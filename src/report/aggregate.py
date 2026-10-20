"""Aggregate the runners' result rows into the Pareto summaries.

Reads ``<runs-dir>/*.jsonl`` and writes ``pareto.csv`` (one row per run),
``pareto_cells.csv`` (one row per dataset, rung, size, norm, algorithm and
tolerance) and ``pareto_coverage.csv`` (the co-solve columns of the cell rows).

A cell's statistics are taken over its co-solved set: the instances scored by
every algorithm of the cell's basis, which is the algorithms with at least one
scorable row in the cell, narrowed by ``--pareto-algos``. An errored, timed-out
or unscorable run is counted per solver, never substituted, and
``cosolve_shrink`` is how many instances the set would regain without that
solver. A run whose t_solve exceeds its time limit is unscorable, whatever its
status, and so is a baseline run without t_total, so every scorable run has a
t_cert. Times are the Pareto block's own clocks, never ``wall_time_s``.

    python -m src.report.aggregate [--runs-dir DIR] [--output-dir DIR]
                                   [--pareto-algos ALGO ...]
"""

import argparse
import csv
import dataclasses
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.protocol.paths import REPO_ROOT
from src.protocol.results_io import iter_rows
from src.protocol.instance_record import (
    POINT_CLOUD,
    SCHEMA_COMPAT,
    InstanceSpec,
)
from src.protocol.regimes import (
    OUR_SOLVERS,
    canonical_algorithm,
    canonical_regime,
    regime_for_algorithm,
)
from src.scoring.kkt_error import PARETO_COMPONENTS, absolute_kkt_error, is_finite_number
from src.report.sgm import sgm, geomean, median, iqr


DEFAULT_RUNS_DIR   = REPO_ROOT / "results" / "bench" / "runs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "bench" / "summaries"



def _load_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """The rows of ``runs_dir/*.jsonl`` with a compatible schema version and
    regime, rung and dataset backfilled, keeping the last row per (instance_id,
    algorithm, tolerance)."""
    if not runs_dir.is_dir():
        return []
    keep: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, rec in iter_rows(sorted(runs_dir.glob("*.jsonl"))):
        if rec.get("schema_version", "v1") not in SCHEMA_COMPAT:
            continue
        algo = rec["algorithm"]
        reg = rec.get("regime")
        if reg is None:
            try:
                rec["regime"] = regime_for_algorithm(algo)
            except ValueError:
                rec["regime"] = None
        else:
            rec["regime"] = canonical_regime(reg)
        # v1 rows have no dataset or rung: they are DOTmark runs on the
        # instance's native rung, which is the algorithm's regime.
        rec.setdefault("dataset", "dotmark")
        rung = rec.get("rung")
        rec["rung"] = (canonical_regime(rung) if rung is not None
                       else rec.get("regime"))
        # Each tolerance arm of our solvers is a separate run of the same
        # (instance, algorithm), so the tolerance belongs in the key.
        tol = rec.get("tolerance")
        key = (rec.get("instance_id"), algo,
               None if tol is None else float(tol))
        # Last wins, so a re-run after a fix replaces the earlier row.
        keep[key] = rec
    return list(keep.values())


# The block fields a row needs, finite, to be scorable. r_DF, not r_Dinf: the
# dual term of the KKT residual is the Frobenius residual.
PARETO_REQUIRED = ("r_P", "r_DF", "p_obj", "d_obj", "gap")

# The time limit of a run whose record does not carry one in
# algo_extras["time_limit_s"]: InstanceSpec's.
DEFAULT_TIME_LIMIT_S = float(next(f.default for f in dataclasses.fields(InstanceSpec)
                                  if f.name == "time_limit_s"))

# Unscorable reasons that come from a record's status flags and Pareto block,
# not from the run's components; pareto.csv keeps them in unscorable_reason.
RECORD_REASONS = frozenset({"timeout_flag", "no_pareto_block", "pareto_absent",
                            "dual_absent", "posthoc_dual_unfilled"})

# t_cert is the time to a solution certified by the KKT residual. Our solvers
# stop on a certificate of their own, so t_cert = t_solve. For a baseline,
# t_cert = t_total, which adds to t_solve the recovery of (X, α, β) from the
# solver's output, less the harness I/O of ``HARNESS_IO_CLOCKS``, plus t_eval,
# the O(mn) evaluation of α ⊕ β ≤ C in the scoring, for the solvers below,
# whose dual is not feasible by construction and whose native stop does not
# measure LP dual feasibility:
#   HALO: potentials on a restricted support, checked against all pairs;
#   HOT:  stops on its reduced flow LP; the lifted duals are only approximately
#         feasible for the original LP;
#   DROT: stops on its own splitting residual;
#   M3S:  entropic on a truncated kernel, so α_i + β_j − C_ij = ε log X_ij ≤ 0
#         holds only on the kept support.
# Not listed: LAMP (the c-transform makes the delivered α ⊕ β ≤ C exact),
# Sinkhorn and SinkhornGrid (the entropic identity holds on the full kernel),
# and cuPDLPx and PDOT (their native stop measures the dual residual inside
# t_solve). A block's own ``dual_certified`` overrides this set.
NEEDS_CERTIFICATE_PASS = frozenset({"HOT", "HALO", "DROT", "M3S"})

# Block clocks inside a baseline's t_total that time the benchmark harness, not
# the solver's recovery, and that t_cert leaves out: DROT's shim writing the
# fp32 plan to disk (t_shim_write) and the runner reading it back and casting it
# to fp64 (t_plan_read). A clock the block lacks is not subtracted.
HARNESS_IO_CLOCKS = {"DROT": ("t_shim_write", "t_plan_read")}

# Block fields other than the components that are copied to pareto.csv.
PARETO_META = ("t_solve", "t_total", "t_eval", "precision", "native_stop",
               "native_stop_is_library_default", "native_stop_is_upstream_default",
               "pareto_partial", "t_solve_boundary", "t_solve_includes",
               "t_solve_excludes", "t_total_boundary", "t_total_includes",
               "t_cost_build", "t_eta_select", "t_plan_read", "t_shim_write",
               "t_shim_recovery", "t_shim_solver_only", "t_shim_checks",
               # declared by a src/runners/baseline.py adapter
               "dual_certified")

PARETO_COVERAGE_COLUMNS = (
    "dataset", "rung", "size", "norm", "algorithm",
    "n_records", "n_scorable", "n_partial", "n_error", "n_timeout", "n_oom",
    "n_unscorable", "unscorable_reasons",
    "in_cosolve_basis", "n_cosolved", "n_cosolved_cell",
    "n_cosolved_if_dropped", "cosolve_shrink",
    "n_cell_solvers", "n_basis_solvers", "cosolve_basis", "n_cell_instances",
)


def _size_of(rec: Dict[str, Any]) -> Optional[int]:
    """The cell's size: m for a native Point Cloud instance, else the grid side
    n_g, else m. A Point Cloud instance lifted to General carries resolution =
    √m, so sizing by m keeps its rows in one cell with the Point Cloud solvers'
    rows, which have no resolution."""
    if rec.get("size") is not None:
        return rec["size"]
    if rec.get("rung") == POINT_CLOUD and rec.get("m") is not None:
        return rec["m"]
    size = rec.get("resolution")
    return size if size is not None else rec.get("m")


def _pareto_cell_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    """(dataset, native rung, size, norm): properties of the instance only, so
    every solver run on an instance lands in the same cell."""
    return (rec.get("dataset") or "dotmark", rec.get("rung"),
            _size_of(rec), rec.get("norm"))


def _resolve_fg_path(recorded: Optional[str], runs_dir: Path) -> Optional[Path]:
    """The recorded ``pareto_fg_path`` if it exists, else the same file name
    searched for under ``runs_dir`` and its parent (the recorded path is stale
    once a results tree is copied to another machine)."""
    if not recorded:
        return None
    p = Path(recorded)
    if p.is_file():
        return p
    name = p.name
    for cand in (runs_dir / "pareto_sidecars" / name,
                 runs_dir / name,
                 runs_dir.parent / "pareto_sidecars" / name,
                 runs_dir.parent / "runs" / "pareto_sidecars" / name):
        if cand.is_file():
            return cand
    return None


def _fill_sinkhorngrid_posthoc(records: List[Dict[str, Any]],
                               runs_dir: Path) -> Dict[str, int]:
    """Fill, in place, the r_Dinf and r_DF that SinkhornGrid's runner defers
    because it runs without torch: from its (f, g) sidecar when found,
    else analytically. A fill that raises (no torch, for one) leaves the block
    deferred, so the row is unscorable as ``posthoc_dual_unfilled``."""
    stats = {"needed": 0, "filled": 0, "analytic": 0, "failed": 0}
    for rec in records:
        blk = ((rec.get("algo_extras") or {}).get("pareto") or {})
        if not blk.get("pareto_needs_posthoc_rDinf"):
            continue
        stats["needed"] += 1
        fg = _resolve_fg_path(blk.get("pareto_fg_path"), runs_dir)
        try:
            from src.scoring.pareto_score import (
                posthoc_fill_sinkhorngrid_analytic, posthoc_fill_sinkhorngrid_rdinf)
            if fg is None:
                # An entropic dual is feasible by construction
                # (α_i + β_j − C_ij = ε log X_ij ≤ 0), so r_DF = r_Dinf = 0; the
                # sidecar only buys a numerical check of that zero, at
                # n_g ≤ 128.
                blk.update(posthoc_fill_sinkhorngrid_analytic())
                blk["pareto_posthoc_source"] = "analytic (no sidecar present)"
                stats["analytic"] += 1
            else:
                blk.update(posthoc_fill_sinkhorngrid_rdinf(str(fg)))
                blk["pareto_posthoc_source"] = str(fg)
            blk["pareto_needs_posthoc_rDinf"] = False
            stats["filled"] += 1
        except Exception as e:                                   # noqa: BLE001
            blk["pareto_posthoc_error"] = f"{type(e).__name__}: {e}"
            stats["failed"] += 1
    return stats


def _flag(v: Any) -> bool:
    """A boolean field, also as pareto.csv writes it ("True"/"False")."""
    return v in (True, "True", "true", "1", 1)


def _recorded_time_limit(rec: Dict[str, Any]) -> Optional[float]:
    """The run's ``spec.time_limit_s``, which the orchestrator records in
    ``algo_extras["time_limit_s"]``; None when the record carries none."""
    v = (rec.get("algo_extras") or {}).get("time_limit_s")
    return float(v) if is_finite_number(v) and float(v) > 0.0 else None


def time_limit_of(run: Dict[str, Any]) -> float:
    """The run's ``time_limit_s``, or ``DEFAULT_TIME_LIMIT_S`` when it has none."""
    v = run.get("time_limit_s")
    return float(v) if is_finite_number(v) and float(v) > 0.0 else DEFAULT_TIME_LIMIT_S


def components_verdict(run: Dict[str, Any]) -> Tuple[bool, str]:
    """(scorable, reason) from a run's components alone: a finite non-negative
    t_solve within the run's time limit, a finite t_total for a baseline (its
    t_cert starts from it), and every ``PARETO_REQUIRED`` field finite. ``run``
    is a Pareto block with the run's ``algorithm`` and ``time_limit_s``, or a
    pareto.csv row."""
    if not is_finite_number(run.get("t_solve")) or float(run["t_solve"]) < 0.0:
        return False, "no_t_solve"
    if float(run["t_solve"]) > time_limit_of(run):
        return False, "time_limit"
    if run.get("algorithm") not in OUR_SOLVERS and not is_finite_number(run.get("t_total")):
        return False, "no_t_total"
    missing = [k for k in PARETO_REQUIRED if not is_finite_number(run.get(k))]
    if missing:
        return False, "nonfinite:" + ",".join(missing)
    return True, ""


def _pareto_scorable(rec: Dict[str, Any]) -> Tuple[bool, str]:
    """(scorable, reason): whether the record is a Pareto point, and if not why."""
    status = rec.get("status")
    if status != "ok":
        return False, str(status or "no_status")
    # iter_rows already reads an ok row with timeout_flag as a timeout; this
    # covers rows that did not come through it.
    if rec.get("timeout_flag") or (rec.get("algo_extras") or {}).get("timeout_flag"):
        return False, "timeout_flag"
    blk = (rec.get("algo_extras") or {}).get("pareto")
    if not blk:
        return False, "no_pareto_block"
    if blk.get("pareto_absent"):
        return False, "pareto_absent"
    if blk.get("dual_absent"):
        return False, "dual_absent"
    if blk.get("pareto_partial"):
        return False, "pareto_partial"
    if blk.get("pareto_needs_posthoc_rDinf"):
        return False, "posthoc_dual_unfilled"
    return components_verdict({**blk, "algorithm": canonical_algorithm(rec.get("algorithm") or ""),
                               "time_limit_s": _recorded_time_limit(rec)})


def row_verdict(row: Dict[str, Any]) -> Tuple[bool, str]:
    """``_pareto_scorable`` for a pareto.csv row, which keeps the components
    but not the record's flags: the status and a recorded reason in
    ``RECORD_REASONS`` stand, and everything else is decided again from the
    components, so a table written by another version of these rules is
    scored by these."""
    status = row.get("status")
    if status != "ok":
        return False, str(status or "no_status")
    if row.get("unscorable_reason") in RECORD_REASONS:
        return False, row["unscorable_reason"]
    if _flag(row.get("pareto_partial")):
        return False, "pareto_partial"
    return components_verdict(row)


def _needs_certificate_pass(run: Dict[str, Any]) -> bool:
    declared = run.get("dual_certified")
    if isinstance(declared, str):                       # a row read back from pareto.csv
        declared = {"True": True, "False": False}.get(declared)
    if isinstance(declared, bool):
        return not declared
    return run.get("algorithm") in NEEDS_CERTIFICATE_PASS


def cert_time(run: Dict[str, Any]) -> Tuple[Optional[float], bool]:
    """(t_cert, complete) of a pareto.csv row (see ``NEEDS_CERTIFICATE_PASS``
    and ``HARNESS_IO_CLOCKS``). A baseline without t_total has no t_cert (such
    a run is unscorable); one that needs the scoring pass but has no t_eval
    gets its t_total less the harness I/O. ``complete`` is False in both
    cases."""
    if not is_finite_number(run.get("t_solve")):
        return None, False
    if run.get("algorithm") in OUR_SOLVERS:
        return float(run["t_solve"]), True
    if not is_finite_number(run.get("t_total")):
        return None, False
    t_recovered = float(run["t_total"]) - sum(
        float(run[k]) for k in HARNESS_IO_CLOCKS.get(run.get("algorithm"), ())
        if is_finite_number(run.get(k)))
    if not _needs_certificate_pass(run):
        return t_recovered, True
    if is_finite_number(run.get("t_eval")):
        return t_recovered + float(run["t_eval"]), True
    return t_recovered, False


def _scalarize(v: Any) -> Any:
    """A list or tuple becomes a ``|``-joined string for the CSV."""
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return v


def _pareto_rows(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One pareto.csv row per record. Names are canonicalised again (the mapping
    is idempotent) so that records not read through ``_load_runs`` group
    correctly too."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        blk = (rec.get("algo_extras") or {}).get("pareto") or {}
        scorable, reason = _pareto_scorable(rec)
        iid = rec.get("instance_id")
        rung = rec.get("rung")
        regime = rec.get("regime")
        row: Dict[str, Any] = {
            "instance_id": iid,
            "dataset": rec.get("dataset") or "dotmark",
            "rung": canonical_regime(rung) if rung else rung,
            "regime": canonical_regime(regime) if regime else regime,
            "algorithm": canonical_algorithm(rec.get("algorithm") or ""),
            "resolution": rec.get("resolution"),
            "m": rec.get("m"),
            "size": _size_of(rec),
            "norm": rec.get("norm"),
            "status": rec.get("status"),
            "scorable": scorable,
            "unscorable_reason": (reason or None),
            "native_termination": rec.get("native_termination"),
            "iters": rec.get("iters"),
            "iter_cap_hit": rec.get("iter_cap_hit"),
            "tolerance": rec.get("tolerance"),
            "wall_time_s": rec.get("wall_time_s"),   # each runner defines it; never aggregated
            "time_limit_s": _recorded_time_limit(rec) or DEFAULT_TIME_LIMIT_S,
        }
        for k in PARETO_META:
            row[k] = _scalarize(blk.get(k))
        row["t_cert"] = cert_time(row)[0]
        for k in PARETO_COMPONENTS:
            row[k] = blk.get(k)
        row["gap_abs"] = abs(blk["gap"]) if is_finite_number(blk.get("gap")) else None
        row.update(absolute_kkt_error(blk))
        # g_feas ≥ 0. The dual is exactly feasible, so g_feas ≥ p_obj − OPT for
        # either sign, and g_feas < 0 means p_obj < OPT (X primal-infeasible).
        # Only our solvers write it, so None means not applicable. It does not
        # gate E.
        row["certificate_valid"] = (rec.get("algo_extras") or {}).get(
            "certificate_valid")
        row.update({
            "peak_gpu_mem_bytes": rec.get("peak_gpu_mem_bytes"),
            "device_name": rec.get("device_name"),
            "device_uuid": rec.get("device_uuid"),
            "git_sha": rec.get("git_sha"),
            "run_uid": rec.get("run_uid"),
            "started_utc": rec.get("started_utc"),
            "ended_utc": rec.get("ended_utc"),
            "schema_version": rec.get("schema_version"),
            "error_kind": rec.get("error_kind"),
        })
        rows.append(row)
    return rows


def _scored_by_instance(rows: List[Dict[str, Any]]
                        ) -> Dict[Tuple[Any, ...], Dict[str, set]]:
    """cell -> instance_id -> the set of algorithms that SCORED that instance."""
    scored: Dict[Tuple[Any, ...], Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        if r["scorable"]:
            scored[_pareto_cell_key(r)][r["instance_id"]].add(r["algorithm"])
    return scored


def _cosolve_basis(rows: List[Dict[str, Any]],
                   algo_filter: Optional[set] = None) -> Dict[Tuple[Any, ...], set]:
    """Per cell, the algorithms that define its co-solved set: those with at
    least one scorable row in the cell (and in ``algo_filter`` when given). A
    solver that never scores in a cell would otherwise empty the set for every
    other solver; one that scores some instances still shrinks it."""
    basis: Dict[Tuple[Any, ...], set] = defaultdict(set)
    for r in rows:
        if not r["scorable"]:
            continue
        if algo_filter is not None and r["algorithm"] not in algo_filter:
            continue
        basis[_pareto_cell_key(r)].add(r["algorithm"])
    return basis


def _intersect(scored_cell: Dict[str, set], basis: set) -> set:
    """The instances scored by every algorithm of ``basis``; empty for an empty basis."""
    if not basis:
        return set()
    return {iid for iid, got in scored_cell.items() if got >= basis}


def _aggregate_pareto_cells(rows: List[Dict[str, Any]],
                            algo_filter: Optional[set] = None
                            ) -> List[Dict[str, Any]]:
    """One row per (dataset, rung, size, norm, algorithm, tolerance), with its
    statistics over the cell's co-solved instances. A group whose co-solved set
    is empty is still emitted, with its counts."""
    basis_by_cell = _cosolve_basis(rows, algo_filter)
    scored = _scored_by_instance(rows)
    algos_in_cell: Dict[Tuple[Any, ...], set] = defaultdict(set)
    insts_in_cell: Dict[Tuple[Any, ...], set] = defaultdict(set)
    for r in rows:
        algos_in_cell[_pareto_cell_key(r)].add(r["algorithm"])
        insts_in_cell[_pareto_cell_key(r)].add(r["instance_id"])

    # Groups are per tolerance arm, but the basis is per algorithm: an instance
    # counts as scored by a solver if any of its arms scored it, so one arm's
    # misses do not shrink the set for the others (n_cosolved is the arm's own
    # count within it).
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        tol = r.get("tolerance")
        groups[_pareto_cell_key(r) + (r["algorithm"],
                                     None if tol is None else float(tol))].append(r)

    out: List[Dict[str, Any]] = []
    for gkey, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        dataset, rung, size, norm, algo, tol = gkey
        cell = (dataset, rung, size, norm)
        basis = basis_by_cell.get(cell, set())
        scored_cell = scored.get(cell, {})
        co = _intersect(scored_cell, basis)
        co_rows = [r for r in rs if r["scorable"] and r["instance_id"] in co]
        # The co-solved set without this solver in the basis.
        in_basis = algo in basis
        co_without = (_intersect(scored_cell, basis - {algo}) if in_basis else co)

        reasons: Dict[str, int] = defaultdict(int)
        for r in rs:
            if not r["scorable"]:
                reasons[r["unscorable_reason"] or "unknown"] += 1

        def _col(rr: List[Dict[str, Any]], field: str) -> List[float]:
            return [r[field] for r in rr if is_finite_number(r.get(field))]

        t_solve = _col(co_rows, "t_solve")
        t_total = _col(co_rows, "t_total")
        # t_cert from each solve's own clocks, never from a t_cert column, then
        # SGM10. n_cert_pass_missing counts the solves whose t_cert lacks a
        # clock (cert_time).
        t_cert, n_cert_missing = [], 0
        for r in co_rows:
            tc, complete = cert_time(r)
            if not complete:
                n_cert_missing += 1
            if tc is not None:
                t_cert.append(tc)
        n_records = len(rs)
        out.append({
            "dataset": dataset, "rung": rung, "size": size, "norm": norm,
            "algorithm": algo,
            "tolerance": tol,
            "regime": rs[0].get("regime"),
            "resolution": rs[0].get("resolution"),
            "n_records": n_records,
            "n_total": n_records,                     # alias of n_records
            "n_ok": sum(1 for r in rs if r["status"] == "ok"),
            "n_scorable": sum(1 for r in rs if r["scorable"]),
            "n_partial": sum(1 for r in rs if r.get("pareto_partial")),
            "n_cosolved": len(co_rows),
            "n_error": sum(1 for r in rs if r["status"] == "error"),
            "n_timeout": sum(1 for r in rs if r["status"] == "timeout"),
            "n_oom": sum(1 for r in rs if r["status"] in ("oom", "oom_recovery")),
            "n_unscorable": sum(1 for r in rs if not r["scorable"]),
            "unscorable_reasons": ";".join(
                f"{k}={v}" for k, v in sorted(reasons.items())) or "",
            "t_solve_sgm10_s": sgm(t_solve, shift=10.0),
            "t_solve_median_s": median(t_solve),
            "t_solve_iqr_s": iqr(t_solve),
            "t_total_sgm10_s": sgm(t_total, shift=10.0),
            "t_total_median_s": median(t_total),
            "t_cert_sgm10_s": sgm(t_cert, shift=10.0),
            "t_cert_median_s": median(t_cert),
            "n_cert_pass_missing": n_cert_missing,
            # E_abs (the KKT residual) is computed per solve, then aggregated;
            # its terms are not aggregated separately, since a max does not
            # commute with a mean. A plain geometric mean, not SGM10: E_abs
            # spans decades below 1, where a shift of 10 would swamp every
            # value. n_E_abs counts the solves with a finite E_abs.
            "E_abs_gm": geomean(_col(co_rows, "E_abs")),
            "E_abs_median": median(_col(co_rows, "E_abs")),
            "n_E_abs": len(_col(co_rows, "E_abs")),
            # Solves with g_feas < 0, i.e. p_obj < OPT. Counted, not removed from
            # E_abs: its gap term is |g|, and dropping them would bias the cell
            # toward the solves with g_feas ≥ 0.
            "n_cert_invalid": sum(1 for r in co_rows
                                  if r.get("certificate_valid") is False),
            "r_P_median": median(_col(co_rows, "r_P")),
            "r_DF_median": median(_col(co_rows, "r_DF")),
            "r_Dinf_median": median(_col(co_rows, "r_Dinf")),
            "gap_abs_median": median(_col(co_rows, "gap_abs")),
            "iters_median": median(_col(co_rows, "iters")),
            "in_cosolve_basis": in_basis,
            "n_cosolved_cell": len(co),
            "n_cosolved_if_dropped": len(co_without),
            "cosolve_shrink": len(co_without) - len(co),
            "n_cell_solvers": len(algos_in_cell[cell]),
            "n_basis_solvers": len(basis),
            "cosolve_basis": "|".join(sorted(basis)),
            "n_cell_instances": len(insts_in_cell[cell]),
        })
    return out


def _pareto_coverage_rows(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The coverage columns of each cell row, for ``pareto_coverage.csv``."""
    return [{k: c.get(k) for k in PARETO_COVERAGE_COLUMNS} for c in cells]


def _write_csv(rows: List[Dict[str, Any]], path: Path,
               extra_columns: Optional[List[str]] = None) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    seen: List[str] = []
    if extra_columns:
        for c in extra_columns:
            if c not in seen:
                seen.append(c)
    for row in rows:
        for c in row.keys():
            if c not in seen:
                seen.append(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seen, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            nice = {}
            for k, v in row.items():
                if isinstance(v, float) and not math.isfinite(v):
                    nice[k] = ""
                else:
                    nice[k] = v
            w.writerow(nice)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir",   type=Path, default=DEFAULT_RUNS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pareto-algos", nargs="+", default=None, metavar="ALGO",
                   help="Restrict the co-solve basis to these algorithms: only "
                        "they define each cell's co-solved instance set. Every "
                        "algorithm is still aggregated and reported; one left "
                        "out of the basis cannot shrink the set. Alias names "
                        "(e.g. BDRS for Grid) are canonicalised. The column "
                        "cosolve_shrink of pareto_coverage.csv shows how many "
                        "instances each solver costs the set.")
    args = p.parse_args(argv)

    algo_filter = ({canonical_algorithm(a) for a in args.pareto_algos}
                   if args.pareto_algos else None)

    raw_records = _load_runs(args.runs_dir)
    print(f"aggregate: loaded {len(raw_records)} unique (instance_id, algorithm, "
          f"tolerance) records from {args.runs_dir}")

    fill = _fill_sinkhorngrid_posthoc(raw_records, args.runs_dir)
    if fill["needed"]:
        print(f"aggregate: SinkhornGrid post-hoc dual fill: {fill['filled']}/"
              f"{fill['needed']} filled "
              f"({fill['analytic']} analytically without a sidecar, "
              f"{fill['failed']} failed)")
    prows = _pareto_rows(raw_records)
    n_scorable = sum(1 for r in prows if r["scorable"])
    if algo_filter is not None:
        print(f"aggregate: co-solve basis restricted to {sorted(algo_filter)} "
              f"(other algorithms are still aggregated + reported)")
    _write_csv(prows, args.output_dir / "pareto.csv",
               extra_columns=["instance_id", "algorithm", "dataset", "rung",
                              "regime", "norm", "resolution", "size", "status",
                              "scorable", "unscorable_reason",
                              "t_solve", "t_total", "t_eval", "t_cert",
                              "time_limit_s",
                              *PARETO_COMPONENTS, "gap_abs",
                              "native_stop",
                              "native_termination", "iters", "precision",
                              "t_solve_boundary", "t_solve_includes"])
    print(f"aggregate: wrote pareto.csv ({len(prows)} rows, {n_scorable} scorable) "
          f"to {args.output_dir / 'pareto.csv'}")
    pcells = _aggregate_pareto_cells(prows, algo_filter)
    _write_csv(pcells, args.output_dir / "pareto_cells.csv",
               extra_columns=["dataset", "rung", "size", "norm", "algorithm",
                              "tolerance",
                              "n_records", "n_total", "n_ok", "n_scorable",
                              "n_partial", "n_cosolved",
                              "n_error", "n_timeout", "n_oom", "n_unscorable",
                              "unscorable_reasons",
                              "in_cosolve_basis", "n_cosolved_cell",
                              "n_cosolved_if_dropped", "cosolve_shrink",
                              "E_abs_gm", "E_abs_median", "n_E_abs",
                              "n_cert_invalid",
                              "t_solve_sgm10_s", "t_solve_median_s",
                              "t_solve_iqr_s", "t_total_sgm10_s",
                              "t_total_median_s", "t_cert_sgm10_s",
                              "t_cert_median_s", "n_cert_pass_missing"])
    print(f"aggregate: wrote pareto_cells.csv ({len(pcells)} rows) to "
          f"{args.output_dir / 'pareto_cells.csv'}")
    _write_csv(_pareto_coverage_rows(pcells),
               args.output_dir / "pareto_coverage.csv",
               extra_columns=list(PARETO_COVERAGE_COLUMNS))
    n_empty = sum(1 for c in pcells if c["n_cosolved_cell"] == 0)
    n_shrink = sum(1 for c in pcells if c["cosolve_shrink"] > 0)
    print(f"aggregate: wrote pareto_coverage.csv ({len(pcells)} rows; "
          f"{n_empty} with an empty co-solved set, {n_shrink} solver-cells that "
          f"shrink their cell) to {args.output_dir / 'pareto_coverage.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
