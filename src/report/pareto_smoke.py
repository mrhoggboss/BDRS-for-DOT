"""Validate the Pareto blocks of a small ``--pareto`` orchestrator run.

Each ok row's ``algo_extras["pareto"]`` block must be present, finite and
internally consistent (``gap == p_obj - d_obj``, ``r_P`` the sum of its three
terms, ``t_total >= t_solve > 0``), and pass a relaxed weak-duality bound.
Where the row persisted an artifact, which today is only SinkhornGrid's
(f, g, r, c) npz, ``d_obj`` and ``gap`` are recomputed offline and
must agree to 1e-9 relative; the other rows are gated on the internal checks
only. Writes ``report.md`` and ``smoke.json`` into the run directory and exits
1 on any failure, 2 when there are no runs.

    python -m src.report.pareto_smoke SMOKE_DIR
"""

import glob
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.protocol.results_io import iter_rows


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _check_block(rec: Dict[str, Any]) -> Dict[str, Any]:
    """The verdict on one row: its named checks, failures and warnings."""
    algo = rec.get("algorithm", "?")
    norm = rec.get("norm", "?")
    ae = rec.get("algo_extras") or {}
    blk = ae.get("pareto")
    _size = rec.get("resolution") if rec.get("resolution") is not None else rec.get("m")
    out: Dict[str, Any] = {"solver": algo, "norm": norm, "n": _size,
                           "status": rec.get("status"), "checks": {}, "block": blk,
                           "fail": [], "warn": []}
    if rec.get("status") != "ok":
        out["warn"].append(f"record status={rec.get('status')}")
        return out
    if blk is None:
        out["fail"].append("no algo_extras['pareto'] block")
        return out
    # A runner that recovered no dual flags its block: a warning, not a failure.
    if blk.get("dual_absent") or blk.get("pareto_absent"):
        out["warn"].append(f"partial: {blk.get('reason', 'dual absent')}")
        return out

    ck = out["checks"]
    ts, tt = blk.get("t_solve"), blk.get("t_total")
    ck["t_solve>0"] = _finite(ts) and ts > 0
    ck["t_total>=t_solve"] = (tt is None) or (_finite(tt) and _finite(ts) and tt >= ts - 1e-9)
    for k in ("r_P", "r_DF", "r_Dinf", "p_obj", "d_obj", "gap"):
        v = blk.get(k)
        ck[f"{k} finite"] = (v is None) or _finite(v)
    # Weak duality d_obj ≤ p_obj holds only for a feasible dual. For a feasible
    # plan and unit-mass marginals, d_obj − p_obj = Σ X_ij (α_i + β_j − C_ij)
    # ≤ r_Dinf, plus O(r_P ‖(α, β)‖) for an infeasible plan. The 5% margin lets
    # an honest native-stop dual through and fails a gross sign or scale error.
    p_obj, d_obj = blk.get("p_obj"), blk.get("d_obj")
    r_Dinf = blk.get("r_Dinf")
    if _finite(p_obj) and _finite(d_obj):
        overshoot = d_obj - p_obj
        bound = (r_Dinf if _finite(r_Dinf) else 0.0) + 0.05 * (1 + abs(p_obj)) + 1e-6
        ck["weak_duality overshoot<=r_Dinf+margin"] = overshoot <= bound
    if _finite(blk.get("gap")) and _finite(p_obj) and _finite(d_obj):
        ck["gap==p-d"] = abs(blk["gap"] - (p_obj - d_obj)) <= 1e-6 * (1 + abs(p_obj))
    parts = [blk.get(k) for k in ("r_P_row", "r_P_col", "r_P_neg")]
    if _finite(blk.get("r_P")) and all(_finite(x) for x in parts):
        ck["r_P==sum"] = abs(blk["r_P"] - sum(parts)) <= 1e-6 * (1 + blk["r_P"])

    for name, ok in ck.items():
        if not ok:
            out["fail"].append(name)
    return out


def _fg_artifact(blk: Optional[Dict[str, Any]], runs_dir: Path) -> Optional[Path]:
    """The row's persisted scoring artifact (SinkhornGrid's npz), re-rooted
    under ``runs_dir``; None when there is none."""
    from src.report.aggregate import _resolve_fg_path
    return _resolve_fg_path((blk or {}).get("pareto_fg_path"), runs_dir)


def _fill_sinkhorngrid(verdict: Dict[str, Any], runs_dir: Path) -> None:
    """Fill SinkhornGrid's deferred dual residuals into the block from its npz;
    unlike the aggregation, a missing npz is a failure."""
    blk = verdict.get("block") or {}
    if not blk.get("pareto_needs_posthoc_rDinf"):
        return
    fg = _fg_artifact(blk, runs_dir)
    if fg is None:
        verdict["fail"].append("SinkhornGrid: pareto_fg_path missing")
        return
    from src.scoring.pareto_score import posthoc_fill_sinkhorngrid_rdinf
    filled = posthoc_fill_sinkhorngrid_rdinf(str(fg))
    blk.update(filled)
    verdict["checks"]["posthoc r_Dinf finite"] = _finite(blk.get("r_Dinf"))
    if not _finite(blk.get("r_Dinf")):
        verdict["fail"].append("SinkhornGrid: post-hoc r_Dinf not finite")


T3_RTOL = 1e-9          # relative agreement required of the offline re-evaluation


def _t3_reeval(verdict: Dict[str, Any], runs_dir: Path) -> None:
    """Recompute ``d_obj`` and ``gap`` in torch fp64 from the row's persisted
    potentials and original marginals, and require agreement with the
    block to ``T3_RTOL``. ``verdict["t3"]`` becomes ``reeval``, ``internal``
    when no artifact exists, or ``reeval(error)`` when loading or recomputing
    raises (a failure). The plan is never persisted, so ``r_P`` and
    ``p_obj`` are not recomputed."""
    blk = verdict.get("block")
    if not blk:
        return
    fg = _fg_artifact(blk, runs_dir)
    if fg is None:
        verdict["t3"] = "internal"
        return
    verdict["t3"] = "reeval"
    ck = verdict["checks"]
    try:
        import numpy as np
        import torch
        from src.scoring.kkt_residual import support_dot
        d = np.load(str(fg), allow_pickle=False)
        t = lambda k: torch.as_tensor(np.asarray(d[k], dtype=np.float64)).reshape(-1)
        f, g, r, c = t("f"), t("g"), t("r"), t("c")
        d_obj_off = support_dot(f, r) + support_dot(g, c)
    except Exception as e:                                       # noqa: BLE001
        verdict["fail"].append(f"offline re-eval failed: {type(e).__name__}")
        verdict["t3"] = "reeval(error)"
        return

    def _agree(name: str, offline: float, recorded: Any) -> None:
        if not _finite(recorded):
            ck[f"offline {name}: recorded value finite"] = False
            return
        ck[f"offline {name} == recorded (1e-9 rel)"] = (
            abs(offline - float(recorded)) <= T3_RTOL * (1.0 + abs(offline)))

    _agree("d_obj", d_obj_off, blk.get("d_obj"))
    p_obj = blk.get("p_obj")
    if _finite(p_obj):
        _agree("gap", float(p_obj) - d_obj_off, blk.get("gap"))
    verdict["t3_offline"] = {"d_obj": d_obj_off}
    for name, ok in ck.items():
        if not ok and name not in verdict["fail"]:
            verdict["fail"].append(name)


def run(smoke_dir: str) -> int:
    d = Path(smoke_dir)
    files = sorted(glob.glob(str(d / "runs" / "*.jsonl")))
    if not files:
        print(f"pareto_smoke: no runs/*.jsonl under {d}", file=sys.stderr)
        return 2
    verdicts: List[Dict[str, Any]] = []
    for _, rec in iter_rows(files):
        v = _check_block(rec)
        v["t3"] = "n/a"
        if (v.get("block") or {}).get("pareto_needs_posthoc_rDinf"):
            _fill_sinkhorngrid(v, d / "runs")
        if v.get("block") is not None and not v["fail"]:
            _t3_reeval(v, d / "runs")
        verdicts.append(v)

    lines = ["# Pareto smoke", "",
             "PASS: the Pareto block is present and finite, satisfies the relaxed "
             "weak-duality bound d_obj - p_obj <= r_Dinf + 0.05(1 + |p_obj|) + 1e-6 "
             "(a native-stop dual may be slightly infeasible), is internally "
             "consistent (gap = p_obj - d_obj, r_P the sum of its terms) and has "
             "t_total >= t_solve > 0.",
             "",
             "Offline column: `reeval` means d_obj and gap were recomputed "
             "in torch fp64 from the row's persisted artifact and agree with the "
             "block to 1e-9 relative. Only SinkhornGrid persists one, its "
             "(f, g, r, c) npz; the plan is not persisted, so r_P and p_obj are "
             "not recomputed. `internal` means no artifact exists and the row is "
             "gated on the internal checks only.",
             "",
             "| solver | norm | n | r_P | r_Dinf | r_DF | gap | t_solve | t_total | offline | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def g(b, k):
        v = (b or {}).get(k)
        return f"{v:.2e}" if _finite(v) else ("—" if v is None else str(v))

    n_pass = n_fail = n_partial = 0
    for v in sorted(verdicts, key=lambda x: (x["solver"], x["norm"] or "")):
        b = v["block"]
        if v["fail"]:
            verdict = "FAIL(" + ",".join(v["fail"][:2]) + ")"; n_fail += 1
        elif v["warn"] and b is None:
            verdict = "n/a(" + v["warn"][0][:20] + ")"; n_partial += 1
        elif v["warn"]:
            verdict = "PARTIAL"; n_partial += 1
        else:
            verdict = "PASS"; n_pass += 1
        lines.append(f"| {v['solver']} | {v['norm']} | {v['n']} | "
                     f"{g(b,'r_P')} | {g(b,'r_Dinf')} | {g(b,'r_DF')} | {g(b,'gap')} | "
                     f"{g(b,'t_solve')} | {g(b,'t_total')} | {v.get('t3', 'n/a')} | "
                     f"{verdict} |")
    n_reeval = sum(1 for v in verdicts if str(v.get("t3", "")).startswith("reeval"))
    n_internal = sum(1 for v in verdicts if v.get("t3") == "internal")
    lines += ["", f"**{n_pass} PASS · {n_fail} FAIL · {n_partial} partial/na** "
              f"over {len(verdicts)} runs.",
              "", f"Offline: {n_reeval} row(s) re-evaluated from persisted "
              f"artifacts (1e-9 relative agreement required); {n_internal} row(s) "
              f"had no persisted artifact and were gated on internal consistency.",
              "" if n_fail == 0 else
              "\n**GATE BLOCKED**: fix the FAIL rows."]
    report = "\n".join(lines)
    (d / "report.md").write_text(report)
    json.dump(verdicts, open(d / "smoke.json", "w"), indent=1, default=str)
    print(report)
    return 1 if n_fail else 0


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("smoke_dir", help="the --pareto orchestrator --output-dir")
    a = p.parse_args(argv)
    return run(a.smoke_dir)


if __name__ == "__main__":
    sys.exit(main())
