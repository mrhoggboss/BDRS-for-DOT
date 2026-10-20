"""
The benchmark scheduler.

Runs the scheduled (algorithm, instance) pairs one at a time, since GPU work is
serialised, and appends one ResultRecord per run to
``<output-dir>/runs/<algorithm in lower case>.jsonl``, without a lock: two
orchestrators must not share an output directory. Resume skips an
(instance_id, algorithm, tolerance) key that already has a settled row, so
``error`` rows are retried.

    python -m src.benchmark.orchestrator --algos Grid General --tolerance 1e-4 --pareto
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.benchmark import env_capture
from src.benchmark.dispatch import (
    append_record,
    _make_synthesized_record,
    run_subprocess,
    _spec_from_index,
    _stamp_time_limit,
)
from src.benchmark.lift import _prepare_ent_for_algo
from src.benchmark.pdot_adapter import PdotAdapter
from src.runners.run_pdot import DEFAULT_DRIVER, DEFAULT_PDOT_DIR
from src.benchmark.schedule import (
    ALL_ALGOS,
    LAMP_ALGOS,
    OPT_IN_ALGOS,
    build_schedule,
    filter_index,
    load_all_indexes,
    load_done_set,
    _normalised_index_entry,
)
from src.protocol.instance_record import utc_now_iso
from src.protocol.paths import REPO_ROOT
from src.protocol.regimes import (
    ALL_NORMS,
    DENSE_ONLY_NORMS,
    GENERAL,
    GRID,
    REGIMES,
    SINKHORN_PREFIX,
    SINKHORNGRID_PREFIX,
    canonical_algorithm,
    canonical_regime,
    eps_variant_name,
    parse_policy_exceptions,
    regime_for_algorithm,
    synthesize_dense_norm_entries,
)


DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "dotmark_instances_index.json"


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "bench"


_LEGACY_ALGO_CLI = ("BDRS", "BDRS_geo", "BDRS_dense")


_LEGACY_REGIME_CLI = ("grid", "geometric", "dense")


def _parse_pair(s: str) -> Tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--pairs expects 'i,j' tokens; got {s!r}")
    return int(parts[0]), int(parts[1])


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algos", nargs="+", default=list(ALL_ALGOS),
                   choices=list(ALL_ALGOS) + list(OPT_IN_ALGOS) + list(_LEGACY_ALGO_CLI),
                   help="Algorithms to run (Grid / PointCloud / General + the "
                        "baselines). The aliases BDRS, BDRS_geo and BDRS_dense "
                        "are accepted and canonicalised. 'Sinkhorn' is a family token: "
                        "it expands to one logical algorithm per --sinkhorn-eps "
                        "value, scheduled as 'Sinkhorn_e<slug>' (e.g. Sinkhorn_e0p01 "
                        "for ε=0.01). LAMP and LAMPPointCloud are not in the "
                        "default and run only without other algorithms.")
    p.add_argument("--sinkhorn-eps", nargs="+", type=float, default=[],
                   help="Sinkhorn entropic regularization values to sweep. "
                        "Required when 'Sinkhorn' is in --algos. Each ε produces a "
                        "logically-distinct algorithm in the schedule and per-cell CSV.")
    p.add_argument("--ott-eps", nargs="+", type=float, default=[],
                   help="SinkhornGrid (OTT Grid Sinkhorn) entropic regularization "
                        "values to sweep. Required when 'SinkhornGrid' is in --algos. "
                        "Each ε produces a 'SinkhornGrid_e<slug>' algorithm. Pass the "
                        "SAME values as --sinkhorn-eps to compare OTT vs POT Sinkhorn "
                        "at matched ε.")
    p.add_argument("--regimes", nargs="+", default=None,
                   choices=list(REGIMES) + list(_LEGACY_REGIME_CLI),
                   help="Restrict the run to algorithms of these input regimes "
                        "(Grid / PointCloud / General; the aliases grid/geometric/"
                        "dense are accepted). The regime is a property of the ALGORITHM, not "
                        "the instance: 'Grid' solvers consume (r, c, n_g, cost_type) "
                        "and never materialise the N x N cost matrix; 'General' "
                        "solvers consume an explicit dense C. Applied after the "
                        "Sinkhorn family expansion. Default: all regimes.")
    p.add_argument("--norms", nargs="+", default=list(ALL_NORMS), choices=list(ALL_NORMS),
                   help="Ground costs to run. Default: all, which the norm "
                        "policy of src/protocol/regimes.py then narrows for each "
                        "algorithm. 'l2' is the unsquared Euclidean cost, distinct "
                        "from sql2, and 'inf' the Chebyshev cost; DOTmark l2/inf "
                        "instances missing from the index are derived from its "
                        "Grid entries, and each runner builds their lattice cost.")
    p.add_argument("--resolutions", nargs="+", type=int, default=[32, 64, 128, 256, 512])
    p.add_argument("--classes", nargs="+", default=None,
                   help="Restrict to these DOTmark classes (default: all in index).")
    p.add_argument("--pairs", nargs="+", type=_parse_pair, default=None,
                   help="Restrict to these (i,j) pairs (each as 'i,j').")
    p.add_argument("--max-instances", type=int, default=None,
                   help="Stop after this many instance-runs.")
    p.add_argument("--spine-first", action="store_true",
                   help="One instance per (algorithm, dataset, size, norm) first, "
                        "then the rest; the size is the resolution, else m.")
    p.add_argument("--sample-first", action="store_true",
                   help="One instance per (algorithm, dataset, size, norm, class) "
                        "first, then the rest; the size is the resolution, else m.")
    p.add_argument("--head-only", action="store_true",
                   help="With --spine-first or --sample-first, run only the head "
                        "(no tail). Useful for quick characterisation across the cell grid.")
    p.add_argument("--no-resume", action="store_true",
                   help="Disable resume-skip; re-run already-settled records.")
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--pareto", action="store_true",
                   help="Pareto protocol: each runner stops on its own native "
                        "rule, recovers (X, α, β) for the original LP and records "
                        "the KKT components and the clocks t_solve and t_total "
                        "under algo_extras['pareto']. Our solvers and PDOT stop "
                        "at --tolerance; the other baselines stop at their own "
                        "defaults.")
    p.add_argument("--pareto-sidecar-dir", type=Path, default=None,
                   help="Where SinkhornGrid writes its (f, g) sidecars under "
                        "--pareto (default <output-dir>/runs/pareto_sidecars). "
                        "The aggregation does not need them: without a sidecar "
                        "it fills SinkhornGrid's dual residuals analytically.")
    p.add_argument("--time-limit", type=int, default=1800,
                   help="Per-run wall-time limit in seconds (default 1800, the "
                        "strict roof: a run past it is a timeout).")
    p.add_argument("--iter-cap",   type=int, default=1_000_000)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH,
                   help="The DOTmark index (default). Additional non-DOTmark "
                        "datasets are added via --datasets / --index-paths.")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Datasets to schedule, and only these (list dotmark "
                        "to keep DOTmark). Each other name (modelnet10, "
                        "random_dense) loads "
                        "data/<name>/index.json, built by `python -m "
                        "src.instances.registry`. Each instance carries its rung "
                        "(Grid, PointCloud or General) and is lifted to the "
                        "solvers that consume it.")
    p.add_argument("--index-paths", nargs="+", type=Path, default=None,
                   help="Explicit extra index.json paths to merge (alternative to "
                        "--datasets).")
    p.add_argument("--rungs", nargs="+", default=None,
                   choices=list(REGIMES) + list(_LEGACY_REGIME_CLI),
                   help="Restrict to instances of these NATIVE rungs "
                        "(Grid/PointCloud/General; the aliases grid/geometric/"
                        "dense are accepted). Distinct from --regimes, which "
                        "filters ALGORITHMS.")
    p.add_argument("--sizes", nargs="+", type=int, default=None,
                   help="Select instances by marginal length N. This is the "
                        "size filter of a Point Cloud or General dataset, whose "
                        "entries carry no resolution or pair, so --resolutions "
                        "and --pairs pass all of them.")
    p.add_argument("--max-per-cell", type=int, default=None,
                   help="Cap instances per (dataset, N, norm), taken in "
                        "instance_id order. Use with --sizes to thin an "
                        "EXPENSIVE tier rather than dropping it: cutting tiers "
                        "loses the top of a scaling curve, while capping within "
                        "a tier keeps it and only widens the error bar.")
    p.add_argument("--check-every", type=int, default=None,
                   help="Override the check cadence of our three solvers. "
                        "Default: the per-n_g table "
                        "src/runners/bdrs_common.CHECK_EVERY_BY_N.")
    p.add_argument("--policy-exception", nargs="+", default=None,
                   metavar="REGIME:NORM",
                   help="Widen ONE regime's benchmark norm policy for ONE "
                        "ground cost, e.g. 'General:sql2', which runs the same "
                        "DOTmark sql2 instance on all three formats (sql2 is "
                        "not in the General policy). The per-algorithm "
                        "restrictions (HOT, HALO and M3S to sql2, LAMPPointCloud "
                        "to l1, sql2 and inf) are never widened. Recorded in "
                        "header.json.")
    p.add_argument("--geo-dense-cap", type=int, default=4096,
                   help="Largest number of points m for which a Point Cloud "
                        "instance runs on a General solver, whose m x m cost is "
                        "materialised in fp64. Native General instances and the "
                        "Point Cloud solvers are not capped.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pdot-driver",  type=Path, default=DEFAULT_DRIVER)
    p.add_argument("--pdot-project", type=Path, default=DEFAULT_PDOT_DIR)
    args = p.parse_args(argv)
    args.algos = [canonical_algorithm(a) for a in args.algos]
    # LAMP's server outlives each runner and holds a CUDA context while idle, so
    # another solver timed in the same job could overlap an orphaned LAMP solve
    # or count the server's memory in its NVML peak.
    lamp_algos = [a for a in args.algos if a in LAMP_ALGOS]
    if lamp_algos and len(lamp_algos) != len(args.algos):
        p.error(f"run {lamp_algos} in its own orchestrator job, without "
                f"{[a for a in args.algos if a not in LAMP_ALGOS]}")
    if args.regimes is not None:
        args.regimes = [canonical_regime(r) for r in args.regimes]
    if args.rungs is not None:
        args.rungs = [canonical_regime(r) for r in args.rungs]
    if args.spine_first and args.sample_first:
        p.error("--spine-first and --sample-first are mutually exclusive")
    # Sinkhorn is General-format: when --regimes omits General every Sinkhorn
    # arm is filtered out, and no ε is needed.
    sinkhorn_can_run = "Sinkhorn" in args.algos and (
        args.regimes is None or GENERAL in args.regimes
    )
    if sinkhorn_can_run and not args.sinkhorn_eps:
        p.error(
            "--sinkhorn-eps is required when --algos includes Sinkhorn "
            "(e.g. --sinkhorn-eps 0.1 0.01 0.001)"
        )
    if args.sinkhorn_eps and "Sinkhorn" not in args.algos:
        print(
            "orchestrator: --sinkhorn-eps was provided but 'Sinkhorn' is not in "
            "--algos; the eps values will be ignored.",
            file=sys.stderr,
        )
    for eps in args.sinkhorn_eps:
        if eps <= 0:
            p.error(f"--sinkhorn-eps values must be positive; got {eps}")
    # Likewise for SinkhornGrid, which is Grid-format.
    sinkhorngrid_can_run = "SinkhornGrid" in args.algos and (
        args.regimes is None or GRID in args.regimes
    )
    if sinkhorngrid_can_run and not args.ott_eps:
        p.error(
            "--ott-eps is required when --algos includes SinkhornGrid "
            "(e.g. --ott-eps 0.1 0.01 0.001)"
        )
    if args.ott_eps and "SinkhornGrid" not in args.algos:
        print(
            "orchestrator: --ott-eps was provided but 'SinkhornGrid' is not in "
            "--algos; the eps values will be ignored.",
            file=sys.stderr,
        )
    for eps in args.ott_eps:
        if eps <= 0:
            p.error(f"--ott-eps values must be positive; got {eps}")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_path = runs_dir / "orchestrator.log"

    # Parsed before any run so a malformed exception fails early; recorded in
    # header.json so the run is marked as off-policy.
    policy_exceptions = parse_policy_exceptions(args.policy_exception or [])
    if policy_exceptions:
        print(
            "orchestrator: NORM-POLICY EXCEPTION in effect: "
            + ", ".join(f"{r}:{n}" for r, n in sorted(policy_exceptions))
            + " (widens the regime policy only; structural per-algorithm "
              "restrictions are unaffected)",
            file=sys.stderr,
        )

    env = env_capture.write(output_dir / "header.json", REPO_ROOT)
    if policy_exceptions:
        _hdr_path = output_dir / "header.json"
        _hdr = json.loads(_hdr_path.read_text())
        _hdr["norm_policy_exceptions"] = sorted(
            f"{r}:{n}" for r, n in policy_exceptions)
        _hdr_path.write_text(json.dumps(_hdr, indent=2))
    print(
        f"orchestrator: header captured (sha={env.get('fingerprint_sha256','?')[:12]}); "
        f"output_dir={output_dir}",
        file=sys.stderr,
    )

    # --datasets both loads data/<name>/index.json and restricts the schedule
    # to the named datasets: `--datasets modelnet10` runs no DOTmark instance,
    # `--datasets dotmark modelnet10` runs both.
    index_paths: List[Path] = [args.index_path]
    if args.datasets:
        index_paths += [REPO_ROOT / "data" / d / "index.json"
                        for d in args.datasets if d != "dotmark"]
    if args.index_paths:
        index_paths += list(args.index_paths)
    index = load_all_indexes(index_paths)
    # DOTmark l2/inf instances missing from the index are derived from its
    # Grid entries, with no cost file (each runner builds the lattice cost).
    # Other datasets index each of their costs themselves.
    dotmark_grid = [e for e in index
                    if e.get("dataset") == "dotmark" and (e.get("rung") or GRID) == GRID]
    for dense_norm in DENSE_ONLY_NORMS:
        if dense_norm in args.norms:
            extra = [_normalised_index_entry(e)
                     for e in synthesize_dense_norm_entries(dotmark_grid, dense_norm)]
            index = index + extra
            print(
                f"orchestrator: synthesized {len(extra)} {dense_norm!r} "
                f"DOTmark instances from the index",
                file=sys.stderr,
            )
    filt = filter_index(
        index,
        classes=args.classes,
        resolutions=args.resolutions,
        norms=args.norms,
        sizes=args.sizes, max_per_cell=args.max_per_cell,
        pairs=args.pairs,
        rungs=args.rungs,
        datasets=args.datasets,
    )
    expanded_algos: List[str] = []
    for a in args.algos:
        if a == "Sinkhorn":
            for eps in args.sinkhorn_eps:
                expanded_algos.append(eps_variant_name(SINKHORN_PREFIX, float(eps)))
        elif a == "SinkhornGrid":
            for eps in args.ott_eps:
                expanded_algos.append(eps_variant_name(SINKHORNGRID_PREFIX, float(eps)))
        else:
            expanded_algos.append(a)
    if expanded_algos != list(args.algos):
        print(
            f"orchestrator: expanded --algos {list(args.algos)} → {expanded_algos}",
            file=sys.stderr,
        )
    if args.regimes:
        wanted = set(args.regimes)
        kept = [a for a in expanded_algos if regime_for_algorithm(a) in wanted]
        if kept != expanded_algos:
            print(
                f"orchestrator: --regimes {args.regimes} filtered algos "
                f"{expanded_algos} → {kept}",
                file=sys.stderr,
            )
        expanded_algos = kept

    schedule = build_schedule(
        filt,
        algos=expanded_algos,
        spine_first=args.spine_first,
        sample_first=args.sample_first,
        head_only=args.head_only,
        policy_exceptions=policy_exceptions,
    )
    cap_str = (f"; will run at most {args.max_instances} (--max-instances)"
               if args.max_instances is not None else "")
    print(
        f"orchestrator: schedule has {len(schedule)} (algo, instance) pairs"
        f"{cap_str}",
        file=sys.stderr,
    )

    done = set() if args.no_resume else load_done_set(runs_dir)
    if done:
        print(f"orchestrator: skipping {len(done)} already-settled (instance, algo) pairs", file=sys.stderr)

    # Checked after the --regimes filter, so a run without PDOT does not pay
    # the Julia/CUDA JIT of starting the adapter.
    pdot_adapter: Optional[PdotAdapter] = None
    if "PDOT" in expanded_algos:
        try:
            pdot_adapter = PdotAdapter(args.pdot_driver, args.pdot_project)
            print("orchestrator: PDOT adapter ready", file=sys.stderr)
        except (FileNotFoundError, RuntimeError, TimeoutError) as e:
            print(
                f"orchestrator: PDOT adapter failed to start: {e!r} -- "
                f"PDOT instances will be recorded as 'error'.",
                file=sys.stderr,
            )

    # Specs, records and lifted inputs; removed on exit.
    scratch = Path(tempfile.mkdtemp(prefix="bdrs_orchestrator_"))
    # Under --pareto, SinkhornGrid writes its (f, g) to
    # `<spec.output_path>.pareto_fg.npz` for the dual residuals that
    # aggregation fills in later; those files must outlive `scratch`, so
    # spec.output_path points here.
    pareto_sidecar_dir = (Path(args.pareto_sidecar_dir)
                          if getattr(args, "pareto_sidecar_dir", None)
                          else runs_dir / "pareto_sidecars")
    if getattr(args, "pareto", False):
        pareto_sidecar_dir.mkdir(parents=True, exist_ok=True)
    # instance_id -> path of the materialised C of a lifted Point Cloud instance
    geo_cache: Dict[str, str] = {}
    n_lift_skipped = 0

    started_at = time.time()
    n_total = len(schedule)
    n_done_pre = len(done)

    try:
        ran = 0
        for idx, (algo, ent) in enumerate(schedule, start=1):
            instance_id = ent["instance_id"]
            if (instance_id, algo, float(args.tolerance)) in done:
                continue
            if args.max_instances is not None and ran >= args.max_instances:
                break

            output_jsonl = runs_dir / f"{algo.lower()}.jsonl"
            # spec.output_path only names side files; run_subprocess reads the
            # record from a path of its own.
            _sidecar_base = (pareto_sidecar_dir if getattr(args, "pareto", False)
                             else scratch)
            spec_output_path = _sidecar_base / f"{algo.lower()}_{instance_id}.out.json"
            # Built from the unlifted entry first, so the error handler below
            # has a spec even when the lift raises.
            spec = _spec_from_index(ent, args=args, output_path=spec_output_path)
            t_start = time.perf_counter()
            try:
                # The lift writes to scratch, so it is inside the try: an I/O
                # error records one error row and the run continues. A pair
                # the runner cannot take (prepared None) is skipped without a row.
                prepared, skip_reason = _prepare_ent_for_algo(
                    ent, algo, scratch=scratch, geo_cache=geo_cache,
                    geo_dense_cap=args.geo_dense_cap)
                if prepared is None:
                    n_lift_skipped += 1
                    print(f"orchestrator: skip {algo} on {instance_id}: {skip_reason}",
                          file=sys.stderr)
                    continue
                spec = _spec_from_index(prepared, args=args, output_path=spec_output_path)

                if algo == "PDOT":
                    if pdot_adapter is None:
                        rec = _make_synthesized_record(
                            spec, "PDOT", status="error", error_kind="adapter_unavailable",
                            error_msg="PDOT adapter failed to start at orchestrator startup",
                        )
                    else:
                        rec = pdot_adapter.run(spec)
                else:
                    rec = run_subprocess(algo, spec, scratch, time_limit_s=args.time_limit)
            except KeyboardInterrupt:
                print("orchestrator: interrupted; the rows already appended are kept",
                      file=sys.stderr)
                raise
            except Exception as e:                          # noqa: BLE001
                rec = _make_synthesized_record(
                    spec, algo, status="error", error_kind=type(e).__name__,
                    error_msg=repr(e),
                )

            # The row carries the instance's native dataset and rung (and m,
            # when the entry has one) over the runner's values: a General run
            # on a lifted Point Cloud instance is recorded with rung PointCloud.
            rec.dataset = ent.get("dataset", "dotmark")
            rec.rung = ent.get("rung") or GRID
            if ent.get("m") is not None:
                rec.m = ent["m"]
            # Every row records its time limit, which the aggregation compares
            # with t_solve. Dispatch and the PDOT adapter stamp the rows they
            # return; this also covers the rows built above.
            _stamp_time_limit(rec, spec)

            # A row without a git SHA or device UUID gets the orchestrator's: the
            # UUID of its current CUDA device, which is the run's device only
            # under --device cuda:0 (PDOT's driver always runs on device 0).
            if getattr(rec, "git_sha", None) is None:
                rec.git_sha = env.get("git_sha")
            if rec.device_uuid is None:
                rec.device_uuid = env_capture.cached_device_uuid()

            append_record(output_jsonl, rec)
            elapsed = time.perf_counter() - t_start
            print(
                f"[{idx}/{n_total}] {algo} {instance_id} status={rec.status} "
                f"wall={rec.wall_time_s} feaserr={rec.feaserr} elapsed={elapsed:.1f}s",
                file=sys.stderr,
            )
            with log_path.open("a") as f:
                f.write(json.dumps({
                    "ts": utc_now_iso(),
                    "algo": algo, "instance_id": instance_id,
                    "regime": rec.regime,
                    "status": rec.status, "elapsed_s": elapsed,
                    "header_sha": env.get("fingerprint_sha256"),
                }) + "\n")
            ran += 1

        total_elapsed = time.time() - started_at
        skip_str = (f"; {n_lift_skipped} lift-skipped (unsupported rung lift)"
                    if n_lift_skipped else "")
        print(
            f"orchestrator: done. ran={ran} (was {n_done_pre} settled before)"
            f"{skip_str}. total wall: {total_elapsed:.1f}s",
            file=sys.stderr,
        )
        return 0

    finally:
        if pdot_adapter is not None:
            pdot_adapter.shutdown()
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
