"""
Runner for HOT, a Halpern-accelerated ADMM solver for optimal transport on a 2-D
grid with the separable sql2 cost (vendor/HOT/HOT_Solver.py, imported by path).

HOT solves a reduced LP: mass goes from source i to target j through an
intermediate node k, moving along one grid axis and then the other. The variable
is a flow x = (x¹, x²) of 2·n_g³ entries with A x = b, b = (0, r, c) (balance at
the N intermediate nodes, source outflows, target inflows), x ≥ 0, and the cost
of path i → k → j is C_ij. ``optimize()`` rescales b by 1/b_scale and the cost w
by 1/c_scale; ``x_bar`` (x̄) and ``y_bar`` (ȳ) are its iterates on the rescaled
problem.

X is the plan HOT's ``recover_transport_plan()`` builds from its Halpern
iterate x, at every n_g. That routine forms X densely (8N² bytes: 34 GB at
n_g = 256) and overwrites x, so the runner does not call it: it evaluates the
same X's row sums, column sums and ⟨C, X⟩ in O(n_g³) (``HotRecoveredPlan``)
and leaves the solver state unchanged.
"""

import inspect
import sys
import time
import uuid
from typing import Optional, Tuple

import numpy as np
import torch

from src.protocol.paths import REPO_ROOT
from src.protocol.instance_record import InstanceSpec, ResultRecord, utc_now_iso
from src.scoring.hot_plan import HotRecoveredPlan
from src.scoring.plan_metrics import evaluate_plan_metrics
from src.runners.gpu_state import device_fingerprint
from src.protocol.regimes import GRID
from src.runners.common import (
    WARMUP_SEED,
    cuda_sync,
    failure_record,
    finite_or_none,
    load_marginal,
    runner_main,
    warmup_marginal,
)


_HOT_DIR = REPO_ROOT / "vendor" / "HOT"
if str(_HOT_DIR) not in sys.path:
    sys.path.insert(0, str(_HOT_DIR))

try:
    from HOT_Solver import HOTSolver       # type: ignore
except ImportError as _e:                   # pragma: no cover - first-import diagnostic
    HOTSolver = None
    _HOT_IMPORT_ERROR = repr(_e)
else:
    _HOT_IMPORT_ERROR = None


# t_solve is the optimize() call alone: HOT's rescaling, iterations and KKT
# checks. HOTSolver.__init__ copies b and w (w has 2·n_g³ entries) to the device
# and allocates the iterates; it is in neither clock (``wall_time_construct_s``).
T_SOLVE_BOUNDARY = "hot_optimize_call"
T_SOLVE_INCLUDES: Tuple[str, ...] = ()
T_SOLVE_EXCLUDES = ("warmup", "marginals_load", "hot_b_c_construction",
                    "hotsolver_construct_h2d_and_alloc")
# t_total adds copying (α, β) out of ȳ and the recovery, which yields X's row
# sums, column sums and ⟨C, X⟩ in the same O(n_g³) pass.
T_TOTAL_BOUNDARY = "hot_optimize_plus_dual_slice_plus_northwest_recovery"
T_TOTAL_INCLUDES = ("plan_row_sums", "plan_col_sums", "plan_objective")

# HOT's period for its KKT check and σ restart (its constructor default).
DEFAULT_CHECK_FREQ = 100


def _hot_signature_default_tolerance() -> float:
    """HOT's tolerance when the keyword is omitted: the default in the signature
    of ``HOTSolver.__init__`` (1e-6 when it cannot be read).

    Not the 1e-4 in the constructor body, which applies only to an explicit
    ``tolerance=None``."""
    if HOTSolver is None:
        return 1e-6
    try:
        d = inspect.signature(HOTSolver.__init__).parameters["tolerance"].default
        return float(d)
    except Exception:                                       # noqa: BLE001
        return 1e-6


HOT_SIGNATURE_DEFAULT_TOLERANCE = _hot_signature_default_tolerance()


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str) -> ResultRecord:
    return failure_record(spec, "HOT", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid)


def _load_marginals(spec: InstanceSpec) -> Tuple[np.ndarray, np.ndarray]:
    """The stored marginals, unfloored: HOT takes no logarithm, so zero entries
    are safe."""
    return load_marginal(spec.marginal_i_path), load_marginal(spec.marginal_j_path)


def _build_b_c_cMax(spec: InstanceSpec) -> Tuple[np.ndarray, np.ndarray, float]:
    """HOT's inputs b = (0, r, c), w and C_max, laid out as ``generate_LP_param``
    in vendor/HOT/demo.py lays them out."""
    m = n = spec.resolution

    r_np, c_np = _load_marginals(spec)
    c_obj, c_Max = _hot_cost(m, n)
    b = np.concatenate([np.zeros(m * n, dtype=np.float64), r_np, c_np])
    return b, c_obj, c_Max


def _hot_cost(m: int, n: int) -> Tuple[np.ndarray, float]:
    """HOT's cost w (the squared axis displacements of x¹ and x²) divided by
    C_max = (m − 1)² + (n − 1)², and C_max."""
    c1 = np.repeat(np.arange(0, m)[np.newaxis, :], m, axis=0)
    c1 = np.square(c1 - np.arange(0, m)[:, np.newaxis])
    c1 = c1.flatten(order="C")
    c1 = np.tile(c1, n)
    assert c1.size == m * m * n

    c2 = np.repeat(np.arange(0, n)[np.newaxis, :], n, axis=0)
    c2 = np.square(c2 - np.arange(0, n)[:, np.newaxis])
    c2 = c2.flatten(order="F")
    c2 = np.repeat(c2, m)
    assert c2.size == n * n * m

    c_obj = np.concatenate([c1, c2]).astype(np.float64)
    c_Max = float(np.max(c1) + np.max(c2))
    if c_Max <= 0:
        raise ValueError(f"HOT input construction: c_Max={c_Max} non-positive at n={n}")
    c_obj /= c_Max
    return c_obj, c_Max


def _warmup(device: torch.device) -> None:
    """An untimed solve on a random 32 × 32 instance."""
    if HOTSolver is None:
        return
    m = n = 32
    rng = np.random.default_rng(WARMUP_SEED)
    sup = warmup_marginal(rng, m * n)
    cus = warmup_marginal(rng, m * n)
    c_obj, c_Max = _hot_cost(m, n)
    b = np.concatenate([np.zeros(m * n, dtype=np.float64), sup, cus])
    solver = HOTSolver(
        b=b, c=c_obj, m=m, n=n, cMax=c_Max,
        max_iters=500, tolerance=1e-4, check_freq=100,
        sigma=None, adjust_sigma=True, logging=True,
        dtype=torch.float64, device=str(device),
    )
    try:
        solver.optimize()
    except Exception:                                       # noqa: BLE001
        # the timed solve reports the error
        pass


def _run_one(spec: InstanceSpec) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    n_total = spec.resolution * spec.resolution

    if HOTSolver is None:
        return _make_failure_record(
            spec, status="error", error_kind="hot_import_failed",
            error_msg=_HOT_IMPORT_ERROR or "HOTSolver could not be imported",
            started_utc=started_utc, run_uid=run_uid,
        )
    if spec.norm != "sql2":
        return _make_failure_record(
            spec, status="error", error_kind="hot_l1_unsupported",
            error_msg=(
                "HOT's reduced model relies on the separability of squared-L₂ "
                f"on a 2D grid, so HOT runs on sql2 only, not {spec.norm!r}."
            ),
            started_utc=started_utc, run_uid=run_uid,
        )
    # "cpu" is for tests; HOT is device-agnostic PyTorch.
    device = torch.device(spec.device if spec.device.startswith("cuda") else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        return _make_failure_record(
            spec, status="error", error_kind="no_cuda",
            error_msg="torch.cuda.is_available() is False; HOT requires CUDA.",
            started_utc=started_utc, run_uid=run_uid,
        )

    if spec.warmup > 0:
        _warmup(device)
        cuda_sync(device)

    try:
        b, c_obj, c_Max = _build_b_c_cMax(spec)
    except (MemoryError, ValueError) as e:
        return _make_failure_record(
            spec, status="error", error_kind="input_construction_failed",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
        )

    pareto = bool(getattr(spec, "pareto", False))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cuda_sync(device)
    log = None
    t_construct = None
    try:
        # Under --pareto the tolerance keyword is omitted, so HOT stops at its
        # own signature default.
        _hot_kw = dict(
            b=b, c=c_obj, m=spec.resolution, n=spec.resolution, cMax=c_Max,
            max_iters=spec.iter_cap, check_freq=DEFAULT_CHECK_FREQ,
            sigma=None, adjust_sigma=True, logging=True,
            dtype=torch.float64, device=str(device),
        )
        if not pareto:
            _hot_kw["tolerance"] = spec.tolerance
        _tc0 = time.perf_counter()
        solver = HOTSolver(**_hot_kw)
        cuda_sync(device)
        t_construct = time.perf_counter() - _tc0
        t0 = time.perf_counter()
        log = solver.optimize()
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_solve",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
        )
    cuda_sync(device)
    t1 = time.perf_counter()
    wall_solve = t1 - t0
    tol_used = (HOT_SIGNATURE_DEFAULT_TOLERANCE if pareto else float(spec.tolerance))

    # ȳ = (λ, α̂, β̂) in the row order of b. Dividing w by c_scale divides the
    # dual by c_scale, so (α, β) = c_scale·(α̂, β̂) is a dual for w. Where ȳ is
    # feasible, summing the reduced constraints λ_k + α_i ≤ w¹ and
    # −λ_k + β_j ≤ w² along i → k → j gives α_i + β_j ≤ C_ij.
    _pareto_alpha = _pareto_beta = None
    _dual_extract_s = 0.0
    if pareto:
        cuda_sync(device)
        _td0 = time.perf_counter()
        _mn = int(solver.mn)
        _cs = float(solver.c_scale)
        _pareto_alpha = (solver.y_bar[_mn:2 * _mn].clone().to(torch.float64) * _cs)
        _pareto_beta = (solver.y_bar[2 * _mn:3 * _mn].clone().to(torch.float64) * _cs)
        cuda_sync(device)
        _dual_extract_s = time.perf_counter() - _td0

    # HOT's own primal error on x = b_scale·x̄:
    # max(‖Ax − b‖₂/(1 + ‖b‖₂), ‖min(x, 0)‖₂/(1 + ‖min(x, 0)‖₂)), the second
    # denominator as the shipped code computes it. HOT's stop does not test it.
    # It writes only scratch buffers (ATy_c, Ax_bar), not x̄, ȳ or x.
    solver.compute_feaserr()
    hot_feaserr = finite_or_none(float(solver.feaserr))

    # flag is 0 iff HOT's KKT test passed (before the loop or at a check), 1 when
    # max_iters ended the loop.
    hot_flag = int(getattr(solver, "flag", 1))
    native_termination = "kkt_below_tol" if hot_flag == 0 else "max_iters"

    if not isinstance(log, dict):
        return _make_failure_record(
            spec, status="error", error_kind="hot_no_log",
            error_msg="HOTSolver.optimize did not return a log dict (logging=True required)",
            started_utc=started_utc, run_uid=run_uid,
        )

    iters_run = int(log.get("iter", solver.iter_count))
    iter_cap_hit = iters_run >= spec.iter_cap
    # HOT sets pr_obj, dual_obj, gap and kkt_error at its checks only, so when
    # the cap ends the loop between checks (spec.iter_cap not a multiple of
    # DEFAULT_CHECK_FREQ) they belong to the last check, not the final iterate.
    pr_obj_raw   = float(log.get("pr_obj"))
    dual_obj_raw = float(log.get("dual_obj"))
    kkt_err  = float(log.get("kkt_error"))
    # HOT's relative gap abs(d − p)/(1 + abs(d) + abs(p)) on its raw-unit
    # objectives: the "1 +" makes it depend on the cost scale, so it is a
    # diagnostic only.
    hot_internal_gap = float(log.get("gap"))

    # pr_obj = ⟨w, b_scale·x̄⟩·C_max is in raw integer-cost units; dual_obj
    # carries the same factor.
    pr_obj_max1 = pr_obj_raw / c_Max
    dual_obj_max1 = dual_obj_raw / c_Max

    cuda_sync(device)
    t2 = time.perf_counter()
    try:
        # The plan recover_transport_plan() builds from the Halpern iterate x
        # (not x̄, on which HOT's KKT test runs); every baseline is scored on
        # what its own recovery returns. x is read, not modified.
        plan = HotRecoveredPlan(solver.x, solver.b_scale, spec.resolution)
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom_recovery", error_kind="cuda_oom_recovery",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
        )
    cuda_sync(device)
    wall_recovery = time.perf_counter() - t2

    r_np, c_np = _load_marginals(spec)
    r_dev = torch.as_tensor(r_np, dtype=torch.float64, device=device)
    c_dev = torch.as_tensor(c_np, dtype=torch.float64, device=device)
    t4 = time.perf_counter()
    metrics = evaluate_plan_metrics(plan, r_dev, c_dev)
    cuda_sync(device)
    wall_eval = time.perf_counter() - t4
    r_P = float(metrics["feaserr"])
    objective = float(metrics["objective"])

    pareto_block = None
    if pareto and _pareto_alpha is not None:
        from src.scoring.pareto_score import score as _pareto_score
        from src.scoring.pareto_score import sanitize_block as _sanitize
        _te0 = time.perf_counter()
        # X ≥ 0 because the recovery clamps x before gluing it.
        _blk = _pareto_score(plan, _pareto_alpha, _pareto_beta, r_dev, c_dev,
                             ng=spec.resolution, cost_type="sql2",
                             x_nonneg_structural=True, compute_frobenius=True)
        cuda_sync(device)
        pareto_block = _sanitize({
            **_blk,
            "t_solve": float(wall_solve),
            "t_total": float(wall_solve + _dual_extract_s + wall_recovery),
            "t_eval": float(time.perf_counter() - _te0),
            "pareto_partial": False,
            "precision": "fp64",
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_total_boundary": T_TOTAL_BOUNDARY,
            "t_total_includes": list(T_TOTAL_INCLUDES),
            "native_stop": f"hot_signature_default_tol_{tol_used:g}",
            "warmup": ("generic_32sq_discarded" if spec.warmup > 0 else "none"),
        })

    if device.type == "cuda":
        peak_mem: Optional[int] = int(torch.cuda.max_memory_allocated(device))
        peak_mem_source: Optional[str] = "torch_cuda_max"
        fp = device_fingerprint(idx=device.index or 0)
    else:
        peak_mem, peak_mem_source, fp = None, None, {}

    rec = ResultRecord(
        instance_id=spec.instance_id, algorithm="HOT", regime=GRID,
        norm=spec.norm, resolution=spec.resolution, n=n_total,
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status="ok",
        wall_time_s=wall_solve + wall_recovery + wall_eval,
        wall_time_solve_s=wall_solve,
        wall_time_recovery_s=wall_recovery,
        iters=iters_run,
        iter_cap_hit=iter_cap_hit,
        native_termination=native_termination,
        tolerance=spec.tolerance,
        feaserr=hot_feaserr,
        native_duality_gap=None,           # HOT's gap is relative, in raw units (hot_internal_gap)
        objective=objective,
        peak_gpu_mem_bytes=peak_mem,
        peak_gpu_mem_source=peak_mem_source,
        device_name=fp.get("device_name"),
        device_uuid=fp.get("device_uuid"),
        algo_extras={
            "feaserr_measure": "hot_compute_feaserr",
            "r_P": r_P,
            "hot_internal_gap": hot_internal_gap,
            "hot_kkt_error": kkt_err,
            "hot_pr_obj": pr_obj_max1,
            "hot_dual_obj": dual_obj_max1,
            "hot_pr_obj_raw": pr_obj_raw,
            "hot_dual_obj_raw": dual_obj_raw,
            "hot_c_max": float(c_Max),
            "objective_units": "max1_cost",
            "hot_flag": hot_flag,
            "hot_tolerance_used": tol_used,
            "hot_tolerance_source": ("signature_default" if pareto else "spec"),
            "certify_unfloored": True,
            "hot_plan_recovery": "recover_transport_plan_closed_form",
            "hot_check_freq": DEFAULT_CHECK_FREQ,
            "row_sum_err_l2": float(metrics["row_sum_err_l2"]),
            "col_sum_err_l2": float(metrics["col_sum_err_l2"]),
            "wall_time_eval_s": wall_eval,
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_total_boundary": T_TOTAL_BOUNDARY,
            "t_total_includes": list(T_TOTAL_INCLUDES),
            "wall_time_construct_s": t_construct,
            "wall_time_dual_extract_s": _dual_extract_s,
            "r_P_negpart_assumed_zero": not metrics["negpart_measured"],
            **({"pareto": pareto_block} if pareto_block is not None else {}),
        },
    )
    return rec


def main(argv=None):
    return runner_main(
        argv, description="HOT runner.",
        run=lambda spec, args: _run_one(spec),
        failure=lambda spec, args, **kw: _make_failure_record(spec, **kw))


if __name__ == "__main__":
    sys.exit(main())
