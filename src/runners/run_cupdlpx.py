"""
Runner for cuPDLPx, a GPU LP solver (restarted Halpern PDHG), on the dense DOT
LP, solved in-process through cuPDLPx's official Python binding
(``bash vendor/cupdlpx/build.sh`` installs it).

The LP is min ⟨C, x⟩ s.t. A x = (r; c), x ≥ 0, with x the plan flattened
row-major (x[i·N + j] = X_ij) and A the 2N × N² marginal operator. The dual
``model.Pi`` holds the duals of the N row constraints, then of the N column
constraints, so α = Pi[:N] and β = Pi[N:2N].
"""

import math
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from src.protocol.instance_record import InstanceSpec, ResultRecord, utc_now_iso
from src.scoring.plan_metrics import DenseTorchPlan, evaluate_plan_metrics
from src.instances.dense_cost import build_dense_cost, resolve_dense_cost
from src.runners.gpu_state import device_fingerprint
from src.runners.nvml_sampler import NvmlPeakSampler
from src.instances.dot_lp import marginal_csr
from src.runners.common import (
    WARMUP_SEED,
    cuda_ordinal,
    failure_record,
    finite_or_none,
    is_oom_exception,
    json_safe,
    load_marginal,
    runner_main,
    warmup_blobs,
)

try:
    import cupdlpx                       # official binding (vendor/cupdlpx/build.sh)
except ImportError as _e:                # pragma: no cover - first-import diagnostic
    cupdlpx = None
    _CUPDLPX_IMPORT_ERROR = repr(_e)
else:
    _CUPDLPX_IMPORT_ERROR = None


MARGINAL_FLOOR = 0.0

# t_solve is the model.optimize() call. cupdlpx.Model(...) and setParams only
# convert the arrays to the binding's dtypes and keep them on the Python
# object, so everything from copying the LP into cuPDLPx to copying X and Pi
# back into numpy arrays is inside t_solve.
T_SOLVE_BOUNDARY = "cupdlpx_optimize_call"
T_SOLVE_INCLUDES = ("pslp_presolve", "rescaling")
T_SOLVE_EXCLUDES = ("marginals_load", "cost_load", "lp_csr_assembly",
                    "warmup", "model_construct_and_setparams")
# t_total adds reading model.X and model.Pi and copying α, β to the device.
T_TOTAL_BOUNDARY = "cupdlpx_optimize_plus_X_and_Pi_readback"
T_TOTAL_INCLUDES: Tuple[str, ...] = ()

WARMUP_ITER_CAP = 200


# cuPDLPx runs iterations in blocks of termination_evaluation_frequency (200,
# left at its default) and tests iteration_limit only between blocks
# (src/solver.cu:163-214), so a positive limit below 200 still runs 200
# iterations. Raising it to 200 makes the recorded limit the one that runs.
_MIN_ITER_LIMIT = 200

# Device bytes per nonzero of A and per variable, estimated from cuPDLPx's
# allocations (src/solver.cu): per nonzero, the fp64 values of the CSR and CSC
# copies and five int32 index arrays; per variable, 13 fp64 vectors.
# Some per-variable buffers are not counted, and presolve can shrink the LP.
# A failed CUDA allocation in cuPDLPx calls exit() (CUDA_CHECK,
# internal/utils.h), which Python cannot catch, so the runner compares this
# estimate with the device memory before solving.
_DEVICE_BYTES_PER_NNZ = 36
_DEVICE_BYTES_PER_VAR = 104


def _estimate_device_bytes(N: int) -> int:
    nnz = 2 * N * N
    return nnz * _DEVICE_BYTES_PER_NNZ + N * N * _DEVICE_BYTES_PER_VAR


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str) -> ResultRecord:
    return failure_record(spec, "CUPDLPX", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, m=spec.dense_shape()[0])


def build_dot_lp(C_flat: np.ndarray, r: np.ndarray, c: np.ndarray
                 ) -> Tuple[np.ndarray, "sp.csr_matrix", np.ndarray,
                            np.ndarray, np.ndarray, None]:
    """The DOT LP in cuPDLPx's form min ⟨c_vec, x⟩ s.t. con_lb ≤ A x ≤ con_ub,
    var_lb ≤ x ≤ var_ub; ``var_ub = None`` is +∞."""
    N = int(r.size)
    nnz = 2 * N * N
    if nnz >= 2 ** 31:
        # cuPDLPx stores nnz and the CSR offsets as int32, and the binding
        # casts them unchecked (python/cupdlpx/model.py:44-47), so they would
        # wrap. Tested before the shapes, so no N²-entry C_flat is needed to
        # reach it.
        raise OverflowError(
            f"LP nnz = 2N² = {nnz} >= 2^31 does not fit cuPDLPx's int32 CSR "
            f"indices")
    if c.size != N or C_flat.size != N * N:
        raise ValueError(
            f"shape mismatch: r={r.size}, c={c.size}, C_flat={C_flat.size}")

    c_vec = np.ascontiguousarray(C_flat, dtype=np.float64)
    A = marginal_csr(N)

    b = np.concatenate([np.asarray(r, dtype=np.float64),
                        np.asarray(c, dtype=np.float64)])
    # Explicit: cuPDLPx fills a missing lower bound with −∞
    # (src/cupdlpx.c:109), not the 0 its Python README states.
    var_lb = np.zeros(N * N, dtype=np.float64)
    return c_vec, A, b, b.copy(), var_lb, None


def _load_marginals(spec: InstanceSpec) -> Tuple[np.ndarray, np.ndarray]:
    """The stored marginals with no floor δ_m: an exact zero is a valid
    equality right-hand side."""
    return load_marginal(spec.marginal_i_path), load_marginal(spec.marginal_j_path)


def _load_cost_flat(spec: InstanceSpec) -> np.ndarray:
    """C flattened row-major as a contiguous fp64 array."""
    C_t = resolve_dense_cost(spec, device="cpu", dtype=torch.float64)
    return np.ascontiguousarray(C_t.numpy()).ravel()


def _set_solver_params(model: "cupdlpx.Model", spec: InstanceSpec,
                       *, iter_limit: int, time_limit_s: float) -> None:
    """Set the iteration and time limits and, outside Pareto mode, both
    relative tolerances to ``spec.tolerance``. In Pareto mode the tolerances
    keep the library default 1e-4; every other parameter keeps its default."""
    params = dict(
        iteration_limit=int(max(iter_limit, _MIN_ITER_LIMIT)),
        time_sec_limit=float(time_limit_s),
    )
    if not getattr(spec, "pareto", False):
        params["eps_optimal_relative"] = float(spec.tolerance)
        params["eps_feasible_relative"] = float(spec.tolerance)
    model.setParams(**params)


def _effective_eps(model: "cupdlpx.Model") -> Tuple[Optional[float], Optional[float]]:
    """The relative tolerances the solve ran with, or (None, None)."""
    try:
        return (float(model.Params["eps_optimal_relative"]),
                float(model.Params["eps_feasible_relative"]))
    except Exception:                                       # noqa: BLE001
        return None, None


def _library_default_params() -> Optional[Dict[str, Any]]:
    """cuPDLPx's default parameters, or None when the binding lacks them."""
    try:
        from cupdlpx._core import get_default_params
        return dict(get_default_params())
    except Exception:                                       # noqa: BLE001
        return None


def _native_stop_label(eps_opt: Optional[float], eps_feas: Optional[float],
                       defaults: Optional[Dict[str, Any]] = None
                       ) -> Tuple[str, Optional[bool]]:
    """``(native_stop label, whether both tolerances are the library
    defaults)``; the flag is None when that cannot be decided."""
    if eps_opt is None or eps_feas is None:
        return "cupdlpx_rel_kkt_unknown", None
    label = f"cupdlpx_rel_kkt_opt_{float(eps_opt):g}_feas_{float(eps_feas):g}"
    if not defaults:
        return label, None
    try:
        is_default = (float(eps_opt) == float(defaults["eps_optimal_relative"])
                      and float(eps_feas) == float(defaults["eps_feasible_relative"]))
    except (KeyError, TypeError, ValueError):
        return label, None
    return label, bool(is_default)


# The warmup only pays a process's one-time costs (CUDA context, kernel and
# library loading), which do not grow with the LP, so a small fixed LP serves
# every instance.
WARMUP_NG = 32
# Not spec.norm: a procedural instance has no lattice cost.
WARMUP_NORM = "l1"


def _warmup(spec: InstanceSpec) -> bool:
    """One untimed solve of a dummy LP; returns whether it completed (a
    failure is swallowed)."""
    ng = WARMUP_NG
    a, b = warmup_blobs(ng, np.random.default_rng(WARMUP_SEED))
    a, b = a.ravel(), b.ravel()
    warmed = False
    try:
        C_w = build_dense_cost(ng, WARMUP_NORM, device="cpu",
                               dtype=torch.float64).numpy().ravel()
        c_vec, A, lb_con, ub_con, var_lb, var_ub = build_dot_lp(C_w, a, b)
        m = cupdlpx.Model(c_vec, A, lb_con, ub_con, var_lb, var_ub)
        _set_solver_params(m, spec, iter_limit=WARMUP_ITER_CAP,
                           time_limit_s=min(120.0, float(spec.time_limit_s)))
        m.optimize()
        del m, A
        warmed = True
    except Exception:                                       # noqa: BLE001
        pass
    return warmed


def _run_one(spec: InstanceSpec) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    # n_g; None for a procedural instance.
    ng = spec.resolution
    N = spec.dense_shape()[0]

    if cupdlpx is None:
        return _make_failure_record(
            spec, status="error", error_kind="cupdlpx_import_failed",
            error_msg=(f"import cupdlpx failed ({_CUPDLPX_IMPORT_ERROR}); "
                       "build the official binding with vendor/cupdlpx/build.sh."),
            started_utc=started_utc, run_uid=run_uid,
        )
    nnz = 2 * N * N
    if nnz >= 2 ** 31:
        # The int32 limit of build_dot_lp, tested before C is built.
        return _make_failure_record(
            spec, status="oom", error_kind="cupdlpx_int32_nnz_structural",
            error_msg=(
                f"the dense DOT LP at N={N} has nnz = 2N² = {nnz} >= 2^31: "
                "cuPDLPx stores its indices as int32 and the Python binding "
                "casts them without a range check (the host CSR alone would be "
                f"~{nnz * 12 / 2**30:.0f} GiB)."),
            started_utc=started_utc, run_uid=run_uid,
        )
    if not torch.cuda.is_available():
        return _make_failure_record(
            spec, status="error", error_kind="no_cuda",
            error_msg="torch.cuda.is_available() is False; cuPDLPx requires CUDA.",
            started_utc=started_utc, run_uid=run_uid,
        )

    est_dev = _estimate_device_bytes(N)
    total_dev = torch.cuda.get_device_properties(0).total_memory
    if est_dev > total_dev:
        return _make_failure_record(
            spec, status="oom", error_kind="cupdlpx_device_oom_structural",
            error_msg=(
                f"estimated device demand {est_dev / 2**30:.1f} GiB "
                f"(nnz·{_DEVICE_BYTES_PER_NNZ} + N²·{_DEVICE_BYTES_PER_VAR}, "
                f"counted from cuPDLPx's allocations) exceeds the device's "
                f"{total_dev / 2**30:.1f} GiB; cuPDLPx calls exit() on a failed "
                f"allocation, so the run is refused before the solve."),
            started_utc=started_utc, run_uid=run_uid,
        )

    device_index = cuda_ordinal(spec.device)

    r_np, c_np = _load_marginals(spec)

    # Host bytes: 12 per nonzero of A (fp64 value, int32 index), 8 per entry
    # of c_vec, var_lb and X.
    est_host_gib = (nnz * 12 + 3 * N * N * 8) / 2 ** 30
    print(f"run_cupdlpx: n_g={ng} N²={N*N} nnz={nnz} "
          f"(~{est_host_gib:.1f} GiB host for CSR + vectors)", file=sys.stderr)
    t_b0 = time.perf_counter()
    try:
        C_flat = _load_cost_flat(spec)
        c_vec, A, lb_con, ub_con, var_lb, var_ub = build_dot_lp(C_flat, r_np, c_np)
    except (MemoryError, OverflowError) as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cpu_oom_lp_build",
            error_msg=repr(e), started_utc=started_utc, run_uid=run_uid,
        )
    lp_build_seconds = time.perf_counter() - t_b0

    warmup_ran = False
    if spec.warmup > 0:
        warmup_ran = _warmup(spec)
        # A Model is in a reference cycle (Model.Params refers back to it), so
        # only the cycle collector frees the warmup's host arrays. cuPDLPx
        # frees its device state before optimize() returns.
        import gc
        gc.collect()
        torch.cuda.synchronize()

    # cuPDLPx allocates with cudaMalloc, which torch's memory counters do not
    # see, so the peak is polled from NVML.
    sampler = NvmlPeakSampler(device_index)
    sampler.start()
    model_build_seconds = None
    try:
        _tm0 = time.perf_counter()
        model = cupdlpx.Model(c_vec, A, lb_con, ub_con, var_lb, var_ub)
        _set_solver_params(model, spec, iter_limit=spec.iter_cap,
                           time_limit_s=float(spec.time_limit_s))
        model_build_seconds = time.perf_counter() - _tm0
        t0 = time.perf_counter()
        model.optimize()
        t1 = time.perf_counter()
    except Exception as e:                                  # noqa: BLE001
        sampler.stop()
        if is_oom_exception(e):
            return _make_failure_record(
                spec, status="oom", error_kind="cuda_oom_solve",
                error_msg=f"{e!r}\n{traceback.format_exc()[-2000:]}",
                started_utc=started_utc, run_uid=run_uid,
            )
        raise
    peak_mem, baseline_mem = sampler.stop()
    wall_solve = t1 - t0

    status_str = model.Status
    _tx0 = time.perf_counter()
    x = model.X
    wall_plan_read = time.perf_counter() - _tx0
    # With r, c ≥ 0 and Σr = Σc the DOT LP is feasible and bounded, so any
    # other status is a numerical failure of cuPDLPx or its presolve. The
    # binding returns an X array for every status, so the None test alone
    # would let it through.
    if (x is None or status_str not in ("OPTIMAL", "ITERATION_LIMIT",
                                        "TIME_LIMIT")):
        return _make_failure_record(
            spec, status="error",
            error_kind=f"cupdlpx_verdict_{status_str or 'NO_SOLUTION'}",
            error_msg=(f"Status={status_str!r} StatusCode={model.StatusCode!r} "
                       f"X={'None' if x is None else 'present'}; the DOT LP is "
                       "feasible and bounded, so this status is a solver-side "
                       "failure."),
            started_utc=started_utc, run_uid=run_uid,
        )

    t2 = time.perf_counter()
    device = torch.device(f"cuda:{device_index}")
    try:
        X_t = torch.as_tensor(np.asarray(x, dtype=np.float64)).reshape(N, N).to(device)
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_metrics",
            error_msg=repr(e), started_utc=started_utc, run_uid=run_uid,
        )
    # For a Grid instance the plan metrics rebuild the lattice cost in row
    # blocks, bitwise equal to C_flat, instead of holding C on the device.
    _Cm = (torch.as_tensor(C_flat, dtype=torch.float64, device=device).reshape(N, N)
           if spec.rung != "Grid" else None)
    rep = DenseTorchPlan(X_t, ng=ng, cost_type=spec.norm, C=_Cm)
    r_dev = torch.as_tensor(r_np, dtype=torch.float64, device=device)
    c_dev = torch.as_tensor(c_np, dtype=torch.float64, device=device)
    metrics = evaluate_plan_metrics(rep, r_dev, c_dev)
    torch.cuda.synchronize(device)
    wall_eval = time.perf_counter() - t2

    _eps_opt, _eps_feas = _effective_eps(model)
    _native_stop, _native_stop_default = _native_stop_label(
        _eps_opt, _eps_feas, _library_default_params())
    pareto_block = None
    if getattr(spec, "pareto", False):
        from src.scoring.pareto_score import score as _pareto_score, sanitize_block as _sanitize
        _tr0 = time.perf_counter()
        try:
            pi = np.asarray(model.Pi, dtype=np.float64).reshape(-1)
        except Exception:                                  # noqa: BLE001
            pi = None
        if pi is not None and pi.size >= 2 * N:
            # PSLP presolve may remove the constraint of a zero-mass row or
            # column; postsolve still returns its dual, at a value of its own
            # choosing, which enters r_D.
            alpha_d = torch.as_tensor(pi[:N], dtype=torch.float64, device=device)
            beta_d = torch.as_tensor(pi[N:2 * N], dtype=torch.float64, device=device)
            torch.cuda.synchronize(device)
            wall_recovery = time.perf_counter() - _tr0
            _te0 = time.perf_counter()
            # The scorer's lattice form covers only l1 and sql2.
            if _Cm is not None:
                _kw = {"C": _Cm}
            elif spec.norm in ("l1", "sql2"):
                _kw = {"ng": ng, "cost_type": spec.norm}
            else:
                _kw = {"C": torch.as_tensor(C_flat, dtype=torch.float64,
                                            device=device).reshape(N, N)}
            _blk = _pareto_score(X_t, alpha_d, beta_d, r_dev, c_dev,
                                 x_nonneg_structural=False, compute_frobenius=True, **_kw)
            torch.cuda.synchronize(device)
            pareto_block = _sanitize({
                **_blk,
                "t_solve": float(wall_solve),
                "t_total": float(wall_solve + wall_plan_read + wall_recovery),
                "t_eval": float(time.perf_counter() - _te0),
                "pareto_partial": False, "precision": "fp64",
                "t_solve_boundary": T_SOLVE_BOUNDARY,
                "t_solve_includes": list(T_SOLVE_INCLUDES),
                "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                "t_total_boundary": T_TOTAL_BOUNDARY,
                "t_total_includes": list(T_TOTAL_INCLUDES),
                "t_plan_read": float(wall_plan_read),
                "t_dual_recovery": float(wall_recovery),
                "native_stop": _native_stop,
                "native_stop_is_library_default": _native_stop_default,
            })
        else:
            pareto_block = {"dual_absent": True, "pareto_partial": True,
                            "t_solve": float(wall_solve), "t_total": None,
                            "t_eval": None,
                            "t_solve_boundary": T_SOLVE_BOUNDARY,
                            "t_solve_includes": list(T_SOLVE_INCLUDES),
                            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                            "t_total_boundary": T_TOTAL_BOUNDARY,
                            "t_total_includes": list(T_TOTAL_INCLUDES),
                            "native_stop": _native_stop,
                            "native_stop_is_library_default": _native_stop_default}

    iters = int(model.IterCount) if model.IterCount is not None else None
    iter_cap_hit = (status_str == "ITERATION_LIMIT") or (
        iters is not None and iters >= spec.iter_cap)
    fp = device_fingerprint(idx=device_index)

    params_dump = {k: model.Params[k] for k in model.Params.keys()}

    return ResultRecord(
        instance_id=spec.instance_id, algorithm="CUPDLPX",
        norm=spec.norm, resolution=ng, n=N, m=N,
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status="ok",
        regime="General",
        timeout_flag=(status_str == "TIME_LIMIT"),
        wall_time_s=wall_solve + wall_plan_read + wall_eval,
        wall_time_solve_s=wall_solve,
        wall_time_recovery_s=wall_plan_read,
        iters=iters,
        iter_cap_hit=iter_cap_hit,
        native_termination=status_str,
        tolerance=spec.tolerance,
        # cuPDLPx's relative primal residual ‖A x − proj_[ℓ_c, u_c](A x)‖ /
        # (1 + ‖b‖), which its stopping test uses, on the presolved LP.
        feaserr=finite_or_none(model.RelPrimalResidual),
        native_duality_gap=finite_or_none(model.RelGap),   # |p − d| / (1 + |p| + |d|)
        objective=float(metrics["objective"]),
        peak_gpu_mem_bytes=peak_mem,
        peak_gpu_mem_source=("nvml_poll" if peak_mem is not None else "unknown"),
        device_name=fp.get("device_name"),
        device_uuid=fp.get("device_uuid"),
        algo_extras={
            "cupdlpx_dtype": "fp64",
            "cupdlpx_version": getattr(cupdlpx, "__version__", None),
            "cupdlpx_params": json_safe(params_dump),
            "cupdlpx_status_code": int(model.StatusCode),
            "marginal_floor": MARGINAL_FLOOR,
            "obj_cupdlpx_native": finite_or_none(model.ObjVal),
            "dual_obj_native": finite_or_none(model.DualObj),
            "gap_abs_native": finite_or_none(model.Gap),
            "feaserr_measure": "cupdlpx_relative_primal_residual",
            "feaserr_norm": params_dump.get("optimality_norm"),
            # The same quantity on the postsolved plan and the original LP:
            # ‖(X 1 − r; Xᵀ 1 − c)‖₂ / (1 + ‖(r; c)‖₂).
            "feaserr_recomputed": (
                math.hypot(float(metrics["row_sum_err_l2"]), float(metrics["col_sum_err_l2"]))
                / (1.0 + float(torch.linalg.vector_norm(torch.cat([r_dev, c_dev]))))),
            "r_P": float(metrics["feaserr"]),
            "rel_dual_residual": finite_or_none(model.RelDualResidual),
            # cuPDLPx's own clock, which time_sec_limit is tested against: process
            # CPU time (clock()) from the first PDHG iteration, so presolve and
            # rescaling are not in it (src/solver.cu:157, src/utils.cu:251).
            "cupdlpx_runtime_native_s": (float(model.Runtime)
                                         if model.Runtime is not None else None),
            "cupdlpx_rescaling_time_s": (float(model.RescalingTime)
                                         if model.RescalingTime is not None else None),
            "lp_nnz": nnz,
            "lp_build_seconds": lp_build_seconds,
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_total_boundary": T_TOTAL_BOUNDARY,
            "t_total_includes": list(T_TOTAL_INCLUDES),
            "model_build_seconds": model_build_seconds,
            "wall_time_plan_read_s": wall_plan_read,
            "warmup_iter_cap": WARMUP_ITER_CAP if warmup_ran else None,
            "warmup_ng": WARMUP_NG if warmup_ran else None,
            "warmup_ran": warmup_ran,
            "cost_family": spec.cost_family if spec.cost_seed is not None else None,
            "cost_seed": spec.cost_seed,
            "cost_source": ("cached_artifact"
                            if (spec.cost_matrix_path
                                and Path(spec.cost_matrix_path).exists())
                            else "build_dense_cost"),
            "wall_time_eval_s": wall_eval,
            **({"pareto": pareto_block} if pareto_block is not None else {}),
            "row_sum_err_l2": float(metrics["row_sum_err_l2"]),
            "col_sum_err_l2": float(metrics["col_sum_err_l2"]),
            "nvml_baseline_bytes": baseline_mem,
            "r_P_negpart_assumed_zero": not metrics["negpart_measured"],
        },
    )


def main(argv=None):
    return runner_main(
        argv, description="cuPDLPx runner (dense regime, norm-agnostic).",
        run=lambda spec, args: _run_one(spec),
        failure=lambda spec, args, **kw: _make_failure_record(spec, **kw))


if __name__ == "__main__":
    sys.exit(main())
