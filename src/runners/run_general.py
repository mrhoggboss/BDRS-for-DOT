"""General runner: the General BDRS solver on a stored m×n cost C.

C comes from ``dense_cost.resolve_dense_cost`` (a cost file, a seeded
procedural cost, or the max-1 lattice cost, ``l2`` and ``inf`` included) and
is resolved afresh for every solve. The solve stops when the row residual
‖X1 − r‖₂ and |g_feas|, the feasible gap of the c-transformed dual (α′, β′)
on C, are below tol, and the delivered (X, α′, β′) is measured against the
original marginals. C must have max C = 1 up to rounding
(``COST_MAX_UNIT_TOL``); any other cost is refused before the solve with
``error/cost_max_not_one``.

t_solve brackets the solver call only: the warm-up, the marginal load and the
cost build (``t_cost_build``) come before it. t_total adds the recovery of
(α′, β′). On ``device="cpu"`` the torch solver runs, so the runner is
testable without a GPU.
"""

import sys
import os
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.instances.dense_cost import (
    is_procedural,
    resolve_dense_cost,
    dense_trace_CX,
)
from src.methods.eta import ETA_ALPHA, eta_from_mean, grid_cost_mean
from src.runners.gpu_state import device_fingerprint
from src.protocol.instance_record import InstanceSpec, ResultRecord, utc_now_iso
from src.scoring.plan_metrics import DenseTorchPlan, evaluate_plan_metrics
from src.protocol.regimes import GENERAL, GRID
from src.runners.bdrs_common import (
    CTRANSFORM_VARIANT,
    T_SOLVE_BOUNDARY,
    assert_dual_feasible,
    certificate_extras,
    delivered_certificate,
    native_stop_label,
    solve_params,
    _load_marginals,
    _resolve_check_every,
    _t_solve_includes,
)
from src.runners.common import (
    WARMUP_SEED,
    cuda_sync,
    failure_record,
    finite_or_none,
    runner_main,
    warmup_marginal,
)
from src.methods.dual_cleanup import (
    _dual_feas_F_storedC,
    feasible_dual_and_gap_storedC,
    make_dense_gap_stop_fn,
)
from src.methods.factored_torch import bregman_dr_solve_general_factored
from src.methods.params import BregmanDRParams
from src.scoring.kkt_residual import r_dinf_storedC


# N >= 512² is refused without an attempt: the fp64 cost alone is 550 GB.
SKIP_RESOLUTION_AT = 512
SKIP_MARGINAL_LEN_AT = SKIP_RESOLUTION_AT * SKIP_RESOLUTION_AT

# Headroom on the size of C for the O(m + n) state, the row-block LSE and
# c-transform temporaries and allocator slack. C is the solver's only m×n
# array while it iterates; the dense plan X it delivers is a second one, which
# this gate does not count (an OOM there is a cuda_oom_solve row).
MEM_HEADROOM = 1.15

T_SOLVE_EXCLUDES = ("marginals_load_h2d", "dense_cost_matrix_build", "warmup")

# The largest |max C − 1| accepted as max C = 1: four units in the last place
# of 1, so that a cost divided by a rounded closed-form maximum passes. The l2
# lattice cost, divided by the rounded √2·(n_g − 1), has max C = 1 − 2⁻⁵³ at
# n_g = 32 and 64, and no lower than 1 − 2⁻⁵² for any n_g ≤ 1024. Inside
# this tolerance the solver's Normalize step still divides C by max C, which
# moves each entry of the iterated cost, and so the objective in the stop's
# g_feas, from C by about 2⁻⁵⁰ relative at most.
COST_MAX_UNIT_TOL = 2.0 ** -50


# The factored iteration log X^k_ij = ξ_i + k·L_ij + ζ_j over the stored C;
# recorded in every row, failure rows included.
GENERAL_IMPL = "factored"


def _general_solve(C, r, c, params, **kw):
    """The CUDA session when ``_use_factored_session`` allows it, else the eager
    ``factored_torch`` solver. On CUDA with the stored-C kernels the two give
    bit-identical iterates."""
    if _use_factored_session(C):
        from src.methods.factored_cuda import bregman_dr_solve_general_factored_cuda
        return bregman_dr_solve_general_factored_cuda(C, r, c, params, **kw)
    return bregman_dr_solve_general_factored(C, r, c, params, **kw)


def _warmup_cost(m: int, n: int, device) -> torch.Tensor:
    """A seeded uniform (m, n) cost scaled to max C = 1 exactly, so the
    solver's Normalize step makes no copy."""
    gen = torch.Generator(device=device).manual_seed(WARMUP_SEED)
    C = torch.rand((m, n), generator=gen, dtype=torch.float64, device=device)
    return C.div_(C.max())


def _use_factored_session(C) -> bool:
    """Whether the CUDA session can run: CUDA, a CUDA ``C``,
    ``BDRS_FACTORED_SESSION`` not ``0``, and an extension that loads and has
    ``DenseFactoredSession``. Otherwise the eager loop runs instead of the
    solve failing; ``algo_extras["factored_session"]`` records which ran."""
    if os.environ.get("BDRS_FACTORED_SESSION", "1") == "0":
        return False
    if not (torch.cuda.is_available() and getattr(C, "is_cuda", False)):
        return False
    try:
        from src.methods import cuda_ext
        return hasattr(cuda_ext.load(), "DenseFactoredSession")
    except Exception:                                   # noqa: BLE001
        return False


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str) -> ResultRecord:
    # Only configuration known before the solve: factored_session and
    # factored_kernels come from the solver's info, which a failed solve does
    # not return.
    return failure_record(spec, "General", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, m=spec.dense_shape()[0],
                          algo_extras={
                              "certify_unfloored": True,
                              "general_impl": GENERAL_IMPL,
                          })


def _run_one(spec: InstanceSpec, *, check_every: Optional[int] = None) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    n_rows, n_cols = spec.dense_shape()
    n_total = n_rows
    # T_chk from the per-n_g table at the instance's n_g (its resolution, or √N
    # for a square N that is a perfect square), else the default; an explicit
    # check_every wins.
    check_every = _resolve_check_every(spec.grid_equivalent_ng() or 0, check_every)

    # The size gates run before any m×n allocation.
    dense_bytes = n_rows * n_cols * 8
    if n_rows >= SKIP_MARGINAL_LEN_AT:
        return _make_failure_record(
            spec, status="oom", error_kind="bdrs_dense_resolution_too_big",
            error_msg=(
                f"General at {n_rows}×{n_cols} skipped "
                f"preemptively: the m×n fp64 cost alone "
                f"would be {dense_bytes / 1e9:.0f} GB."
            ),
            started_utc=started_utc, run_uid=run_uid,
        )

    if spec.device.startswith("cuda"):
        if not torch.cuda.is_available():
            return _make_failure_record(
                spec, status="error", error_kind="no_cuda",
                error_msg=("torch.cuda.is_available() is False; spec requested "
                           f"device={spec.device!r}."),
                started_utc=started_utc, run_uid=run_uid,
            )
        device = torch.device(spec.device)
        total_mem = int(torch.cuda.get_device_properties(device).total_memory)
        if dense_bytes * MEM_HEADROOM > total_mem:
            return _make_failure_record(
                spec, status="oom", error_kind="bdrs_dense_resolution_too_big",
                error_msg=(
                    f"General at {n_rows}×{n_cols} skipped "
                    f"preemptively: the m×n fp64 cost × {MEM_HEADROOM} headroom "
                    f"= {dense_bytes * MEM_HEADROOM / 1e9:.1f} GB exceeds device "
                    f"memory ({total_mem / 1e9:.1f} GB)."
                ),
                started_utc=started_utc, run_uid=run_uid,
            )
    else:
        device = torch.device(spec.device)

    # The warm-up runs before the instance is loaded, on a seeded problem of
    # its shape, so it cannot see the instance. It pays the extension load,
    # the first kernel launches and the allocator's first touch; each solve
    # builds its own session and graph. Its cost is freed before the
    # instance's is built, so the two never coexist.
    if spec.warmup > 0:
        C_warm = None
        try:
            C_warm = _warmup_cost(n_rows, n_cols, device)
            warmup_params = BregmanDRParams(
                eta=eta_from_mean(C_warm.mean()),
                max_iters=10,
                epsilon_primal=0.0,            # never stops early; checks only the proxy
                epsilon_dual=0.0,
                epsilon_gap=0.0,
                check_every=10,
                verbosity=False,
            )
            _wrng = np.random.default_rng(WARMUP_SEED)
            r_warm = torch.as_tensor(warmup_marginal(_wrng, n_rows), dtype=torch.float64,
                                     device=device)
            c_warm = torch.as_tensor(warmup_marginal(_wrng, n_cols), dtype=torch.float64,
                                     device=device)
            for _ in range(spec.warmup):
                _general_solve(C_warm, r_warm, c_warm, warmup_params)
            cuda_sync(device)
        except torch.cuda.OutOfMemoryError as e:
            return _make_failure_record(
                spec, status="oom", error_kind="cuda_oom_warmup",
                error_msg=repr(e),
                started_utc=started_utc, run_uid=run_uid,
            )
        finally:
            del C_warm

    # The original marginals; the solver floors its own copies.
    r, c = _load_marginals(spec, device)
    tol = float(spec.tolerance)

    # C is resolved for every solve, never cached across instances, and its
    # build is outside t_solve and wall_time_s.
    cuda_sync(device)
    t_build0 = time.perf_counter()
    try:
        C = resolve_dense_cost(spec, device=device)
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_cost_build",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
        )
    cuda_sync(device)
    wall_cost_build = time.perf_counter() - t_build0

    # The solver iterates on C / max C, while η, the feasible-gap stop's
    # c-transform and the scoring below use C, so the two costs must agree: a
    # C whose maximum is not 1 to COST_MAX_UNIT_TOL (NaN included) is refused
    # before the solve.
    cost_max = float(C.max())
    if not abs(cost_max - 1.0) <= COST_MAX_UNIT_TOL:
        return _make_failure_record(
            spec, status="error", error_kind="cost_max_not_one",
            error_msg=(
                f"max C = {cost_max!r}, not 1: the General runner needs a cost "
                f"normalized to max C = 1. The solver would iterate on C / max C "
                f"while η, the feasible-gap stop and the scoring use C."),
            started_utc=started_utc, run_uid=run_uid,
        )

    # η = 0.7·mean(C): grid_cost_mean on a Grid-rung spec, else the mean of
    # every entry of the C built above.
    eta_alpha = ETA_ALPHA
    if is_procedural(spec):
        eta = eta_from_mean(C.mean())
        eta_scheme = "proc_mean"
    elif spec.rung == GRID:
        eta = eta_from_mean(grid_cost_mean(spec.resolution, spec.norm))
        eta_scheme = {"inf": "linf_mean", "l2": "l2_mean"}.get(spec.norm, "mean")
    else:
        eta = eta_from_mean(C.mean())
        eta_scheme = "stored_mean"

    params = solve_params(eta, spec, check_every=check_every)
    # The last gap check's certificate, reused at delivery.
    _gap_cache: Dict[str, Any] = {}
    gap_fn = make_dense_gap_stop_fn(C, r, c,
                                    variant=CTRANSFORM_VARIANT, cache=_gap_cache)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        X, info = _general_solve(
            C, r, c, params,
            gap_stop_fn=gap_fn,
            gap_stop_tol=tol,
        )
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_solve",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
        )
    cuda_sync(device)
    t1 = time.perf_counter()
    wall_solve = t1 - t0

    rep = DenseTorchPlan(X, ng=spec.resolution, cost_type=spec.norm, C=C)
    cuda_sync(device)
    t2 = time.perf_counter()
    metrics = evaluate_plan_metrics(rep, r, c)
    cuda_sync(device)
    t3 = time.perf_counter()
    wall_eval = t3 - t2

    # (α′, β′) and g_feas of the delivered plan. ⟨C, X⟩ is summed in row
    # blocks: (C * X).sum() would allocate a third m×n array.
    cuda_sync(device)
    t_rec0 = time.perf_counter()
    iters_run = int(info["iters"])
    obj_true_t = dense_trace_CX(C, X)
    alpha_p, beta_p, g_feas = delivered_certificate(
        _gap_cache, iters_run,
        lambda: feasible_dual_and_gap_storedC(
            info["log_u"], info["log_v"], eta, r, c, C, obj_true_t,
            variant=CTRANSFORM_VARIANT,
        ))
    # The recovery that t_total counts ends here; the feasibility checks
    # below verify (α′, β′) and are outside it.
    if getattr(spec, "pareto", False):
        cuda_sync(device)
        _wall_pair_recovery = time.perf_counter() - t_rec0
    dual_infeas_abs = float(_dual_feas_F_storedC(alpha_p, beta_p, C))
    r_dinf_final = r_dinf_storedC(alpha_p, beta_p, C)
    r_dinf_bound = assert_dual_feasible(r_dinf_final, alpha_p, beta_p)
    obj_true = float(obj_true_t)
    cuda_sync(device)
    wall_recovery = time.perf_counter() - t_rec0

    pareto_block = None
    if getattr(spec, "pareto", False):
        from src.scoring.pareto_score import (score as _pareto_score,
                                                        sanitize_block as _sanitize)
        cuda_sync(device)
        _te0 = time.perf_counter()
        _blk = _pareto_score(X, alpha_p, beta_p, r, c, C=C,
                             x_nonneg_structural=True, compute_frobenius=True)
        cuda_sync(device)
        pareto_block = _sanitize({
            **_blk,
            "t_solve": float(wall_solve),
            "t_total": float(wall_solve + _wall_pair_recovery),
            "t_eval": float(time.perf_counter() - _te0),
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": _t_solve_includes(spec, device),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_cost_build": float(wall_cost_build),
            "pareto_partial": False,
            "precision": "fp64",
            "native_stop": native_stop_label(tol),
        })

    objective = float(metrics["objective"])

    iter_cap_hit = iters_run >= spec.iter_cap
    # primal_res_L2 is the full row and column residual against (r, c) (the
    # in-loop proxy is row-only). dual_res_F, r_D of the uncorrected duals, is
    # NaN unless native_dual_residual, and gap_abs is their gap.
    primal_l2 = float(info["primal_res_L2"])
    dual_F = finite_or_none(info["dual_res_F"])
    gap_abs_native = float(info["gap_abs"])
    native_term = info["native_termination"]

    if device.type == "cuda":
        peak_mem: Optional[int] = int(torch.cuda.max_memory_allocated(device))
        peak_mem_source: Optional[str] = "torch_cuda_max"
        fp = device_fingerprint(idx=device.index or 0)
        device_name = fp.get("device_name")
        device_uuid = fp.get("device_uuid")
    else:
        peak_mem = None
        peak_mem_source = None
        device_name = None
        device_uuid = None

    rec = ResultRecord(
        instance_id=spec.instance_id, algorithm="General",
        norm=spec.norm, resolution=spec.resolution, n=n_total, m=n_rows,
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status="ok",
        wall_time_s=wall_solve + wall_eval + wall_recovery,
        wall_time_solve_s=wall_solve,
        wall_time_recovery_s=wall_recovery,
        iters=iters_run,
        iter_cap_hit=iter_cap_hit,
        native_termination=native_term,
        tolerance=spec.tolerance,
        feaserr=float(metrics["feaserr"]),
        native_duality_gap=gap_abs_native,
        objective=objective,
        peak_gpu_mem_bytes=peak_mem,
        peak_gpu_mem_source=peak_mem_source,
        device_name=device_name,
        device_uuid=device_uuid,
        regime=GENERAL,
        algo_extras={
            "certify_unfloored": True,
            "general_impl": GENERAL_IMPL,
            # The paths that ran: the session and the stored-C kernels each
            # fall back to a slower path when unavailable, without an error.
            "factored_session": bool(info.get("factored_session", False)),
            "factored_kernels": bool(info.get("factored_kernels", False)),
            "factored_rows_per_chunk": info.get("factored_rows_per_chunk"),
            "eta": eta,
            "eta_scheme": eta_scheme,
            "eta_alpha": eta_alpha,
            "check_every": check_every,
            **certificate_extras(
                r_dinf_final=r_dinf_final, r_dinf_bound=r_dinf_bound,
                g_feas=g_feas, dual_infeas_abs=dual_infeas_abs),
            "objective_storedC": obj_true,
            # Diagnostics; the stop does not test them.
            "primal_res_L2": primal_l2,
            "dual_res_F": dual_F,
            "native_dual_residual": bool(info.get("native_dual_residual", False)),
            "marginal_floor": finite_or_none(info.get("marginal_floor")),
            "cost_scale": finite_or_none(info.get("cost_scale")),
            "gap_abs": gap_abs_native,
            "row_sum_err_l2": float(metrics["row_sum_err_l2"]),
            "col_sum_err_l2": float(metrics["col_sum_err_l2"]),
            "wall_time_eval_s": wall_eval,
            "wall_time_recovery_s": wall_recovery,
            "wall_time_cost_build_s": wall_cost_build,
            "feaserr_measure": "r_P",
            "feaserr_negpart_assumed_zero": not metrics["negpart_measured"],
            "proxy_passes": int(info.get("proxy_passes", 0)),
            "kkt_full_checks": int(info.get("full_kkt_checks", 0)),
            "sep_obj_checks": int(info.get("sep_obj_checks", 0)),
            "primal_res_proxy_final": finite_or_none(info.get("primal_res_proxy_final")),
            "cuda_graph": bool(info.get("cuda_graph", False)),
            **({"pareto": pareto_block} if pareto_block is not None else {}),
        },
    )
    return rec


def main(argv=None):
    return runner_main(
        argv, description="General runner (dense-DOT regime).",
        run=lambda spec, args: _run_one(spec, check_every=spec.check_every),
        failure=lambda spec, args, **kw: _make_failure_record(spec, **kw))


if __name__ == "__main__":
    sys.exit(main())
