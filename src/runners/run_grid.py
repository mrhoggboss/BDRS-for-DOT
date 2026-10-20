"""
Grid runner: the BDRS CUDA solver on one Grid instance (l1 or sql2 lattice
cost), without forming the N×N plan.

    python -m src.runners.run_grid --instance spec.json --output result.json

The record describes the triple (X, α′, β′): X is the factored plan and
(α′, β′) is the alternating c-transform of the final potentials
(α, β) = (η log u, η log v), dual feasible up to rounding. The solve stops at
the first check where the row residual ‖X1 − r‖₂ is below ε_P = tol
(``solve_params``) and |g_feas|, with g_feas = ⟨C, X⟩ − (rᵀα′ + cᵀβ′), is
below tol (``make_gap_stop_fn``).
"""

import sys
import time
import uuid
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.protocol.instance_record import InstanceSpec, ResultRecord, utc_now_iso
from src.scoring.plan_metrics import BdrsFactoredPlan, evaluate_plan_metrics
from src.runners.gpu_state import device_fingerprint
from src.methods.eta import ETA_ALPHA, eta_from_mean, grid_cost_mean
from src.methods.params import BregmanDRParams
from src.methods.grid_cuda import bregman_dr_solve_grid_cuda
from src.methods.dual_cleanup import (
    _dual_feas_F,
    feasible_dual_and_gap,
    make_gap_stop_fn,
)
from src.scoring.kkt_residual import r_dinf_separable
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
    failure_record,
    finite_or_none,
    runner_main,
    warmup_marginal,
)


# The work before t0, recorded as the Pareto block's ``t_solve_excludes``.
T_SOLVE_EXCLUDES = ("marginals_load_h2d", "cost_axis_table_build", "warmup")


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str) -> ResultRecord:
    return failure_record(spec, "Grid", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid)


def _run_one(spec: InstanceSpec, *, check_every: Optional[int] = None) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    n_total = spec.resolution * spec.resolution
    check_every = _resolve_check_every(spec.resolution, check_every)

    if not torch.cuda.is_available():
        return _make_failure_record(
            spec, status="error", error_kind="no_cuda",
            error_msg="torch.cuda.is_available() is False; BDRS requires a CUDA device.",
            started_utc=started_utc, run_uid=run_uid,
        )
    device = torch.device(spec.device if spec.device.startswith("cuda") else "cuda:0")

    # The original marginals: the solver applies the δ_m floor internally, and
    # every check and recorded metric uses (r, c).
    r, c = _load_marginals(spec, device)
    eta_alpha = ETA_ALPHA
    eta = eta_from_mean(grid_cost_mean(spec.resolution, spec.norm))

    tol = float(spec.tolerance)
    params = solve_params(eta, spec, check_every=check_every)

    # Shared by every c-transform of this solve, never across instances.
    from src.methods.grid_cost import _cost_axis
    cost_axis = _cost_axis(spec.norm, spec.resolution, device=device,
                           dtype=torch.float64)
    # The stop hook's last certificate, reused at delivery when it belongs to
    # the final iterate.
    _gap_cache: Dict[str, Any] = {}

    def _certify(info: Dict[str, Any], objective):
        return feasible_dual_and_gap(
            info["log_u"], info["log_v"], eta, r, c,
            spec.resolution, spec.norm, objective,
            variant=CTRANSFORM_VARIANT,
            cost_axis=cost_axis,
        )

    _gap_fn = make_gap_stop_fn(_certify, cache=_gap_cache)

    # Untimed warm-up on seeded dummy marginals of the instance's n_g, never on
    # (r, c). It moves the one-time costs of a process's first solve (the
    # extension's JIT build, the first CUDA-graph capture) off the timed solve,
    # which still captures its own graph.
    if spec.warmup > 0:
        warmup_params = BregmanDRParams(
            eta=eta,
            max_iters=10,
            epsilon_primal=spec.tolerance,
            epsilon_dual=spec.tolerance,
            epsilon_gap=spec.tolerance,
            check_every=10,
            verbosity=False,
        )
        _wrng = np.random.default_rng(WARMUP_SEED)
        _N = spec.resolution * spec.resolution
        r_warm = torch.as_tensor(warmup_marginal(_wrng, _N), dtype=torch.float64, device=device)
        c_warm = torch.as_tensor(warmup_marginal(_wrng, _N), dtype=torch.float64, device=device)
        for _ in range(spec.warmup):
            try:
                bregman_dr_solve_grid_cuda(
                    None, r_warm, c_warm, warmup_params,
                    cost_type=spec.norm,
                    grid_size=spec.resolution,
                    return_plan=False,
                )
            except torch.cuda.OutOfMemoryError as e:
                return _make_failure_record(
                    spec, status="oom", error_kind="cuda_oom_warmup",
                    error_msg=repr(e),
                    started_utc=started_utc, run_uid=run_uid,
                )
        torch.cuda.synchronize(device)

    # t_solve is this call: the session setup and graph capture, the
    # iterations, the checks and the delivery pass.
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        _, info = bregman_dr_solve_grid_cuda(
            None, r, c, params,
            cost_type=spec.norm,
            grid_size=spec.resolution,
            return_plan=False,
            gap_stop_fn=_gap_fn,
            gap_stop_tol=tol,
        )
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_solve",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
        )
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    wall_solve = t1 - t0

    rep = BdrsFactoredPlan(
        A=info["A"].to(torch.float64),
        B=info["B"].to(torch.float64),
        ng=spec.resolution,
        cost_type=spec.norm,
        eta=eta,
        iters=int(info["iters"]),
    )
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    metrics = evaluate_plan_metrics(rep, r, c)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter()
    wall_eval = t3 - t2

    # The delivered dual (α′, β′). obj_factored is the solver's ⟨C, X⟩, the
    # value a reused in-loop certificate was computed with.
    torch.cuda.synchronize(device)
    t_rec0 = time.perf_counter()
    log_u = info["log_u"].to(torch.float64).flatten()
    log_v = info["log_v"].to(torch.float64).flatten()
    obj_factored = float(info.get("objective", metrics["objective"]))
    alpha_p, beta_p, g_feas = delivered_certificate(
        _gap_cache, int(info["iters"]),
        lambda: feasible_dual_and_gap(
            log_u, log_v, eta, r, c, spec.resolution, spec.norm, obj_factored,
            variant=CTRANSFORM_VARIANT,
            cost_axis=cost_axis,
        ))
    # The recovery for t_total ends here (a cached certificate, or one
    # c-transform when the hook did not run on the final iterate).
    # wall_recovery also contains the O(N²) r_D of (α′, β′) and the r_Dinf
    # check, which verify the pair rather than recover it.
    if getattr(spec, "pareto", False):
        torch.cuda.synchronize(device)
        _wall_pair_recovery = time.perf_counter() - t_rec0
    dual_infeas_abs = float(_dual_feas_F(alpha_p, beta_p, spec.resolution, spec.norm))
    r_dinf_final = r_dinf_separable(alpha_p, beta_p, spec.resolution, spec.norm)
    r_dinf_bound = assert_dual_feasible(r_dinf_final, alpha_p, beta_p)
    torch.cuda.synchronize(device)
    wall_recovery = time.perf_counter() - t_rec0

    # Pareto block of (rep, α′, β′); rep is X ≥ 0 by construction. The scoring
    # (t_eval) is on neither clock.
    pareto_block = None
    if getattr(spec, "pareto", False):
        from src.scoring.pareto_score import (score as _pareto_score,
                                                        sanitize_block as _sanitize)
        torch.cuda.synchronize(device)
        _te0 = time.perf_counter()
        _blk = _pareto_score(rep, alpha_p, beta_p, r, c,
                             ng=spec.resolution, cost_type=spec.norm,
                             x_nonneg_structural=True, compute_frobenius=True)
        torch.cuda.synchronize(device)
        pareto_block = _sanitize({
            **_blk,
            "t_solve": float(wall_solve),
            "t_total": float(wall_solve + _wall_pair_recovery),
            "t_eval": float(time.perf_counter() - _te0),
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": _t_solve_includes(spec, device),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "pareto_partial": False,
            "precision": "fp64",
            "native_stop": native_stop_label(tol),
        })

    objective = float(metrics["objective"])
    # The native gap |⟨C, X⟩ − η(rᵀ log u + cᵀ log v)| of (α, β), which is in
    # general not dual feasible, so this gap certifies nothing; a diagnostic.
    dual_obj_native = eta * (float(log_u.dot(r).item()) + float(log_v.dot(c).item()))
    native_gap = abs(obj_factored - dual_obj_native)

    iters_run = int(info["iters"])
    iter_cap_hit = iters_run >= spec.iter_cap
    # The full ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂ from the delivery pass, not the row proxy.
    primal_l2 = float(info.get("primal_res_L2", float("nan")))
    # The native r_D of (α, β) is not requested: dual_res_F is NaN, recorded as None.
    dual_F = finite_or_none(info.get("dual_res_F"))
    gap_abs = float(info.get("gap_abs", native_gap))
    native_term = info["native_termination"]

    peak_mem = int(torch.cuda.max_memory_allocated(device))
    fp = device_fingerprint(idx=device.index or 0)
    device_name = fp.get("device_name")
    device_uuid = fp.get("device_uuid")

    rec = ResultRecord(
        instance_id=spec.instance_id, algorithm="Grid",
        norm=spec.norm, resolution=spec.resolution, n=n_total,
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
        native_duality_gap=float(native_gap),
        objective=objective,
        peak_gpu_mem_bytes=peak_mem,
        peak_gpu_mem_source="torch_cuda_max",
        device_name=device_name,
        device_uuid=device_uuid,
        algo_extras={
            "certify_unfloored": True,
            "eta": eta,
            "eta_scheme": "mean",
            "eta_alpha": eta_alpha,
            "check_every": check_every,
            **certificate_extras(
                r_dinf_final=r_dinf_final, r_dinf_bound=r_dinf_bound,
                g_feas=g_feas, dual_infeas_abs=dual_infeas_abs),
            "objective_factored": obj_factored,
            "primal_res_L2": primal_l2,
            "dual_res_F": dual_F,
            "native_dual_residual": bool(info.get("native_dual_residual", False)),
            "marginal_floor": finite_or_none(info.get("marginal_floor")),
            "gap_abs_factored": gap_abs,
            "row_sum_err_l2": float(metrics["row_sum_err_l2"]),
            "col_sum_err_l2": float(metrics["col_sum_err_l2"]),
            "wall_time_eval_s": wall_eval,
            "wall_time_recovery_s": wall_recovery,
            "feaserr_measure": "r_P",
            "feaserr_negpart_assumed_zero": not metrics["negpart_measured"],
            "kkt_proxy_passes": int(info.get("proxy_passes", 0)),
            "kkt_full_checks": int(info.get("full_kkt_checks", 0)),
            "primal_res_proxy_final": finite_or_none(info.get("primal_res_proxy_final")),
            **({"pareto": pareto_block} if pareto_block is not None else {}),
        },
    )
    return rec


def main(argv=None):
    return runner_main(
        argv, description="Grid runner (BDRS, grid regime).",
        run=lambda spec, args: _run_one(spec, check_every=spec.check_every),
        failure=lambda spec, args, **kw: _make_failure_record(spec, **kw))


if __name__ == "__main__":
    sys.exit(main())
