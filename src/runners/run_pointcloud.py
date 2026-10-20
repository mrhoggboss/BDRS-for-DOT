"""
Point Cloud runner: BDRS on one Point Cloud instance, whose cost
C_ij = κ(p_i, q_j) / c_max is evaluated from the clouds P, Q block by block, so
neither C nor the plan X is formed. ``spec.c_max`` must be the exact pairwise
maximum of κ (see ``c_max_not_exact`` below).

    python -m src.runners.run_pointcloud --instance spec.json --output result.json

The stop, the delivered triple (X, α′, β′) and the recorded certificate are
those of ``run_grid``, with every cost sweep (the c-transform, r_D, the KKT
components) run on the cost blocks. With a CPU ``spec.device`` the solve runs
in the torch Point Cloud solver.
"""

import sys
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from src.protocol.instance_record import (
    InstanceSpec, ResultRecord, geo_cost_type,
    utc_now_iso,
)
from src.runners.gpu_state import device_fingerprint
from src.methods import geo_cost
from src.methods.params import BregmanDRParams
from src.methods.factored_cuda import bregman_dr_solve_pointcloud_cuda
from src.methods.dual_cleanup import (
    _dual_feas_F_symbolic,
    _dual_feas_inf_symbolic,
    cost_inf_symbolic,
    feasible_dual_and_gap_symbolic,
    make_pointcloud_gap_stop_fn,
)
from src.methods.eta import ETA_ALPHA, eta_from_mean, grid_cost_mean
from src.runners.bdrs_common import (
    CHECK_EVERY_BY_N,
    CTRANSFORM_VARIANT,
    T_SOLVE_BOUNDARY,
    assert_dual_feasible,
    certificate_extras,
    delivered_certificate,
    native_stop_label,
    solve_params,
    _resolve_check_every,
    _t_solve_includes,
)
from src.runners.common import (
    WARMUP_SEED,
    cuda_sync,
    failure_record,
    finite_or_none,
    load_marginal_unit_mass,
    runner_main,
    warmup_marginal,
)

# The work before t0, recorded as the Pareto block's ``t_solve_excludes``.
# Unless the instance is a lifted l1/sql2 grid, η needs the mean of C, an O(mn)
# pass; the warm-up needs η, so the pass runs before it and is timed as
# ``t_eta_select``.
T_SOLVE_EXCLUDES = ("marginals_and_clouds_load_h2d", "eta_selection",
                    "cost_inf_precompute", "warmup")


# The largest |cost_scale − 1| accepted as a max-1 instance. cost_scale is
# (max_ij κ_ij) / c_max as the solver recomputes it; it can miss 1 by an ulp
# when c_max was computed on another device, or is the closed-form
# ``grid_cmax`` of a lifted l2 grid.
COST_SCALE_UNIT_TOL = 1e-9


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str) -> ResultRecord:
    return failure_record(spec, "PointCloud", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, m=spec.m)


def _load_marginal(path: str, device: torch.device) -> torch.Tensor:
    """The original marginal (unit mass, no floor); the solver applies δ_m internally."""
    a = load_marginal_unit_mass(path)
    return torch.as_tensor(a, dtype=torch.float64, device=device)


def _load_cloud(path: str, device: torch.device) -> torch.Tensor:
    P = np.load(path).astype(np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    return torch.as_tensor(P, dtype=torch.float64, device=device)


def _is_lifted_grid(spec: InstanceSpec, m: int, n: int) -> bool:
    """A Grid instance lifted to a 2-D cloud: resolution set, d = 2, m = n = n_g²."""
    return (spec.resolution is not None and spec.dim == 2
            and m == n == int(spec.resolution) ** 2)


def _mean_cost(cost_fn: Callable[[int, int], torch.Tensor], m: int, n: int) -> float:
    """Mean of C over all m·n entries, in row blocks of at most 2^21 entries (at
    least one row), so the (rows, n, d) temporary inside ``cost_fn`` stays
    bounded."""
    chunk = max(1, min(m, (2 * 1024 * 1024) // max(1, n)))
    acc = 0.0
    cnt = 0
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        blk = cost_fn(i0, i1)
        acc += float(blk.sum().item())
        cnt += blk.numel()
    return acc / cnt



def _resolve_geo_eta(spec: InstanceSpec, cost_fn, m: int, n: int) -> Tuple[float, str, float]:
    """``(η, eta_scheme, eta_alpha)``. A lifted l1/sql2 grid takes the closed-form
    lattice mean, so it runs with the Grid runner's η; every other cloud
    averages C over all m·n entries."""
    if _is_lifted_grid(spec, m, n) and spec.norm in ("l1", "sql2"):
        return (eta_from_mean(grid_cost_mean(int(spec.resolution), spec.norm)),
                "mean", ETA_ALPHA)
    mean_cost = _mean_cost(cost_fn, m, n)
    return eta_from_mean(mean_cost), "geo_mean", ETA_ALPHA


def _equivalent_ng(m: int, n: int) -> int:
    """The lattice side n_eq = 2^⌈log₂(N)/2⌉, N = max(m, n), whose T_chk a cloud
    uses, clamped to the range of ``CHECK_EVERY_BY_N``; computed in integers."""
    N = max(int(m), int(n))
    if N <= 1:
        k = 0
    else:
        k = (N - 1).bit_length()          # = ceil(log2 N)
    n_eq = 1 << ((k + 1) // 2)            # = 2^ceil(log2(N)/2)
    lo, hi = min(CHECK_EVERY_BY_N), max(CHECK_EVERY_BY_N)
    return min(max(n_eq, lo), hi)


def _resolve_geo_check_every(m: int, n: int, override: Optional[int] = None) -> int:
    """T_chk of a cloud: the Grid schedule at ``_equivalent_ng``, unless ``override`` is given."""
    if override is not None:
        return int(override)
    return _resolve_check_every(_equivalent_ng(m, n))


def _primal_terms(info: Dict[str, Any], r: torch.Tensor, c: torch.Tensor,
                  cost_fn, *, m: int, n: int, eta: float, iters: int,
                  check_every: int, primal_l2: float,
                  native_term: str) -> Tuple[Optional[float], Optional[float], str]:
    """``(‖X1 − r‖₂, ‖Xᵀ1 − c‖₂, source)`` of the delivered X, without forming X.

    The CUDA session returns ξ, ζ (``A``, ``B``), and both sums are recomputed
    as the solver's delivery pass computes them. The torch solver does not;
    then the row term is the row-only proxy and the column term is the
    delivered r_P minus it, valid only when the last check ran on the
    delivered iterate (a feasible-gap exit, or k a multiple of T_chk).
    Otherwise both are None."""
    A = info.get("A")
    B = info.get("B")
    if A is not None and B is not None:
        from src.methods.factored_torch import (
            _pointcloud_block_chunk,
            _pointcloud_col_lse,
            _pointcloud_row_lse,
        )
        chunk = _pointcloud_block_chunk(m, n)
        scale = iters / float(eta)
        A = A.to(torch.float64).reshape(-1)
        B = B.to(torch.float64).reshape(-1)
        lse_row = torch.empty(m, dtype=torch.float64, device=A.device)
        _pointcloud_row_lse(lse_row, cost_fn, B, scale, m, chunk)
        row = float(((A + lse_row).exp() - r).norm(2).item())
        lse_col = torch.empty(n, dtype=torch.float64, device=A.device)
        _pointcloud_col_lse(lse_col, cost_fn, A, scale, m, chunk)
        col = float(((B + lse_col).exp() - c).norm(2).item())
        return row, col, "factored_state"

    proxy = finite_or_none(info.get("primal_res_proxy_final"))
    fresh = (native_term == "feasible_gap_below_tol"
             or (check_every > 0 and iters % check_every == 0))
    if proxy is not None and fresh and np.isfinite(primal_l2):
        return proxy, max(0.0, primal_l2 - proxy), "delivered_split"
    return None, None, "unavailable"


def _run_one(spec: InstanceSpec, *, check_every: Optional[int] = None) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())

    want_cuda = spec.device.startswith("cuda")
    if want_cuda and not torch.cuda.is_available():
        return _make_failure_record(
            spec, status="error", error_kind="no_cuda",
            error_msg="spec.device requests CUDA but torch.cuda.is_available() is False.",
            started_utc=started_utc, run_uid=run_uid)
    device = torch.device(spec.device if (want_cuda and torch.cuda.is_available()) else "cpu")

    r = _load_marginal(spec.marginal_i_path, device)
    c = _load_marginal(spec.marginal_j_path, device)
    P = _load_cloud(spec.support_i_path, device)
    Q = _load_cloud(spec.support_j_path, device)
    m = int(r.numel())
    n = int(c.numel())
    dim = int(P.shape[1])
    cost_type = geo_cost_type(spec.norm)          # inf -> linf
    c_max = float(spec.c_max)
    cost_fn = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max)

    check_every = _resolve_geo_check_every(m, n, check_every)

    cuda_sync(device)
    _t_eta0 = time.perf_counter()
    eta, eta_scheme, eta_alpha = _resolve_geo_eta(spec, cost_fn, m, n)
    cuda_sync(device)
    wall_eta_select = time.perf_counter() - _t_eta0

    tol = float(spec.tolerance)
    params = solve_params(eta, spec, check_every=check_every)
    # ‖C‖_∞ of the Pareto block: one O(mn) pass, off both clocks.
    cost_inf = (cost_inf_symbolic(cost_fn, m, n)
                if getattr(spec, "pareto", False) else None)

    _gap_cache: Dict[str, Any] = {}
    gap_fn = make_pointcloud_gap_stop_fn(
        cost_fn, r, c, variant=CTRANSFORM_VARIANT, cache=_gap_cache)

    # Untimed warm-up on seeded random clouds and marginals of the instance's
    # (m, n, d), never on the instance. It moves the one-time costs of a
    # process's first solve (the extension's JIT build, the first CUDA-graph
    # capture) off the timed solve, which still captures its own graph.
    if spec.warmup > 0:
        _wrng = np.random.default_rng(WARMUP_SEED)
        Pw = torch.as_tensor(_wrng.standard_normal((m, dim)), dtype=torch.float64, device=device)
        Qw = torch.as_tensor(_wrng.standard_normal((n, dim)), dtype=torch.float64, device=device)
        r_warm = torch.as_tensor(warmup_marginal(_wrng, m), dtype=torch.float64, device=device)
        c_warm = torch.as_tensor(warmup_marginal(_wrng, n), dtype=torch.float64, device=device)
        warmup_params = BregmanDRParams(
            eta=eta, max_iters=10, epsilon_primal=tol, epsilon_dual=tol,
            epsilon_gap=tol, check_every=10, verbosity=False)
        for _ in range(spec.warmup):
            try:
                bregman_dr_solve_pointcloud_cuda(
                    Pw, Qw, r_warm, c_warm, warmup_params,
                    cost_type=cost_type, c_max=c_max, return_plan=False)
            except torch.cuda.OutOfMemoryError as e:
                return _make_failure_record(
                    spec, status="oom", error_kind="cuda_oom_warmup",
                    error_msg=repr(e), started_utc=started_utc, run_uid=run_uid)
        cuda_sync(device)

    # t_solve is this call: the Normalize step's O(mnd) recompute of C_max, the
    # session setup and CUDA-graph capture (CUDA path), the iterations, the
    # checks and the delivery pass.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cuda_sync(device)
    t0 = time.perf_counter()
    try:
        _, info = bregman_dr_solve_pointcloud_cuda(
            P, Q, r, c, params,
            cost_type=cost_type, c_max=c_max,
            return_plan=False,
            gap_stop_fn=gap_fn, gap_stop_tol=tol,
        )
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_solve",
            error_msg=repr(e), started_utc=started_utc, run_uid=run_uid)
    cuda_sync(device)
    wall_solve = time.perf_counter() - t0

    # A cost_scale beyond COST_SCALE_UNIT_TOL of 1 means spec.c_max is not the
    # attained max_ij κ_ij: the solver iterated on κ / C_max, while cost_fn,
    # and so the certificate and every metric below, is κ / c_max. The run
    # fails rather than certify across two costs.
    cost_scale = float(info.get("cost_scale", 1.0))
    if abs(cost_scale - 1.0) > COST_SCALE_UNIT_TOL:
        return _make_failure_record(
            spec, status="error", error_kind="c_max_not_exact",
            error_msg=(
                f"instance c_max={c_max!r} is not the attained pairwise maximum: "
                f"the solver's exact recompute gives cost_scale="
                f"max_ij C_ij = {cost_scale!r} on the instance cost; the dataset "
                f"builder must store geo_cost.exact_c_max(P, Q, cost_type). The "
                f"solver's objective and this runner's certificate would be on "
                f"different scales."),
            started_utc=started_utc, run_uid=run_uid)

    iters_run = int(info["iters"])
    iter_cap_hit = iters_run >= spec.iter_cap
    obj = float(info["objective"])
    # primal_res_L2 is ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂ of the delivered iterate. The r_D
    # of the uncorrected (α, β) is not requested, so info["dual_res_F"] is NaN
    # and dual_F is None.
    primal_l2 = float(info.get("primal_res_L2", float("nan")))
    dual_F = finite_or_none(info.get("dual_res_F"))
    gap_abs_diag = float(info.get("gap_abs", float("nan")))

    cuda_sync(device)
    t_rec0 = time.perf_counter()
    alpha_p, beta_p, g_feas = delivered_certificate(
        _gap_cache, iters_run,
        lambda: feasible_dual_and_gap_symbolic(
            info["log_u"], info["log_v"], float(info["eta"]), r, c, cost_fn,
            float(obj), variant=CTRANSFORM_VARIANT))
    # The recovery for t_total ends here (a cached certificate, or one
    # c-transform when the hook did not run on the final iterate).
    # wall_recovery also contains the O(mn) r_Dinf check, which verifies the
    # pair rather than recovers it.
    if getattr(spec, "pareto", False):
        cuda_sync(device)
        _wall_pair_recovery = time.perf_counter() - t_rec0
    r_dinf_final = _dual_feas_inf_symbolic(alpha_p, beta_p, cost_fn, m, n)
    r_dinf_bound = assert_dual_feasible(r_dinf_final, alpha_p, beta_p,
                                        defect="c-transform defect.")
    cuda_sync(device)
    wall_recovery = time.perf_counter() - t_rec0

    # r_D of (α′, β′), the dual term of the KKT residual, is the Frobenius
    # ‖(α′ ⊕ β′ − C)₊‖_F; r_Dinf above is a diagnostic.
    dual_infeas_abs = float(_dual_feas_F_symbolic(alpha_p, beta_p, cost_fn, m, n))
    # With gap_stop_fn the solver exits only on the feasible gap or at K_max.
    native_term = info["native_termination"]

    # Pareto block of (X, α′, β′). score_symbolic evaluates C block by block,
    # since pareto_score.score needs a lattice or a stored C. The scoring
    # (t_eval) is on neither clock.
    pareto_block = None
    if getattr(spec, "pareto", False):
        from src.scoring.pareto_score import sanitize_block as _sanitize
        from src.scoring.pareto_score import score_symbolic
        cuda_sync(device)
        _te0 = time.perf_counter()
        rp_row, rp_col, rp_source = _primal_terms(
            info, r, c, cost_fn, m=m, n=n, eta=float(info["eta"]),
            iters=iters_run, check_every=int(check_every),
            primal_l2=primal_l2, native_term=native_term)
        # X = exp(·) > 0, so the negative-part term of r_P is 0. r_Dinf and
        # r_DF were computed above on the same (α′, β′) and cost; passing them
        # skips two O(mn) sweeps.
        blk = score_symbolic(
            p_obj=obj, r_P_row=rp_row, r_P_col=rp_col, r_P_neg=0.0,
            alpha=alpha_p, beta=beta_p, r=r, c=c, cost_fn=cost_fn, m=m, n=n,
            r_Dinf=r_dinf_final, r_DF=dual_infeas_abs, C_inf=cost_inf)
        cuda_sync(device)
        pareto_block = _sanitize({
            **blk,
            "t_solve": float(wall_solve),
            "t_total": float(wall_solve + _wall_pair_recovery),
            "t_eval": float(time.perf_counter() - _te0),
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": _t_solve_includes(spec, device),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_eta_select": float(wall_eta_select),
            # Partial only when a primal term was not delivered; a non-finite
            # term is a separate case, which aggregation counts as "nonfinite".
            "pareto_partial": bool(rp_row is None or rp_col is None),
            "x_nonneg_structural": True,
            "r_P_source": rp_source,
            "precision": "fp64",
            "native_stop": native_stop_label(tol),
        })

    peak_mem = None
    device_name = device_uuid = None
    peak_src = "unknown"
    if device.type == "cuda":
        peak_mem = int(torch.cuda.max_memory_allocated(device))
        peak_src = "torch_cuda_max"
        fp = device_fingerprint(idx=device.index or 0)
        device_name = fp.get("device_name")
        device_uuid = fp.get("device_uuid")
    kernels_used = bool(info.get("cuda_graph", False))

    return ResultRecord(
        instance_id=spec.instance_id, algorithm="PointCloud",
        norm=spec.norm, resolution=spec.resolution, n=m,   # ResultRecord.n is the length of r
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status="ok",
        wall_time_s=wall_solve + wall_recovery,
        wall_time_solve_s=wall_solve,
        wall_time_recovery_s=wall_recovery,
        iters=iters_run, iter_cap_hit=iter_cap_hit,
        native_termination=native_term, tolerance=tol,
        feaserr=primal_l2, native_duality_gap=finite_or_none(gap_abs_diag),
        objective=obj,
        peak_gpu_mem_bytes=peak_mem, peak_gpu_mem_source=peak_src,
        device_name=device_name, device_uuid=device_uuid,
        regime="PointCloud", rung="PointCloud", dataset=spec.dataset, m=m,
        algo_extras={
            "feaserr_measure": "r_P",
            "certify_unfloored": True,
            "eta": eta, "eta_scheme": eta_scheme, "eta_alpha": eta_alpha,
            "wall_time_eta_select_s": wall_eta_select,
            "check_every": int(check_every),
            **certificate_extras(
                r_dinf_final=r_dinf_final, r_dinf_bound=r_dinf_bound,
                g_feas=g_feas, dual_infeas_abs=dual_infeas_abs),
            "objective_factored": obj,
            "primal_res_L2": primal_l2,
            "dual_res_F": dual_F,
            "native_dual_residual": bool(info.get("native_dual_residual", False)),
            "marginal_floor": finite_or_none(info.get("marginal_floor")),
            "gap_abs_factored": finite_or_none(gap_abs_diag),
            "m": m, "n": n, "dim": dim, "c_max": c_max,
            "cost_scale": cost_scale,
            "geo_cost_type": cost_type, "kernels_used": kernels_used,
            "wall_time_recovery_s": wall_recovery,
            "kkt_proxy_passes": int(info.get("proxy_passes", 0)),
            "kkt_full_checks": int(info.get("full_kkt_checks", 0)),
            "primal_res_proxy_final": finite_or_none(info.get("primal_res_proxy_final")),
            **({"pareto": pareto_block} if pareto_block is not None else {}),
        },
    )


def main(argv=None):
    return runner_main(
        argv, description="Point Cloud runner (BDRS).",
        run=lambda spec, args: _run_one(spec, check_every=spec.check_every),
        failure=lambda spec, args, **kw: _make_failure_record(spec, **kw))


if __name__ == "__main__":
    sys.exit(main())
