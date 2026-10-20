"""Point Cloud and General formats on CUDA, one captured graph per iteration.

Both formats iterate the factored state log X^k_ij = ξ_i + k·L_ij + ζ_j with
L = −C/η (ξ, ζ are ``A``, ``B``), so no m×n iterate is stored. Each solver
drives a session of the CUDA extension through ``_factored_session_loop``,
whose checks use the tiled reductions of ``factored_torch``. Without CUDA,
with the inputs on the CPU, or with the format's environment switch set to
``0``, it falls back to the torch solver in ``factored_torch``.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from src.methods import cuda_ext
from src.methods.dual_cleanup import make_dense_gap_stop_fn, make_pointcloud_gap_stop_fn
from src.methods.params import (
    BregmanDRParams,
    _check_marginals,
    _resolve_stop,
    floor_marginal,
)


# Must match the branches of pointcloud_cost_raw in cuda/bdrs_kernels.cu, which
# silently treats an unknown code as l1.
_POINTCLOUD_COST_TYPES: Dict[str, int] = {
    "l1":   0,
    "sql2": 1,
    "l2":   2,
    "linf": 3,
    "inf":  3,
}


def _pointcloud_check_primal_row(A, B, r, cost_fn, scale, m, chunk) -> float:
    """Row-only primal proxy ‖X1 − r‖₂ on the factored state (``scale`` = k/η).

    The column term is left out: the ζ-update makes Xᵀ1 = c̃ up to rounding,
    so it is the constant ‖c̃ − c‖₂ at every check."""
    from src.methods.factored_torch import _pointcloud_row_lse
    dev, dt = A.device, A.dtype
    lse_row = torch.empty(m, device=dev, dtype=dt)
    _pointcloud_row_lse(lse_row, cost_fn, B, scale, m, chunk)
    row_sums = (A + lse_row).exp()
    return float((row_sums - r).norm(2).item())


def _pointcloud_check_primal(A, B, r, c, cost_fn, scale, m, n, chunk) -> float:
    """Full primal residual r_P = ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂ on the factored state.

    X is an exponential, hence nonnegative, so r_P has no negative-part term."""
    from src.methods.factored_torch import _pointcloud_row_lse, _pointcloud_col_lse
    dev, dt = A.device, A.dtype
    lse_row = torch.empty(m, device=dev, dtype=dt)
    _pointcloud_row_lse(lse_row, cost_fn, B, scale, m, chunk)
    row_sums = (A + lse_row).exp()
    lse_col = torch.empty(n, device=dev, dtype=dt)
    _pointcloud_col_lse(lse_col, cost_fn, A, scale, m, chunk)
    col_sums = (B + lse_col).exp()
    return float(((row_sums - r).norm(2) + (col_sums - c).norm(2)).item())


def _pointcloud_full_metrics(A, B, log_u, log_v, r, c, cost_fn, eta, k, m, n, chunk,
                             *, native_dual: bool = False, objective=None):
    """(r_P, r_D, |g|, ⟨C,X⟩) as floats at iteration ``k``, for α = η·log u, β = η·log v.

    r_D costs an O(mn) pass and is NaN unless ``native_dual=True``.
    ``objective`` (float or 0-dim tensor) reuses a ⟨C,X⟩ already computed on
    this state."""
    from src.methods.factored_torch import _pointcloud_trace_CX, _pointcloud_dual_viol_sq
    scale = k / eta
    primal = _pointcloud_check_primal(A, B, r, c, cost_fn, scale, m, n, chunk)
    if native_dual:
        dual = float((eta * _pointcloud_dual_viol_sq(cost_fn, log_u, log_v, eta, m, chunk).sqrt()).item())
    else:
        dual = float("nan")
    obj_t = objective if objective is not None else _pointcloud_trace_CX(cost_fn, A, B, scale, m, chunk)
    obj = float(obj_t.item()) if isinstance(obj_t, torch.Tensor) else float(obj_t)
    gap = abs(obj - eta * float((log_u.dot(r) + log_v.dot(c)).item()))
    return primal, dual, gap, obj


def _factored_session_loop(
    sess: Any,
    cost_fn: Callable[[int, int], torch.Tensor],
    *,
    r_chk: torch.Tensor,
    c_chk: torch.Tensor,
    params: BregmanDRParams,
    eta_f: float,
    m: int,
    n: int,
    chunk: int,
    device: torch.device,
    max_iters_total: int,
    check_every: int,
    return_plan: bool,
    traj_callback: Optional[Callable[[Dict[str, Any]], None]],
    gap_stop_fn: Optional[Callable[[Dict[str, Any]], float]],
    gap_stop_tol: float,
    native_dual_residual: bool,
    cost_scale: float,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """Run a captured factored session to termination and return ``(X, info)``.

    ``sess.step(n)`` returns the cumulative iteration count k, and ``sess.A``,
    ``sess.B``, ``sess.log_u``, ``sess.log_v`` are device tensors it updates in
    place. Each batch runs min(T_chk, K_max − k) iterations (``check_every``,
    ``max_iters_total``), then one check against ``r_chk``, ``c_chk``:

      * with ``traj_callback``: the full metrics go to the callback, and the
        solve stops (``"kkt_below_tol"``) when r_P, r_D and |g| are below
        ``epsilon_primal``, ``epsilon_dual`` and ``epsilon_gap``;
      * otherwise, once the row-only proxy is below ``epsilon_primal``: with
        ``gap_stop_fn`` (the feasible-gap stop) the solve stops
        (``"feasible_gap_below_tol"``) when its value is below
        ``gap_stop_tol``; without it (the KKT-triple stop) the full metrics
        decide as above.

    The ``log_u``, ``log_v`` given to the callback and to ``gap_stop_fn`` are
    the session's own tensors. The next ``step()`` overwrites them and does not
    wait for the caller's stream, so a callback that keeps them must copy them
    and finish the copy (e.g. ``.cpu()``) before it returns. X is the dense
    m×n fp64 plan with ``return_plan``, else ``None``.
    """
    from src.methods.factored_torch import _pointcloud_trace_CX


    A_state = sess.A
    B_state = sess.B
    log_u_state = sess.log_u
    log_v_state = sess.log_v

    iters_run = 0
    native_termination = "max_iters"
    proxy_passes = 0
    full_checks = 0          # r_D evaluations
    sep_obj_checks = 0
    last_gap_val = None
    last_full = None
    last_full_iter = -1
    last_obj_t = None
    last_obj_iter = -1
    primal_proxy_value = float("inf")


    while iters_run < max_iters_total:
        n_step = min(check_every, max_iters_total - iters_run)
        iters_run = int(sess.step(int(n_step)))

        scale = iters_run / eta_f
        if traj_callback is not None:
            pr, du, ga, ob = _pointcloud_full_metrics(
                A_state, B_state, log_u_state, log_v_state, r_chk, c_chk, cost_fn,
                eta_f, iters_run, m, n, chunk, native_dual=True)
            full_checks += 1
            last_full, last_full_iter = (pr, du, ga, ob), iters_run
            traj_callback({
                "k": iters_run, "eta": eta_f,
                "primal_res_L2": pr, "dual_res_F": du, "gap_abs": ga,
                "objective": ob, "log_u": log_u_state, "log_v": log_v_state,
            })
            if (pr < params.epsilon_primal and du < params.epsilon_dual
                    and ga < params.epsilon_gap):
                native_termination = "kkt_below_tol"
                break
        else:
            primal_proxy_value = _pointcloud_check_primal_row(
                A_state, B_state, r_chk, cost_fn, scale, m, chunk)
            if primal_proxy_value < params.epsilon_primal:
                proxy_passes += 1
                if gap_stop_fn is not None:
                    obj_t = _pointcloud_trace_CX(cost_fn, A_state, B_state, scale, m, chunk)
                    last_obj_t, last_obj_iter = obj_t, iters_run
                    sep_obj_checks += 1
                    gap_val = float(gap_stop_fn({
                        "iters": iters_run, "eta": eta_f,
                        "log_u": log_u_state, "log_v": log_v_state,
                        "objective": obj_t,
                    }))
                    last_gap_val = gap_val
                    if (primal_proxy_value < params.epsilon_primal
                            and gap_val < gap_stop_tol):
                        native_termination = "feasible_gap_below_tol"
                        break
                else:
                    pr, du, ga, ob = _pointcloud_full_metrics(
                        A_state, B_state, log_u_state, log_v_state, r_chk, c_chk, cost_fn,
                        eta_f, iters_run, m, n, chunk, native_dual=True)
                    full_checks += 1
                    last_full, last_full_iter = (pr, du, ga, ob), iters_run
                    if (pr < params.epsilon_primal and du < params.epsilon_dual
                            and ga < params.epsilon_gap):
                        native_termination = "kkt_below_tol"
                        break

    # Delivery on the final state. A full check taken on this state is reused,
    # its r_D included; otherwise r_D is computed only with native_dual_residual.
    if last_full is not None and last_full_iter == iters_run:
        primal_f, dual_f, gap_f, obj_f = last_full
    else:
        obj_reuse = last_obj_t if last_obj_iter == iters_run else None
        primal_f, dual_f, gap_f, obj_f = _pointcloud_full_metrics(
            A_state, B_state, log_u_state, log_v_state, r_chk, c_chk, cost_fn,
            eta_f, iters_run, m, n, chunk,
            native_dual=bool(native_dual_residual), objective=obj_reuse)
        if native_dual_residual:
            full_checks += 1

    X = None
    if return_plan:
        scale = iters_run / eta_f
        X = torch.empty((m, n), dtype=torch.float64, device=device)
        for i0 in range(0, m, chunk):
            i1 = min(m, i0 + chunk)
            cb = cost_fn(i0, i1)
            X[i0:i1] = ((A_state[i0:i1, None] - scale * cb) + B_state[None, :]).exp_()
            del cb

    info: Dict[str, Any] = {
        "iters": iters_run,
        "primal_res_L2": primal_f,
        "gap_abs": gap_f,
        "objective": obj_f,
        "eta": eta_f,
        "dual_res_F": dual_f,
        "log_u": log_u_state,
        "log_v": log_v_state,
        "A": A_state,
        "B": B_state,
        "native_termination": native_termination,
        "proxy_passes": proxy_passes,
        "full_kkt_checks": full_checks,
        "sep_obj_checks": sep_obj_checks,
        "primal_res_proxy_final": primal_proxy_value,
        "native_dual_residual": bool(native_dual_residual),
        "marginal_floor": float(params.eps_mass),
        "cost_scale": cost_scale,
        "cuda_graph": True,
    }
    if last_gap_val is not None:
        info["gap_stop_value"] = last_gap_val
    return X, info


def bregman_dr_solve_pointcloud_cuda(
    P: torch.Tensor,                  # [m, d] fp64 row-support coordinates
    Q: torch.Tensor,                  # [n, d] fp64 column-support coordinates
    r: torch.Tensor,                  # [m] fp64, the original (unfloored) marginal
    c: torch.Tensor,                  # [n] fp64, the original (unfloored) marginal
    params: BregmanDRParams,
    *,
    cost_type: str,                   # "l1"|"sql2"|"l2"|"linf" ("inf" = "linf")
    c_max: float = 1.0,               # the instance cost is κ / c_max
    normalize_cost: bool = True,      # recompute C_max = max κ exactly
    return_plan: bool = True,
    stop: str = "feasible_gap",       # or "kkt_triple" (params.STOP_RULES)
    traj_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    gap_stop_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    check_r: Optional[torch.Tensor] = None,
    check_c: Optional[torch.Tensor] = None,
    gap_stop_tol: Optional[float] = None,   # None: params.epsilon_primal
    native_dual_residual: bool = False,
    _ext_override: Any = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """Point Cloud BDRS on CUDA over the clouds P, Q with ground cost κ (``cost_type``).

    Runs ``ext.PointCloudBdrsSession`` through ``_factored_session_loop``.
    The stop rule ``stop`` is that of ``factored_torch.bregman_dr_solve_pointcloud``:
    by default ``"feasible_gap"``, with ``gap_stop_tol`` defaulting to
    ``params.epsilon_primal`` and, without ``gap_stop_fn``,
    ``dual_cleanup.make_pointcloud_gap_stop_fn`` built by the solver on the
    iterated cost and the check marginals; or ``"kkt_triple"``. A
    ``gap_stop_fn`` passed in must be built on the cost the solver iterates
    on: κ / c_max, or κ / C_max when the Normalize step rescales
    (``info["cost_scale"]`` ≠ 1). ``gap_stop_fn`` and ``traj_callback`` are
    mutually exclusive.

    Normalize step: with ``normalize_cost=True`` the solver computes
    C_max = max_ij κ(p_i, q_j) exactly on every solve (``geo_cost.exact_c_max``)
    and iterates on κ / C_max unless C_max / c_max is exactly 1 or C_max is 0;
    ``info["cost_scale"] = C_max / c_max``, and every output is on the
    iterated scale. The iteration uses the marginals floored with δ_m =
    ``params.eps_mass``; every check and delivered metric uses ``check_r``,
    ``check_c``, which default to the original r, c. r_D (``dual_res_F``) is
    NaN unless ``native_dual_residual=True`` or a full check ran on the final
    state.

    With ``BDRS_GEO_KERNELS=0``, without CUDA, or with ``P`` on the CPU, the
    solve runs in ``factored_torch.bregman_dr_solve_pointcloud`` on the same
    cost. Its iterates agree with the kernels' only to rounding: nvcc may
    contract each term's multiply-add into a fused multiply-add, and the
    reductions run in a different order. It returns the iteration count,
    residuals, objective and η as 0-dim tensors, not Python numbers;
    ``info["cuda_graph"]`` records which path ran. ``_ext_override`` replaces
    the extension and forces the session path (a test seam).

    Returns ``(X, info)``, X the m×n plan or ``None`` with ``return_plan=False``.
    Raises ValueError for an unknown ``cost_type`` or ``stop``, both
    ``gap_stop_fn`` and ``traj_callback``, ``gap_stop_fn`` with
    ``stop="kkt_triple"``, non-fp64 inputs, or ``c_max <= 0``; the session
    raises RuntimeError for a dimension d outside [1, 16].
    """
    from src.methods import geo_cost
    from src.methods.factored_torch import (
        _pointcloud_block_chunk,
        bregman_dr_solve_pointcloud,
    )

    if cost_type not in _POINTCLOUD_COST_TYPES:
        raise ValueError(
            f"Unknown cost_type {cost_type!r}; valid: {sorted(_POINTCLOUD_COST_TYPES)}")
    build_default, gap_stop_tol = _resolve_stop(
        params, stop, gap_stop_fn, traj_callback, gap_stop_tol)
    if P.dtype != torch.float64 or Q.dtype != torch.float64 \
            or r.dtype != torch.float64 or c.dtype != torch.float64:
        raise ValueError("P, Q, r, c must be torch.float64.")
    if float(c_max) <= 0.0:
        raise ValueError(f"c_max must be positive; got {c_max!r}")

    r_orig, c_orig = r, c
    r_it = floor_marginal(r_orig, params.eps_mass)
    c_it = floor_marginal(c_orig, params.eps_mass)
    r_chk, c_chk = _check_marginals(r_orig, c_orig, check_r, check_c)

    # max_ij(κ_ij / c_max) = (max_ij κ_ij) / c_max bit for bit (rounded division
    # by a positive constant is monotone), so this no-op test agrees with the
    # one the torch solver applies to the instance cost_fn.
    cost_scale = 1.0
    c_max_eff = float(c_max)
    if normalize_cost:
        raw_max = geo_cost.exact_c_max(P, Q, cost_type)
        cmax_inst = raw_max / float(c_max)
        if cmax_inst > 0.0 and cmax_inst != 1.0:
            cost_scale = cmax_inst
            c_max_eff = raw_max                    # iterate on κ / C_max, maximum 1
    cost_fn = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max_eff)

    use_kernels = (_ext_override is not None) or (
        os.environ.get("BDRS_GEO_KERNELS", "1") != "0"
        and torch.cuda.is_available() and P.is_cuda)
    if not use_kernels:
        # The torch solver builds the default stop itself, on the same cost_fn
        # and check marginals.
        X, info = bregman_dr_solve_pointcloud(
            r_orig, c_orig, params, cost_fn=cost_fn, return_plan=return_plan,
            stop=stop, traj_callback=traj_callback, gap_stop_fn=gap_stop_fn,
            gap_stop_tol=gap_stop_tol, check_r=r_chk, check_c=c_chk,
            normalize_cost=False, native_dual_residual=native_dual_residual)
        info["cost_scale"] = cost_scale
        return X, info

    if build_default:
        gap_stop_fn = make_pointcloud_gap_stop_fn(cost_fn, r_chk, c_chk)
    ext = _ext_override if _ext_override is not None else cuda_ext.load()
    device = r.device
    eta_f = float(params.eta)
    m = int(r_it.numel())
    n = int(c_it.numel())
    chunk = _pointcloud_block_chunk(m, n)
    max_iters_total = int(params.max_iters)
    check_every = max(1, int(params.check_every))

    sess = ext.PointCloudBdrsSession(
        P=P.contiguous(), Q=Q.contiguous(), r=r_it.contiguous(), c=c_it.contiguous(),
        eta=eta_f, cost_type=_POINTCLOUD_COST_TYPES[cost_type], c_max=float(c_max_eff),
    )
    return _factored_session_loop(
        sess, cost_fn,
        r_chk=r_chk, c_chk=c_chk, params=params, eta_f=eta_f,
        m=m, n=n, chunk=chunk, device=device,
        max_iters_total=max_iters_total, check_every=check_every,
        return_plan=return_plan, traj_callback=traj_callback,
        gap_stop_fn=gap_stop_fn, gap_stop_tol=gap_stop_tol,
        native_dual_residual=native_dual_residual, cost_scale=cost_scale,
    )


def bregman_dr_solve_general_factored_cuda(
    C: torch.Tensor,                  # [m, n] stored cost
    r: torch.Tensor,                  # [m], the original (unfloored) marginal
    c: torch.Tensor,                  # [n], the original (unfloored) marginal
    params: BregmanDRParams,
    *,
    return_plan: bool = True,
    stop: str = "feasible_gap",       # or "kkt_triple" (params.STOP_RULES)
    traj_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    gap_stop_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    gap_stop_tol: Optional[float] = None,   # None: params.epsilon_primal
    check_r: Optional[torch.Tensor] = None,
    check_c: Optional[torch.Tensor] = None,
    normalize_cost: bool = True,
    row_chunk: int = 0,
    native_dual_residual: bool = False,
    _ext_override: Any = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """General BDRS on CUDA over a stored cost ``C``, through ``ext.DenseFactoredSession``.

    The eager stored-C kernels of ``factored_torch`` take −k/η by value, so a
    captured graph of them would replay the capture-time k; the session keeps
    k in device memory and advances it there. Its iterates are bit-identical
    to the eager kernel path (each term keeps the ``__dmul_rn``/``__dadd_rn``
    form, not a fused multiply-add, and the column pass the same two-stage
    chunk-then-merge order), as ``tests/methods/test_dense_factored_session.py``
    asserts with ``torch.equal``.

    Normalize step: ``C``, ``r``, ``c`` are cast to fp64, and C is divided by
    max C unless that maximum is 1 or not positive (``info["cost_scale"]`` is
    the divisor). The iteration uses r, c floored with δ_m =
    ``params.eps_mass``; checks and delivery are those of
    ``_factored_session_loop``. The stop rule ``stop`` is that of
    ``factored_torch.bregman_dr_solve_general_factored``: by default
    ``"feasible_gap"``, with ``gap_stop_tol`` defaulting to
    ``params.epsilon_primal`` and, without ``gap_stop_fn``,
    ``dual_cleanup.make_dense_gap_stop_fn`` built by the solver on the
    normalized C and the check marginals; or ``"kkt_triple"``.

    ``BDRS_FACTORED_SESSION=0``, a CPU ``C`` or no CUDA selects the eager
    solver of ``factored_torch``; ``info["factored_session"]`` records which
    path ran. ``_ext_override`` replaces the extension and forces the session
    path (a test seam). Raises ValueError if ``C`` is not 2-D or ``r``, ``c``
    do not match its shape.
    """
    from src.methods.factored_torch import (
        _pointcloud_block_chunk,
        bregman_dr_solve_general_factored,
    )
    from src.methods.params import _as_f64, _check_dense_shapes

    C = _as_f64(C)
    # r, c become the check marginals of every gate and delivered metric.
    r = _as_f64(r)
    c = _as_f64(c)
    _check_dense_shapes(C, r, c)
    build_default, gap_stop_tol = _resolve_stop(
        params, stop, gap_stop_fn, traj_callback, gap_stop_tol)
    cost_scale = 1.0
    if normalize_cost:
        _cmax = float(C.max())
        if _cmax > 0.0 and _cmax != 1.0:
            C = C / _cmax
            cost_scale = _cmax

    use_session = (_ext_override is not None) or (
        os.environ.get("BDRS_FACTORED_SESSION", "1") != "0"
        and torch.cuda.is_available() and C.is_cuda)
    if not use_session:
        # The torch solver builds the default stop itself, on the same C and
        # check marginals.
        X, info = bregman_dr_solve_general_factored(
            C, r, c, params, return_plan=return_plan, stop=stop,
            traj_callback=traj_callback,
            gap_stop_fn=gap_stop_fn, gap_stop_tol=gap_stop_tol,
            check_r=check_r, check_c=check_c, normalize_cost=False,
            row_chunk=row_chunk, native_dual_residual=native_dual_residual)
        info["cost_scale"] = cost_scale
        info["factored_session"] = False
        return X, info

    C = C.contiguous()
    r_orig, c_orig = r, c
    r_it = floor_marginal(r_orig, params.eps_mass)
    c_it = floor_marginal(c_orig, params.eps_mass)
    r_chk, c_chk = _check_marginals(r_orig, c_orig, check_r, check_c)
    if build_default:
        gap_stop_fn = make_dense_gap_stop_fn(C, r_chk, c_chk)

    ext = _ext_override if _ext_override is not None else cuda_ext.load()
    device = C.device
    eta_f = float(params.eta)
    m, n = int(C.shape[0]), int(C.shape[1])
    chunk = _pointcloud_block_chunk(m, n, row_chunk)
    # The rule of the eager path (``_krpc`` in factored_torch): the bit-identity
    # needs the column reduction split into the same chunks, merged in the same
    # order.
    rows_per_chunk = 128 if m <= 4096 else 1024

    def _cost_rows(i0: int, i1: int) -> torch.Tensor:
        return C[i0:i1]

    sess = ext.DenseFactoredSession(
        C=C, r=r_it.contiguous(), c=c_it.contiguous(),
        eta=eta_f, rows_per_chunk=rows_per_chunk,
    )

    X, info = _factored_session_loop(
        sess, _cost_rows,
        r_chk=r_chk, c_chk=c_chk, params=params, eta_f=eta_f,
        m=m, n=n, chunk=chunk, device=device,
        max_iters_total=int(params.max_iters),
        check_every=max(1, int(params.check_every)),
        return_plan=return_plan, traj_callback=traj_callback,
        gap_stop_fn=gap_stop_fn, gap_stop_tol=gap_stop_tol,
        native_dual_residual=native_dual_residual, cost_scale=cost_scale,
    )
    info["factored_kernels"] = True
    info["factored_session"] = True
    info["factored_rows_per_chunk"] = int(rows_per_chunk)
    return X, info
