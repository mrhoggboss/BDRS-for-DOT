"""Point Cloud and General formats in torch, over blocks of cost rows.

Both formats iterate the factored state log X^k_ij = ξ_i + k·L_ij + ζ_j, with
ξ (``A``), ζ (``B``) and L = −C/η, so no m×n iterate is stored. Each half-step
is a log-sum-exp over one axis, evaluated over row blocks of the cost returned
by ``cost_fn(i0, i1)``: recomputed from coordinates for Point Cloud, row slices
of a stored C for General. On CUDA the General solver can instead run its two
per-iteration LSEs in fused kernels that read the whole stored C.
"""

import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from src.methods.dual_cleanup import make_dense_gap_stop_fn, make_pointcloud_gap_stop_fn
from src.methods.params import (
    _as_f64,
    BregmanDRParams,
    _check_dense_shapes,
    _check_marginals,
    floor_marginal,
    _nan_t,
    _resolve_stop,
)


def _stored_kernels_ext(device):
    """Return the CUDA extension if it has the stored-C LSE kernels, else None.

    None, so the tiled torch reductions run, when ``device`` is not CUDA or
    BDRS_FACTORED_KERNELS is "0"; a failed load or a missing kernel also gives
    None and warns once per process. The kernels form ``neg_scale·C + w``
    inside the kernel, so k·L is never written to a second m×n array. Their
    results differ from the torch reductions' by the rounding of a different
    LSE accumulation order.
    """
    import os as _os
    global _FALLBACK_WARNED
    if device.type != "cuda" or _os.environ.get("BDRS_FACTORED_KERNELS", "1") == "0":
        return None
    try:
        from src.methods import cuda_ext
        ext = cuda_ext.load()
        ok = hasattr(ext, "dense_row_lse_scaled") and hasattr(ext, "dense_col_lse_scaled")
        why = None if ok else "the extension lacks dense_row/col_lse_scaled (a stale build?)"
    except Exception as e:                                     # noqa: BLE001
        ext, why = None, f"loading the CUDA extension failed: {e!r}"
    if why is None:
        return ext
    if not _FALLBACK_WARNED:
        _FALLBACK_WARNED = True
        import warnings
        warnings.warn(f"{why}; the General solver falls back to the chunked torch "
                      f"reductions (BDRS_FACTORED_KERNELS=0 selects them explicitly)",
                      RuntimeWarning, stacklevel=2)
    return None


_FALLBACK_WARNED = False


def _stored_row_lse(out, C, w, neg_scale: float, ext) -> None:
    """Set out[i] = LSE_j(neg_scale·C_ij + w_j) with the fused CUDA row kernel.

    ``neg_scale`` is −k/η, already negated (a Python float), unlike the
    ``scale`` = +k/η of the torch helpers; a wrong sign raises no error and
    gives finite values. It is passed by value, so the call must not be
    captured in a CUDA graph: a replay would reuse the capture-time k.
    """
    assert not torch.cuda.is_current_stream_capturing(), (
        "dense_row_lse_scaled must not be graph-captured: neg_scale is a by-value "
        "argument, so a captured graph would replay the capture-time k forever")
    ext.dense_row_lse_scaled(out, C, w, neg_scale)


def _stored_col_lse(out, C, w, neg_scale: float, ext, scratch, rows_per_chunk: int) -> None:
    """Set out[j] = LSE_i(neg_scale·C_ij + w_i) with the fused CUDA column kernel.

    Same sign and capture rules as ``_stored_row_lse``. The kernel reduces
    blocks of ``rows_per_chunk`` rows, then merges their partial (max, sum)
    pairs, held in ``scratch``: two fp64 buffers of at least
    ceil(m/rows_per_chunk)·n entries.
    """
    assert not torch.cuda.is_current_stream_capturing(), (
        "dense_col_lse_scaled must not be graph-captured: neg_scale is a by-value "
        "argument, so a captured graph would replay the capture-time k forever")
    ext.dense_col_lse_scaled(out, C, w, neg_scale, scratch[0], scratch[1], rows_per_chunk)


def _pointcloud_block_chunk(m: int, n: int, row_chunk: int = 0) -> int:
    """Source rows per block of the tiled reductions.

    ``row_chunk > 0`` is used as given (capped at m); otherwise a (chunk, n)
    fp64 block is about 16 MiB. A metric ``cost_fn`` builds a (chunk, n, d)
    coordinate-difference temporary, d times larger.
    """
    if row_chunk > 0:
        return min(int(row_chunk), m)
    return max(1, min(m, (2 * 1024 * 1024) // max(1, n)))


def _pointcloud_row_lse(out, cost_fn, w_b, scale: float, m: int, chunk: int) -> None:
    """Set out[i] = LSE_j(−scale·C_ij + w_b[j]), tiled over source rows i (``scale`` = +k/η)."""
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        lk = cost_fn(i0, i1).mul(-scale).add_(w_b[None, :])
        out[i0:i1] = torch.logsumexp(lk, dim=1)
        del lk


def _pointcloud_col_lse(out, cost_fn, w_a, scale: float, m: int, chunk: int) -> None:
    """Set out[j] = LSE_i(−scale·C_ij + w_a[i]), tiled over source rows i (``scale`` = +k/η)."""
    m_run = None
    s_run = None
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        blk = cost_fn(i0, i1).mul(-scale).add_(w_a[i0:i1, None])
        m_c = blk.amax(dim=0)
        s_c = (blk - m_c[None, :]).exp_().sum(dim=0)
        if m_run is None:
            m_run, s_run = m_c, s_c
        else:
            m_new = torch.maximum(m_run, m_c)
            s_run = s_run * (m_run - m_new).exp_() + s_c * (m_c - m_new).exp_()
            m_run = m_new
        del blk
    out.copy_(m_run + s_run.log())


def _pointcloud_trace_CX(cost_fn, A, B, scale: float, m: int, chunk: int):
    """Return ⟨C, X⟩ = Σ_ij C_ij · exp(ξ_i − scale·C_ij + ζ_j) as a 0-dim tensor, tiled over rows."""
    device = A.device
    acc = torch.zeros((), dtype=A.dtype, device=device)
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        cb = cost_fn(i0, i1)
        x_blk = ((A[i0:i1, None] - scale * cb) + B[None, :]).exp_()
        acc = acc + (cb * x_blk).sum()
        del cb, x_blk
    return acc


def _pointcloud_dual_viol_sq(cost_fn, log_u, log_v, eta: float, m: int, chunk: int):
    """Return Σ_ij ReLU(log u_i + log v_j − C_ij/η)² as a 0-dim tensor, tiled over rows.

    η times its square root is r_D = ‖(α ⊕ β − C)₊‖_F for α = η·log u,
    β = η·log v.
    """
    device = log_u.device
    acc = torch.zeros((), dtype=log_u.dtype, device=device)
    inv_eta = 1.0 / eta
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        viol = (log_u[i0:i1, None] + log_v[None, :]
                - cost_fn(i0, i1).mul(inv_eta)).clamp_min_(0.0)  # .mul, not .mul_: cost_fn may return a view of a stored C
        acc = acc + (viol * viol).sum()
        del viol
    return acc


@torch.no_grad()
def bregman_dr_solve_pointcloud(
    r: torch.Tensor,                 # length m, original source marginal
    c: torch.Tensor,                 # length n, original target marginal
    params: BregmanDRParams,
    *,
    cost_fn: Callable[[int, int], torch.Tensor],   # cost_fn(i0, i1) -> [i1-i0, n]
    return_plan: bool = True,
    stop: str = "feasible_gap",      # or "kkt_triple" (params.STOP_RULES)
    traj_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    gap_stop_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    gap_stop_tol: Optional[float] = None,   # None: params.epsilon_primal
    check_r: Optional[torch.Tensor] = None,
    check_c: Optional[torch.Tensor] = None,
    row_chunk: int = 0,
    normalize_cost: bool = True,
    native_dual_residual: bool = False,
    stored_C: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Point Cloud BDRS on the factored state in torch, with the cost given by ``cost_fn``.

    ``cost_fn(i0, i1)`` returns the cost block of source rows i0..i1−1
    (``geo_cost.make_metric_cost_fn`` builds one); ``row_chunk`` overrides the
    block height (``_pointcloud_block_chunk``).

    Normalize step: with ``normalize_cost=True`` the solver computes
    C_max = max_ij cost_fn exactly on every solve (``geo_cost.exact_cost_fn_max``)
    and iterates on cost_fn / C_max, unless C_max is 1 or not positive, which
    makes a max-1 cost bit-identical to ``normalize_cost=False``.
    ``info["cost_scale"]`` is the divisor, and every output is on the iterated
    scale. The iteration uses the marginals r̃, c̃ floored with δ_m =
    ``params.eps_mass``; every check and delivered metric uses ``check_r``,
    ``check_c``, which default to the original r, c.

    Stopping, checked every T_chk = ``params.check_every`` iterations, by the
    rule ``stop``:
      - ``"feasible_gap"`` (the default): once the row-only residual
        ‖X1 − r‖₂ is below ``epsilon_primal``, ⟨C, X⟩ and ``gap_stop_fn`` are
        evaluated, and the solve stops when its value is below
        ``gap_stop_tol`` (default ``epsilon_primal``). Without
        ``gap_stop_fn`` the solver builds
        ``dual_cleanup.make_pointcloud_gap_stop_fn`` on the iterated cost and
        the check marginals, whose value is |g_feas| of the alternating
        c-transform. The column term of the residual is left out: the
        ζ-update makes Xᵀ1 = c̃ to rounding, so it is the constant
        ‖c̃ − c‖₂.
      - ``"kkt_triple"``: the solve stops when r_P, r_D and |g| of
        (η·log u, η·log v) are at most ``epsilon_primal``, ``epsilon_dual``
        and ``epsilon_gap``. r_D and g are evaluated only once
        r_P ≤ ``epsilon_primal``, or at every check with ``params.verbosity``.
    With ``traj_callback`` (trajectory mode) the full metrics go to the
    callback at every check and the KKT-triple test applies, whatever
    ``stop`` is. ``gap_stop_fn`` together with ``traj_callback`` or with
    ``stop="kkt_triple"`` raises ValueError.

    Delivery, on the final iterate: r_P, ⟨C, X⟩ and the native gap
    |⟨C, X⟩ − η(rᵀlog u + cᵀlog v)|. r_D (``dual_res_F``) is NaN unless
    ``native_dual_residual=True`` or, under the KKT-triple test, the last
    full check ran on the final iterate.

    ``stored_C``: an optional contiguous fp64 CUDA (m, n) copy of the cost.
    When the stored-C kernels are available (``_stored_kernels_ext``) the two
    per-iteration LSEs read it; checks, delivery and X still use ``cost_fn``.
    It must equal the blocks of ``cost_fn`` (not checked); with
    ``normalize_cost=True`` a cost whose maximum is positive and not 1 raises
    ValueError, since only ``cost_fn`` could be rescaled.

    Returns ``(X, info)``, X the m×n plan or ``None`` with ``return_plan=False``.
    ``info["log_u"]``, ``info["log_v"]`` are the last increments ξ^k − ξ^(k−1),
    ζ^k − ζ^(k−1); the native duals are α = η·log u, β = η·log v.
    """
    build_default, gap_stop_tol = _resolve_stop(
        params, stop, gap_stop_fn, traj_callback, gap_stop_tol)

    r_orig = _as_f64(r)
    c_orig = _as_f64(c)
    r = floor_marginal(r_orig, params.eps_mass)
    c = floor_marginal(c_orig, params.eps_mass)
    r_chk, c_chk = _check_marginals(r_orig, c_orig, check_r, check_c)
    m = int(r.numel())
    n = int(c.numel())
    device = r.device
    dtype = torch.float64
    eta = float(params.eta)

    chunk = _pointcloud_block_chunk(m, n, row_chunk)
    _kern = _stored_kernels_ext(device) if stored_C is not None else None
    _kscr = None
    _krpc = 1024
    if _kern is not None:
        # The column kernel's first stage launches ceil(n/256)·ceil(m/_krpc)
        # blocks: at m = n = 4096 and 1024 rows per chunk that is 64, fewer
        # than the 108 SMs of an A100, so small m takes 128 rows per chunk.
        # factored_cuda's General session uses the same rule; its bit-identity
        # with this path depends on it.
        _krpc = 128 if m <= 4096 else 1024
        _nch = (m + _krpc - 1) // _krpc
        _kscr = (torch.empty(_nch * n, dtype=dtype, device=device),
                 torch.empty(_nch * n, dtype=dtype, device=device))

    cost_scale = 1.0
    if normalize_cost:
        from src.methods.geo_cost import exact_cost_fn_max
        _cmax = exact_cost_fn_max(cost_fn, m, row_chunk=chunk)
        if _cmax > 0.0 and _cmax != 1.0:
            if stored_C is not None:
                # The iteration would read the unscaled stored_C while every
                # check reads cost_fn / C_max.
                raise ValueError(
                    "stored_C needs a cost with max 1: normalize it first and pass "
                    f"normalize_cost=False (max of the cost: {_cmax!r})")
            cost_scale = _cmax
            _cost_fn_inst = cost_fn

            def cost_fn(i0: int, i1: int, _f=_cost_fn_inst, _s=float(_cmax)) -> torch.Tensor:
                return _f(i0, i1) / _s          # new tensor: never mutates a stored-C view

    if build_default:
        # On the cost the iteration reads and the marginals every check uses.
        gap_stop_fn = make_pointcloud_gap_stop_fn(cost_fn, r_chk, c_chk)

    log_r = r.log()
    log_c = c.log()

    # Uniform X^0 = 1/(mn), split as ξ^0 = −log m, ζ^0 = −log n; the duals and
    # X do not depend on the split.
    A = torch.full((m,), -float(math.log(m)), dtype=dtype, device=device)
    B = torch.full((n,), -float(math.log(n)), dtype=dtype, device=device)
    log_u = torch.zeros(m, dtype=dtype, device=device)
    log_v = torch.zeros(n, dtype=dtype, device=device)

    lse_u = torch.empty(m, dtype=dtype, device=device)
    lse_v = torch.empty(n, dtype=dtype, device=device)   # also read by _primal_t
    lse_rowck = torch.empty(m, dtype=dtype, device=device)

    def _primal_row_t():
        # ‖X1 − r‖₂ only: the ζ-update makes Xᵀ1 = c̃ to rounding.
        _pointcloud_row_lse(lse_rowck, cost_fn, B, k / eta, m, chunk)
        return ((A + lse_rowck).exp() - r_chk).norm(2)

    def _primal_t():
        # r_P = ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂. The column sums exp(ζ + lse_v) reuse the
        # ζ-update's LSE_i(k·L_ij + ξ_i).
        _pointcloud_row_lse(lse_rowck, cost_fn, B, k / eta, m, chunk)
        row = ((A + lse_rowck).exp() - r_chk).norm(2)
        col = ((B + lse_v).exp() - c_chk).norm(2)
        return row + col

    def _obj_t():
        return _pointcloud_trace_CX(cost_fn, A, B, k / eta, m, chunk)

    def _full_metrics(with_dual: bool = True):
        """Return (r_P, r_D, |g|, ⟨C, X⟩) as floats; r_D is NaN with ``with_dual=False``."""
        primal_t = _primal_t()
        if with_dual:
            dual_t = eta * _pointcloud_dual_viol_sq(cost_fn, log_u, log_v, eta, m, chunk).sqrt()
        else:
            dual_t = _nan_t(device)
        obj_t = _obj_t()
        gap_t = (obj_t - eta * (log_u.dot(r_chk) + log_v.dot(c_chk))).abs()
        vals = torch.stack([primal_t, dual_t, gap_t, obj_t]).tolist()   # one host sync
        return vals[0], vals[1], vals[2], vals[3]

    max_iters = int(params.max_iters)
    check_every = max(1, int(params.check_every))

    k = 0
    native_termination = "max_iters"
    proxy_passes = 0
    full_checks = 0        # r_D evaluations
    sep_obj_checks = 0     # ⟨C, X⟩ evaluations on the gap-stop path, named as in grid_cuda
    last = None
    last_iter = -1
    last_gap_val = None
    last_obj_t = None
    last_obj_iter = -1
    primal_proxy_value = float("inf")

    while k < max_iters:
        k += 1
        scale = k / eta
        # ξ_i = log r̃_i − LSE_j(−(k/η)·C_ij + β̃_j), with β̃ = ζ + log v.
        beta_tilde = B + log_v
        if _kern is not None:
            _stored_row_lse(lse_u, stored_C, beta_tilde, -scale, _kern)
        else:
            _pointcloud_row_lse(lse_u, cost_fn, beta_tilde, scale, m, chunk)
        A_old = A
        A = log_r - lse_u
        log_u = A - A_old
        # ζ_j = log c̃_j − LSE_i(−(k/η)·C_ij + ξ_i).
        if _kern is not None:
            _stored_col_lse(lse_v, stored_C, A, -scale, _kern, _kscr, _krpc)
        else:
            _pointcloud_col_lse(lse_v, cost_fn, A, scale, m, chunk)
        B_old = B
        B = log_c - lse_v
        log_v = B - B_old

        if k % check_every == 0:
            if gap_stop_fn is not None:
                primal_proxy_value = float(_primal_row_t().item())
                gap_val = None
                if primal_proxy_value < params.epsilon_primal:
                    proxy_passes += 1
                    obj_t = _obj_t()
                    last_obj_t, last_obj_iter = obj_t, k
                    sep_obj_checks += 1
                    gap_val = float(gap_stop_fn({
                        "iters": k, "eta": eta,
                        "log_u": log_u, "log_v": log_v,
                        "objective": obj_t,
                    }))
                    last_gap_val = gap_val
                if params.verbosity and (k % params.log_every == 0):
                    print(f"iter {k}: rP(row)={primal_proxy_value:.3e}"
                          + ("" if gap_val is None else f", |g_feas|={gap_val:.3e}"))
                # gap_val is set only when the proxy is below epsilon_primal.
                if gap_val is not None and gap_val < gap_stop_tol:
                    native_termination = "feasible_gap_below_tol"
                    break
            else:
                need_full = (traj_callback is not None) or params.verbosity
                primal = float(_primal_t().item())
                if primal <= params.epsilon_primal or need_full:
                    pr, du, ga, ob = _full_metrics()
                    full_checks += 1
                    last = (pr, du, ga, ob)
                    last_iter = k
                    if traj_callback is not None:
                        extremes = torch.stack([
                            log_u.abs().max(), log_v.abs().max()
                        ]).tolist()
                        traj_callback({
                            "k": k, "eta": eta,
                            "primal_res_L2": pr, "dual_res_F": du,
                            "gap_abs": ga, "objective": ob,
                            "max_abs_log_u": extremes[0],
                            "max_abs_log_v": extremes[1],
                        })
                    if params.verbosity and (k % params.log_every == 0):
                        print(f"iter {k}: obj={ob:.6e}, rP(L2)={pr:.3e}, "
                              f"rD(F)={du:.3e}, gap={ga:.3e}")
                    if (pr <= params.epsilon_primal
                            and du <= params.epsilon_dual
                            and ga <= params.epsilon_gap):
                        native_termination = "kkt_below_tol"
                        break

    # Delivery on the final iterate. A full in-loop check on this iterate is
    # reused, its r_D included; otherwise r_D only with native_dual_residual.
    if k == 0:
        _pointcloud_col_lse(lse_v, cost_fn, A, 0.0, m, chunk)   # max_iters ≤ 0: no ζ-update has filled lse_v
    if last is not None and last_iter == k:
        primal_f, dual_f, gap_f, obj_f = last
    elif gap_stop_fn is not None:
        primal_t = _primal_t()
        obj_t = last_obj_t if (last_obj_iter == k and last_obj_t is not None) else _obj_t()
        if native_dual_residual:
            dual_t = eta * _pointcloud_dual_viol_sq(cost_fn, log_u, log_v, eta, m, chunk).sqrt()
            full_checks += 1
        else:
            dual_t = _nan_t(device)
        gap_t = (obj_t - eta * (log_u.dot(r_chk) + log_v.dot(c_chk))).abs()
        primal_f, dual_f, gap_f, obj_f = torch.stack(
            [primal_t, dual_t, gap_t, obj_t.reshape(())]).tolist()   # one host sync
    else:
        primal_f, dual_f, gap_f, obj_f = _full_metrics(with_dual=native_dual_residual)
        if native_dual_residual:
            full_checks += 1

    X = None
    if return_plan:
        scale = k / eta
        X = torch.empty((m, n), dtype=dtype, device=device)
        for i0 in range(0, m, chunk):
            i1 = min(m, i0 + chunk)
            cb = cost_fn(i0, i1)
            X[i0:i1] = ((A[i0:i1, None] - scale * cb) + B[None, :]).exp_()
            del cb

    info = {
        "iters": torch.tensor(k, dtype=torch.int32),
        "primal_res_L2": torch.tensor(primal_f, dtype=torch.float64, device=device),
        "gap_abs": torch.tensor(gap_f, dtype=torch.float64, device=device),
        "objective": torch.tensor(obj_f, dtype=torch.float64, device=device),
        "eta": torch.tensor(eta, dtype=torch.float64),
        "dual_res_F": torch.tensor(dual_f, dtype=torch.float64, device=device),
        "log_u": log_u,
        "log_v": log_v,
        "native_termination": native_termination,
        # The fallback to the torch LSEs raises no error, so a kernel-vs-torch
        # comparison must check this key.
        "factored_kernels": bool(_kern is not None),
        "factored_rows_per_chunk": int(_krpc) if _kern is not None else None,
        "proxy_passes": proxy_passes,
        "full_kkt_checks": full_checks,
        "sep_obj_checks": sep_obj_checks,
        "primal_res_proxy_final": primal_proxy_value,
        "native_dual_residual": bool(native_dual_residual),
        "marginal_floor": float(params.eps_mass),
        "cost_scale": cost_scale,
        "cuda_graph": False,
    }
    if last_gap_val is not None:
        info["gap_stop_value"] = last_gap_val
    return X, info


@torch.no_grad()
def bregman_dr_solve_general_factored(
    C: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    params: BregmanDRParams,
    *,
    return_plan: bool = True,
    stop: str = "feasible_gap",      # or "kkt_triple" (params.STOP_RULES)
    traj_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    gap_stop_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    gap_stop_tol: Optional[float] = None,   # None: params.epsilon_primal
    check_r: Optional[torch.Tensor] = None,
    check_c: Optional[torch.Tensor] = None,
    normalize_cost: bool = True,
    row_chunk: int = 0,
    native_dual_residual: bool = False,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """General BDRS on the factored state in torch, over a stored cost C of shape (m, n).

    The solve is ``bregman_dr_solve_pointcloud`` with ``cost_fn`` returning row
    slices of C, so each LSE re-reads C and no further m×n array is formed
    during the iteration; on a contiguous CUDA C the two per-iteration LSEs use
    the fused stored-C kernels when available. The iterates match the dense
    log-domain recursion up to the rounding of the reassociated LSE sums.

    Normalize step: C is cast to fp64 and replaced by the new array C / max C
    unless that maximum is 1 or not positive, so a max-1 cost runs
    bit-identically; the caller's C is never modified and stays allocated
    beside the copy. ``info["cost_scale"]`` is the divisor.

    The stop is that of ``bregman_dr_solve_pointcloud``. Under the default
    ``stop="feasible_gap"`` without ``gap_stop_fn``, the solver builds
    ``dual_cleanup.make_dense_gap_stop_fn`` on the normalized C and the check
    marginals; a ``gap_stop_fn`` passed in must be built on the normalized C
    too. Raises ValueError unless C is 2-D and r, c have shapes (m,), (n,).
    """
    C = _as_f64(C)
    _check_dense_shapes(C, r, c)
    build_default, gap_stop_tol = _resolve_stop(
        params, stop, gap_stop_fn, traj_callback, gap_stop_tol)
    cost_scale = 1.0
    if normalize_cost:
        _cmax = float(C.max())
        if _cmax > 0.0 and _cmax != 1.0:
            C = C / _cmax
            cost_scale = _cmax
    if build_default:
        r_chk, c_chk = _check_marginals(_as_f64(r), _as_f64(c), check_r, check_c)
        gap_stop_fn = make_dense_gap_stop_fn(C, r_chk, c_chk)

    def _cost_rows(i0: int, i1: int) -> torch.Tensor:
        return C[i0:i1]

    X, info = bregman_dr_solve_pointcloud(
        r, c, params,
        cost_fn=_cost_rows,
        stored_C=C if (C.is_cuda and C.is_contiguous()) else None,
        check_r=check_r,
        check_c=check_c,
        return_plan=return_plan,
        stop=stop,
        traj_callback=traj_callback,
        gap_stop_fn=gap_stop_fn,
        gap_stop_tol=gap_stop_tol,
        row_chunk=row_chunk,
        normalize_cost=False,            # C is normalized above
        native_dual_residual=native_dual_residual,
    )
    info["cost_scale"] = cost_scale
    return X, info
