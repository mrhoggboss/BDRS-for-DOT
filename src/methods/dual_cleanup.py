"""Exactly dual-feasible potentials by the c-transform, and the feasible-gap certificate.

The BDRS potentials α = η·log u, β = η·log v (``log_u``, ``log_v``) satisfy
α_i + β_j ≤ C_ij only approximately. The c-transform restores feasibility up
to rounding:

    β'_j = min_i (C_ij − α_i)   ⇒   α_i + β'_j ≤ C_ij for all i, j.

Given α, β' is the largest feasible β, so for c ≥ 0 it maximizes the dual
objective rᵀα + cᵀβ over the feasible β. By weak duality, for any plan X the
feasible gap g_feas = ⟨C,X⟩ − (rᵀα' + cᵀβ') bounds ⟨C,X⟩ − OT* from above
without knowing OT*; the solvers stop on it.

Every ``r``, ``c`` argument is the instance's original marginal pair (the
floor δ_m acts inside the solver iteration only), and ``C`` or ``cost_fn`` is
the max-1 cost the solver iterates on, so the certificate and ⟨C,X⟩ share one
scale.
"""

import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from src.methods.grid_factored import _check_chunk_rows
from src.methods.grid_cost import (
    _build_grid_indices,
    _cost_axis,
    _cost_row_chunk,
    _grid_blocks,
    _phi_and_cmax,
    _row_chunk_for,
)


# ============================================================================
# The core: a c-transform is a minimum over the cost, taken over row blocks
# from a block source ``cost_fn(i0, i1) -> C[i0:i1, :]``:
#
#   β'_j = min_i (C_ij − α_i)   (column minimum: transform of the row potential)
#   α'_i = min_j (C_ij − β_j)   (row minimum: transform of the column potential)
#
# A minimum does not round, so the block size never changes a c-transform; the
# sum r_D does depend on it.
# ============================================================================

def _pointcloud_ct_chunk(m: int, n: int, row_chunk: int = 0) -> int:
    """Rows per (rows, n) block, between 1 and m: ``row_chunk`` if positive,
    else ⌊2 Mi / n⌋ (2 Mi fp64 entries are 16 MiB)."""
    if row_chunk > 0:
        return min(int(row_chunk), m)
    return max(1, min(m, (2 * 1024 * 1024) // max(1, n)))


def _ctransform_symbolic_colmin(p: torch.Tensor, cost_fn, m: int, n: int,
                                *, row_chunk: int = 0) -> torch.Tensor:
    """q_j = min_i (C_ij − p_i), so p = α gives β'."""
    p = p.reshape(-1)
    chunk = _pointcloud_ct_chunk(m, n, row_chunk)
    q: Optional[torch.Tensor] = None
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        block_min = (cost_fn(i0, i1) - p[i0:i1, None]).min(dim=0).values
        q = block_min if q is None else torch.minimum(q, block_min)
    assert q is not None
    return q


def _ctransform_symbolic_rowmin(p: torch.Tensor, cost_fn, m: int, n: int,
                                *, row_chunk: int = 0) -> torch.Tensor:
    """q_i = min_j (C_ij − p_j), so p = β gives α'."""
    p = p.reshape(-1)
    chunk = _pointcloud_ct_chunk(m, n, row_chunk)
    return torch.cat([(cost_fn(i0, min(m, i0 + chunk)) - p[None, :]).min(dim=1).values
                      for i0 in range(0, m, chunk)])


def _dual_violation_sweep(alpha: torch.Tensor, beta: torch.Tensor, cost_fn, m: int, *,
                          chunk: int, acc_like: torch.Tensor, want_fro: bool = True,
                          want_max: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """``(‖V₊‖_F, max_ij V_ij)`` for V = α ⊕ β − C, each None unless requested.

    The maximum is not clamped at 0. The square sum accumulates block by block
    in ``acc_like``'s dtype (promoted when V is wider), so its last bits
    depend on ``chunk``: each caller keeps a fixed block size so that its r_D
    is reproducible."""
    acc = torch.zeros((), device=acc_like.device, dtype=acc_like.dtype)
    mx = torch.full((), float("-inf"), device=acc_like.device, dtype=acc_like.dtype)
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        viol = alpha[i0:i1, None] + beta[None, :] - cost_fn(i0, i1)
        if want_max:
            mx = torch.maximum(mx, viol.max())
        if want_fro:
            viol.clamp_min_(0.0)
            acc = acc + (viol * viol).sum()
        del viol
    return (float(acc.sqrt().item()) if want_fro else None,
            float(mx.item()) if want_max else None)


def _dual_feas_F_symbolic(alpha: torch.Tensor, beta: torch.Tensor, cost_fn,
                          m: int, n: int, *, row_chunk: int = 0,
                          out_like: Optional[torch.Tensor] = None) -> float:
    """r_D = ‖(α ⊕ β − C)₊‖_F of a symbolic cost, with ``acc_like`` of
    ``_dual_violation_sweep`` set to ``out_like`` (default ``alpha``)."""
    alpha = alpha.reshape(-1)
    return _dual_violation_sweep(alpha, beta.reshape(-1), cost_fn, m,
                                 chunk=_pointcloud_ct_chunk(m, n, row_chunk),
                                 acc_like=alpha if out_like is None else out_like)[0]


def _dual_feas_inf_symbolic(alpha: torch.Tensor, beta: torch.Tensor, cost_fn,
                            m: int, n: int, *, row_chunk: int = 0) -> float:
    """r_Dinf = max(0, max_ij (α_i + β_j − C_ij)) of a symbolic cost: a
    diagnostic, since the KKT residual uses the Frobenius r_D."""
    alpha = alpha.reshape(-1)
    beta = beta.reshape(-1)
    chunk = _pointcloud_ct_chunk(m, n, row_chunk)
    worst = -float("inf")
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        blk = alpha[i0:i1, None] + beta[None, :] - cost_fn(i0, i1)
        worst = max(worst, float(blk.max().item()))
    return max(0.0, worst)


def cost_inf_symbolic(cost_fn, m: int, n: int, *, row_chunk: int = 0) -> float:
    """‖C‖_∞ = max_ij |C_ij| of a symbolic cost, in fp64 over the row blocks of
    the symbolic c-transform."""
    chunk = _pointcloud_ct_chunk(m, n, row_chunk)
    mx = None
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        b_mx = cost_fn(i0, i1).to(torch.float64).abs().max()
        mx = b_mx if mx is None else torch.maximum(mx, b_mx)
    return float(mx.item())


# ============================================================================
# Shared by every backend: the variant dispatch, the certificate of a feasible
# pair, and the gap stop built on it.
# ============================================================================

def _by_variant(alpha: torch.Tensor, beta: torch.Tensor, variant: str,
                row_to_col: Callable[[torch.Tensor], torch.Tensor],
                col_to_row: Callable[[torch.Tensor], torch.Tensor],
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(α', β')`` for ``variant``, given ``row_to_col(α) = β'`` and
    ``col_to_row(β) = α'``.

    Every variant gives α' ⊕ β' ≤ C up to rounding. Under ``"alternating"``,
    α ⊕ β' ≤ C gives α' ≥ α entrywise, so for r ≥ 0 its dual objective is at
    least that of ``"beta"``.
    """
    if variant == "beta":
        return alpha, row_to_col(alpha)
    if variant == "alpha":
        return col_to_row(beta), beta
    if variant == "alternating":
        bp = row_to_col(alpha)
        return col_to_row(bp), bp
    raise ValueError(f"unknown variant {variant!r}")


def _certificate(ap: torch.Tensor, bp: torch.Tensor, r: torch.Tensor, c: torch.Tensor,
                 objective: Union[float, torch.Tensor],
                 ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """``(α', β', g_feas)`` with g_feas = ⟨C,X⟩ − (rᵀα' + cᵀβ').

    ``objective`` is ⟨C,X⟩ as a float or a 0-dim tensor; the tensor form
    reads back only g_feas, so a check synchronizes once."""
    dual_obj_t = (r.reshape(-1) * ap).sum() + (c.reshape(-1) * bp).sum()
    if isinstance(objective, torch.Tensor):
        g_feas = float((objective.reshape(()) - dual_obj_t).item())
    else:
        g_feas = float(objective) - float(dual_obj_t.item())
    return ap, bp, float(g_feas)


def make_gap_stop_fn(
    certify: Callable[[Dict[str, Any], Union[float, torch.Tensor]],
                      Tuple[torch.Tensor, torch.Tensor, float]],
    *, cache: Optional[Dict[str, Any]] = None,
) -> Callable[[Dict[str, Any]], float]:
    """Build a solver's ``gap_stop_fn`` around one backend's certificate.

    ``certify(info, objective)`` returns ``(α', β', g_feas)``. The solvers call
    the returned function only at checks whose primal proxy is below
    ``epsilon_primal``, and stop when its value |g_feas| is below
    ``gap_stop_tol`` (default ``epsilon_primal``); the absolute value is
    needed because X is only near-feasible, so g_feas can be negative.

    ``cache``, when given, receives the last evaluated certificate (``iters``,
    ``alpha_p``, ``beta_p``, ``g_feas``), so that delivery can reuse it instead
    of recomputing it.
    """
    def _gap_fn(info: Dict[str, Any]) -> float:
        ap, bp, g_feas = certify(info, info["objective"])
        if cache is not None:
            cache.update({
                "iters": int(info["iters"]), "alpha_p": ap, "beta_p": bp,
                "g_feas": g_feas,
            })
        return abs(g_feas)

    return _gap_fn


# ============================================================================
# Grid backends ({l1, sql2} on an n_g × n_g lattice). With the flat index
# m ↔ (m_a, m_b) = (m // n_g, m % n_g), C[m, k] = axis[m_a, k_a] +
# axis[m_b, k_b], where axis = φ(|Δ|)/C_max. The Grid cost is symmetric, so
# one function q_m = min_k (C_mk − p_k) gives β' (p = α) and α' (p = β).
# ============================================================================

def _cost_axis_1d(cost_type: str, ng: int, *, device, dtype=torch.float64) -> torch.Tensor:
    """ax[d] = φ(d)/C_max for d = 0..n_g−1, row 0 of ``_cost_axis``."""
    phi, c_max = _phi_and_cmax(cost_type, ng, dtype=dtype, device=device)
    return phi / c_max


def _ctransform(p: torch.Tensor, ng: int, cost_type: str, *, row_chunk: int = 0,
                cost_axis: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dense Grid c-transform q_m = min_k (C_mk − p_k), with C rebuilt in row
    blocks: O(n_g⁴) work, the reference for ``_ctransform_separable``."""
    device, dtype = p.device, p.dtype
    N = ng * ng
    if cost_axis is None:
        cost_axis = _cost_axis(cost_type, ng, device=device, dtype=dtype)
    a_r_full, a_c_full = _build_grid_indices(ng, device=device)

    def rows(i0: int, i1: int) -> torch.Tensor:
        return _cost_row_chunk(a_r_full[i0:i1], a_c_full[i0:i1], a_r_full, a_c_full, cost_axis)

    return _ctransform_symbolic_rowmin(p, rows, N, N,
                                       row_chunk=_row_chunk_for(N, row_chunk=row_chunk))


def _ctransform_separable(p: torch.Tensor, ng: int, cost_type: str, *, row_chunk: int = 0,
                          cost_axis: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Separable Grid c-transform q_m = min_k (C_mk − p_k): O(n_g³) work, no N × N array.

    With P[k_a, k_b] = p[k_a·n_g + k_b], the minimum splits into two 1-D
    min-plus passes,

        H[m_a, k_b] = min_{k_a} ( axis[m_a, k_a] − P[k_a, k_b] )
        Q[m_a, m_b] = min_{k_b} ( axis[m_b, k_b] + H[m_a, k_b] )

    the zero-temperature counterpart of the solver's two-pass log-sum-exp, in
    cost units (no η). The result agrees with ``_ctransform`` to rounding: the
    two add the same three terms in a different order.

    On CUDA in fp64, when the extension is already loaded and
    ``BDRS_CT_KERNEL`` is not ``0``, a fused kernel computes the same terms
    and the same output bit for bit, NaN included. The kernel reads
    only row 0 of ``cost_axis``, which must therefore be the ``_cost_axis``
    table of this (ng, cost_type). ``row_chunk`` is ignored (it matches
    ``_ctransform``'s signature).
    """
    device, dtype = p.device, p.dtype

    if (p.is_cuda and dtype == torch.float64
            and os.environ.get("BDRS_CT_KERNEL", "1") != "0"):
        # loaded(), not load(): a c-transform never triggers a JIT build.
        from src.methods import cuda_ext
        ext = cuda_ext.loaded()
        if ext is not None and hasattr(ext, "ctransform_separable"):
            if cost_axis is not None:
                ax1 = cost_axis[0].contiguous()
            else:
                ax1 = _cost_axis_1d(cost_type, ng, device=device, dtype=dtype).contiguous()
            return ext.ctransform_separable(p.contiguous().reshape(-1), ax1, ng)

    if cost_axis is None:
        axis = _cost_axis(cost_type, ng, device=device, dtype=dtype)
    else:
        axis = cost_axis
    P = p.reshape(ng, ng)
    rows = _check_chunk_rows(ng, ng * ng)

    def _min_streamed(block, dim):
        return torch.cat([block(i0, min(ng, i0 + rows)).min(dim=dim).values
                          for i0 in range(0, ng, rows)], dim=0)

    # H: (m_a, k_a, k_b) → (m_a, k_b).
    H = _min_streamed(lambda i0, i1: axis[i0:i1].unsqueeze(2) - P.unsqueeze(0), 1)
    # Q: (m_a, m_b, k_b) → (m_a, m_b).
    Q = _min_streamed(lambda i0, i1: axis.unsqueeze(0) + H[i0:i1].unsqueeze(1), 2)
    return Q.reshape(ng * ng)


def _resolve_ctransform(backend, cost_type: str):
    """The Grid c-transform for ``backend``: ``"separable"`` (the default for
    None) or the ``"dense"`` reference."""
    if cost_type not in ("l1", "sql2"):
        raise ValueError(
            f"the Grid c-transform needs a separable cost (l1, sql2); got {cost_type!r}")
    if backend is None:
        backend = "separable"
    if backend == "separable":
        return _ctransform_separable
    if backend == "dense":
        return _ctransform
    raise ValueError(f"unknown c-transform backend {backend!r}")


def dual_cleanup_ctransform(
    alpha: torch.Tensor, beta: torch.Tensor, ng: int, cost_type: str,
    *, variant: str = "alternating", backend=None, row_chunk: int = 0,
    cost_axis: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A feasible (α', β') for the Grid cost (``variant`` as in ``_by_variant``,
    ``backend`` as in ``_resolve_ctransform``)."""
    ct = _resolve_ctransform(backend, cost_type)

    def transform(p):
        return ct(p, ng, cost_type, row_chunk=row_chunk, cost_axis=cost_axis)

    return _by_variant(alpha, beta, variant, transform, transform)


def _dual_feas_F(alpha: torch.Tensor, beta: torch.Tensor, ng: int, cost_type: str,
                 *, row_chunk: int = 0) -> float:
    """r_D = ‖(α ⊕ β − C)₊‖_F on the Grid, with C rebuilt in row blocks in
    α's dtype."""
    cost_axis = _cost_axis(cost_type, ng, device=alpha.device, dtype=alpha.dtype)
    return _dual_violation_sweep(alpha, beta, _grid_blocks(cost_axis, ng), ng * ng,
                                 chunk=_row_chunk_for(ng * ng, row_chunk=row_chunk),
                                 acc_like=alpha)[0]


def feasible_dual_and_gap(
    log_u: torch.Tensor, log_v: torch.Tensor, eta: float,
    r: torch.Tensor, c: torch.Tensor, ng: int, cost_type: str,
    objective: Union[float, torch.Tensor],
    *, variant: str = "alternating", backend=None, row_chunk: int = 0,
    cost_axis: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Grid certificate ``(α', β', g_feas)`` from α = η·log u,
    β = η·log v; ``objective`` is the current ⟨C,X⟩."""
    alpha = eta * log_u.reshape(-1)
    beta = eta * log_v.reshape(-1)
    ap, bp = dual_cleanup_ctransform(
        alpha, beta, ng, cost_type, variant=variant, backend=backend,
        row_chunk=row_chunk, cost_axis=cost_axis)
    return _certificate(ap, bp, r, c, objective)


def make_grid_gap_stop_fn(
    r: torch.Tensor, c: torch.Tensor, ng: int, cost_type: str,
    *, variant: str = "alternating", backend=None,
    cost_axis: Optional[torch.Tensor] = None,
    cache: Optional[Dict[str, Any]] = None,
) -> Callable[[Dict[str, Any]], float]:
    """The Grid solver's ``gap_stop_fn``: ``feasible_dual_and_gap`` wrapped by
    ``make_gap_stop_fn``, with η read from ``info["eta"]``.

    ``cost_axis`` is the ``_cost_axis`` table of this (ng, cost_type); it is
    built here once, in fp64 on r's device, when not given.
    """
    if cost_axis is None:
        cost_axis = _cost_axis(cost_type, ng, device=r.device, dtype=torch.float64)

    def certify(info: Dict[str, Any], objective):
        return feasible_dual_and_gap(
            info["log_u"], info["log_v"], float(info["eta"]), r, c, ng, cost_type,
            objective, variant=variant, backend=backend, cost_axis=cost_axis)

    return make_gap_stop_fn(certify, cache=cache)


# ============================================================================
# Stored-C backend (General format): an in-memory m × n cost C, not assumed
# symmetric, so β' is a column minimum and α' a row minimum.
# ============================================================================

def _stored_blocks(C: torch.Tensor) -> Callable[[int, int], torch.Tensor]:
    """The stored C as a block source of row views (no copy)."""
    return lambda i0, i1: C[i0:i1, :]


def _stored_chunk(C: torch.Tensor, row_chunk: int = 0) -> int:
    """Rows per block for a stored C, those of a symbolic cost of the same
    shape; the last bits of r_D depend on this block size."""
    return _pointcloud_ct_chunk(C.shape[0], C.shape[1], row_chunk)


def _ctransform_storedC_colmin(p: torch.Tensor, C: torch.Tensor,
                               *, row_chunk: int = 0) -> torch.Tensor:
    """q_j = min_i (C_ij − p_i) over a stored C, so p = α gives β'."""
    return _ctransform_symbolic_colmin(p, _stored_blocks(C), C.shape[0], C.shape[1],
                                       row_chunk=_stored_chunk(C, row_chunk))


def _ctransform_storedC_rowmin(p: torch.Tensor, C: torch.Tensor,
                               *, row_chunk: int = 0) -> torch.Tensor:
    """q_i = min_j (C_ij − p_j) over a stored C, so p = β gives α'."""
    return _ctransform_symbolic_rowmin(p, _stored_blocks(C), C.shape[0], C.shape[1],
                                       row_chunk=_stored_chunk(C, row_chunk))


def dual_cleanup_ctransform_storedC(
    alpha: torch.Tensor, beta: torch.Tensor, C: torch.Tensor,
    *, variant: str = "alternating", row_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stored-C counterpart of ``dual_cleanup_ctransform``."""
    return _by_variant(
        alpha, beta, variant,
        lambda p: _ctransform_storedC_colmin(p, C, row_chunk=row_chunk),
        lambda p: _ctransform_storedC_rowmin(p, C, row_chunk=row_chunk))


def _dual_feas_F_storedC(alpha: torch.Tensor, beta: torch.Tensor, C: torch.Tensor,
                         *, row_chunk: int = 0) -> float:
    """r_D = ‖(α ⊕ β − C)₊‖_F over a stored C, with ``acc_like = C``."""
    return _dual_feas_F_symbolic(alpha, beta, _stored_blocks(C), C.shape[0], C.shape[1],
                                 row_chunk=_stored_chunk(C, row_chunk), out_like=C)


def feasible_dual_and_gap_storedC(
    log_u: torch.Tensor, log_v: torch.Tensor, eta: float,
    r: torch.Tensor, c: torch.Tensor, C: torch.Tensor,
    objective: Union[float, torch.Tensor],
    *, variant: str = "alternating", row_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """General-format certificate ``(α', β', g_feas)``.

    ``objective`` must be ⟨C,X⟩ for this same C: g_feas bounds ⟨C,X⟩ − OT*
    only for the cost the dual is feasible for, and an objective computed from
    any other array (for example a clamped log-kernel) gives no bound.
    """
    alpha = eta * log_u.reshape(-1)
    beta = eta * log_v.reshape(-1)
    ap, bp = dual_cleanup_ctransform_storedC(
        alpha, beta, C, variant=variant, row_chunk=row_chunk)
    return _certificate(ap, bp, r, c, objective)


def make_dense_gap_stop_fn(
    C: torch.Tensor, r: torch.Tensor, c: torch.Tensor,
    *, variant: str = "alternating", row_chunk: int = 0,
    cache: Optional[Dict[str, Any]] = None,
) -> Callable[[Dict[str, Any]], float]:
    """The General solvers' ``gap_stop_fn``: ``feasible_dual_and_gap_storedC``
    wrapped by ``make_gap_stop_fn``."""
    def certify(info: Dict[str, Any], objective):
        return feasible_dual_and_gap_storedC(
            info["log_u"], info["log_v"], float(info["eta"]), r, c, C,
            objective, variant=variant, row_chunk=row_chunk)

    return make_gap_stop_fn(certify, cache=cache)


# ============================================================================
# Symbolic backend (Point Cloud format): an injected
# ``cost_fn(i0, i1) -> C[i0:i1, :]`` that recomputes the cost κ from the point
# coordinates (``geo_cost.make_metric_cost_fn`` builds one).
# ============================================================================

def dual_cleanup_ctransform_symbolic(
    alpha: torch.Tensor, beta: torch.Tensor, cost_fn, m: int, n: int,
    *, variant: str = "alternating", row_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symbolic counterpart of ``dual_cleanup_ctransform_storedC`` (α of
    length m, β of length n)."""
    return _by_variant(
        alpha, beta, variant,
        lambda p: _ctransform_symbolic_colmin(p, cost_fn, m, n, row_chunk=row_chunk),
        lambda p: _ctransform_symbolic_rowmin(p, cost_fn, m, n, row_chunk=row_chunk))


def feasible_dual_and_gap_symbolic(
    log_u: torch.Tensor, log_v: torch.Tensor, eta: float,
    r: torch.Tensor, c: torch.Tensor, cost_fn,
    objective: Union[float, torch.Tensor],
    *, variant: str = "alternating", row_chunk: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Point Cloud certificate ``(α', β', g_feas)``; ``objective``
    must be ⟨C,X⟩ for the cost ``cost_fn`` returns."""
    alpha = eta * log_u.reshape(-1)
    beta = eta * log_v.reshape(-1)
    m = int(alpha.numel())
    n = int(beta.numel())
    ap, bp = dual_cleanup_ctransform_symbolic(
        alpha, beta, cost_fn, m, n, variant=variant, row_chunk=row_chunk)
    return _certificate(ap, bp, r, c, objective)


def make_pointcloud_gap_stop_fn(
    cost_fn, r: torch.Tensor, c: torch.Tensor,
    *, variant: str = "alternating", row_chunk: int = 0,
    cache: Optional[Dict[str, Any]] = None,
) -> Callable[[Dict[str, Any]], float]:
    """The Point Cloud solvers' ``gap_stop_fn``: ``feasible_dual_and_gap_symbolic``
    wrapped by ``make_gap_stop_fn``.

    ``cost_fn`` must be the cost the solver iterates on: a max-1 cost, which
    the solver's Normalize step leaves unchanged (the solver then returns
    ``info["cost_scale"] == 1``).
    """
    def certify(info: Dict[str, Any], objective):
        return feasible_dual_and_gap_symbolic(
            info["log_u"], info["log_v"], float(info["eta"]), r, c, cost_fn,
            objective, variant=variant, row_chunk=row_chunk)

    return make_gap_stop_fn(certify, cache=cache)


