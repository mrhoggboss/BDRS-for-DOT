"""Grid format: the torch reference solvers on square and rectangular lattices.

On an h x w lattice (n_g x n_g for the square solver) with row-major flat
indices a = (a_r, a_c), b = (b_r, b_c), the cost is separable,
C_ab = (φ_r(|a_r − b_r|) + φ_c(|a_c − b_c|)) / C_max with φ(δ) = δ for l1 and
δ² for sql2, and enters only through ``cost_type`` and the lattice size. The
iteration is that of the torch Point Cloud and General solvers
(``factored_torch``); separability changes only how each log-sum-exp is
evaluated. In the factored state log X^k_ab = ξ_a + k·L_ab + ζ_b (``A``, ``B``)
the kernel term splits as k·L_ab = kL_r[a_r, b_r] + kL_c[a_c, b_c] (one table,
``kL_mat``, for both axes on the square lattice), so each log-sum-exp over a
row or column of X is two one-dimensional log-sum-exps along the lattice axes:
O(n_g³) work and memory per iteration on the square lattice, and no N x N
array. Both solvers run exactly K_max iterations with no termination check;
``grid_cuda`` is the square-lattice solver with the stop.
"""

from typing import Dict, List, Optional, Tuple

import torch

from src.methods.grid_cost import _base_ell, _phi_and_cmax
from src.methods.grid_factored import (
    factored_col_sums,
    factored_dual_sq_sum,
    factored_row_sums,
)
from src.methods.params import (
    _as_f64,
    BregmanDRParams,
    _check_marginals,
    floor_marginal,
    _nan_t,
)


@torch.no_grad()
def bregman_dr_solve_factored_logdomain(
    r: torch.Tensor,                 # length N = n_g², original marginal
    c: torch.Tensor,                 # length N, original marginal
    params: BregmanDRParams,
    *,
    grid_size: int,                  # n_g
    cost_type: str,                  # "l1" or "sql2"
    return_plan: bool = True,
    check_r: Optional[torch.Tensor] = None,
    check_c: Optional[torch.Tensor] = None,
    native_dual_residual: bool = False,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Square-lattice BDRS in log domain, run for K_max (``params.max_iters``) iterations.

    The torch reference of the CUDA Grid solver, without its termination
    check: of ``params`` only ``eta``, ``max_iters`` and ``eps_mass`` are
    read. The Normalize step floors (r, c) with δ_m (``eps_mass``) into
    (r̃, c̃), which only the iteration uses; r_P and the native gap are
    measured against ``check_r``/``check_c``, default (r, c). Raises
    ValueError unless ``grid_size`` is a power of 2 (as ``grid_cuda``
    requires) and at least 2, both marginals have length N = n_g² and
    ``cost_type`` is l1 or sql2.

    Returns ``(X, info)``. X is the N x N plan, or None with
    ``return_plan=False``, in which case every metric comes from the
    factored state, so the solver also runs where X (32 GiB in fp64 at
    n_g = 256) does not fit. ``info``: ``iters``, ``primal_res_L2``
    (r_P = ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂), ``dual_res_F`` (the native r_D, an
    O(n_g⁴) pass computed only with ``native_dual_residual=True``, NaN
    otherwise), ``gap_abs`` (the native gap), ``objective`` (⟨C, X⟩),
    ``eta``, ``log_u`` and ``log_v`` (length N; the native duals are
    α = η·log u, β = η·log v), ``native_dual_residual`` and
    ``marginal_floor`` (δ_m).
    """
    ng = int(grid_size)
    if ng < 2 or (ng & (ng - 1)) != 0:
        # n_g = 1 has C_max = 0, so the cost cannot be normalized.
        raise ValueError(f"grid_size must be a power of 2 and at least 2; got {ng}")
    N = ng * ng

    r_orig = _as_f64(r)
    c_orig = _as_f64(c)
    if r_orig.numel() != N or c_orig.numel() != N:
        raise ValueError(f"marginals must have length n_g*n_g = {N}")
    r = floor_marginal(r_orig, params.eps_mass)
    c = floor_marginal(c_orig, params.eps_mass)
    r_chk, c_chk = _check_marginals(r_orig, c_orig, check_r, check_c)

    device = r.device
    dtype = torch.float64

    r2 = r.view(ng, ng)
    c2 = c.view(ng, ng)
    log_r = r2.log()
    log_c = c2.log()

    base_ell = _base_ell(cost_type, ng, params.eta, dtype=dtype, device=device)

    idx = torch.arange(ng, device=device)
    D_idx = (idx[:, None] - idx[None, :]).abs()

    # ξ^0 = ζ^0 = −log N: X^0 = 1/N² is the uniform plan.
    logN = float(torch.tensor(N, dtype=dtype).log())
    A = torch.full((ng, ng), -logN, dtype=dtype, device=device)
    B = torch.full((ng, ng), -logN, dtype=dtype, device=device)
    log_v = torch.zeros((ng, ng), dtype=dtype, device=device)
    log_u = torch.zeros_like(A)

    k = 0
    while k < params.max_iters:
        k += 1
        kL = k * base_ell
        kL_mat = kL[D_idx]

        # ξ^k = log r̃ − LSE_b(k·L_ab + β̃_b), β̃ = ζ^{k−1} + log v^{k−1}, in two
        # one-axis passes; log u^k = ξ^k − ξ^{k−1}.
        beta_tilde = B + log_v

        # Pass 1: logS[a_r, b_c] = LSE_{b_r}(kL_mat[a_r, b_r] + beta_tilde[b_r, b_c]),
        # the inner reduction over b_r as in the CUDA kernel fused_u.
        M_uR = kL_mat.unsqueeze(1) + beta_tilde.T.unsqueeze(0) # (a_r, b_c, b_r)
        logS = torch.logsumexp(M_uR, dim=2)
        del M_uR

        # Pass 2: A[a_r, a_c] = log_r - LSE_{b_c}(kL_mat[a_c, b_c] + logS[a_r, b_c])
        M_u2 = kL_mat.unsqueeze(0) + logS.unsqueeze(1)         # (a_r, a_c, b_c)
        lse_u = torch.logsumexp(M_u2, dim=2)
        del M_u2

        A_old = A
        A = log_r - lse_u
        log_u = A - A_old

        # ζ^k = log c̃ − LSE_a(k·L_ab + ξ^k_a), the mirror (nesting of fused_v);
        # log v^k = ζ^k − ζ^{k−1}.
        # Pass 1: logT[b_r, a_c] = LSE_{a_r}(kL_mat[a_r, b_r] + A[a_r, a_c])
        M_vR = kL_mat.T.unsqueeze(1) + A.T.unsqueeze(0)        # (b_r, a_c, a_r)
        logT = torch.logsumexp(M_vR, dim=2)
        del M_vR

        # Pass 2: B[b_r, b_c] = log_c - LSE_{a_c}(kL_mat[a_c, b_c] + logT[b_r, a_c])
        M_v2 = kL_mat.T.unsqueeze(0) + logT.unsqueeze(1)       # (b_r, b_c, a_c)
        lse_v = torch.logsumexp(M_v2, dim=2)
        del M_v2

        B_old = B
        B = log_c - lse_v
        log_v = B - B_old

    if k == 0:
        kL_mat = (k * base_ell)[D_idx]                 # max_iters <= 0: k·L = 0, X = X^0

    if return_plan:
        # log X in a_r slabs (rows a_r·n_g .. a_r·n_g + n_g − 1, N·n_g entries).
        # The concatenation holds the slab list and X at once, 2N² fp64 entries.
        log_X_rows: List[torch.Tensor] = []
        for a_r in range(ng):
            chunk = (
                A[a_r, :, None, None]              # (a_c, 1, 1)
                + kL_mat[a_r, None, :, None]       # (1, b_r, 1)
                + kL_mat[:, None, :]               # (a_c, 1, b_c)
                + B[None, :, :]                    # (1, b_r, b_c)
            )                                       # (a_c, b_r, b_c)
            log_X_rows.append(chunk.reshape(ng, N))
            del chunk
        X = torch.cat(log_X_rows, dim=0).view(N, N).exp_()
        del log_X_rows
        row_sums = X.sum(dim=1)
        col_sums = X.sum(dim=0)
    else:
        row_sums, _ = factored_row_sums(A, B, kL_mat, kL_mat)
        col_sums = factored_col_sums(A, B, kL_mat, kL_mat)
        X = None

    primal_res = (row_sums - r_chk).norm(2) + (col_sums - c_chk).norm(2)

    log_u_flat = log_u.reshape(N)
    log_v_flat = log_v.reshape(N)

    def _logK_block(a_r: int) -> torch.Tensor:
        # Block a_r of log K = L = −C/η, indexed (a_c, b_r, b_c), computed as
        # k·L / k from the iteration's tables, the expression grid_cuda uses
        # for its native r_D. At k = 0 those tables are zero, so ℓ
        # (``base_ell``) is used.
        if k > 0:
            return (kL_mat[a_r, None, :, None]        # (1, b_r, 1)
                    + kL_mat[:, None, :]) / float(k)  # (a_c, 1, b_c) → (a_c, b_r, b_c)
        ell_mat = base_ell[D_idx]
        return ell_mat[a_r, None, :, None] + ell_mat[:, None, :]

    # Native r_D = η·‖(log u ⊕ log v + log K)₊‖_F = ‖(α ⊕ β − C)₊‖_F. Opt-in
    # because it is a diagnostic: the dual a caller certifies is the
    # c-transformed pair (α′, β′) of dual_cleanup, feasible by construction.
    if native_dual_residual:
        dual_sq_sum = factored_dual_sq_sum(log_u, log_v, _logK_block, ng,
                                           dtype=dtype, device=device)
        dual_res = float(params.eta) * dual_sq_sum.sqrt()
    else:
        dual_res = _nan_t(device)

    # ⟨C, X⟩ from the factored state, with C = −η·log K.
    obj = torch.zeros((), dtype=dtype, device=device)
    for a_r in range(ng):
        logK_block = _logK_block(a_r)
        C_block = -float(params.eta) * logK_block
        logX_block = (
            A[a_r, :, None, None]
            + kL_mat[a_r, None, :, None]
            + kL_mat[:, None, :]
            + B[None, :, :]
        )
        X_block = logX_block.exp_()
        obj = obj + (C_block * X_block).sum()
        del logK_block, C_block, logX_block, X_block

    # Native gap: a diagnostic, since (η·log u, η·log v) need not be dual feasible.
    gap_abs = (obj - float(params.eta) * (log_u_flat.dot(r_chk) + log_v_flat.dot(c_chk))).abs()

    info = {
        "iters": torch.tensor(k, dtype=torch.int32),
        "primal_res_L2": primal_res,
        "gap_abs": gap_abs,
        "objective": obj,
        "eta": torch.tensor(params.eta, dtype=torch.float64),
        "dual_res_F": dual_res,
        "log_u": log_u_flat,
        "log_v": log_v_flat,
        "native_dual_residual": bool(native_dual_residual),
        "marginal_floor": float(params.eps_mass),
    }
    return X, info


@torch.no_grad()
def bregman_dr_solve_grid_rect_logdomain(
    r: torch.Tensor,                 # length N = h*w, row-major on the lattice
    c: torch.Tensor,                 # length N
    params: BregmanDRParams,
    *,
    grid_shape: Tuple[int, int],     # (h, w)
    cost_type,                       # "l1"/"sql2", or a (row, col) pair of them
    return_plan: bool = True,
    check_r: Optional[torch.Tensor] = None,
    check_c: Optional[torch.Tensor] = None,
    native_dual_residual: bool = False,
) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
    """Rectangular-lattice BDRS in log domain, run for K_max (``params.max_iters``) iterations.

    The square solver on an h x w lattice with any h, w >= 2 and axis costs
    φ_r, φ_c that may differ, normalized by C_max = φ_r(h − 1) + φ_c(w − 1)
    so that max C = 1. For h = w and one cost type every tensor operation
    coincides with the square solver's, so the results are bit-identical.
    The arguments, the Normalize step, the opt-in native r_D (O(N²) work)
    and ``(X, info)`` are as for the square solver. Raises ValueError for a side below 2, a
    marginal not of length h·w or an unknown cost type.
    """
    h, w = int(grid_shape[0]), int(grid_shape[1])
    if h < 2 or w < 2:
        raise ValueError(f"grid_shape sides must be >= 2; got {(h, w)}")
    N = h * w
    ct_r, ct_c = (cost_type, cost_type) if isinstance(cost_type, str) else cost_type

    r_orig = _as_f64(r)
    c_orig = _as_f64(c)
    if r_orig.numel() != N or c_orig.numel() != N:
        raise ValueError(f"marginals must have length h*w = {N}")
    r = floor_marginal(r_orig, params.eps_mass)
    c = floor_marginal(c_orig, params.eps_mass)
    r_chk, c_chk = _check_marginals(r_orig, c_orig, check_r, check_c)

    device = r.device
    dtype = torch.float64

    r2 = r.view(h, w)
    c2 = c.view(h, w)
    log_r = r2.log()
    log_c = c2.log()

    # Only φ is taken: the C_max _phi_and_cmax returns is that of a square lattice.
    phi_r = _phi_and_cmax(ct_r, h, dtype=dtype, device=device)[0]
    phi_c = _phi_and_cmax(ct_c, w, dtype=dtype, device=device)[0]
    c_max = float(phi_r[-1] + phi_c[-1])
    ell_r = -phi_r / (c_max * params.eta)
    ell_c = -phi_c / (c_max * params.eta)

    idx_r = torch.arange(h, device=device)
    idx_c = torch.arange(w, device=device)
    D_r = (idx_r[:, None] - idx_r[None, :]).abs()
    D_c = (idx_c[:, None] - idx_c[None, :]).abs()

    # ξ^0 = ζ^0 = −log N: X^0 is the uniform plan.
    logN = float(torch.tensor(N, dtype=dtype).log())
    A = torch.full((h, w), -logN, dtype=dtype, device=device)
    B = torch.full((h, w), -logN, dtype=dtype, device=device)
    log_v = torch.zeros((h, w), dtype=dtype, device=device)
    log_u = torch.zeros_like(A)

    k = 0
    while k < params.max_iters:
        k += 1
        kL_r = (k * ell_r)[D_r]
        kL_c = (k * ell_c)[D_c]

        # The square solver's ξ- and ζ-updates with kL_r on the row axis and
        # kL_c on the column axis, in its nesting (inner reduction over the row
        # axis), so that h = w with one cost reproduces it bit for bit.
        beta_tilde = B + log_v

        # Pass 1: logS[a_r, b_c] = LSE_{b_r}(kL_r[a_r, b_r] + beta_tilde[b_r, b_c])
        M_uR = kL_r.unsqueeze(1) + beta_tilde.T.unsqueeze(0)   # (a_r, b_c, b_r)
        logS = torch.logsumexp(M_uR, dim=2)
        del M_uR

        # Pass 2: A[a_r, a_c] = log_r - LSE_{b_c}(kL_c[a_c, b_c] + logS[a_r, b_c])
        M_u2 = kL_c.unsqueeze(0) + logS.unsqueeze(1)           # (a_r, a_c, b_c)
        lse_u = torch.logsumexp(M_u2, dim=2)
        del M_u2

        A_old = A
        A = log_r - lse_u
        log_u = A - A_old

        # Pass 1: logT[b_r, a_c] = LSE_{a_r}(kL_r[a_r, b_r] + A[a_r, a_c])
        M_vR = kL_r.T.unsqueeze(1) + A.T.unsqueeze(0)          # (b_r, a_c, a_r)
        logT = torch.logsumexp(M_vR, dim=2)
        del M_vR

        # Pass 2: B[b_r, b_c] = log_c - LSE_{a_c}(kL_c[a_c, b_c] + logT[b_r, a_c])
        M_v2 = kL_c.T.unsqueeze(0) + logT.unsqueeze(1)         # (b_r, b_c, a_c)
        lse_v = torch.logsumexp(M_v2, dim=2)
        del M_v2

        B_old = B
        B = log_c - lse_v
        log_v = B - B_old

    if k == 0:
        kL_r = (k * ell_r)[D_r]                        # max_iters <= 0: k·L = 0, X = X^0
        kL_c = (k * ell_c)[D_c]

    if return_plan:
        # log X in a_r slabs of N·w entries; as in the square solver, the
        # concatenation holds 2N² fp64 entries.
        log_X_rows: List[torch.Tensor] = []
        for a_r in range(h):
            chunk = (
                A[a_r, :, None, None]              # (a_c, 1, 1)
                + kL_r[a_r, None, :, None]         # (1, b_r, 1)
                + kL_c[:, None, :]                 # (a_c, 1, b_c)
                + B[None, :, :]                    # (1, b_r, b_c)
            )                                       # (a_c, b_r, b_c)
            log_X_rows.append(chunk.reshape(w, N))
            del chunk
        X = torch.cat(log_X_rows, dim=0).view(N, N).exp_()
        del log_X_rows
        row_sums = X.sum(dim=1)
        col_sums = X.sum(dim=0)
    else:
        row_sums, _ = factored_row_sums(A, B, kL_r, kL_c)
        col_sums = factored_col_sums(A, B, kL_r, kL_c)
        X = None

    primal_res = (row_sums - r_chk).norm(2) + (col_sums - c_chk).norm(2)

    log_u_flat = log_u.reshape(N)
    log_v_flat = log_v.reshape(N)

    def _logK_block(a_r: int) -> torch.Tensor:
        # Block a_r of log K = L = −C/η, indexed (a_c, b_r, b_c), by the
        # square solver's expression.
        if k > 0:
            return (kL_r[a_r, None, :, None]          # (1, b_r, 1)
                    + kL_c[:, None, :]) / float(k)    # (a_c, 1, b_c) → (a_c, b_r, b_c)
        return ell_r[D_r][a_r, None, :, None] + ell_c[D_c][:, None, :]

    if native_dual_residual:
        dual_sq_sum = factored_dual_sq_sum(log_u, log_v, _logK_block, h,
                                           dtype=dtype, device=device)
        dual_res = float(params.eta) * dual_sq_sum.sqrt()
    else:
        dual_res = _nan_t(device)

    obj = torch.zeros((), dtype=dtype, device=device)
    for a_r in range(h):
        logK_block = _logK_block(a_r)
        C_block = -float(params.eta) * logK_block
        logX_block = (
            A[a_r, :, None, None]
            + kL_r[a_r, None, :, None]
            + kL_c[:, None, :]
            + B[None, :, :]
        )
        obj = obj + (C_block * logX_block.exp_()).sum()
        del logK_block, C_block, logX_block

    gap_abs = (obj - float(params.eta) * (log_u_flat.dot(r_chk) + log_v_flat.dot(c_chk))).abs()

    info = {
        "iters": torch.tensor(k, dtype=torch.int32),
        "primal_res_L2": primal_res,
        "gap_abs": gap_abs,
        "objective": obj,
        "eta": torch.tensor(params.eta, dtype=torch.float64),
        "dual_res_F": dual_res,
        "log_u": log_u_flat,
        "log_v": log_v_flat,
        "native_dual_residual": bool(native_dual_residual),
        "marginal_floor": float(params.eps_mass),
    }
    return X, info
