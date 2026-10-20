"""CUDA driver of the Grid solver: the factored BDRS iteration on an n_g × n_g grid.

With flat indices a = (a_r, a_c) and b = (b_r, b_c) the iterate is kept as

    log X^k[a, b] = ξ[a] + k·ℓ[a_r, b_r] + k·ℓ[a_c, b_c] + ζ[b],

with ξ (`A`), ζ (`B`) and the per-axis log-kernel ℓ[i, j] = −φ(|i − j|)/(C_max·η)
(`base_ell` holds it by distance |i − j|), so only O(N) state exists. The
checks evaluate r_P, ⟨C, X⟩ and the native gap by separable log-sum-exps in
O(n_g³), and r_D in O(n_g⁴), without forming X.
"""

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from src.methods import cuda_ext
from src.methods.dual_cleanup import make_grid_gap_stop_fn
from src.methods.grid_cost import _base_ell, _phi_and_cmax
from src.methods.grid_factored import (
    _lse_streamed,
    factored_col_sums,
    factored_dual_sq_sum,
    factored_row_sums,
)
from src.methods.params import (
    BregmanDRParams,
    _check_marginals,
    _resolve_stop,
    floor_marginal,
)


# Must match the cost codes of build_base_ell in cuda/bdrs_solver.cpp.
_COST_TYPES: Dict[str, int] = {
    "l1":   0,
    "sql2": 1,
}


# The fused kernels launch n_g·W threads per block, with W = 1024/n_g at
# n_g ∈ {32, 64, 128} (W = 1 otherwise, or everywhere with BDRS_LSE_SPLIT=0),
# and CUDA allows at most 1024 threads per block. Their shared memory,
# (2·n_g + 2·n_g·W)·8 B, peaks at 32 KiB at n_g = 1024, within the default
# 48 KiB per block.
MAX_NG = 1024


def _check_invariants(grid_size: int, cost_type: str, eta: float,
                      device) -> Dict[str, Any]:
    """The fp64 tables of the checks: ℓ by distance (`base_ell`), the index
    table |i − j| (`D_idx`) and ψ[i, j] = φ(|i − j|)/C_max (`psi_mat`).

    The solver builds them once per solve and never caches them across
    solves, so each solve pays for its own."""
    ng = int(grid_size)
    dtype = torch.float64
    phi, c_max = _phi_and_cmax(cost_type, ng, dtype=dtype, device=device)
    base_ell = _base_ell(cost_type, ng, eta, dtype=dtype, device=device)
    idx = torch.arange(ng, device=device)
    D_idx = (idx[:, None] - idx[None, :]).abs()
    # Entrywise equal to grid_cost._cost_axis, from which the blocks of C are
    # built, so the separable ⟨C, X⟩ uses the same cost entries.
    psi_mat = (phi / c_max)[D_idx]
    return {"ng": ng, "base_ell": base_ell, "D_idx": D_idx, "psi_mat": psi_mat}


def _kl_mat(invariants: Dict[str, Any], k_iter: int) -> torch.Tensor:
    """The (n_g, n_g) table k·ℓ[i, j] of iteration k."""
    kL = float(k_iter) * invariants["base_ell"]
    return kL[invariants["D_idx"]]


def bregman_dr_solve_grid_cuda(
    C: Optional[torch.Tensor],        # ignored; positional parity with the General solvers
    r: torch.Tensor,                  # [N], fp64, the original marginal
    c: torch.Tensor,                  # [N], fp64, the original marginal
    params: BregmanDRParams,
    *,
    cost_type: str,
    grid_size: int,                   # n_g
    return_plan: bool = True,         # form the N×N plan X (8·N² bytes: 34 GB at n_g = 256)
    stop: str = "feasible_gap",       # or "kkt_triple" (params.STOP_RULES)
    traj_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    gap_stop_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    check_r: Optional[torch.Tensor] = None,
    check_c: Optional[torch.Tensor] = None,
    gap_stop_tol: Optional[float] = None,   # None: params.epsilon_primal
    init: str = "uniform",            # "uniform" (X⁰ = 1/N²) or "product" (X⁰ = r̃ c̃ᵀ)
    native_dual_residual: bool = False,
    _ext_override: Any = None,
    check_at: Optional[Sequence[int]] = None,   # explicit check iterations, in place of every T_chk
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    """Solve the Grid DOT problem with the BDRS CUDA session.

    The session runs batches of T_chk (`params.check_every`) iterations. After
    each batch the row-only primal proxy ‖X1 − r‖₂ is computed; when it is
    below `params.epsilon_primal` the test of ``stop`` runs:

      * ``"feasible_gap"`` (the default): the solve stops when
        ``gap_stop_fn(info) < gap_stop_tol``, with ``gap_stop_tol`` defaulting
        to `epsilon_primal`. Without ``gap_stop_fn`` the solver builds
        ``dual_cleanup.make_grid_gap_stop_fn(r, c, n_g, cost_type)``, whose
        value is |g_feas| of the alternating c-transform (α′, β′) of
        (η·log u, η·log v);
      * ``"kkt_triple"``: the solve stops when r_P, r_D and the native gap of
        (η·log u, η·log v) are below `epsilon_primal`, `epsilon_dual` and
        `epsilon_gap`; r_D costs O(n_g⁴) at each such check.

    With ``traj_callback`` (trajectory mode) the full check runs after every
    batch and the KKT-triple test applies, whatever ``stop`` is (tolerances
    of 0 disable it). If no stop fires, the solve runs K_max
    (`params.max_iters`) iterations.

    The iteration uses the marginals floored with δ_m (`params.eps_mass`);
    every gate and delivered metric uses the original (r, c), or
    ``check_r``/``check_c`` when given.

    ``gap_stop_fn`` receives ``info["objective"]`` as a 0-dim fp64 device
    tensor and reads back only its own value, one host sync per check. Its
    value is stored as ``info["gap_stop_value"]``.

    ``traj_callback`` receives the session's ``log_u`` and ``log_v``, which
    the next batch overwrites in place on the session's stream, and that
    stream does not wait for work queued on the caller's stream: a callback
    that keeps them must copy them and wait for the copy (e.g. ``.cpu()``).

    ``native_dual_residual`` computes r_D of the uncorrected (α, β) =
    (η·log u, η·log v) at delivery, an O(n_g⁴) diagnostic. Without it
    ``dual_res_F`` is NaN unless the final batch ran the full check.

    Returns ``(X, info)``, X None unless ``return_plan``. ``info["primal_res_L2"]``
    is the full r_P = ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂ (the proxy inside the loop),
    ``gap_abs`` the native gap of (α, β), and ``primal_res_proxy_final`` is
    inf when K_max = 0.
    """
    if cost_type not in _COST_TYPES:
        raise ValueError(
            f"Unknown cost_type: {cost_type!r}. Valid: {sorted(_COST_TYPES)}"
        )
    if grid_size < 4 or (grid_size & (grid_size - 1)) != 0:
        # The session's kernels are built for powers of two from 4 to MAX_NG.
        raise ValueError(f"grid_size must be a power of 2 and at least 4; got {grid_size}")
    if grid_size > MAX_NG:
        raise ValueError(
            f"grid_size={grid_size} exceeds MAX_NG={MAX_NG}. The fused kernels "
            f"use n_g*W threads per block; CUDA caps blocks at 1024 threads."
        )
    if _ext_override is None:
        if r.device.type != "cuda" or c.device.type != "cuda":
            raise ValueError("r and c must be CUDA tensors.")
    if r.dtype != torch.float64 or c.dtype != torch.float64:
        raise ValueError("r and c must be torch.float64.")
    N_expected = grid_size * grid_size
    if r.numel() != N_expected or c.numel() != N_expected:
        raise ValueError(
            f"marginals must have length ng*ng = {N_expected}; got r={r.numel()}, c={c.numel()}"
        )
    if bool((r < 0).any().item()) or bool((c < 0).any().item()):
        raise ValueError("r and c must be non-negative (they are probability vectors).")

    # The floored (r̃, c̃) drive the iteration only; every gate and delivered
    # metric uses (r_chk, c_chk), the originals by default.
    r_orig, c_orig = r, c
    r_it = floor_marginal(r_orig, params.eps_mass)
    c_it = floor_marginal(c_orig, params.eps_mass)
    r_chk, c_chk = _check_marginals(r_orig, c_orig, check_r, check_c)
    # log r̃ enters ξ = log r̃ − LSE(·) directly, so a zero entry would make
    # ξ = −∞. With δ_m > 0 every floored entry is positive.
    if not bool((r_it > 0).all().item()) or not bool((c_it > 0).all().item()):
        raise ValueError(
            "the floored iteration marginals contain a non-positive entry; "
            "params.eps_mass = 0 disables the marginal floor delta_m and the "
            "log-domain updates need strictly positive marginals (the default "
            "eps_mass = MARGINAL_EPS = 1e-12 floors them)."
        )

    build_default, gap_stop_tol = _resolve_stop(
        params, stop, gap_stop_fn, traj_callback, gap_stop_tol)
    if build_default:
        gap_stop_fn = make_grid_gap_stop_fn(r_chk, c_chk, int(grid_size), cost_type)

    # Test seam: a fake extension drives this loop on CPU.
    ext = _ext_override if _ext_override is not None else cuda_ext.load()

    device = r.device
    cost_type_int = _COST_TYPES[cost_type]
    eta_f = float(params.eta)
    max_iters_total = int(params.max_iters)
    check_every = max(1, int(params.check_every))

    inv: Optional[Dict[str, Any]] = None

    # The session captures the per-iteration CUDA graph once; step(n) replays it
    # n times on the session's stream and makes the caller's current stream
    # wait for those iterations, so the checks below read the new state.
    # "uniform" passes empty tensors, so the session starts from its default
    # ξ⁰ = ζ⁰ = −log N, log v⁰ = 0, i.e. X⁰ = 1/N².
    # "product" starts from X⁰ = r̃ c̃ᵀ: ξ⁰ = log r̃, ζ⁰ = log c̃, log v⁰ = 0.
    # The ξ-update reads the state only through β̃⁰ = ζ⁰ + log v⁰, so ξ⁰
    # enters only log u¹ = ξ¹ − ξ⁰.
    if init == "product":
        init_A = r_it.contiguous().log()
        init_B = c_it.contiguous().log()
        init_log_v = torch.zeros_like(init_A)
    elif init == "uniform":
        init_A = init_B = init_log_v = torch.empty(0, dtype=torch.float64, device=device)
    else:
        raise ValueError(f"init must be 'uniform' or 'product'; got {init!r}")
    sess = ext.BdrsSession(
        r=r_it.contiguous(), c=c_it.contiguous(),
        ng=int(grid_size), eta=eta_f, cost_type=cost_type_int,
        init_A=init_A, init_B=init_B, init_log_v=init_log_v,
    )
    # The kernels update these session tensors in place, so the references
    # always hold the current iterate.
    A_state: torch.Tensor = sess.A
    B_state: torch.Tensor = sess.B
    log_u_state: torch.Tensor = sess.log_u
    log_v_state: torch.Tensor = sess.log_v

    iters_run = 0
    info_out: Dict[str, Any] = {}
    native_termination = "max_iters"
    proxy_passes = 0
    full_checks  = 0     # evaluations of the native r_D (O(n_g⁴) each)
    sep_obj_checks = 0
    primal_proxy_value = float("inf")

    def _mk_info() -> Dict[str, Any]:
        return {
            "iters": iters_run,
            "eta": eta_f,
            "ng": int(grid_size),
            "N": N_expected,
            "cost_type": cost_type,
            "A": A_state,
            "B": B_state,
            "log_u": log_u_state,
            "log_v": log_v_state,
        }

    # The iterates do not depend on the check schedule; it decides only where
    # a stop can fire.
    check_iters = (sorted({int(k) for k in check_at if 0 < int(k) <= max_iters_total})
                   if check_at is not None else None)
    ci = 0
    while iters_run < max_iters_total:
        if check_iters is not None:
            while ci < len(check_iters) and check_iters[ci] <= iters_run:
                ci += 1
            nxt = check_iters[ci] if ci < len(check_iters) else max_iters_total
            n_step = nxt - iters_run
        else:
            n_step = min(check_every, max_iters_total - iters_run)

        iters_run = int(sess.step(int(n_step)))
        info_out = _mk_info()

        if inv is None:
            inv = _check_invariants(grid_size, cost_type, eta_f, device)
        primal_proxy_value, logS_row = _compute_factored_primal_res_ex(
            A_state, B_state, r_chk, c_chk,
            grid_size, cost_type, eta_f, iters_run,
            invariants=inv,
        )
        info_out["primal_res_L2"] = primal_proxy_value

        if traj_callback is not None:
            _add_factored_kkt(info_out, r_chk, c_chk, grid_size, cost_type, eta_f,
                              invariants=inv, native_dual=True)
            full_checks += 1
            traj_callback({
                "k": iters_run,
                "eta": eta_f,
                "primal_res_L2": info_out["primal_res_L2"],
                "dual_res_F": info_out["dual_res_F"],
                "gap_abs": info_out["gap_abs"],
                "objective": info_out["objective"],
                "primal_res_proxy": primal_proxy_value,
                "log_u": info_out["log_u"],
                "log_v": info_out["log_v"],
            })
            if (info_out["primal_res_L2"] < params.epsilon_primal
                and info_out["dual_res_F"] < params.epsilon_dual
                and info_out["gap_abs"]    < params.epsilon_gap):
                native_termination = "kkt_below_tol"
                break
        elif primal_proxy_value < params.epsilon_primal:
            proxy_passes += 1
            if gap_stop_fn is not None:
                # The hook needs only ⟨C, X⟩: the separable evaluation reuses
                # the proxy's table S and leaves the value on the device, so
                # the hook does the one host readback.
                obj_t = _compute_factored_objective_t(
                    A_state, B_state, grid_size, cost_type, eta_f, iters_run,
                    S=logS_row, invariants=inv,
                )
                info_out["objective"] = obj_t
                sep_obj_checks += 1
                gap_val = float(gap_stop_fn(info_out))
                info_out["gap_stop_value"] = gap_val
                if (info_out["primal_res_L2"] < params.epsilon_primal
                        and gap_val < gap_stop_tol):
                    native_termination = "feasible_gap_below_tol"
                    break
            else:
                # _add_factored_kkt replaces the proxy in primal_res_L2 with
                # the full r_P.
                _add_factored_kkt(info_out, r_chk, c_chk, grid_size, cost_type, eta_f,
                                  invariants=inv, native_dual=True)
                full_checks += 1
                if (info_out["primal_res_L2"] < params.epsilon_primal
                        and info_out["dual_res_F"] < params.epsilon_dual
                        and info_out["gap_abs"]    < params.epsilon_gap):
                    native_termination = "kkt_below_tol"
                    break

    if not info_out:
        info_out = _mk_info()

    if isinstance(info_out.get("objective"), torch.Tensor):
        info_out["objective"] = float(info_out["objective"].item())

    # Delivery: the full r_P, ⟨C, X⟩ and the native gap of the final iterate,
    # unless the final batch already ran _add_factored_kkt (which leaves
    # "dual_res_F"). An objective computed on this state for the hook is
    # reused.
    if info_out and "dual_res_F" not in info_out:
        if inv is None:
            inv = _check_invariants(grid_size, cost_type, eta_f, device)
        _add_factored_kkt(info_out, r_chk, c_chk, grid_size, cost_type, eta_f,
                          invariants=inv, native_dual=bool(native_dual_residual))
        if native_dual_residual:
            full_checks += 1

    info_out["native_termination"] = native_termination
    info_out["check_every"]        = check_every
    info_out["proxy_passes"]       = proxy_passes
    info_out["full_kkt_checks"]    = full_checks
    info_out["sep_obj_checks"]     = sep_obj_checks
    info_out["primal_res_proxy_final"] = primal_proxy_value
    info_out["marginal_floor"]     = float(params.eps_mass)
    info_out["iters"] = iters_run

    if return_plan and info_out:
        X = ext.materialize_X(
            A=A_state, B=B_state,
            ng=int(grid_size),
            eta=eta_f,
            cost_type=cost_type_int,
            k_iter=int(iters_run),
        )
    else:
        X = None
    return X, info_out


def _compute_factored_primal_res_ex(
    A: torch.Tensor,
    B: torch.Tensor,
    r: torch.Tensor,
    c: torch.Tensor,
    grid_size: int,
    cost_type: str,
    eta: float,
    k_iter: int,
    *,
    invariants: Optional[Dict[str, Any]] = None,
) -> Tuple[float, torch.Tensor]:
    """The row-only primal proxy ‖X^k 1 − r‖₂ as a float, and the first-pass table S.

    S[a_r, b_c] = LSE_{b_r}(k·ℓ[a_r, b_r] + ζ[b_r, b_c]) is reused by
    ``_compute_factored_objective_t``. ``c`` is not read: the v-update makes
    Xᵀ1 = c̃ up to rounding, so the omitted column term is ‖c̃ − c‖₂ =
    δ_m‖1 − N·c‖₂/(1 + N·δ_m) for a probability vector c.
    """
    ng = int(grid_size)
    N = ng * ng
    device = A.device

    A2 = A.view(ng, ng)
    B2 = B.view(ng, ng)

    if invariants is None:
        invariants = _check_invariants(grid_size, cost_type, eta, device)
    kL_mat = _kl_mat(invariants, k_iter)

    row_sums, logS_row = factored_row_sums(A2, B2, kL_mat, kL_mat)

    r_flat = r.reshape(N)
    primal_res = (row_sums - r_flat).norm(2)
    return float(primal_res.item()), logS_row


def _compute_factored_objective_t(
    A: torch.Tensor,
    B: torch.Tensor,
    grid_size: int,
    cost_type: str,
    eta: float,
    k_iter: int,
    *,
    S: Optional[torch.Tensor] = None,
    invariants: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """⟨C, X⟩ of the factored iterate as a 0-dim device tensor, in O(n_g³) with no host sync.

    The cost is axis-additive, C[a, b] = ψ[a_r, b_r] + ψ[a_c, b_c] with
    ψ[i, j] = φ(|i − j|)/C_max, so

        ⟨C, X⟩ = Σ_{a_r,b_r} ψ[a_r, b_r]·R_r[a_r, b_r] + Σ_{a_c,b_c} ψ[a_c, b_c]·R_c[a_c, b_c],

    where R_r[a_r, b_r] = Σ_{a_c,b_c} X and R_c[a_c, b_c] = Σ_{a_r,b_r} X are
    each computed by two log-sum-exp passes and then exponentiated. A linear-domain
    evaluation would lose mass: the most negative entry of k·ℓ is −k/(2η),
    below the fp64 underflow threshold of exp (about −745) once k > 1490·η,
    while ξ + ζ offsets it on the support of X. Each entry of R_r and R_c is
    at most Σc̃ = 1, so the final exp cannot overflow.

    ``S``: the proxy's table S; passing it skips one pass.
    """
    ng = int(grid_size)
    device = A.device

    A2 = A.reshape(ng, ng)
    B2 = B.reshape(ng, ng)

    if invariants is None:
        invariants = _check_invariants(grid_size, cost_type, eta, device)
    kL_mat = _kl_mat(invariants, k_iter)
    psi_mat = invariants["psi_mat"]

    # Each pass streams over an axis it does not reduce: a_r for S and T', a_c
    # for T and S'.
    #   S[a_r,b_c] = LSE_{b_r}( kL_mat[a_r,b_r] + B2[b_r,b_c] )
    if S is None:
        S = _lse_streamed(                                                  # (a_r, b_c)
            lambda i0, i1: kL_mat[i0:i1].unsqueeze(2) + B2.unsqueeze(0), 1, ng, ng * ng)
    #   T[a_c,b_c] = LSE_{a_r}( A2[a_r,a_c] + S[a_r,b_c] )
    T = _lse_streamed(                                                      # (a_c, b_c)
        lambda j0, j1: A2[:, j0:j1].unsqueeze(2) + S.unsqueeze(1), 0, ng, ng * ng)
    R_c = (kL_mat + T).exp()                                                # (a_c, b_c)
    del T

    #   S'[a_c,b_r] = LSE_{b_c}( kL_mat[a_c,b_c] + B2[b_r,b_c] )
    Sp = _lse_streamed(                                                     # (a_c, b_r)
        lambda j0, j1: kL_mat[j0:j1].unsqueeze(1) + B2.unsqueeze(0), 2, ng, ng * ng)
    #   T'[a_r,b_r] = LSE_{a_c}( A2[a_r,a_c] + S'[a_c,b_r] )
    Tp = _lse_streamed(                                                     # (a_r, b_r)
        lambda i0, i1: A2[i0:i1].unsqueeze(2) + Sp.unsqueeze(0), 1, ng, ng * ng)
    R_r = (kL_mat + Tp).exp()                                               # (a_r, b_r)

    return (psi_mat * R_r).sum() + (psi_mat * R_c).sum()


def _compute_factored_objective(
    A: torch.Tensor,
    B: torch.Tensor,
    grid_size: int,
    cost_type: str,
    eta: float,
    k_iter: int,
    *,
    S: Optional[torch.Tensor] = None,
    invariants: Optional[Dict[str, Any]] = None,
) -> float:
    """``_compute_factored_objective_t`` read back as a float."""
    return float(_compute_factored_objective_t(
        A, B, grid_size, cost_type, eta, k_iter,
        S=S, invariants=invariants).item())


def _add_factored_kkt(
    info: Dict[str, Any],
    r: torch.Tensor,
    c: torch.Tensor,
    grid_size: int,
    cost_type: str,
    eta: float,
    *,
    invariants: Optional[Dict[str, Any]] = None,
    native_dual: bool = False,
) -> None:
    """Write r_P, ⟨C, X⟩, the native gap and optionally r_D of the factored iterate into ``info``.

    Reads ξ, ζ, log u, log v and k (``A``, ``B``, ``log_u``, ``log_v``,
    ``iters``) from ``info``. An ``objective`` already in ``info`` (float or
    0-dim tensor) is taken to belong to this state and reused. ``gap_abs`` is
    |⟨C, X⟩ − η(rᵀ log u + cᵀ log v)|; ``dual_res_F`` is r_D with
    ``native_dual`` (an O(n_g⁴) pass), NaN otherwise.
    """
    ng = int(grid_size)
    N = ng * ng
    device = r.device
    dtype = torch.float64

    A = info["A"].view(ng, ng)
    B = info["B"].view(ng, ng)
    log_u = info["log_u"].view(ng, ng)
    log_v = info["log_v"].view(ng, ng)

    k_iter = int(info["iters"])
    if invariants is None:
        invariants = _check_invariants(grid_size, cost_type, eta, device)
    kL_mat = _kl_mat(invariants, k_iter)

    row_sums, _ = factored_row_sums(A, B, kL_mat, kL_mat)
    col_sums = factored_col_sums(A, B, kL_mat, kL_mat)

    r_flat = r.reshape(N)
    c_flat = c.reshape(N)
    primal_res = (row_sums - r_flat).norm(2) + (col_sums - c_flat).norm(2)

    # log K = L = −C/η, recovered as k·ℓ / k from the iteration-k table, or
    # from ℓ when k = 0.
    if native_dual:
        ell_mat = invariants["base_ell"][invariants["D_idx"]]

        def logK_block(a_r: int) -> torch.Tensor:          # (a_c, b_r, b_c)
            if k_iter > 0:
                return (kL_mat[a_r, None, :, None] + kL_mat[:, None, :]) / float(k_iter)
            return ell_mat[a_r, None, :, None] + ell_mat[:, None, :]

        dual_sq_sum = factored_dual_sq_sum(log_u, log_v, logK_block, ng,
                                           dtype=dtype, device=device)
        dual_res = float((eta * dual_sq_sum.sqrt()).item())
    else:
        dual_res = float("nan")

    obj_in = info.get("objective")
    if isinstance(obj_in, torch.Tensor):
        obj_val = float(obj_in.item())
    elif obj_in is not None:
        obj_val = float(obj_in)
    else:
        obj_val = _compute_factored_objective(
            info["A"], info["B"], grid_size, cost_type, eta, k_iter,
            invariants=invariants)

    log_u_flat = log_u.reshape(N)
    log_v_flat = log_v.reshape(N)
    dual_obj = eta * float((log_u_flat.dot(r_flat) + log_v_flat.dot(c_flat)).item())
    gap_abs = abs(obj_val - dual_obj)

    info["primal_res_L2"] = float(primal_res.item())
    info["dual_res_F"]    = dual_res
    info["gap_abs"]       = float(gap_abs)
    info["objective"]     = float(obj_val)
    info["native_dual_residual"] = bool(native_dual)
