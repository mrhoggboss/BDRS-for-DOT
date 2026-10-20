"""One iteration of the Grid CUDA solver against a dense torch reference.

The ξ-update kernel (``fused_u_update_t`` in bdrs_kernels.cu) evaluates
LSE_b(k·L_ab + β̃_b) as two per-axis online log-sum-exps over the separable
lattice cost; the reference is one ``torch.logsumexp`` over the dense N×N
log-kernel. ξ (``A``) and log u are compared after iteration 1.

Run on a CUDA machine:
    python -m tests.methods.test_lse
"""

import math
import sys

import numpy as np
import torch

from src.methods.params import floor_marginal
from tests.helpers import requires_cuda


# The reference floors r as the solver does, so the errors measured are the
# kernel's own: two per-axis online LSEs against one dense torch.logsumexp.
RTOL = 1e-12


def build_L_matrix(ng: int, eta: float, cost_type: str) -> torch.Tensor:
    """The dense N×N log-kernel L = −C/η of the max-1 ``l1`` or ``sql2``
    lattice cost, on the CPU. The session's per-axis table ℓ
    (``build_base_ell`` in bdrs_solver.cpp) gives L_ab = ℓ[|Δrow|] + ℓ[|Δcol|],
    equal to this up to rounding.
    """
    N = ng * ng
    ar = torch.arange(N, dtype=torch.int64) // ng
    ac = torch.arange(N, dtype=torch.int64) %  ng
    dr = (ar[:, None] - ar[None, :]).abs().to(torch.float64)
    dc = (ac[:, None] - ac[None, :]).abs().to(torch.float64)
    if cost_type == "sql2":
        c_max = 2.0 * (ng - 1) ** 2
        L = -((dr * dr + dc * dc) / (c_max * eta))
    elif cost_type == "l1":
        c_max = 2.0 * (ng - 1)
        L = -((dr + dc) / (c_max * eta))
    else:
        raise ValueError(cost_type)
    return L


def run():
    if not torch.cuda.is_available():
        print("[test_lse] SKIP: CUDA not available")
        return 0

    device = torch.device("cuda")
    torch.manual_seed(0)

    ng = 8
    N = ng * ng
    eta = 0.3
    cost_type = "sql2"

    r_cpu = torch.tensor(
        np.random.default_rng(42).dirichlet([1.0] * N), dtype=torch.float64,
    )
    c_cpu = torch.tensor(
        np.random.default_rng(43).dirichlet([1.0] * N), dtype=torch.float64,
    )
    r = r_cpu.to(device)
    c = c_cpu.to(device)

    L_cpu = build_L_matrix(ng, eta, cost_type)

    # The solver's default start (init="uniform"): ξ⁰ = ζ⁰ = −log N, log v⁰ = 0.
    A0 = torch.full((N,), -math.log(N), dtype=torch.float64)
    B0 = torch.full((N,), -math.log(N), dtype=torch.float64)
    log_v0 = torch.zeros(N, dtype=torch.float64)

    log_r_cpu = torch.log(floor_marginal(r_cpu))

    # ξ¹_a = log r_a − LSE_b(L_ab + β̃⁰_b), β̃⁰ = ζ⁰ + log v⁰; log u¹ = ξ¹ − ξ⁰.
    beta_tilde = B0 + log_v0
    lse = torch.logsumexp(L_cpu + beta_tilde[None, :], dim=1)  # (N,)
    A1_ref = log_r_cpu - lse
    log_u_ref = A1_ref - A0

    from src.methods.params import BregmanDRParams
    from src.methods.grid_cuda import bregman_dr_solve_grid_cuda

    params = BregmanDRParams(
        eta=eta,
        max_iters=1,
        check_every=1,
        epsilon_primal=0.0,
        epsilon_dual=0.0,
        epsilon_gap=0.0,
    )
    X, info = bregman_dr_solve_grid_cuda(
        None, r, c, params,
        cost_type=cost_type, grid_size=ng,
    )

    A1_cuda = info["A"].detach().cpu()
    log_u_cuda = info["log_u"].detach().cpu()

    rtol = RTOL
    diff_A = (A1_cuda - A1_ref).abs()
    diff_u = (log_u_cuda - log_u_ref).abs()
    max_rel_A = float(diff_A.max() / (A1_ref.abs().max() + 1e-300))
    max_rel_u = float(diff_u.max() / (log_u_ref.abs().max() + 1e-300))

    print(f"[test_lse] n_g={ng} N={N} cost={cost_type} eta={eta}")
    print(f"  max |A_cuda - A_ref|       = {diff_A.max().item():.3e}")
    print(f"  max |log_u_cuda - log_u_ref| = {diff_u.max().item():.3e}")
    print(f"  max relative error (A)     = {max_rel_A:.3e}")
    print(f"  max relative error (log_u) = {max_rel_u:.3e}")

    ok = (max_rel_A < rtol) and (max_rel_u < rtol)
    if ok:
        print("[test_lse] PASS")
        return 0
    else:
        print(f"[test_lse] FAIL (rtol = {rtol})")
        return 1


@requires_cuda
def test_lse_matches_two_pass_reference():
    """The pytest entry of ``run()``, whose printed errors appear in the
    captured output on failure."""
    assert run() == 0, f"online-LSE vs two-pass reference exceeded rtol={RTOL}"


if __name__ == "__main__":
    sys.exit(run())
