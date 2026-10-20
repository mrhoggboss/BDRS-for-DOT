"""The BDRS log-domain recursion on a dense m x n array, with no chunking,
kernels or factored state. The General and Point Cloud solvers agree with it
up to the reassociation of the LSE sums.
"""

import torch

from src.methods.params import MARGINAL_EPS, floor_marginal


def naive_logdomain(C: torch.Tensor, r: torch.Tensor, c: torch.Tensor,
                    eta: float, iters: int):
    """``iters`` BDRS iterations on the dense cost C (m x n), which is not
    divided by its maximum.

    The iteration runs on the marginals floored by ``floor_marginal`` (δ_m), as
    in the solvers. Returns (X, log_u, log_v)."""
    r = floor_marginal(r, MARGINAL_EPS)
    c = floor_marginal(c, MARGINAL_EPS)
    logK = -C / eta
    m, n = C.shape
    buf = torch.full(
        (m, n), -float(torch.tensor(m * n, dtype=torch.float64).log()),
        dtype=torch.float64)
    log_v = torch.zeros(n, dtype=torch.float64)
    log_r = r.log()
    log_c = c.log()
    log_u = torch.zeros(m, dtype=torch.float64)
    for _ in range(iters):
        buf.add_(logK)
        log_u = log_r - torch.logsumexp(buf + log_v[None, :], dim=1)
        log_v = log_c - torch.logsumexp(buf + log_u[:, None], dim=0)
        buf.add_(log_u[:, None])
        buf.add_(log_v[None, :])
    return buf.exp(), log_u, log_v
