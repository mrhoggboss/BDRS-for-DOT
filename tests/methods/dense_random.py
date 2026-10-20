"""Random rectangular discrete-OT instances (C, r, c) with no grid structure,
so the row count m and the column count n are independent.

The draw uses a CPU ``torch.Generator`` and is then moved to ``device``, so a
given ``(m, n, seed, dist)`` is the same instance on CPU and GPU.
"""

from typing import Tuple

import torch

# Keeps the marginals strictly positive.
MARGINAL_EPS = 1e-12

_VALID_DIST = ("uniform", "exp", "absnormal")


def _sample_nonneg(shape, dist: str, gen: torch.Generator) -> torch.Tensor:
    if dist == "uniform":
        return torch.rand(shape, generator=gen, dtype=torch.float64)
    if dist == "exp":
        # Exponential(1) by inverse CDF; the clamp avoids log 0.
        u = torch.rand(shape, generator=gen, dtype=torch.float64).clamp_min_(1e-300)
        return -torch.log(u)
    if dist == "absnormal":
        return torch.randn(shape, generator=gen, dtype=torch.float64).abs_()
    raise ValueError(f"unknown dist {dist!r}; valid: {_VALID_DIST}")


def make_random_dense_dot(
    m: int,
    n: int,
    *,
    seed: int = 0,
    dist: str = "uniform",
    device="cpu",
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(C, r, c)``: C (m x n) has iid ``dist`` entries scaled to max 1;
    r and c are uniform draws plus ``MARGINAL_EPS``, rescaled to unit mass."""
    if m <= 0 or n <= 0:
        raise ValueError(f"m, n must be positive; got m={m}, n={n}")
    if dist not in _VALID_DIST:
        raise ValueError(f"unknown dist {dist!r}; valid: {_VALID_DIST}")

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    C = _sample_nonneg((m, n), dist, gen)
    cmax = C.max()
    if float(cmax) > 0.0:
        C = C / cmax

    r = torch.rand(m, generator=gen, dtype=torch.float64).add_(MARGINAL_EPS)
    r = r / r.sum()
    c = torch.rand(n, generator=gen, dtype=torch.float64).add_(MARGINAL_EPS)
    c = c / c.sum()

    C = C.to(device=device, dtype=dtype).contiguous()
    r = r.to(device=device, dtype=dtype).contiguous()
    c = c.to(device=device, dtype=dtype).contiguous()
    return C, r, c
