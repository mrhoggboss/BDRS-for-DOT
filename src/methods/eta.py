"""The BDRS step size η = 0.7·mean(C), one rule for all three formats.

C is the max-1 cost, and mean(C) is its exact mean, never a sample.

The expressions in ``grid_cost_mean`` must keep their operation order. η
enters every iterate, and an algebraically equal rewrite (for example
0.7·(n_g+1)/(3 n_g) for l1) rounds differently for some n_g (n_g = 256 is
one), which changes η in the last bit and with it the whole iterate sequence.
"""


import numpy as np

from src.methods.grid_cmax import grid_cmax

ETA_ALPHA = 0.7


def eta_from_mean(mean_c) -> float:
    """η = ETA_ALPHA·mean(C) as a Python float; ``mean_c`` is a float or a 0-dim tensor."""
    return ETA_ALPHA * float(mean_c)


def grid_cost_mean(ng: int, norm: str) -> float:
    """Exact mean of the max-1 lattice cost C on an n_g × n_g grid.

    With d1, d2 independent, each distributed as |i − j| for i, j uniform on
    {0, …, n_g−1}:
      l1:   E[d1 + d2] / (2 (n_g−1))
      sql2: E[d1² + d2²] / (2 (n_g−1)²)
      l2:   E[√(d1² + d2²)] / (√2 (n_g−1))
      inf:  E[max(d1, d2)] / (n_g−1)
    l2 and inf are exact O(n_g²) sums over the joint law of (d1, d2). Raises
    ValueError for any other ``norm``, ``linf`` included (``grid_cmax``
    accepts it).
    """
    if norm in ("l1", "sql2"):
        n = int(ng)
        E_abs = (n * n - 1) / (3.0 * n)
        E_sq = (n * n - 1) / 6.0                # E[(i - j)²] = 2 Var(i)
        raw_mean = 2.0 * E_abs if norm == "l1" else 2.0 * E_sq
        return raw_mean / grid_cmax(norm, n)
    if norm == "l2":
        d, w = _abs_delta_weights(ng)
        M = np.sqrt(np.add.outer(d * d, d * d))
        p = np.outer(w, w) / (ng * ng) ** 2     # p[a, b] = P(d1 = a, d2 = b)
        return float((p * M).sum() / grid_cmax(norm, ng))
    if norm == "inf":
        d, w = _abs_delta_weights(ng)
        M = np.maximum.outer(d, d)
        p = np.outer(w, w) / (ng * ng) ** 2
        return float((p * M).sum() / grid_cmax(norm, ng))
    raise ValueError(f"no closed-form mean for norm={norm!r}; valid: l1, sql2, l2, inf")


def _abs_delta_weights(ng: int) -> "tuple[np.ndarray, np.ndarray]":
    """The values d = |i − j| = 0, …, n_g−1 over i, j in {0, …, n_g−1}, and
    their pair counts w, which sum to n_g² (both float64)."""
    d = np.arange(ng, dtype=np.float64)
    w = 2.0 * (ng - d)
    w[0] = ng
    return d, w
