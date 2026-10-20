"""C_max, the largest raw ground cost on an n_g × n_g grid, the divisor that normalizes C to max 1.

The raw costs of an integer offset (Δrow, Δcol) are l1: Δrow + Δcol,
sql2: Δrow² + Δcol², l2: √(Δrow² + Δcol²) (unsquared), inf/linf:
max(Δrow, Δcol). Each is nondecreasing in both offsets, so the maximum is
attained between opposite corners, Δrow = Δcol = n_g − 1. For l2 the rounded
√2·(n_g − 1) can exceed the attained fp64 maximum, so the normalized maximum
can fall short of 1 (among powers of two up to 1024: by 2^-53 at n_g = 32
and 64, by 2^-52 at n_g = 4).

The module must not import torch: the OTT-JAX runner, which never loads
torch, imports it.
"""

import math

GRID_COST_TYPES = ("l1", "sql2", "l2", "linf", "inf")


def grid_cmax(cost_type: str, ng: int) -> float:
    """Return C_max of ``cost_type`` on an n_g × n_g grid (n_g = ``ng``)."""
    if ng < 2:
        raise ValueError(f"a grid needs at least 2 points per side; got ng={ng}")
    if cost_type == "l1":
        return 2.0 * (ng - 1)
    if cost_type == "sql2":
        return 2.0 * (ng - 1) ** 2
    if cost_type == "l2":
        return math.sqrt(2.0) * (ng - 1)
    if cost_type in ("linf", "inf"):
        return float(ng - 1)
    raise ValueError(f"unknown cost_type {cost_type!r}; valid: {GRID_COST_TYPES}")
