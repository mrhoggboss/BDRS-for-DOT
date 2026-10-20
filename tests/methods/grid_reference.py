"""An independent reference for the DOTmark lattice cost, built from grid
coordinates with numpy and deliberately NOT from ``src``, so a test that solves
on it does not share the cost construction of the code under test.
"""

import numpy as np
import torch


def grid_cost(ng: int, norm: str) -> torch.Tensor:
    """The dense max-1 ``(ng², ng²)`` fp64 lattice cost, ``sql2`` or else ``l1``.

    The sum of the two axis terms is divided once, so entries can differ in
    the last bit from ``src.methods.grid_cost``, which divides each axis term."""
    N = ng * ng
    idx = np.arange(N)
    dr = idx[:, None] // ng - idx[None, :] // ng
    dc = idx[:, None] % ng - idx[None, :] % ng
    if norm == "sql2":
        C = (dr ** 2 + dc ** 2) / (2.0 * (ng - 1) ** 2)
    else:  # l1
        C = (np.abs(dr) + np.abs(dc)) / (2.0 * (ng - 1))
    return torch.from_numpy(C.astype(np.float64))
