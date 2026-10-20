"""Point Cloud ground costs κ, evaluated block by block from the support coordinates.

A cost callable ``cost_fn(i0, i1)`` returns the fp64 block κ(P[i0:i1], Q) / C_max
of shape (i1 − i0, n), recomputed from the coordinates of P (m × d) and Q (n × d)
on every call, so the Point Cloud solver and its c-transform never hold an m × n
cost matrix.
"""

from typing import Callable, Optional

import torch


CostFn = Callable[[int, int], torch.Tensor]

_VALID = ("l1", "sql2", "l2", "linf", "inf")


def _metric_block(P_blk: torch.Tensor, Q: torch.Tensor, cost_type: str) -> torch.Tensor:
    """Raw cost κ between the rows of ``P_blk`` (b, d) and ``Q`` (n, d), as a (b, n) tensor."""
    diff = P_blk[:, None, :] - Q[None, :, :]          # [b, n, d]
    if cost_type == "l1":
        return diff.abs().sum(dim=2)
    if cost_type == "sql2":
        return (diff * diff).sum(dim=2)
    if cost_type == "l2":
        return torch.sqrt((diff * diff).sum(dim=2))
    if cost_type in ("linf", "inf"):
        return diff.abs().amax(dim=2)
    raise ValueError(f"unknown cost_type {cost_type!r}; valid: {_VALID}")


def _max_row_chunk(m: int, n: int, row_chunk: int = 0) -> int:
    """Source rows per block of the exact-max passes: ``row_chunk`` if positive,
    else at most 2^23 entries (64 MiB in fp64) per cost block but at least one
    row. The (chunk, n, d) difference temporary is d times larger."""
    if row_chunk > 0:
        return min(int(row_chunk), m)
    return max(1, min(m, (8 * 1024 * 1024) // max(1, n)))


def exact_cost_fn_max(cost_fn: CostFn, m: int, *, row_chunk: int = 0) -> float:
    """Attained maximum of ``cost_fn`` over all ``m`` source rows.

    For ``cost_fn = make_metric_cost_fn(P, Q, ct, c_max)`` it equals
    ``exact_c_max(P, Q, ct) / c_max`` exactly: rounded division by a positive
    constant is monotone, so it commutes with the maximum."""
    m = int(m)
    if m <= 0:
        raise ValueError("exact_cost_fn_max needs m >= 1")
    # Row 0 alone gives n, the dtype and the device; the sweep starts at row 1.
    head = cost_fn(0, 1)
    n = int(head.shape[1])
    chunk = _max_row_chunk(m, n, row_chunk)
    best = head.max()
    del head
    for i0 in range(1, m, chunk):
        i1 = min(m, i0 + chunk)
        best = torch.maximum(best, cost_fn(i0, i1).max())
    return float(best.item())


def exact_c_max(P: torch.Tensor, Q: torch.Tensor, cost_type: str,
                row_chunk: int = 0) -> float:
    """Attained maximum C_max = max_ij κ(p_i, q_j) of the raw cost, not a bounding-box bound.

    One tiled fp64 pass on the clouds' device, O(mnd) time."""
    if cost_type not in _VALID:
        raise ValueError(f"unknown cost_type {cost_type!r}; valid: {_VALID}")
    P = P.to(torch.float64)
    Q = Q.to(torch.float64)
    m = int(P.shape[0])
    n = int(Q.shape[0])
    if m <= 0 or n <= 0:
        raise ValueError("exact_c_max needs non-empty clouds")
    chunk = _max_row_chunk(m, n, row_chunk)
    best = None
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        blk_max = _metric_block(P[i0:i1], Q, cost_type).max()
        best = blk_max if best is None else torch.maximum(best, blk_max)
    return float(best.item())


def make_metric_cost_fn(
    P: torch.Tensor, Q: torch.Tensor, cost_type: str,
    c_max: Optional[float] = 1.0,
) -> CostFn:
    """Tiled cost ``cost_fn(i0, i1) = κ(P[i0:i1], Q) / c_max`` in fp64.

    ``c_max=None`` divides by ``exact_c_max(P, Q, cost_type)``. On
    ``grid_coords(n_g)`` with ``c_max = grid_cmax(cost_type, n_g)`` the blocks
    equal ``instances.dense_cost.lattice_cost_rows`` entry for entry: the
    offsets are integers, so the only roundings are the final division and the
    l2 square root."""
    if cost_type not in _VALID:
        raise ValueError(f"unknown cost_type {cost_type!r}; valid: {_VALID}")
    if c_max is None:
        c_max = exact_c_max(P, Q, cost_type)
    P = P.to(torch.float64)
    Q = Q.to(torch.float64)
    cm = float(c_max)

    def cost_fn(i0: int, i1: int) -> torch.Tensor:
        return _metric_block(P[i0:i1], Q, cost_type) / cm

    return cost_fn


def grid_coords(ng: int, *, device, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """The n_g × n_g grid as an (N, 2) point cloud in the flat-index order of
    ``grid_cost._build_grid_indices``."""
    idx = torch.arange(ng * ng, device=device)
    rows = (idx // ng).to(dtype)
    cols = (idx % ng).to(dtype)
    return torch.stack([rows, cols], dim=1)


def materialize_cost(cost_fn: CostFn, m: int, n: int, *, device, dtype=torch.float64,
                     row_chunk: int = 0) -> torch.Tensor:
    """Dense (m, n) cost matrix assembled from the blocks of ``cost_fn``."""
    C = torch.empty((m, n), device=device, dtype=dtype)
    chunk = row_chunk if row_chunk > 0 else max(1, min(m, (2 * 1024 * 1024) // max(1, n)))
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        C[i0:i1] = cost_fn(i0, i1)
    return C
