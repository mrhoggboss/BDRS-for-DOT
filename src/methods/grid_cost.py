"""The separable Grid costs (l1, sql2), assembled one block of rows at a time.

With the row-major flat index a = a_r·n_g + a_c,

    C[a, b] = T[a_r, b_r] + T[a_c, b_c],    T[i, j] = φ(|i − j|) / C_max,

where φ(d) = d for l1 and d² for sql2, so any block of rows of C is built from
the (n_g, n_g) axis table T without forming the N × N matrix. Each axis term
is divided by C_max before the two are added; ``instances.dense_cost``
divides the sum, so the two can differ in the last bit of an entry.
"""

from typing import Callable, Tuple

import torch

from src.methods.grid_cmax import grid_cmax


def _phi_and_cmax(cost_type: str, ng: int, *, dtype, device) -> Tuple[torch.Tensor, float]:
    """Return the axis table φ(d), d = 0..n_g − 1, and C_max."""
    delta = torch.arange(ng, dtype=dtype, device=device)
    if cost_type == "sql2":
        return delta * delta, grid_cmax(cost_type, ng)
    if cost_type == "l1":
        return delta, grid_cmax(cost_type, ng)
    raise ValueError(
        f"unknown cost_type {cost_type!r}; separable grid costs: 'l1', 'sql2'")


def _base_ell(cost_type: str, ng: int, eta: float, *, dtype, device) -> torch.Tensor:
    """Return the log-kernel axis table ℓ(d) = −φ(d) / (C_max·η).

    L = −C/η splits as L[a, b] = ℓ(|a_r − b_r|) + ℓ(|a_c − b_c|). The CUDA
    extension builds the same table on the host (``build_base_ell`` in
    ``cuda/bdrs_solver.cpp``)."""
    phi, c_max = _phi_and_cmax(cost_type, ng, dtype=dtype, device=device)
    return -phi / (c_max * eta)


def _row_chunk_for(N: int, *, row_chunk: int = 0) -> int:
    """Rows per block of an N-column matrix: ``row_chunk`` if positive, else at
    most 2^25 entries (256 MiB in fp64) per block."""
    if row_chunk > 0:
        return int(row_chunk)
    return max(1, min(N, (32 * 1024 * 1024) // max(1, N)))


def _cost_row_chunk(
    a_r: torch.Tensor, a_c: torch.Tensor,
    a_r_full: torch.Tensor, a_c_full: torch.Tensor,
    cost_axis: torch.Tensor,
) -> torch.Tensor:
    """Return the (b, N) rows of C whose flat indices have coordinates (``a_r``, ``a_c``).

    The column coordinates (``_build_grid_indices``) and T (``_cost_axis``)
    are arguments so a caller builds them once for all blocks. The peak
    memory is three (b, N) arrays."""
    return cost_axis[a_r][:, a_r_full] + cost_axis[a_c][:, a_c_full]


def _build_grid_indices(ng: int, *, device):
    """Return (a // n_g, a % n_g) for every flat index a, as two int64 tensors of length N."""
    N = ng * ng
    idx_full = torch.arange(N, device=device)
    a_r_full = idx_full // ng
    a_c_full = idx_full % ng
    return a_r_full, a_c_full


def _cost_axis(cost_type: str, ng: int, *, device, dtype) -> torch.Tensor:
    """Return the (n_g, n_g) axis table T[i, j] = φ(|i − j|) / C_max."""
    phi, c_max = _phi_and_cmax(cost_type, ng, dtype=dtype, device=device)
    idx = torch.arange(ng, device=device)
    D = (idx[:, None] - idx[None, :]).abs()
    return phi[D] / c_max


def _grid_blocks(cost_axis: torch.Tensor, ng: int) -> Callable[[int, int], torch.Tensor]:
    """Return C as a block source ``(i0, i1) -> C[i0:i1, :]`` built from T = ``cost_axis``."""
    a_r_full, a_c_full = _build_grid_indices(ng, device=cost_axis.device)
    return lambda i0, i1: _cost_row_chunk(a_r_full[i0:i1], a_c_full[i0:i1],
                                          a_r_full, a_c_full, cost_axis)
