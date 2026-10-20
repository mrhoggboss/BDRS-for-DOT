"""Row sums, column sums and the squared dual violation of the factored Grid plan, without forming X.

On an h × w grid (N = h·w; h = w = n_g for the square solver) with row-major
flat indices a = (a_r, a_c) and b = (b_r, b_c), the plan at iteration k is

    log X_ab = ξ_a + kL_r[a_r, b_r] + kL_c[a_c, b_c] + ζ_b,

with ξ (``A``) and ζ (``B``) as (h, w) arrays and the per-axis tables
kL_r = k·ℓ_r (h × h) and kL_c = k·ℓ_c (w × w), ℓ = −φ/(C_max·η). A sum over b
separates into two one-dimensional log-sum-exps, O(N·(h + w)) in all:

    S[a_r, b_c]  = LSE_{b_r} (kL_r[a_r, b_r] + B[b_r, b_c])
    row_sum[a]   = exp(A[a_r, a_c] + LSE_{b_c} (kL_c[a_c, b_c] + S[a_r, b_c]))

    T[b_r, a_c]  = LSE_{a_r} (kL_r[a_r, b_r] + A[a_r, a_c])
    col_sum[b]   = exp(B[b_r, b_c] + LSE_{a_c} (kL_c[a_c, b_c] + T[b_r, a_c]))

The passes stay in the log domain: |k·ℓ| grows linearly in k, so exp(k·ℓ)
underflows in fp64 while the entries of X that carry mass stay representable.
Each pass reduces a three-axis broadcast of N·h or N·w entries (8 GiB in fp64
at n_g = 1024), which ``_lse_streamed`` builds in blocks of rows of an axis it
does not reduce, so a pass holds about two blocks at a time (a block and the
log-sum-exp's temporary).
"""

import os
from typing import Callable, Tuple

import torch

_CHECK_CHUNK_ELEMS = 1 << 24      # entries per block: 128 MiB in fp64


def _check_chunk_rows(n_rows: int, row_elems: int) -> int:
    """Rows per block, for a broadcast of ``row_elems`` entries per row.

    ``BDRS_GRID_CHECK_CHUNK_ELEMS`` overrides the entry budget and
    ``BDRS_GRID_CHECK_STREAM=0`` gives one block; both are read at every call."""
    n_rows = int(n_rows)
    if os.environ.get("BDRS_GRID_CHECK_STREAM", "1") == "0":
        return n_rows
    raw = os.environ.get("BDRS_GRID_CHECK_CHUNK_ELEMS")
    budget = _CHECK_CHUNK_ELEMS if raw is None else int(raw)
    if budget <= 0:
        raise ValueError(f"BDRS_GRID_CHECK_CHUNK_ELEMS must be positive, got {raw!r}")
    return max(1, min(n_rows, budget // int(row_elems)))


def _lse_streamed(block: Callable[[int, int], torch.Tensor], dim: int,
                  n_rows: int, row_elems: int) -> torch.Tensor:
    """``torch.logsumexp(block(0, n_rows), dim)``, computed one block of rows at a time.

    The streamed axis must not be reduced and must become dim 0 of the result.
    Each entry reduces the same elements, but torch may order the sum by the
    launch size: on CUDA with the default budget the result is bitwise that of
    one block up to n_g = 512 and equal to rounding at 1024; on CPU a reduction
    over a strided axis can differ by rounding."""
    rows = _check_chunk_rows(n_rows, row_elems)
    if rows >= n_rows:
        return torch.logsumexp(block(0, n_rows), dim=dim)
    return torch.cat([torch.logsumexp(block(i0, min(n_rows, i0 + rows)), dim=dim)
                      for i0 in range(0, n_rows, rows)], dim=0)


def factored_row_sums(A: torch.Tensor, B: torch.Tensor, kL_r: torch.Tensor,
                      kL_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Row sums of the factored plan, flattened row-major to (N,), and the first-pass table S.

    S (h, w) is returned for the separable objective in ``grid_cuda``, which reuses it.
    """
    h, w = A.shape
    BT = B.T
    logS_row = _lse_streamed(                              # (a_r, b_c)
        lambda i0, i1: kL_r[i0:i1].unsqueeze(1) + BT.unsqueeze(0),        # (a_r, b_c, b_r)
        2, h, w * h)
    lse_row = _lse_streamed(                               # (a_r, a_c)
        lambda i0, i1: kL_c.unsqueeze(0) + logS_row[i0:i1].unsqueeze(1),  # (a_r, a_c, b_c)
        2, h, w * w)
    return (A + lse_row).exp().reshape(-1), logS_row


def factored_col_sums(A: torch.Tensor, B: torch.Tensor, kL_r: torch.Tensor,
                      kL_c: torch.Tensor) -> torch.Tensor:
    """Column sums of the factored plan, flattened row-major to (N,)."""
    h, w = A.shape
    kLrT, kLcT, AT = kL_r.T, kL_c.T, A.T
    logT_col = _lse_streamed(                              # (b_r, a_c)
        lambda i0, i1: kLrT[i0:i1].unsqueeze(1) + AT.unsqueeze(0),        # (b_r, a_c, a_r)
        2, h, w * h)
    lse_col = _lse_streamed(                               # (b_r, b_c)
        lambda i0, i1: kLcT.unsqueeze(0) + logT_col[i0:i1].unsqueeze(1),  # (b_r, b_c, a_c)
        2, h, w * w)
    return (B + lse_col).exp().reshape(-1)


def factored_dual_sq_sum(log_u: torch.Tensor, log_v: torch.Tensor,
                         logK_block: Callable[[int], torch.Tensor], n_rows: int, *,
                         dtype: torch.dtype, device) -> torch.Tensor:
    """Σ_ab max(log u_a + log v_b + log K_ab, 0)², one grid row a_r of sources at a time.

    ``log_u``, ``log_v`` are (h, w); ``logK_block(a_r)`` must return the
    (w, h, w) block of log K = L = −C/η indexed (a_c, b_r, b_c), so each block
    has N·w entries and the work is O(N²). η times the square root of the sum
    is r_D = ‖(α ⊕ β − C)₊‖_F of α = η·log u, β = η·log v.
    """
    dual_sq_sum = torch.zeros((), dtype=dtype, device=device)
    for a_r in range(n_rows):
        viol = (log_u[a_r, :, None, None] + log_v[None, :, :]
                + logK_block(a_r)).clamp_min_(0.0)
        dual_sq_sum = dual_sq_sum + (viol * viol).sum()
        del viol
    return dual_sq_sum
