"""The stored-C LSE kernels ``dense_row_lse_scaled`` and ``dense_col_lse_scaled``
against the torch tiled LSEs of ``factored_torch``.

The kernels take neg_scale = −k/η; the torch helpers take +k/η and negate it
themselves. A wrong sign gives finite values, so each call below derives one
from the other. The kernels round the product and the sum separately, as torch
does (``__dmul_rn``/``__dadd_rn``, no FMA contraction), so only the summation
order differs: the bar is 1e-12 relative, not equality.
"""

import pytest
import torch

from src.methods.cuda_ext import load as _load_extension
from src.methods.factored_torch import _pointcloud_col_lse, _pointcloud_row_lse
from tests.helpers import requires_cuda

pytestmark = requires_cuda

DEV = "cuda"
DT = torch.float64
BAR = 1e-12
# m < n, m > n, a small square, and N × N with N = n_g² at n_g = 32 and 64.
SHAPES = [(40, 24), (24, 40), (64, 64), (32 * 32, 32 * 32), (64 * 64, 64 * 64)]
SCALES = [1.0, 1e3, 1e5]          # k/η, positive
ROWS_PER_CHUNK = 1024


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d / max(s, 1.0)


def _inputs(m: int, n: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    C = torch.rand((m, n), generator=g, device=DEV, dtype=DT)
    w_n = torch.randn(n, generator=g, device=DEV, dtype=DT)
    w_m = torch.randn(m, generator=g, device=DEV, dtype=DT)
    return C, w_n, w_m


@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("k_over_eta", SCALES)
def test_dense_row_lse_scaled(m, n, k_over_eta):
    ext = _load_extension()
    C, w_n, _ = _inputs(m, n, seed=1000 + m + n)
    neg_scale = -(k_over_eta)
    out = torch.empty(m, device=DEV, dtype=DT)
    ext.dense_row_lse_scaled(out, C, w_n, neg_scale)
    ref = torch.empty(m, device=DEV, dtype=DT)
    _pointcloud_row_lse(ref, lambda i0, i1: C[i0:i1], w_n, k_over_eta, m, ROWS_PER_CHUNK)
    assert torch.isfinite(out).all(), "kernel produced non-finite output"
    r = _rel(out, ref)
    assert r < BAR, f"row LSE m={m} n={n} k/eta={k_over_eta:g}: rel {r:.3e} >= {BAR:g}"


@pytest.mark.parametrize("m,n", SHAPES)
@pytest.mark.parametrize("k_over_eta", SCALES)
def test_dense_col_lse_scaled(m, n, k_over_eta):
    ext = _load_extension()
    C, _, w_m = _inputs(m, n, seed=2000 + m + n)
    neg_scale = -(k_over_eta)
    n_chunks = (m + ROWS_PER_CHUNK - 1) // ROWS_PER_CHUNK
    out = torch.empty(n, device=DEV, dtype=DT)
    sm = torch.empty(n_chunks * n, device=DEV, dtype=DT)
    ss = torch.empty(n_chunks * n, device=DEV, dtype=DT)
    ext.dense_col_lse_scaled(out, C, w_m, neg_scale, sm, ss, ROWS_PER_CHUNK)
    ref = torch.empty(n, device=DEV, dtype=DT)
    _pointcloud_col_lse(ref, lambda i0, i1: C[i0:i1], w_m, k_over_eta, m, ROWS_PER_CHUNK)
    assert torch.isfinite(out).all(), "kernel produced non-finite output"
    r = _rel(out, ref)
    assert r < BAR, f"col LSE m={m} n={n} k/eta={k_over_eta:g}: rel {r:.3e} >= {BAR:g}"


def test_scaled_agrees_with_unscaled_row():
    """Folding −1/η into the kernel matches pre-scaling C to the bar, not
    bitwise: (−1/η)·C and (−C)/η round differently."""
    ext = _load_extension()
    eta = 7.5
    m, n = 64, 96
    C, w_n, _ = _inputs(m, n, seed=31337)
    out1 = torch.logsumexp((-C / eta) + w_n[None, :], dim=1)
    out2 = torch.empty(m, device=DEV, dtype=DT)
    ext.dense_row_lse_scaled(out2, C, w_n, -1.0 / eta)
    r = _rel(out2, out1)
    assert r < BAR, f"scaled vs pre-scaled row LSE: rel {r:.3e} >= {BAR:g}"


def test_scaled_agrees_with_unscaled_col():
    ext = _load_extension()
    eta = 7.5
    m, n = 96, 64
    C, _, w_m = _inputs(m, n, seed=31338)
    n_chunks = (m + ROWS_PER_CHUNK - 1) // ROWS_PER_CHUNK
    out1 = torch.logsumexp((-C / eta) + w_m[:, None], dim=0)
    out2 = torch.empty(n, device=DEV, dtype=DT)
    sm = torch.empty(n_chunks * n, device=DEV, dtype=DT)
    ss = torch.empty(n_chunks * n, device=DEV, dtype=DT)
    ext.dense_col_lse_scaled(out2, C, w_m, -1.0 / eta, sm, ss, ROWS_PER_CHUNK)
    r = _rel(out2, out1)
    assert r < BAR, f"scaled vs pre-scaled col LSE: rel {r:.3e} >= {BAR:g}"


def test_wrong_sign_is_detectably_different():
    """A wrong sign gives finite values (the solver would only fail to
    converge), so check that it changes the output."""
    ext = _load_extension()
    m, n, k_over_eta = 64, 64, 1e3
    C, w_n, _ = _inputs(m, n, seed=4242)
    a = torch.empty(m, device=DEV, dtype=DT)
    b = torch.empty(m, device=DEV, dtype=DT)
    ext.dense_row_lse_scaled(a, C, w_n, -(k_over_eta))
    ext.dense_row_lse_scaled(b, C, w_n, +(k_over_eta))
    assert _rel(a, b) > 1.0, "sign of neg_scale has no effect -- convention is broken"
