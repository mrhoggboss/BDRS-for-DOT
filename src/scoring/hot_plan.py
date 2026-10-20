"""The plan HOT's ``recover_transport_plan()`` returns, scored without forming it.

HOT's flow x = (x¹, x²) lives on a reduced graph of the n_g × n_g lattice. x¹
moves mass along a row, from source (p, q) to the intermediate node (p, q'), and
x² along a column, from (p, q') to target (p', q'). In HOT's flat layout,

    x¹ from (p, q) to (p, q')    = x[p·n_g² + q'·n_g + q],
    x² from (p, q') to (p', q')  = x[n_g³ + p'·n_g² + p·n_g + q'].

``recover_transport_plan()`` scales x by b_scale, clamps it at 0, and at each
intermediate node k glues the in-flows u_q (from the sources (p, q), in the
order of q) to the out-flows v_p' (to the targets (p', q'), in the order of p')
by the north-west-corner rule. With U = Σ u, V = Σ v and prefix sums U_<q,
V_<p', source q sends s_q = min(u_q, (V − U_<q)₊) through k and target p'
receives t_p' = min(v_p', (U − V_<p')₊). A source and a target are joined
through one intermediate node only, so, with s and t taken at k = (p, q'),

    (X1)_(p,q)     = Σ_q' s_q,
    (Xᵀ1)_(p',q')  = Σ_p t_p',
    ⟨C, X⟩         = Σ_k (Σ_q T[q, q']·s_q + Σ_p' T[p, p']·t_p'),

with T[i, j] = φ(|i − j|)/C_max the per-axis table of the sql2 cost. Each sum
is O(n_g³), the size of x; X itself has N² = n_g⁴ entries and is never formed.
X ≥ 0 because x is clamped first. The prefix sums reassociate HOT's sequential
subtractions, so the values agree with those of the dense X up to rounding.
"""

import torch

from src.methods.grid_cost import _cost_axis

# Entries of x¹ per block of lattice rows p: 2²⁴, 128 MiB in fp64.
_BLOCK_ENTRIES = 1 << 24


def _exclusive_cumsum(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Σ_{j < i} t_j along ``dim``, 0 at i = 0."""
    s = t.cumsum(dim)
    out = torch.zeros_like(t)
    n = t.size(dim)
    if n > 1:
        out.narrow(dim, 1, n - 1).copy_(s.narrow(dim, 0, n - 1))
    return out


class HotRecoveredPlan:
    """X = ``recover_transport_plan()`` of HOT's Halpern iterate ``x``,
    exposed through its row sums, column sums and ⟨C, X⟩ on the max-1 sql2
    cost.

    The constructor computes all three in one pass over blocks of
    ``block_rows`` lattice rows p (default: about 2²⁴ entries of x¹ per
    block), reading ``x`` without modifying it. It has no ``neg_part_norm``:
    X ≥ 0 by construction, so a scorer must declare ``x_nonneg_structural``.
    """

    def __init__(self, x: torch.Tensor, b_scale, ng: int, *, block_rows: int = 0):
        ng = int(ng)
        n3 = ng ** 3
        if x.dim() != 1 or x.numel() != 2 * n3:
            raise ValueError(
                f"x must be HOT's flat flow of 2·n_g³ = {2 * n3} entries; "
                f"got shape {tuple(x.shape)}")
        if block_rows <= 0:
            block_rows = max(1, min(ng, _BLOCK_ENTRIES // (ng * ng)))
        dtype, device = x.dtype, x.device
        zero = torch.zeros((), dtype=dtype, device=device)
        x1 = x[:n3].view(ng, ng, ng)          # [p, q', q]
        x2 = x[n3:].view(ng, ng, ng)          # [p', p, q']
        T = _cost_axis("sql2", ng, device=device, dtype=dtype)

        rows = torch.empty(ng, ng, dtype=dtype, device=device)    # [p, q]
        cols = torch.zeros(ng, ng, dtype=dtype, device=device)    # [p', q']
        obj = torch.zeros((), dtype=dtype, device=device)
        for p0 in range(0, ng, block_rows):
            p1 = min(ng, p0 + block_rows)
            # HOT's own scaling and clamp, on a copy.
            u = torch.maximum(torch.mul(x1[p0:p1], b_scale), zero)      # [p, q', q]
            v = torch.maximum(torch.mul(x2[:, p0:p1], b_scale), zero)   # [p', p, q']
            U = u.sum(dim=2)                                            # [p, q']
            V = v.sum(dim=0)                                            # [p, q']
            sent = torch.minimum(u, (V[:, :, None] - _exclusive_cumsum(u, 2)).clamp_min(0))
            recv = torch.minimum(v, (U[None] - _exclusive_cumsum(v, 0)).clamp_min(0))
            rows[p0:p1] = sent.sum(dim=1)
            cols += recv.sum(dim=1)
            obj += (sent * T[None]).sum() + (recv * T[:, p0:p1, None]).sum()
            del u, v, sent, recv

        self._rows = rows.reshape(-1)
        self._cols = cols.reshape(-1)
        self._obj = float(obj.item())

    def row_sums(self) -> torch.Tensor:
        return self._rows

    def col_sums(self) -> torch.Tensor:
        return self._cols

    def trace_with_C(self) -> float:
        return self._obj
