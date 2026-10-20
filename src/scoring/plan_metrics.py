"""The plan-side metrics of a transport plan X: ``feaserr`` = r_P =
‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂ + ‖min(X, 0)‖_F and ``objective`` = ⟨C, X⟩.

A plan representation (``PlanRep``) exposes X through row sums, column sums and
⟨C, X⟩, so a factored, dense or sparse plan is scored by the same code. The
negative part is measured when the entries are available and is zero for the
factored plans, whose entries are exponentials. The duality gap is left to the
runner, since it depends on the solver's dual.
"""

import math
from typing import Any, Dict, Protocol

import numpy as np
import torch

from src.methods.factored_torch import (
    _pointcloud_block_chunk,
    _pointcloud_col_lse,
    _pointcloud_row_lse,
    _pointcloud_trace_CX,
)
from src.methods.grid_cmax import grid_cmax
from src.methods.grid_cost import _base_ell, _phi_and_cmax
from src.methods.grid_factored import factored_col_sums, factored_row_sums


def _nonsep_combine(cost_type: str, d_r: torch.Tensor, d_c: torch.Tensor,
                    ng: int) -> torch.Tensor:
    """The max-1 cost from per-axis distances |Δ|: l2 is
    √(d_r² + d_c²)/(√2(n_g − 1)), inf is max(d_r, d_c)/(n_g − 1)."""
    if cost_type == "l2":
        return torch.sqrt(d_r * d_r + d_c * d_c) / grid_cmax(cost_type, ng)
    if cost_type == "inf":
        return torch.maximum(d_r, d_c) / grid_cmax(cost_type, ng)
    raise ValueError(f"Not a non-separable cost_type: {cost_type!r}")


class PlanRep(Protocol):
    """What the scorers read from a plan."""

    def row_sums(self) -> torch.Tensor: ...      # shape (N,)
    def col_sums(self) -> torch.Tensor: ...      # shape (N,)
    def trace_with_C(self) -> float: ...         # <C, X>


class BdrsFactoredPlan:
    """The factored Grid plan, never formed:

        log X_ab = ξ_a + k·ℓ(|a_r − b_r|) + k·ℓ(|a_c − b_c|) + ζ_b,

    with ξ (``A``), ζ (``B``), k = ``iters`` and ℓ = −φ/(C_max·η).
    """

    def __init__(self, *, A: torch.Tensor, B: torch.Tensor,
                 ng: int, cost_type: str, eta: float, iters: int):
        if A.numel() != ng * ng or B.numel() != ng * ng:
            raise ValueError(
                f"A, B must have ng*ng = {ng*ng} elements; got {A.numel()}, {B.numel()}"
            )
        self._A = A.view(ng, ng).contiguous()
        self._B = B.view(ng, ng).contiguous()
        self._ng = int(ng)
        self._N = self._ng * self._ng
        self._cost_type = cost_type
        self._eta = float(eta)
        self._iters = int(iters)

        device, dtype = A.device, A.dtype
        base_ell = _base_ell(cost_type, self._ng, self._eta, dtype=dtype, device=device)
        kL = self._iters * base_ell
        idx = torch.arange(self._ng, device=device)
        D = (idx[:, None] - idx[None, :]).abs()
        self._kL_mat = kL[D].contiguous()            # kL[|i - j|]

    def row_sums(self) -> torch.Tensor:
        """Σ_b X_ab by the two-pass separable log-sum-exp, O(n_g³)."""
        return factored_row_sums(self._A, self._B, self._kL_mat, self._kL_mat)[0]

    def col_sums(self) -> torch.Tensor:
        """Σ_a X_ab, likewise."""
        return factored_col_sums(self._A, self._B, self._kL_mat, self._kL_mat)

    def trace_with_C(self) -> float:
        """⟨C, X⟩ by ``grid_cuda``'s separable objective, O(n_g³)."""
        from src.methods.grid_cuda import _compute_factored_objective
        return _compute_factored_objective(
            self._A, self._B, self._ng, self._cost_type, self._eta, self._iters)


class DenseTorchPlan:
    """A dense (N, N) plan X; only ``neg_part_norm`` upcasts X to fp64.

    ``trace_with_C`` rebuilds the Grid cost (l1, sql2, l2, inf) of ``ng`` and
    ``cost_type`` in row blocks, or reads a stored ``C`` of X's shape, which is
    required when ``ng`` is None.
    """

    def __init__(self, X: torch.Tensor, *, ng: "int | None" = None,
                 cost_type: str = "", row_chunk: int = 0,
                 C: "torch.Tensor | None" = None):
        if X.dim() != 2 or X.shape[0] != X.shape[1]:
            raise ValueError(f"X must be square; got shape {tuple(X.shape)}")
        if ng is None:
            if C is None:
                raise ValueError(
                    "DenseTorchPlan needs ng (to rebuild a grid cost) or C "
                    "(a stored cost); got neither")
            N_expected = int(X.shape[0])
        else:
            N_expected = ng * ng
            if X.shape[0] != N_expected:
                raise ValueError(f"X side {X.shape[0]} != ng*ng = {N_expected}")
        if C is not None and tuple(C.shape) != tuple(X.shape):
            raise ValueError(
                f"stored C shape {tuple(C.shape)} != X shape {tuple(X.shape)}")
        self._X = X
        self._C = C
        self._ng = None if ng is None else int(ng)
        self._N = N_expected
        self._cost_type = cost_type
        if row_chunk <= 0:
            # At most 2^25 entries (256 MiB in fp64) per block.
            row_chunk = max(1, min(self._N, 32 * 1024 * 1024 // max(1, self._N)))
        self._row_chunk = int(row_chunk)

    def row_sums(self) -> torch.Tensor:
        return self._X.sum(dim=1)

    def col_sums(self) -> torch.Tensor:
        return self._X.sum(dim=0)

    def neg_part_norm(self) -> float:
        """‖min(X, 0)‖_F in fp64, row block by row block (no second N x N array)."""
        acc = 0.0
        for i0 in range(0, self._X.shape[0], self._row_chunk):
            blk = self._X[i0:i0 + self._row_chunk].to(torch.float64).clamp(max=0.0)
            acc += float((blk * blk).sum().item())
        return acc ** 0.5

    def trace_with_C(self) -> float:
        ng, N = self._ng, self._N
        device, dtype = self._X.device, self._X.dtype
        chunk = max(1, self._row_chunk)
        obj = torch.zeros((), dtype=dtype, device=device)

        if self._C is not None:
            for i0 in range(0, N, chunk):
                i1 = min(N, i0 + chunk)
                obj = obj + (self._C[i0:i1] * self._X[i0:i1]).sum()
            return float(obj.item())

        idx = torch.arange(ng, device=device)
        D = (idx[:, None] - idx[None, :]).abs()
        nonsep = self._cost_type in ("l2", "inf")
        if nonsep:
            axis_dist = D.to(dtype)
            c_max = None
        else:
            phi, c_max = _phi_and_cmax(self._cost_type, ng, dtype=dtype, device=device)
            axis_dist = phi[D]

        a_r_full = torch.arange(N, device=device) // ng
        a_c_full = torch.arange(N, device=device) % ng

        for i0 in range(0, N, chunk):
            i1 = min(N, i0 + chunk)
            a_r = a_r_full[i0:i1]
            a_c = a_c_full[i0:i1]
            row_part = axis_dist[a_r][:, a_r_full]                # (chunk, N)
            col_part = axis_dist[a_c][:, a_c_full]                # (chunk, N)
            if nonsep:
                C_block = _nonsep_combine(self._cost_type, row_part, col_part, ng)
            else:
                C_block = (row_part + col_part) / c_max
            obj = obj + (C_block * self._X[i0:i1, :]).sum()
            del C_block, row_part, col_part
        return float(obj.item())


class SparseTorchPlan:
    """A COO plan: ``indices`` (2, nnz) and ``values`` (nnz,), with C evaluated
    only at the stored entries."""

    def __init__(self, indices: torch.Tensor, values: torch.Tensor,
                 *, ng: int, cost_type: str):
        if indices.dim() != 2 or indices.shape[0] != 2:
            raise ValueError(f"indices must have shape (2, nnz); got {tuple(indices.shape)}")
        if values.dim() != 1 or values.numel() != indices.shape[1]:
            raise ValueError("values must be 1-D with length matching indices' second dim")
        self._indices = indices
        self._values = values
        self._ng = int(ng)
        self._N = self._ng * self._ng
        self._cost_type = cost_type

    def row_sums(self) -> torch.Tensor:
        out = torch.zeros(self._N, dtype=self._values.dtype, device=self._values.device)
        out.scatter_add_(0, self._indices[0], self._values)
        return out

    def col_sums(self) -> torch.Tensor:
        out = torch.zeros(self._N, dtype=self._values.dtype, device=self._values.device)
        out.scatter_add_(0, self._indices[1], self._values)
        return out

    def neg_part_norm(self) -> float:
        """‖min(X, 0)‖_F over the stored entries (the others are zero), the
        values at a repeated (i, j) first summed into one entry."""
        vals = self._values.to(torch.float64)
        if bool((vals < 0).any()):
            flat = self._indices[0] * self._N + self._indices[1]
            uniq, inv = torch.unique(flat, return_inverse=True)
            if uniq.numel() < flat.numel():
                vals = torch.zeros(uniq.numel(), dtype=vals.dtype,
                                   device=vals.device).index_add_(0, inv, vals)
        return float(vals.clamp(max=0.0).norm(2).item())

    def trace_with_C(self) -> float:
        ng = self._ng
        dtype = self._values.dtype
        device = self._values.device

        i_idx = self._indices[0]
        j_idx = self._indices[1]
        a_r = i_idx // ng;   a_c = i_idx % ng
        b_r = j_idx // ng;   b_c = j_idx % ng
        d_r = (a_r - b_r).abs()
        d_c = (a_c - b_c).abs()
        if self._cost_type in ("l2", "inf"):
            C_at = _nonsep_combine(self._cost_type, d_r.to(dtype), d_c.to(dtype), ng)
        else:
            phi, c_max = _phi_and_cmax(self._cost_type, ng, dtype=dtype, device=device)
            C_at = (phi[d_r] + phi[d_c]) / c_max
        return float((C_at * self._values).sum().item())


def sparse_plan_from_scipy_coo(coo, *, ng: int, cost_type: str) -> "SparseTorchPlan":
    """A ``scipy.sparse.coo_matrix`` plan as a ``SparseTorchPlan``, with the
    values upcast to fp64 (HALO's coupling is fp32)."""
    indices = torch.from_numpy(
        np.ascontiguousarray(np.stack([coo.row, coo.col]).astype(np.int64)))
    values = torch.from_numpy(coo.data.astype(np.float64))
    return SparseTorchPlan(indices, values, ng=ng, cost_type=cost_type)


class SymbolicFactoredPlan:
    """The m × n plan log X_ij = a_i + s·C_ij + b_j, never formed, over a block
    source ``cost_fn(i0, i1) -> C[i0:i1, :]``: row slices of a stored C, or a
    Point Cloud cost (``geo_cost.make_metric_cost_fn``).

    The reductions are those of ``factored_torch``, whose plans are
    log X = A − scale·C + B, so scale = −s.
    """

    def __init__(self, *, a: torch.Tensor, b: torch.Tensor, s: float, cost_fn,
                 m: int, n: int, row_chunk: int = 0):
        self._a = a.reshape(-1).to(torch.float64)
        self._b = b.reshape(-1).to(torch.float64)
        if self._a.numel() != m or self._b.numel() != n:
            raise ValueError(f"a has {self._a.numel()} entries and b {self._b.numel()}, "
                             f"expected m={m} and n={n}")
        if not math.isfinite(float(s)):
            raise ValueError(f"exponent scale s must be finite, got {s}")
        if not (bool(torch.isfinite(self._a).all()) and bool(torch.isfinite(self._b).all())):
            raise ValueError("a and b must be finite (a zero-mass row gives log 0 = -inf)")
        self._s = float(s)
        self._cost_fn = cost_fn
        self.m, self.n = int(m), int(n)
        self._chunk = _pointcloud_block_chunk(self.m, self.n, row_chunk)

    def row_sums(self) -> torch.Tensor:
        out = torch.empty(self.m, dtype=torch.float64, device=self._a.device)
        _pointcloud_row_lse(out, self._cost_fn, self._b, -self._s, self.m, self._chunk)
        return (self._a + out).exp()

    def col_sums(self) -> torch.Tensor:
        out = torch.empty(self.n, dtype=torch.float64, device=self._b.device)
        _pointcloud_col_lse(out, self._cost_fn, self._a, -self._s, self.m, self._chunk)
        return (self._b + out).exp()

    def trace_with_C(self) -> float:
        return float(_pointcloud_trace_CX(self._cost_fn, self._a, self._b, -self._s,
                                          self.m, self._chunk).item())


def evaluate_plan_metrics(rep: PlanRep, r: torch.Tensor, c: torch.Tensor) -> Dict[str, Any]:
    """``feaserr`` = r_P and ``objective`` = ⟨C, X⟩, with r_P's three terms.

    The negative part is 0.0, and ``negpart_measured`` False, when ``rep`` has
    no ``neg_part_norm``.
    """
    rs = rep.row_sums()
    cs = rep.col_sums()
    err_r = float((rs - r).norm(2).item())
    err_c = float((cs - c).norm(2).item())
    measured = hasattr(rep, "neg_part_norm")
    neg = float(rep.neg_part_norm()) if measured else 0.0
    return {
        "feaserr": err_r + err_c + neg,
        "objective": float(rep.trace_with_C()),
        "row_sum_err_l2": err_r,
        "col_sum_err_l2": err_c,
        "neg_part_l2": neg,
        "negpart_measured": measured,
    }
