"""The KKT components of a candidate (X, α, β) on a DOT instance (C, r, c).

Dual feasibility is α_i + β_j ≤ C_ij, and

    r_P    = ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂ + ‖min(X, 0)‖_F
    r_D    = ‖max(α ⊕ β − C, 0)‖_F                    (``r_DF``)
    r_Dinf = max(0, max_ij (α_i + β_j − C_ij))         (a diagnostic)
    p_obj  = ⟨C, X⟩,   d_obj = ⟨α, r⟩ + ⟨β, c⟩,   gap = |p_obj − d_obj|

with ‖C‖_∞ = max_ij |C_ij| computed from the data, never assumed. Dense
inputs are reduced in fp64 whatever their dtype; a plan representation
computes its marginals and ⟨C, X⟩ in its own dtype.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from src.methods.dual_cleanup import _ctransform_separable, _dual_violation_sweep
from src.methods.grid_cost import (
    _build_grid_indices,
    _cost_axis,
    _cost_row_chunk,
    _grid_blocks,
    _row_chunk_for,
)


_SEPARABLE_COSTS = ("l1", "sql2")


# Above this many entries, max_ij |C_ij| and ‖min(X, 0)‖_F are taken in row
# blocks: C.abs() or X.clamp() would allocate a second m x n array next to the
# resident C and X (32 GiB each at n_g = 256). The blocked and whole-array
# reductions differ only by reassociation.
_CHUNK_ABOVE = 1 << 26


def _abs_max_chunked(C: torch.Tensor) -> float:
    """max_ij |C_ij|, in row blocks above ``_CHUNK_ABOVE`` entries (a max does
    not depend on the blocking)."""
    m, n = C.shape[0], C.shape[1] if C.dim() > 1 else 1
    if C.numel() <= _CHUNK_ABOVE:
        return float(C.abs().max().item())
    worst = float("-inf")
    step = _row_chunk_for(n)
    for i0 in range(0, m, step):
        worst = max(worst, float(C[i0:i0 + step].abs().max().item()))
    return worst


def _neg_norm_chunked(X: torch.Tensor) -> float:
    """‖min(X, 0)‖_F in row blocks, accumulated in fp64."""
    m, n = X.shape
    acc = torch.zeros((), dtype=torch.float64, device=X.device)
    step = _row_chunk_for(n)
    for i0 in range(0, m, step):
        blk = X[i0:i0 + step].clamp(max=0.0)
        acc = acc + (blk.to(torch.float64) ** 2).sum()
        del blk
    return float(acc.sqrt().item())


@dataclass
class Instance:
    """A DOT instance (C, r, c) with ‖C‖_∞ (``C_inf``) computed once.

    Exactly one cost form is set: the Grid cost (``ng``, ``cost_type`` in
    {l1, sql2}), rebuilt in row blocks and never formed, or a stored dense
    ``C`` of any dtype. Raises ValueError otherwise.
    """

    r: torch.Tensor
    c: torch.Tensor
    ng: Optional[int] = None
    cost_type: Optional[str] = None
    C: Optional[torch.Tensor] = None
    C_inf: float = field(init=False)

    def __post_init__(self) -> None:
        if (self.C is None) == (self.ng is None or self.cost_type is None):
            raise ValueError("set exactly one cost form: (ng, cost_type) or C")
        if self.C is None and self.cost_type not in _SEPARABLE_COSTS:
            raise ValueError(
                f"grid form only represents {_SEPARABLE_COSTS}; "
                f"got {self.cost_type!r}; pass a stored C instead")
        self.r = self.r.reshape(-1).to(torch.float64)
        self.c = self.c.reshape(-1).to(torch.float64)
        if self.C is not None:
            self.C_inf = _abs_max_chunked(self.C)
        else:
            ax = self._grid_axis(self.r.device)
            # C = T[a_r, b_r] + T[a_c, b_c], and both terms reach max T at once
            # when a and b are opposite corners of the lattice, so max C = 2·max T.
            self.C_inf = float(2.0 * ax.max().item())

    @property
    def m(self) -> int:
        return int(self.r.numel())

    @property
    def n(self) -> int:
        return int(self.c.numel())

    @property
    def is_grid(self) -> bool:
        return self.C is None

    def _grid_axis(self, device) -> torch.Tensor:
        return _cost_axis(str(self.cost_type), int(self.ng), device=device,
                          dtype=torch.float64)

    def assert_normalized_max1(self, *, rtol: float = 1e-9) -> None:
        """Assert the max-1 normalisation, |‖C‖_∞ − 1| ≤ ``rtol``."""
        if abs(self.C_inf - 1.0) > rtol:
            raise AssertionError(
                f"instance not max-1 normalised: ||C||_inf = {self.C_inf!r}")


@dataclass
class Candidate:
    """A candidate (X, α, β), with X a dense (m, n) tensor or a ``plan_metrics``
    plan representation.

    ``x_nonneg_structural=True`` declares X ≥ 0 by construction, and
    ‖min(X, 0)‖_F is recorded as 0. A representation without
    ``neg_part_norm`` (``BdrsFactoredPlan``) must declare it, or ``evaluate``
    raises ValueError.
    """

    X: Any
    alpha: torch.Tensor
    beta: torch.Tensor
    x_nonneg_structural: bool = False


def _primal_components(cand: Candidate, inst: Instance) -> Dict[str, float]:
    """r_P with its three terms, and p_obj = ⟨C, X⟩."""
    X = cand.X
    if isinstance(X, torch.Tensor):
        if X.shape != (inst.m, inst.n):
            raise ValueError(f"X shape {tuple(X.shape)} != ({inst.m}, {inst.n})")
        # On an fp32 X each sum first casts a full fp64 copy of X.
        rs = X.sum(dim=1, dtype=torch.float64)
        cs = X.sum(dim=0, dtype=torch.float64)
        if cand.x_nonneg_structural:
            negpart = 0.0
        else:
            if X.numel() <= _CHUNK_ABOVE:
                neg = X.clamp(max=0.0)
                negpart = float((neg.to(torch.float64) ** 2).sum().sqrt().item())
            else:
                negpart = _neg_norm_chunked(X)
        p_obj = _trace_fp64(X, inst)
    else:
        rs = X.row_sums().reshape(-1).to(torch.float64)
        cs = X.col_sums().reshape(-1).to(torch.float64)
        p_obj = float(X.trace_with_C())
        if cand.x_nonneg_structural:
            negpart = 0.0
        elif hasattr(X, "neg_part_norm"):
            negpart = float(X.neg_part_norm())
        else:
            raise ValueError(
                "representation has no neg_part_norm; declare "
                "x_nonneg_structural=True or provide one")
    err_r = float((rs - inst.r).norm(2).item())
    err_c = float((cs - inst.c).norm(2).item())
    return {"r_P": err_r + err_c + negpart, "p_obj": p_obj,
            "r_P_row": err_r, "r_P_col": err_c, "r_P_neg": negpart}


def _trace_fp64(X: torch.Tensor, inst: Instance, *, row_chunk: int = 0) -> float:
    """⟨C, X⟩ in row blocks, accumulated in fp64."""
    if inst.is_grid:
        ax = inst._grid_axis(X.device)
        a_r_full, a_c_full = _build_grid_indices(int(inst.ng), device=X.device)
        N = inst.m
        chunk = _row_chunk_for(N, row_chunk=row_chunk)
        acc = torch.zeros((), dtype=torch.float64, device=X.device)
        for i0 in range(0, N, chunk):
            i1 = min(N, i0 + chunk)
            C_blk = _cost_row_chunk(a_r_full[i0:i1], a_c_full[i0:i1],
                                    a_r_full, a_c_full, ax)
            acc = acc + (C_blk * X[i0:i1, :].to(torch.float64)).sum()
            del C_blk
        return float(acc.item())
    C = inst.C
    chunk = _row_chunk_for(inst.n, row_chunk=row_chunk)
    acc = torch.zeros((), dtype=torch.float64, device=X.device)
    for i0 in range(0, inst.m, chunk):
        i1 = min(inst.m, i0 + chunk)
        acc = acc + (C[i0:i1, :].to(torch.float64)
                     * X[i0:i1, :].to(torch.float64)).sum()
    return float(acc.item())


def _dual_sweep(alpha: torch.Tensor, beta: torch.Tensor, inst: Instance,
                *, want_fro: bool, row_chunk: int = 0) -> Dict[str, float]:
    """r_Dinf, and r_D when ``want_fro``, in one fp64 pass over row blocks of
    α ⊕ β − C (the Grid C rebuilt block by block)."""
    a = alpha.reshape(-1).to(torch.float64)
    b = beta.reshape(-1).to(torch.float64)
    if inst.is_grid:
        cost_fn = _grid_blocks(inst._grid_axis(alpha.device), int(inst.ng))
    else:
        def cost_fn(i0, i1):
            return inst.C[i0:i1, :].to(torch.float64)
    fro, mx = _dual_violation_sweep(a, b, cost_fn, inst.m,
                                    chunk=_row_chunk_for(inst.n, row_chunk=row_chunk),
                                    acc_like=a, want_fro=want_fro, want_max=True)
    out = {"r_Dinf": max(0.0, mx)}
    if want_fro:
        out["r_DF"] = fro
    return out


def support_dot(potential: torch.Tensor, mass: torch.Tensor) -> float:
    """⟨potential, mass⟩ summed over mass > 0 only, i.e. with 0·(±∞) = 0.

    Given a zero-mass entry r_i = 0, an entropic baseline may return
    α_i = −∞ (POT's ``sinkhorn_log`` takes log r as is), and the plain product
    would make d_obj and the gap NaN. On a strictly positive mass this is the
    ordinary dot product, bit for bit."""
    m = mass.reshape(-1)
    p = potential.reshape(-1)
    prod = torch.where(m > 0, p * m, torch.zeros((), dtype=m.dtype, device=m.device))
    return float(prod.sum().item())


def r_dinf_storedC(alpha: torch.Tensor, beta: torch.Tensor, C: torch.Tensor,
                   *, row_chunk: int = 0) -> float:
    """r_Dinf = max(0, max_ij (α_i + β_j − C_ij)) over a stored C, in fp64 row
    blocks."""
    a = alpha.reshape(-1).to(torch.float64)
    b = beta.reshape(-1).to(torch.float64)
    m, n = C.shape
    chunk = _row_chunk_for(n, row_chunk=row_chunk)
    max_acc = torch.full((), float("-inf"), dtype=torch.float64, device=a.device)
    for i0 in range(0, m, chunk):
        i1 = min(m, i0 + chunk)
        viol = a[i0:i1, None] + b[None, :] - C[i0:i1, :].to(torch.float64)
        max_acc = torch.maximum(max_acc, viol.max())
        del viol
    return max(0.0, float(max_acc.item()))


def r_dinf_separable(alpha: torch.Tensor, beta: torch.Tensor,
                     ng: int, cost_type: str) -> float:
    """r_Dinf on the Grid cost in O(n_g³), without C, through the c-transform
    β^c_i = min_j (C_ij − β_j): max_ij (α_i + β_j − C_ij) = max_i (α_i − β^c_i)."""
    a = alpha.reshape(-1).to(torch.float64)
    b = beta.reshape(-1).to(torch.float64)
    beta_c = _ctransform_separable(b, int(ng), cost_type)
    return max(0.0, float((a - beta_c).max().item()))


def evaluate(candidate: Candidate, instance: Instance,
             *, compute_frobenius_dual: bool = False,
             dense_dual_check: bool = False) -> Dict[str, float]:
    """The components of ``candidate`` on ``instance``.

    Returns r_P with its terms ``r_P_row``, ``r_P_col`` and ``r_P_neg``,
    ``r_Dinf``, ``p_obj``, ``d_obj`` and the unsigned ``gap`` |p_obj − d_obj|,
    plus r_D (``r_DF``) when ``compute_frobenius_dual``. r_Dinf (and r_D, if
    asked) come from one row-block sweep over C, except on the Grid cost with
    neither ``compute_frobenius_dual`` nor ``dense_dual_check`` set, where
    r_Dinf comes from the separable c-transform.
    """
    out = _primal_components(candidate, instance)
    a = candidate.alpha.reshape(-1).to(torch.float64)
    b = candidate.beta.reshape(-1).to(torch.float64)
    if a.numel() != instance.m or b.numel() != instance.n:
        raise ValueError(
            f"potentials ({a.numel()}, {b.numel()}) != marginals "
            f"({instance.m}, {instance.n})")
    d_obj = support_dot(a, instance.r) + support_dot(b, instance.c)
    out["d_obj"] = d_obj
    out["gap"] = abs(out["p_obj"] - d_obj)

    if instance.is_grid and not (compute_frobenius_dual or dense_dual_check):
        out["r_Dinf"] = r_dinf_separable(a, b, int(instance.ng),
                                         str(instance.cost_type))
    else:
        out.update(_dual_sweep(a, b, instance, want_fro=compute_frobenius_dual))
    return out
