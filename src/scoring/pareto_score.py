"""The Pareto block: the raw KKT components of a recovered (X, α, β), which a
runner stores as ``algo_extras["pareto"]``.

The block holds components, not an error (``kkt_error`` combines them), and
its gap is signed, g = p_obj − d_obj. Non-finite values are stored as None.
"""

import math
from typing import Any, Dict, Optional

import torch

from src.methods.dual_cleanup import (
    cost_inf_symbolic,
    _dual_feas_F_symbolic,
    _dual_feas_inf_symbolic,
)
from src.scoring.kkt_residual import Candidate, Instance, evaluate, support_dot


def _finite(v: Any) -> Any:
    """A non-finite float as None (json.dumps would write the non-standard
    NaN/Infinity tokens); anything else unchanged."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def sanitize_block(d: Dict[str, Any]) -> Dict[str, Any]:
    """The block with every non-finite float value replaced by None."""
    return {k: _finite(v) for k, v in d.items()}


def make_instance(
    r: torch.Tensor, c: torch.Tensor, *,
    ng: Optional[int] = None, cost_type: Optional[str] = None,
    C: Optional[torch.Tensor] = None,
    assert_max1: bool = True, assert_rtol: float = 1e-6,
) -> Instance:
    """The ``kkt_residual.Instance`` of one cost form; with ``assert_max1``,
    asserts |‖C‖_∞ − 1| ≤ ``assert_rtol``."""
    inst = Instance(r=r, c=c, ng=ng, cost_type=cost_type, C=C)
    if assert_max1:
        inst.assert_normalized_max1(rtol=assert_rtol)
    return inst


def score(
    X: Any, alpha: torch.Tensor, beta: torch.Tensor,
    r: torch.Tensor, c: torch.Tensor, *,
    ng: Optional[int] = None, cost_type: Optional[str] = None,
    C: Optional[torch.Tensor] = None,
    x_nonneg_structural: bool = False,
    compute_frobenius: bool = True,
    assert_max1: bool = True,
    instance: Optional[Instance] = None,
) -> Dict[str, Any]:
    """The Pareto block of a recovered (X, α, β) on the Grid cost (``ng``,
    ``cost_type``) or a stored ``C``.

    ``X`` and ``x_nonneg_structural`` are as in ``kkt_residual.Candidate``. A
    prebuilt ``instance``, reused across candidates, replaces r, c, the cost
    arguments and ``assert_max1``.
    """
    inst = instance if instance is not None else make_instance(
        r, c, ng=ng, cost_type=cost_type, C=C, assert_max1=assert_max1)
    cand = Candidate(X=X, alpha=alpha, beta=beta,
                     x_nonneg_structural=x_nonneg_structural)
    m = evaluate(cand, inst, compute_frobenius_dual=compute_frobenius)
    return sanitize_block({
        "r_P": m["r_P"],
        "r_P_row": m["r_P_row"],
        "r_P_col": m["r_P_col"],
        "r_P_neg": m["r_P_neg"],
        "r_Dinf": m["r_Dinf"],
        "r_DF": m.get("r_DF"),                 # None unless compute_frobenius
        "p_obj": m["p_obj"],
        "d_obj": m["d_obj"],
        "gap": m["p_obj"] - m["d_obj"],        # signed; evaluate's gap is |·|
        "C_inf": inst.C_inf,
    })


def score_symbolic(
    *, p_obj: float, r_P_row: Optional[float], r_P_col: Optional[float],
    r_P_neg: Optional[float], alpha: torch.Tensor, beta: torch.Tensor,
    r: torch.Tensor, c: torch.Tensor, cost_fn, m: int, n: int,
    r_Dinf: Optional[float] = None, r_DF: Optional[float] = None,
    C_inf: Optional[float] = None,
) -> Dict[str, Any]:
    """The Pareto block for a cost given only as a block source
    ``cost_fn(i0, i1) -> C[i0:i1, :]`` (a Point Cloud cost).

    p_obj and the primal terms come from the caller; ``r_P`` is None when one
    of its terms is. ``r_Dinf`` and ``r_DF``, when passed, must come from the
    same ``dual_cleanup`` sweeps on this (α, β) and cost, and ``C_inf`` must
    be ‖C‖_∞ of this cost; they are then not recomputed."""
    a = alpha.reshape(-1).to(torch.float64)
    b = beta.reshape(-1).to(torch.float64)
    r = r.reshape(-1).to(torch.float64)
    c = c.reshape(-1).to(torch.float64)
    if r_Dinf is None:
        r_Dinf = _dual_feas_inf_symbolic(a, b, cost_fn, m, n)
    if r_DF is None:
        r_DF = _dual_feas_F_symbolic(a, b, cost_fn, m, n)
    if C_inf is None:
        C_inf = cost_inf_symbolic(cost_fn, m, n)
    d_obj = support_dot(a, r) + support_dot(b, c)
    terms = (r_P_row, r_P_col, r_P_neg)
    return sanitize_block({
        "r_P": None if None in terms else r_P_row + r_P_col + r_P_neg,
        "r_P_row": r_P_row, "r_P_col": r_P_col, "r_P_neg": r_P_neg,
        "r_Dinf": r_Dinf, "r_DF": r_DF,
        "p_obj": p_obj, "d_obj": d_obj,
        "gap": p_obj - d_obj,
        "C_inf": C_inf,
    })


# Largest n_g at which SinkhornGrid's zero dual residuals are also measured:
# the sweep rebuilds all N² entries of C in fp64, 2 GiB at n_g = 128 and
# 8 TiB at 1024.
POSTHOC_VERIFY_NG_MAX = 128


def posthoc_fill_sinkhorngrid_analytic() -> Dict[str, Any]:
    """SinkhornGrid's deferred dual fields without its sidecar: r_D = r_Dinf = 0,
    since an entropic dual is feasible."""
    return sanitize_block({"r_Dinf": 0.0, "r_DF": 0.0,
                           "dual_residual_source": "analytic_entropic_feasibility"})


def posthoc_fill_sinkhorngrid_rdinf(fg_path: str) -> Dict[str, Any]:
    """SinkhornGrid's deferred dual fields from its ``.pareto_fg.npz`` sidecar.

    SinkhornGrid runs without torch, so it saves (f, g, r, c, ng, cost_type)
    and leaves r_D and r_Dinf to this function. The dual is α = f,
    β = g: OTT's potentials are already in cost units. r_D = r_Dinf = 0 by
    entropic feasibility; for n_g ≤ ``POSTHOC_VERIFY_NG_MAX`` both are also
    measured, and a positive measurement replaces both zeros. Returns the
    fields to merge into the block."""
    import numpy as np
    from src.scoring.kkt_residual import Instance, _dual_sweep
    d = np.load(fg_path, allow_pickle=False)
    f = torch.as_tensor(np.asarray(d["f"], dtype=np.float64))
    g = torch.as_tensor(np.asarray(d["g"], dtype=np.float64))
    r = torch.as_tensor(np.asarray(d["r"], dtype=np.float64))
    c = torch.as_tensor(np.asarray(d["c"], dtype=np.float64))
    ng = int(np.asarray(d["ng"]))
    cost_type = str(np.asarray(d["cost_type"]))

    # Entropic feasibility: X_ij = exp((α_i + β_j − C_ij)/ε), and after a
    # half-update X matches one marginal exactly, so X_ij ≤ r_i ≤ 1 (or
    # ≤ c_j ≤ 1). So α ⊕ β ≤ C, and both residuals are 0 in exact arithmetic.
    out = {"r_Dinf": 0.0, "r_DF": 0.0,
           "dual_residual_source": "analytic_entropic_feasibility"}

    if ng <= POSTHOC_VERIFY_NG_MAX:
        inst = Instance(r=r.reshape(-1), c=c.reshape(-1), ng=ng, cost_type=cost_type)
        sw = _dual_sweep(f, g, inst, want_fro=True)
        meas_inf, meas_fro = float(sw["r_Dinf"]), float(sw.get("r_DF") or 0.0)
        out["dual_residual_source"] = "verified_zero"
        out["r_Dinf_measured"] = meas_inf
        out["r_DF_measured"] = meas_fro
        if max(meas_inf, meas_fro) > 0.0:
            out["r_Dinf"], out["r_DF"] = meas_inf, meas_fro
            out["dual_residual_source"] = "measured_nonzero_premise_violated"
    return sanitize_block(out)
