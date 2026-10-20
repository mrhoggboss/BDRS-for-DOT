"""The KKT residual of a Pareto block.

A block stores the raw components rather than an error, so the residual is
computed from recorded rows. Torch-free: the report layer imports it.
"""

import math
from typing import Any, Dict

# The component keys of a scored Pareto block.
PARETO_COMPONENTS = (
    "r_P", "r_P_row", "r_P_col", "r_P_neg", "r_Dinf", "r_DF",
    "p_obj", "d_obj", "gap", "C_inf",
)


def is_finite_number(v: Any) -> bool:
    """An int or float that is finite and not a bool."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def absolute_kkt_error(blk: Dict[str, Any]) -> Dict[str, Any]:
    """The KKT residual E_abs = max{r_P, r_D, |g|}."""
    def _raw(x):
        return float(abs(x)) if is_finite_number(x) else None

    a_p = _raw(blk.get("r_P"))
    a_d = _raw(blk.get("r_DF"))
    a_g = _raw(blk.get("gap"))
    terms = [a_p, a_d, a_g]
    return {"E_abs_primal": a_p, "E_abs_dual": a_d, "E_abs_gap": a_g,
            "E_abs": max(terms) if all(t is not None for t in terms) else None}
