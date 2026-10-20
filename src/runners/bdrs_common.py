"""What the three BDRS runners (``run_grid``, ``run_general``, ``run_pointcloud``)
share, from the solve parameters to the recorded certificate of the delivered
(X, α′, β′).
"""

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from src.methods.params import BregmanDRParams
from src.protocol.instance_record import InstanceSpec
from src.runners.common import load_marginal_unit_mass


# β′_j = min_i(C_ij − α_i), then α′_i = min_j(C_ij − β′_j). The stop and the
# delivered dual use the same variant, so after a feasible-gap stop the delivered
# (α′, β′) is the pair the stop certified.
CTRANSFORM_VARIANT = "alternating"

# t_solve brackets the solver call: the solver's own setup (session, workspace,
# CUDA-graph capture), its iterations and its checks are inside; loading the
# marginals, the runner's cost construction and the warm-up are outside (each
# runner lists what it excludes in its T_SOLVE_EXCLUDES). t_total adds the recovery of (X, α′, β′); the scoring is on neither clock.
T_SOLVE_BOUNDARY = "solver_call"


def _t_solve_includes(spec: InstanceSpec,
                      device: Optional[torch.device] = None) -> list:
    """The one-time costs inside ``t_solve`` that a warm-up would move out: the
    extension JIT and the process's first CUDA-graph capture. Empty after a
    warm-up and on CPU."""
    if int(getattr(spec, "warmup", 0) or 0) > 0:
        return []
    if device is not None and torch.device(device).type != "cuda":
        return []
    return ["extension_jit", "first_call_graph_capture"]

DEFAULT_CHECK_EVERY = 200

# T_chk by lattice side n_g; Point Cloud and General look up an equivalent n_g.
# T_chk moves only the iteration at which the stop lands, never the iterates.
# One schedule serves every tolerance: the acceptance test is not monotone in k
# (the residuals dip below tol and rise again), so a coarser T_chk at a tight
# tol can step over a dip and stop much later.
CHECK_EVERY_BY_N = {32: 100, 64: 200, 128: 500, 256: 1000, 512: 2000, 1024: 4000}


def _resolve_check_every(resolution: int, override: Optional[int] = None) -> int:
    if override is not None:
        return int(override)
    return int(CHECK_EVERY_BY_N.get(int(resolution), DEFAULT_CHECK_EVERY))


def _load_marginals(spec: InstanceSpec, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """The original (r, c), unit mass. Never floored here: the solver floors its
    own copy for the iteration only, and every check and metric uses (r, c)."""
    r_np = load_marginal_unit_mass(spec.marginal_i_path)
    c_np = load_marginal_unit_mass(spec.marginal_j_path)
    return (
        torch.as_tensor(r_np, dtype=torch.float64, device=device),
        torch.as_tensor(c_np, dtype=torch.float64, device=device),
    )


def solve_params(eta: float, spec: InstanceSpec, *,
                 check_every: int) -> BregmanDRParams:
    """The timed solve's parameters. ``epsilon_primal`` = tol gates the
    feasible-gap stop. ``epsilon_dual`` and ``epsilon_gap`` are +inf; a solver
    given a ``gap_stop_fn`` never reads them."""
    tol = float(spec.tolerance)
    return BregmanDRParams(
        eta=eta,
        max_iters=spec.iter_cap,
        epsilon_primal=tol,
        epsilon_dual=float("inf"),
        epsilon_gap=float("inf"),
        check_every=int(check_every),
        verbosity=False,
    )


def delivered_certificate(cache: Dict[str, Any], iters: int,
                          compute: Callable[[], Tuple[torch.Tensor, torch.Tensor, float]],
                          ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """``(α′, β′, g_feas)`` of the final iterate: the certificate the last
    ``gap_stop_fn`` call left in ``cache`` when it was computed at iteration
    ``iters`` (always so after a feasible-gap stop), else ``compute()``."""
    if cache.get("iters") == iters:
        return cache["alpha_p"], cache["beta_p"], cache["g_feas"]
    return compute()


def assert_dual_feasible(r_dinf_final: float, alpha_p: torch.Tensor,
                         beta_p: torch.Tensor, *,
                         defect: str = "the delivered dual is not exactly feasible; "
                                       "c-transform defect.") -> float:
    """Raise AssertionError if r_Dinf of the delivered (α′, β′) exceeds the
    rounding bound 1e-12·(1 + max|α′| + max|β′|); return the bound. The
    c-transform gives α′_i + β′_j ≤ C_ij up to rounding."""
    r_dinf_bound = 1e-12 * (1.0 + float(alpha_p.abs().max().item())
                            + float(beta_p.abs().max().item()))
    if r_dinf_final > r_dinf_bound:
        raise AssertionError(
            f"r_Dinf(α', β') = {r_dinf_final!r} exceeds the rounding bound "
            f"{r_dinf_bound!r}: {defect}")
    return r_dinf_bound


def native_stop_label(tol: float) -> str:
    return f"production_feasible_gap_{tol:g}_absolute_abs"


def certificate_extras(*, r_dinf_final: float, r_dinf_bound: float,
                       g_feas: float, dual_infeas_abs: float) -> Dict[str, Any]:
    """The ``algo_extras`` fields of the delivered (X, α′, β′) and its certificate."""
    return {
        "ctransform_variant": CTRANSFORM_VARIANT,
        "r_Dinf_final": r_dinf_final,          # 0 up to rounding, by the c-transform
        "r_Dinf_machine_bound": r_dinf_bound,
        "g_feas": g_feas,
        # g_feas certifies suboptimality only while g_feas ≥ 0. X meets the
        # marginals only to within r_P, so once g_feas is of that order its sign
        # can flip. The stop compares |g_feas|, so such a run still stops; this
        # flag marks its certificate void.
        "certificate_valid": bool(g_feas >= 0.0),
        "dual_infeas_abs": dual_infeas_abs,    # r_D = ‖(α′⊕β′ − C)₊‖_F
    }
