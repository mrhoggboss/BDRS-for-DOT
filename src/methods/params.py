"""The parameters for BDRS, and the Normalize step's marginal floor."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def _as_f64(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float64)
    return torch.as_tensor(x, dtype=torch.float64)


MARGINAL_EPS = 1e-12 # δ_m, the default of BregmanDRParams.eps_mass


def floor_marginal(v: torch.Tensor, delta: float = MARGINAL_EPS) -> torch.Tensor:
    """The floored marginal ṽ = (v + δ·1) / (1 + kδ) of a probability vector v
    of length k, as a flat fp64 vector."""
    v = torch.clamp_min(_as_f64(v).reshape(-1), 0.0)
    k = int(v.numel())
    s = v.sum()
    if delta != 0.0:
        return (v + float(delta)) / (s + k * float(delta))
    return v / torch.clamp_min(s, 1e-300)


def _check_marginals(
    r_orig: torch.Tensor, c_orig: torch.Tensor,
    check_r: Optional[torch.Tensor], check_c: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The flat (r, c) every check and delivered metric uses: the originals, or
    ``check_r``/``check_c`` when given (tests, and the Point Cloud CUDA entry's
    torch fallback)."""
    r_chk = r_orig.reshape(-1) if check_r is None else _as_f64(check_r).reshape(-1)
    c_chk = c_orig.reshape(-1) if check_c is None else _as_f64(check_c).reshape(-1)
    return r_chk, c_chk


def _check_dense_shapes(C: torch.Tensor, r: torch.Tensor, c: torch.Tensor) -> None:
    """Raise ValueError unless C is (m, n), r is (m,) and c is (n,)."""
    if C.ndim != 2:
        raise ValueError(f"C must be 2-D (m,n). Got shape {tuple(C.shape)}")
    m, n = C.shape
    if tuple(r.shape) != (m,) or tuple(c.shape) != (n,):
        raise ValueError(
            f"r,c must have shape (m,)=({m},), (n,)=({n},) to match C ({m},{n}). "
            f"Got {tuple(r.shape)}, {tuple(c.shape)}")


def _nan_t(device) -> torch.Tensor:
    """A 0-dim fp64 NaN, reported for r_D when it is not computed (an O(mn) pass)."""
    return torch.full((), float("nan"), dtype=torch.float64, device=device)


# The termination rules of the solvers with a check (their ``stop`` argument),
# tested every T_chk iterations:
#   "feasible_gap" (the default): once ‖X1 − r‖₂ < epsilon_primal, the
#       c-transformed pair (α′, β′) and its gap g_feas = ⟨C,X⟩ − (rᵀα′ + cᵀβ′)
#       are formed, and the solve stops when |g_feas| < gap_stop_tol, which
#       defaults to epsilon_primal;
#   "kkt_triple": the solve stops when r_P, r_D and |g| of the solver's own
#       (α, β) are below epsilon_primal, epsilon_dual and epsilon_gap.
STOP_RULES = ("feasible_gap", "kkt_triple")


def _resolve_stop(params: "BregmanDRParams", stop: str, gap_stop_fn, traj_callback,
                  gap_stop_tol: Optional[float]) -> Tuple[bool, float]:
    """Check a solver's stop arguments; return ``(build_default, gap_stop_tol)``.

    ``build_default`` is True when the solver must build the feasible-gap stop
    of its own format: ``stop="feasible_gap"`` with neither ``gap_stop_fn``
    nor ``traj_callback``. A ``gap_stop_tol`` of None becomes
    ``params.epsilon_primal``. Raises ValueError for an unknown ``stop``, for
    ``gap_stop_fn`` together with ``traj_callback`` (trajectory mode runs the
    full check at every batch and stops on the KKT triple), and for
    ``gap_stop_fn`` with ``stop="kkt_triple"``.
    """
    if stop not in STOP_RULES:
        raise ValueError(f"unknown stop {stop!r}; valid: {list(STOP_RULES)}")
    if gap_stop_fn is not None and traj_callback is not None:
        raise ValueError(
            "gap_stop_fn and traj_callback are mutually exclusive: trajectory mode "
            "runs the full KKT check at every batch and stops on the KKT triple, "
            "whereas the feasible-gap stop runs behind the primal-proxy gate.")
    if gap_stop_fn is not None and stop == "kkt_triple":
        raise ValueError(
            "gap_stop_fn is the feasible-gap stop; stop='kkt_triple' takes none.")
    tol = float(params.epsilon_primal) if gap_stop_tol is None else float(gap_stop_tol)
    build_default = (stop == "feasible_gap" and gap_stop_fn is None
                     and traj_callback is None)
    return build_default, tol


@dataclass(frozen=True)
class BregmanDRParams:
    """Step size, iteration budget, tolerances and marginal floor of a BDRS solve."""

    eta: float                       # η; the log-kernel is L = −C/η
    max_iters: int = 1_000_000       # K_max

    # tol of the default stop (``stop="feasible_gap"``, see STOP_RULES): it
    # ends at the first check where ‖X1 − r‖₂ < tol and |g_feas| < tol (the
    # solvers' ``gap_stop_tol`` defaults to tol), g_feas the gap of the
    # c-transformed pair (α′, β′).
    epsilon_primal: float = 1e-4
    # Read only by the KKT-triple stop (``stop="kkt_triple"``) and trajectory
    # mode, which end at the first check where r_P = ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂,
    # r_D and |g| of the solver's own (α, β) = (η·log u, η·log v) are below
    # epsilon_primal, epsilon_dual and epsilon_gap.
    epsilon_dual: float = 1e-4       # r_D = ‖(α⊕β − C)₊‖_F
    epsilon_gap: float = 1e-4        # |g| = |⟨C,X⟩ − (rᵀα + cᵀβ)|

    # Progress printing of the torch implementations of the Point Cloud and General solvers.
    verbosity: bool = False
    log_every: int = 50

    # δ_m. 0 normalizes without flooring; a zero entry then gives log r̃ = −∞.
    eps_mass: float = MARGINAL_EPS

    check_every: int = 10            # T_chk; values below 1 act as 1
