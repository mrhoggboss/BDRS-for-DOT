"""The Grid CUDA solver's separable ⟨C, X⟩ and the work of its feasible-gap stop.

The separable objective is an O(n_g³) log-sum-exp and must equal the dense
⟨C, X⟩ also where exp(k·L) underflows. With ``gap_stop_fn`` the O(n_g⁴) native
r_D is computed only with ``native_dual_residual=True``.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.methods.params import BregmanDRParams
from src.methods.grid_cuda import (
    bregman_dr_solve_grid_cuda,
    _compute_factored_objective,
    _add_factored_kkt,
)
from src.methods.dual_cleanup import _dual_feas_F, feasible_dual_and_gap
from tests.dotmark_load import (
    get_dense_cost_matrix, load_marginal, require_marginals,
)
from tests.helpers import requires_cuda
from src.methods.eta import eta_from_mean, grid_cost_mean

pytestmark = requires_cuda


def _eta(ng, norm):
    return eta_from_mean(grid_cost_mean(ng, norm))


def _factored_state(ng, norm, *, iters=250, seed=0):
    """Run the CUDA Grid solver for exactly ``iters`` iterations on random
    positive marginals; return (r, c, X, info, η)."""
    dev = torch.device("cuda")
    rng = np.random.default_rng(seed + ng + (norm == "sql2") * 97)
    N = ng * ng
    r = np.abs(rng.standard_normal(N)) + 1e-3; r /= r.sum()
    c = np.abs(rng.standard_normal(N)) + 1e-3; c /= c.sum()
    rt = torch.as_tensor(r, dtype=torch.float64, device=dev)
    ct = torch.as_tensor(c, dtype=torch.float64, device=dev)
    eta = _eta(ng, norm)
    params = BregmanDRParams(
        eta=eta, max_iters=iters, check_every=iters + 1,   # one batch; ε = 0 never stops
        epsilon_primal=0.0, epsilon_dual=0.0, epsilon_gap=0.0, verbosity=False,
    )
    X, info = bregman_dr_solve_grid_cuda(
        None, rt, ct, params, cost_type=norm, grid_size=ng,
        return_plan=True,
    )
    return rt, ct, X, info, eta


@pytest.mark.parametrize("norm", ["l1", "sql2"])
@pytest.mark.parametrize("ng", [8, 16, 32, 64])
def test_factored_objective_matches_dense_and_kkt(ng, norm):
    r, c, X, info, eta = _factored_state(ng, norm)
    k = int(info["iters"])
    obj_sep = _compute_factored_objective(info["A"], info["B"], ng, norm, eta, k)

    # (a) the dense ⟨C, X⟩ of the materialized plan.
    C = get_dense_cost_matrix(ng, norm, device=X.device)
    truth = float((C * X).sum().item())
    assert abs(obj_sep - truth) <= 1e-9 * (1.0 + abs(truth)), \
        f"separable {obj_sep!r} vs dense {truth!r} (ng={ng}, {norm})"

    # (b) the objective the delivery pass recorded, which _add_factored_kkt
    # reuses, is the separable objective of the delivered state (A, B, k).
    info2 = dict(info)
    _add_factored_kkt(info2, r, c, ng, norm, eta)
    assert abs(obj_sep - info2["objective"]) <= 1e-9 * (1.0 + abs(info2["objective"]))

    assert np.isfinite(obj_sep) and obj_sep > 0.0


def test_objective_underflow_regime_is_log_domain():
    """Where exp(k·L) underflows in fp64 (k·L = −k/η ≈ −1247 at opposite
    corners here), the separable objective still equals the dense ⟨C, X⟩."""
    ng, norm = 32, "l1"
    r, c, X, info, eta = _factored_state(ng, norm, iters=300)
    k = int(info["iters"])
    # k·L at opposite corners is twice the per-axis minimum k·ℓ = −k/(2η).
    delta = torch.arange(ng, dtype=torch.float64)
    base_ell = -(delta) / (2.0 * (ng - 1) * eta)          # the l1 per-axis table ℓ
    kL_corner = 2.0 * float(k * base_ell.min())
    assert torch.tensor(kL_corner, dtype=torch.float64).exp().item() == 0.0, \
        f"kL_corner={kL_corner} did not underflow in fp64"
    obj_sep = _compute_factored_objective(info["A"], info["B"], ng, norm, eta, k)
    truth = float((get_dense_cost_matrix(ng, norm, device=X.device) * X).sum().item())
    assert abs(obj_sep - truth) <= 1e-9 * (1.0 + abs(truth))


@pytest.mark.parametrize("cls,ng,ia,ib,norm", [
    ("GRFmoderate", 32, 1001, 1002, "l1"),
    ("CauchyDensity", 32, 1001, 1003, "sql2"),
])
def test_gap_stop_path_delivery_without_native_dual(cls, ng, ia, ib, norm):
    """With ``gap_stop_fn`` no check computes the O(n_g⁴) native r_D; the
    delivery pass computes it once only with ``native_dual_residual=True``."""
    require_marginals(cls, ng, ia, ib)
    dev = torch.device("cuda")
    r = load_marginal(cls, ia, ng, device=dev)
    c = load_marginal(cls, ib, ng, device=dev)
    eta = _eta(ng, norm)
    tol = 1e-4
    params = BregmanDRParams(
        eta=eta, max_iters=300_000,
        epsilon_primal=tol, epsilon_dual=float("inf"),
        epsilon_gap=float("inf"), check_every=200, verbosity=False,
    )

    def gap_fn(info):
        _ap, _bp, g_feas = feasible_dual_and_gap(
            info["log_u"], info["log_v"], eta, r, c, ng, norm, info["objective"],
            variant="alternating",
        )
        return g_feas

    _, info = bregman_dr_solve_grid_cuda(
        None, r, c, params, cost_type=norm, grid_size=ng,
        return_plan=False, gap_stop_fn=gap_fn, gap_stop_tol=tol,
    )

    assert info["native_termination"] == "feasible_gap_below_tol"
    assert info["full_kkt_checks"] == 0, f"expected 0, got {info['full_kkt_checks']}"
    assert np.isnan(info["dual_res_F"]) and info["native_dual_residual"] is False
    assert info["sep_obj_checks"] >= 1
    assert "gap_abs" in info and info["gap_abs"] >= 0.0 and info["objective"] > 0.0
    assert info["primal_res_L2"] < tol
    # The delivered r_P adds the column term that the row-only proxy omits.
    assert info["primal_res_L2"] >= info["primal_res_proxy_final"] - 1e-15

    _, info_d = bregman_dr_solve_grid_cuda(
        None, r, c, params, cost_type=norm, grid_size=ng,
        return_plan=False, gap_stop_fn=gap_fn, gap_stop_tol=tol,
        native_dual_residual=True,
    )
    assert info_d["iters"] == info["iters"]
    assert info_d["full_kkt_checks"] == 1 and info_d["native_dual_residual"] is True
    assert np.isfinite(info_d["dual_res_F"]) and info_d["dual_res_F"] >= 0.0

    # The c-transform of the delivered potentials is feasible and certifies the stop.
    ap, bp, g_feas = feasible_dual_and_gap(
        info["log_u"], info["log_v"], eta, r, c, ng, norm, info["objective"],
        variant="alternating",
    )
    assert g_feas < tol
    assert _dual_feas_F(ap, bp, ng, norm) <= 1e-9


def test_kkt_triple_stop_computes_r_D():
    """The KKT-triple stop computes r_D at every check whose proxy passes."""
    ng, norm = 32, "l1"
    dev = torch.device("cuda")
    require_marginals("GRFmoderate", ng, 1001, 1002)
    r = load_marginal("GRFmoderate", 1001, ng, device=dev)
    c = load_marginal("GRFmoderate", 1002, ng, device=dev)
    eta = _eta(ng, norm)
    params = BregmanDRParams(
        eta=eta, max_iters=300_000, epsilon_primal=1e-4, epsilon_dual=1e-3,
        epsilon_gap=1e-4, check_every=200, verbosity=False,
    )
    _, info = bregman_dr_solve_grid_cuda(
        None, r, c, params, cost_type=norm, grid_size=ng,
        return_plan=False, stop="kkt_triple",
    )
    assert info["native_termination"] in ("kkt_below_tol", "max_iters")
    assert "dual_res_F" in info and "gap_abs" in info and "objective" in info
    if info["native_termination"] == "kkt_below_tol":
        # The stopping check computed r_D; its record is delivered as is.
        assert info["full_kkt_checks"] >= 1 and np.isfinite(info["dual_res_F"])
    else:
        # The delivery pass computes r_D only with native_dual_residual=True.
        assert np.isnan(info["dual_res_F"])
    assert info["sep_obj_checks"] == 0   # counts only objectives computed for gap_stop_fn
