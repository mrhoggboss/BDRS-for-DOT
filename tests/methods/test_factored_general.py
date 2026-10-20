"""The General torch solver ``bregman_dr_solve_general_factored`` on a stored cost C, on CPU."""

import numpy as np
import pytest
import torch

from src.methods import dual_cleanup as DC
from src.methods.dual_cleanup import make_dense_gap_stop_fn
from src.methods.factored_torch import bregman_dr_solve_general_factored
from src.methods.params import BregmanDRParams
from tests.methods.grid_reference import grid_cost
from tests.methods.naive_ademm import naive_logdomain


def _random_instance(N=36, seed=0):
    rng = np.random.default_rng(seed)
    C = torch.tensor(np.abs(rng.standard_normal((N, N))) + 0.02, dtype=torch.float64)
    C /= C.max()
    r = torch.tensor(np.abs(rng.standard_normal(N)) + 1e-3, dtype=torch.float64)
    r /= r.sum()
    c = torch.tensor(np.abs(rng.standard_normal(N)) + 1e-3, dtype=torch.float64)
    c /= c.sum()
    return C, r, c


def _grid_instance(ng: int, norm: str, seed: int):
    rng = np.random.default_rng(seed)
    N = ng * ng
    r = rng.dirichlet(np.ones(N) * 2.0) + 1e-12
    c = rng.dirichlet(np.ones(N) * 2.0) + 1e-12
    r, c = r / r.sum(), c / c.sum()
    return grid_cost(ng, norm), torch.from_numpy(r), torch.from_numpy(c)


@pytest.mark.parametrize("iters", [1, 2, 7])
def test_matches_naive_recursion(iters):
    C, r, c = _random_instance(N=30, seed=2)
    eta = 0.4
    params = BregmanDRParams(eta=eta, max_iters=iters, check_every=10**9,
                             epsilon_primal=0.0, epsilon_dual=0.0, epsilon_gap=0.0)
    X, info = bregman_dr_solve_general_factored(C, r, c, params)
    X_ref, lu_ref, lv_ref = naive_logdomain(C, r, c, eta, iters)
    assert int(info["iters"]) == iters
    # Equal up to the reassociation of the LSE sums.
    assert torch.allclose(info["log_u"], lu_ref, rtol=1e-10, atol=1e-12)
    assert torch.allclose(info["log_v"], lv_ref, rtol=1e-10, atol=1e-12)
    assert torch.allclose(X, X_ref, rtol=1e-10, atol=1e-14)


def test_gap_stop_fires_and_certifies():
    C, r, c = _random_instance(N=36, seed=4)
    tol = 1e-3
    check_every = 5
    params = BregmanDRParams(
        eta=0.5, max_iters=20000, epsilon_primal=tol,
        epsilon_dual=float("inf"), epsilon_gap=float("inf"),
        check_every=check_every)
    cache = {}
    gap_fn = DC.make_dense_gap_stop_fn(C, r, c, cache=cache)
    X, info = bregman_dr_solve_general_factored(
        C, r, c, params, gap_stop_fn=gap_fn, gap_stop_tol=tol)

    assert info["native_termination"] == "feasible_gap_below_tol"
    k = int(info["iters"])
    assert k % check_every == 0, "the stop must land on a check boundary"
    assert info["gap_stop_value"] < tol
    pinf = float((X.sum(1) - r).norm(2) + (X.sum(0) - c).norm(2))
    assert pinf < tol
    # g_feas is ⟨C, X⟩ of the delivered X minus the dual objective of the
    # cached (α', β').
    assert cache["iters"] == k
    assert DC._dual_feas_F_storedC(cache["alpha_p"], cache["beta_p"], C) <= 1e-12
    true_obj = float((C * X).sum())
    dual_obj = float((r * cache["alpha_p"]).sum() + (c * cache["beta_p"]).sum())
    assert abs((true_obj - dual_obj) - cache["g_feas"]) <= 1e-9
    assert dual_obj <= true_obj + 1e-12          # weak duality


@pytest.mark.parametrize("norm,alpha", [("sql2", 0.7), ("l1", 0.7)])
def test_grid_cost_solves_to_the_exact_lp(norm, alpha):
    C, r, c = _grid_instance(16, norm, seed=20260828)
    eta = alpha * float(C.mean())
    tol = 1e-4
    params = BregmanDRParams(eta=eta, max_iters=20000, check_every=25,
                             epsilon_primal=tol)
    C_before = C.clone()
    X, info = bregman_dr_solve_general_factored(
        C, r, c, params, gap_stop_fn=make_dense_gap_stop_fn(C, r, c),
        gap_stop_tol=tol)
    assert torch.equal(C, C_before)
    assert info["native_termination"] == "feasible_gap_below_tol"
    obj = float((C * X).sum())
    ot = pytest.importorskip("ot")
    p_star = float(ot.emd2(r.numpy(), c.numpy(), C.numpy()))
    assert abs(obj - p_star) <= 10.0 * tol


def test_gap_stop_tol_zero_never_fires():
    C, r, c = _random_instance(N=16, seed=5)
    params = BregmanDRParams(eta=0.5, max_iters=40, check_every=5,
                             epsilon_primal=10.0,   # proxy passes immediately
                             epsilon_dual=float("inf"), epsilon_gap=float("inf"))
    X, info = bregman_dr_solve_general_factored(
        C, r, c, params, gap_stop_fn=DC.make_dense_gap_stop_fn(C, r, c),
        gap_stop_tol=0.0)
    assert info["native_termination"] == "max_iters"
    assert int(info["iters"]) == 40
    assert info["proxy_passes"] >= 1   # the gap stop was evaluated and never fired


def test_gap_stop_and_traj_callback_are_mutually_exclusive():
    C, r, c = _random_instance(N=9, seed=6)
    with pytest.raises(ValueError):
        bregman_dr_solve_general_factored(
            C, r, c, BregmanDRParams(eta=0.5, max_iters=5),
            traj_callback=lambda d: None, gap_stop_fn=lambda info: 0.0)


def test_kkt_triple_stop():
    """With ``stop="kkt_triple"`` the solver stops when r_P, r_D and |g| are
    each below their ``epsilon_*``."""
    C, r, c = _random_instance(N=16, seed=7)
    params = BregmanDRParams(eta=0.5, max_iters=20000, check_every=10,
                             epsilon_primal=1e-2, epsilon_dual=1e-1,
                             epsilon_gap=1e-1)
    X, info = bregman_dr_solve_general_factored(C, r, c, params, stop="kkt_triple")
    assert info["native_termination"] in ("kkt_below_tol", "max_iters")
    if info["native_termination"] == "kkt_below_tol":
        assert float(info["primal_res_L2"]) <= params.epsilon_primal
        assert float(info["dual_res_F"]) <= params.epsilon_dual
        assert float(info["gap_abs"]) <= params.epsilon_gap
