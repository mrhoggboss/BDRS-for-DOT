"""The rectangular torch Grid solver ``bregman_dr_solve_grid_rect_logdomain``:
h × w lattices with any sides h, w ≥ 2 and per-axis costs φ_r, φ_c."""
import numpy as np
import pytest
import torch

from src.methods.grid_torch import (
    bregman_dr_solve_factored_logdomain,
    bregman_dr_solve_grid_rect_logdomain,
)
from src.methods.params import BregmanDRParams, floor_marginal


def _phi(delta: torch.Tensor, cost_type: str) -> torch.Tensor:
    return delta * delta if cost_type == "sql2" else delta


def _rect_cost(h: int, w: int, ct_r: str, ct_c: str) -> torch.Tensor:
    """The dense (h·w, h·w) cost with max C = 1, row-major."""
    rr = torch.arange(h, dtype=torch.float64)
    cc = torch.arange(w, dtype=torch.float64)
    coords_r = rr.repeat_interleave(w)
    coords_c = cc.repeat(h)
    dr = (coords_r[:, None] - coords_r[None, :]).abs()
    dc = (coords_c[:, None] - coords_c[None, :]).abs()
    c_max = float(_phi(torch.tensor(float(h - 1)), ct_r)
                  + _phi(torch.tensor(float(w - 1)), ct_c))
    return (_phi(dr, ct_r) + _phi(dc, ct_c)) / c_max


def _marginals(N: int, seed: int):
    rng = np.random.default_rng(seed)
    r = torch.from_numpy(rng.dirichlet(np.ones(N) * 2.0))
    c = torch.from_numpy(rng.dirichlet(np.ones(N) * 2.0))
    return r, c


def _ref_dense_logdomain(C, r, c, eta: float, iters: int, eps_mass: float):
    """The BDRS log-domain recursion on the dense C, on floored marginals as the
    solvers iterate; returns (log X, log u, log v)."""
    N = C.shape[0]
    r = floor_marginal(r.to(torch.float64), eps_mass)
    c = floor_marginal(c.to(torch.float64), eps_mass)
    log_r, log_c = r.log(), c.log()
    log_M = -C / eta - float(np.log(N))         # log M^k = log X^k − C/η; X^0 constant
    log_v = torch.zeros(N, dtype=torch.float64)
    log_u = torch.zeros(N, dtype=torch.float64)
    for _ in range(iters):
        log_u = log_r - torch.logsumexp(log_M + log_v[None, :], dim=1)
        log_v = log_c - torch.logsumexp(log_M + log_u[:, None], dim=0)
        log_M = log_M + log_u[:, None] + log_v[None, :] - C / eta
    return log_M + C / eta, log_u, log_v


@pytest.mark.parametrize("h,w,ct_r,ct_c", [
    (5, 7, "sql2", "sql2"),      # rectangular, non-power-of-two
    (6, 4, "l1", "l1"),
    (5, 7, "l1", "sql2"),        # distinct axis costs
])
def test_rect_matches_dense_reference(h, w, ct_r, ct_c):
    N = h * w
    r, c = _marginals(N, seed=20260831)
    C = _rect_cost(h, w, ct_r, ct_c)
    eta = 0.5 * float(C.mean())
    iters = 40
    params = BregmanDRParams(eta=eta, max_iters=iters, check_every=10**9)

    X, info = bregman_dr_solve_grid_rect_logdomain(
        r, c, params, grid_shape=(h, w), cost_type=(ct_r, ct_c))
    log_X_ref, log_u_ref, log_v_ref = _ref_dense_logdomain(
        C, r, c, eta, iters, params.eps_mass)

    assert int(info["iters"]) == iters
    assert (X - log_X_ref.exp()).abs().max().item() <= 1e-12
    assert (info["log_u"] - log_u_ref).abs().max().item() <= 1e-10
    assert (info["log_v"] - log_v_ref).abs().max().item() <= 1e-10
    obj_ref = float((C * log_X_ref.exp()).sum())
    assert abs(float(info["objective"]) - obj_ref) <= 1e-12


def test_square_path_bit_identical():
    ng = 8
    N = ng * ng
    r, c = _marginals(N, seed=20260831)
    params = BregmanDRParams(eta=0.05, max_iters=30, check_every=10**9)

    # native_dual_residual=True so that the comparison covers r_D, NaN by default.
    X_sq, info_sq = bregman_dr_solve_factored_logdomain(
        r, c, params, grid_size=ng, cost_type="sql2", native_dual_residual=True)
    X_rt, info_rt = bregman_dr_solve_grid_rect_logdomain(
        r, c, params, grid_shape=(ng, ng), cost_type="sql2", native_dual_residual=True)

    assert torch.equal(X_sq, X_rt)
    for key in ("log_u", "log_v", "objective", "dual_res_F", "primal_res_L2", "gap_abs"):
        assert torch.equal(info_sq[key], info_rt[key]), key
    assert bool(torch.isfinite(info_sq["dual_res_F"]))

    _, info_def = bregman_dr_solve_factored_logdomain(
        r, c, params, grid_size=ng, cost_type="sql2")
    assert bool(torch.isnan(info_def["dual_res_F"]))
    assert info_def["native_dual_residual"] is False
    for key in ("log_u", "log_v", "objective", "primal_res_L2", "gap_abs"):
        assert torch.equal(info_def[key], info_sq[key]), key


def test_return_plan_false_metrics_agree():
    h, w = 5, 7
    r, c = _marginals(h * w, seed=7)
    params = BregmanDRParams(eta=0.03, max_iters=25, check_every=10**9)
    X, info_full = bregman_dr_solve_grid_rect_logdomain(
        r, c, params, grid_shape=(h, w), cost_type="sql2")
    X_none, info_lean = bregman_dr_solve_grid_rect_logdomain(
        r, c, params, grid_shape=(h, w), cost_type="sql2", return_plan=False)
    assert X_none is None and X is not None
    assert abs(float(info_full["primal_res_L2"]) - float(info_lean["primal_res_L2"])) <= 1e-13
    assert torch.equal(info_full["objective"], info_lean["objective"])
