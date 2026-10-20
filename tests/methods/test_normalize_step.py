"""The Normalize step and the delivery rules of the torch BDRS solvers and of
the Point Cloud CUDA entry's torch fallback.

Every solver receives the original marginals (r, c), exact zeros allowed,
floors them to (r̃, c̃) with δ_m for the iteration only, and measures every
gate and delivered metric against (r, c). The Point Cloud solvers divide the
cost by its exact pairwise maximum C_max, recomputed on every solve, which is
a bit-exact no-op on a max-1 cost.
"""

import numpy as np
import pytest
import torch

from src.methods import geo_cost
from src.methods.factored_cuda import bregman_dr_solve_pointcloud_cuda
from src.methods.dual_cleanup import make_dense_gap_stop_fn, make_pointcloud_gap_stop_fn
from src.methods.factored_torch import (
    bregman_dr_solve_general_factored,
    bregman_dr_solve_pointcloud,
)
from src.methods.grid_torch import (
    bregman_dr_solve_factored_logdomain,
    bregman_dr_solve_grid_rect_logdomain,
)
from src.methods.params import MARGINAL_EPS, BregmanDRParams, floor_marginal
from tests.methods.grid_reference import grid_cost

CPU = torch.device("cpu")


def _zero_mass_marginals(N: int, seed: int):
    """Probability vectors with exact zeros."""
    rng = np.random.default_rng(seed)
    r = rng.dirichlet(np.ones(N) * 2.0)
    c = rng.dirichlet(np.ones(N) * 2.0)
    r[: N // 4] = 0.0
    c[N // 2: N // 2 + N // 4] = 0.0
    r, c = r / r.sum(), c / c.sum()
    return torch.from_numpy(r), torch.from_numpy(c)


def _dense_dual_res(C, log_u, log_v, eta):
    viol = (log_u[:, None] + log_v[None, :] - C / eta).clamp_min(0.0)
    return float(eta * viol.norm())


def _xres(X, r, c) -> float:
    return float((X.sum(1) - r).norm(2) + (X.sum(0) - c).norm(2))


def _run_dense(kind, C, r, c, tol=1e-5, **kw):
    """One solve with the feasible-gap stop built on the original marginals;
    ``kind`` is "general" for the General solver on the stored C, or
    "pointcloud" for the Point Cloud solver reading C through ``cost_fn``.
    Returns (X, info, cache, eta)."""
    eta = 0.6 * float(C.mean())
    params = BregmanDRParams(eta=eta, max_iters=50_000, check_every=25,
                             epsilon_primal=tol, epsilon_dual=float("inf"),
                             epsilon_gap=float("inf"))
    cache: dict = {}
    if kind == "pointcloud":
        cost_fn = lambda i0, i1: C[i0:i1]
        stop = make_pointcloud_gap_stop_fn(cost_fn, r, c, cache=cache)
        X, info = bregman_dr_solve_pointcloud(
            r, c, params, cost_fn=cost_fn, gap_stop_fn=stop, gap_stop_tol=tol, **kw)
    elif kind == "general":
        stop = make_dense_gap_stop_fn(C, r, c, cache=cache)
        X, info = bregman_dr_solve_general_factored(
            C, r, c, params, gap_stop_fn=stop, gap_stop_tol=tol, **kw)
    else:
        raise ValueError(kind)
    return X, info, cache, eta


def test_floor_marginal_formula_and_properties():
    N = 40
    r, _ = _zero_mass_marginals(N, seed=1)
    d = MARGINAL_EPS
    rf = floor_marginal(r, d)
    # (r + δ_m)/(1 + N·δ_m) on a probability vector, up to the rounding of Σr.
    expected = (r + d) / (1.0 + N * d)
    assert torch.allclose(rf, expected, rtol=1e-14, atol=0.0)
    assert float(rf.min()) > 0.0
    assert abs(float(rf.sum()) - 1.0) < 1e-12
    assert float((rf - r).norm(2)) <= N * d            # the floor bias bound
    # δ_m = 0: mass normalization only, zeros kept.
    r0 = floor_marginal(r, 0.0)
    assert torch.equal(r0, r / r.sum())
    # An input of any total mass is normalized by the same expression.
    raw = 3.0 * r
    assert abs(float(floor_marginal(raw, d).sum()) - 1.0) < 1e-12
    assert BregmanDRParams(eta=1.0).eps_mass == MARGINAL_EPS == 1e-12


# ---- The General and Point Cloud torch solvers on the same stored cost -------

@pytest.mark.parametrize("kind", ["general", "pointcloud"])
def test_zero_mass_originals_floored_internally_delivery_against_originals(kind):
    ng = 8
    C = grid_cost(ng, "sql2")
    r, c = _zero_mass_marginals(ng * ng, seed=3)
    assert float(r.min()) == 0.0 and float(c.min()) == 0.0

    X, info, cache, eta = _run_dense(kind, C, r, c)
    assert info["native_termination"] == "feasible_gap_below_tol"
    assert info["marginal_floor"] == MARGINAL_EPS
    assert bool(torch.isfinite(info["log_u"]).all()) and bool(torch.isfinite(info["log_v"]).all())

    assert abs(float(info["primal_res_L2"]) - _xres(X, r, c)) <= 1e-12
    assert float(info["primal_res_L2"]) < 1e-5 + (ng * ng) * MARGINAL_EPS
    # The certificate uses (r, c) too: weak duality, rᵀα′ + cᵀβ′ ≤ ⟨C, X⟩.
    d_obj = float((r * cache["alpha_p"]).sum() + (c * cache["beta_p"]).sum())
    assert d_obj <= float(info["objective"]) + 1e-12
    assert bool(torch.isnan(info["dual_res_F"]))
    assert info["native_dual_residual"] is False
    assert info["full_kkt_checks"] == 0


@pytest.mark.parametrize("kind", ["general", "pointcloud"])
def test_row_only_proxy_equals_row_residual_and_full_at_delivery(kind):
    """The gate is the row residual; the delivered r_P adds the column term,
    which is at most the floor bias ‖c̃ − c‖₂ plus rounding because the
    v-update makes Xᵀ1 = c̃."""
    ng = 8
    C = grid_cost(ng, "l1")
    r, c = _zero_mass_marginals(ng * ng, seed=5)
    X, info, cache, eta = _run_dense(kind, C, r, c)
    assert info["native_termination"] == "feasible_gap_below_tol"

    proxy = float(info["primal_res_proxy_final"])
    full = float(info["primal_res_L2"])
    row = float((X.sum(1) - r).norm(2))
    col = float((X.sum(0) - c).norm(2))
    col_bias = float((floor_marginal(c) - c).norm(2))
    assert abs(proxy - row) <= 1e-12
    assert abs(full - (row + col)) <= 1e-12
    assert full >= proxy - 1e-15
    assert full - proxy <= col_bias + 1e-12
    assert col <= col_bias + 1e-12


@pytest.mark.parametrize("kind", ["general", "pointcloud"])
def test_check_seams_override_default_originals(kind):
    ng = 8
    C = grid_cost(ng, "sql2")
    r, c = _zero_mass_marginals(ng * ng, seed=7)
    rf, cf = floor_marginal(r), floor_marginal(c)
    X, info, _, _ = _run_dense(kind, C, r, c, check_r=rf, check_c=cf)
    # Against (r̃, c̃) the column term is rounding only, since Xᵀ1 = c̃.
    assert abs(float(info["primal_res_L2"]) - _xres(X, rf, cf)) <= 1e-12
    assert float((X.sum(0) - cf).norm(2)) <= 1e-12


@pytest.mark.parametrize("kind", ["general", "pointcloud"])
def test_native_dual_residual_opt_in(kind):
    ng = 8
    C = grid_cost(ng, "sql2")
    r, c = _zero_mass_marginals(ng * ng, seed=9)
    X0, info0, _, eta = _run_dense(kind, C, r, c)
    X1, info1, _, _ = _run_dense(kind, C, r, c, native_dual_residual=True)
    # The opt-in adds r_D at delivery and leaves the iterates unchanged.
    assert int(info0["iters"]) == int(info1["iters"])
    assert torch.equal(X0, X1)
    assert bool(torch.isnan(info0["dual_res_F"])) and info0["full_kkt_checks"] == 0
    assert info1["native_dual_residual"] is True and info1["full_kkt_checks"] == 1
    ref = _dense_dual_res(C, info1["log_u"], info1["log_v"], eta)
    assert float(info1["dual_res_F"]) == pytest.approx(ref, rel=1e-10, abs=1e-14)


def test_kkt_triple_stop_keeps_r_D_in_the_loop():
    """The KKT-triple stop evaluates r_D in the loop, as part of its test, and
    delivers the final check's values; a solve that ends at K_max after its
    last check delivers r_D only when opted in."""
    ng = 6
    C = grid_cost(ng, "sql2")
    r, c = _zero_mass_marginals(ng * ng, seed=11)
    eta = 0.6 * float(C.mean())
    p_stop = BregmanDRParams(eta=eta, max_iters=50_000, check_every=20,
                             epsilon_primal=1e-4, epsilon_dual=1e-2, epsilon_gap=1e-3)
    X, info = bregman_dr_solve_general_factored(C, r, c, p_stop, stop="kkt_triple")
    assert info["native_termination"] == "kkt_below_tol"
    assert bool(torch.isfinite(info["dual_res_F"])) and info["full_kkt_checks"] >= 1
    assert abs(float(info["primal_res_L2"]) - _xres(X, r, c)) <= 1e-12

    p_cap = BregmanDRParams(eta=eta, max_iters=30, check_every=7,
                            epsilon_primal=1e-4, epsilon_dual=1e-2, epsilon_gap=1e-3)
    X, info = bregman_dr_solve_general_factored(C, r, c, p_cap, stop="kkt_triple")
    assert info["native_termination"] == "max_iters"
    assert bool(torch.isnan(info["dual_res_F"]))
    assert abs(float(info["primal_res_L2"]) - _xres(X, r, c)) <= 1e-12
    _, info_d = bregman_dr_solve_general_factored(
        C, r, c, p_cap, stop="kkt_triple", native_dual_residual=True)
    assert float(info_d["dual_res_F"]) == pytest.approx(
        _dense_dual_res(C, info_d["log_u"], info_d["log_v"], eta), rel=1e-10)


def test_grid_reference_solvers_floor_internally():
    ng = 8
    r, c = _zero_mass_marginals(ng * ng, seed=13)
    params = BregmanDRParams(eta=0.05, max_iters=25, check_every=10**9)
    X_sq, i_sq = bregman_dr_solve_factored_logdomain(
        r, c, params, grid_size=ng, cost_type="sql2")
    X_rt, i_rt = bregman_dr_solve_grid_rect_logdomain(
        r, c, params, grid_shape=(ng, ng), cost_type="sql2")
    for X, info in ((X_sq, i_sq), (X_rt, i_rt)):
        assert bool(torch.isfinite(X).all())
        assert abs(float(info["primal_res_L2"]) - _xres(X, r, c)) <= 1e-12
        assert bool(torch.isnan(info["dual_res_F"]))
        assert info["marginal_floor"] == MARGINAL_EPS
    assert torch.equal(X_sq, X_rt)
    rf, cf = floor_marginal(r), floor_marginal(c)
    X_s, i_s = bregman_dr_solve_grid_rect_logdomain(
        r, c, params, grid_shape=(ng, ng), cost_type="sql2", check_r=rf, check_c=cf)
    assert abs(float(i_s["primal_res_L2"]) - _xres(X_s, rf, cf)) <= 1e-12
    # Opt-in r_D on an h × w lattice with l1 rows and sql2 columns, against the
    # dense formula with C_max = (h − 1) + (w − 1)².
    h, w = 5, 7
    r2, c2 = _zero_mass_marginals(h * w, seed=17)
    X2, i2 = bregman_dr_solve_grid_rect_logdomain(
        r2, c2, params, grid_shape=(h, w), cost_type=("l1", "sql2"),
        native_dual_residual=True)
    rr = torch.arange(h, dtype=torch.float64).repeat_interleave(w)
    cc = torch.arange(w, dtype=torch.float64).repeat(h)
    C2 = ((rr[:, None] - rr[None, :]).abs() + (cc[:, None] - cc[None, :]) ** 2) / (
        (h - 1) + (w - 1) ** 2)
    assert float(i2["dual_res_F"]) == pytest.approx(
        _dense_dual_res(C2, i2["log_u"], i2["log_v"], params.eta), rel=1e-10)


# ---- Exact C_max in the Point Cloud solvers ---------------------------------

def _clouds(m, n, d, seed):
    rng = np.random.default_rng(seed)
    P = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float64)
    Q = torch.tensor(rng.standard_normal((n, d)) * 1.3 + 0.2, dtype=torch.float64)
    r = torch.tensor(rng.dirichlet(np.ones(m)), dtype=torch.float64)
    c = torch.tensor(rng.dirichlet(np.ones(n)), dtype=torch.float64)
    return P, Q, r, c


@pytest.mark.parametrize("cost_type", ["l1", "sql2", "l2", "linf"])
def test_exact_c_max_is_the_brute_force_pairwise_max(cost_type):
    P, Q, _, _ = _clouds(37, 53, 3, seed=21)
    raw = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=1.0)
    C = geo_cost.materialize_cost(raw, 37, 53, device=CPU)
    exact = geo_cost.exact_c_max(P, Q, cost_type)
    assert exact == float(C.max())
    assert exact == geo_cost.exact_c_max(P, Q, cost_type, row_chunk=5)  # tiling-independent
    assert geo_cost.exact_cost_fn_max(raw, 37) == exact
    assert geo_cost.exact_cost_fn_max(raw, 37, row_chunk=4) == exact
    # Rounded division by a positive constant is monotone: max(κ/b) = max(κ)/b exactly.
    bbox = 1.7 * exact
    scaled = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=bbox)
    assert geo_cost.exact_cost_fn_max(scaled, 37) == exact / bbox
    assert geo_cost.exact_cost_fn_max(scaled, 37) == float(
        geo_cost.materialize_cost(scaled, 37, 53, device=CPU).max())
    unit = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=exact)
    assert geo_cost.exact_cost_fn_max(unit, 37) == 1.0
    assert geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=None)(0, 37).max().item() == 1.0


@pytest.mark.parametrize("cost_type", ["sql2", "l2"])
def test_pointcloud_solver_normalizes_by_exact_cmax(cost_type):
    m, n, d = 40, 32, 3
    P, Q, r, c = _clouds(m, n, d, seed=23)
    exact = geo_cost.exact_c_max(P, Q, cost_type)
    params = BregmanDRParams(eta=0.15, max_iters=15, check_every=10**9)

    raw = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=1.0)
    unit = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=exact)

    X_raw, i_raw = bregman_dr_solve_pointcloud(r, c, params, cost_fn=raw)
    X_unit, i_unit = bregman_dr_solve_pointcloud(r, c, params, cost_fn=unit)
    X_off, i_off = bregman_dr_solve_pointcloud(r, c, params, cost_fn=unit,
                                               normalize_cost=False)
    assert i_raw["cost_scale"] == exact
    # raw = κ / 1.0 = κ, so the solver's raw / C_max and the max-1 cost κ / C_max
    # are the same rounded quotient.
    assert torch.equal(i_raw["log_u"], i_unit["log_u"])
    assert torch.equal(i_raw["log_v"], i_unit["log_v"])
    assert torch.equal(X_raw, X_unit)
    assert torch.equal(i_raw["objective"], i_unit["objective"])
    # On a max-1 cost the Normalize step is a bit-exact no-op.
    assert i_unit["cost_scale"] == 1.0 and i_off["cost_scale"] == 1.0
    assert torch.equal(X_unit, X_off) and torch.equal(i_unit["objective"], i_off["objective"])
    # Outputs are on the normalized scale: objective × cost_scale = ⟨κ, X⟩.
    C_raw = geo_cost.materialize_cost(raw, m, n, device=CPU)
    assert float(i_raw["objective"]) * i_raw["cost_scale"] == pytest.approx(
        float((C_raw * X_raw).sum()), rel=1e-12)

    # A bounding-box c_max is corrected (cost_scale = C_max / bbox); the cost is
    # then divided twice, so the solve agrees with the exact one up to rounding.
    bbox = 1.7 * exact
    boxed = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=bbox)
    X_box, i_box = bregman_dr_solve_pointcloud(r, c, params, cost_fn=boxed)
    assert i_box["cost_scale"] == exact / bbox
    assert torch.allclose(X_box, X_unit, rtol=1e-11, atol=1e-14)
    assert torch.allclose(i_box["log_u"], i_unit["log_u"], rtol=1e-11, atol=1e-12)


def test_pointcloud_cuda_entry_cpu_fallback_exact_cmax(monkeypatch):
    """The CUDA entry computes the exact C_max of the clouds on every solve and
    passes the normalized cost to its torch fallback."""
    monkeypatch.setenv("BDRS_GEO_KERNELS", "0")
    m, n, d = 36, 30, 3
    P, Q, r, c = _clouds(m, n, d, seed=29)
    r = r.clone(); r[:5] = 0.0; r = r / r.sum()
    cost_type = "sql2"
    exact = geo_cost.exact_c_max(P, Q, cost_type)
    params = BregmanDRParams(eta=0.15, max_iters=20, check_every=10**9)

    # (a) c_max exact: no rescaling, bit-identical to the torch solver.
    X_e, i_e = bregman_dr_solve_pointcloud_cuda(P, Q, r, c, params,
                                                cost_type=cost_type, c_max=exact)
    unit = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=exact)
    X_t, i_t = bregman_dr_solve_pointcloud(r, c, params, cost_fn=unit)
    assert i_e["cost_scale"] == 1.0
    assert torch.equal(X_e, X_t) and torch.equal(i_e["log_u"], i_t["log_u"])
    assert abs(float(i_e["primal_res_L2"]) - _xres(X_e, r, c)) <= 1e-12
    assert bool(torch.isnan(i_e["dual_res_F"]))
    assert i_e["marginal_floor"] == MARGINAL_EPS

    # (b) c_max a bound above C_max: cost_scale = C_max / c_max, iterating on κ / C_max.
    bbox = 2.5 * exact
    X_b, i_b = bregman_dr_solve_pointcloud_cuda(P, Q, r, c, params,
                                                cost_type=cost_type, c_max=bbox)
    assert i_b["cost_scale"] == exact / bbox
    assert torch.allclose(X_b, X_t, rtol=1e-11, atol=1e-14)
    # objective (solver scale) × cost_scale = ⟨κ/bbox, X⟩ (instance scale)
    boxed = geo_cost.materialize_cost(
        geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=bbox), m, n, device=CPU)
    assert float(i_b["objective"]) * i_b["cost_scale"] == pytest.approx(
        float((boxed * X_b).sum()), rel=1e-12)

    # (c) normalize_cost=False iterates on κ / c_max as given.
    X_n, i_n = bregman_dr_solve_pointcloud_cuda(P, Q, r, c, params, cost_type=cost_type,
                                                c_max=bbox, normalize_cost=False)
    assert i_n["cost_scale"] == 1.0
    assert not torch.equal(X_n, X_t)

    # (d) native_dual_residual reaches the fallback.
    _, i_d = bregman_dr_solve_pointcloud_cuda(P, Q, r, c, params, cost_type=cost_type,
                                              c_max=exact, native_dual_residual=True)
    C_unit = geo_cost.materialize_cost(unit, m, n, device=CPU)
    assert float(i_d["dual_res_F"]) == pytest.approx(
        _dense_dual_res(C_unit, i_d["log_u"], i_d["log_v"], params.eta), rel=1e-10)


def test_pointcloud_gap_stop_zero_mass_marginals_certified_against_originals():
    """The feasible-gap stop on an off-grid cloud with zero-mass marginals: the
    certificate, built on the max-1 cost with the original marginals, and the
    delivered metrics agree with a dense evaluation."""
    m, n, d = 48, 40, 2
    P, Q, r, c = _clouds(m, n, d, seed=31)
    r = r.clone(); r[:6] = 0.0; r = r / r.sum()
    cost_type = "l1"
    exact = geo_cost.exact_c_max(P, Q, cost_type)
    cost_fn = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=exact)
    C = geo_cost.materialize_cost(cost_fn, m, n, device=CPU)
    tol = 1e-5
    eta = 0.7 * float(C.mean())
    params = BregmanDRParams(eta=eta, max_iters=50_000, check_every=25,
                             epsilon_primal=tol, epsilon_dual=float("inf"),
                             epsilon_gap=float("inf"))
    cache: dict = {}
    stop = make_pointcloud_gap_stop_fn(cost_fn, r, c, cache=cache)
    X, info = bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn,
                                          gap_stop_fn=stop, gap_stop_tol=tol)
    assert info["native_termination"] == "feasible_gap_below_tol"
    assert info["cost_scale"] == 1.0
    assert cache["iters"] == int(info["iters"])
    obj = float((C * X).sum())
    assert float(info["objective"]) == pytest.approx(obj, rel=1e-12)
    ap, bp = cache["alpha_p"], cache["beta_p"]
    assert float((ap[:, None] + bp[None, :] - C).clamp_min(0.0).max()) <= 1e-12
    g_feas = obj - float((r * ap).sum() + (c * bp).sum())
    assert abs(g_feas) < tol and abs(g_feas - cache["g_feas"]) <= 1e-12
    assert abs(float(info["primal_res_L2"]) - _xres(X, r, c)) <= 1e-12
    assert float(info["primal_res_L2"]) < tol + (m + n) * MARGINAL_EPS
