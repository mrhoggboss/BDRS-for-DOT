"""The Point Cloud solvers against the dense recursion and the Grid solver, and
their feasible-gap stop.

The torch solver reads the cost only through ``cost_fn(i0, i1) -> C[i0:i1, :]``;
the CUDA solver recomputes it from the coordinates. Iterates are compared on
log u, log v, X and ⟨C, X⟩, which do not depend on how log X is split between
ξ (`A`) and ζ (`B`).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.methods.grid_cmax import grid_cmax
from src.methods.grid_cost import _cost_axis, _grid_blocks
from src.methods import dual_cleanup as DC
from src.methods import geo_cost
from src.methods.factored_torch import bregman_dr_solve_pointcloud
from src.methods.params import BregmanDRParams
from tests.helpers import requires_cuda
from tests.methods.naive_ademm import naive_logdomain


def make_grid_cost_fn(
    ng: int, cost_type: str, *, device, dtype: torch.dtype = torch.float64,
):
    """``cost_fn`` for the l1/sql2 grid cost from ``grid_cost``'s per-axis table.

    Its maximum is exactly 1, so the solver's Normalize step leaves it unchanged."""
    return _grid_blocks(_cost_axis(cost_type, ng, device=device, dtype=dtype), ng)


CPU = torch.device("cpu")


def _marginals(N, seed, device=CPU):
    rng = np.random.default_rng(seed)
    r = torch.tensor(np.abs(rng.standard_normal(N)) + 1e-3, dtype=torch.float64, device=device)
    r /= r.sum()
    c = torch.tensor(np.abs(rng.standard_normal(N)) + 1e-3, dtype=torch.float64, device=device)
    c /= c.sum()
    return r, c


def _fixed_k_params(eta, k):
    """Exactly k iterations, with no early stop."""
    return BregmanDRParams(eta=eta, max_iters=k, check_every=10**9,
                           epsilon_primal=0.0, epsilon_dual=0.0, epsilon_gap=0.0)


def _dmax(a, b):
    return float((a - b).abs().max())


@pytest.mark.parametrize("cost_type", ["l1", "sql2"])
@pytest.mark.parametrize("iters", [1, 2, 3, 5, 10])
def test_pointcloud_vs_dense_iterate_identity_cpu(cost_type, iters):
    """Both read the same cost entries, so they differ only in reduction order."""
    ng = 8
    N = ng * ng
    eta = 0.05
    r, c = _marginals(N, seed=1)
    cost_fn = make_grid_cost_fn(ng, cost_type, device=CPU)
    C = geo_cost.materialize_cost(cost_fn, N, N, device=CPU)
    params = _fixed_k_params(eta, iters)

    Xg, ig = bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn)
    Xd, lu, lv = naive_logdomain(C, r, c, eta, iters)

    assert int(ig["iters"]) == iters
    assert torch.allclose(ig["log_u"], lu, rtol=1e-10, atol=1e-12)
    assert torch.allclose(ig["log_v"], lv, rtol=1e-10, atol=1e-12)
    assert torch.allclose(Xg, Xd, rtol=1e-10, atol=1e-14)
    obj_d = float((C * Xd).sum())
    assert abs(float(ig["objective"]) - obj_d) <= 1e-10 * max(1.0, abs(obj_d))


@requires_cuda
@pytest.mark.parametrize("cost_type", ["l1", "sql2"])
@pytest.mark.parametrize("iters", [1, 3, 10])
def test_pointcloud_cuda_vs_torch_and_gridfactored(cost_type, iters):
    from src.methods.factored_cuda import bregman_dr_solve_pointcloud_cuda
    from src.methods.grid_cuda import bregman_dr_solve_grid_cuda
    dev = torch.device("cuda")
    ng = 16
    N = ng * ng
    eta = 0.05
    r, c = _marginals(N, seed=2, device=dev)
    P = geo_cost.grid_coords(ng, device=dev)
    cm = grid_cmax(cost_type, ng)
    cost_fn = geo_cost.make_metric_cost_fn(P, P, cost_type, cm)
    params = _fixed_k_params(eta, iters)

    Xc, ic = bregman_dr_solve_pointcloud_cuda(P, P, r, c, params, cost_type=cost_type, c_max=cm)
    Xt, it = bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn)
    Xf, iff = bregman_dr_solve_grid_cuda(None, r, c, params, cost_type=cost_type,
                                    grid_size=ng)

    # Same cost entries; only the LSE reductions differ.
    assert _dmax(ic["log_u"], it["log_u"]) < 1e-9
    assert _dmax(Xc, Xt) < 1e-10
    # The Grid solver builds k·L from per-axis tables and reduces it in two
    # separable passes, so it rounds differently.
    assert _dmax(ic["log_u"], iff["log_u"]) < 1e-6
    assert _dmax(Xc, Xf) < 1e-6


@pytest.mark.parametrize("cost_type", ["l1", "sql2"])
@pytest.mark.parametrize("variant", ["beta", "alpha", "alternating"])
def test_symbolic_ctransform_matches_storedC(cost_type, variant):
    """Bitwise: both read the same entries, and a minimum does not round."""
    ng = 6
    N = ng * ng
    rng = np.random.default_rng(3)
    alpha = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
    beta = torch.tensor(rng.standard_normal(N), dtype=torch.float64)
    cost_fn = make_grid_cost_fn(ng, cost_type, device=CPU)
    C = geo_cost.materialize_cost(cost_fn, N, N, device=CPU)

    ap_s, bp_s = DC.dual_cleanup_ctransform_symbolic(alpha, beta, cost_fn, N, N, variant=variant)
    ap_c, bp_c = DC.dual_cleanup_ctransform_storedC(alpha, beta, C, variant=variant)
    assert torch.equal(ap_s, ap_c)
    assert torch.equal(bp_s, bp_c)


def test_pointcloud_gap_stop_fires_and_certifies():
    ng = 6
    N = ng * ng
    eta = 0.5
    tol = 1e-3
    r, c = _marginals(N, seed=4)
    cost_fn = make_grid_cost_fn(ng, "l1", device=CPU)
    check_every = 5
    params = BregmanDRParams(
        eta=eta, max_iters=20000, epsilon_primal=tol,
        epsilon_dual=float("inf"), epsilon_gap=float("inf"), check_every=check_every)
    cache = {}
    gap_fn = DC.make_pointcloud_gap_stop_fn(cost_fn, r, c, cache=cache)
    X, info = bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn,
                                         gap_stop_fn=gap_fn, gap_stop_tol=tol)

    assert info["native_termination"] == "feasible_gap_below_tol"
    k = int(info["iters"])
    assert k % check_every == 0
    assert info["proxy_passes"] >= 1 and info["sep_obj_checks"] >= 1
    assert info["gap_stop_value"] < tol

    pinf = float((X.sum(1) - r).norm(2) + (X.sum(0) - c).norm(2))
    assert pinf < tol
    assert cache["iters"] == k
    # r_Dinf of the corrected dual (α′, β′).
    assert DC._dual_feas_inf_symbolic(cache["alpha_p"], cache["beta_p"], cost_fn, N, N) <= 1e-12
    C = geo_cost.materialize_cost(cost_fn, N, N, device=CPU)
    true_obj = float((C * X).sum())
    dual_obj = float((r * cache["alpha_p"]).sum() + (c * cache["beta_p"]).sum())
    assert abs((true_obj - dual_obj) - cache["g_feas"]) <= 1e-9
    assert dual_obj <= true_obj + 1e-12
    for key in ("dual_res_F", "gap_abs", "objective", "primal_res_L2"):
        assert key in info


def test_pointcloud_gap_stop_tol_zero_never_fires():
    ng = 4
    N = ng * ng
    r, c = _marginals(N, seed=5)
    cost_fn = make_grid_cost_fn(ng, "l1", device=CPU)
    params = BregmanDRParams(eta=0.5, max_iters=40, check_every=5,
                             epsilon_primal=10.0,
                             epsilon_dual=float("inf"), epsilon_gap=float("inf"))
    gap_fn = DC.make_pointcloud_gap_stop_fn(cost_fn, r, c)
    X, info = bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn,
                                         gap_stop_fn=gap_fn, gap_stop_tol=0.0)
    assert info["native_termination"] == "max_iters"
    assert int(info["iters"]) == 40
    assert info["proxy_passes"] >= 1


def test_pointcloud_gap_stop_traj_callback_mutually_exclusive():
    ng = 3
    N = ng * ng
    r, c = _marginals(N, seed=6)
    cost_fn = make_grid_cost_fn(ng, "l1", device=CPU)
    params = BregmanDRParams(eta=0.5, max_iters=5)
    with pytest.raises(ValueError):
        bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn,
                                   traj_callback=lambda d: None,
                                   gap_stop_fn=lambda info: 0.0)


@pytest.mark.parametrize("cost_type", ["l2", "linf"])
def test_pointcloud_offgrid_square_vs_dense(cost_type):
    """l2 and linf are not separable, so no Grid solver can run them."""
    N, d = 60, 3
    eta = 0.2
    rng = np.random.default_rng(7)
    P = torch.tensor(rng.standard_normal((N, d)), dtype=torch.float64)
    Q = torch.tensor(rng.standard_normal((N, d)), dtype=torch.float64)
    r, c = _marginals(N, seed=8)
    cost_fn = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=1.0)
    C = geo_cost.materialize_cost(cost_fn, N, N, device=CPU)
    params = _fixed_k_params(eta, 30)

    Xg, ig = bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn)
    # The solver divides the cost by its exact maximum (the Normalize step) and
    # the dense recursion does not, so the recursion gets C / max C.
    Xd, lu, lv = naive_logdomain(C / C.max(), r, c, eta, 30)
    assert torch.allclose(ig["log_u"], lu, rtol=1e-9, atol=1e-11)
    assert torch.allclose(Xg, Xd, rtol=1e-9, atol=1e-12)


def test_pointcloud_rectangular_vs_dense():
    """On an m ≠ n cloud the iterate matches the dense recursion on C / max C,
    and the symbolic c-transform of its dual matches the stored-C one."""
    m, n, d = 40, 25, 2
    eta = 0.2
    rng = np.random.default_rng(9)
    P = torch.tensor(rng.standard_normal((m, d)), dtype=torch.float64)
    Q = torch.tensor(rng.standard_normal((n, d)), dtype=torch.float64)
    r, _ = _marginals(m, seed=10)
    _, c = _marginals(n, seed=11)
    cost_fn = geo_cost.make_metric_cost_fn(P, Q, "l2", c_max=1.0)
    C = geo_cost.materialize_cost(cost_fn, m, n, device=CPU)
    params = _fixed_k_params(eta, 25)
    X, info = bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn)
    assert X.shape == (m, n)
    assert info["cost_scale"] == float(C.max())

    C1 = C / C.max()                    # the cost the solver iterates on
    Xd, lu, lv = naive_logdomain(C1, r, c, eta, 25)
    assert torch.allclose(info["log_u"], lu, rtol=1e-9, atol=1e-11)
    assert torch.allclose(info["log_v"], lv, rtol=1e-9, atol=1e-11)
    assert torch.allclose(X, Xd, rtol=1e-9, atol=1e-12)

    # α = η·log u, β = η·log v are duals of the iterated cost C / max C.
    cost_fn1 = lambda i0, i1: cost_fn(i0, i1) / C.max()
    alpha, beta = eta * info["log_u"], eta * info["log_v"]
    ap, bp = DC.dual_cleanup_ctransform_symbolic(alpha, beta, cost_fn1, m, n)
    ap_c, bp_c = DC.dual_cleanup_ctransform_storedC(alpha, beta, C1)
    assert torch.equal(ap, ap_c) and torch.equal(bp, bp_c)
    assert DC._dual_feas_inf_symbolic(ap, bp, cost_fn1, m, n) <= 1e-12


@requires_cuda
def test_pointcloud_cuda_certificate_on_a_lifted_grid():
    """The CUDA solver's gap stop fires on a lifted 32 x 32 grid with a feasible
    corrected dual and g_feas < tol."""
    from src.methods.factored_cuda import bregman_dr_solve_pointcloud_cuda
    dev = torch.device("cuda")
    ng = 32
    N = ng * ng
    eta = 3.0 * float(geo_cost.materialize_cost(
        make_grid_cost_fn(ng, "l1", device=dev), N, N, device=dev).mean())
    r, c = _marginals(N, seed=12, device=dev)
    P = geo_cost.grid_coords(ng, device=dev)
    cm = grid_cmax("l1", ng)
    cost_fn = geo_cost.make_metric_cost_fn(P, P, "l1", cm)
    tol = 1e-4
    cache = {}
    gap_fn = DC.make_pointcloud_gap_stop_fn(cost_fn, r, c, cache=cache)
    params = BregmanDRParams(eta=eta, max_iters=200000, check_every=100,
                             epsilon_primal=tol,
                             epsilon_dual=float("inf"), epsilon_gap=float("inf"))
    X, info = bregman_dr_solve_pointcloud_cuda(
        P, P, r, c, params, cost_type="l1", c_max=cm,
        gap_stop_fn=gap_fn, gap_stop_tol=tol, return_plan=False)
    assert info["native_termination"] == "feasible_gap_below_tol"
    assert DC._dual_feas_inf_symbolic(cache["alpha_p"], cache["beta_p"], cost_fn, N, N) <= 1e-10
    assert cache["g_feas"] < tol
