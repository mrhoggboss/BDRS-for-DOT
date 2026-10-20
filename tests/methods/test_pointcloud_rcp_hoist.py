"""The hoisted reciprocal of C_max in the Point Cloud kernels gives bit-identical iterates to the division.

``pointcloud_div_hoisted`` in ``bdrs_kernels.cu`` replaces the per-term division
κ / C_max by a reciprocal computed once and a Markstein correction, which rounds
to the bits of the IEEE division. ``BDRS_PC_RCP_HOIST=0`` selects the division.
The switch is read when a session captures its graph, so two solves in one
process can run the two variants.
"""
import pytest

torch = pytest.importorskip("torch")

from tests.helpers import requires_cuda

pytestmark = requires_cuda


def _cloud(m, n, d, seed):
    g = torch.Generator().manual_seed(seed)
    P = torch.rand(m, d, generator=g, dtype=torch.float64)
    Q = torch.rand(n, d, generator=g, dtype=torch.float64) * 0.8 + 0.1
    r = torch.rand(m, generator=g, dtype=torch.float64) + 0.1
    c = torch.rand(n, generator=g, dtype=torch.float64) + 0.1
    return P.cuda(), Q.cuda(), (r / r.sum()).cuda(), (c / c.sum()).cuda()


def _grid(ng, seed):
    from src.methods import geo_cost
    P = geo_cost.grid_coords(ng, device="cuda")
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(ng * ng, generator=g, dtype=torch.float64) + 0.05
    c = torch.rand(ng * ng, generator=g, dtype=torch.float64) + 0.05
    return P, P.clone(), (r / r.sum()).cuda(), (c / c.sum()).cuda()


def _state(P, Q, r, c, cost, iters, hoist, monkeypatch):
    from src.methods.factored_cuda import bregman_dr_solve_pointcloud_cuda
    from src.methods.params import BregmanDRParams
    monkeypatch.setenv("BDRS_PC_RCP_HOIST", "1" if hoist else "0")
    params = BregmanDRParams(eta=0.1, max_iters=iters, epsilon_primal=0.0,
                             epsilon_dual=0.0, epsilon_gap=0.0, check_every=iters)
    _, info = bregman_dr_solve_pointcloud_cuda(P, Q, r, c, params, cost_type=cost,
                                               c_max=1.0, normalize_cost=True,
                                               return_plan=False)
    return {k: info[k].clone() for k in ("A", "B", "log_v")}, info


@pytest.mark.parametrize("cost", ["l1", "sql2", "l2", "linf"])
@pytest.mark.parametrize("d", [2, 3])
def test_hoisted_reciprocal_is_bit_identical_on_clouds(cost, d, monkeypatch):
    P, Q, r, c = _cloud(257, 300, d, seed=11 + d)
    s0, i0 = _state(P, Q, r, c, cost, 20000, hoist=False, monkeypatch=monkeypatch)
    s1, i1 = _state(P, Q, r, c, cost, 20000, hoist=True, monkeypatch=monkeypatch)
    assert i0.get("cost_scale") == i1.get("cost_scale")
    for k in s0:
        assert torch.equal(s0[k], s1[k]), (cost, d, k)


@pytest.mark.parametrize("cost", ["l1", "sql2", "l2", "linf"])
def test_hoisted_reciprocal_is_bit_identical_on_a_lifted_grid(cost, monkeypatch):
    P, Q, r, c = _grid(32, seed=5)
    s0, _ = _state(P, Q, r, c, cost, 20000, hoist=False, monkeypatch=monkeypatch)
    s1, _ = _state(P, Q, r, c, cost, 20000, hoist=True, monkeypatch=monkeypatch)
    for k in s0:
        assert torch.equal(s0[k], s1[k]), (cost, k)
