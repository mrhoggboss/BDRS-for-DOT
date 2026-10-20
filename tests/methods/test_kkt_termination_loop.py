"""The termination loop of ``grid_cuda.bregman_dr_solve_grid_cuda``, on the CPU.

The real loop runs with a fake extension in place of the CUDA session (the
``_ext_override`` seam) and with the row-only proxy and ``_add_factored_kkt``
patched by stubs, so these tests check the control flow (batch sizes, gates,
termination labels, the delivery pass, which marginals reach which step) and
not the numerics, which the CUDA tests check.
"""

import math
import unittest.mock as mock

import pytest
import torch

from src.methods import grid_cuda
from src.methods.params import BregmanDRParams, MARGINAL_EPS, floor_marginal


class _FakeSession:
    """Stands in for ext.BdrsSession: owns state tensors and counts step calls."""

    def __init__(self, holder, *, r, c, ng, eta, cost_type,
                 init_A, init_B, init_log_v):
        N = ng * ng
        self.holder = holder
        self.r = r
        self.c = c
        self._iters = 0
        logN = math.log(N)
        self.A = torch.full((N,), -logN, dtype=torch.float64)
        self.B = torch.full((N,), -logN, dtype=torch.float64)
        self.log_u = torch.zeros(N, dtype=torch.float64)
        self.log_v = torch.zeros(N, dtype=torch.float64)
        self.step_calls = []
        holder.sessions.append(self)

    def step(self, n_iters):
        self._iters += int(n_iters)
        self.step_calls.append(int(n_iters))
        return self._iters

    @property
    def iters(self):
        return self._iters


class _FakeExt:
    """The extension namespace handed to the loop via _ext_override."""

    def __init__(self):
        self.sessions = []
        self.materialize_calls = []

    def BdrsSession(self, **kwargs):
        return _FakeSession(self, **kwargs)

    def materialize_X(self, *, A, B, ng, eta, cost_type, k_iter):
        self.materialize_calls.append(int(k_iter))
        N = ng * ng
        return torch.zeros((N, N), dtype=torch.float64)


def _params(max_iters, check_every, eps=1e-4):
    return BregmanDRParams(
        eta=1.0, max_iters=max_iters, check_every=check_every,
        epsilon_primal=eps, epsilon_dual=eps, epsilon_gap=eps,
    )


def _marginals(ng):
    N = ng * ng
    r = torch.full((N,), 1.0 / N, dtype=torch.float64)
    c = torch.full((N,), 1.0 / N, dtype=torch.float64)
    return r, c


def _proxy_patch(fn, seen=None):
    """Patch the row-only proxy with a (value, logS) stub driven by k_iter;
    ``seen`` (a list) records the (r, c) the proxy was handed."""
    def _stub(A, B, r, c, grid_size, cost_type, eta, k_iter, *, invariants=None):
        if seen is not None:
            seen.append((r, c))
        return float(fn(k_iter)), torch.zeros(
            (int(grid_size), int(grid_size)), dtype=torch.float64)
    return mock.patch.object(grid_cuda, "_compute_factored_primal_res_ex", _stub)


def _kkt_patch(fn, seen=None):
    """Patch _add_factored_kkt: fn(info) merges the fake triple; calls are
    logged as (iters, native_dual); ``seen`` records the (r, c) handed in."""
    calls = []

    def _stub(info, r, c, grid_size, cost_type, eta, *, invariants=None,
              native_dual=False):
        calls.append((int(info["iters"]), bool(native_dual)))
        if seen is not None:
            seen.append((r, c))
        fn(info)
        if not native_dual:
            info["dual_res_F"] = float("nan")
        obj = info.get("objective", 0.0)
        info["objective"] = (float(obj.item()) if isinstance(obj, torch.Tensor)
                             else float(obj))
        info["native_dual_residual"] = bool(native_dual)
    return mock.patch.object(grid_cuda, "_add_factored_kkt", _stub), calls


def _const_kkt(primal, dual, gap, obj=0.5):
    def _fn(info):
        info["primal_res_L2"] = primal
        info["dual_res_F"] = dual
        info["gap_abs"] = gap
        info.setdefault("objective", obj)
    return _fn


def test_max_iters_no_proxy_passes():
    ng = 8
    ext = _FakeExt()
    r, c = _marginals(ng)
    kkt_patch, kkt_calls = _kkt_patch(_const_kkt(1.0, 1.0, 1.0))
    with _proxy_patch(lambda k: 1.0), kkt_patch:
        X, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=200, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=False, _ext_override=ext)

    assert info["iters"] == 200
    assert info["native_termination"] == "max_iters"
    # One delivery pass, without the native r_D.
    assert kkt_calls == [(200, False)]
    assert info["full_kkt_checks"] == 0
    assert info["native_dual_residual"] is False
    assert math.isnan(info["dual_res_F"])
    assert info["proxy_passes"] == 0
    assert len(ext.sessions) == 1, "exactly one session per solve"
    assert ext.sessions[0].step_calls == [50, 50, 50, 50]
    assert X is None


def test_max_iters_native_dual_opt_in():
    """native_dual_residual=True adds the O(n_g⁴) native r_D to the delivery."""
    ng = 8
    ext = _FakeExt()
    r, c = _marginals(ng)
    kkt_patch, kkt_calls = _kkt_patch(_const_kkt(1.0, 0.25, 1.0))
    with _proxy_patch(lambda k: 1.0), kkt_patch:
        _, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=100, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=False, native_dual_residual=True, _ext_override=ext)
    assert kkt_calls == [(100, True)]
    assert info["full_kkt_checks"] == 1
    assert info["native_dual_residual"] is True
    assert info["dual_res_F"] == 0.25


def test_kkt_below_tol_terminates():
    ng = 8
    ext = _FakeExt()
    r, c = _marginals(ng)
    kkt_patch, kkt_calls = _kkt_patch(_const_kkt(1e-6, 1e-6, 1e-6))
    with _proxy_patch(lambda k: 1e-6), kkt_patch:
        _, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=200, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=False, stop="kkt_triple", _ext_override=ext)
    assert info["iters"] == 50
    assert info["native_termination"] == "kkt_below_tol"
    assert info["proxy_passes"] == 1
    # The KKT-triple stop evaluates r_D in the loop, as part of its test; the
    # final batch's values are delivered as they are, with no delivery pass.
    assert info["full_kkt_checks"] == 1
    assert kkt_calls == [(50, True)]
    assert info["dual_res_F"] == 1e-6


def test_proxy_passes_but_full_kkt_does_not():
    """A passing proxy does not stop the solve while r_D is above epsilon_dual."""
    ng = 8
    ext = _FakeExt()
    r, c = _marginals(ng)
    n_calls = {"n": 0}

    def fake_kkt(info):
        n_calls["n"] += 1
        info["primal_res_L2"] = 3e-5
        info["dual_res_F"] = 1e-3 if n_calls["n"] < 4 else 1e-6
        info["gap_abs"] = 1e-6
        info.setdefault("objective", 0.0)

    kkt_patch, kkt_calls = _kkt_patch(fake_kkt)
    with _proxy_patch(lambda k: 3e-5), kkt_patch:
        _, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=200, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=False, stop="kkt_triple", _ext_override=ext)
    assert info["iters"] == 200
    assert info["native_termination"] == "kkt_below_tol"
    assert info["proxy_passes"] == 4
    assert info["full_kkt_checks"] == 4
    assert kkt_calls == [(50, True), (100, True), (150, True), (200, True)]


def test_gap_stop_path_delivery_without_native_dual_and_objective_is_float():
    ng = 8
    ext = _FakeExt()
    r, c = _marginals(ng)
    kkt_patch, kkt_calls = _kkt_patch(_const_kkt(1e-7, 0.1, 0.1))
    obj_stub = mock.patch.object(
        grid_cuda, "_compute_factored_objective_t",
        lambda *a, **kw: torch.tensor(0.5, dtype=torch.float64))
    gap_calls = []

    def gap_fn(info):
        gap_calls.append(int(info["iters"]))
        assert isinstance(info["objective"], torch.Tensor), \
            "the in-loop objective is handed to gap_stop_fn as a tensor"
        return 1.0 if info["iters"] < 150 else 1e-9

    with _proxy_patch(lambda k: 1e-7 if k >= 100 else 1.0), kkt_patch, obj_stub:
        _, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=10**6, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=False, gap_stop_fn=gap_fn, gap_stop_tol=1e-4,
            _ext_override=ext)

    assert info["native_termination"] == "feasible_gap_below_tol"
    assert info["iters"] == 150
    assert gap_calls == [100, 150]
    assert info["sep_obj_checks"] == 2
    assert info["proxy_passes"] == 2
    # One delivery pass, at the final state, without the O(n_g⁴) native r_D:
    # the feasible-gap stop certifies the corrected dual (α′, β′) instead.
    assert kkt_calls == [(150, False)]
    assert info["full_kkt_checks"] == 0
    assert math.isnan(info["dual_res_F"])
    assert info["native_dual_residual"] is False
    assert isinstance(info["objective"], float)
    assert info["gap_stop_value"] == 1e-9


def test_gap_stop_path_native_dual_opt_in():
    ng = 8
    ext = _FakeExt()
    r, c = _marginals(ng)
    kkt_patch, kkt_calls = _kkt_patch(_const_kkt(1e-7, 0.1, 0.1))
    obj_stub = mock.patch.object(
        grid_cuda, "_compute_factored_objective_t",
        lambda *a, **kw: torch.tensor(0.5, dtype=torch.float64))
    with _proxy_patch(lambda k: 1e-7), kkt_patch, obj_stub:
        _, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=10**6, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=False, gap_stop_fn=lambda i: 0.0, gap_stop_tol=1e-4,
            native_dual_residual=True, _ext_override=ext)
    assert info["native_termination"] == "feasible_gap_below_tol"
    assert kkt_calls == [(50, True)]
    assert info["full_kkt_checks"] == 1
    assert info["dual_res_F"] == 0.1


def test_gap_stop_and_traj_callback_mutually_exclusive():
    ng = 4
    r, c = _marginals(ng)
    with pytest.raises(ValueError):
        grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=10, check_every=5),
            cost_type="sql2", grid_size=ng,
            traj_callback=lambda d: None, gap_stop_fn=lambda i: 0.0,
            _ext_override=_FakeExt())


def test_trajectory_mode_full_kkt_every_batch():
    ng = 4
    ext = _FakeExt()
    r, c = _marginals(ng)
    kkt_patch, kkt_calls = _kkt_patch(_const_kkt(1.0, 1.0, 1.0))
    seen = []
    with _proxy_patch(lambda k: 1.0), kkt_patch:
        grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c,
            BregmanDRParams(eta=1.0, max_iters=150, check_every=50,
                            epsilon_primal=0.0, epsilon_dual=0.0,
                            epsilon_gap=0.0),
            cost_type="l1", grid_size=ng,
            return_plan=False, traj_callback=seen.append, _ext_override=ext)
    assert kkt_calls == [(50, True), (100, True), (150, True)]
    assert [d["k"] for d in seen] == [50, 100, 150]


def test_materialize_called_with_final_state_when_return_plan():
    ng = 8
    ext = _FakeExt()
    r, c = _marginals(ng)
    kkt_patch, _ = _kkt_patch(_const_kkt(1.0, 1.0, 1.0))
    with _proxy_patch(lambda k: 1.0), kkt_patch:
        X, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=100, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=True, _ext_override=ext)
    N = ng * ng
    assert X is not None and X.shape == (N, N)
    assert ext.materialize_calls == [100]


# ---- The session iterates on (r̃, c̃); gates and delivery use (r, c) ---------

def _zero_mass_marginals(ng, seed=0):
    """Unit-mass marginals with exact zeros, like an image with a zero background."""
    N = ng * ng
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(N, generator=g, dtype=torch.float64)
    c = torch.rand(N, generator=g, dtype=torch.float64)
    r[: N // 3] = 0.0
    c[N // 2:] = 0.0
    return r / r.sum(), c / c.sum()


def test_session_iterates_on_floored_marginals_checks_use_originals():
    ng = 8
    ext = _FakeExt()
    r, c = _zero_mass_marginals(ng)
    assert float(r.min()) == 0.0 and float(c.min()) == 0.0
    proxy_seen, kkt_seen = [], []
    kkt_patch, _ = _kkt_patch(_const_kkt(1.0, 1.0, 1.0), seen=kkt_seen)
    with _proxy_patch(lambda k: 1.0, seen=proxy_seen), kkt_patch:
        _, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=100, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=False, _ext_override=ext)

    sess = ext.sessions[0]
    assert float(sess.r.min()) > 0.0 and float(sess.c.min()) > 0.0
    assert torch.equal(sess.r, floor_marginal(r, MARGINAL_EPS))
    assert torch.equal(sess.c, floor_marginal(c, MARGINAL_EPS))
    assert abs(float(sess.r.sum()) - 1.0) < 1e-12
    assert info["marginal_floor"] == MARGINAL_EPS == 1e-12
    assert len(proxy_seen) == 2
    for r_seen, c_seen in proxy_seen + kkt_seen:
        assert torch.equal(r_seen, r) and torch.equal(c_seen, c)


def test_check_seams_override_the_default_originals():
    ng = 8
    ext = _FakeExt()
    r, c = _zero_mass_marginals(ng)
    r_alt, c_alt = _marginals(ng)
    proxy_seen, kkt_seen = [], []
    kkt_patch, _ = _kkt_patch(_const_kkt(1.0, 1.0, 1.0), seen=kkt_seen)
    with _proxy_patch(lambda k: 1.0, seen=proxy_seen), kkt_patch:
        grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=50, check_every=50),
            cost_type="sql2", grid_size=ng,
            return_plan=False, check_r=r_alt, check_c=c_alt, _ext_override=ext)
    for r_seen, c_seen in proxy_seen + kkt_seen:
        assert torch.equal(r_seen, r_alt) and torch.equal(c_seen, c_alt)


def test_eps_mass_zero_with_zero_mass_marginal_raises():
    ng = 4
    r, c = _zero_mass_marginals(ng)
    params = BregmanDRParams(eta=1.0, max_iters=10, check_every=5, eps_mass=0.0)
    with pytest.raises(ValueError, match="floored iteration marginals"):
        grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, params, cost_type="sql2", grid_size=ng,
            _ext_override=_FakeExt())


def test_negative_marginal_entry_raises():
    ng = 4
    r, c = _marginals(ng)
    r = r.clone()
    r[0] = -1e-3
    with pytest.raises(ValueError, match="non-negative"):
        grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=10, check_every=5),
            cost_type="sql2", grid_size=ng, _ext_override=_FakeExt())


def test_product_init_uses_floored_marginals():
    """init='product' sets ξ⁰ = log r̃ and ζ⁰ = log c̃, finite even where r or c is 0."""
    ng = 4
    r, c = _zero_mass_marginals(ng)
    captured = {}

    class _Ext(_FakeExt):
        def BdrsSession(self, **kwargs):
            captured.update(kwargs)
            return _FakeSession(self, **kwargs)

    ext = _Ext()
    kkt_patch, _ = _kkt_patch(_const_kkt(1.0, 1.0, 1.0))
    with _proxy_patch(lambda k: 1.0), kkt_patch:
        grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=10, check_every=5),
            cost_type="l1", grid_size=ng,
            return_plan=False, init="product", _ext_override=ext)
    assert bool(torch.isfinite(captured["init_A"]).all())
    assert bool(torch.isfinite(captured["init_B"]).all())
    assert torch.equal(captured["init_A"], floor_marginal(r).log())


def test_check_at_schedule_drives_checks_and_step_sizes():
    """Checks fall at the listed iterations in (0, K_max], sorted and without
    duplicates, and at K_max; the step batches are the gaps between them."""
    ng = 8
    ext = _FakeExt()
    r, c = _marginals(ng)
    kkt_patch, _ = _kkt_patch(_const_kkt(1e-7, 0.1, 0.1))
    obj_stub = mock.patch.object(
        grid_cuda, "_compute_factored_objective_t",
        lambda *a, **kw: torch.tensor(0.5, dtype=torch.float64))
    gap_calls = []

    def gap_fn(info):
        gap_calls.append(int(info["iters"]))
        return 1.0                      # never stops: the solve runs to K_max

    with _proxy_patch(lambda k: 1e-7), kkt_patch, obj_stub:
        _, info = grid_cuda.bregman_dr_solve_grid_cuda(
            None, r, c, _params(max_iters=100, check_every=7),
            cost_type="sql2", grid_size=ng,
            return_plan=False, gap_stop_fn=gap_fn, gap_stop_tol=1e-4,
            _ext_override=ext, check_at=[89, 1, 2, 3, 5, 8, 13, 21, 34, 55, 55, 0, 400])
    assert info["native_termination"] == "max_iters"
    assert info["iters"] == 100
    assert gap_calls == [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 100]
    assert ext.sessions[0].step_calls == [1, 1, 1, 2, 3, 5, 8, 13, 21, 34, 11]
    assert len(ext.sessions) == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
