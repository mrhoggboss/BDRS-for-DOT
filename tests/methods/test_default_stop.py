"""The solvers' default stop, ``stop="feasible_gap"``, on the CPU.

Without a ``gap_stop_fn`` each solver builds the certificate of its format
(``dual_cleanup.make_grid_gap_stop_fn``, ``make_pointcloud_gap_stop_fn`` or
``make_dense_gap_stop_fn``) on the cost it iterates on and on the check
marginals, with ``gap_stop_tol`` = ``epsilon_primal``. Each test compares a
default solve against the same solve given that certificate explicitly, as
the runners give it, and requires bitwise equal iterates, stops and
delivered values. The CUDA sessions are replaced by torch sessions that run
the same iteration (the ``_ext_override`` seam), so the session paths run
here too.
"""

import inspect
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.methods import dual_cleanup as DC
from src.methods import geo_cost
from src.methods.eta import eta_from_mean, grid_cost_mean
from src.methods.factored_cuda import (
    bregman_dr_solve_general_factored_cuda,
    bregman_dr_solve_pointcloud_cuda,
)
from src.methods.factored_torch import (
    bregman_dr_solve_general_factored,
    bregman_dr_solve_pointcloud,
)
from src.methods.grid_cost import _base_ell, _cost_axis
from src.methods.grid_cuda import bregman_dr_solve_grid_cuda
from src.methods.params import STOP_RULES, BregmanDRParams

F64 = torch.float64


# ---- torch stand-ins for the CUDA sessions ------------------------------------

class _TorchGridSession:
    """``ext.BdrsSession``: the Grid iteration on the separable lattice cost,
    with ``step(n)`` updating ξ, ζ, log u and log v in place."""

    def __init__(self, *, r, c, ng, eta, cost_type, init_A, init_B, init_log_v):
        self.ng = int(ng)
        N = self.ng * self.ng
        name = {0: "l1", 1: "sql2"}[int(cost_type)]
        self.log_r = r.reshape(self.ng, self.ng).log()
        self.log_c = c.reshape(self.ng, self.ng).log()
        self.base_ell = _base_ell(name, self.ng, eta, dtype=F64, device=r.device)
        idx = torch.arange(self.ng)
        self.D = (idx[:, None] - idx[None, :]).abs()
        self.A = torch.full((N,), -math.log(N), dtype=F64)
        self.B = torch.full((N,), -math.log(N), dtype=F64)
        self.log_u = torch.zeros(N, dtype=F64)
        self.log_v = torch.zeros(N, dtype=F64)
        self.k = 0

    def step(self, n):
        ng = self.ng
        for _ in range(int(n)):
            self.k += 1
            kL = (self.k * self.base_ell)[self.D]
            A = self.A.view(ng, ng)
            B = self.B.view(ng, ng)
            beta_tilde = B + self.log_v.view(ng, ng)
            logS = torch.logsumexp(kL.unsqueeze(1) + beta_tilde.T.unsqueeze(0), dim=2)
            A_new = self.log_r - torch.logsumexp(kL.unsqueeze(0) + logS.unsqueeze(1), dim=2)
            self.log_u.copy_((A_new - A).reshape(-1))
            self.A.copy_(A_new.reshape(-1))
            logT = torch.logsumexp(kL.T.unsqueeze(1) + A_new.T.unsqueeze(0), dim=2)
            B_new = self.log_c - torch.logsumexp(kL.T.unsqueeze(0) + logT.unsqueeze(1), dim=2)
            self.log_v.copy_((B_new - B).reshape(-1))
            self.B.copy_(B_new.reshape(-1))
        return self.k


class _TorchFactoredSession:
    """``ext.DenseFactoredSession`` and ``ext.PointCloudBdrsSession``: the
    factored iteration on a dense (m, n) cost, updating its state in place."""

    def __init__(self, C, r, c, eta):
        m, n = C.shape
        self.C, self.eta = C, float(eta)
        self.log_r, self.log_c = r.log(), c.log()
        self.A = torch.full((m,), -math.log(m), dtype=F64)
        self.B = torch.full((n,), -math.log(n), dtype=F64)
        self.log_u = torch.zeros(m, dtype=F64)
        self.log_v = torch.zeros(n, dtype=F64)
        self.k = 0

    def step(self, n_steps):
        for _ in range(int(n_steps)):
            self.k += 1
            kL = self.C.mul(-(self.k / self.eta))
            A_new = self.log_r - torch.logsumexp(kL + (self.B + self.log_v)[None, :], dim=1)
            self.log_u.copy_(A_new - self.A)
            self.A.copy_(A_new)
            B_new = self.log_c - torch.logsumexp(kL + self.A[:, None], dim=0)
            self.log_v.copy_(B_new - self.B)
            self.B.copy_(B_new)
        return self.k


_PC_CODES = {0: "l1", 1: "sql2", 2: "l2", 3: "linf"}


class _TorchExt:
    def BdrsSession(self, **kw):
        return _TorchGridSession(**kw)

    def DenseFactoredSession(self, *, C, r, c, eta, rows_per_chunk):
        return _TorchFactoredSession(C, r, c, eta)

    def PointCloudBdrsSession(self, *, P, Q, r, c, eta, cost_type, c_max):
        cost_fn = geo_cost.make_metric_cost_fn(P, Q, _PC_CODES[int(cost_type)], c_max)
        C = geo_cost.materialize_cost(cost_fn, P.shape[0], Q.shape[0], device=P.device)
        return _TorchFactoredSession(C, r, c, eta)


# ---- instances and comparison --------------------------------------------------

def _marginals(m, n, seed, n_zero=0):
    rng = np.random.default_rng(seed)
    r = np.abs(rng.standard_normal(m)) + 1e-3
    c = np.abs(rng.standard_normal(n)) + 1e-3
    r[:n_zero] = 0.0
    return torch.from_numpy(r / r.sum()), torch.from_numpy(c / c.sum())


def _dense_instance(m, n, seed):
    rng = np.random.default_rng(seed)
    C = torch.from_numpy(np.abs(rng.standard_normal((m, n))) + 0.02)
    r, c = _marginals(m, n, seed + 1, n_zero=3)
    return C / C.max(), r, c


def _clouds(m, n, d, seed):
    rng = np.random.default_rng(seed)
    P = torch.from_numpy(rng.standard_normal((m, d)))
    Q = torch.from_numpy(rng.standard_normal((n, d)) + 0.3)
    r, c = _marginals(m, n, seed + 1, n_zero=2)
    return P, Q, r, c


def _params(eta, tol, check_every, max_iters=20000):
    return BregmanDRParams(eta=eta, max_iters=max_iters, check_every=check_every,
                           epsilon_primal=tol)


def _as_float(v):
    return float(v.item()) if isinstance(v, torch.Tensor) else float(v)


def _assert_same_solve(a, b):
    """Two ``(X, info)`` results with bitwise equal iterates, stop and
    delivered values."""
    (Xa, ia), (Xb, ib) = a, b
    assert int(ia["iters"]) == int(ib["iters"])
    assert ia["native_termination"] == ib["native_termination"]
    for key in ("log_u", "log_v"):
        assert torch.equal(ia[key], ib[key]), key
    if Xa is not None or Xb is not None:
        assert torch.equal(Xa, Xb)
    for key in ("primal_res_L2", "gap_abs", "objective", "gap_stop_value",
                "primal_res_proxy_final", "proxy_passes", "sep_obj_checks",
                "full_kkt_checks", "cost_scale"):
        assert (key in ia) == (key in ib), key
        if key in ia:
            assert _as_float(ia[key]) == _as_float(ib[key]), key


def _assert_certified_stop(info, cache, tol):
    """The default stop fired on the certificate in ``cache``."""
    assert info["native_termination"] == "feasible_gap_below_tol"
    assert int(info["iters"]) == cache["iters"]
    assert info["gap_stop_value"] == abs(cache["g_feas"]) < tol
    assert info["primal_res_proxy_final"] < tol


# ---- every entry point defaults to the feasible-gap stop ------------------------

ENTRIES = [bregman_dr_solve_grid_cuda, bregman_dr_solve_pointcloud_cuda,
           bregman_dr_solve_general_factored_cuda, bregman_dr_solve_pointcloud,
           bregman_dr_solve_general_factored]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda f: f.__name__)
def test_entry_defaults(entry):
    sig = inspect.signature(entry).parameters
    assert sig["stop"].default == "feasible_gap" == STOP_RULES[0]
    assert sig["gap_stop_tol"].default is None
    assert sig["gap_stop_fn"].default is None
    assert set(STOP_RULES) == {"feasible_gap", "kkt_triple"}


def test_params_defaults_match_the_benchmark_spec():
    """A direct call and a benchmark run (``InstanceSpec``) share the
    tolerance, 1e-4, and K_max, 10⁶; δ_m is 1e-12."""
    from src.protocol.instance_record import InstanceSpec
    p = BregmanDRParams(eta=1.0)
    spec = inspect.signature(InstanceSpec).parameters
    assert p.max_iters == spec["iter_cap"].default == 10**6
    assert p.epsilon_primal == spec["tolerance"].default == 1e-4
    assert p.eps_mass == 1e-12


# ---- General ----------------------------------------------------------------------

@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_general_torch_default_is_the_dense_certificate_on_the_normalized_cost(scale):
    C, r, c = _dense_instance(30, 24, seed=1)
    Cs = scale * C
    tol = 1e-6
    params = _params(eta=0.7 * float(C.mean()), tol=tol, check_every=10)
    default = bregman_dr_solve_general_factored(Cs, r, c, params)
    cache: dict = {}
    explicit = bregman_dr_solve_general_factored(
        Cs, r, c, params, gap_stop_tol=tol,
        gap_stop_fn=DC.make_dense_gap_stop_fn(Cs / Cs.max(), r, c, cache=cache))
    _assert_same_solve(default, explicit)
    _assert_certified_stop(default[1], cache, tol)
    assert default[1]["cost_scale"] == float(Cs.max())


@pytest.mark.parametrize("session", [False, True], ids=["fallback", "session"])
def test_general_cuda_entry_default_on_the_cpu(session):
    C, r, c = _dense_instance(28, 36, seed=3)
    tol = 1e-6
    params = _params(eta=0.7 * float(C.mean()), tol=tol, check_every=10)
    kw = {"_ext_override": _TorchExt()} if session else {}
    default = bregman_dr_solve_general_factored_cuda(C, r, c, params, **kw)
    cache: dict = {}
    explicit = bregman_dr_solve_general_factored_cuda(
        C, r, c, params, gap_stop_tol=tol,
        gap_stop_fn=DC.make_dense_gap_stop_fn(C, r, c, cache=cache),
        **({"_ext_override": _TorchExt()} if session else {}))
    _assert_same_solve(default, explicit)
    _assert_certified_stop(default[1], cache, tol)
    assert default[1]["factored_session"] is session


def test_gap_stop_tol_defaults_to_epsilon_primal():
    C, r, c = _dense_instance(20, 20, seed=5)
    tol = 1e-5
    params = _params(eta=0.6 * float(C.mean()), tol=tol, check_every=5)
    implicit = bregman_dr_solve_general_factored(
        C, r, c, params, gap_stop_fn=DC.make_dense_gap_stop_fn(C, r, c))
    explicit = bregman_dr_solve_general_factored(
        C, r, c, params, gap_stop_fn=DC.make_dense_gap_stop_fn(C, r, c), gap_stop_tol=tol)
    _assert_same_solve(implicit, explicit)
    assert implicit[1]["native_termination"] == "feasible_gap_below_tol"


def test_kkt_triple_is_the_option_and_differs_from_the_default():
    """``stop="kkt_triple"`` tests the uncorrected (η·log u, η·log v), whose
    r_D lags, so on the same tolerances it stops later than the default or
    not at all."""
    C, r, c = _dense_instance(24, 24, seed=7)
    tol = 1e-6
    params = BregmanDRParams(eta=0.7 * float(C.mean()), max_iters=3000, check_every=10,
                             epsilon_primal=tol, epsilon_dual=tol, epsilon_gap=tol)
    _, d = bregman_dr_solve_general_factored(C, r, c, params, return_plan=False)
    _, t = bregman_dr_solve_general_factored(C, r, c, params, return_plan=False,
                                             stop="kkt_triple")
    assert d["native_termination"] == "feasible_gap_below_tol"
    assert t["native_termination"] in ("kkt_below_tol", "max_iters")
    assert int(t["iters"]) > int(d["iters"])
    assert "gap_stop_value" not in t and t["sep_obj_checks"] == 0
    assert t["full_kkt_checks"] >= 1 and d["full_kkt_checks"] == 0


# ---- Point Cloud ------------------------------------------------------------------

@pytest.mark.parametrize("cost_type", ["sql2", "l2"])
@pytest.mark.parametrize("bound", [1.0, 1.6], ids=["exact_cmax", "bounding_cmax"])
def test_pointcloud_torch_default_is_the_symbolic_certificate_on_the_iterated_cost(
        cost_type, bound):
    P, Q, r, c = _clouds(36, 30, 2, seed=11)
    exact = geo_cost.exact_c_max(P, Q, cost_type)
    cost_fn = geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max=bound * exact)
    cmax = geo_cost.exact_cost_fn_max(cost_fn, 36)
    iterated = (cost_fn if cmax == 1.0
                else (lambda i0, i1: cost_fn(i0, i1) / cmax))   # the Normalize step's cost
    tol = 1e-5
    params = _params(eta=0.3, tol=tol, check_every=10)
    default = bregman_dr_solve_pointcloud(r, c, params, cost_fn=cost_fn)
    cache: dict = {}
    explicit = bregman_dr_solve_pointcloud(
        r, c, params, cost_fn=cost_fn, gap_stop_tol=tol,
        gap_stop_fn=DC.make_pointcloud_gap_stop_fn(iterated, r, c, cache=cache))
    _assert_same_solve(default, explicit)
    _assert_certified_stop(default[1], cache, tol)


@pytest.mark.parametrize("session", [False, True], ids=["fallback", "session"])
def test_pointcloud_cuda_entry_default_on_the_cpu(monkeypatch, session):
    monkeypatch.setenv("BDRS_GEO_KERNELS", "0")
    P, Q, r, c = _clouds(32, 40, 3, seed=13)
    cost_type = "l1"
    c_max = geo_cost.exact_c_max(P, Q, cost_type)
    tol = 1e-5
    params = _params(eta=0.3, tol=tol, check_every=10)

    def solve(**kw):
        if session:
            kw["_ext_override"] = _TorchExt()
        return bregman_dr_solve_pointcloud_cuda(P, Q, r, c, params, cost_type=cost_type,
                                                c_max=c_max, **kw)

    default = solve()
    cache: dict = {}
    explicit = solve(gap_stop_tol=tol, gap_stop_fn=DC.make_pointcloud_gap_stop_fn(
        geo_cost.make_metric_cost_fn(P, Q, cost_type, c_max), r, c, cache=cache))
    _assert_same_solve(default, explicit)
    _assert_certified_stop(default[1], cache, tol)
    assert default[1]["cuda_graph"] is session


# ---- Grid ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", ["l1", "sql2"])
def test_grid_default_is_the_runner_certificate(norm):
    """The default against the certificate as ``run_grid`` builds it."""
    ng = 8
    r, c = _marginals(ng * ng, ng * ng, seed=17, n_zero=5)
    eta = eta_from_mean(grid_cost_mean(ng, norm))
    tol = 1e-5
    params = _params(eta=eta, tol=tol, check_every=20)
    default = bregman_dr_solve_grid_cuda(None, r, c, params, cost_type=norm, grid_size=ng,
                                         return_plan=False, _ext_override=_TorchExt())
    cost_axis = _cost_axis(norm, ng, device="cpu", dtype=F64)
    cache: dict = {}

    def certify(info, objective):
        return DC.feasible_dual_and_gap(info["log_u"], info["log_v"], eta, r, c, ng, norm,
                                        objective, variant="alternating", cost_axis=cost_axis)

    explicit = bregman_dr_solve_grid_cuda(
        None, r, c, params, cost_type=norm, grid_size=ng, return_plan=False,
        gap_stop_fn=DC.make_gap_stop_fn(certify, cache=cache), gap_stop_tol=tol,
        _ext_override=_TorchExt())
    _assert_same_solve(default, explicit)
    _assert_certified_stop(default[1], cache, tol)
    assert default[1]["full_kkt_checks"] == 0


def test_make_grid_gap_stop_fn_matches_feasible_dual_and_gap():
    ng, norm = 6, "sql2"
    N = ng * ng
    r, c = _marginals(N, N, seed=19, n_zero=4)
    rng = np.random.default_rng(20)
    info = {"log_u": torch.from_numpy(rng.standard_normal(N)),
            "log_v": torch.from_numpy(rng.standard_normal(N)),
            "eta": 0.3, "iters": 40, "objective": torch.tensor(0.25, dtype=F64)}
    cache: dict = {}
    val = DC.make_grid_gap_stop_fn(r, c, ng, norm, cache=cache)(info)
    ap, bp, g = DC.feasible_dual_and_gap(info["log_u"], info["log_v"], 0.3, r, c, ng, norm,
                                         info["objective"], variant="alternating")
    assert val == abs(g) and cache["g_feas"] == g and cache["iters"] == 40
    assert torch.equal(cache["alpha_p"], ap) and torch.equal(cache["beta_p"], bp)


# ---- trajectory mode and argument checks -----------------------------------------

def test_trajectory_mode_keeps_the_kkt_triple_test():
    C, r, c = _dense_instance(16, 16, seed=23)
    params = BregmanDRParams(eta=0.5, max_iters=400, check_every=20,
                             epsilon_primal=1e-3, epsilon_dual=1e-2, epsilon_gap=1e-2)
    seen_d, seen_t = [], []
    a = bregman_dr_solve_general_factored(C, r, c, params, traj_callback=seen_d.append)
    b = bregman_dr_solve_general_factored(C, r, c, params, traj_callback=seen_t.append,
                                          stop="kkt_triple")
    _assert_same_solve(a, b)
    assert seen_d == seen_t and len(seen_d) >= 1
    assert a[1]["native_termination"] in ("kkt_below_tol", "max_iters")
    assert "gap_stop_value" not in a[1]


def _call(entry, **kw):
    """One call of ``entry`` on a small CPU instance."""
    params = BregmanDRParams(eta=0.5, max_iters=5, check_every=5)
    if entry is bregman_dr_solve_grid_cuda:
        r, c = _marginals(16, 16, seed=29)
        return entry(None, r, c, params, cost_type="l1", grid_size=4, return_plan=False,
                     _ext_override=_TorchExt(), **kw)
    if entry is bregman_dr_solve_pointcloud_cuda:
        P, Q, r, c = _clouds(6, 5, 2, seed=29)
        return entry(P, Q, r, c, params, cost_type="l1", c_max=1.0, **kw)
    C, r, c = _dense_instance(6, 5, seed=29)
    if entry is bregman_dr_solve_pointcloud:
        return entry(r, c, params, cost_fn=lambda i0, i1: C[i0:i1], **kw)
    return entry(C, r, c, params, **kw)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda f: f.__name__)
def test_stop_arguments_are_checked(entry):
    with pytest.raises(ValueError, match="unknown stop"):
        _call(entry, stop="kkt")
    with pytest.raises(ValueError, match="kkt_triple"):
        _call(entry, stop="kkt_triple", gap_stop_fn=lambda info: 0.0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _call(entry, traj_callback=lambda d: None, gap_stop_fn=lambda info: 0.0)
    _, info = _call(entry, stop="kkt_triple")
    assert int(info["iters"]) == 5


def test_verbosity_prints_the_feasible_gap_checks(capsys):
    C, r, c = _dense_instance(12, 12, seed=31)
    params = BregmanDRParams(eta=0.5, max_iters=40, check_every=10, log_every=10,
                             epsilon_primal=10.0, verbosity=True)
    bregman_dr_solve_general_factored(C, r, c, params, gap_stop_tol=0.0)
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split(":")[0] for ln in lines] == ["iter 10", "iter 20", "iter 30", "iter 40"]
    assert all("|g_feas|=" in ln for ln in lines)
