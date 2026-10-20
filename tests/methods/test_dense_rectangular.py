"""The General solver (``factored_torch.bregman_dr_solve_general_factored``)
on random rectangular (m != n) instances."""

import pytest

torch = pytest.importorskip("torch")
ot = pytest.importorskip("ot")

from tests.helpers import requires_cuda
from tests.methods.dense_random import make_random_dense_dot
from src.methods.dual_cleanup import (
    feasible_dual_and_gap_storedC,
    make_dense_gap_stop_fn,
)
from src.methods.factored_torch import bregman_dr_solve_general_factored
from src.methods.params import BregmanDRParams

SOLVERS = [bregman_dr_solve_general_factored]
SOLVER_IDS = ["factored"]
# Both orientations and a square control.
SHAPES = [(40, 24), (24, 40), (33, 33)]


def _r_dinf(ap, bp, C):
    """r_Dinf = max_ij (α'_i + β'_j − C_ij)₊, formed densely."""
    return float((ap[:, None] + bp[None, :] - C).clamp_min(0.0).max().item())


def _solve_and_certify(solver, C, r, c, *, tol):
    """Solve with the feasible-gap stop and return the delivered certificate,
    reused from the stop's cache when it was taken on the final iterate."""
    eta = float(C.mean())
    cache: dict = {}
    gap_fn = make_dense_gap_stop_fn(C, r, c, variant="alternating", cache=cache)
    params = BregmanDRParams(
        eta=eta, max_iters=200_000, epsilon_primal=tol,
        epsilon_dual=float("inf"), epsilon_gap=float("inf"), check_every=50)
    X, info = solver(C, r, c, params, gap_stop_fn=gap_fn, gap_stop_tol=tol)
    obj_t = (C * X).sum()
    if cache.get("iters") == int(info["iters"]):
        ap, bp, g_feas = cache["alpha_p"], cache["beta_p"], cache["g_feas"]
    else:
        ap, bp, g_feas = feasible_dual_and_gap_storedC(
            info["log_u"], info["log_v"], eta, r, c, C, obj_t,
            variant="alternating")
    return X, ap, bp, float(g_feas), info


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("m,n", SHAPES, ids=[f"{m}x{n}" for m, n in SHAPES])
def test_rectangular_correct_cpu(solver, m, n):
    C, r, c = make_random_dense_dot(m, n, seed=7)
    tol = 1e-6
    X, ap, bp, g_feas, info = _solve_and_certify(
        solver, C, r, c, tol=tol)

    assert ap.shape == (m,)
    assert bp.shape == (n,)
    assert X.shape == (m, n)

    # Primal feasibility (the proxy gate fired below tol).
    pr = float((X.sum(dim=1) - r).norm(2))
    pc = float((X.sum(dim=0) - c).norm(2))
    assert (pr + pc) < tol * 5.0
    assert float(X.min()) >= 0.0

    # Exactly dual feasible by construction (c-transform).
    bound = 1e-12 * (1.0 + float(ap.abs().max()) + float(bp.abs().max()))
    assert _r_dinf(ap, bp, C) <= bound

    assert g_feas < tol
    # The delivered g_feas is the certificate of the delivered plan: recomputed
    # from the final potentials and ⟨C, X⟩ of the returned X, it is unchanged.
    obj_t = (C * X).sum()
    _, _, g_chk = feasible_dual_and_gap_storedC(
        info["log_u"], info["log_v"], float(C.mean()), r, c, C, obj_t,
        variant="alternating")
    assert abs(g_feas - g_chk) <= 1e-12
    obj = float(obj_t)

    p_star = float(ot.emd2(r.numpy(), c.numpy(), C.numpy()))
    assert abs(obj - p_star) < 10.0 * tol


@requires_cuda
@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("m,n", [(40, 24), (24, 40)], ids=["40x24", "24x40"])
def test_cpu_cuda_bit_close(solver, m, n):
    # The generator is device-independent: the same instance on both devices.
    Cc, rc, cc = make_random_dense_dot(m, n, seed=5)
    dev = torch.device("cuda")
    Cg, rg, cg = make_random_dense_dot(m, n, seed=5, device=dev)
    eta = float(Cc.mean())
    # A fixed iteration count with no check bounds the reassociation drift.
    p_cpu = BregmanDRParams(eta=eta, max_iters=300, epsilon_primal=0.0,
                            epsilon_dual=0.0, epsilon_gap=0.0, check_every=1000)
    p_gpu = BregmanDRParams(eta=eta, max_iters=300, epsilon_primal=0.0,
                            epsilon_dual=0.0, epsilon_gap=0.0, check_every=1000)
    Xc, ic = solver(Cc, rc, cc, p_cpu)
    Xg, ig = solver(Cg, rg, cg, p_gpu)
    assert int(ic["iters"]) == int(ig["iters"]) == 300
    assert float((Xc - Xg.cpu()).abs().max()) < 1e-10
    assert float((ic["log_u"] - ig["log_u"].cpu()).abs().max()) < 1e-9
    assert float((ic["log_v"] - ig["log_v"].cpu()).abs().max()) < 1e-9


@requires_cuda
@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
@pytest.mark.parametrize("m,n", [(40, 24), (24, 40)], ids=["40x24", "24x40"])
def test_rectangular_correct_cuda(solver, m, n):
    dev = torch.device("cuda")
    C, r, c = make_random_dense_dot(m, n, seed=7, device=dev)
    tol = 1e-6
    X, ap, bp, g_feas, info = _solve_and_certify(
        solver, C, r, c, tol=tol)
    assert ap.shape == (m,) and bp.shape == (n,)
    bound = 1e-12 * (1.0 + float(ap.abs().max()) + float(bp.abs().max()))
    assert _r_dinf(ap, bp, C) <= bound
    assert g_feas < tol
    obj = float((C * X).sum())
    p_star = float(ot.emd2(r.cpu().numpy(), c.cpu().numpy(), C.cpu().numpy()))
    assert abs(obj - p_star) < 10.0 * tol


@pytest.mark.parametrize("solver", SOLVERS, ids=SOLVER_IDS)
def test_shape_validation(solver):
    C, r, c = make_random_dense_dot(12, 8, seed=1)
    params = BregmanDRParams(eta=0.3, max_iters=10)
    with pytest.raises(ValueError):
        solver(C, c, r, params)
    with pytest.raises(ValueError):
        solver(C, r[:-1], c, params)
    with pytest.raises(ValueError):
        solver(C.reshape(-1), r, c, params)
