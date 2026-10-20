"""The stored-C c-transform and certificate (``dual_cleanup`` ``*_storedC``), on CPU."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.dotmark_load import get_dense_cost_matrix
from src.methods import dual_cleanup as DC

CPU = torch.device("cpu")


def _C_grid(ng, norm):
    return get_dense_cost_matrix(ng, norm, device=CPU)


def _C_asym(N, seed=0):
    """A non-symmetric, strictly positive cost, so the two transform directions differ."""
    rng = np.random.default_rng(seed)
    C = torch.tensor(np.abs(rng.standard_normal((N, N))) + 0.05,
                     dtype=torch.float64)
    assert not torch.allclose(C, C.T)
    return C


@pytest.mark.parametrize("seed", [0, 1])
def test_storedC_colmin_rowmin_vs_brute(seed):
    N = 37    # not a grid size: the stored-C backend takes any C
    C = _C_asym(N, seed)
    rng = np.random.default_rng(seed + 100)
    p = torch.tensor(rng.standard_normal(N) * 0.3, dtype=torch.float64)

    # colmin: q_j = min_i (C_ij − p_i)
    brute_col = (C - p[:, None]).min(dim=0).values
    got_col = DC._ctransform_storedC_colmin(p, C, row_chunk=7)   # force chunking
    assert torch.equal(got_col, brute_col), "colmin must be bit-identical (min is exact)"

    # rowmin: q_i = min_j (C_ij − p_j)
    brute_row = (C - p[None, :]).min(dim=1).values
    got_row = DC._ctransform_storedC_rowmin(p, C, row_chunk=5)
    assert torch.equal(got_row, brute_row), "rowmin must be bit-identical"


@pytest.mark.parametrize("variant", ["beta", "alpha", "alternating"])
def test_storedC_feasibility_machine_eps(variant):
    """α' ⊕ β' ≤ C up to the rounding of the subtraction inside the minimum."""
    N = 64
    C = _C_asym(N, 3)
    rng = np.random.default_rng(7)
    alpha = torch.tensor(rng.standard_normal(N) * 0.2, dtype=torch.float64)
    beta = torch.tensor(rng.standard_normal(N) * 0.2, dtype=torch.float64)

    ap, bp = DC.dual_cleanup_ctransform_storedC(alpha, beta, C, variant=variant)
    viol = (ap[:, None] + bp[None, :] - C).clamp_min(0.0).max()
    assert float(viol) <= 1e-12
    assert DC._dual_feas_F_storedC(ap, bp, C) <= 1e-12


def test_storedC_alternating_order():
    """``alternating``: β'_j = min_i(C_ij − α_i) from the input α, then
    α'_i = min_j(C_ij − β'_j)."""
    N = 16
    C = _C_asym(N, 11)
    alpha = torch.linspace(-0.1, 0.2, N, dtype=torch.float64)
    beta = torch.zeros(N, dtype=torch.float64)
    ap, bp = DC.dual_cleanup_ctransform_storedC(alpha, beta, C, variant="alternating")
    assert torch.equal(bp, DC._ctransform_storedC_colmin(alpha, C))
    assert torch.equal(ap, DC._ctransform_storedC_rowmin(bp, C))


@pytest.mark.parametrize("norm", ["l1", "sql2"])
@pytest.mark.parametrize("ng", [4, 8, 16])
def test_storedC_equals_grid_backends(norm, ng):
    C = _C_grid(ng, norm)
    N = ng * ng
    rng = np.random.default_rng(1000 + ng)
    alpha = torch.tensor(rng.standard_normal(N) * 0.3, dtype=torch.float64)
    beta = torch.tensor(rng.standard_normal(N) * 0.3, dtype=torch.float64)

    ap_s, bp_s = DC.dual_cleanup_ctransform_storedC(alpha, beta, C, variant="alternating")
    ap_g, bp_g = DC.dual_cleanup_ctransform(alpha, beta, ng, norm,
                                            variant="alternating", backend="separable")
    # The Grid cost is symmetric, so the Grid backend's one transform serves
    # both stored-C directions. The backends add the same terms in a different
    # order, so they agree to rounding, not bitwise.
    assert float((bp_s - bp_g).abs().max()) <= 1e-12
    assert float((ap_s - ap_g).abs().max()) <= 1e-12


@pytest.mark.parametrize("norm", ["l1", "sql2"])
def test_feasible_dual_and_gap_storedC_parity(norm):
    ng = 8
    N = ng * ng
    C = _C_grid(ng, norm)
    rng = np.random.default_rng(42)
    r = torch.tensor(np.abs(rng.standard_normal(N)) + 1e-3, dtype=torch.float64)
    r /= r.sum()
    c = torch.tensor(np.abs(rng.standard_normal(N)) + 1e-3, dtype=torch.float64)
    c /= c.sum()
    eta = 0.3
    log_u = torch.tensor(rng.standard_normal(N) * 0.1, dtype=torch.float64)
    log_v = torch.tensor(rng.standard_normal(N) * 0.1, dtype=torch.float64)
    objective = 0.123

    ap_s, bp_s, g_s = DC.feasible_dual_and_gap_storedC(
        log_u, log_v, eta, r, c, C, objective)
    ap_g, bp_g, g_g = DC.feasible_dual_and_gap(
        log_u, log_v, eta, r, c, ng, norm, objective)
    assert float((ap_s - ap_g).abs().max()) <= 1e-10
    assert float((bp_s - bp_g).abs().max()) <= 1e-10
    assert abs(g_s - g_g) <= 1e-12

    ap_t, bp_t, g_t = DC.feasible_dual_and_gap_storedC(
        log_u, log_v, eta, r, c, C,
        torch.tensor(objective, dtype=torch.float64))
    assert g_t == g_s


def test_make_dense_gap_stop_fn_cache():
    N = 25
    C = _C_asym(N, 5)
    rng = np.random.default_rng(9)
    r = torch.tensor(np.abs(rng.standard_normal(N)) + 1e-3, dtype=torch.float64)
    r /= r.sum()
    c = torch.tensor(np.abs(rng.standard_normal(N)) + 1e-3, dtype=torch.float64)
    c /= c.sum()
    info = {
        "iters": 7, "eta": 0.2,
        "log_u": torch.tensor(rng.standard_normal(N) * 0.05, dtype=torch.float64),
        "log_v": torch.tensor(rng.standard_normal(N) * 0.05, dtype=torch.float64),
        "objective": 0.456,
    }
    cache = {}
    fn = DC.make_dense_gap_stop_fn(C, r, c, cache=cache)
    value = fn(info)
    assert set(cache) == {"iters", "alpha_p", "beta_p", "g_feas"}
    assert cache["iters"] == 7 and value == abs(cache["g_feas"])
    assert DC._dual_feas_F_storedC(cache["alpha_p"], cache["beta_p"], C) <= 1e-12


# A minimum does not round, so the block size cannot change the result.
def test_storedC_chunk_invariance():
    N = 53
    C = _C_asym(N, 21)
    p = torch.linspace(-1, 1, N, dtype=torch.float64)
    full_col = DC._ctransform_storedC_colmin(p, C, row_chunk=N)
    full_row = DC._ctransform_storedC_rowmin(p, C, row_chunk=N)
    for chunk in (1, 3, 17, 52):
        assert torch.equal(DC._ctransform_storedC_colmin(p, C, row_chunk=chunk), full_col)
        assert torch.equal(DC._ctransform_storedC_rowmin(p, C, row_chunk=chunk), full_row)
