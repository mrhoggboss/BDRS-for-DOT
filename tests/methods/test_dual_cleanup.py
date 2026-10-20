"""The c-transform and the feasible-gap certificate (``src.methods.dual_cleanup``)."""

from src.methods.dual_cleanup import _dual_feas_F
from typing import Dict
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.dotmark_load import get_dense_cost_matrix
from tests.helpers import requires_cuda
from src.methods import dual_cleanup as DC


def dual_objective(alpha, beta, r, c) -> float:
    """rᵀα + cᵀβ, invariant under (α + t, β − t) since r, c have unit mass."""
    return float((r * alpha).sum() + (c * beta).sum())

def duality_gap(alpha, beta, r, c, objective: float) -> float:
    return float(objective) - dual_objective(alpha, beta, r, c)

def evaluate(alpha, beta, ng, cost_type, r, c, objective) -> Dict[str, float]:
    return {
        "r_D": _dual_feas_F(alpha, beta, ng, cost_type),
        "gap": duality_gap(alpha, beta, r, c, objective),
        "dual_obj": dual_objective(alpha, beta, r, c),
    }

def best_ctransform(
    alpha: torch.Tensor, beta: torch.Tensor, ng: int, cost_type: str,
    r: torch.Tensor, c: torch.Tensor, objective: float,
) -> Dict[str, object]:
    """The c-transform variant with the smallest |g|, plus per-variant metrics."""
    out = {}
    for v in ("beta", "alpha", "alternating"):
        ap, bp = DC.dual_cleanup_ctransform(alpha, beta, ng, cost_type, variant=v)
        out[v] = {**evaluate(ap, bp, ng, cost_type, r, c, objective), "alpha": ap, "beta": bp}
    best = min(out, key=lambda v: abs(out[v]["gap"]))
    return {"best_variant": best, "best": out[best], "per_variant": out}

CPU = torch.device("cpu")


def _C(ng, norm):
    return get_dense_cost_matrix(ng, norm, device=CPU)


def test_ctransform_tiny_hand():
    ng, norm = 2, "l1"
    C = _C(ng, norm)
    alpha = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64)
    beta_p = DC._ctransform(alpha, ng, norm)
    brute = (C - alpha[:, None]).min(dim=0).values      # β'_j = min_i (C_ij − α_i)
    assert torch.allclose(beta_p, brute, atol=1e-12)
    assert float((alpha[:, None] + beta_p[None, :] - C).clamp_min(0).max()) <= 1e-12


def test_ctransform_separable_tiny_hand():
    ng, norm = 4, "l1"
    C = _C(ng, norm)
    p = torch.tensor([0.05 * i for i in range(ng * ng)], dtype=torch.float64)
    brute = (C - p[None, :]).min(dim=1).values          # q_m = min_k (C_mk − p_k)
    q_sep = DC._ctransform_separable(p, ng, norm)
    assert q_sep.shape == (ng * ng,)
    assert torch.allclose(q_sep, brute, atol=1e-12)
    assert float((p[:, None] + q_sep[None, :] - C).clamp_min(0).max()) <= 1e-12


@pytest.mark.parametrize("norm", ["l1", "sql2"])
@pytest.mark.parametrize("ng", [16, 32, 64])
def test_ctransform_separable_matches_dense(norm, ng):
    """Separable against dense to 1e-10, also on the second transform of the
    alternating variant."""
    rng = np.random.default_rng(1000 * ng + (norm == "sql2"))
    for _ in range(4):
        p = torch.tensor(rng.standard_normal(ng * ng) * 0.5, dtype=torch.float64)
        q_dense = DC._ctransform(p, ng, norm)
        q_sep = DC._ctransform_separable(p, ng, norm)
        assert q_sep.shape == q_dense.shape == (ng * ng,)
        assert float((q_sep - q_dense).abs().max()) <= 1e-10
        q_dense2 = DC._ctransform(q_dense, ng, norm)
        q_sep2 = DC._ctransform_separable(q_sep, ng, norm)
        assert float((q_sep2 - q_dense2).abs().max()) <= 1e-10


@pytest.mark.parametrize("norm", ["l1", "sql2"])
@pytest.mark.parametrize("variant", ["beta", "alpha", "alternating"])
def test_dual_cleanup_backend_equivalence(norm, variant):
    ng = 32
    rng = np.random.default_rng(7 * len(variant) + (norm == "sql2") * 13)
    alpha = torch.tensor(rng.standard_normal(ng * ng) * 0.3, dtype=torch.float64)
    beta = torch.tensor(rng.standard_normal(ng * ng) * 0.3, dtype=torch.float64)
    ap_d, bp_d = DC.dual_cleanup_ctransform(alpha, beta, ng, norm, variant=variant, backend="dense")
    ap_s, bp_s = DC.dual_cleanup_ctransform(alpha, beta, ng, norm, variant=variant, backend="separable")
    assert float((ap_s - ap_d).abs().max()) <= 1e-10
    assert float((bp_s - bp_d).abs().max()) <= 1e-10
    assert _dual_feas_F(ap_s, bp_s, ng, norm) <= 1e-9


def test_default_backend_is_separable_end_to_end():
    """Through feasible_dual_and_gap, backend=None equals "separable" bitwise
    and "dense" to 1e-10."""
    ng, norm = 16, "sql2"
    rng = np.random.default_rng(11)
    a = rng.random(ng * ng) + 0.05; a /= a.sum()
    b = rng.random(ng * ng) + 0.05; b /= b.sum()
    r = torch.tensor(a); c = torch.tensor(b)
    log_u = torch.tensor(rng.standard_normal(ng * ng) * 0.2, dtype=torch.float64)
    log_v = torch.tensor(rng.standard_normal(ng * ng) * 0.2, dtype=torch.float64)
    obj = 0.3
    ap0, bp0, g0 = DC.feasible_dual_and_gap(log_u, log_v, 0.2, r, c, ng, norm, obj)
    aps, bps, gs = DC.feasible_dual_and_gap(log_u, log_v, 0.2, r, c, ng, norm, obj, backend="separable")
    apd, bpd, gd = DC.feasible_dual_and_gap(log_u, log_v, 0.2, r, c, ng, norm, obj, backend="dense")
    assert torch.equal(ap0, aps) and torch.equal(bp0, bps)
    assert g0 == gs
    assert float((aps - apd).abs().max()) <= 1e-10
    assert float((bps - bpd).abs().max()) <= 1e-10
    assert abs(gs - gd) <= 1e-10


@requires_cuda
def test_ctransform_separable_no_NxN_alloc():
    """The separable path's peak device memory stays below a tenth of one dense
    fp64 N x N cost, and it is faster than the dense backend. CUDA only: the
    check reads the allocator's peak."""
    import time
    dev = torch.device("cuda")
    ng, norm = 128, "sql2"
    N = ng * ng
    p = torch.randn(N, dtype=torch.float64, device=dev)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    base = torch.cuda.memory_allocated(dev)
    q = DC._ctransform_separable(p, ng, norm)
    torch.cuda.synchronize(dev)
    peak = torch.cuda.max_memory_allocated(dev) - base
    nbyn_bytes = N * N * 8
    assert peak < 0.1 * nbyn_bytes
    assert q.shape == (N,)
    q_dense = DC._ctransform(p, ng, norm)
    assert float((q - q_dense).abs().max()) <= 1e-10
    for _ in range(2):                            # untimed warm-up
        DC._ctransform_separable(p, ng, norm); DC._ctransform(p, ng, norm)
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    for _ in range(10):
        DC._ctransform_separable(p, ng, norm)
    torch.cuda.synchronize(dev); t_sep = time.perf_counter() - t0
    torch.cuda.synchronize(dev); t0 = time.perf_counter()
    for _ in range(10):
        DC._ctransform(p, ng, norm)
    torch.cuda.synchronize(dev); t_dense = time.perf_counter() - t0
    assert t_sep < t_dense


@requires_cuda
@pytest.mark.parametrize("norm", ["l1", "sql2"])
def test_ctransform_kernel_equals_torch_and_propagates_nan(norm, monkeypatch):
    """The fused c-transform kernel equals the torch passes bit for bit, and one
    NaN in the potential makes every output NaN on both paths."""
    from src.methods import cuda_ext
    cuda_ext.load()
    ng = 32
    p = torch.randn(ng * ng, dtype=torch.float64, device="cuda")
    p_nan = p.clone()
    p_nan[5] = float("nan")
    q_kernel = [DC._ctransform_separable(x, ng, norm) for x in (p, p_nan)]
    monkeypatch.setenv("BDRS_CT_KERNEL", "0")
    q_torch = [DC._ctransform_separable(x, ng, norm) for x in (p, p_nan)]
    assert torch.equal(q_kernel[0], q_torch[0])
    assert bool(torch.isnan(q_torch[1]).all()) and bool(torch.isnan(q_kernel[1]).all())


@pytest.mark.parametrize("norm", ["l1", "sql2"])
@pytest.mark.parametrize("variant", ["beta", "alpha", "alternating"])
def test_ctransform_feasible_by_construction(norm, variant):
    ng = 8
    rng = np.random.default_rng(0)
    alpha = torch.tensor(rng.standard_normal(ng * ng) * 0.3, dtype=torch.float64)
    beta = torch.tensor(rng.standard_normal(ng * ng) * 0.3, dtype=torch.float64)
    ap, bp = DC.dual_cleanup_ctransform(alpha, beta, ng, norm, variant=variant)
    assert _dual_feas_F(ap, bp, ng, norm) <= 1e-9


def test_ctransform_matches_exact_dual():
    """The c-transform of POT's optimal duals is feasible and attains the optimum."""
    ot = pytest.importorskip("ot")
    ng, norm = 6, "l1"
    C = _C(ng, norm); M = C.numpy()
    rng = np.random.default_rng(1)
    a = rng.random(ng * ng) + 0.05; a /= a.sum()
    b = rng.random(ng * ng) + 0.05; b /= b.sum()
    G, log = ot.emd(a, b, M, log=True)
    cost = float((G * M).sum())
    u, v = log["u"], log["v"]
    r = torch.tensor(a); c = torch.tensor(b)
    alpha = torch.tensor(u, dtype=torch.float64); beta = torch.tensor(v, dtype=torch.float64)
    res = best_ctransform(alpha, beta, ng, norm, r, c, objective=cost)
    best = res["best"]
    assert best["r_D"] <= 1e-9
    assert abs(best["gap"]) <= 1e-7
    assert abs(best["dual_obj"] - cost) <= 1e-7


def test_feasible_dual_and_gap_certifies_optimum():
    """On POT's optimal duals: a feasible pair and g_feas ≈ 0."""
    ot = pytest.importorskip("ot")
    ng, norm = 6, "l1"
    C = _C(ng, norm); M = C.numpy()
    rng = np.random.default_rng(3)
    a = rng.random(ng * ng) + 0.05; a /= a.sum()
    b = rng.random(ng * ng) + 0.05; b /= b.sum()
    G, log = ot.emd(a, b, M, log=True)
    cost = float((G * M).sum())
    r = torch.tensor(a); c = torch.tensor(b)
    # α = η·log u, so with η = 1 log_u is the potential itself.
    log_u = torch.tensor(log["u"], dtype=torch.float64)
    log_v = torch.tensor(log["v"], dtype=torch.float64)
    ap, bp, g_feas = DC.feasible_dual_and_gap(log_u, log_v, 1.0, r, c, ng, norm, cost)
    assert _dual_feas_F(ap, bp, ng, norm) <= 1e-9
    assert abs(g_feas) <= 1e-12


def test_feasible_dual_and_gap_scaling_and_bound():
    """g_feas ≥ 0 here, and the certificate depends only on (α, β) = η·(log u,
    log v): doubling η and halving log u, log v changes nothing."""
    ng, norm = 8, "sql2"
    rng = np.random.default_rng(7)
    a = rng.random(ng * ng) + 0.05; a /= a.sum()
    b = rng.random(ng * ng) + 0.05; b /= b.sum()
    r = torch.tensor(a); c = torch.tensor(b)
    log_u = torch.tensor(rng.standard_normal(ng * ng) * 0.2, dtype=torch.float64)
    log_v = torch.tensor(rng.standard_normal(ng * ng) * 0.2, dtype=torch.float64)
    obj = 0.5
    _, _, g1 = DC.feasible_dual_and_gap(log_u, log_v, 0.2, r, c, ng, norm, obj)
    _, _, g2 = DC.feasible_dual_and_gap(log_u * 0.5, log_v * 0.5, 0.4, r, c, ng, norm, obj)
    assert g1 >= -1e-9
    assert abs(g1 - g2) <= 1e-9


def test_ctransform_repairs_drifted_dual():
    """POT's optimal duals plus 1e-3 Gaussian noise on every entry: the
    c-transform restores feasibility and keeps |g| ≤ 1e-2."""
    ot = pytest.importorskip("ot")
    ng, norm = 6, "sql2"
    C = _C(ng, norm); M = C.numpy()
    rng = np.random.default_rng(2)
    a = rng.random(ng * ng) + 0.05; a /= a.sum()
    b = rng.random(ng * ng) + 0.05; b /= b.sum()
    G, log = ot.emd(a, b, M, log=True)
    cost = float((G * M).sum())
    u = log["u"] + rng.standard_normal(ng * ng) * 1e-3
    v = log["v"] + rng.standard_normal(ng * ng) * 1e-3
    r = torch.tensor(a); c = torch.tensor(b)
    alpha = torch.tensor(u, dtype=torch.float64); beta = torch.tensor(v, dtype=torch.float64)
    rD_before = _dual_feas_F(alpha, beta, ng, norm)
    res = best_ctransform(alpha, beta, ng, norm, r, c, objective=cost)
    assert res["best"]["r_D"] <= 1e-9
    assert res["best"]["r_D"] < rD_before
    assert abs(res["best"]["gap"]) <= 1e-2


@pytest.mark.parametrize("m,n", [(12, 8), (8, 12)], ids=["12x8", "8x12"])
@pytest.mark.parametrize("variant", ["beta", "alpha", "alternating"])
def test_storedC_ctransform_rectangular_feasible(m, n, variant):
    from tests.methods.dense_random import make_random_dense_dot
    C, _, _ = make_random_dense_dot(m, n, seed=4)
    rng = np.random.default_rng(0)
    alpha = torch.tensor(rng.standard_normal(m) * 0.3, dtype=torch.float64)
    beta = torch.tensor(rng.standard_normal(n) * 0.3, dtype=torch.float64)
    ap, bp = DC.dual_cleanup_ctransform_storedC(alpha, beta, C, variant=variant)
    assert ap.shape == (m,) and bp.shape == (n,)
    assert float((ap[:, None] + bp[None, :] - C).clamp_min(0).max()) <= 1e-12
    assert DC._dual_feas_F_storedC(ap, bp, C) <= 1e-9


@pytest.mark.parametrize("m,n", [(10, 7), (7, 10)], ids=["10x7", "7x10"])
def test_storedC_feasible_dual_and_gap_certifies_rectangular_optimum(m, n):
    """On POT's optimal duals of a rectangular LP: a feasible pair and g_feas ≈ 0."""
    ot = pytest.importorskip("ot")
    from tests.methods.dense_random import make_random_dense_dot
    C, r, c = make_random_dense_dot(m, n, seed=9)
    M = C.numpy()
    G, log = ot.emd(r.numpy(), c.numpy(), M, log=True)
    cost = float((G * M).sum())
    # α = η·log u, so with η = 1 log_u is the potential itself.
    log_u = torch.tensor(log["u"], dtype=torch.float64)
    log_v = torch.tensor(log["v"], dtype=torch.float64)
    ap, bp, g_feas = DC.feasible_dual_and_gap_storedC(
        log_u, log_v, 1.0, r, c, C, cost)
    assert ap.shape == (m,) and bp.shape == (n,)
    assert DC._dual_feas_F_storedC(ap, bp, C) <= 1e-9
    assert abs(g_feas) <= 1e-12


def test_the_grid_ctransform_refuses_a_non_separable_cost():
    """Any cost but l1 and sql2 is refused, not evaluated as a separable cost
    under its name."""
    a = torch.zeros(16, dtype=torch.float64)
    for backend in (None, "separable", "dense"):
        with pytest.raises(ValueError, match="separable"):
            DC.dual_cleanup_ctransform(a, a, 4, "l2", backend=backend)
