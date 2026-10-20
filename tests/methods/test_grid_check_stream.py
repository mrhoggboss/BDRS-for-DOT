"""The Grid check passes computed in blocks of rows equal the single-block ones.

The blocks split an axis the pass does not reduce, so each output reduces the
same elements. ``BDRS_GRID_CHECK_CHUNK_ELEMS`` forces the block size and
``BDRS_GRID_CHECK_STREAM=0`` gives one block. The comparison is bitwise except
where torch orders a reduction by the launch size: over a strided axis on CPU,
and on CUDA at n_g = 1024.
"""

import math

import pytest
import torch

from src.methods import grid_cuda
from src.methods.grid_cuda import (
    _add_factored_kkt,
    _check_invariants,
    _compute_factored_objective_t,
    _compute_factored_primal_res_ex,
)
from src.methods.grid_factored import _check_chunk_rows, factored_col_sums, factored_row_sums
from tests.helpers import requires_cuda


def _set_rows(monkeypatch, ng, rows):
    """rows=None: one block; otherwise blocks of ``rows`` rows."""
    if rows is None:
        monkeypatch.setenv("BDRS_GRID_CHECK_STREAM", "0")
        monkeypatch.delenv("BDRS_GRID_CHECK_CHUNK_ELEMS", raising=False)
    else:
        monkeypatch.setenv("BDRS_GRID_CHECK_STREAM", "1")
        monkeypatch.setenv("BDRS_GRID_CHECK_CHUNK_ELEMS", str(rows * ng * ng))
        assert _check_chunk_rows(ng, ng * ng) == rows


def _state(ng, cost_type, k_iter, eta, scale, seed, device):
    """A factored state (ξ, ζ, log u, log v) and marginals. "large" puts ξ near
    +1e3 and ζ near −1e3 − log N, so every LSE shifts by about 1e3 while ξ + ζ
    stays near −log N."""
    g = torch.Generator().manual_seed(seed)
    N = ng * ng
    xs = torch.linspace(0.0, 1.0, ng, dtype=torch.float64)
    smooth = (torch.sin(3.0 * xs)[:, None] * torch.cos(2.0 * xs)[None, :]).reshape(N)
    if scale == "realistic":
        A = -math.log(N) + 2.0 * smooth + 0.5 * torch.randn(N, generator=g, dtype=torch.float64)
        B = -math.log(N) - 1.5 * smooth + 0.5 * torch.randn(N, generator=g, dtype=torch.float64)
    else:
        A = 1e3 + 2.0 * smooth + 0.5 * torch.randn(N, generator=g, dtype=torch.float64)
        B = -1e3 - math.log(N) + 0.5 * torch.randn(N, generator=g, dtype=torch.float64)
    log_u = torch.randn(N, generator=g, dtype=torch.float64)
    log_v = torch.randn(N, generator=g, dtype=torch.float64)
    r = torch.rand(N, generator=g, dtype=torch.float64) + 1e-3
    c = torch.rand(N, generator=g, dtype=torch.float64) + 1e-3
    r, c = r / r.sum(), c / c.sum()
    to = dict(device=device)
    return {
        "A": A.to(**to), "B": B.to(**to), "log_u": log_u.to(**to), "log_v": log_v.to(**to),
        "r": r.to(**to), "c": c.to(**to), "ng": ng, "cost_type": cost_type,
        "k_iter": k_iter, "eta": eta,
    }


def _evaluate(st):
    """Every output of the check passes."""
    ng, ct, eta, k = st["ng"], st["cost_type"], st["eta"], st["k_iter"]
    inv = _check_invariants(ng, ct, eta, st["A"].device)
    proxy, logS = _compute_factored_primal_res_ex(
        st["A"], st["B"], st["r"], st["c"], ng, ct, eta, k, invariants=inv)
    obj_fresh = _compute_factored_objective_t(st["A"], st["B"], ng, ct, eta, k, invariants=inv)
    obj_reuse = _compute_factored_objective_t(st["A"], st["B"], ng, ct, eta, k,
                                              S=logS, invariants=inv)
    info = {"A": st["A"], "B": st["B"], "log_u": st["log_u"], "log_v": st["log_v"],
            "iters": k}
    _add_factored_kkt(info, st["r"], st["c"], ng, ct, eta, invariants=inv, native_dual=False)
    kkt = {key: info[key] for key in
           ("primal_res_L2", "dual_res_F", "gap_abs", "objective", "native_dual_residual")}
    return {"proxy": proxy, "logS": logS, "obj_fresh": obj_fresh,
            "obj_reuse": obj_reuse, "kkt": kkt}


def _same_float(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b


def _assert_identical(got, ref, *, cpu_ulps=False):
    """Blocked == single block. The row proxy and its S table are compared
    bitwise. With ``cpu_ulps`` the objective, the values derived from it and
    the full r_P may differ by a few ulps: on CPU, torch's order of a reduction
    over a strided axis can depend on the block size. On CUDA, where the Grid
    solver runs, the comparison is bitwise up to n_g = 512."""
    assert _same_float(got["proxy"], ref["proxy"])
    assert torch.equal(got["logS"], ref["logS"])
    obj_scale = float(ref["obj_fresh"].abs())

    def _match(name, a, b):
        a, b = float(a), float(b)
        if not cpu_ulps or math.isnan(b):
            assert _same_float(a, b), name
            return
        scale = max(abs(b), obj_scale)
        assert abs(a - b) <= 1e-15 * scale, f"{name}: diff {abs(a - b):.3e} at scale {scale:.3e}"

    _match("obj_fresh", got["obj_fresh"], ref["obj_fresh"])
    _match("obj_reuse", got["obj_reuse"], ref["obj_reuse"])
    for key, val in ref["kkt"].items():
        if isinstance(val, float):
            _match(f"kkt.{key}", got["kkt"][key], val)
        else:
            assert got["kkt"][key] == val, key


def test_chunk_rows_default_and_switches(monkeypatch):
    monkeypatch.delenv("BDRS_GRID_CHECK_STREAM", raising=False)
    monkeypatch.delenv("BDRS_GRID_CHECK_CHUNK_ELEMS", raising=False)
    assert [_check_chunk_rows(n, n * n) for n in (32, 64, 128, 256, 512, 1024)] == \
        [32, 64, 128, 256, 64, 16]
    monkeypatch.setenv("BDRS_GRID_CHECK_STREAM", "0")
    assert _check_chunk_rows(1024, 1024 * 1024) == 1024
    monkeypatch.setenv("BDRS_GRID_CHECK_STREAM", "1")
    monkeypatch.setenv("BDRS_GRID_CHECK_CHUNK_ELEMS", "1")
    assert _check_chunk_rows(1024, 1024 * 1024) == 1     # floor of one row
    monkeypatch.setenv("BDRS_GRID_CHECK_CHUNK_ELEMS", "0")
    with pytest.raises(ValueError):
        _check_chunk_rows(64, 64 * 64)


@pytest.mark.parametrize("scale", ["realistic", "large"])
@pytest.mark.parametrize("k_iter", [1, 1000, 50000])
@pytest.mark.parametrize("cost_type,eta", [("l1", 0.03), ("sql2", 0.012), ("l1", 0.05)])
@pytest.mark.parametrize("ng", [8, 16, 32])
def test_streamed_check_bitwise_cpu(monkeypatch, ng, cost_type, eta, k_iter, scale):
    st = _state(ng, cost_type, k_iter, eta, scale, seed=ng * 7 + k_iter % 97,
                device=torch.device("cpu"))
    _set_rows(monkeypatch, ng, None)
    ref = _evaluate(st)
    assert math.isfinite(ref["proxy"]) and torch.isfinite(ref["obj_fresh"])
    for rows in sorted({1, 3, ng // 2, ng}):
        _set_rows(monkeypatch, ng, rows)
        _assert_identical(_evaluate(st), ref, cpu_ulps=True)


def test_streamed_proxy_s_table_feeds_objective_cpu(monkeypatch):
    """The objective that reuses the proxy's blocked S equals, bitwise, the one
    that recomputes it."""
    ng = 32
    st = _state(ng, "sql2", 4000, 0.02, "realistic", seed=5, device=torch.device("cpu"))
    _set_rows(monkeypatch, ng, 3)
    out = _evaluate(st)
    assert torch.equal(out["obj_fresh"], out["obj_reuse"])


def _unstreamed_reference(st):
    """The single-block expressions written out, independent of the blocking
    helpers."""
    ng, ct, eta, k = st["ng"], st["cost_type"], st["eta"], st["k_iter"]
    inv = _check_invariants(ng, ct, eta, st["A"].device)
    kL_mat = grid_cuda._kl_mat(inv, k)
    A2, B2 = st["A"].view(ng, ng), st["B"].view(ng, ng)
    M_rR = kL_mat.unsqueeze(1) + B2.T.unsqueeze(0)
    logS_row = torch.logsumexp(M_rR, dim=2)
    M_r2 = kL_mat.unsqueeze(0) + logS_row.unsqueeze(1)
    lse_row = torch.logsumexp(M_r2, dim=2)
    row_sums = (A2 + lse_row).exp().reshape(ng * ng)
    proxy = float((row_sums - st["r"].reshape(ng * ng)).norm(2).item())
    psi_mat = inv["psi_mat"]
    S = torch.logsumexp(kL_mat.unsqueeze(2) + B2.unsqueeze(0), dim=1)
    T = torch.logsumexp(A2.unsqueeze(2) + S.unsqueeze(1), dim=0)
    R_c = (kL_mat + T).exp()
    Sp = torch.logsumexp(kL_mat.unsqueeze(1) + B2.unsqueeze(0), dim=2)
    Tp = torch.logsumexp(A2.unsqueeze(2) + Sp.unsqueeze(0), dim=1)
    R_r = (kL_mat + Tp).exp()
    obj = (psi_mat * R_r).sum() + (psi_mat * R_c).sum()
    M_cR = kL_mat.T.unsqueeze(1) + A2.T.unsqueeze(0)
    logT_col = torch.logsumexp(M_cR, dim=2)
    M_c2 = kL_mat.T.unsqueeze(0) + logT_col.unsqueeze(1)
    lse_col = torch.logsumexp(M_c2, dim=2)
    col_sums = (B2 + lse_col).exp().reshape(ng * ng)
    primal = (row_sums - st["r"].reshape(-1)).norm(2) + (col_sums - st["c"].reshape(-1)).norm(2)
    return proxy, logS_row, obj, float(primal.item())


@pytest.mark.parametrize("rows", [None, 1, 3, 8, 16])
@pytest.mark.parametrize("cost_type,k_iter", [("l1", 1000), ("sql2", 50000)])
def test_matches_unstreamed_expressions_cpu(monkeypatch, rows, cost_type, k_iter):
    ng = 16
    st = _state(ng, cost_type, k_iter, 0.02, "realistic", seed=11, device=torch.device("cpu"))
    proxy, logS, obj, primal = _unstreamed_reference(st)
    _set_rows(monkeypatch, ng, rows)
    out = _evaluate(st)
    assert out["proxy"] == proxy
    assert torch.equal(out["logS"], logS)
    assert torch.equal(out["obj_fresh"], obj)
    assert out["kkt"]["objective"] == float(obj.item())
    assert out["kkt"]["primal_res_L2"] == primal


def test_streaming_engages(monkeypatch):
    """With 3-row blocks at n_g = 8, each log-sum-exp runs as ceil(8/3) = 3 calls."""
    calls = []
    real = torch.logsumexp

    def counting(x, dim, *a, **kw):
        calls.append(tuple(x.shape))
        return real(x, dim, *a, **kw)

    ng = 8
    st = _state(ng, "l1", 100, 0.03, "realistic", seed=1, device=torch.device("cpu"))
    _set_rows(monkeypatch, ng, 3)
    monkeypatch.setattr(torch, "logsumexp", counting)
    _compute_factored_primal_res_ex(st["A"], st["B"], st["r"], st["c"], ng, "l1", 0.03, 100)
    assert len(calls) == 6
    assert sorted({sh[0] for sh in calls}) == [2, 3]


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("cost_type", ["l1", "sql2"])
def test_delivery_helpers_streamed_cpu(monkeypatch, rows, cost_type):
    """``BdrsFactoredPlan``'s row sums, column sums and ⟨C, X⟩ and the torch
    c-transform are blocked too, with unchanged values."""
    ng = 16
    st = _state(ng, cost_type, 3000, 0.02, "realistic", seed=3, device=torch.device("cpu"))

    _set_rows(monkeypatch, ng, None)
    ref = _evaluate_delivery(st)
    _set_rows(monkeypatch, ng, rows)
    _assert_delivery_identical(_evaluate_delivery(st), ref)


def _evaluate_delivery(st):
    from src.methods.dual_cleanup import _ctransform_separable
    from src.scoring.plan_metrics import BdrsFactoredPlan
    plan = BdrsFactoredPlan(A=st["A"], B=st["B"], ng=st["ng"], cost_type=st["cost_type"],
                            eta=st["eta"], iters=st["k_iter"])
    return (plan.row_sums(), plan.col_sums(), plan.trace_with_C(),
            _ctransform_separable(st["log_u"], st["ng"], st["cost_type"]))


def _assert_delivery_identical(got, ref):
    """Row sums and the c-transform (a minimum) bitwise; the column sums and
    ⟨C, X⟩ include reductions over a strided axis, so on CPU to a few ulps."""
    assert torch.equal(got[0], ref[0])
    assert float((got[1] - ref[1]).abs().max()) <= 1e-15 * float(ref[1].abs().max())
    assert abs(got[2] - ref[2]) <= 1e-15 * abs(ref[2])
    assert torch.equal(got[3], ref[3])


@pytest.mark.parametrize("blocks", [1, 2])
@pytest.mark.parametrize("h,w", [(6, 10), (10, 6)])
def test_rectangular_row_and_col_sums_streamed_cpu(monkeypatch, h, w, blocks):
    """On an h × w grid each pass blocks with its own row size; both
    orientations run, so a pass that used w for h would drop or duplicate rows.
    Row sums and S bitwise; the column chain reduces over a strided axis, so on
    CPU to a few ulps."""
    g = torch.Generator().manual_seed(17)
    A = torch.randn(h, w, generator=g, dtype=torch.float64) - math.log(h * w)
    B = torch.randn(h, w, generator=g, dtype=torch.float64) - math.log(h * w)
    d_r = torch.arange(h, dtype=torch.float64)
    d_c = torch.arange(w, dtype=torch.float64)
    kL_r = -40.0 * (d_r[:, None] - d_r[None, :]).abs() / h
    kL_c = -40.0 * (d_c[:, None] - d_c[None, :]).abs() ** 2 / w ** 2
    monkeypatch.setenv("BDRS_GRID_CHECK_STREAM", "0")
    ref_rows, ref_S = factored_row_sums(A, B, kL_r, kL_c)
    ref_cols = factored_col_sums(A, B, kL_r, kL_c)
    monkeypatch.setenv("BDRS_GRID_CHECK_STREAM", "1")
    monkeypatch.setenv("BDRS_GRID_CHECK_CHUNK_ELEMS", str(blocks * w * h))
    # Both passes of each chain run in more than one block.
    assert _check_chunk_rows(h, w * h) < h and _check_chunk_rows(h, w * w) < h
    got_rows, got_S = factored_row_sums(A, B, kL_r, kL_c)
    got_cols = factored_col_sums(A, B, kL_r, kL_c)
    assert torch.equal(got_rows, ref_rows) and torch.equal(got_S, ref_S)
    assert float((got_cols - ref_cols).abs().max()) <= 1e-15 * float(ref_cols.abs().max())


def _peak_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    return out, torch.cuda.max_memory_allocated() - base


def _fields(check, delivery):
    """Both evaluations flattened into name -> tensor or float."""
    out = {"proxy": check["proxy"], "logS_row": check["logS"],
           "objective_S_none": check["obj_fresh"], "objective_S_reuse": check["obj_reuse"]}
    out.update({f"kkt.{k}": v for k, v in check["kkt"].items()
                if isinstance(v, float) and not math.isnan(v)})
    out.update({"plan.row_sums": delivery[0], "plan.col_sums": delivery[1],
                "plan.trace_with_C": delivery[2], "ctransform_torch": delivery[3]})
    return out


def _compare(got, ref):
    """name -> (bitwise_equal, max relative difference)."""
    res = {}
    for name, r in ref.items():
        g = got[name]
        if isinstance(r, torch.Tensor):
            same = torch.equal(g, r)
            den = max(float(r.abs().max().item()), 1e-300)
            rel = float((g - r).abs().max().item()) / den
        else:
            same = g == r
            rel = abs(g - r) / max(abs(r), 1e-300)
        res[name] = (same, rel)
    return res


@requires_cuda
@pytest.mark.parametrize("ng", [512, 1024])
@pytest.mark.parametrize("cost_type,eta", [("sql2", 0.012), ("l1", 0.03)])
def test_streamed_check_cuda(monkeypatch, ng, cost_type, eta):
    """Bitwise at n_g = 512. At n_g = 1024 torch splits each 8 GiB single-block
    broadcast into 32-bit-indexable pieces and splits strided reductions across
    thread blocks, so the arms may differ by rounding: the bar is 1e-12
    relative. Prints the peak memory of both arms and which fields are bitwise
    equal."""
    st = _state(ng, cost_type, 20000, eta, "realistic", seed=ng, device=torch.device("cuda"))
    monkeypatch.setenv("BDRS_CT_KERNEL", "0")      # exercise the torch c-transform path

    def both():
        return _evaluate(st), _evaluate_delivery(st)

    _set_rows(monkeypatch, ng, None)
    (ref_c, ref_d), peak_single = _peak_bytes(both)
    monkeypatch.delenv("BDRS_GRID_CHECK_STREAM", raising=False)
    monkeypatch.delenv("BDRS_GRID_CHECK_CHUNK_ELEMS", raising=False)
    rows = _check_chunk_rows(ng, ng * ng)
    (got_c, got_d), peak_stream = _peak_bytes(both)

    cmp = _compare(_fields(got_c, got_d), _fields(ref_c, ref_d))
    print(f"\nng={ng} {cost_type} rows={rows}: peak single-block "
          f"{peak_single / 2**30:.3f} GiB, blocked {peak_stream / 2**30:.3f} GiB")
    for name, (same, rel) in cmp.items():
        print(f"  {name:24s} bitwise={same!s:5s} max_rel_diff={rel:.3e}")
    assert peak_stream < 2 * 2**30
    if ng <= 512:
        assert all(same for same, _ in cmp.values()), cmp
    else:
        assert all(rel <= 1e-12 for _, rel in cmp.values()), cmp
