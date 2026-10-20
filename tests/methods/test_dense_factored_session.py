"""The captured General session (``DenseFactoredSession``) against the eager
stored-C kernel loop, compared with ``torch.equal``.

The eager kernels take ``neg_scale = -k/eta`` by value, so a captured graph of
them would replay the capture-time k forever; the session reads k from device
memory instead. Its kernels keep the eager ``__dmul_rn``/``__dadd_rn`` terms
(no FMA contraction) and the two-stage column reduction, so arithmetic and
reduction order both match.
"""

import pytest
import torch

from tests.helpers import requires_cuda

SHAPES = [(64, 64), (40, 24), (24, 40)]


def _ext():
    from src.methods import cuda_ext
    ext = cuda_ext.load()
    for sym in ("DenseFactoredSession", "dense_row_lse_scaled", "dense_col_lse_scaled"):
        if not hasattr(ext, sym):
            pytest.skip(f"extension lacks {sym}")
    return ext


def _instance(m: int, n: int, seed: int = 20260914):
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = torch.rand(m, n, generator=g, dtype=torch.float64).cuda().contiguous()
    C = C / C.max()
    C = C.contiguous()
    r = torch.rand(m, generator=g, dtype=torch.float64).cuda() + 1e-3
    c = torch.rand(n, generator=g, dtype=torch.float64).cuda() + 1e-3
    r = r / r.sum()
    c = c / c.sum()
    return C, r, c


def _eager_reference(C, r, c, eta, iters, rows_per_chunk):
    """The eager kernel loop of ``factored_torch.bregman_dr_solve_pointcloud``,
    transcribed because the solver's ``info`` exposes log_u, log_v but not A, B."""
    from src.methods.factored_torch import (
        _stored_kernels_ext,
        _stored_row_lse,
        _stored_col_lse,
    )
    import math
    ext = _stored_kernels_ext(C.device)
    assert ext is not None, "BDRS_FACTORED_KERNELS=1 must select the kernel path"
    m, n = C.shape
    dt, dev = C.dtype, C.device
    A = torch.full((m,), -float(math.log(m)), dtype=dt, device=dev)
    B = torch.full((n,), -float(math.log(n)), dtype=dt, device=dev)
    log_u = torch.zeros(m, dtype=dt, device=dev)
    log_v = torch.zeros(n, dtype=dt, device=dev)
    lse_u = torch.empty(m, dtype=dt, device=dev)
    lse_v = torch.empty(n, dtype=dt, device=dev)
    log_r, log_c = r.log(), c.log()
    n_chunks = (m + rows_per_chunk - 1) // rows_per_chunk
    scratch = (torch.empty(n_chunks * n, dtype=dt, device=dev),
               torch.empty(n_chunks * n, dtype=dt, device=dev))
    k = 0
    while k < iters:
        k += 1
        scale = k / eta
        beta_tilde = B + log_v
        _stored_row_lse(lse_u, C, beta_tilde, -scale, ext)
        A_old = A
        A = log_r - lse_u
        log_u = A - A_old
        _stored_col_lse(lse_v, C, A, -scale, ext, scratch, rows_per_chunk)
        B_old = B
        B = log_c - lse_v
        log_v = B - B_old
    return A, B, log_u, log_v


@requires_cuda
@pytest.mark.parametrize("m,n", SHAPES)
def test_session_matches_eager_kernel_loop_bitwise(m, n, monkeypatch):
    monkeypatch.setenv("BDRS_FACTORED_KERNELS", "1")
    ext = _ext()
    C, r, c = _instance(m, n)
    eta, iters, rpc = 0.17, 200, 128

    A_ref, B_ref, lu_ref, lv_ref = _eager_reference(C, r, c, eta, iters, rpc)

    sess = ext.DenseFactoredSession(C=C, r=r, c=c, eta=eta,
                                    rows_per_chunk=rpc)
    sess.step(iters)
    sess.synchronize()

    assert sess.iters == iters
    assert torch.equal(sess.A, A_ref), f"A differs, max|d|={(sess.A - A_ref).abs().max().item():.3e}"
    assert torch.equal(sess.B, B_ref), f"B differs, max|d|={(sess.B - B_ref).abs().max().item():.3e}"
    assert torch.equal(sess.log_u, lu_ref), \
        f"log_u differs, max|d|={(sess.log_u - lu_ref).abs().max().item():.3e}"
    assert torch.equal(sess.log_v, lv_ref), \
        f"log_v differs, max|d|={(sess.log_v - lv_ref).abs().max().item():.3e}"


@requires_cuda
def test_device_k_tracks_host_k_over_10k_iterations(monkeypatch):
    """A graph that replayed its capture-time k would still give a plausible
    state, so the device k is checked directly as well as bitwise."""
    monkeypatch.setenv("BDRS_FACTORED_KERNELS", "1")
    ext = _ext()
    C, r, c = _instance(24, 24)
    eta, rpc = 0.17, 128
    sess = ext.DenseFactoredSession(C=C, r=r, c=c, eta=eta,
                                    rows_per_chunk=rpc)
    for batch in (1, 9, 90, 900, 9000):
        sess.step(batch)
    sess.synchronize()
    total = 1 + 9 + 90 + 900 + 9000
    assert sess.iters == total
    # k_device holds the next iteration's k, so after `total` replays it reads total+1.
    assert int(sess.k_device.item()) == total + 1, \
        f"device k is {int(sess.k_device.item())}, expected {total + 1}"

    A_ref, B_ref, lu_ref, lv_ref = _eager_reference(C, r, c, eta, total, rpc)
    assert torch.equal(sess.A, A_ref)
    assert torch.equal(sess.B, B_ref)
    assert torch.equal(sess.log_u, lu_ref)
    assert torch.equal(sess.log_v, lv_ref)


@requires_cuda
def test_batching_does_not_change_the_result(monkeypatch):
    monkeypatch.setenv("BDRS_FACTORED_KERNELS", "1")
    ext = _ext()
    C, r, c = _instance(48, 48)
    eta, rpc = 0.17, 128
    a = ext.DenseFactoredSession(C=C, r=r, c=c, eta=eta, rows_per_chunk=rpc)
    a.step(150); a.synchronize()
    b = ext.DenseFactoredSession(C=C, r=r, c=c, eta=eta, rows_per_chunk=rpc)
    for _ in range(150):
        b.step(1)
    b.synchronize()
    assert torch.equal(a.A, b.A)
    assert torch.equal(a.B, b.B)
    assert torch.equal(a.log_u, b.log_u)
    assert torch.equal(a.log_v, b.log_v)


@requires_cuda
def test_rejects_noncontiguous_C(monkeypatch):
    monkeypatch.setenv("BDRS_FACTORED_KERNELS", "1")
    ext = _ext()
    C, r, c = _instance(32, 32)
    with pytest.raises(RuntimeError, match="contiguous"):
        ext.DenseFactoredSession(C=C.t(), r=c, c=r, eta=0.17,
                                 rows_per_chunk=128)


@requires_cuda
@pytest.mark.parametrize("m,n", [(64, 64), (40, 24)])
def test_solver_entry_matches_eager_solver_bitwise(m, n, monkeypatch):
    """The session solver entry, which run_general calls, against the eager
    solver: same iteration count, log u, log v and plan."""
    monkeypatch.setenv("BDRS_FACTORED_KERNELS", "1")
    _ext()
    from src.methods.factored_torch import bregman_dr_solve_general_factored
    from src.methods.params import BregmanDRParams
    from src.methods.factored_cuda import bregman_dr_solve_general_factored_cuda

    C, r, c = _instance(m, n)
    params = BregmanDRParams(eta=0.17, max_iters=200, check_every=201,
                             epsilon_primal=0.0, epsilon_dual=float("inf"),
                             epsilon_gap=float("inf"), verbosity=False)

    monkeypatch.setenv("BDRS_FACTORED_SESSION", "1")
    X_s, info_s = bregman_dr_solve_general_factored_cuda(C, r, c, params, return_plan=True)
    monkeypatch.setenv("BDRS_FACTORED_SESSION", "0")
    X_e, info_e = bregman_dr_solve_general_factored(C, r, c, params, return_plan=True)

    assert info_s["factored_session"] is True
    assert int(info_s["iters"]) == int(info_e["iters"]) == 200
    assert torch.equal(info_s["log_u"], info_e["log_u"]), \
        f"log_u differs, max|d|={(info_s['log_u'] - info_e['log_u']).abs().max().item():.3e}"
    assert torch.equal(info_s["log_v"], info_e["log_v"]), \
        f"log_v differs, max|d|={(info_s['log_v'] - info_e['log_v']).abs().max().item():.3e}"
    assert torch.equal(X_s, X_e), \
        f"plan differs, max|d|={(X_s - X_e).abs().max().item():.3e}"


@requires_cuda
def test_kill_switch_selects_the_eager_loop(monkeypatch):
    monkeypatch.setenv("BDRS_FACTORED_KERNELS", "1")
    monkeypatch.setenv("BDRS_FACTORED_SESSION", "0")
    _ext()
    from src.methods.params import BregmanDRParams
    from src.methods.factored_cuda import bregman_dr_solve_general_factored_cuda
    C, r, c = _instance(32, 32)
    params = BregmanDRParams(eta=0.17, max_iters=50, check_every=51,
                             epsilon_primal=0.0, epsilon_dual=float("inf"),
                             epsilon_gap=float("inf"), verbosity=False)
    _, info = bregman_dr_solve_general_factored_cuda(C, r, c, params, return_plan=False)
    assert info["factored_session"] is False
    assert info["factored_kernels"] is True      # the eager loop still uses the kernels


@requires_cuda
def test_session_runs_the_feasible_gap_stop(monkeypatch):
    """The session loop's ⟨C, X⟩ call runs only under the feasible-gap stop
    and after the primal proxy passes, so the fixed-iteration tests above,
    whose check period exceeds K_max, never reach it."""
    monkeypatch.setenv("BDRS_FACTORED_KERNELS", "1")
    monkeypatch.setenv("BDRS_FACTORED_SESSION", "1")
    _ext()
    from src.methods.params import BregmanDRParams
    from src.methods.factored_cuda import bregman_dr_solve_general_factored_cuda
    from src.methods import dual_cleanup as DC

    C, r, c = _instance(32, 32)
    tol = 1e-4
    gap_fn = DC.make_dense_gap_stop_fn(C, r, c)
    params = BregmanDRParams(eta=0.17, max_iters=200000, check_every=100,
                             epsilon_primal=tol,
                             epsilon_dual=float("inf"), epsilon_gap=float("inf"),
                             verbosity=False)
    _, info = bregman_dr_solve_general_factored_cuda(
        C, r, c, params, gap_stop_fn=gap_fn, gap_stop_tol=tol, return_plan=False)
    assert info["factored_session"] is True
    assert info["native_termination"] == "feasible_gap_below_tol", info["native_termination"]
    assert info["proxy_passes"] >= 1
    assert info["sep_obj_checks"] >= 1          # the _pointcloud_trace_CX call site
