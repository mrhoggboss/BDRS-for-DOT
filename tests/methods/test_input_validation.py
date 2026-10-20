"""Inputs the solvers refuse or convert before iterating, and two edge cases:
a K_max = 0 solve and a CUDA extension that fails to load."""

import pytest

torch = pytest.importorskip("torch")

from src.methods import factored_cuda as FC
from src.methods.factored_torch import bregman_dr_solve_pointcloud
from src.methods.grid_cmax import grid_cmax
from src.methods.grid_cuda import bregman_dr_solve_grid_cuda
from src.methods.grid_torch import bregman_dr_solve_factored_logdomain
from src.methods.params import BregmanDRParams
from tests.helpers import requires_cuda


def _params(iters=10):
    return BregmanDRParams(eta=0.1, max_iters=iters, epsilon_primal=0.0,
                           epsilon_dual=0.0, epsilon_gap=0.0, check_every=max(1, iters))


def test_the_general_cuda_entry_casts_fp32_marginals(monkeypatch):
    """Both the session and the check loop receive fp64 marginals; an fp32
    check marginal would make the delivered gap's dot product raise after the
    whole solve."""
    seen = {}

    class _Stop(Exception):
        pass

    class _Ext:
        class DenseFactoredSession:
            def __init__(self, **kw):
                seen["r_it"], seen["c_it"] = kw["r"].dtype, kw["c"].dtype

    def _loop(sess, cost_rows, *, r_chk, c_chk, **kw):
        seen["r_chk"], seen["c_chk"] = r_chk.dtype, c_chk.dtype
        raise _Stop

    C = torch.rand(5, 4, dtype=torch.float64)
    r = torch.full((5,), 0.2, dtype=torch.float32)
    c = torch.full((4,), 0.25, dtype=torch.float32)
    monkeypatch.setattr(FC, "_factored_session_loop", _loop)
    with pytest.raises(_Stop):
        FC.bregman_dr_solve_general_factored_cuda(C, r, c, _params(), _ext_override=_Ext())
    assert set(seen.values()) == {torch.float64}


def test_a_grid_too_small_to_normalize_is_refused():
    """A one-point grid has C_max = 0 and the CUDA session needs n_g >= 4; both
    raise ValueError, not a NaN or a C++ error."""
    with pytest.raises(ValueError, match="at least 2"):
        grid_cmax("l1", 1)
    r = torch.ones(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 2"):
        bregman_dr_solve_factored_logdomain(r, r, _params(), grid_size=1, cost_type="l1")
    r4 = torch.full((4,), 0.25, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least 4"):
        bregman_dr_solve_grid_cuda(None, r4, r4, _params(), cost_type="l1", grid_size=2)


def test_a_stored_cost_must_already_have_max_one():
    """The Normalize step can rescale only cost_fn; a stored_C whose max is not
    1 would be iterated unscaled while every check reads cost_fn / C_max."""
    C = 2.0 * torch.rand(6, 5, dtype=torch.float64) + 0.5
    r = torch.full((6,), 1 / 6, dtype=torch.float64)
    c = torch.full((5,), 0.2, dtype=torch.float64)
    with pytest.raises(ValueError, match="max 1"):
        bregman_dr_solve_pointcloud(r, c, _params(), cost_fn=lambda i0, i1: C[i0:i1],
                                    stored_C=C, normalize_cost=True)
    Cn = C / C.max()
    _, info = bregman_dr_solve_pointcloud(r, c, _params(), cost_fn=lambda i0, i1: Cn[i0:i1],
                                          stored_C=Cn, normalize_cost=True, return_plan=False)
    assert info["iters"] == 10


@requires_cuda
def test_zero_iterations_deliver_from_a_defined_state():
    """With K_max = 0 no step runs, so every session delivers its initial
    log u = 0 and a gap computed from it."""
    from src.methods.geo_cost import grid_coords
    ng, N = 8, 64
    r = torch.full((N,), 1 / N, dtype=torch.float64, device="cuda")
    _, info = bregman_dr_solve_grid_cuda(None, r, r, _params(0), cost_type="l1",
                                         grid_size=ng, return_plan=False)
    assert torch.count_nonzero(info["log_u"]) == 0
    P = grid_coords(ng, device="cuda")
    _, info = FC.bregman_dr_solve_pointcloud_cuda(P, P, r, r, _params(0), cost_type="l1",
                                                   c_max=float(2 * (ng - 1)), return_plan=False)
    assert torch.count_nonzero(info["log_u"]) == 0
    C = torch.rand(N, N, dtype=torch.float64, device="cuda")
    _, info = FC.bregman_dr_solve_general_factored_cuda(C, r, r, _params(0), return_plan=False)
    assert torch.count_nonzero(info["log_u"]) == 0


def test_a_failed_extension_load_warns_once(monkeypatch):
    """A failed extension load sends the General solver to the chunked torch
    reductions with one RuntimeWarning per process."""
    import warnings
    from src.methods import cuda_ext
    from src.methods import factored_torch as FT

    def _broken():
        raise RuntimeError("compile error")
    monkeypatch.setattr(cuda_ext, "load", _broken)
    monkeypatch.setattr(FT, "_FALLBACK_WARNED", False)
    monkeypatch.delenv("BDRS_FACTORED_KERNELS", raising=False)
    dev = torch.device("cuda")
    with pytest.warns(RuntimeWarning, match="falls back"):
        assert FT._stored_kernels_ext(dev) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert FT._stored_kernels_ext(dev) is None
