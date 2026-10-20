"""The step-size rule η = 0.7·mean(C) (``src/methods/eta.py``).

η enters every iterate, so the pinned values are compared with ==: a rewrite of
the closed form that rounds differently changes every solve.
"""

import pytest
import torch

from src.instances.dense_cost import build_dense_cost
from src.methods.eta import ETA_ALPHA, eta_from_mean, grid_cost_mean

PINNED = {
    (32, "l1"): 0.24062499999999998, (32, "sql2"): 0.12419354838709677,
    (64, "l1"): 0.23697916666666666, (64, "sql2"): 0.12037037037037035,
    (128, "l1"): 0.23515624999999998, (128, "sql2"): 0.11850393700787401,
    (256, "l1"): 0.23424479166666667, (256, "sql2"): 0.11758169934640524,
    (512, "l1"): 0.23378906249999998, (512, "sql2"): 0.11712328767123287,
    (1024, "l1"): 0.23356119791666666, (1024, "sql2"): 0.11689475399152818,
    (32, "l2"): 0.26628049346606464, (32, "inf"): 0.336968994140625,
    (64, "l2"): 0.26214820964731067, (64, "inf"): 0.3317939758300781,
    (128, "l2"): 0.26010706910612474, (128, "inf"): 0.32922449111938473,
    (256, "l2"): 0.25909272514039206, (256, "inf"): 0.3279441380500793,
}


def test_alpha_is_0p7():
    assert ETA_ALPHA == 0.7


@pytest.mark.parametrize("ng,norm", sorted(PINNED))
def test_eta_pinned_values(ng, norm):
    assert eta_from_mean(grid_cost_mean(ng, norm)) == PINNED[(ng, norm)]


@pytest.mark.parametrize("norm", ["l1", "sql2", "l2", "inf"])
@pytest.mark.parametrize("ng", [8, 16])
def test_closed_form_is_the_mean_of_the_built_cost(ng, norm):
    C = build_dense_cost(ng, norm, device="cpu")
    assert grid_cost_mean(ng, norm) == pytest.approx(float(C.mean()), rel=1e-13)


def test_eta_from_mean_accepts_a_tensor():
    C = torch.rand(7, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    assert eta_from_mean(C.mean()) == 0.7 * float(C.mean())


def test_unknown_norm_raises():
    with pytest.raises(ValueError):
        grid_cost_mean(32, "rand_uniform")
