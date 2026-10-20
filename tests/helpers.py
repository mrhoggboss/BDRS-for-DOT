"""Helpers shared by several test modules.

The module imports only numpy, pytest and ``src.protocol`` (torch only inside
``has_cuda``), so it also loads in the torch-free JAX environment of the
SinkhornGrid tests.
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pytest

from src.protocol.instance_record import InstanceSpec, ResultRecord
from src.protocol.regimes import GRID, algo_applies_instance


def has_cuda() -> bool:
    """Whether torch is installed and sees a CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


requires_cuda = pytest.mark.skipif(not has_cuda(), reason="needs a CUDA device")


def random_marginals(n: int, *, seed: int, offset: float = 1e-3, n_zero: int = 0,
                     identical: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Probability vectors r, c of length ``n``, |N(0,1)| + ``offset`` from
    ``default_rng(seed)``, each with ``n_zero`` entries set to exactly 0.
    ``identical`` copies r into c before the zeros are placed, so with
    ``n_zero > 0`` the two in general differ."""
    rng = np.random.default_rng(seed)
    r = np.abs(rng.standard_normal(n)) + offset
    c = r.copy() if identical else np.abs(rng.standard_normal(n)) + offset
    if n_zero:
        r[rng.choice(n, n_zero, replace=False)] = 0.0
        c[rng.choice(n, n_zero, replace=False)] = 0.0
    return r / r.sum(), c / c.sum()


def save_marginals(directory: Path, r: np.ndarray, c: np.ndarray,
                   names: Tuple[str, str] = ("r.npy", "c.npy")) -> Tuple[Path, Path]:
    """Save r and c as ``.npy`` files in ``directory``; return their paths."""
    r_path, c_path = Path(directory) / names[0], Path(directory) / names[1]
    np.save(r_path, r)
    np.save(c_path, c)
    return r_path, c_path


def lattice_spec(directory: Path, *, ng: int, norm: str, cls: str = "WhiteNoise",
                 marginal_paths: Optional[Sequence] = None, **fields) -> InstanceSpec:
    """A DOTmark-shaped InstanceSpec for pair (0, 1) of class ``cls``, with its
    files in ``directory``; ``fields`` sets any other InstanceSpec field."""
    directory = Path(directory)
    r_path, c_path = marginal_paths or (directory / "r.npy", directory / "c.npy")
    fields.setdefault("cost_matrix_path", None)
    fields.setdefault("output_path", str(directory / "out.json"))
    return InstanceSpec(
        instance_id=f"{cls}_n{ng}_i0_j1_{norm}", class_name=cls,
        resolution=ng, i=0, j=1, id_i=1001, id_j=1002, norm=norm,
        marginal_i_path=str(r_path), marginal_j_path=str(c_path), **fields)


def run_main(module, spec: InstanceSpec, directory: Path,
             *extra_args: str) -> Tuple[int, Optional[ResultRecord]]:
    """Run a runner's ``main`` on ``spec`` through files in ``directory``; return
    its exit code and the record it wrote (None when it wrote none)."""
    directory = Path(directory)
    spec_path, out_path = directory / "spec.json", directory / "result.json"
    spec.to_path(spec_path)
    rc = module.main(["--instance", str(spec_path), "--output", str(out_path), *extra_args])
    return rc, (ResultRecord.from_path(out_path) if out_path.exists() else None)


def algo_applies_on_grid(algo: str, norm: str, exceptions=None) -> bool:
    """``algo_applies_instance`` on a Grid instance of cost ``norm`` (every
    regime can consume a Grid instance, so this isolates the norm policy)."""
    return algo_applies_instance(algo, {"norm": norm, "rung": GRID}, exceptions)
