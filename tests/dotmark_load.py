"""DOTmark marginals and the dense l1/sql2 lattice cost, for tests.

The marginals are read from ``data/marginals/<Class>/n<res>/data<res>_<id>.npy``
(image ids 1001 to 1010), which ``python -m src.instances.dotmark_build`` writes;
a test that needs them calls ``require_marginals`` and is skipped when they are
absent.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

from src.instances.dotmark_build import MARGINAL_DIR
from src.runners.common import load_marginal_unit_mass
from src.methods.grid_cost import (
    _build_grid_indices,
    _cost_axis,
)


def _marginal_path(class_name: str, image_id: int, resolution: int) -> Path:
    res = int(resolution)
    return Path(MARGINAL_DIR) / class_name / f"n{res}" / f"data{res}_{int(image_id)}.npy"


def require_marginals(class_name: str, resolution: int, *image_ids: int) -> List[Path]:
    """The marginal files of these DOTmark images; the calling test is skipped
    when one of them has not been built."""
    paths = [_marginal_path(class_name, i, resolution) for i in image_ids]
    if not all(p.is_file() for p in paths):
        pytest.skip(f"marginals for {class_name} n{resolution} not built "
                    f"(run python -m src.instances.dotmark_build)")
    return paths


def load_marginal(
    class_name: str,
    image_id: int,
    resolution: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """One DOTmark marginal as our runners load it: the stored vector divided by
    its sum, not floored."""
    path = _marginal_path(class_name, image_id, resolution)
    if not path.is_file():
        raise FileNotFoundError(
            f"Marginal not found: {path}. Build DOTmark marginals first with "
            f"`python -m src.instances.dotmark_build --resolutions {int(resolution)}`."
        )
    return torch.as_tensor(load_marginal_unit_mass(str(path)), dtype=dtype, device=device)


_dense_cost_cache: Dict[Tuple[int, str, str, str], torch.Tensor] = {}


def get_dense_cost_matrix(
    resolution: int,
    norm: str,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
    row_chunk: int = 2048,
) -> torch.Tensor:
    """The dense (N, N) max-1 lattice cost of ``norm`` (``l1`` or ``sql2``), N = n_g².

    Each axis term φ(|Δ|) is divided by C_max before the two are added, as in
    ``src.methods.grid_cost``, so entries can differ in the last bit from a
    construction that divides their sum once (``tests/methods/grid_reference.py``).
    The result is memoized and the same tensor goes to every caller, so a caller
    must not modify it in place. C is built ``row_chunk`` rows at a time, with
    two (row_chunk, N) temporaries.
    """
    key = (int(resolution), norm, str(device), str(dtype))
    cached = _dense_cost_cache.get(key)
    if cached is not None:
        return cached
    ng = int(resolution)
    N = ng * ng
    cost_axis = _cost_axis(norm, ng, device=device, dtype=dtype)  # (ng, ng)
    a_r_full, a_c_full = _build_grid_indices(ng, device=device)
    # C[i, j] = cost_axis[a_r[i], a_r[j]] + cost_axis[a_c[i], a_c[j]]
    C = torch.empty((N, N), device=device, dtype=dtype)
    for i0 in range(0, N, int(row_chunk)):
        i1 = min(i0 + int(row_chunk), N)
        torch.add(cost_axis[a_r_full[i0:i1]][:, a_r_full],
                  cost_axis[a_c_full[i0:i1]][:, a_c_full],
                  out=C[i0:i1])
    _dense_cost_cache[key] = C
    return C
