"""Lifting an instance to the format of the solver it is dispatched to.

A Grid instance reaches a Point Cloud solver as the cloud of its pixel
coordinates, and a General solver unchanged (the runner reads or builds the
lattice C).
A Point Cloud instance reaches a General solver through a dense C materialised
once per instance.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.methods.grid_cmax import grid_cmax
from src.protocol.regimes import (
    GENERAL,
    GRID,
    POINT_CLOUD,
    base_algorithm,
    canonical_regime,
    regime_for_algorithm,
)


# A Point Cloud instance is lifted to General only when N = n_g² for one of
# the Grid resolutions; the lifted entry carries resolution = n_g, which the
# dense runners read as the lattice side.
_SUPPORTED_SIDES = (32, 64, 128, 256, 512, 1024)


def _save_npy(path: Path, arr) -> str:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr, dtype=np.float64))
    return str(path)


def _lift_grid_to_geo(ent: Dict[str, Any], scratch: Path) -> Dict[str, Any]:
    """Grid entry -> Point Cloud entry on ``grid_coords(n_g)``. ``resolution`` is
    kept, so run_pointcloud recognises the lifted grid and, for l1/sql2, takes
    the Grid runner's η."""
    from src.methods import geo_cost
    ng = int(ent["resolution"]); norm = ent["norm"]
    coords = geo_cost.grid_coords(ng, device="cpu").numpy()
    d = scratch / f"geo_{ent['instance_id']}"
    out = dict(ent)
    out.update({
        "rung": POINT_CLOUD,
        "support_i_path": _save_npy(d / "support_i.npy", coords),
        "support_j_path": _save_npy(d / "support_j.npy", coords),
        "c_max": grid_cmax(norm, ng),
        "m": ng * ng, "n": ng * ng, "dim": 2,
    })
    return out


def _materialize_geo_cost(ent: Dict[str, Any], scratch: Path,
                          geo_cache: Dict[str, str]) -> str:
    """Path of the dense C of a Point Cloud instance, written once per
    instance_id and reused by every General solver scheduled on it."""
    cached = geo_cache.get(ent["instance_id"])
    if cached and os.path.exists(cached):
        return cached
    import numpy as np
    import torch
    from src.methods import geo_cost
    from src.protocol.instance_record import geo_cost_type
    P = torch.as_tensor(np.load(ent["support_i_path"]), dtype=torch.float64)
    Q = torch.as_tensor(np.load(ent["support_j_path"]), dtype=torch.float64)
    cf = geo_cost.make_metric_cost_fn(P, Q, geo_cost_type(ent["norm"]), float(ent["c_max"]))
    C = geo_cost.materialize_cost(cf, int(ent["m"]), int(ent["n"]), device="cpu")
    path = str(scratch / f"geocost_{ent['instance_id']}.npy")
    np.save(path, C.numpy())
    geo_cache[ent["instance_id"]] = path
    return path


def _prepare_ent_for_algo(ent: Dict[str, Any], algo: str, *, scratch: Path,
                          geo_cache: Dict[str, str], geo_dense_cap: int
                          ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The entry in the form ``algo``'s runner consumes: ``(entry, None)``, or
    ``(None, reason)`` when that runner cannot take the instance."""
    ent = {**ent, "rung": canonical_regime(ent.get("rung") or GRID)}
    rung = ent["rung"]
    regime = regime_for_algorithm(algo)
    base = base_algorithm(algo)

    if regime == GRID:
        return ent, None                                  # grid algo, grid instance

    if regime == POINT_CLOUD:
        if rung == GRID:
            if base in ("HALO", "M3S"):
                return ent, None                          # grid-mode runner on the grid spec
            return _lift_grid_to_geo(ent, scratch), None
        # M3S's shim runs image grids only.
        if base == "M3S":
            return None, "m3s-image-grid-only"
        # LAMP's matrix-free kernels read three coordinates and assume m = n.
        if base == "LAMPPointCloud" and int(ent.get("dim") or 0) > 3:
            return None, "lamppointcloud-dim-gt-3"
        if base == "LAMPPointCloud" and ent.get("m") != ent.get("n"):
            return None, "lamppointcloud-needs-square"
        return ent, None                                  # PointCloud, HALO tree mode, LAMPPointCloud

    # General regime
    if rung == GRID:
        return ent, None                                  # the runner reads or builds the lattice C
    if rung == GENERAL:
        # A native General instance is used as it stands: the √N and cap
        # gates of the Point Cloud lift below do not apply.
        m, n = ent.get("m"), ent.get("n")
        if m is None or n is None or m != n:
            return None, "dense-runner-needs-square-mn"
        cost_path = ent.get("cost_matrix_path")
        has_cost = (ent.get("cost_seed") is not None
                    or bool(cost_path and os.path.exists(cost_path)))
        if not has_cost:
            # Without a cost file or seed, dense_cost.resolve_dense_cost would
            # fall back to the lattice cost of (resolution, norm), which is not
            # this instance's.
            return None, "dense-rung-missing-cost"
        return ent, None

    # Point Cloud instance: C is materialised and the entry restamped General
    # with resolution = √N.
    m, n = ent.get("m"), ent.get("n")
    if m is None or n is None or m != n:
        return None, "dense-runner-needs-square-mn"
    side = int(round(m ** 0.5))
    if side * side != m or side not in _SUPPORTED_SIDES:
        return None, "dense-runner-needs-supported-sqrtN"
    if m > geo_dense_cap:
        return None, f"geo-dense-cap-{geo_dense_cap}-exceeded"
    cost_path = _materialize_geo_cost(ent, scratch, geo_cache)
    out = dict(ent)
    out.update({"rung": GENERAL, "resolution": side, "cost_matrix_path": cost_path})
    return out, None
