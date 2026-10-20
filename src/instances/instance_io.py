"""On-disk writers for the registry datasets: Point Cloud instances
(``modelnet10``) and procedural General instances (``random_dense``). Each
instance writer returns its index entries."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.protocol.paths import REPO_ROOT
from src.protocol.regimes import procedural_norm

DATA_ROOT = REPO_ROOT / "data"


def normalize_marginal(a: np.ndarray) -> np.ndarray:
    """Flatten and divide by the sum, with no floor: an instance stores the
    original marginals, and our solvers apply δ_m to their own copy."""
    a = np.asarray(a, dtype=np.float64).flatten()
    a = a / a.sum()
    return a


def exact_c_max_by_type(P: np.ndarray, Q: np.ndarray,
                        cost_types: Sequence[str]) -> Dict[str, float]:
    """The attained maximum C_max = max_ij κ(p_i, q_j) of each cost type.

    It must be attained, not a bound: the Point Cloud solver recomputes it on
    every solve, and the runner records ``c_max_not_exact`` unless the stored
    value gives max C = 1 to within 1e-9."""
    import torch
    from src.methods import geo_cost
    from src.protocol.instance_record import geo_cost_type
    Pt = torch.as_tensor(np.asarray(P, dtype=np.float64))
    Qt = torch.as_tensor(np.asarray(Q, dtype=np.float64))
    return {ct: float(geo_cost.exact_c_max(Pt, Qt, geo_cost_type(ct)))
            for ct in cost_types}


def _save(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr, dtype=np.float64))
    return str(path)


def write_geo_instance(
    dataset: str, inst_name: str, P: np.ndarray, Q: np.ndarray,
    r: np.ndarray, c: np.ndarray, cost_types: Sequence[str],
    *, c_max: float = 1.0, c_max_by_type: Optional[Dict[str, float]] = None,
    out_root: Optional[Path] = None,
) -> List[Dict]:
    """Write the clouds P (m × d), Q (n × d) and the marginals, and return one
    index entry per cost type.

    Pass ``c_max_by_type = exact_c_max_by_type(P, Q, cost_types)``; a cost type
    missing from it gets the scalar ``c_max``."""
    out_root = out_root or DATA_ROOT
    base = out_root / dataset / "clouds" / inst_name
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if Q.ndim == 1:
        Q = Q.reshape(-1, 1)
    m, d = P.shape
    n = Q.shape[0]
    pP = _save(base / "support_i.npy", P)
    pQ = _save(base / "support_j.npy", Q)
    pr = _save(base / "r.npy", normalize_marginal(r))
    pc = _save(base / "c.npy", normalize_marginal(c))
    out = []
    for ct in cost_types:
        cm = (c_max_by_type or {}).get(ct, c_max)
        out.append({
            "instance_id": f"{dataset}__{inst_name}_{ct}",
            "dataset": dataset, "rung": "PointCloud", "class_name": dataset,
            "resolution": None, "norm": ct,
            "marginal_i_path": pr, "marginal_j_path": pc,
            "cost_matrix_path": None,
            "support_i_path": pP, "support_j_path": pQ,
            "c_max": float(cm), "m": int(m), "n": int(n), "dim": int(d),
        })
    return out


def write_procedural_dense_instance(
    dataset: str, inst_name: str, m: int, n: int, seed: int, family: str,
    r: np.ndarray, c: np.ndarray, *, out_root: Optional[Path] = None,
) -> List[Dict]:
    """Write the two marginals of a General instance whose cost is regenerated
    from ``(m, n, seed, family)``, and return its index entry."""
    out_root = out_root or DATA_ROOT
    base = out_root / dataset / "dense" / inst_name
    pr = _save(base / "r.npy", normalize_marginal(r))
    pcm = _save(base / "c.npy", normalize_marginal(c))
    return [{
        "instance_id": f"{dataset}__{inst_name}_dense",
        "dataset": dataset, "rung": "General", "class_name": dataset,
        # The family is part of the norm token, not only of cost_family:
        # aggregation keys on norm, and the two families are different costs
        # (mean(C), hence η, differs by a factor of about 3 to 4).
        "resolution": None, "norm": procedural_norm(family),
        "marginal_i_path": pr, "marginal_j_path": pcm,
        "cost_matrix_path": None,
        "cost_seed": int(seed), "cost_family": str(family),
        "m": int(m), "n": int(n),
    }]


def write_index(dataset: str, entries: List[Dict], *, out_root: Optional[Path] = None) -> Path:
    """Write ``<out_root>/<dataset>/index.json`` (an empty list when the data is absent)."""
    import json
    out_root = out_root or DATA_ROOT
    d = out_root / dataset
    d.mkdir(parents=True, exist_ok=True)
    p = d / "index.json"
    p.write_text(json.dumps(entries, indent=0))
    return p
