"""Registry of the non-DOTmark dataset loaders, and a CLI that builds them.

Every loader is called as ``build(out_dir, scales, n_pairs, seed)`` but reads
the arguments its own way: ``modelnet10`` takes ``scales`` as the cloud sizes N
and ``n_pairs`` as the number of meshes per class; ``random_dense`` takes
``n_pairs`` as the pairs per (N, cost family) and does not read ``scales``.

    python -m src.instances.registry --datasets random_dense --n-pairs 10 --seed 20260907
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence

from src.protocol.regimes import PROCEDURAL_NORMS
from src.instances import instance_io as io
from src.instances import synthetic, modelnet


@dataclass
class LoaderSpec:
    name: str
    rung: str                       # Grid | PointCloud | General
    default_cost_types: Sequence[str]
    build: Callable[[Path, Sequence[int], int, int], Dict]
    notes: str = ""


def _registry() -> Dict[str, LoaderSpec]:
    reg: Dict[str, LoaderSpec] = {}
    reg["modelnet10"] = LoaderSpec("modelnet10", "PointCloud", modelnet.COST_TYPES,
                                   modelnet.build_modelnet10,
                                   "ModelNet10 CAD meshes → unit-cube surface point clouds (sql2, l2)")
    reg["random_dense"] = LoaderSpec("random_dense", "General", PROCEDURAL_NORMS,
                                     synthetic.build_random_dense, "random nonneg cost + Dirichlet marginals")
    return reg


REGISTRY: Dict[str, LoaderSpec] = _registry()


def get(name: str) -> LoaderSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def build_all(dataset_names: Sequence[str], *, scales: Sequence[int],
              n_pairs: int, seed: int, out_root: Path = io.DATA_ROOT) -> Dict[str, Dict]:
    reports: Dict[str, Dict] = {}
    for nm in dataset_names:
        spec = get(nm)
        try:
            reports[nm] = spec.build(out_root, scales, n_pairs, seed)
        except Exception as e:                               # noqa: BLE001
            import traceback
            reports[nm] = {"name": nm, "rung": spec.rung, "n_instances": 0,
                           "data_present": False,
                           "reason": f"build error: {e!r}", "instance_ids": [],
                           "traceback": traceback.format_exc()[:2000]}
    return reports


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Build non-DOTmark dataset instances.")
    p.add_argument("--datasets", nargs="+", default=["all"],
                   help=f"dataset names ({', '.join(sorted(REGISTRY))}) or 'all'")
    p.add_argument("--scales", nargs="+", type=int, default=[32, 256],
                   help="modelnet10: the cloud sizes N; random_dense does not read it")
    p.add_argument("--n-pairs", type=int, default=2,
                   help="modelnet10: meshes per class (at least 2); random_dense: "
                        "pairs per (N, cost family)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    ds = sorted(REGISTRY) if args.datasets == ["all"] else args.datasets
    reports = build_all(ds, scales=args.scales, n_pairs=args.n_pairs, seed=args.seed)
    print(f"{'dataset':<16} {'rung':<10} {'#inst':>6}  reason")
    for nm in ds:
        r = reports[nm]
        print(f"{nm:<16} {r['rung']:<10} {r['n_instances']:>6}  "
              f"{'' if r['data_present'] else (r.get('reason') or '')}")
        if r.get("traceback"):
            print(r["traceback"])
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
