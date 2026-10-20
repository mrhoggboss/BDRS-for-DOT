"""ModelNet10 loader: Point Cloud instances (d = 3) from Princeton ModelNet10 meshes.

Each mesh used (the first k test-set meshes of a class, by filename) is sampled
once into BASE_N surface points in [0,1]³, in a random order fixed by the seed.
The size-N cloud is the first N of them, so a mesh's clouds are nested across sizes.

The meshes are read from ``RAW_ROOT/<class>/test/<class>_XXXX.off``, where
``RAW_ROOT`` is the archive's own top-level folder ``ModelNet10/`` unpacked at
the repository root. Everything the build writes goes under
``<out_dir>/modelnet10/``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.instances import instance_io as io
from src.protocol.paths import REPO_ROOT

# Surface samples per mesh, which is also the largest cloud size N.
BASE_N = 524288
# sql2 is the one cost HALO runs on point clouds; the unsquared l2 is run, among
# the Point Cloud solvers, by PointCloud alone.
COST_TYPES = ("sql2", "l2")
DIM = 3

CLASSES = ("bathtub", "bed", "chair", "desk", "dresser",
           "monitor", "night_stand", "sofa", "table", "toilet")

MODELNET10_URL = "https://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
MODELNET10_SHA256 = "9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51"
RAW_ROOT = REPO_ROOT / "ModelNet10"


def download_hint(raw_root: Optional[Path] = None) -> str:
    """Where the meshes were looked for, and the README's commands that put them
    at ``RAW_ROOT``."""
    raw = RAW_ROOT if raw_root is None else Path(raw_root)
    return (f"no meshes at {raw}/<class>/test/<class>_XXXX.off. From the repository "
            f"root, download the archive (~473 MB) and unpack its test meshes, as in "
            f"the README's Data section:\n"
            f"  curl -LO {MODELNET10_URL}\n"
            f"  echo \"{MODELNET10_SHA256}  ModelNet10.zip\" | sha256sum -c\n"
            f"  unzip -q ModelNet10.zip 'ModelNet10/*/test/*'\n"
            f"On macOS, use shasum -a 256 -c in place of sha256sum -c.")


def _test_shapes(raw_root: Path, class_name: str, k: int) -> List[Path]:
    d = raw_root / class_name / "test"
    offs = sorted(d.glob(f"{class_name}_*.off"))
    return offs[:k]


def _shape_id(off_path: Path) -> str:
    """'<class>_0001.off' -> '0001'."""
    return off_path.stem.split("_")[-1]


def _sample_base_cloud(off_path: Path, seed: int) -> np.ndarray:
    """(BASE_N, 3) area-weighted surface samples in [0,1]³, rows in a random
    order fixed by `seed`."""
    import trimesh
    mesh = trimesh.load(str(off_path), process=False, force="mesh")
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, BASE_N, seed=seed)
    except TypeError:                                  # older trimesh: no seed kw
        _st = np.random.get_state()
        np.random.seed(seed)
        try:
            pts, _ = trimesh.sample.sample_surface(mesh, BASE_N)
        finally:
            np.random.set_state(_st)                    # do not leak into the global RNG
    pts = np.asarray(pts, dtype=np.float64)
    # One scale for all axes. Both costs are translation invariant and κ/C_max
    # is scale invariant, so up to rounding this fixes only the frame.
    pts = pts - pts.mean(axis=0, keepdims=True)
    span = float((pts.max(axis=0) - pts.min(axis=0)).max())
    if span > 0:
        pts = pts / span
    pts = pts - pts.min(axis=0, keepdims=True)
    perm = np.random.default_rng(seed).permutation(BASE_N)
    return np.ascontiguousarray(pts[perm])


def _base_cloud(out_root: Path, class_name: str, off_path: Path, seed: int) -> np.ndarray:
    """Base cloud of a mesh, read from the .npy cache or sampled and cached."""
    cache = out_root / "modelnet10" / "base" / f"{class_name}__{_shape_id(off_path)}__s{seed}.npy"
    if cache.is_file():
        arr = np.load(cache)
        if arr.shape == (BASE_N, DIM):
            return arr
    arr = _sample_base_cloud(off_path, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def _c_max_by_type(P: np.ndarray, Q: np.ndarray) -> dict:
    """The attained C_max = max_ij κ(p_i, q_j) of this pair, per cost type.

    Not a bounding-box upper bound: the Point Cloud runner records an instance
    whose normalized cost does not attain 1 as an error (``c_max_not_exact``)."""
    return io.exact_c_max_by_type(P, Q, COST_TYPES)


def _within_class_pairs(k: int) -> List[Tuple[int, int]]:
    return [(a, b) for a in range(k) for b in range(a + 1, k)]


def _empty_index_unless_built(out_dir: Path) -> None:
    """Write an empty index, keeping one an earlier build wrote."""
    if not (Path(out_dir) / "modelnet10" / "index.json").is_file():
        io.write_index("modelnet10", [], out_root=out_dir)


def build_modelnet10(out_dir: Path, scales: Sequence[int], n_pairs: int, seed: int,
                     *, raw_root: Optional[Path] = None) -> Dict:
    """Registry loader. ``n_pairs`` is the number k of meshes per class (at least
    2), giving C(k, 2) within-class pairs per class; ``scales`` are cloud sizes N.
    The meshes are read from ``raw_root`` (default ``RAW_ROOT``)."""
    raw_root = RAW_ROOT if raw_root is None else Path(raw_root)
    shapes_per_class = max(2, n_pairs)
    ladder = sorted({int(s) for s in scales if int(s) <= BASE_N})
    dropped = sorted({int(s) for s in scales if int(s) > BASE_N})
    if dropped:
        print(f"[modelnet10] dropping scales > BASE_N={BASE_N} (unsupported): {dropped}")
    if not ladder:
        _empty_index_unless_built(out_dir)
        return {"name": "modelnet10", "rung": "PointCloud", "n_instances": 0,
                "data_present": False,
                "reason": f"no requested scale ≤ BASE_N={BASE_N}", "instance_ids": []}

    if not raw_root.is_dir() or not _test_shapes(raw_root, CLASSES[0], 1):
        _empty_index_unless_built(out_dir)
        return {"name": "modelnet10", "rung": "PointCloud", "n_instances": 0,
                "data_present": False,
                "reason": f"data_absent: {download_hint(raw_root)}", "instance_ids": []}

    all_entries: List[Dict] = []
    for ci, class_name in enumerate(CLASSES):
        offs = _test_shapes(raw_root, class_name, shapes_per_class)
        if len(offs) < 2:
            continue
        clouds = [_base_cloud(out_dir, class_name, off,
                              seed + ci * 1000 + i) for i, off in enumerate(offs)]
        ids = [_shape_id(off) for off in offs]
        for (a, b) in _within_class_pairs(len(offs)):
            for N in ladder:
                P = np.ascontiguousarray(clouds[a][:N])
                Q = np.ascontiguousarray(clouds[b][:N])
                r = np.ones(N); c = np.ones(N)
                inst = f"{class_name}_{ids[a]}_{ids[b]}_n{N}"
                ents = io.write_geo_instance(
                    "modelnet10", inst, P, Q, r, c, COST_TYPES,
                    c_max_by_type=_c_max_by_type(P, Q), out_root=out_dir)
                all_entries.extend(ents)
    io.write_index("modelnet10", all_entries, out_root=out_dir)
    return {"name": "modelnet10", "rung": "PointCloud", "n_instances": len(all_entries),
            "data_present": True, "reason": None,
            "instance_ids": [e["instance_id"] for e in all_entries]}
