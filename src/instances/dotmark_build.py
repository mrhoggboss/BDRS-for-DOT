"""Build the DOTmark data under ``data/`` from the raw CSVs in ``DOTmark_1.0/Data/``:
unit-mass marginals (no floor), max-1 cost matrices and the instance index.

Usage:
    python -m src.instances.dotmark_build                       # all three norms
    python -m src.instances.dotmark_build --norms sql2 l1       # the index then holds only these
"""

import argparse
import json
import os
import time
from glob import glob

import numpy as np

from src.methods.grid_cmax import grid_cmax


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_CSV_ROOT = os.path.join(REPO_ROOT, "DOTmark_1.0", "Data")
OUTPUT_ROOT = os.path.join(REPO_ROOT, "data")
MARGINAL_DIR = os.path.join(OUTPUT_ROOT, "marginals")
COST_DIR = os.path.join(OUTPUT_ROOT, "cost_matrices")

DEFAULT_RESOLUTIONS = [32, 64, 128]

# ``inf`` has no cost file: regimes.synthesize_dense_norm_entries derives its
# DOTmark entries from this index.
ALL_NORMS = ("l1", "l2", "sql2")


def _parse_filename(path: str):
    """``data<res>_<id>.csv`` -> ``(res, id)``."""
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split("_")
    res = int(parts[0].replace("data", ""))
    id_val = int(parts[1])
    return res, id_val


def build_marginals(resolutions=None):
    """Save every raw image at ``resolutions`` as a unit-mass marginal; return the count."""
    resolutions = list(resolutions if resolutions is not None else DEFAULT_RESOLUTIONS)
    class_dirs = sorted(
        d for d in os.listdir(DATA_CSV_ROOT)
        if os.path.isdir(os.path.join(DATA_CSV_ROOT, d))
    )
    count = 0
    for class_name in class_dirs:
        class_path = os.path.join(DATA_CSV_ROOT, class_name)
        csv_files = glob(os.path.join(class_path, "*.csv"))

        for fpath in csv_files:
            res, id_val = _parse_filename(fpath)
            if res not in resolutions:
                continue

            matrix = np.loadtxt(fpath, delimiter=",", dtype=np.float64)
            flat = matrix.flatten()
            total = flat.sum()
            if total > 1e-12:
                flat = flat / total

            out_dir = os.path.join(MARGINAL_DIR, class_name, f"n{res}")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"data{res}_{id_val}.npy")
            np.save(out_path, flat)
            count += 1

    print(f"Saved {count} marginal .npy files.")
    return count


def build_cost_matrix(n: int, norm: str, block: int = 1024):
    """Write the max-1 cost matrix of ``norm`` on the n × n grid and return its path.

    An existing file is kept when its shape and dtype are right; its values are
    not read.
    """
    if norm not in ALL_NORMS:
        raise ValueError(f"Unknown norm: {norm!r}. Valid: {ALL_NORMS}")

    out_path = os.path.join(COST_DIR, f"cost_n{n}_{norm}.npy")
    nn = n * n
    if os.path.exists(out_path):
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape == (nn, nn) and arr.dtype == np.float64:
            print(f"Cost matrix exists and valid: {out_path}")
            return out_path
        print(f"Cost matrix invalid, rebuilding: {out_path}")

    print(f"Building normalized {norm} cost matrix n={n} ({nn}x{nn}) ...")
    os.makedirs(COST_DIR, exist_ok=True)

    row_idx = np.repeat(np.arange(n, dtype=np.int64), n)
    col_idx = np.tile(np.arange(n, dtype=np.int64), n)

    c_max = grid_cmax(norm, n)

    # open_memmap allocates the whole file before any row is written, so an
    # interrupted build under the final name would pass the test above. The
    # file is written under a temporary name and renamed once complete.
    tmp_path = out_path + ".partial"
    try:
        C = np.lib.format.open_memmap(
            tmp_path, mode="w+", dtype=np.float64, shape=(nn, nn)
        )

        for i0 in range(0, nn, block):
            i1 = min(nn, i0 + block)
            dr = row_idx[i0:i1, None].astype(np.float64) - row_idx[None, :].astype(np.float64)
            dc = col_idx[i0:i1, None].astype(np.float64) - col_idx[None, :].astype(np.float64)
            if norm == "l1":
                C[i0:i1, :] = (np.abs(dr) + np.abs(dc)) / c_max
            elif norm == "l2":
                C[i0:i1, :] = np.sqrt(dr * dr + dc * dc) / c_max
            else:  # sql2
                C[i0:i1, :] = (dr * dr + dc * dc) / c_max

        C.flush()
        del C
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print(f"Saved: {out_path}  (max = 1.0, raw max was {c_max:.4f})")
    return out_path


def _discover_built_resolutions():
    """The resolutions with at least one marginal file on disk, in any class."""
    if not os.path.isdir(MARGINAL_DIR):
        return []
    found = set()
    for class_name in os.listdir(MARGINAL_DIR):
        class_path = os.path.join(MARGINAL_DIR, class_name)
        if not os.path.isdir(class_path):
            continue
        for entry in os.listdir(class_path):
            if not entry.startswith("n"):
                continue
            try:
                n = int(entry[1:])
            except ValueError:
                continue
            res_dir = os.path.join(class_path, entry)
            if any(f.endswith(".npy") for f in os.listdir(res_dir)):
                found.add(n)
    return sorted(found)


def build_instance_index(norms, resolutions=None):
    """Write the index, one entry per (class, resolution, pair i < j, norm), and
    return its path.

    It covers every resolution with marginals on disk (``resolutions`` only when
    there are none), so runs with different ``--resolutions`` accumulate. The
    file is rewritten, however: it holds only ``norms``, and it loses the
    n_g = 1024 entries that ``dotmark_upsample_1024`` adds (no raw CSV exists at
    that size).
    """
    norms = tuple(norms)
    for nm in norms:
        if nm not in ALL_NORMS:
            raise ValueError(f"Unknown norm: {nm!r}. Valid: {ALL_NORMS}")

    class_names = sorted(
        d for d in os.listdir(DATA_CSV_ROOT)
        if os.path.isdir(os.path.join(DATA_CSV_ROOT, d))
    )

    discovered = _discover_built_resolutions()
    if not discovered:
        discovered = list(resolutions if resolutions is not None else DEFAULT_RESOLUTIONS)
    print(f"Indexing resolutions present on disk: {discovered}")

    index = []
    for class_name in class_names:
        class_path = os.path.join(DATA_CSV_ROOT, class_name)

        for n in discovered:
            files = sorted(glob(os.path.join(class_path, f"data{n}_*.csv")))
            if not files:
                continue

            ids = [_parse_filename(f)[1] for f in files]

            for norm in norms:
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        instance_id = f"{class_name}_n{n}_i{i}_j{j}_{norm}"
                        index.append({
                            "instance_id": instance_id,
                            "class_name": class_name,
                            "resolution": n,
                            "i": i,
                            "j": j,
                            "id_i": ids[i],
                            "id_j": ids[j],
                            "norm": norm,
                            "marginal_i_path": os.path.join(
                                MARGINAL_DIR, class_name, f"n{n}", f"data{n}_{ids[i]}.npy"
                            ),
                            "marginal_j_path": os.path.join(
                                MARGINAL_DIR, class_name, f"n{n}", f"data{n}_{ids[j]}.npy"
                            ),
                            "cost_matrix_path": os.path.join(
                                COST_DIR, f"cost_n{n}_{norm}.npy"
                            ),
                        })

    out_path = os.path.join(OUTPUT_ROOT, "dotmark_instances_index.json")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Index: {len(index)} instances across norms={list(norms)} -> {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Prepare DOTmark data.")
    parser.add_argument(
        "--norms", nargs="+", default=list(ALL_NORMS), choices=list(ALL_NORMS),
        help=f"Cost-matrix norms to build (default: all {list(ALL_NORMS)}).",
    )
    parser.add_argument(
        "--resolutions", nargs="+", type=int, default=list(DEFAULT_RESOLUTIONS),
        help=f"Grid sizes to prep (default: {DEFAULT_RESOLUTIONS}). "
             f"At n=256 the cost matrix is ~34 GB per norm; use --skip-costs.",
    )
    parser.add_argument(
        "--skip-marginals", action="store_true",
        help="Skip the marginal-building step.",
    )
    parser.add_argument(
        "--skip-costs", action="store_true",
        help="Skip cost-matrix building. Useful at large n where C won't fit "
             "on disk or in memory; the factored solvers don't need C anyway.",
    )
    args = parser.parse_args()
    # The index is built from the CSV file names, so every run needs them.
    if not os.path.isdir(DATA_CSV_ROOT):
        raise SystemExit(
            f"{DATA_CSV_ROOT} not found: download and unpack the DOTmark 1.0 "
            f"archive as in the README's Data section.")

    t0 = time.time()
    if not args.skip_marginals:
        print(f"=== Building marginals for resolutions {args.resolutions} ===")
        build_marginals(resolutions=args.resolutions)
    else:
        print("=== Skipping marginals (--skip-marginals) ===")

    if not args.skip_costs:
        print(f"\n=== Building cost matrices for norms {args.norms} ===")
        for norm in args.norms:
            for n in args.resolutions:
                build_cost_matrix(n, norm)
    else:
        print("\n=== Skipping cost matrices (--skip-costs) ===")

    print("\n=== Building instance index ===")
    build_instance_index(args.norms, resolutions=args.resolutions)

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
