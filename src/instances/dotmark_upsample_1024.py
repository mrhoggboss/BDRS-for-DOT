"""The derived n_g = 1024 DOTmark tier, upsampled from the raw integer 512 × 512 CSVs.

HALO's recipe: bilinear 2× upsampling of the raw intensities, each original
pixel kept at its position, rounding to integers and clipping to the original
intensity range, all before the usual normalization to unit mass. The recipe
does not fix the alignment; this module anchors the corners:

    up[2i, 2j]     = orig[i, j]
    up[2i, 2j+1]   = (orig[i, j] + orig[i, j+1]) / 2
    up[2i+1, 2j]   = (orig[i, j] + orig[i+1, j]) / 2
    up[2i+1, 2j+1] = mean of the four surrounding originals

with the last row and column (index 2m−1) copied from row and column 2m−2,
then ``np.rint`` and ``np.clip`` to [orig.min(), orig.max()]. On an integer
image the clip is a no-op, kept because the recipe prescribes it. The files
follow the recipe but are not bit-identical to HALO's, which are not
distributed.

Run ``dotmark_build`` first: it rewrites the index and drops the n1024
entries. Re-running this script re-appends them (existing marginals are kept
unless ``--force``), and ``--check`` detects the loss.

Usage:
    python -m src.instances.dotmark_upsample_1024            # build
    python -m src.instances.dotmark_upsample_1024 --check    # verify only
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from src.protocol.paths import REPO_ROOT
from src.protocol.instance_record import utc_now_iso

DATA_CSV_ROOT = REPO_ROOT / "DOTmark_1.0" / "Data"
MARGINAL_DIR = REPO_ROOT / "data" / "marginals"
INDEX_PATH = REPO_ROOT / "data" / "dotmark_instances_index.json"
PROVENANCE_PATH = MARGINAL_DIR / "n1024_provenance.json"

SRC_RES = 512
OUT_RES = 1024
IMAGE_IDS = list(range(1001, 1011))         # DOTmark ids, 10 per class

RECIPE = (
    "HALO Appendix-E: corner-anchored 2x bilinear upsampling "
    "of the RAW integer 512 CSVs; np.rint; np.clip to [orig.min(), orig.max()]; "
    "then the standard dotmark_build probability normalization (flatten, /sum)."
)
ALIGNMENT = (
    "up[2i,2j]=orig[i,j] (anchors exact); odd positions = 2-pt/4-pt means of "
    "the surrounding originals; last row/col (index 2m-1) copied from row/col 2m-2."
)


def upsample2x_corner_anchored(orig: np.ndarray) -> np.ndarray:
    """The corner-anchored 2× upsample of a square image (module docstring)."""
    if orig.ndim != 2 or orig.shape[0] != orig.shape[1]:
        raise ValueError(f"expected a square 2D array; got {orig.shape}")
    orig = orig.astype(np.float64)
    m = orig.shape[0]
    up = np.empty((2 * m, 2 * m), dtype=np.float64)

    up[0::2, 0::2] = orig                                            # anchors
    up[0::2, 1:-1:2] = 0.5 * (orig[:, :-1] + orig[:, 1:])            # even r, odd c
    up[0::2, -1] = orig[:, -1]                                       # last col clamp
    up[1:-1:2, 0::2] = 0.5 * (orig[:-1, :] + orig[1:, :])            # odd r, even c
    up[-1, 0::2] = orig[-1, :]                                       # last row clamp
    up[1:-1:2, 1:-1:2] = 0.25 * (orig[:-1, :-1] + orig[:-1, 1:]
                                 + orig[1:, :-1] + orig[1:, 1:])     # odd r, odd c
    up[1:-1:2, -1] = 0.5 * (orig[:-1, -1] + orig[1:, -1])            # last col, odd r
    up[-1, 1:-1:2] = 0.5 * (orig[-1, :-1] + orig[-1, 1:])            # last row, odd c
    up[-1, -1] = orig[-1, -1]

    up = np.rint(up)
    return np.clip(up, orig.min(), orig.max())


def normalize_marginal(img: np.ndarray) -> np.ndarray:
    """The normalization of ``dotmark_build.build_marginals``."""
    flat = img.flatten().astype(np.float64)
    total = flat.sum()
    if total > 1e-12:
        flat = flat / total
    return flat


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                       # noqa: BLE001
        return "unknown"


def _classes(*, for_check: bool = False) -> List[str]:
    """The classes of the raw CSV tree, or for ``--check`` without that tree the
    classes with n1024 marginals on disk. Never the index's classes: after
    ``dotmark_build`` the index has no n1024 entry, and an empty class list
    would pass the check vacuously."""
    if DATA_CSV_ROOT.is_dir():
        return sorted(
            d for d in os.listdir(DATA_CSV_ROOT)
            if (DATA_CSV_ROOT / d).is_dir()
        )
    if for_check and MARGINAL_DIR.is_dir():
        found = sorted(
            d.name for d in MARGINAL_DIR.iterdir()
            if (d / f"n{OUT_RES}").is_dir()
        )
        if found:
            return found
    raise FileNotFoundError(
        f"{DATA_CSV_ROOT} missing"
        + (" and no n1024 marginals found under data/marginals"
           if for_check else "; download and unpack the DOTmark 1.0 archive "
                             "as in the README's Data section"))


def _out_path(cls: str, image_id: int) -> Path:
    return MARGINAL_DIR / cls / f"n{OUT_RES}" / f"data{OUT_RES}_{image_id}.npy"


def _repo_rel(p: Path) -> str:
    """``p`` relative to the repository root, or absolute when outside it."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def build(classes: List[str], norms: List[str], force: bool) -> int:
    sources: Dict[str, str] = {}
    built = kept = 0
    for cls in classes:
        for image_id in IMAGE_IDS:
            src = DATA_CSV_ROOT / cls / f"data{SRC_RES}_{image_id}.csv"
            if not src.exists():
                print(f"FATAL: missing raw CSV {src}", file=sys.stderr)
                return 1
            out = _out_path(cls, image_id)
            sources[_repo_rel(out)] = _repo_rel(src)
            if out.exists() and not force:
                kept += 1
                continue
            orig = np.loadtxt(src, delimiter=",", dtype=np.float64)
            if orig.shape != (SRC_RES, SRC_RES):
                print(f"FATAL: {src} has shape {orig.shape}", file=sys.stderr)
                return 1
            up = upsample2x_corner_anchored(orig)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, normalize_marginal(up))
            built += 1
    print(f"upsample_1024: {built} marginals built, {kept} kept "
          f"({len(classes)} classes)")

    # Merge into the existing record, so a subset run keeps the other sources.
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_sources: Dict[str, str] = {}
    if PROVENANCE_PATH.exists():
        try:
            merged_sources = dict(json.loads(PROVENANCE_PATH.read_text()).get("sources", {}))
        except (json.JSONDecodeError, AttributeError):
            merged_sources = {}
    merged_sources.update(sources)
    PROVENANCE_PATH.write_text(json.dumps({
        "recipe": RECIPE,
        "alignment": ALIGNMENT,
        "rounding": "np.rint",
        "source_resolution": SRC_RES,
        "output_resolution": OUT_RES,
        "script": "src/instances/dotmark_upsample_1024.py",
        "git_sha": _git_sha(),
        "generated_utc": utc_now_iso(),
        "n_files": len(merged_sources),
        "sources": merged_sources,
    }, indent=1))
    print(f"upsample_1024: provenance -> {PROVENANCE_PATH} "
          f"({len(merged_sources)} sources)")

    # Replace only the (class, norm) entries being built, so a subset run keeps
    # the other classes' and norms' entries.
    cls_set, norm_set = set(classes), set(norms)
    index = json.loads(INDEX_PATH.read_text())
    index = [e for e in index
             if not (e.get("resolution") == OUT_RES
                     and e.get("class_name") in cls_set
                     and e.get("norm") in norm_set)]
    added = 0
    for cls in classes:
        for norm in norms:
            for i, j in itertools.combinations(range(len(IMAGE_IDS)), 2):
                index.append({
                    "instance_id": f"{cls}_n{OUT_RES}_i{i}_j{j}_{norm}",
                    "class_name": cls,
                    "resolution": OUT_RES,
                    "i": i,
                    "j": j,
                    "id_i": IMAGE_IDS[i],
                    "id_j": IMAGE_IDS[j],
                    "norm": norm,
                    "marginal_i_path": str(_out_path(cls, IMAGE_IDS[i])),
                    "marginal_j_path": str(_out_path(cls, IMAGE_IDS[j])),
                    "cost_matrix_path": None,     # an fp64 C would be 8 TiB
                    "derived": True,
                })
                added += 1
    INDEX_PATH.write_text(json.dumps(index, indent=2))
    print(f"upsample_1024: appended {added} derived n{OUT_RES} index entries "
          f"(norms={norms})")
    return 0


def check(classes: List[str], norms: List[str]) -> int:
    ok = True
    n_marg = 0
    for cls in classes:
        for image_id in IMAGE_IDS:
            out = _out_path(cls, image_id)
            if not out.exists():
                print(f"CHECK FAIL: missing {out}", file=sys.stderr)
                ok = False
                continue
            v = np.load(out)
            if v.shape != (OUT_RES * OUT_RES,):
                print(f"CHECK FAIL: {out} shape {v.shape}", file=sys.stderr)
                ok = False
            elif (v < 0).any() or abs(float(v.sum()) - 1.0) > 1e-12:
                print(f"CHECK FAIL: {out} not a probability vector", file=sys.stderr)
                ok = False
            else:
                n_marg += 1
    if not PROVENANCE_PATH.exists():
        print(f"CHECK FAIL: missing {PROVENANCE_PATH}", file=sys.stderr)
        ok = False
    index = json.loads(INDEX_PATH.read_text()) if INDEX_PATH.exists() else []
    ents = [e for e in index if e.get("resolution") == OUT_RES]
    # Count only the in-scope (class, norm) entries, so a subset --check judges
    # its own subset; the derived flag is required of every n1024 entry.
    cls_set, norm_set = set(classes), set(norms)
    in_scope = [e for e in ents
                if e.get("class_name") in cls_set and e.get("norm") in norm_set]
    expected = len(classes) * len(norms) * (len(IMAGE_IDS) * (len(IMAGE_IDS) - 1) // 2)
    if len(in_scope) != expected or not all(e.get("derived") for e in ents):
        print(f"CHECK FAIL: index has {len(in_scope)} in-scope n{OUT_RES} "
              f"entries (expected {expected}, all derived=true); if "
              f"dotmark_build ran after the upsampler, re-run this script to "
              f"re-append", file=sys.stderr)
        ok = False
    # A class with n1024 marginals on disk but no index entry most likely lost
    # its entries; warn without failing.
    indexed_classes = {e.get("class_name") for e in ents}
    for d in sorted(MARGINAL_DIR.iterdir()) if MARGINAL_DIR.is_dir() else []:
        if (d / f"n{OUT_RES}").is_dir() and d.name not in indexed_classes:
            print(f"CHECK WARN: {d.name} has n{OUT_RES} marginals on disk "
                  f"but no index entries (lost? re-run with "
                  f"--classes {d.name})", file=sys.stderr)
    print(f"upsample_1024 --check: {n_marg} marginals OK, "
          f"{len(in_scope)}/{len(ents)} in-scope/total index entries, "
          f"provenance {'present' if PROVENANCE_PATH.exists() else 'MISSING'} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Derived n1024 DOTmark tier.")
    ap.add_argument("--classes", nargs="+", default=None)
    # sql2, the only cost HALO runs, by default. The marginals do not depend on
    # the norm, so a later run with --norms adds entries without rebuilding them.
    ap.add_argument("--norms", nargs="+", default=["sql2"],
                    choices=["l1", "sql2", "l2", "inf"])
    ap.add_argument("--check", action="store_true",
                    help="verify marginals/provenance/index; build nothing")
    ap.add_argument("--force", action="store_true",
                    help="rebuild marginals even when present")
    args = ap.parse_args(argv)

    try:
        classes = args.classes or _classes(for_check=args.check)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    if args.check:
        return check(classes, args.norms)
    return build(classes, args.norms, args.force)


if __name__ == "__main__":
    sys.exit(main())
