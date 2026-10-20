"""What the orchestrator runs: the merged instance index, its filters, the
ordered (algorithm, instance) schedule, and what an existing runs directory
already settled."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.protocol.instance_record import SETTLED_STATUSES, InstanceSpec
from src.protocol.results_io import iter_rows
from src.protocol.regimes import (
    GRID,
    POINT_CLOUD,
    algo_applies_instance,
    canonical_regime,
)


# The default and the choices of --algos, in their run order within each
# (dataset, rung, resolution, norm) block. The orchestrator expands Sinkhorn
# and SinkhornGrid into one arm per ε; the arms are not in this tuple and run
# last in a block.
ALL_ALGOS = ("Grid", "PointCloud", "General", "DROT", "PDOT", "HOT", "Sinkhorn",
             "SinkhornGrid", "HALO", "M3S", "CUPDLPX")

# --algos choices outside the default run. LAMP's solve server outlives each
# runner and holds a CUDA context while idle, so its two names run only in an
# orchestrator job without other algorithms.
LAMP_ALGOS = ("LAMP", "LAMPPointCloud")
OPT_IN_ALGOS = LAMP_ALGOS


def _normalised_index_entry(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Default ``dataset`` to "dotmark" and ``rung`` to Grid, canonicalise the
    rung name, and drop a ``cost_matrix_path`` whose file is missing."""
    raw = dict(raw)
    raw.setdefault("dataset", "dotmark")
    raw["rung"] = canonical_regime(raw.get("rung") or GRID)
    cmp = raw.get("cost_matrix_path")
    if cmp and not os.path.exists(cmp):
        raw["cost_matrix_path"] = None
    return raw


def _inputs_exist(ent: Dict[str, Any]) -> bool:
    """Whether the marginals, and for a PointCloud entry the clouds, exist; C is
    not checked."""
    if not (os.path.exists(ent.get("marginal_i_path", "")) and
            os.path.exists(ent.get("marginal_j_path", ""))):
        return False
    if ent.get("rung") == POINT_CLOUD:
        if not (os.path.exists(ent.get("support_i_path", "")) and
                os.path.exists(ent.get("support_j_path", ""))):
            return False
    return True


def load_index(path: Path) -> List[Dict[str, Any]]:
    """One index file (a JSON list, a JSON object or JSONL), normalised, without
    the entries whose input files are missing."""
    raw_text = path.read_text()
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            data = [data]
    except json.JSONDecodeError:
        data = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
    out = []
    for ent in data:
        ent = _normalised_index_entry(ent)
        if _inputs_exist(ent):
            out.append(ent)
    return out


def load_all_indexes(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    """Merge index files by ``instance_id``, the last file winning."""
    merged: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"orchestrator: warning: index not found, skipping: {p}", file=sys.stderr)
            continue
        for ent in load_index(p):
            merged[ent["instance_id"]] = ent
    return list(merged.values())


def filter_index(
    index: List[Dict[str, Any]], *,
    classes: Optional[List[str]] = None,
    resolutions: Optional[List[int]] = None,
    norms: Optional[List[str]] = None,
    pairs: Optional[List[Tuple[int, int]]] = None,
    rungs: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    sizes: Optional[List[int]] = None,
    max_per_cell: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Select index entries.

    ``resolutions`` and ``pairs`` pass an entry that has no resolution or pair.
    ``sizes`` selects on the marginal length N, and ``max_per_cell`` keeps the
    first k entries per (dataset, N, norm) in ``instance_id`` order, so the cap
    is deterministic.
    """
    out = []
    wanted_rungs = {canonical_regime(r) for r in rungs} if rungs else None
    for ent in index:
        if datasets and ent.get("dataset") not in datasets:
            continue
        if wanted_rungs and canonical_regime(ent.get("rung") or GRID) not in wanted_rungs:
            continue
        if classes and ent.get("class_name") not in classes:
            continue
        if resolutions and ent.get("resolution") is not None and ent["resolution"] not in resolutions:
            continue
        if norms and ent["norm"] not in norms:
            continue
        if pairs and ent.get("i") is not None and (ent["i"], ent["j"]) not in pairs:
            continue
        if sizes and _problem_n(ent_spec_like(ent)) not in sizes:
            continue
        out.append(ent)
    if max_per_cell:
        capped: List[Dict[str, Any]] = []
        seen: Dict[Tuple[Any, ...], int] = defaultdict(int)
        for ent in sorted(out, key=lambda e: str(e.get("instance_id"))):
            key = (ent.get("dataset", "dotmark"),
                   _problem_n(ent_spec_like(ent)), ent["norm"])
            if seen[key] >= max_per_cell:
                continue
            seen[key] += 1
            capped.append(ent)
        out = capped
    return out


class _EntShim:
    """An index entry with the ``dense_shape()`` that ``_problem_n`` reads."""
    __slots__ = ("resolution", "m", "n", "instance_id")

    def __init__(self, ent: Dict[str, Any]):
        self.resolution = ent.get("resolution")
        self.m = ent.get("m")
        self.n = ent.get("n")
        self.instance_id = ent.get("instance_id", "?")

    def dense_shape(self):
        if self.resolution is not None:
            side = int(self.resolution) ** 2
            return side, side
        if not self.m:
            raise ValueError(f"{self.instance_id}: no size")
        return int(self.m), int(self.n or self.m)


def ent_spec_like(ent: Dict[str, Any]) -> "_EntShim":
    return _EntShim(ent)


def _algo_priority_index(algo: str) -> int:
    return ALL_ALGOS.index(algo) if algo in ALL_ALGOS else len(ALL_ALGOS)


def build_schedule(
    index: List[Dict[str, Any]],
    *,
    algos: List[str],
    spine_first: bool,
    sample_first: bool,
    head_only: bool = False,
    policy_exceptions: "frozenset | None" = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """The (algorithm, instance) pairs ``algo_applies_instance`` admits, in run
    order.

    ``spine_first`` runs one pair per (algorithm, dataset, size, norm) first,
    ``sample_first`` one per (algorithm, dataset, size, norm, class), where the
    size is the resolution, else m. ``head_only`` returns only those first
    pairs, and has no effect without one of the two.
    """
    # The sort key below ends in instance_id, so the run order does not depend
    # on the order of ``index``.
    full = [(a, ent) for ent in index for a in algos
            if algo_applies_instance(a, ent, policy_exceptions)]
    full.sort(key=lambda x: (
        x[1].get("dataset", "dotmark"), x[1].get("rung") or GRID,
        x[1].get("resolution") or 0, x[1]["norm"], _algo_priority_index(x[0]),
        x[1].get("class_name") or "", x[1].get("i") or 0, x[1].get("j") or 0,
        x[1]["instance_id"],
    ))

    if not (spine_first or sample_first):
        return full

    seen_keys = set()
    head: List[Tuple[str, Dict[str, Any]]] = []
    for algo, ent in full:
        size = ent.get("resolution")
        if size is None:
            size = ent.get("m")
        ds = ent.get("dataset", "dotmark")
        if sample_first:
            key = (algo, ds, size, ent["norm"], ent.get("class_name"))
        else:
            key = (algo, ds, size, ent["norm"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        head.append((algo, ent))

    if head_only:
        return head

    head_pairs = {(a, e["instance_id"]) for a, e in head}
    tail = [(a, e) for a, e in full if (a, e["instance_id"]) not in head_pairs]
    return head + tail


def load_done_set(runs_dir: Path) -> set:
    """The (instance_id, algorithm, tolerance) keys of the settled rows."""
    done: set = set()
    if not runs_dir.is_dir():
        return done
    for jsonl in runs_dir.glob("*.jsonl"):
        try:
            for _, rec in iter_rows([jsonl]):
                if rec.get("status") in SETTLED_STATUSES:
                    # The tolerance is in the key, so a run of the same
                    # instance at another tolerance is not skipped.
                    tol = rec.get("tolerance")
                    done.add((rec["instance_id"], rec["algorithm"],
                              None if tol is None else float(tol)))
        except OSError as e:
            print(f"orchestrator: warning: skipping {jsonl}: {e}", file=sys.stderr)
    return done


def _problem_n(spec: InstanceSpec) -> int:
    """Marginal length N, a record's ``n``: n_g² with a resolution, else m."""
    try:
        return spec.dense_shape()[0]
    except ValueError:                     # malformed spec: keep records writable
        return 0
