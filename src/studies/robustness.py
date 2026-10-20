"""How the iteration count and the solve time of the Grid and General solvers
change with the step size η and with the check period T_chk, on DOTmark pairs
at tol = 1e-4.

An η run solves with η = m·0.7·mean(C) for a multiplier m; a T_chk run solves
at m = 1 with T_chk forced.
"""

import argparse
import contextlib
import dataclasses
import itertools
import json
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from src.benchmark import env_capture
from src.instances.dotmark_build import MARGINAL_DIR
from src.methods.eta import eta_from_mean, grid_cost_mean
from src.protocol.instance_record import GRID, SETTLED_STATUSES, InstanceSpec
from src.protocol.paths import REPO_ROOT
from src.protocol.results_io import iter_rows
from src.runners import run_general, run_grid
from src.runners.common import scratch_dir
from src.runners.gpu_state import device_uuid


SEED = 20260610                 # draws the image pairs; also seeds the warm-up marginals
TOL = 1e-4
ITER_CAP = 50_000               # K_max
ITER_CAP_WIDE = 1_000_000       # K_max of the wide η ladder
DEVICE = "cuda:0"
CAMPAIGN_TAG = "robustness"

MULTS = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
MULTS_WIDE = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 32.0)
TCHKS = (50, 100, 200, 500, 1000, 2000, 5000)
T_CHK_FINE = 10                 # forced T_chk of the eta_fine arm

CLASSES10 = ("CauchyDensity", "ClassicImages", "GRFmoderate", "GRFrough", "GRFsmooth",
             "LogGRF", "LogitGRF", "MicroscopyImages", "Shapes", "WhiteNoise")
CLASSES3 = ("GRFmoderate", "Shapes", "WhiteNoise")
IMAGE_IDS = tuple(range(1001, 1011))    # the ten images of a DOTmark class

SMOKE_CLASSES = ("GRFmoderate",)
SMOKE_MULTS_WIDE = (1.0, 3.0)


@dataclasses.dataclass(frozen=True)
class Arm:
    """One block of runs: classes × resolutions × norms × pairs × values.

    An η arm sweeps ``multipliers`` and solves at ``T_chk_forced`` (None: the
    runner's T_chk for n_g); a T_chk arm sweeps ``tchks`` at m = 1."""

    tree: str
    sweep: str
    solver: str
    classes: Tuple[str, ...]
    resolutions: Tuple[int, ...]
    norms: Tuple[str, ...]
    n_pairs: int
    multipliers: Tuple[float, ...] = ()
    tchks: Tuple[int, ...] = ()
    T_chk_forced: Optional[int] = None
    iter_cap: int = ITER_CAP


ARMS: Dict[str, Arm] = {
    "1": Arm("eta_sweep", "eta", "Grid", CLASSES10, (64, 128), ("l1", "sql2"), 5,
             multipliers=MULTS),
    "2": Arm("eta_sweep", "eta", "General", CLASSES3, (64,), ("l1",), 2,
             multipliers=MULTS),
    "3": Arm("eta_sweep", "eta_fine", "Grid", CLASSES3, (64, 128), ("l1", "sql2"), 3,
             multipliers=MULTS, T_chk_forced=T_CHK_FINE),
    "4": Arm("eta_sweep_wide", "eta_wide", "Grid", CLASSES10, (64, 128), ("sql2",), 3,
             multipliers=MULTS_WIDE, iter_cap=ITER_CAP_WIDE),
    "5": Arm("tchk_sweep", "tchk", "Grid", CLASSES10, (32, 64, 128), ("l1", "sql2"), 3,
             tchks=TCHKS),
    "6": Arm("tchk_sweep", "tchk", "General", CLASSES3, (64,), ("l1",), 2,
             tchks=TCHKS),
}
# Cheap arms first, the wide ladder (whose m = 0.01 runs reach K_max) last.
ARM_ORDER = ("1", "2", "3", "5", "6", "4")

RUNNERS = {"Grid": run_grid, "General": run_general}

# algo_extras fields copied onto every row. The KKT residual of the delivered
# (X, α′, β′) is E = max{r_P, r_D, |g|} with r_P = primal_res_L2,
# r_D = dual_infeas_abs = ‖(α′ ⊕ β′ − C)₊‖_F and g = g_feas.
EXTRA_KEYS = (
    "primal_res_L2", "dual_infeas_abs", "g_feas",
    "dual_res_F", "native_dual_residual",
    "r_Dinf_final", "r_Dinf_machine_bound",
    "eta", "eta_scheme", "eta_alpha", "check_every",
    "certificate_valid", "general_impl",
)


@dataclasses.dataclass(frozen=True)
class Run:
    """One planned solve: the pair (ia, ib) of ``class_name`` on the n_g lattice."""

    arm: str
    tree: str
    sweep: str
    solver: str
    class_name: str
    resolution: int
    norm: str
    ia: int
    ib: int
    multiplier: Optional[float]     # η = m·η_prod; None on a T_chk arm
    T_chk: Optional[int]            # the swept T_chk; None on an η arm
    T_chk_forced: Optional[int]     # an η arm's T_chk; None: the runner's T_chk
    iter_cap: int

    @property
    def check_every(self) -> Optional[int]:
        """The forced T_chk, or None for the runner's T_chk at n_g."""
        return self.T_chk if self.T_chk is not None else self.T_chk_forced

    @property
    def key(self) -> Tuple[Any, ...]:
        """(sweep, solver, class, n_g, norm, ia, ib, m or T_chk), as ``row_key`` reads it off a row."""
        value = self.multiplier if self.multiplier is not None else self.T_chk
        return (self.sweep, self.solver, self.class_name, self.resolution,
                self.norm, self.ia, self.ib, value)


def row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    value = row["T_chk"] if row["sweep"] == "tchk" else row["multiplier"]
    return (row["sweep"], row["solver"], row["class"], row["resolution"],
            row["norm"], row["ia"], row["ib"], value)


def image_pairs(n_pairs: int) -> List[Tuple[int, int]]:
    """The first ``n_pairs`` of the C(10, 2) unordered image pairs, in sorted
    order shuffled by ``random.Random(SEED)``; the same for every class and n_g."""
    pairs = list(itertools.combinations(IMAGE_IDS, 2))
    random.Random(SEED).shuffle(pairs)
    return pairs[:n_pairs]


def smoke_arm(arm: Arm) -> Arm:
    """``arm`` cut to one class and two values of its ladder."""
    if arm.tchks:
        return dataclasses.replace(arm, classes=SMOKE_CLASSES, tchks=arm.tchks[:2])
    mults = SMOKE_MULTS_WIDE if arm.multipliers == MULTS_WIDE else arm.multipliers[:2]
    return dataclasses.replace(arm, classes=SMOKE_CLASSES, multipliers=mults)


def plan(arms: Sequence[str] = ARM_ORDER, *, smoke: bool = False) -> List[Run]:
    """Every run of ``arms``, in order arm, class, n_g, norm, pair, value."""
    runs: List[Run] = []
    for name in arms:
        arm = smoke_arm(ARMS[name]) if smoke else ARMS[name]
        values = arm.multipliers or arm.tchks
        for cls, res, norm, (ia, ib), v in itertools.product(
                arm.classes, arm.resolutions, arm.norms, image_pairs(arm.n_pairs), values):
            runs.append(Run(
                arm=name, tree=arm.tree, sweep=arm.sweep, solver=arm.solver,
                class_name=cls, resolution=res, norm=norm, ia=ia, ib=ib,
                multiplier=v if arm.multipliers else None,
                T_chk=None if arm.multipliers else v,
                T_chk_forced=arm.T_chk_forced,
                iter_cap=arm.iter_cap))
    return runs


def marginal_path(marginal_dir, class_name: str, resolution: int, image_id: int) -> Path:
    res = int(resolution)
    return Path(marginal_dir) / class_name / f"n{res}" / f"data{res}_{int(image_id)}.npy"


def instance_spec(run: Run, *, device: str = DEVICE, marginal_dir=MARGINAL_DIR) -> InstanceSpec:
    """The spec of ``run``, with the forced T_chk in ``check_every``.

    The rung is Grid for both solvers, as for a DOTmark instance in the
    benchmark: ``run_general`` then builds C from the lattice and takes
    η = 0.7·mean(C) from the closed-form mean, the η of ``run_grid``. The
    runner's warm-up is off; ``synthetic_warmup`` runs once per process."""
    i, j = run.ia - IMAGE_IDS[0], run.ib - IMAGE_IDS[0]
    return InstanceSpec(
        instance_id=f"{run.class_name}_n{run.resolution}_i{i}_j{j}_{run.norm}",
        dataset="dotmark", rung=GRID, class_name=run.class_name,
        resolution=run.resolution, i=i, j=j, id_i=run.ia, id_j=run.ib, norm=run.norm,
        marginal_i_path=str(marginal_path(marginal_dir, run.class_name, run.resolution, run.ia)),
        marginal_j_path=str(marginal_path(marginal_dir, run.class_name, run.resolution, run.ib)),
        tolerance=TOL, iter_cap=run.iter_cap, warmup=0, device=device,
        check_every=run.check_every)


@contextlib.contextmanager
def eta_multiplier(runner, m: Optional[float]) -> Iterator[None]:
    """Within the block ``runner`` solves with η = (0.7·mean(C))·m: its
    ``eta_from_mean`` is wrapped and restored on exit. ``m = None`` leaves it."""
    if m is None:
        yield
        return
    base = runner.eta_from_mean
    runner.eta_from_mean = lambda mean_c: base(mean_c) * m
    try:
        yield
    finally:
        runner.eta_from_mean = base


def row_head(run: Run) -> Dict[str, Any]:
    head: Dict[str, Any] = {
        "sweep": run.sweep, "solver": run.solver, "class": run.class_name,
        "resolution": run.resolution, "norm": run.norm, "ia": run.ia, "ib": run.ib,
    }
    if run.multiplier is not None:
        head["multiplier"] = run.multiplier
        head["prod_eta"] = eta_from_mean(grid_cost_mean(run.resolution, run.norm))
        head["T_chk_forced"] = run.T_chk_forced
    else:
        head["T_chk"] = run.T_chk
    head["iter_cap"] = run.iter_cap
    return head


def _device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if device.startswith("cuda:") else 0


def result_row(rec, head: Dict[str, Any], *, git_sha: Optional[str],
               device: str = DEVICE) -> Dict[str, Any]:
    """``head`` plus the record's outcome and the ``EXTRA_KEYS`` of its
    algo_extras; a record that is not ok adds its ``error_kind`` and ``error_msg``."""
    extras = rec.algo_extras or {}
    row = dict(head)
    row.update({
        "status": rec.status,
        "iters": rec.iters,
        "iter_cap_hit": rec.iter_cap_hit,
        "wall_time_s": rec.wall_time_s,
        "wall_time_solve_s": rec.wall_time_solve_s,
        "native_termination": rec.native_termination,
        "objective": rec.objective,
    })
    for k in EXTRA_KEYS:
        row[k] = extras.get(k)
    row["settled"] = rec.status == "ok" and not rec.iter_cap_hit
    row.update({"tolerance": TOL, "seed": SEED, "campaign_tag": CAMPAIGN_TAG,
                "git_sha": git_sha,
                "device_uuid": rec.device_uuid or _gpu_uuid(device)})
    if rec.status != "ok":
        row["error_kind"] = rec.error_kind
        row["error_msg"] = rec.error_msg
    return row


def error_row(head: Dict[str, Any], exc: BaseException, *, git_sha: Optional[str],
              device: str = DEVICE) -> Dict[str, Any]:
    """The row of a run whose runner raised."""
    row = dict(head)
    row.update({"status": "driver_error", "err": repr(exc)[:400], "settled": False,
                "tolerance": TOL, "seed": SEED, "campaign_tag": CAMPAIGN_TAG,
                "git_sha": git_sha, "device_uuid": _gpu_uuid(device)})
    return row


def _gpu_uuid(device: str) -> Optional[str]:
    return device_uuid(_device_index(device)) if device.startswith("cuda") else None


def run_one(run: Run, *, git_sha: Optional[str] = None, device: str = DEVICE,
            marginal_dir=MARGINAL_DIR) -> Dict[str, Any]:
    """Solve ``run`` with its runner and return its row; an exception becomes a
    ``driver_error`` row."""
    head = row_head(run)
    runner = RUNNERS[run.solver]
    try:
        spec = instance_spec(run, device=device, marginal_dir=marginal_dir)
        with eta_multiplier(runner, run.multiplier):
            rec = runner._run_one(spec, check_every=spec.check_every)
    except Exception as exc:                                        # noqa: BLE001
        return error_row(head, exc, git_sha=git_sha, device=device)
    return result_row(rec, head, git_sha=git_sha, device=device)


def settled_keys(path: Path) -> Set[Tuple[Any, ...]]:
    """Keys of the rows of ``path`` with a settled status; a run with another
    status (``error``, ``driver_error``) is run again."""
    if not path.is_file():
        return set()
    return {row_key(row) for _, row in iter_rows([path])
            if row.get("status") in SETTLED_STATUSES}


def end_last_line(path: Path) -> None:
    """Terminate the last line of ``path`` if a killed append left it without
    its newline, so that the fragment stays alone on its line and the next row
    starts a line of its own."""
    if not path.is_file():
        return
    with path.open("ab+") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() > 0:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")


class Tree:
    """One output tree, ``<out>/<name>/header.json`` and
    ``<out>/<name>/runs/sweep.jsonl``. The header is written on the first row
    of an invocation; each row is appended and flushed. One writer per tree."""

    def __init__(self, out_root: Path, name: str):
        self.name = name
        self.dir = Path(out_root) / name
        self.path = self.dir / "runs" / "sweep.jsonl"
        self.done = settled_keys(self.path)
        self.git_sha: Optional[str] = None
        self.n_written = 0
        self._fh = None

    def open(self) -> None:
        if self._fh is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = env_capture.write(self.dir / "header.json", REPO_ROOT)
        self.git_sha = header.get("git_sha")
        end_last_line(self.path)
        self._fh = open(self.path, "a")

    def write(self, row: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        self.n_written += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def synthetic_warmup(*, general: bool, device: str = DEVICE,
                     resolutions: Sequence[int] = (64, 128),
                     norms: Sequence[str] = ("l1", "sql2")) -> List[str]:
    """Pay the process's one-time costs (the extension build and load, the
    first solves) on seeded synthetic marginals, never on a measured pair:
    Grid at every (n_g, norm), and General at n_g = 64, l1 when ``general``.
    Returns one line per warm-up solve that did not return ok."""
    tmp = Path(tempfile.mkdtemp(prefix="bdrs_robustness_warmup_", dir=scratch_dir()))
    rng = np.random.default_rng(SEED)
    failed: List[str] = []
    try:
        for res in resolutions:
            pa, pb = tmp / f"synth_n{res}_a.npy", tmp / f"synth_n{res}_b.npy"
            np.save(pa, (rng.random(res * res) + 0.5).reshape(res, res))
            np.save(pb, (rng.random(res * res) + 0.5).reshape(res, res))
            for norm in norms:
                solvers = ["Grid", "General"] if (general and res == 64 and norm == "l1") else ["Grid"]
                for solver in solvers:
                    spec = InstanceSpec(
                        instance_id=f"SYNTH_n{res}_{norm}", dataset="dotmark", rung=GRID,
                        class_name="SYNTH", resolution=res, norm=norm,
                        marginal_i_path=str(pa), marginal_j_path=str(pb),
                        tolerance=TOL, iter_cap=200, warmup=0, device=device)
                    label = f"warm-up {solver} n_g={res} {norm}"
                    try:
                        rec = RUNNERS[solver]._run_one(spec)
                    except Exception as exc:                        # noqa: BLE001
                        failed.append(f"{label}: {exc!r}"[:400])
                        print(f"  {failed[-1]}", flush=True)
                        continue
                    print(f"  {label}: {rec.status}", flush=True)
                    if rec.status != "ok":
                        failed.append(f"{label}: {rec.status}/{rec.error_kind}: "
                                      f"{rec.error_msg}"[:400])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return failed


def _parse_arms(tokens: Sequence[str]) -> List[str]:
    """The arms named in ``tokens``, each once, in the order first named."""
    arms = list(dict.fromkeys(a for t in tokens for a in t.split()))
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; valid: {' '.join(ARM_ORDER)}")
    return arms


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="The η and T_chk robustness sweeps of the Grid and General solvers.")
    ap.add_argument("--out", required=True, type=Path,
                    help="output root; trees eta_sweep, eta_sweep_wide, tchk_sweep")
    ap.add_argument("--arms", nargs="+", default=list(ARM_ORDER),
                    help=f"arms to run, in order (default: {' '.join(ARM_ORDER)}). 1: η, Grid; "
                         "2: η, General; 3: η at T_chk = 10, Grid; 4: wide η ladder, Grid; "
                         "5: T_chk, Grid; 6: T_chk, General")
    ap.add_argument("--smoke", action="store_true",
                    help="one class and two ladder values per arm; the rows are a "
                         "subset of the full runs, so give a separate --out")
    args = ap.parse_args(argv)
    arms = _parse_arms(args.arms)

    runs = plan(arms, smoke=args.smoke)
    trees: Dict[str, Tree] = {}
    for run in runs:
        if run.tree not in trees:
            trees[run.tree] = Tree(args.out, run.tree)
    todo = [r for r in runs if r.key not in trees[r.tree].done]
    print(f"robustness: out={args.out} arms={' '.join(arms)} smoke={args.smoke} "
          f"planned={len(runs)} settled={len(runs) - len(todo)} to_run={len(todo)}", flush=True)
    if not todo:
        return 0

    t0 = time.time()
    failed = synthetic_warmup(general=any(r.solver == "General" for r in todo))
    if failed and torch.cuda.is_available():
        print("robustness: a warm-up solve on the GPU did not return ok; no run was "
              "started:\n  " + "\n  ".join(failed), file=sys.stderr, flush=True)
        return 2
    n_bad = 0
    for name in dict.fromkeys(r.arm for r in todo):
        ta = time.time()
        arm_runs = [r for r in todo if r.arm == name]
        n_settled = 0
        for run in arm_runs:
            tree = trees[run.tree]
            tree.open()
            row = run_one(run, git_sha=tree.git_sha)
            tree.write(row)
            n_settled += bool(row.get("settled"))
            n_bad += row.get("status") != "ok"
        print(f"arm {name} -> {ARMS[name].tree}: {len(arm_runs)} rows, {n_settled} settled, "
              f"{time.time() - ta:.1f} s", flush=True)
    for tree in trees.values():
        tree.close()
        print(f"  {tree.name}: {tree.n_written} rows appended to {tree.path}", flush=True)
    print(f"robustness done in {time.time() - t0:.1f} s; {n_bad} rows not ok", flush=True)
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
