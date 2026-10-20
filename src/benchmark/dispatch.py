"""Running one (algorithm, instance) pair as a runner subprocess.

A run that writes no record still gets a row: a synthesized record whose
status is read from the exit (timeout, OOM, crash).
"""

import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from src.benchmark.schedule import _problem_n
from src.protocol.instance_record import (
    WALL_BUFFER_S, InstanceSpec, ResultRecord, utc_now_iso,
)
from src.runners.common import mentions_oom
from src.protocol.regimes import (
    GRID,
    SINKHORN_PREFIX,
    SINKHORNGRID_PREFIX,
    base_algorithm,
    canonical_algorithm,
    parse_eps_variant,
    regime_for_algorithm,
)


def _stamp_regime(rec: ResultRecord, algo: str) -> ResultRecord:
    """Overwrite ``rec.regime`` with ``algo``'s format; an unknown name leaves it as is."""
    try:
        rec.regime = regime_for_algorithm(algo)
    except ValueError:
        pass
    return rec


def _stamp_time_limit(rec: ResultRecord, spec: InstanceSpec) -> ResultRecord:
    """Record ``spec.time_limit_s`` in ``rec.algo_extras["time_limit_s"]``,
    where the aggregation reads a run's time limit."""
    rec.algo_extras = dict(rec.algo_extras or {})
    rec.algo_extras["time_limit_s"] = float(spec.time_limit_s)
    return rec


def _make_synthesized_record(spec: InstanceSpec, algo: str, *,
                             status: str, error_kind: str, error_msg: str,
                             wall_time_s: Optional[float] = None) -> ResultRecord:
    now = utc_now_iso()
    return _stamp_regime(ResultRecord(
        instance_id=spec.instance_id, algorithm=algo,
        norm=spec.norm, resolution=spec.resolution, n=_problem_n(spec),
        run_uid=str(uuid.uuid4()), started_utc=now, ended_utc=now,
        status=status,
        oom_flag=(status == "oom"),
        timeout_flag=(status == "timeout"),
        error_kind=error_kind, error_msg=error_msg[:4096],
        wall_time_s=wall_time_s,
        tolerance=spec.tolerance,
        dataset=spec.dataset, rung=spec.rung, m=spec.m,
    ), algo)


def _runner_module(algo: str) -> str:
    if algo.startswith(SINKHORNGRID_PREFIX):
        return "src.runners.run_sinkhorngrid"
    if algo.startswith(SINKHORN_PREFIX):
        return "src.runners.run_sinkhorn"
    return f"src.runners.run_{canonical_algorithm(algo).lower()}"


def _spec_from_index(ent: Dict[str, Any], *, args, output_path: Path) -> InstanceSpec:
    """The InstanceSpec of a (possibly lifted) index entry under the run controls ``args``."""
    return InstanceSpec(
        instance_id=ent["instance_id"],
        dataset=ent.get("dataset", "dotmark"),
        rung=ent.get("rung") or GRID,
        class_name=ent.get("class_name", ""),
        resolution=ent.get("resolution"),
        i=ent.get("i"), j=ent.get("j"),
        id_i=ent.get("id_i"), id_j=ent.get("id_j"),
        norm=ent["norm"],
        marginal_i_path=ent["marginal_i_path"],
        marginal_j_path=ent["marginal_j_path"],
        cost_matrix_path=ent.get("cost_matrix_path"),
        cost_seed=ent.get("cost_seed"), cost_family=ent.get("cost_family"),
        support_i_path=ent.get("support_i_path"),
        support_j_path=ent.get("support_j_path"),
        c_max=ent.get("c_max"), m=ent.get("m"), n=ent.get("n"), dim=ent.get("dim"),
        tolerance=args.tolerance,
        time_limit_s=args.time_limit,
        iter_cap=args.iter_cap,
        output_path=str(output_path),
        warmup=args.warmup,
        device=args.device,
        check_every=args.check_every,
        pareto=getattr(args, "pareto", False),
    )


def run_subprocess(
    algo: str, spec: InstanceSpec, scratch: Path,
    *, time_limit_s: int,
) -> ResultRecord:
    """Run ``algo``'s runner on ``spec`` in a new process group, killed at
    ``time_limit_s + WALL_BUFFER_S``. Returns the runner's record, or a
    synthesized timeout, oom or error row when it wrote none, with
    ``spec.time_limit_s`` recorded in either."""
    return _stamp_time_limit(
        _run_runner(algo, spec, scratch, time_limit_s=time_limit_s), spec)


def _run_runner(
    algo: str, spec: InstanceSpec, scratch: Path,
    *, time_limit_s: int,
) -> ResultRecord:
    """``run_subprocess`` without the time-limit stamp."""
    # Keyed on (algorithm, instance), with a stale output removed first: a
    # runner that exits without writing must not leave another run's record
    # to be read as its own.
    _tag = f"{algo}__{spec.instance_id}"
    spec_path = scratch / f"{_tag}.spec.json"
    out_path  = scratch / f"{_tag}.out.json"
    try:
        out_path.unlink()
    except FileNotFoundError:
        pass
    spec.to_path(spec_path)

    # SinkhornGrid runs in the separate JAX environment (no torch).
    interpreter = sys.executable
    if parse_eps_variant(algo, SINKHORNGRID_PREFIX) is not None:
        interpreter = os.environ.get("BDRS_OTT_PYTHON")
        if not interpreter:
            raise RuntimeError(
                f"{algo} needs the OTT-JAX env, but BDRS_OTT_PYTHON is unset. "
                f"Export it to the python of the environment built from "
                f"requirements-ott.txt (README, Installation).")

    cmd = [interpreter, "-m", _runner_module(algo),
           "--instance", str(spec_path), "--output", str(out_path)]

    # repr() passes ε to the runner exactly. The two prefixes are disjoint, so
    # at most one matches.
    sk_eps = parse_eps_variant(algo, SINKHORN_PREFIX)
    if sk_eps is None:
        sk_eps = parse_eps_variant(algo, SINKHORNGRID_PREFIX)
    if sk_eps is not None:
        cmd += ["--epsilon", repr(sk_eps)]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=time_limit_s + WALL_BUFFER_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        rec = _make_synthesized_record(
            spec, algo, status="timeout",
            error_kind="subprocess_timeout",
            error_msg=f"Killed after {time_limit_s}s; stderr tail:\n{stderr[-4000:]}",
            wall_time_s=float(time_limit_s),
        )
        return rec

    # A runner exits 1 after writing an error record (for a caught exception,
    # with the traceback in error_msg, which stderr does not get); that record
    # is kept over a row synthesized from stderr.
    runner_record: Optional[ResultRecord] = None
    mismatched: Optional[str] = None
    if out_path.exists():
        try:
            runner_record = ResultRecord.from_path(out_path)
        except (json.JSONDecodeError, TypeError, ValueError):
            runner_record = None
        if runner_record is not None:
            # Compared on base names, so an alias (BDRS_dense) matches its
            # runner's record.
            if base_algorithm(runner_record.algorithm) != base_algorithm(algo):
                mismatched = runner_record.algorithm
                runner_record = None
            else:
                _stamp_regime(runner_record, algo)
    if mismatched is not None:
        return _make_synthesized_record(
            spec, algo, status="error", error_kind="record_algorithm_mismatch",
            error_msg=(
                f"{out_path} holds a record for {mismatched!r}, not {algo!r}; "
                "refusing to attribute another solver's result to this cell. "
                f"runner exit={proc.returncode}; stderr tail:\n{stderr[-3000:]}"),
        )

    if proc.returncode != 0:
        if runner_record is not None:
            return runner_record
        oom = mentions_oom(stderr) or (proc.returncode in (-9, 137))
        if oom:
            return _make_synthesized_record(
                spec, algo, status="oom",
                error_kind=f"runner_killed_{proc.returncode}",
                error_msg=f"stderr tail:\n{stderr[-4000:]}",
            )
        return _make_synthesized_record(
            spec, algo, status="error",
            error_kind=f"runner_exit_{proc.returncode}",
            error_msg=f"stderr tail:\n{stderr[-4000:]}\n\nstdout tail:\n{stdout[-2000:]}",
        )

    if runner_record is not None:
        return runner_record
    if not out_path.exists():
        return _make_synthesized_record(
            spec, algo, status="error", error_kind="no_output_file",
            error_msg=f"runner exited cleanly but {out_path} not written",
        )
    return _make_synthesized_record(
        spec, algo, status="error", error_kind="invalid_output",
        error_msg=f"runner exit 0 but record unparseable; raw:\n{out_path.read_text()[:4000]}",
    )


def append_record(jsonl_path: Path, rec: ResultRecord) -> None:
    """Append ``rec`` to ``jsonl_path`` as one line.

    A run killed while appending leaves a last line without its newline. That
    line is terminated first; otherwise the new record would merge into it and
    the row reader would skip both."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("ab+") as f:
        f.seek(0, os.SEEK_END)
        if f.tell() > 0:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")
        f.write((rec.to_json() + "\n").encode("utf-8"))
