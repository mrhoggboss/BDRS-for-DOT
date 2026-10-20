"""
PDOT runner: one Julia process per instance, for standalone runs and preflight.

    python -m src.runners.run_pdot --instance spec.json --output result.json

The Julia driver ``vendor/PDOT.jl/bench_driver.jl`` implements the protocol
(the original marginals passed as stored, the clocks, PDOT's own stop at
eps_optimal_absolute = eps_optimal_relative = tol, the Pareto block); this
module starts it and relays its record. The orchestrator instead keeps one warm
Julia process for all PDOT instances (``src/benchmark/pdot_adapter.py``),
through the same ``JuliaPdot`` and ``pdot_size_gate``. The orchestrator imports
this module, so it must import without torch.
"""

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

from src.runners.common import failure_record, mentions_oom, runner_main
from src.protocol.paths import REPO_ROOT
from src.protocol.instance_record import (
    WALL_BUFFER_S, InstanceSpec, ResultRecord, utc_now_iso,
)


DEFAULT_PDOT_DIR = REPO_ROOT / "vendor" / "PDOT.jl"
DEFAULT_DRIVER = DEFAULT_PDOT_DIR / "bench_driver.jl"


# Measured on an 80 GB A100 (79.25 GiB usable): at N = 65536 PDOT holds the
# dense C and the dense plan on the device (32 GiB each) and fails 59 s in,
# allocating a third block of that size. At N = 16384 each buffer is 2 GiB.
SKIP_MARGINAL_LEN_GE = 65536


def pdot_size_gate(spec: InstanceSpec) -> Optional[str]:
    """The reason to record this instance as ``oom`` without running it, or None."""
    n_total = spec.dense_shape()[0]
    if n_total < SKIP_MARGINAL_LEN_GE:
        return None
    gib = n_total * n_total * 8 / 2**30
    return (f"PDOT at N = {n_total} skipped preemptively: the dense cost and "
            f"the dense plan are {gib:.2f} GiB each on device, and PDOT runs out "
            f"of memory requesting a third block of that size (at N = 65536 on "
            f"an 80 GB A100: 96.00 GiB requested, 79.25 GiB available).")


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str,
                         **fields) -> ResultRecord:
    return failure_record(spec, "PDOT", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, m=spec.dense_shape()[0],
                          **fields)


class JuliaPdot:
    """One Julia process running PDOT's bench driver.

    The driver warms up, prints ``READY`` on stderr, then answers each spec,
    sent as one JSON line on stdin, with one ResultRecord JSON line on stdout;
    stdout lines that do not start with ``{`` are skipped. Two daemon threads
    drain stdout and stderr, so every wait ends at its deadline even while
    Julia prints nothing, and Julia never blocks on a full stderr pipe."""

    def __init__(self, driver: Path, project: Path, *, startup_timeout_s: float = 300.0,
                 wall_buffer_s: float = WALL_BUFFER_S):
        self.cmd = ["julia", f"--project={project}", str(driver)]
        self.startup_timeout_s = startup_timeout_s
        self.wall_buffer_s = wall_buffer_s
        self.proc: Optional[subprocess.Popen] = None
        self._stdout: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stderr: Deque[str] = deque(maxlen=200)
        self._stderr_done = threading.Event()
        self._ready = threading.Event()
        self._oom_after_ready = threading.Event()

    def _drain_stdout(self, stream, lines: "queue.Queue[Optional[str]]") -> None:
        for line in iter(stream.readline, ""):
            lines.put(line)
        lines.put(None)                                    # end of stdout: Julia exited

    def _drain_stderr(self, stream, tail: Deque[str], ready: threading.Event,
                      done: threading.Event, oom_after_ready: threading.Event) -> None:
        """Keep the last 200 lines, and flag any line after READY that names an
        allocation failure, so a death is classified as ``oom`` even when the
        stack trace that follows pushes that line out of the tail."""
        for line in iter(stream.readline, ""):
            s = line.rstrip()
            tail.append(s)
            if ready.is_set():
                if mentions_oom(s):
                    oom_after_ready.set()
            elif s.strip() == "READY":
                ready.set()
        done.set()

    def _stderr_tail(self) -> str:
        self._stderr_done.wait(timeout=5)
        return "\n".join(self._stderr)

    def launch(self) -> None:
        """Start Julia and wait for READY.

        Raises RuntimeError if Julia exits first, and TimeoutError (after
        killing it) if READY does not come within ``startup_timeout_s``."""
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, start_new_session=True,
        )
        self._stdout = queue.Queue()
        self._stderr = deque(maxlen=200)
        self._stderr_done = threading.Event()
        self._ready = threading.Event()
        self._oom_after_ready = threading.Event()
        threading.Thread(target=self._drain_stdout, args=(self.proc.stdout, self._stdout),
                         daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(
            self.proc.stderr, self._stderr, self._ready, self._stderr_done,
            self._oom_after_ready), daemon=True).start()
        deadline = time.monotonic() + self.startup_timeout_s
        while not self._ready.wait(timeout=0.05):
            if self.proc.poll() is not None:
                code = self.proc.returncode
                tail = self._stderr_tail()
                self.proc = None
                raise RuntimeError(f"Julia exited code={code} before READY:\n{tail[-4000:]}")
            if time.monotonic() > deadline:
                tail = "\n".join(self._stderr)
                self.kill()
                raise TimeoutError(
                    f"Julia did not print READY within {self.startup_timeout_s}s:\n{tail[-4000:]}")

    def solve(self, spec: InstanceSpec, *,
              failure: Callable[..., ResultRecord]) -> ResultRecord:
        """Send ``spec`` and return its record, launching Julia first if it is not running.

        ``failure(status=, error_kind=, error_msg=[, wall_time_s=])`` builds
        the row when no record comes back:

        - Julia exits: ``oom`` when its exit code is -9 or 137 (SIGKILL, as
          the Linux OOM killer sends) or a stderr line after READY names an
          allocation failure, else ``error``. The next call relaunches Julia.
        - A reply that is not a ResultRecord: ``error``.
        - No reply within ``time_limit_s + wall_buffer_s``: ``timeout`` with
          wall time ``time_limit_s``; Julia is killed.

        A returned ``oom`` or ``oom_recovery`` record means the driver caught
        the failure while Julia still holds its device memory, so Julia is
        killed before the next solver runs."""
        if self.proc is None or self.proc.poll() is not None:
            self.launch()
        try:
            self.proc.stdin.write(json.dumps(spec_to_dict(spec)) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            return failure(status="error", error_kind="adapter_pipe_broken", error_msg=repr(e))

        deadline = time.monotonic() + spec.time_limit_s + self.wall_buffer_s
        skipped_lines: List[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                line = self._stdout.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                try:
                    code = self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    code = None
                tail = self._stderr_tail()               # waits for the drain, so the OOM flag is final
                oom_killed = (code in (-9, 137)) or self._oom_after_ready.is_set()
                self.kill()
                return failure(
                    status="oom" if oom_killed else "error",
                    error_kind="adapter_julia_oom_killed" if oom_killed else "adapter_julia_died",
                    error_msg=(
                        f"exit={code}\n"
                        f"stdout skipped: {skipped_lines!r}\n"
                        f"stderr tail:\n{tail[-4000:]}"
                    ),
                )
            stripped = line.strip()
            if not stripped or not stripped.startswith("{"):
                skipped_lines.append(stripped[:200])
                if len(skipped_lines) > 50:
                    skipped_lines = skipped_lines[-50:]
                continue
            try:
                rec_dict = json.loads(stripped)
            except json.JSONDecodeError as e:
                return failure(
                    status="error", error_kind="adapter_bad_json",
                    error_msg=f"{e!r}; raw: {stripped[:1000]!r}; skipped: {skipped_lines!r}",
                )
            valid_fields = set(ResultRecord.__dataclass_fields__.keys())
            filtered = {k: v for k, v in rec_dict.items() if k in valid_fields}
            try:
                rec = ResultRecord(**filtered)
            except TypeError as e:
                return failure(
                    status="error", error_kind="adapter_record_invalid",
                    error_msg=f"{e!r}; rec_dict keys: {list(rec_dict.keys())}",
                )
            if rec.status in ("oom", "oom_recovery"):
                self.kill()
            return rec

        self.kill()
        return failure(
            status="timeout", error_kind="adapter_timeout",
            error_msg=f"No response within {spec.time_limit_s + self.wall_buffer_s}s",
            wall_time_s=float(spec.time_limit_s),
        )

    def kill(self) -> None:
        """SIGKILL Julia's process group, so its device memory is released."""
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None

    def shutdown(self) -> None:
        """Close stdin (the driver exits at end of input); kill after 10 s."""
        if self.proc is None:
            return
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:                                   # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.kill()
            return
        self.proc = None


def _run_one(spec: InstanceSpec) -> ResultRecord:
    # The gate runs before the cost is materialized: a gated procedural C
    # (32 GiB at N = 65536) would be written to TMPDIR only to be deleted.
    _gate = pdot_size_gate(spec)
    if _gate is not None:
        return _make_failure_record(
            spec, status="oom", error_kind="pdot_device_oom_structural",
            error_msg=_gate, started_utc=utc_now_iso(), run_uid=str(uuid.uuid4()),
        )
    # The driver reads C from a file, so a procedural cost is written out for
    # the solve and removed after it. Imported here because dense_cost imports
    # torch.
    from src.instances.dense_cost import materialized_cost_path
    with materialized_cost_path(spec) as spec_with_cost:
        return _run_one_inner(spec_with_cost)


def _run_one_inner(spec: InstanceSpec) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    driver = Path(os.environ.get("BDRS_PDOT_DRIVER", str(DEFAULT_DRIVER)))
    project = Path(os.environ.get("BDRS_PDOT_PROJECT", str(DEFAULT_PDOT_DIR)))
    if not driver.is_file():
        return _make_failure_record(
            spec, status="error", error_kind="driver_missing",
            error_msg=f"bench_driver.jl not found at {driver}",
            started_utc=started_utc, run_uid=run_uid,
        )
    if not (project / "Project.toml").is_file():
        return _make_failure_record(
            spec, status="error", error_kind="project_missing",
            error_msg=f"PDOT.jl project not found at {project} (no Project.toml)",
            started_utc=started_utc, run_uid=run_uid,
        )

    # Julia's first compile of PDOT.optimize and its CUDA kernels can take
    # 30-90 s on a cold node.
    julia = JuliaPdot(driver, project, startup_timeout_s=float(
        os.environ.get("BDRS_PDOT_WARMUP_TIMEOUT", "300")))
    try:
        try:
            julia.launch()
        except (RuntimeError, TimeoutError) as e:
            return _make_failure_record(
                spec, status="error", error_kind="warmup_failed", error_msg=str(e),
                started_utc=started_utc, run_uid=run_uid,
            )
        return julia.solve(spec, failure=lambda **kw: _make_failure_record(
            spec, started_utc=started_utc, run_uid=run_uid, **kw))
    finally:
        julia.shutdown()


def spec_to_dict(spec: InstanceSpec) -> Dict[str, Any]:
    """The spec as the flat JSON object the driver reads from one stdin line."""
    return {
        "instance_id": spec.instance_id,
        "class_name": spec.class_name,
        "resolution": spec.resolution,
        # The driver sizes an instance without a lattice from m.
        "m": spec.dense_shape()[0],
        "n_cols": spec.dense_shape()[1],
        "i": spec.i, "j": spec.j,
        "id_i": spec.id_i, "id_j": spec.id_j,
        "norm": spec.norm,
        "marginal_i_path": spec.marginal_i_path,
        "marginal_j_path": spec.marginal_j_path,
        "cost_matrix_path": spec.cost_matrix_path,
        "tolerance": spec.tolerance,
        "time_limit_s": spec.time_limit_s,
        "iter_cap": spec.iter_cap,
        "warmup": spec.warmup,
        "device": spec.device,              # unread: PDOT runs on CUDA device 0
        "pareto": bool(getattr(spec, "pareto", False)),   # the driver scores the Pareto block itself
    }


def main(argv=None):
    return runner_main(
        argv, description="PDOT runner (one-shot Julia per call).",
        run=lambda spec, args: _run_one(spec),
        failure=lambda spec, args, **kw: _make_failure_record(spec, **kw))


if __name__ == "__main__":
    sys.exit(main())
