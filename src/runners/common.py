"""Helpers every runner may use, and the runners' command-line entry point.

Torch-free at import: the SinkhornGrid runner, which runs in the JAX
environment without torch, imports it.
"""

import argparse
import math
import os
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from src.protocol.instance_record import (
    SETTLED_STATUSES, InstanceSpec, ResultRecord, utc_now_iso,
)
from src.protocol.regimes import canonical_regime, regime_for_algorithm


def load_marginal(path: str) -> np.ndarray:
    return np.load(path).astype(np.float64).flatten()


def load_marginal_unit_mass(path: str) -> np.ndarray:
    a = load_marginal(path)
    return a / a.sum()


def finite_or_none(v: Any) -> Optional[float]:
    """``v`` as a float, or None when it is None, NaN or infinite (``json.dumps``
    would write the non-standard token ``NaN``)."""
    if v is None:
        return None
    f = float(v)
    return f if np.isfinite(f) else None


def json_safe(obj: Any, *, nonfinite_to_none: bool = False) -> Any:
    """``obj`` with JSON-serialisable values only (an unknown type becomes its
    repr); with ``nonfinite_to_none`` a NaN or infinite float becomes None."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v, nonfinite_to_none=nonfinite_to_none) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v, nonfinite_to_none=nonfinite_to_none) for v in obj]
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if nonfinite_to_none and not math.isfinite(f) else f
    return repr(obj)


def cuda_sync(device) -> None:
    """Wait for the work queued on ``device``, so that a clock read afterwards covers it."""
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def cuda_ordinal(device: str) -> int:
    """The ordinal ``k`` of ``"cuda:<k>"``; 0 for any other device string."""
    if device.startswith("cuda") and ":" in device:
        try:
            return int(device.split(":")[1])
        except ValueError:
            return 0
    return 0


_SCRATCH_CANDIDATES = ("/dev/shm", "/tmp")


def scratch_dir() -> str:
    """Where a runner writes its temporary files, preferring RAM-backed /dev/shm."""
    for cand in _SCRATCH_CANDIDATES:
        if os.path.isdir(cand) and os.access(cand, os.W_OK):
            return cand
    return tempfile.gettempdir()


# Warm-up dummies: seeded, so a runner warms up on the same data in every run,
# and independent of the instance being measured.
WARMUP_SEED = 20260427


def warmup_marginal(rng: np.random.Generator, n: int) -> np.ndarray:
    """A strictly positive unit-mass dummy marginal of length ``n``."""
    x = np.abs(rng.standard_normal(n)) + 1e-3
    return x / x.sum()


def warmup_blobs(ng: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit-mass Gaussian-blob dummy marginals on an (ng, ng) grid."""
    yy, xx = np.mgrid[0:ng, 0:ng].astype(np.float64)

    def blob(cx, cy, s):
        return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * s * s)))
    a = blob(ng * 0.3, ng * 0.35, ng * 0.15) + 1e-6 * rng.random((ng, ng))
    b = blob(ng * 0.7, ng * 0.6, ng * 0.2) + 1e-6 * rng.random((ng, ng))
    a /= a.sum()
    b /= b.sum()
    return a, b


def _marginal_length(spec: InstanceSpec) -> int:
    """The source marginal's length; 0 for a spec too malformed to give it, so a
    failure row can still be written."""
    try:
        return spec.dense_shape()[0]
    except ValueError:
        return 0


def failure_record(spec: InstanceSpec, algorithm: str, *, status: str, error_kind: str,
                   error_msg: str, started_utc: str, run_uid: str,
                   m: Optional[int] = None, algo_extras: Optional[Dict[str, Any]] = None,
                   **fields: Any) -> ResultRecord:
    """The row of a run that produced no solution (refused, out of memory, timed
    out or crashed), so that every scheduled pair leaves exactly one row."""
    return ResultRecord(
        instance_id=spec.instance_id, algorithm=algorithm,
        norm=spec.norm, resolution=spec.resolution, n=_marginal_length(spec), m=m,
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status=status, error_kind=error_kind, error_msg=error_msg[:4096],
        oom_flag=status in ("oom", "oom_recovery"),
        tolerance=spec.tolerance,
        regime=regime_for_algorithm(algorithm),
        dataset=spec.dataset,
        rung=canonical_regime(spec.rung) if spec.rung else None,
        algo_extras=dict(algo_extras or {}),
        **fields,
    )


# Phrases that name an allocation failure, matched case-insensitively: torch and
# the CUDA runtime; cuPDLPx's CUDA_CHECK, which prints the error name and then
# exits; host allocation in C and C++ solver layers (HALO's embedded cuPDLPx
# prints "malloc failed"); JAX/XLA.
OOM_NEEDLES = (
    "out of memory", "outofmemoryerror", "out of gpu memory", "cuda oom",
    "cudamalloc", "cudaerrormemoryallocation", "cuda error 2:", "bad_alloc",
    "cannot allocate memory", "malloc failed", "resource_exhausted",
)


def mentions_oom(text: str) -> bool:
    """Whether ``text`` (an exception's repr, or a process's stderr) names an allocation failure."""
    s = text.lower()
    return any(n in s for n in OOM_NEEDLES)


def is_oom_exception(exc: BaseException) -> bool:
    """Whether ``exc`` is an allocation failure. Torch's ``OutOfMemoryError`` is
    matched by class name, so this module needs no torch."""
    if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
        return True
    return mentions_oom(repr(exc))


def runner_main(argv, *, description: str,
                run: Callable[[InstanceSpec, argparse.Namespace], ResultRecord],
                failure: Callable[..., ResultRecord],
                add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
                check_args: Optional[Callable[[argparse.Namespace], Optional[str]]] = None,
                ) -> int:
    """The entry point of ``python -m src.runners.run_<x> --instance spec.json
    --output record.json``.

    An exception escaping ``run`` becomes the record ``failure(...)`` builds;
    a spec that fails validation raises and writes no record. Exit code: 0 for
    a settled status, 1 otherwise (the orchestrator uses the written record
    either way), 2 when ``check_args`` rejects the arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--instance", required=True, type=Path,
                        help="Path to InstanceSpec JSON.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to write ResultRecord JSON.")
    if add_args is not None:
        add_args(parser)
    args = parser.parse_args(argv)
    if check_args is not None:
        problem = check_args(args)
        if problem:
            print(problem, file=sys.stderr)
            return 2

    spec = InstanceSpec.from_path(args.instance)
    spec.validate()

    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    try:
        rec = run(spec, args)
    except Exception as e:                                  # noqa: BLE001
        oom = is_oom_exception(e)
        rec = failure(spec, args, status="oom" if oom else "error",
                      error_kind="cuda_oom" if oom else type(e).__name__,
                      error_msg=f"{e!r}\n{traceback.format_exc()}",
                      started_utc=started_utc, run_uid=run_uid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rec.to_path(args.output)
    print(f"{rec.algorithm} {spec.instance_id} status={rec.status} "
          f"wall={rec.wall_time_s} iters={rec.iters} feaserr={rec.feaserr} "
          f"objective={rec.objective}")
    return 0 if rec.status in SETTLED_STATUSES else 1
