"""Checks of the environment and the solver builds before a benchmark run.

The per-check results go to ``--report-aux``; a failed check makes the exit
code non-zero only under ``--strict``. Checks (``--check``, repeatable):

  env                  the environment fingerprint, written to ``--report``
  builds               Grid, General, DROT, PDOT, HOT and HALO on the first
                       n_g = 32 sql2 DOTmark instance; a settled status passes
  hot_recovery_format  whether HOT's recover_transport_plan returns a dense
                       or a sparse tensor
  pdot_jit             five PDOT solves in one Julia process; flags wall times
                       that never increase (JIT cost left after the warm-up)
  sinkhorn             POT >= 0.9.4, and a run_sinkhorn solve at ε = 0.01 that
                       ends ok with its plan on the GPU

    python -m src.benchmark.preflight --check env --check builds [--strict]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.protocol.paths import REPO_ROOT
from src.benchmark import env_capture
from src.protocol.instance_record import InstanceSpec, ResultRecord, utc_now_iso


DEFAULT_REPORT_PATH = REPO_ROOT / "results" / "bench" / "header.json"


def _make_smoke_spec(scratch: Path) -> Optional[InstanceSpec]:
    """The first n_g = 32 sql2 DOTmark entry whose marginals exist, or None."""
    idx_path = REPO_ROOT / "data" / "dotmark_instances_index.json"
    if not idx_path.is_file():
        return None
    try:
        index = json.loads(idx_path.read_text())
    except json.JSONDecodeError:
        return None
    pick = None
    for ent in index:
        if ent.get("resolution") == 32 and ent.get("norm") == "sql2":
            if os.path.exists(ent["marginal_i_path"]) and os.path.exists(ent["marginal_j_path"]):
                pick = ent
                break
    if pick is None:
        return None
    out_path = scratch / "smoke.out.json"
    return InstanceSpec(
        instance_id=pick["instance_id"],
        class_name=pick["class_name"], resolution=32,
        i=pick["i"], j=pick["j"], id_i=pick["id_i"], id_j=pick["id_j"],
        norm="sql2",
        marginal_i_path=pick["marginal_i_path"],
        marginal_j_path=pick["marginal_j_path"],
        cost_matrix_path=pick.get("cost_matrix_path"),
        tolerance=1e-4, time_limit_s=120, iter_cap=10_000,
        output_path=str(out_path), warmup=1, device="cuda:0",
    )


def check_env(report_path: Path) -> Dict[str, Any]:
    info = env_capture.write(report_path, REPO_ROOT)
    return {
        "check": "env",
        "ok": True,
        "fingerprint_sha256": info["fingerprint_sha256"],
        "header_path": str(report_path),
    }


def _smoke_one_runner(algo: str, spec: InstanceSpec, scratch: Path,
                      time_limit_s: int) -> Dict[str, Any]:
    spec_path = scratch / f"{algo}_spec.json"
    out_path  = scratch / f"{algo}_out.json"
    spec_for_run = InstanceSpec(
        **{**{k: getattr(spec, k) for k in spec.__dataclass_fields__.keys()},
           "output_path": str(out_path)}
    )
    spec_for_run.to_path(spec_path)
    cmd = [sys.executable, "-m", f"src.runners.run_{algo.lower()}",
           "--instance", str(spec_path), "--output", str(out_path)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit_s + 60)
    elapsed = time.perf_counter() - t0
    out: Dict[str, Any] = {
        "algo": algo,
        "elapsed_s": elapsed,
        "exit_code": proc.returncode,
    }
    if not out_path.exists():
        out.update({
            "ok": False,
            "reason": "no_output",
            "stderr_tail": proc.stderr[-2000:],
            "stdout_tail": proc.stdout[-2000:],
        })
        return out
    try:
        rec = ResultRecord.from_path(out_path)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        out.update({"ok": False, "reason": f"invalid_record: {e!r}"})
        return out
    out.update({
        "ok": rec.status in ("ok", "timeout", "oom", "oom_recovery"),
        "status": rec.status,
        "wall_time_s": rec.wall_time_s,
        "iters": rec.iters,
        "feaserr": rec.feaserr,
        "error_kind": rec.error_kind,
        "error_msg": (rec.error_msg or "")[:4000] if rec.error_msg else None,
    })
    return out


def check_builds(scratch: Path) -> Dict[str, Any]:
    spec = _make_smoke_spec(scratch)
    if spec is None:
        return {
            "check": "builds",
            "ok": False,
            "reason": "no_32_sql2_instance_in_index; run `python -m src.instances.dotmark_build` first",
        }
    results = []
    for algo in ("Grid", "General", "DROT", "PDOT", "HOT", "HALO"):
        try:
            r = _smoke_one_runner(algo, spec, scratch, time_limit_s=spec.time_limit_s)
        except subprocess.TimeoutExpired as e:
            r = {"algo": algo, "ok": False, "reason": f"timeout: {e!r}"}
        except Exception as e:                              # noqa: BLE001
            r = {"algo": algo, "ok": False, "reason": f"smoke_exception: {e!r}"}
        results.append(r)
    all_ok = all(r.get("ok") for r in results)
    return {"check": "builds", "ok": all_ok, "per_runner": results}


def check_hot_recovery_format(scratch: Path) -> Dict[str, Any]:
    """Type, shape and dtype of what HOT's `recover_transport_plan` returns."""
    try:
        from src.runners.run_hot import HOTSolver, _build_b_c_cMax  # noqa
    except ImportError as e:
        return {"check": "hot_recovery_format", "ok": False, "reason": repr(e)}
    if HOTSolver is None:
        return {"check": "hot_recovery_format", "ok": False, "reason": "HOTSolver unimportable"}
    try:
        import torch
    except ImportError as e:
        return {"check": "hot_recovery_format", "ok": False, "reason": repr(e)}
    if not torch.cuda.is_available():
        return {"check": "hot_recovery_format", "ok": False, "reason": "no_cuda"}

    spec = _make_smoke_spec(scratch)
    if spec is None or spec.norm != "sql2":
        return {"check": "hot_recovery_format", "ok": False,
                "reason": "no_smoke_sql2_instance"}
    b, c, c_Max = _build_b_c_cMax(spec)
    solver = HOTSolver(
        b=b, c=c, m=spec.resolution, n=spec.resolution, cMax=c_Max,
        max_iters=200, tolerance=1e-3, check_freq=50,
        sigma=None, adjust_sigma=True, logging=True,
        dtype=torch.float64, device="cuda",
    )
    solver.optimize()
    X = solver.recover_transport_plan()
    info = {
        "type": str(type(X)),
        "shape": tuple(getattr(X, "shape", ())),
        "is_dense": isinstance(X, torch.Tensor) and not X.is_sparse,
        "is_sparse": isinstance(X, torch.Tensor) and X.is_sparse,
        "dtype": str(getattr(X, "dtype", None)),
    }
    return {"check": "hot_recovery_format", "ok": True, **info}


def check_pdot_jit(scratch: Path, n_warmups: int = 5) -> Dict[str, Any]:
    """Time `n_warmups` solves of the smoke instance in one PDOT process.

    A wall_time_s that never increases from one solve to the next indicates
    JIT or setup cost left after the driver's per-instance warm-up.
    """
    from src.benchmark.pdot_adapter import PdotAdapter
    from src.runners.run_pdot import DEFAULT_DRIVER, DEFAULT_PDOT_DIR
    spec = _make_smoke_spec(scratch)
    if spec is None or spec.norm != "sql2":
        return {"check": "pdot_jit", "ok": False, "reason": "no_smoke_sql2_instance"}
    try:
        adapter = PdotAdapter(DEFAULT_DRIVER, DEFAULT_PDOT_DIR)
    except Exception as e:                                  # noqa: BLE001
        return {"check": "pdot_jit", "ok": False, "reason": f"adapter_failed: {e!r}"}

    times = []
    try:
        for _ in range(n_warmups):
            r = adapter.run(spec)
            times.append(r.wall_time_s)
    finally:
        adapter.shutdown()

    valid = [t for t in times if t is not None]
    monotonic_decrease = len(valid) >= 2 and all(
        valid[i] >= valid[i + 1] for i in range(len(valid) - 1)
    )
    return {
        "check": "pdot_jit",
        "ok": True,
        "wall_times_s": times,
        "monotonic_decrease": monotonic_decrease,
        "warning": (
            "Wall times decrease monotonically — residual JIT/setup. Consider "
            "an extra warmup at this size in bench_driver.jl."
            if monotonic_decrease else None
        ),
    }


def check_sinkhorn(scratch: Path) -> Dict[str, Any]:
    """Require POT >= 0.9.4 and solve the smoke instance with run_sinkhorn at
    ε = 0.01. Fails unless POT returned the plan as a CUDA tensor: a silent CPU
    fallback would make the timings meaningless.
    """
    try:
        import ot                                           # noqa: F401
    except ImportError as e:
        return {"check": "sinkhorn", "ok": False,
                "reason": f"pot_not_installed: {e!r}"}
    pot_version = getattr(ot, "__version__", "0.0.0")
    try:
        version_tuple = tuple(int(x) for x in pot_version.split(".")[:3])
    except (ValueError, AttributeError):
        version_tuple = (0, 0, 0)
    if version_tuple < (0, 9, 4):
        return {"check": "sinkhorn", "ok": False,
                "reason": f"pot_version_too_old: {pot_version} (need >= 0.9.4)"}

    spec = _make_smoke_spec(scratch)
    if spec is None:
        return {"check": "sinkhorn", "ok": False,
                "reason": "no_smoke_instance; run `python -m src.instances.dotmark_build` first"}

    spec_path = scratch / "sinkhorn_spec.json"
    out_path  = scratch / "sinkhorn_out.json"
    spec_for_run = InstanceSpec(
        **{**{k: getattr(spec, k) for k in spec.__dataclass_fields__.keys()},
           "output_path": str(out_path)}
    )
    spec_for_run.to_path(spec_path)
    cmd = [sys.executable, "-m", "src.runners.run_sinkhorn",
           "--instance", str(spec_path), "--output", str(out_path),
           "--epsilon", "0.01"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=spec_for_run.time_limit_s + 60)
    except subprocess.TimeoutExpired as e:
        return {"check": "sinkhorn", "ok": False, "reason": f"timeout: {e!r}"}
    elapsed = time.perf_counter() - t0

    if not out_path.exists():
        return {"check": "sinkhorn", "ok": False, "reason": "no_output",
                "exit_code": proc.returncode,
                "stderr_tail": proc.stderr[-2000:],
                "stdout_tail": proc.stdout[-2000:],
                "elapsed_s": elapsed}
    try:
        rec = ResultRecord.from_path(out_path)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        return {"check": "sinkhorn", "ok": False,
                "reason": f"invalid_record: {e!r}", "elapsed_s": elapsed}

    extras = rec.algo_extras or {}
    pot_backend = extras.get("pot_backend")
    backend_ok = pot_backend == "torch_cuda"

    return {
        "check": "sinkhorn",
        "ok": rec.status == "ok" and backend_ok,
        "status": rec.status,
        "algorithm": rec.algorithm,
        "wall_time_s": rec.wall_time_s,
        "iters": rec.iters,
        "feaserr": rec.feaserr,
        "objective": rec.objective,
        "pot_version": pot_version,
        "pot_backend": pot_backend,
        "backend_ok": backend_ok,
        "warning": (
            None if backend_ok
            else f"POT did not keep result on CUDA (pot_backend={pot_backend!r}); "
                 "wall-time numbers may be uninterpretable due to CPU fallback."
        ),
        "error_kind": rec.error_kind,
        "error_msg": (rec.error_msg or "")[:4000] if rec.error_msg else None,
        "elapsed_s": elapsed,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--check", action="append", default=None,
        choices=["env", "builds", "hot_recovery_format", "pdot_jit", "sinkhorn"],
        help="Repeat to run multiple checks (default: env + builds).",
    )
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH,
                   help="Where to write the env fingerprint JSON.")
    p.add_argument("--report-aux", type=Path, default=None,
                   help="Where to write the aux check log JSON. "
                        "Defaults to <report>.preflight.json.")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on any failed check.")
    args = p.parse_args(argv)

    checks = args.check or ["env", "builds"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    aux_path = args.report_aux or args.report.with_suffix(args.report.suffix + ".preflight.json")

    scratch = Path(tempfile.mkdtemp(prefix="bdrs_preflight_"))
    results: List[Dict[str, Any]] = []
    overall_ok = True
    try:
        for c in checks:
            print(f"preflight: running check {c}...")
            try:
                if c == "env":
                    r = check_env(args.report)
                elif c == "builds":
                    r = check_builds(scratch)
                elif c == "hot_recovery_format":
                    r = check_hot_recovery_format(scratch)
                elif c == "pdot_jit":
                    r = check_pdot_jit(scratch)
                elif c == "sinkhorn":
                    r = check_sinkhorn(scratch)
                else:
                    r = {"check": c, "ok": False, "reason": "unknown check"}
            except Exception as e:                          # noqa: BLE001
                r = {"check": c, "ok": False, "reason": repr(e)}
            results.append(r)
            ok = bool(r.get("ok"))
            print(f"  {c}: {'OK' if ok else 'FAIL'} -- {json.dumps({k:v for k,v in r.items() if k!='per_runner'})[:240]}")
            if not ok:
                overall_ok = False
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    aux_path.write_text(json.dumps({
        "captured_utc": utc_now_iso(),
        "checks": results,
        "overall_ok": overall_ok,
    }, indent=2))
    print(f"preflight: aux log written to {aux_path}")
    return 0 if (overall_ok or not args.strict) else 1


if __name__ == "__main__":
    sys.exit(main())
