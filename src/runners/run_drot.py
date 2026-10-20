"""DROT runner: upstream DROT (Douglas–Rachford splitting for OT, CUDA) through
the shim ``vendor/drot/shim/bdrs_shim``, run as a subprocess.

DROT is fp32 only: its ``step`` launches the float kernels whatever T is. The
runner writes C and the original marginals (r, c) as fp32 blobs, with no
floor (DROT needs no strictly positive marginals), and scores the returned
plan, cast to fp64, against the original fp64 marginals.

DROT's kernels read C column-major (C[i, j] at offset i + j·N in
``kernels.hpp``) and the shim copies the blob to the device unchanged, so the
blob holds Cᵀ row-major; a row-major C would be solved as Cᵀ.

t_solve is the shim's ``solve_seconds``: DROT's iteration loop, its residual
polls included. The shim starts that clock after DROT's workspace allocation
and init kernels and does not time them. t_total adds the shim's plan
recovery and copies, its file writes, the runner's plan read and cast, and
the dual read and division by σ; each part is recorded separately.

``spec.device="cpu"`` is a test seam: the shim is CUDA-only and the tests fake it.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.protocol.paths import REPO_ROOT
from src.protocol.instance_record import (InstanceSpec, ResultRecord, WALL_BUFFER_S,
                                         utc_now_iso)
from src.scoring.plan_metrics import DenseTorchPlan, evaluate_plan_metrics
from src.instances.dense_cost import (build_dense_cost, lattice_cost_rows,
                                      materialized_cost_path)
from src.scoring.pareto_score import score as _pareto_score, sanitize_block as _sanitize
from src.runners.gpu_state import device_fingerprint
from src.protocol.regimes import GENERAL, GRID
from src.runners.common import (
    cuda_sync,
    failure_record,
    load_marginal,
    mentions_oom,
    runner_main,
    scratch_dir,
)


# BDRS_DROT_SHIM, when set, names another build.
DEFAULT_SHIM_PATH = REPO_ROOT / "vendor" / "drot" / "shim" / "bdrs_shim"

# n_g >= 256 is refused before the solve. At n_g = 256 the fp32 cost and plan
# are 17 GB each, and the upstream kernels hit an illegal memory access there;
# their 32-bit int offsets i + j·N into C and X overflow at N = 65536.
SKIP_RESOLUTION_ABOVE = 128
# The same limit on N, for instances without a lattice.
SKIP_MARGINAL_LEN_ABOVE = SKIP_RESOLUTION_ABOVE * SKIP_RESOLUTION_ABOVE

# The shim polls DROT's residual every DROT_CHECK_FREQ iterations (upstream's
# drot() polls every iteration). Under --pareto the runner does not pass
# --tol, so the native stop is the shim's default tol = 1e-4, the tolerance of
# upstream's examples/multiexp.cu.
DROT_CHECK_FREQ = 100
DROT_NATIVE_STOP_LABEL = f"shim_default_tol_1e-4_checkfreq_{DROT_CHECK_FREQ}"

T_SOLVE_BOUNDARY = "shim_solve_seconds"
# Work inside solve_seconds that t_solve should exclude: none. The residual
# polls are DROT's own termination checks, which t_solve includes.
T_SOLVE_INCLUDES: Tuple[str, ...] = ()
# Work outside solve_seconds. The last two are solver setup, which belongs in
# t_solve, but the shim starts its clock after them and does not time them.
T_SOLVE_EXCLUDES = ("process_spawn", "cost_blob_read_and_h2d",
                    "drot_device_workspace_alloc", "drot_init_kernels")


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str) -> ResultRecord:
    timeout = status == "timeout"
    return failure_record(spec, "DROT", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, m=spec.dense_shape()[0],
                          algo_extras={"drot_dtype": "fp32"}, timeout_flag=timeout,
                          wall_time_s=float(spec.time_limit_s) if timeout else None)


def _load_marginals(spec: InstanceSpec) -> Tuple[np.ndarray, np.ndarray]:
    """The original marginals (r, c), fp64, neither floored nor renormalised."""
    return load_marginal(spec.marginal_i_path), load_marginal(spec.marginal_j_path)


def _load_marginals_fp32(spec: InstanceSpec) -> Tuple[np.ndarray, np.ndarray]:
    r, c = _load_marginals(spec)
    return r.astype(np.float32, copy=False), c.astype(np.float32, copy=False)


def _write_cost_fp32_blob(spec: InstanceSpec, out_path: Path) -> None:
    """Write C as fp32 to ``out_path`` in column-major order (Cᵀ row-major), in
    blocks, so C is never held whole.

    The cost is ``spec.cost_matrix_path`` when that file exists (fp64, max 1,
    possibly asymmetric), else the lattice cost of ``spec.resolution``.
    """
    nn = spec.dense_shape()[0]

    cached = spec.cost_matrix_path
    if cached and os.path.exists(cached):
        # Each block is Cᵀ[j0:j1, :] = (C[:, j0:j1])ᵀ, a strided read of the
        # row-major memmap.
        C64 = np.load(cached, mmap_mode="r")
        if C64.shape != (nn, nn):
            raise ValueError(f"Cached cost shape {C64.shape} != ({nn},{nn})")
        with open(out_path, "wb") as f:
            chunk = max(1, min(nn, 1024 * 1024 * 1024 // (nn * 4)))   # ~1 GB per chunk
            for j0 in range(0, nn, chunk):
                j1 = min(nn, j0 + chunk)
                # Casts the strided view straight to fp32, with no fp64 copy.
                f.write(np.ascontiguousarray(
                    C64[:, j0:j1].T, dtype=np.float32).tobytes())
        return

    # The lattice costs are symmetric, so the rows of C are the rows of Cᵀ.
    # Every other dense instance has a cost file (a procedural cost is written
    # to one by materialized_cost_path).
    if spec.resolution is None:
        raise ValueError(
            f"{spec.instance_id}: no cost file and no grid resolution to build "
            "the dense cost from")
    chunk = max(1, min(nn, (32 * 1024 * 1024) // nn))                # ~256 MB fp64 per block
    with open(out_path, "wb") as f:
        for j0 in range(0, nn, chunk):
            j1 = min(nn, j0 + chunk)
            rows = lattice_cost_rows(spec.resolution, spec.norm, j0, j1, device="cpu")
            f.write(np.ascontiguousarray(rows.numpy(), dtype=np.float32).tobytes())


def _opt_float(info: Dict[str, Any], key: str) -> Optional[float]:
    """``info[key]`` as a float, or ``None`` when absent or not finite (a record
    never holds NaN or inf)."""
    v = info.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pareto_clocks(info: Dict[str, Any], *, t_plan_read: float,
                   t_dual_recovery: float, t_process_wall: float) -> Dict[str, Any]:
    """The Pareto clock block, from the shim's info.json.

    ``t_solve`` is ``solve_seconds``, with no fallback (RuntimeError when
    absent). ``t_total`` adds ``recovery_seconds`` and ``write_seconds`` (both
    None and left out, with ``shim_recovery_timed`` False, unless the shim
    wrote both), ``t_plan_read`` and ``t_dual_recovery``.
    """
    if "solve_seconds" not in info:
        raise RuntimeError(
            "DROT pareto: shim info.json has no 'solve_seconds'; t_solve must be "
            "the shim's own solve clock, so the subprocess wall time is not used.")
    t_solve = float(info["solve_seconds"])
    timed = ("recovery_seconds" in info) and ("write_seconds" in info)
    t_rec = float(info["recovery_seconds"]) if timed else None
    t_wr = float(info["write_seconds"]) if timed else None
    t_total = t_solve + (t_rec or 0.0) + (t_wr or 0.0) + float(t_plan_read) + float(t_dual_recovery)
    return {
        "t_solve": t_solve,
        "t_total": float(t_total),
        "t_shim_recovery": t_rec,
        "t_shim_write": t_wr,
        "t_plan_read": float(t_plan_read),
        "t_dual_recovery": float(t_dual_recovery),
        "t_process_wall": float(t_process_wall),
        "shim_recovery_timed": bool(timed),
        "t_solve_boundary": T_SOLVE_BOUNDARY,
        "t_solve_includes": list(T_SOLVE_INCLUDES),
        "t_solve_excludes": list(T_SOLVE_EXCLUDES),
    }


def _run_one(spec: InstanceSpec) -> ResultRecord:
    # The size gate comes before materialized_cost_path, which would write the
    # refused cost to disk (32 GiB of fp64 at N = 65536).
    n_total = spec.dense_shape()[0]
    if n_total > SKIP_MARGINAL_LEN_ABOVE:
        return _make_failure_record(
            spec, status="oom", error_kind="cpu_skipped_resolution_too_big",
            error_msg=(
                f"DROT at N = {n_total} skipped preemptively: the fp32 cost "
                f"matrix would be {n_total ** 2 * 4 / 1e9:.0f} GB, and at "
                f"N = 65536 the upstream kernels hit an illegal memory access; "
                f"their 32-bit int offsets i + j·N into C and X overflow there."
            ),
            started_utc=utc_now_iso(), run_uid=str(uuid.uuid4()),
        )
    # The shim reads C from a file, so a procedural cost is written to one.
    with materialized_cost_path(spec) as spec_with_cost:
        return _run_one_inner(spec_with_cost)


def _run_one_inner(spec: InstanceSpec) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    n_total = spec.dense_shape()[0]
    pareto = bool(getattr(spec, "pareto", False))

    if spec.device.startswith("cuda") and not torch.cuda.is_available():
        return _make_failure_record(
            spec, status="error", error_kind="no_cuda",
            error_msg="torch.cuda.is_available() is False; DROT shim requires CUDA.",
            started_utc=started_utc, run_uid=run_uid,
        )
    device = torch.device(spec.device if spec.device.startswith("cuda") else "cpu")

    shim_path = Path(os.environ.get("BDRS_DROT_SHIM", str(DEFAULT_SHIM_PATH)))
    if not shim_path.is_file() or not os.access(shim_path, os.X_OK):
        return _make_failure_record(
            spec, status="error", error_kind="shim_not_built",
            error_msg=(
                f"DROT shim not found at {shim_path}. Build it via "
                f"`make -C vendor/drot/shim`."
            ),
            started_utc=started_utc, run_uid=run_uid,
        )

    scratch_root = scratch_dir()
    scratch = Path(tempfile.mkdtemp(prefix=f"bdrs_drot_{spec.instance_id}_", dir=scratch_root))
    # The basenames differ in more than letter case: on a case-insensitive
    # filesystem (macOS APFS) "C.bin" and "c.bin" are one file, and the cost
    # would overwrite the target marginal.
    cost_blob = scratch / "cost.bin"
    r_blob    = scratch / "marg_r.bin"
    c_blob    = scratch / "marg_c.bin"
    plan_blob = scratch / "plan.bin"
    info_path = scratch / "info.json"
    alpha_blob = scratch / "alpha.bin"   # φ₁ (--pareto only)
    beta_blob  = scratch / "beta.bin"    # φ₂ (--pareto only)

    try:
        try:
            r_np, c_np = _load_marginals_fp32(spec)
            r_np.tofile(r_blob)
            c_np.tofile(c_blob)
        except (MemoryError, OSError) as e:
            return _make_failure_record(
                spec, status="oom", error_kind="cpu_oom_marginals",
                error_msg=repr(e),
                started_utc=started_utc, run_uid=run_uid,
            )

        try:
            _write_cost_fp32_blob(spec, cost_blob)
        except (MemoryError, OSError) as e:
            return _make_failure_record(
                spec, status="oom", error_kind="cpu_oom_cost_matrix",
                error_msg=repr(e),
                started_utc=started_utc, run_uid=run_uid,
            )

        # No warm-up: upstream DROT ships none. --n is N (n_g² on a grid), not
        # n_g; the shim takes one size, so m = n.
        cmd = [
            str(shim_path),
            "--cost", str(cost_blob),
            "--r", str(r_blob),
            "--c", str(c_blob),
            "--n", str(n_total),
            "--max-iters", str(spec.iter_cap),
            "--check-freq", str(DROT_CHECK_FREQ),
            "--plan-out", str(plan_blob),
            "--info-out", str(info_path),
        ]
        if pareto:
            # No --tol: the native stop is the shim's default.
            cmd += ["--alpha-out", str(alpha_blob), "--beta-out", str(beta_blob)]
        else:
            cmd += ["--tol", str(spec.tolerance)]

        cuda_sync(device)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                # The orchestrator kills the runner at time_limit_s +
                # WALL_BUFFER_S, which comes first; this deadline serves a
                # direct invocation, where it is the only one.
                timeout=spec.time_limit_s + WALL_BUFFER_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _make_failure_record(
                spec, status="timeout", error_kind="shim_timeout",
                error_msg=f"bdrs_shim exceeded {spec.time_limit_s + WALL_BUFFER_S}s",
                started_utc=started_utc, run_uid=run_uid,
            )
        t1 = time.perf_counter()
        wall_process = t1 - t0

        if proc.returncode == 2 or mentions_oom(proc.stderr):
            return _make_failure_record(
                spec, status="oom", error_kind="cuda_oom_shim",
                error_msg=f"shim exit={proc.returncode}\nstderr:\n{proc.stderr[-4000:]}",
                started_utc=started_utc, run_uid=run_uid,
            )
        if proc.returncode != 0:
            return _make_failure_record(
                spec, status="error", error_kind=f"shim_exit_{proc.returncode}",
                error_msg=f"shim stderr:\n{proc.stderr[-4000:]}",
                started_utc=started_utc, run_uid=run_uid,
            )
        if not info_path.exists() or not plan_blob.exists():
            return _make_failure_record(
                spec, status="error", error_kind="shim_missing_output",
                error_msg=f"info or plan missing under {scratch}",
                started_utc=started_utc, run_uid=run_uid,
            )

        info = json.loads(info_path.read_text())
        shim_solve_seconds = _opt_float(info, "solve_seconds")

        # The plan file is column-major (X[i, j] at offset i + j·N), hence
        # order='F'. The read and the fp64 cast are timed as t_plan_read.
        _pp0 = time.perf_counter()
        X_fp32 = torch.from_numpy(
            np.ascontiguousarray(
                np.fromfile(plan_blob, dtype=np.float32, count=n_total * n_total)
                .reshape(n_total, n_total, order="F")
            )
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cuda_sync(device)
        t2 = time.perf_counter()
        try:
            X = X_fp32.to(device=device, dtype=torch.float64)
        except torch.cuda.OutOfMemoryError as e:
            return _make_failure_record(
                spec, status="oom", error_kind="cuda_oom_post_cast",
                error_msg=repr(e),
                started_utc=started_utc, run_uid=run_uid,
            )
        cuda_sync(device)
        t_plan_read = time.perf_counter() - _pp0
        del X_fp32

        r64, c64 = _load_marginals(spec)
        r_dev = torch.from_numpy(r64).to(device=device, dtype=torch.float64)
        c_dev = torch.from_numpy(c64).to(device=device, dtype=torch.float64)

        # Off the Grid rung the cost is the fp64 file whose fp32 rounding the
        # shim solved; a cost rebuilt from (resolution, norm) would be a
        # different one. A Grid instance rebuilds its cost in row blocks.
        _Cm = None
        if spec.rung != GRID and spec.cost_matrix_path and Path(spec.cost_matrix_path).exists():
            _Cm = torch.as_tensor(np.load(spec.cost_matrix_path),
                                  dtype=torch.float64, device=device)
        rep = DenseTorchPlan(X=X, ng=spec.resolution, cost_type=spec.norm, C=_Cm)
        metrics = evaluate_plan_metrics(rep, r_dev, c_dev)
        cuda_sync(device)
        t3 = time.perf_counter()
        wall_eval = t3 - t2

        # The peak of this process's allocations since the reset before the
        # fp64 cast (the shim's memory is another process's), read before the
        # Pareto scoring builds a dense C (l2, inf).
        if device.type == "cuda":
            peak_mem: Optional[int] = int(torch.cuda.max_memory_allocated(device))
            peak_mem_source: Optional[str] = "torch_cuda_max"
        else:
            peak_mem, peak_mem_source = None, None

        # The dual is (α, β) = (φ₁/σ, φ₂/σ): DROT's projection multipliers
        # divided by its step size σ = 2/(m + n). X ≥ 0 is not assumed for the
        # fp32 plan, so ‖min(X, 0)‖ is measured.
        pareto_block = None
        clocks: Optional[Dict[str, Any]] = None
        wall_dual_recovery = 0.0
        if pareto:
            cuda_sync(device)
            _tr0 = time.perf_counter()
            if "stepsize" not in info:
                raise RuntimeError(
                    "DROT pareto: shim info.json has no 'stepsize', so α = φ₁/σ "
                    "cannot be formed; a guessed σ would corrupt the dual.")
            sigma = float(info["stepsize"])
            phi1 = torch.from_numpy(np.fromfile(alpha_blob, dtype=np.float32,
                                                count=n_total)).to(device=device, dtype=torch.float64)
            phi2 = torch.from_numpy(np.fromfile(beta_blob, dtype=np.float32,
                                                count=n_total)).to(device=device, dtype=torch.float64)
            alpha_d = phi1 / sigma
            beta_d = phi2 / sigma
            cuda_sync(device)
            wall_dual_recovery = time.perf_counter() - _tr0
            # Scored off the clocks against the cost the shim solved, in fp64:
            # the stored C off the Grid rung, else the Grid form (l1, sql2),
            # else a dense lattice C built here (l2, inf).
            _te0 = time.perf_counter()
            if _Cm is None and spec.norm in ("l1", "sql2"):
                _blk = _pareto_score(X, alpha_d, beta_d, r_dev, c_dev,
                                     ng=spec.resolution, cost_type=spec.norm,
                                     x_nonneg_structural=False, compute_frobenius=True)
            else:
                _Csc = _Cm if _Cm is not None else build_dense_cost(
                    spec.resolution, spec.norm, device=device, dtype=torch.float64)
                _blk = _pareto_score(X, alpha_d, beta_d, r_dev, c_dev, C=_Csc,
                                     x_nonneg_structural=False, compute_frobenius=True)
            cuda_sync(device)
            clocks = _pareto_clocks(info, t_plan_read=t_plan_read,
                                    t_dual_recovery=wall_dual_recovery,
                                    t_process_wall=wall_process)
            if not clocks["shim_recovery_timed"]:
                _msg = ("DROT pareto: shim info.json lacks 'recovery_seconds'/"
                        "'write_seconds' (a stale shim build), so t_total omits the "
                        "shim's plan recovery, device-to-host copies and file "
                        "writes; rebuild with `make -C vendor/drot/shim`.")
                if os.environ.get("BDRS_PARETO_STRICT"):
                    raise AssertionError(_msg)
                print("run_drot: WARNING " + _msg, file=sys.stderr)
            pareto_block = _sanitize({
                **_blk,
                **clocks,
                "t_eval": float(time.perf_counter() - _te0),
                "pareto_partial": False,
                "precision": "fp32",
                "native_stop": DROT_NATIVE_STOP_LABEL,
                "drot_sigma": sigma,
                "warmup": "none_upstream_ships_none",
            })

        fp = device_fingerprint(idx=device.index or 0) if device.type == "cuda" else {}

        iters_run = int(info.get("iters", 0))
        iter_cap_hit = iters_run >= spec.iter_cap
        term_reason = info.get("term_reason", "max_iters")
        # DROT's stopping residual sqrt(‖X1 − r‖² + ‖Xᵀ1 − c‖²), in fp32
        # against the fp32 marginals, is primal, not a duality gap: the
        # info.json keys primal_res_sq and dual_res_sq hold its row and
        # column terms (loginfo[0], loginfo[1] of update_auxs_*).
        native_residual = _opt_float(info, "residual")
        row_res_sq = _opt_float(info, "primal_res_sq")
        col_res_sq = _opt_float(info, "dual_res_sq")
        shim_recovery_seconds = _opt_float(info, "recovery_seconds")
        shim_write_seconds = _opt_float(info, "write_seconds")
        wall_recovery = (float(shim_recovery_seconds or 0.0) + float(shim_write_seconds or 0.0)
                         + t_plan_read + wall_dual_recovery)

        rec = ResultRecord(
            instance_id=spec.instance_id, algorithm="DROT", regime=GENERAL,
            norm=spec.norm, resolution=spec.resolution, n=n_total, m=n_total,
            run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
            status="ok",
            wall_time_s=wall_process + wall_eval,
            wall_time_solve_s=shim_solve_seconds,
            wall_time_recovery_s=wall_recovery,
            iters=iters_run,
            iter_cap_hit=iter_cap_hit,
            native_termination=term_reason,
            tolerance=spec.tolerance,
            feaserr=native_residual,
            native_duality_gap=None,
            objective=float(metrics["objective"]),
            peak_gpu_mem_bytes=peak_mem,
            peak_gpu_mem_source=peak_mem_source,
            device_name=fp.get("device_name"),
            device_uuid=fp.get("device_uuid"),
            algo_extras={
                "drot_dtype": "fp32",
                "certify_unfloored": True,
                "stepsize": _opt_float(info, "stepsize"),
                "feaserr_measure": "drot_residual",
                "drot_native_residual": native_residual,
                "drot_native_residual_kind": "primal_marginal_l2_combined_fp32",
                "drot_row_res_sq": row_res_sq,
                "drot_col_res_sq": col_res_sq,
                "native_duality_gap_reason": "drot_exposes_no_native_lp_gap",
                "shim_solve_seconds": shim_solve_seconds,
                "shim_recovery_seconds": shim_recovery_seconds,
                "shim_write_seconds": shim_write_seconds,
                "wall_time_process_s": wall_process,
                "wall_time_plan_read_s": t_plan_read,
                "row_sum_err_l2": float(metrics["row_sum_err_l2"]),
                "col_sum_err_l2": float(metrics["col_sum_err_l2"]),
                "wall_time_eval_s": wall_eval,
                "r_P": float(metrics["feaserr"]),
                "r_P_negpart_assumed_zero": not metrics["negpart_measured"],
                "warmup": "none",
                # True when the last step was odd-indexed: DROT then stores
                # the splitting variable X − σC, and the shim added σC back.
                "shim_recovered_from_splitting": bool(info.get("recovered_from_splitting", False)),
                "shim_plan_layout": str(info.get("plan_layout", "column_major")),
                "cost_blob_layout": "column_major",
                **({"pareto": pareto_block} if pareto_block is not None else {}),
            },
        )
        return rec

    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None):
    return runner_main(
        argv, description="DROT runner.",
        run=lambda spec, args: _run_one(spec),
        failure=lambda spec, args, **kw: _make_failure_record(spec, **kw))


if __name__ == "__main__":
    sys.exit(main())
