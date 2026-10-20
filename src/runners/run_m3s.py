"""
Runner for M3S, the multiscale sparse ε-scaling Sinkhorn solver of Chen, Li and
Lu (CVPR 2022; upstream ucascnic/CudaOT, fetch-pinned at vendor/m3s/PIN).
Upstream's API returns the distance only, so the runner calls the first-party
shim vendor/m3s/_build/m3s_shim as a subprocess: it runs the same pipeline and
also writes the plan (COO) and the finest-layer potentials.

M3S solves the entropic problem at a final ε of about 0.1 in its native pixel²
cost units, with every solver parameter hard-wired upstream; its stop is an L1
row-marginal error of at most 1e-4, and ``spec.tolerance`` and ``spec.iter_cap``
are not passed to it. Its GPU solver handles square image grids with n_g in
{32, 64, 128, 256, 512} and the sql2 cost only, so the orchestrator skips it on
point clouds.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.protocol.paths import REPO_ROOT
from src.protocol.instance_record import (
    WALL_BUFFER_S, InstanceSpec, ResultRecord, utc_now_iso,
)
from src.scoring.plan_metrics import SparseTorchPlan, evaluate_plan_metrics
from src.runners.gpu_state import device_fingerprint, visible_selector
from src.runners.nvml_sampler import NvmlPeakSampler
from src.methods.params import MARGINAL_EPS as TORCH_MARGINAL_EPS, floor_marginal
from src.runners.common import (
    cuda_ordinal,
    failure_record,
    load_marginal,
    mentions_oom,
    runner_main,
    scratch_dir,
)


DEFAULT_SHIM_PATH = REPO_ROOT / "vendor" / "m3s" / "_build" / "m3s_shim"
# The line upstream's getError prints at every convergence check.
_M3S_ERROR_LINE = re.compile(r"^\terror = (\S+)$", re.MULTILINE)

# Upstream rejects nonpositive mass (msg 10001, ERR_PREP_INIT_MUXNEG), so the
# shim solves on marginals floored at δ_m, the floor of our solvers; every
# metric except feaserr is scored against the originals. The floor moves a
# unit-mass marginal by at most 2N·δ_m in ℓ1.
MARGINAL_FLOOR = TORCH_MARGINAL_EPS

# t_solve is the shim's shim_solve_seconds: the upstream pipeline (multiscale
# setup and the whole ε ladder) between two device synchronisations. It
# includes the host-to-device copy of the marginals, which the shim cannot
# separate, and excludes the spawn, the blob I/O and the shim's warm-up solve.
T_SOLVE_BOUNDARY = "m3s_shim_solve_seconds"
T_SOLVE_INCLUDES = ("marginal_h2d",)
T_SOLVE_EXCLUDES = ("marginals_load", "marginal_blob_write",
                    "subprocess_spawn", "shim_blob_read",
                    "shim_warmup_solve", "shim_output_blob_write")
# t_total adds the shim's extraction window (the plan and dual copies, and its
# recomputation of the native distance, a diagnostic), the runner's read of the
# plan blobs, and the read, fold and rescale of the duals.
T_TOTAL_BOUNDARY = ("m3s_shim_solve_plus_shim_extract_plus_blob_read"
                    "_plus_dual_fold")
T_TOTAL_INCLUDES = ("shim_native_distance_recompute",)

# The literals of upstream's cuda_image_ot_solver (SparseSinkhorn/solver.cu),
# which the shim runs unchanged.
M3S_HARDWIRED_CONFIG = {
    "eps_start": 1e8,
    "eps_target": 1e-1,
    "eps_steps": 20,
    "layer_coarsest": 3,
    "max_error": 1e-4,
    "max_iterations": 100000,
    "inner_iterations": 1000,
    "max_absorption_loops": 20,
    "absorption_scaling_bound": 1e3,
    "absorption_scaling_lower_bound": 1e3,
    "truncation_thresh": 1e-1,
    "refine_kernel": 1,
    "p_index": 2,
}

def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str,
                         wall_time_s: float | None = None,
                         extras: Optional[dict] = None) -> ResultRecord:
    return failure_record(spec, "M3S", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, algo_extras=extras,
                          timeout_flag=(status == "timeout"), wall_time_s=wall_time_s)


def _load_marginals(spec: InstanceSpec) -> tuple[np.ndarray, np.ndarray]:
    """The stored marginals, unfloored; every metric but ``feaserr`` is scored
    against them."""
    return load_marginal(spec.marginal_i_path), load_marginal(spec.marginal_j_path)


def _floored_for_solver(r: np.ndarray, c: np.ndarray, delta: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """The copy the shim solves on, r̃ = floor_marginal(r, δ), the same operation
    as our solvers' floor. Only ``feaserr``, M3S's own error, is measured against
    it."""
    rf = floor_marginal(torch.from_numpy(r), delta).numpy()
    cf = floor_marginal(torch.from_numpy(c), delta).numpy()
    return np.ascontiguousarray(rf, dtype=np.float64), np.ascontiguousarray(cf, dtype=np.float64)


def _fold_reference_measure(alpha: np.ndarray, beta: np.ndarray,
                            r: np.ndarray, c: np.ndarray, eps: float
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Fold the reference measure into M3S's potentials, in native pixel² units.

    M3S's kernel carries the reference measure,
    K_ij = exp((α_i + β_j − c_ij)/ε)·r̃_i·c̃_j, and its scalings are 1 at
    delivery, so π = K and, on the support of π,
    ε·log π_ij = (α_i + ε·log r̃_i) + (β_j + ε·log c̃_j) − c_ij: the folded
    potentials satisfy the entropic optimality condition. The fold uses the
    original r, c, which shifts α_i by ε·log(r_i/r̃_i) (negligible unless r_i
    is near δ_m), β_j likewise, and gives −∞ where the mass is zero. That is
    the exact potential of a point without mass, and harmless: ReLU(−∞) = 0 in
    r_D, and the dual objective takes 0·(−∞) = 0 (``kkt_residual.support_dot``).
    The folded pair has g ≈ ε·H(π) > 0 with H(π) = −Σ π log π, the entropic
    floor, up to the marginal error.

    ``eps`` is the final ε of the ladder; the caller rescales by
    1/(2(n_g − 1)²) afterwards.
    """
    eps = float(eps)
    log_r = np.log(np.where(r > 0.0, r, 1.0))
    log_c = np.log(np.where(c > 0.0, c, 1.0))
    # -inf only at zero mass; never 0*(-inf) -> NaN even if eps were 0.
    a_out = np.where(r > 0.0, alpha + eps * log_r, -np.inf)
    b_out = np.where(c > 0.0, beta + eps * log_c, -np.inf)
    return a_out, b_out


def _native_termination(msg: int) -> str:
    if msg == 0:
        # Returned only when every (layer, ε) stage drove the L1 row error
        # below maxError.
        return "max_error_reached"
    if msg == 30101:
        return "max_iterations"
    if msg == 30201:
        return "too_many_absorptions"
    if msg == 30102:
        return "nan_scaling"
    if msg == -2:
        return "nan_error"
    return f"m3s_msg_{msg}"


def _run_one(spec: InstanceSpec) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    ng = spec.resolution
    n_total = ng * ng

    if spec.norm != "sql2":
        # p_index = 2 is hard-wired upstream; p = 1 would be the Euclidean
        # distance, not l1.
        return _make_failure_record(
            spec, status="error", error_kind=f"m3s_{spec.norm}_unsupported",
            error_msg=(
                "M3S's cost kernel hard-codes squared-Euclidean grid cost "
                "(p_index=2); the protocol restricts M3S to sql2 "
                "(src/protocol/regimes.py)."),
            started_utc=started_utc, run_uid=run_uid,
        )
    if ng == 1024:
        # Upstream's fixed MAXSIZE = 512 buffers overflow at n_g = 1024
        # (memory corruption, not an allocation failure), so the row is a
        # preemptive oom.
        return _make_failure_record(
            spec, status="oom", error_kind="m3s_1024_structurally_unsupported",
            error_msg=(
                "M3S as-published cannot run n_g=1024: XData.data[512²·3] and "
                "crs_row_ptrl[512²+1] (MAXSIZE=512, Common/basic_settings.h) "
                "are overflowed by the 1024² atomic hierarchy layer (host heap "
                "and device corruption). See vendor/m3s/README.md."),
            started_utc=started_utc, run_uid=run_uid,
        )

    shim_path = Path(os.environ.get("BDRS_M3S_SHIM", str(DEFAULT_SHIM_PATH)))
    if not shim_path.is_file() or not os.access(shim_path, os.X_OK):
        return _make_failure_record(
            spec, status="error", error_kind="shim_not_built",
            error_msg=(
                f"M3S shim not found at {shim_path}. Build it via "
                f"`bash vendor/m3s/fetch.sh && bash vendor/m3s/build.sh`."),
            started_utc=started_utc, run_uid=run_uid,
        )

    r_np, c_np = _load_marginals(spec)
    # Attached to every row from here on, failure rows included.
    marg_extras = {
        "marginal_floor": MARGINAL_FLOOR,
        "n_zero_mass_r": int((r_np == 0.0).sum()),
        "n_zero_mass_c": int((c_np == 0.0).sum()),
    }

    scratch = Path(tempfile.mkdtemp(
        prefix=f"bdrs_m3s_{spec.instance_id}_", dir=scratch_dir()))
    try:
        r_blob = scratch / "r.bin"
        c_blob = scratch / "c.bin"
        r_solver, c_solver = _floored_for_solver(r_np, c_np, MARGINAL_FLOOR)
        r_solver.tofile(r_blob)
        c_solver.tofile(c_blob)

        cmd = [str(shim_path), str(ng), str(r_blob), str(c_blob), str(scratch)]
        if spec.warmup > 0:
            cmd.append("--warmup")

        # The shim allocates with cudaMalloc, outside torch's allocator, so
        # device memory is polled through NVML; the poll also spans the shim's
        # warm-up solve, which runs at the same n_g.
        device_index = cuda_ordinal(spec.device)
        sampler = NvmlPeakSampler(device_index)
        sampler.start()

        # The shim runs the warm-up solve and the timed solve in one process.
        budget = spec.time_limit_s * (2 if spec.warmup > 0 else 1) + WALL_BUFFER_S
        # The shim always uses CUDA device 0, so it sees only the spec's device,
        # mapped through this process's CUDA_VISIBLE_DEVICES as the sampler and
        # the fingerprint map it.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible_selector(device_index))
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=budget, check=False, env=env,
            )
        except subprocess.TimeoutExpired:
            sampler.stop()
            return _make_failure_record(
                spec, status="timeout", error_kind="shim_timeout",
                error_msg=f"m3s_shim exceeded {budget}s (incl. untimed warmup)",
                started_utc=started_utc, run_uid=run_uid,
                wall_time_s=float(spec.time_limit_s), extras=marg_extras,
            )
        peak_mem, baseline_mem = sampler.stop()

        meta_path = scratch / "meta.json"
        meta = None
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = None

        if proc.returncode == 2 or mentions_oom(proc.stderr or ""):
            return _make_failure_record(
                spec, status="oom", error_kind="cuda_oom_shim",
                error_msg=f"shim exit={proc.returncode}\nstderr:\n{proc.stderr[-4000:]}",
                started_utc=started_utc, run_uid=run_uid, extras=marg_extras,
            )
        if proc.returncode == 3:
            msg = int(meta["msg"]) if meta else -1
            return _make_failure_record(
                spec, status="error", error_kind=f"m3s_msg_{msg}",
                error_msg=(f"M3S solver msg={msg} "
                           f"({_native_termination(msg)})\n"
                           f"stderr:\n{proc.stderr[-2000:]}"),
                started_utc=started_utc, run_uid=run_uid, extras=marg_extras,
            )
        if proc.returncode != 0:
            return _make_failure_record(
                spec, status="error", error_kind=f"shim_exit_{proc.returncode}",
                error_msg=f"shim stderr:\n{proc.stderr[-4000:]}",
                started_utc=started_utc, run_uid=run_uid, extras=marg_extras,
            )
        needed = ["plan_row.bin", "plan_col.bin", "plan_val.bin",
                  "alpha.bin", "beta.bin", "meta.json"]
        if meta is None or any(not (scratch / f).exists() for f in needed):
            return _make_failure_record(
                spec, status="error", error_kind="shim_missing_output",
                error_msg=f"expected {needed} under {scratch}",
                started_utc=started_utc, run_uid=run_uid, extras=marg_extras,
            )

        # Reading the plan is part of t_total; the metric evaluation below is
        # in neither clock.
        t_pr0 = time.perf_counter()
        row = np.fromfile(scratch / "plan_row.bin", dtype=np.int32)
        col = np.fromfile(scratch / "plan_col.bin", dtype=np.int32)
        val = np.fromfile(scratch / "plan_val.bin", dtype=np.float64)
        nnz = int(meta["nnz"])
        if not (row.size == col.size == val.size == nnz):
            return _make_failure_record(
                spec, status="error", error_kind="shim_blob_size_mismatch",
                error_msg=(f"nnz={nnz} but blobs hold "
                           f"({row.size}, {col.size}, {val.size}) entries"),
                started_utc=started_utc, run_uid=run_uid, extras=marg_extras,
            )
        indices = torch.from_numpy(
            np.ascontiguousarray(np.stack([row, col]).astype(np.int64)))
        rep = SparseTorchPlan(indices, torch.from_numpy(val),
                              ng=ng, cost_type="sql2")
        wall_plan_read = time.perf_counter() - t_pr0

        t0 = time.perf_counter()
        metrics = evaluate_plan_metrics(
            rep, torch.from_numpy(r_np), torch.from_numpy(c_np))
        # M3S's own error, the quantity its stop tests: ‖X1 − r̃‖₁ on the row
        # marginal only (its v-update makes the column sums exact), against the
        # floored r̃ it was given. M3S evaluates it before the final kernel
        # rebuild, which drops entries, so it is recomputed on the delivered plan.
        row_sums = rep.row_sums()
        feaserr = float((row_sums - torch.from_numpy(r_solver)).abs().sum())
        feaserr_vs_original = float((row_sums - torch.from_numpy(r_np)).abs().sum())
        wall_eval = time.perf_counter() - t0
        # The value M3S printed (eight decimals) at the last check of the timed
        # solve, the one that accepted.
        _errs = _M3S_ERROR_LINE.findall(proc.stdout or "")
        m3s_error_at_stop = float(_errs[-1]) if _errs else None

        objective = float(metrics["objective"])
        obj_native = float(meta["distance_native"])
        scale = 1.0 / (2.0 * (ng - 1) ** 2)
        # distance_native is in pixel² units, so native·scale must equal the
        # recomputed objective up to summation order. A mismatch means upstream's
        # position convention changed: warn, do not fail.
        crosscheck_rel = (abs(obj_native * scale - objective)
                          / max(abs(objective), 1e-300))
        if crosscheck_rel > 1e-9:
            print(
                f"run_m3s: WARNING objective cross-check off by "
                f"{crosscheck_rel:.3e} rel (native*scale="
                f"{obj_native * scale!r} vs recomputed={objective!r})",
                file=sys.stderr,
            )

        iters_per_layer = [int(x) for x in meta.get("iters_per_layer", [])]
        wall_solve = float(meta["shim_solve_seconds"])
        wall_extract = float(meta["shim_extract_seconds"])

        # The potentials are read before the finally clause deletes scratch,
        # folded (``_fold_reference_measure``) and rescaled like the objective.
        # The entropic duals are scored as they are, without a c-transform.
        pareto_block = None
        if getattr(spec, "pareto", False):
            from src.scoring.pareto_score import (score as _pareto_score,
                                                            sanitize_block as _sanitize)
            _tr0 = time.perf_counter()
            a_raw = np.fromfile(scratch / "alpha.bin", dtype=np.float64)
            b_raw = np.fromfile(scratch / "beta.bin", dtype=np.float64)
            if a_raw.size != n_total or b_raw.size != n_total:
                raise RuntimeError(
                    f"M3S pareto: alpha/beta.bin length ({a_raw.size},{b_raw.size}) "
                    f"!= N={n_total}; the shim's dual export is malformed.")
            eps_native = float(meta["eps_final"])
            a_fold, b_fold = _fold_reference_measure(
                a_raw, b_raw, r_np, c_np, eps_native)
            alpha_d = torch.from_numpy(a_fold).to(torch.float64) * scale
            beta_d = torch.from_numpy(b_fold).to(torch.float64) * scale
            wall_recovery = time.perf_counter() - _tr0
            _te0 = time.perf_counter()
            _blk = _pareto_score(
                rep, alpha_d, beta_d,
                torch.from_numpy(r_np).to(torch.float64),
                torch.from_numpy(c_np).to(torch.float64),
                ng=ng, cost_type="sql2",
                x_nonneg_structural=False, compute_frobenius=True)
            # Weak duality d_obj ≤ p_obj holds for a feasible dual. The slack
            # admits a small infeasibility but not a scale or sign error, such as
            # a dropped factor 2(n_g − 1)². A violation raises only under
            # BDRS_PARETO_STRICT, so one bad row does not stop a sweep. d_obj or
            # p_obj is None when sanitize_block removed a non-finite value; the
            # check is then skipped.
            _r_dinf = _blk.get("r_Dinf") or 0.0
            _d_obj, _p_obj = _blk.get("d_obj"), _blk.get("p_obj")
            _checkable = (isinstance(_d_obj, float) and isinstance(_p_obj, float)
                          and math.isfinite(_d_obj) and math.isfinite(_p_obj))
            if _checkable:
                _overshoot = _d_obj - _p_obj
                _wd_bound = _r_dinf + 0.05 * (1.0 + abs(_p_obj)) + 1e-6
                _wd_ok = _overshoot <= _wd_bound
                if not _wd_ok:
                    _msg = (f"M3S weak-duality violated: d_obj={_d_obj!r} − "
                            f"p_obj={_p_obj!r} = {_overshoot:.3e} > bound "
                            f"{_wd_bound:.3e} (r_Dinf={_r_dinf:.3e}); the dual's "
                            f"scale or sign is wrong")
                    if os.environ.get("BDRS_PARETO_STRICT"):
                        raise AssertionError(_msg)
                    print("run_m3s: WARNING " + _msg, file=sys.stderr)
            else:
                _wd_ok = None
                print("run_m3s: WARNING weak-duality check skipped: d_obj or p_obj "
                      f"is not finite ({_d_obj!r}, {_p_obj!r})",
                      file=sys.stderr)
            pareto_block = _sanitize({
                **_blk,
                "t_solve": float(wall_solve),
                "t_total": float(wall_solve + wall_extract + wall_plan_read
                                 + wall_recovery),
                "t_eval": float(time.perf_counter() - _te0),
                "t_solve_boundary": T_SOLVE_BOUNDARY,
                "t_solve_includes": list(T_SOLVE_INCLUDES),
                "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                "t_total_boundary": T_TOTAL_BOUNDARY,
                "t_total_includes": list(T_TOTAL_INCLUDES),
                "t_shim_extract": float(wall_extract),
                "t_plan_read": float(wall_plan_read),
                "t_dual_recovery": float(wall_recovery),
                "pareto_partial": False, "precision": "fp64",
                "weak_duality_ok": (None if _wd_ok is None else bool(_wd_ok)),
                "native_stop": "m3s_hardwired_maxError_1e-4_final_eps_0.1",
                "m3s_dual_scale": scale,
                "m3s_reference_measure_folded": True,
                "m3s_eps_final_native": eps_native,
                "m3s_eps_final_repo_units": eps_native * scale,
                "m3s_n_neg_inf_alpha": int(np.isneginf(a_fold).sum()),
                "m3s_n_neg_inf_beta": int(np.isneginf(b_fold).sum()),
            })

        fp = device_fingerprint(idx=device_index)

        return ResultRecord(
            instance_id=spec.instance_id, algorithm="M3S",
            norm=spec.norm, resolution=ng, n=n_total,
            run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
            status="ok",
            wall_time_s=wall_solve + wall_extract + wall_plan_read + wall_eval,
            wall_time_solve_s=wall_solve,
            wall_time_recovery_s=wall_extract + wall_plan_read,
            iters=sum(iters_per_layer),
            iter_cap_hit=False,           # spec.iter_cap is not passed to the shim
            native_termination=_native_termination(int(meta["msg"])),
            tolerance=spec.tolerance,     # recorded, not passed to the shim
            feaserr=feaserr,
            native_duality_gap=None,      # M3S reports no duality gap
            objective=objective,
            peak_gpu_mem_bytes=peak_mem,
            peak_gpu_mem_source=("nvml_poll" if peak_mem is not None else "unknown"),
            device_name=fp.get("device_name"),
            device_uuid=fp.get("device_uuid"),
            algo_extras={
                **marg_extras,
                "feaserr_measure": "m3s_row_l1_error",
                "feaserr_vs_original": feaserr_vs_original,
                "m3s_error_at_stop": m3s_error_at_stop,
                "r_P": float(metrics["feaserr"]),
                "m3s_dtype": "fp64",
                "m3s_integration": "first_party_extraction_shim",
                "m3s_hardwired_config": M3S_HARDWIRED_CONFIG,
                "m3s_class": "diagnostic_regularized_like_sinkhorn",
                "eps_final": float(meta["eps_final"]),
                "obj_m3s_native": obj_native,
                "m3s_obj_scale_factor": scale,
                **({"pareto": pareto_block} if pareto_block is not None else {}),
                "objective_crosscheck_rel": crosscheck_rel,
                "iters_per_layer": iters_per_layer,
                "nnz_final": nnz,
                "shim_solve_seconds": wall_solve,
                "shim_extract_seconds": wall_extract,
                "wall_time_plan_read_s": wall_plan_read,
                "wall_time_eval_s": wall_eval,
                "t_solve_boundary": T_SOLVE_BOUNDARY,
                "t_solve_includes": list(T_SOLVE_INCLUDES),
                "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                "t_total_boundary": T_TOTAL_BOUNDARY,
                "t_total_includes": list(T_TOTAL_INCLUDES),
                "row_sum_err_l2": float(metrics["row_sum_err_l2"]),
                "col_sum_err_l2": float(metrics["col_sum_err_l2"]),
                "nvml_baseline_bytes": baseline_mem,
                "nvml_poll_spans_warmup": bool(spec.warmup > 0),
                "r_P_negpart_assumed_zero": not metrics["negpart_measured"],
            },
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main(argv=None):
    return runner_main(
        argv, description="M3S runner (sql2 image grids only; extraction shim).",
        run=lambda spec, args: _run_one(spec),
        failure=lambda spec, args, **kw: _make_failure_record(spec, **kw))


if __name__ == "__main__":
    sys.exit(main())
