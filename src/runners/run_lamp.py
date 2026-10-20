"""LAMP runner: Log-Averaged Mirror Prox (Burns and Liang,
https://github.com/mxburns2022/CuLAMP.jl, pinned at ``vendor/lamp/PIN``).

Two algorithm names share this module. ``LAMP`` (General) runs LAMP's dense path
on the stored C, on the GPU, or on the CPU when the spec's device is "cpu".
``LAMPPointCloud`` (Point Cloud, ``run_lamppointcloud``) runs its matrix-free
path, which recomputes the cost from coordinates: l1, sql2 or inf, dimension at
most 3, m = n, CUDA only. The solve runs in a persistent Julia server
(``lamp_client``, ``vendor/lamp/lamp_server.jl``).

- Marginals: LAMP solves on upstream's preprocessing r̃ = (r + 10⁻⁶)/‖r + 10⁻⁶‖₁,
  and likewise c̃, which also keeps log r̃ finite. The Pareto block and r_P
  are scored against r and c, so r_P includes the shift r̃ − r; ``feaserr``,
  the crosscheck's ``col_l1_ours`` and ``lamp_d_obj_solved_marginals`` use r̃
  and c̃.
- Stop: LAMP's own test at ε = ``spec.tolerance``, with every other setting
  from upstream's ``configurations/default_0.json`` (η_p = η_μ = 0, 10⁶
  iterations, t_max = 600 s) and a check every 100 iterations. The dense GPU
  path stops when ‖Xᵀ1 − c̃‖₁ < ε/2 and its penalised gap is below ε/2, the CPU
  and matrix-free paths at ε/6. A run that ends at t_max is an ``ok`` row with
  ``native_termination`` "tmax", not a timeout.
- Plan: log X_ij = log r̃_i − log Z_i + s·C_ij + b_j with s = −st/(2Wη_t),
  b = −ν/η_t and Z the row normaliser, rebuilt from LAMP's O(m + n) state and
  never formed. W converts LAMP's cost units to ours: W = W_∞ on the dense
  paths; for p ≠ ∞ the matrix-free path divides the coordinates by
  W_∞-scaling^(1/p), so W = W_∞·W_∞-scaling there.
- Dual: β = −2Wθ, from the θ that LAMP's GPU stopping tests evaluate (the CPU
  test evaluates another iterate, γ), and α the c-transform of β, so α ⊕ β ≤ C.
  The pair built from ν instead of θ is recorded as ``lamp_nu_dual``.
- Clocks: t_solve is the upstream call between two device synchronisations in
  the server; t_total adds the state read, Z and the c-transform. Server
  start-up, warm-up included, precedes the solve's time budget and is on no
  clock.
- The matrix-free path never rescales for p = ∞, and its stopping test is then
  consistent only at W_∞ = 1. The coordinates are divided by C_max (√C_max for
  sql2), which gives W_∞ = 1 when ``spec.c_max`` is the exact maximum;
  otherwise the row is flagged ``lamp_kernel_W_eff_not_one``, and an inf row
  also ``lamp_native_gap_test_inconsistent``.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.instances.dense_cost import resolve_dense_cost
from src.methods import geo_cost
from src.methods.dual_cleanup import (
    _ctransform_storedC_rowmin,
    _ctransform_symbolic_rowmin,
    _dual_feas_F_storedC,
    _dual_feas_F_symbolic,
)
from src.methods.factored_torch import _pointcloud_block_chunk, _pointcloud_row_lse
from src.protocol.instance_record import InstanceSpec, ResultRecord, geo_cost_type, utc_now_iso
from src.protocol.regimes import GENERAL, POINT_CLOUD
from src.runners import lamp_client as server
from src.runners.common import cuda_ordinal, failure_record, runner_main, scratch_dir
from src.runners.gpu_state import device_fingerprint
from src.runners.nvml_sampler import NvmlPeakSampler
from src.scoring.pareto_score import sanitize_block, score, score_symbolic
from src.scoring.plan_metrics import SymbolicFactoredPlan

ALGO_GENERAL = "LAMP"
ALGO_PC = "LAMPPointCloud"
PRECISION = "fp64"
MARGINAL_FLOOR = 1e-6                      # upstream CmdLineUtils.jl run_dot: normalize(r .+ 1e-6, 1)
MARGINAL_FLOOR_SOURCE = "upstream CmdLineUtils.jl run_dot"
NATIVE_FREQUENCY = 100                     # the --frequency default of CmdLineUtils.jl
NATIVE_SETTINGS_FILE = "configurations/default_0.json"
EPS_TEST_FACTOR = {"gpu_dense": 0.5, "cpu_dense": 1.0 / 6.0, "kernel": 1.0 / 6.0}
KERNEL_P = {"l1": "1", "sql2": "2", "inf": "Inf"}
# The dense GPU path keeps C and its transpose resident, 2N² fp64 values (64 GiB
# at N = 65536), plus N² temporaries at each check; such a size is refused
# before any cost is built.
DENSE_SKIP_N_GE = 65536
T_SOLVE_BOUNDARY = "lamp_patched_solver_call_in_server"
T_SOLVE_INCLUDES = ("lamp_internal_setup", "native_checks_and_check_log_writes")
T_SOLVE_EXCLUDES = ("julia_server_startup", "startup_warmup", "gc_reclaim_reset",
                    "input_blob_write_read_and_h2d", "result_blob_write",
                    "color_transfer_outputs_not_computed", "scorer_cost_h2d")
T_TOTAL_BOUNDARY = "lamp_solve_plus_state_read_normaliser_and_theta_ctransform"
T_TOTAL_INCLUDES = ("state_blob_read", "logZ_recompute", "theta_dual_ctransform")
WALL_MARGIN_S = 45.0                       # left for scoring inside the orchestrator's wall
FEASERR_MEASURE = "lamp_col_l1_error"


def _regime(algo: str) -> str:
    return POINT_CLOUD if algo == ALGO_PC else GENERAL


def _clip(msg: str, limit: int = 4096) -> str:
    """Head and tail of ``msg``: a server failure ends with the log lines that matter."""
    msg = str(msg)
    return msg if len(msg) <= limit else msg[:1024] + "\n[...]\n" + msg[-(limit - 1032):]


def _failure(spec: InstanceSpec, algo: str, *, status: str, error_kind: str, error_msg: str,
             started_utc: str, run_uid: str, extras: Optional[Dict[str, Any]] = None) -> ResultRecord:
    return failure_record(
        spec, algo, status=status, error_kind=error_kind, error_msg=_clip(error_msg),
        started_utc=started_utc, run_uid=run_uid, m=spec.m,
        timeout_flag=(status == "timeout"),
        algo_extras={"marginal_floor": MARGINAL_FLOOR, **(extras or {})})


def _write_f64(path: Path, arr: np.ndarray) -> None:
    np.ascontiguousarray(arr, dtype=np.float64).tofile(path)


def native_settings() -> Dict[str, Any]:
    """Upstream's ``default_0.json``, whose sha256 the shim checks."""
    s = json.loads((server.upstream_dir() / NATIVE_SETTINGS_FILE).read_text())
    for k in ("eta_p", "eta_mu", "tau_p", "tau_mu", "alpha", "B", "epsilon", "tmax"):
        s[k] = float(s[k])
    s["itermax"] = int(s["itermax"])
    s["inner_iter"] = int(s.get("inner_iter", 10))
    s["verbose"] = bool(s["verbose"])
    return s


def _settings_for(spec: Optional[InstanceSpec]) -> Tuple[Dict[str, Any], bool]:
    """(settings, is_upstream_default): ``default_0.json`` with epsilon =
    ``spec.tolerance``. BDRS_LAMP_TEST_OVERRIDES (JSON) overrides settings for
    tests, and the row is then never the upstream default."""
    s = native_settings()
    upstream_eps = s["epsilon"]
    if spec is not None and spec.tolerance is not None:
        s["epsilon"] = float(spec.tolerance)
    default = s["epsilon"] == upstream_eps
    raw = os.environ.get("BDRS_LAMP_TEST_OVERRIDES")
    if not raw:
        return s, default
    s.update(json.loads(raw))
    return s, False


def thin_trace(rows: List[List[float]], k: int = 48) -> List[List[float]]:
    """At most k + 1 of LAMP's check rows, log-spaced in check index, the last
    included, each as [time, iteration, infeasibility, primal − dual]."""
    if not rows:
        return []
    n = len(rows)
    idx = sorted({min(n - 1, int(round(math.expm1(j * math.log1p(n - 1) / k))))
                  for j in range(k + 1)} | {n - 1})
    return [[float(rows[i][0]), float(rows[i][1]), float(rows[i][2]),
             float(rows[i][4] - rows[i][5])] for i in idx if len(rows[i]) >= 6]


def upstream_floor(v: np.ndarray) -> np.ndarray:
    """Julia's normalize(v .+ 1e-6, 1)."""
    w = np.asarray(v, dtype=np.float64) + MARGINAL_FLOOR
    return w / np.abs(w).sum()


def coord_prescale(norm: str, c_max: float) -> float:
    """The divisor of the coordinates that makes LAMP's distance C = κ/C_max:
    l1 and inf are homogeneous of degree 1, sql2 of degree 2."""
    return math.sqrt(c_max) if norm == "sql2" else c_max


def effective_scale(w_inf: float, w_inf_scaling: float) -> float:
    """W, with LAMP's cost equal to C/W."""
    return w_inf * w_inf_scaling


def plan_parameters(nu: np.ndarray, eta_t: float, st: float, w: float) -> Tuple[float, np.ndarray]:
    """(s, b) of log X = a + s·C + b. η_t = ∞ (no iteration completed) gives
    the plan with uniform rows, s = 0 and b = 0."""
    if not math.isfinite(eta_t):
        return 0.0, np.zeros_like(nu)
    return -st / (2.0 * w * eta_t), -nu / eta_t


def theta_dual(theta: torch.Tensor, w: float, rowmin) -> Tuple[torch.Tensor, torch.Tensor]:
    """(α, β) with β = −2Wθ and α = ``rowmin(β)``, the c-transform."""
    beta = -2.0 * w * theta
    return rowmin(beta), beta


def _check_rows(log_path: Path) -> list:
    """LAMP's check rows: time, iteration, infeasibility, objective, primal, dual."""
    rows = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 6 and parts[0][:1].isdigit():
            try:
                rows.append([float(x) for x in parts[:6]])
            except ValueError:
                continue
    return rows


def _gates(spec: InstanceSpec, algo: str) -> Optional[Tuple[str, str, str]]:
    """(status, error_kind, message) for an instance refused before any work, else None."""
    if algo == ALGO_PC:
        if spec.rung != POINT_CLOUD:
            return ("error", "lamppointcloud_needs_pointcloud_spec",
                    f"rung={spec.rung}; the orchestrator lifts grids to point clouds")
        if spec.norm not in KERNEL_P:
            return ("error", f"lamppointcloud_{spec.norm}_unsupported",
                    "LAMP's kernels compute sum|d|, sum d^2 or max|d|; no unsquared l2")
        if int(spec.dim or 0) > 3:
            return ("error", "lamppointcloud_dim_gt_3", f"dim={spec.dim}; kernels read 3 coordinates")
        if spec.m != spec.n:
            return ("error", "lamppointcloud_needs_square",
                    f"m={spec.m} n={spec.n}; the kernel path assumes m = n")
    else:
        m, n = spec.dense_shape()
        if m != n:
            return ("error", "lamp_needs_square", f"m={m} n={n}")
        if m >= DENSE_SKIP_N_GE and spec.device.startswith("cuda"):
            return ("oom", "lamp_device_oom_structural",
                    f"N={m}: C and its transpose alone need {2 * m * m * 8 / 2**30:.0f} GiB")
    return None


def _run_one(spec: InstanceSpec, algo: str) -> ResultRecord:
    started_utc, run_uid = utc_now_iso(), str(uuid.uuid4())
    t_wall0 = time.perf_counter()
    gate = _gates(spec, algo)
    if gate is not None:
        status, kind, msg = gate
        return _failure(spec, algo, status=status, error_kind=kind, error_msg=msg,
                        started_utc=started_utc, run_uid=run_uid)
    why = server.upstream_ready()
    if why is not None:
        return _failure(spec, algo, status="error", error_kind="upstream_not_fetched",
                        error_msg=why, started_utc=started_utc, run_uid=run_uid)
    cuda = spec.device.startswith("cuda")
    if algo == ALGO_PC and not cuda:
        return _failure(spec, algo, status="error", error_kind="no_cuda",
                        error_msg="LAMP's matrix-free path is CUDA-only",
                        started_utc=started_utc, run_uid=run_uid)
    if cuda and not torch.cuda.is_available():
        return _failure(spec, algo, status="error", error_kind="no_cuda",
                        error_msg="CUDA unavailable", started_utc=started_utc, run_uid=run_uid)
    mode = "kernel" if algo == ALGO_PC else ("gpu_dense" if cuda else "cpu_dense")
    settings, is_default = _settings_for(spec)
    if mode == "kernel" and not settings["verbose"]:
        raise AssertionError("verbose must be true: LAMP's kernel epsilon test is gated on it")

    r_np = np.load(spec.marginal_i_path).astype(np.float64).flatten()
    c_np = np.load(spec.marginal_j_path).astype(np.float64).flatten()
    r_solve = upstream_floor(r_np)
    c_solve = upstream_floor(c_np)
    m, n = r_np.size, c_np.size
    blobs = Path(tempfile.mkdtemp(prefix="lamp_", dir=scratch_dir()))
    extras: Dict[str, Any] = {}
    try:
        # Inputs, outside every clock.
        t_b0 = time.perf_counter()
        _write_f64(blobs / "row_marginal.f64", r_solve)
        _write_f64(blobs / "col_marginal.f64", c_solve)
        payload: Dict[str, Any] = {"mode": mode, "m": m, "n": n, "frequency": NATIVE_FREQUENCY,
                                   "settings": settings, "instance_id": spec.instance_id,
                                   "r_path": str(blobs / "row_marginal.f64"),
                                   "c_path": str(blobs / "col_marginal.f64"),
                                   "out_dir": str(blobs), "log_path": str(blobs / "checks.csv")}
        C_host = P = Q = None
        if mode == "kernel":
            P = np.load(spec.support_i_path).astype(np.float64)
            Q = np.load(spec.support_j_path).astype(np.float64)
            P = P.reshape(-1, 1) if P.ndim == 1 else P
            Q = Q.reshape(-1, 1) if Q.ndim == 1 else Q
            div = coord_prescale(spec.norm, float(spec.c_max))
            pad = lambda X: np.hstack([X / div, np.zeros((X.shape[0], 3 - X.shape[1]))])
            _write_f64(blobs / "points_row.f64", pad(P))        # (m, 3) row-major = 3 × m column-major
            _write_f64(blobs / "points_col.f64", pad(Q))
            payload.update({"P_path": str(blobs / "points_row.f64"),
                            "Q_path": str(blobs / "points_col.f64"), "p": KERNEL_P[spec.norm]})
            extras["lamp_coord_prescale"] = div
        else:
            C_host = resolve_dense_cost(spec, device="cpu", dtype=torch.float64)
            if tuple(C_host.shape) != (m, n):
                raise ValueError(f"cost shape {tuple(C_host.shape)} != marginals ({m}, {n})")
            _write_f64(blobs / "cost.f64", C_host.numpy().T)   # column-major, as Julia reads it
            payload["cost_path"] = str(blobs / "cost.f64")
        t_marshal = time.perf_counter() - t_b0

        # Server start-up, before the time budget and the NVML window.
        dev_idx = cuda_ordinal(spec.device) if cuda else 0
        t_up0 = time.perf_counter()
        try:
            pong = server.ensure_server(dev_idx)
        except server.LampServerError as e:
            return _failure(spec, algo, status="error", error_kind="lamp_server_unavailable",
                            error_msg=str(e), started_utc=started_utc, run_uid=run_uid)
        t_startup = time.perf_counter() - t_up0
        cold_start = t_startup > 5.0
        if cuda and not pong.get("has_gpu"):
            return _failure(spec, algo, status="error", error_kind="lamp_server_no_gpu",
                            error_msg=f"server pong: {pong}", started_utc=started_utc,
                            run_uid=run_uid)
        if not pong.get("warm", False):
            return _failure(spec, algo, status="error", error_kind="lamp_server_cold",
                            error_msg=f"server pong: {pong}", started_utc=started_utc,
                            run_uid=run_uid)
        extras.update({"lamp_server_startup_s": t_startup, "lamp_server_cold_start": cold_start})

        # The solve, timed in the server.
        budget = float(spec.time_limit_s) - (time.perf_counter() - t_wall0) - WALL_MARGIN_S
        if budget <= 1.0:
            # Start-up took the wall time and LAMP never ran: a retryable row.
            return _failure(spec, algo, status="error", error_kind="lamp_no_budget_after_startup",
                            error_msg=f"server start-up took {t_startup:.0f} s",
                            started_utc=started_utc, run_uid=run_uid, extras=extras)
        sampler = NvmlPeakSampler(dev_idx) if cuda else None
        if sampler is not None:
            sampler.start()
        try:
            resp = server.solve(payload, timeout_s=budget, device=dev_idx)
        except server.LampTimeout as e:
            if cold_start:
                # This runner's own start-up shortened the budget: retryable on
                # resume, when the server is warm.
                return _failure(spec, algo, status="error", error_kind="lamp_timeout_after_cold_start",
                                error_msg=f"{e!r}; start-up {t_startup:.0f} s",
                                started_utc=started_utc, run_uid=run_uid, extras=extras)
            return _failure(spec, algo, status="timeout", error_kind="lamp_server_timeout",
                            error_msg=repr(e), started_utc=started_utc, run_uid=run_uid)
        except server.LampServerDied as e:
            oomish = any(k in str(e).lower() for k in ("out of memory", "killed", "oom"))
            return _failure(spec, algo, status="oom" if oomish else "error",
                            error_kind="lamp_server_died", error_msg=str(e),
                            started_utc=started_utc, run_uid=run_uid)
        except server.LampServerError as e:
            return _failure(spec, algo, status="error", error_kind="lamp_server_unavailable",
                            error_msg=str(e), started_utc=started_utc, run_uid=run_uid)
        finally:
            peak_mem, base_mem = sampler.stop() if sampler is not None else (None, None)
        if resp.get("status") != "ok":
            return _failure(spec, algo, status=resp.get("status", "error"),
                            error_kind=resp.get("error_kind", "lamp_solve_failed"),
                            error_msg=resp.get("error_msg", ""), started_utc=started_utc,
                            run_uid=run_uid)

        # Recovery of (X, α, β), in t_total. This runner's own copy of C, for
        # recovery and scoring, is moved to the device outside every clock.
        device = torch.device(spec.device if cuda else "cpu")
        C = C_host.to(device) if C_host is not None else None
        if cuda:
            torch.cuda.synchronize(device)
        t_r0 = time.perf_counter()
        theta = torch.from_numpy(np.fromfile(blobs / "theta.f64", dtype=np.float64)).to(device)
        nu_np = np.fromfile(blobs / "nu.f64", dtype=np.float64)
        r = torch.from_numpy(r_np).to(device)
        c = torch.from_numpy(c_np).to(device)
        r_floored = torch.from_numpy(r_solve).to(device)
        eta_t, st = float(resp["eta_t"]), float(resp["st"])
        w_inf, w_inf_scaling = float(resp["W_inf"]), float(resp["W_inf_scaling"])
        w_eff = effective_scale(w_inf, w_inf_scaling)
        s, b_np = plan_parameters(nu_np, eta_t, st, w_eff)
        b = torch.from_numpy(b_np).to(device)
        if mode == "kernel":
            Pt = torch.from_numpy(P).to(device)
            Qt = torch.from_numpy(Q).to(device)
            cost_fn = geo_cost.make_metric_cost_fn(Pt, Qt, geo_cost_type(spec.norm), float(spec.c_max))
            rowmin = lambda p: _ctransform_symbolic_rowmin(p, cost_fn, m, n)
        else:
            cost_fn = lambda i0, i1: C[i0:i1]
            rowmin = lambda p: _ctransform_storedC_rowmin(p, C)
        logZ = torch.empty(m, dtype=torch.float64, device=device)
        _pointcloud_row_lse(logZ, cost_fn, b, -s, m, _pointcloud_block_chunk(m, n))
        a = r_floored.log() - logZ
        rep = SymbolicFactoredPlan(a=a, b=b, s=s, cost_fn=cost_fn, m=m, n=n)
        alpha, beta = theta_dual(theta, w_eff, rowmin)
        c_floored = torch.from_numpy(c_solve).to(device)
        if cuda:
            torch.cuda.synchronize(device)
        wall_recovery = time.perf_counter() - t_r0

        # Scoring, on neither clock.
        t_e0 = time.perf_counter()
        rs, cs = rep.row_sums(), rep.col_sums()
        row_err = float((rs - r).norm(2).item())
        col_err = float((cs - c).norm(2).item())
        # LAMP's own infeasibility, ‖Xᵀ1 − c̃‖₁ against the c̃ it solved on, on
        # the delivered plan (X1 = r̃ by construction of Z).
        feaserr = float((cs - c_floored).abs().sum().item())
        p_obj = rep.trace_with_C()
        blk: Dict[str, Any] = {}
        if spec.pareto:
            if mode == "kernel":
                blk = score_symbolic(p_obj=p_obj, r_P_row=row_err, r_P_col=col_err,
                                     r_P_neg=0.0, alpha=alpha, beta=beta, r=r, c=c,
                                     cost_fn=cost_fn, m=m, n=n)
            else:
                blk = score(rep, alpha, beta, r, c, C=C, x_nonneg_structural=True,
                            compute_frobenius=True)
            alpha_nu, beta_nu = theta_dual(torch.from_numpy(nu_np).to(device), w_eff, rowmin)
            d_obj_nu = float((r * alpha_nu).sum().item() + (c * beta_nu).sum().item())
            if mode == "kernel":
                r_df_nu = _dual_feas_F_symbolic(alpha_nu, beta_nu, cost_fn, m, n)
            else:
                r_df_nu = _dual_feas_F_storedC(alpha_nu, beta_nu, C)
            gap_nu = p_obj - d_obj_nu
            extras["lamp_nu_dual"] = sanitize_block({
                "d_obj": d_obj_nu, "gap": gap_nu, "r_DF": r_df_nu,
                "E_abs": max(abs(blk.get("r_P") or 0.0), abs(r_df_nu), abs(gap_nu))})
        # Cross-checks against the plan and normaliser LAMP itself computed.
        xc: Dict[str, Any] = {}
        if "xc_obj" in resp:
            xc["obj_abs_diff"] = abs(float(resp["xc_obj"]) - p_obj)
            xc["col_l1_lamp"] = float(resp["xc_col_l1"])
            xc["col_l1_ours"] = feaserr
            xc["row_l1_lamp"] = float(resp["xc_row_l1"])
        phi_path = blobs / "phi.f64"
        if phi_path.exists():
            phi = np.fromfile(phi_path, dtype=np.float64)
            xc["normaliser_max_abs_diff"] = float(np.max(np.abs(phi - a.cpu().numpy())))
        extras["lamp_crosscheck"] = sanitize_block(xc)
        if cuda:
            torch.cuda.synchronize(device)
        wall_eval = time.perf_counter() - t_e0

        rows = _check_rows(blobs / "checks.csv")
        t_solve = float(resp["t_solve"])
        reason = str(resp["reason"])
        native_stop = (f"lamp_native_eps{settings['epsilon']:g}_tmax{settings['tmax']:g}"
                       f"_itermax{settings['itermax']}_check{NATIVE_FREQUENCY}")
        pareto = None
        if spec.pareto:
            pareto = sanitize_block({
                **blk,
                "t_solve": t_solve, "t_total": t_solve + wall_recovery,
                "t_dual_recovery": wall_recovery, "t_eval": wall_eval,
                "t_cost_build": t_marshal, "precision": PRECISION,
                "native_stop": native_stop, "native_stop_is_upstream_default": is_default,
                "native_stop_is_library_default": is_default,
                "t_solve_boundary": T_SOLVE_BOUNDARY, "t_solve_includes": list(T_SOLVE_INCLUDES),
                "t_solve_excludes": list(T_SOLVE_EXCLUDES), "t_total_boundary": T_TOTAL_BOUNDARY,
                "t_total_includes": list(T_TOTAL_INCLUDES), "pareto_partial": False,
                "x_nonneg_structural": True, "r_P_neg_source": "gibbs_form_strictly_positive",
                "lamp_exit": reason})
        fp = device_fingerprint(idx=dev_idx) if cuda else {}
        extras.update({
            "feaserr_measure": FEASERR_MEASURE, "r_P": row_err + col_err,
            "lamp_path": mode, "lamp_eps_test_factor": EPS_TEST_FACTOR[mode],
            "lamp_settings": settings, "lamp_check_period": NATIVE_FREQUENCY,
            "lamp_exit": reason, "lamp_eta_t": eta_t, "lamp_st": st, "lamp_W_inf": w_inf,
            "lamp_W_inf_scaling": w_inf_scaling, "lamp_W_eff": w_eff, "lamp_s": s,
            "lamp_t_loop_s": float(resp["t_loop"]), "lamp_check_rows": len(rows),
            "lamp_jit_in_solve": resp.get("jit_in_solve"),
            "lamp_last_check": rows[-1] if rows else None,
            "tolerance_consumed": True, "certify_unfloored": True,
            "lamp_epsilon_source": "benchmark_tolerance",
            "lamp_upstream_default_epsilon": float(native_settings()["epsilon"]),
            "lamp_check_trace": thin_trace(rows),
            "marginal_floor": MARGINAL_FLOOR, "marginal_floor_source": MARGINAL_FLOOR_SOURCE,
            "marginal_floor_shift_l2": [float(np.linalg.norm(r_solve - r_np)),
                                        float(np.linalg.norm(c_solve - c_np))],
            # The dual objective on (r̃, c̃), the marginals LAMP's own test uses.
            "lamp_d_obj_solved_marginals": float((r_floored * alpha).sum().item()
                                                 + (c_floored * beta).sum().item()),
            "lamp_pin": server.PIN_FILE.read_text().strip(),
            "lamp_server_fingerprint": server.fingerprint(),
            "wall_time_eval_s": wall_eval, "nvml_baseline_bytes": base_mem,
            "row_sum_err_l2": row_err, "col_sum_err_l2": col_err,
            "t_solve_boundary": T_SOLVE_BOUNDARY, "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES), "t_total_boundary": T_TOTAL_BOUNDARY,
            "t_total_includes": list(T_TOTAL_INCLUDES)})
        if mode == "kernel" and abs(w_eff - 1.0) > 1e-9:
            extras["lamp_kernel_W_eff_not_one"] = True          # spec.c_max is not the exact maximum
        gap_test_inconsistent = (mode == "kernel" and spec.norm == "inf"
                                 and abs(w_inf - 1.0) > 1e-9)
        if gap_test_inconsistent:
            extras["lamp_native_gap_test_inconsistent"] = True
        if pareto is not None:
            extras["pareto"] = pareto
        return ResultRecord(
            instance_id=spec.instance_id, algorithm=algo, regime=_regime(algo),
            norm=spec.norm, resolution=spec.resolution, n=m,
            m=spec.m if algo == ALGO_PC else m,
            run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
            status="ok", timeout_flag=False,
            wall_time_s=t_solve + wall_recovery + wall_eval,
            wall_time_solve_s=t_solve, wall_time_recovery_s=wall_recovery,
            iters=int(resp["iters"]), iter_cap_hit=(reason == "itermax"),
            native_termination=reason, tolerance=spec.tolerance,
            feaserr=feaserr, objective=float(p_obj),
            # LAMP prints its primal and dual values in its own cost units, C/W∞-scaling.
            native_duality_gap=(w_inf_scaling * (rows[-1][4] - rows[-1][5])
                                if rows and reason == "epsilon" and not gap_test_inconsistent
                                else None),
            peak_gpu_mem_bytes=peak_mem,
            peak_gpu_mem_source=("nvml_poll" if peak_mem is not None else "unknown"),
            device_name=fp.get("device_name"), device_uuid=fp.get("device_uuid"),
            dataset=spec.dataset, rung=spec.rung, algo_extras=extras)
    finally:
        shutil.rmtree(blobs, ignore_errors=True)


def main_for(algo: str, argv=None) -> int:
    return runner_main(
        argv, description=f"{algo} runner (LAMP, Log-Averaged Mirror Prox).",
        run=lambda spec, args: _run_one(spec, algo),
        failure=lambda spec, args, **kw: _failure(spec, algo, **kw))


def main(argv=None) -> int:
    return main_for(ALGO_GENERAL, argv)


if __name__ == "__main__":
    sys.exit(main())
