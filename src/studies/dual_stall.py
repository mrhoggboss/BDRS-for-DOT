"""The dual stall of three Douglas–Rachford solvers, BDRS, DROT and HOT, on
DOTmark pairs with the sql2 cost.

Each solver runs exactly N iterations with no stop. At every check iteration k
of a fixed schedule two triples are scored against the instance's unit-mass
marginals (r, c) and the max-1 lattice cost C. The native triple
(X^k, α^k, β^k) is the solver's own:

    BDRS  X^k = exp(ξ^k ⊕ k·L ⊕ ζ^k), the factored plan;
          (α^k, β^k) = (η·log u^k, η·log v^k), η = 0.7·mean(C).
    DROT  X^k the plan after k steps, with σC added back when the last step
          index k − 1 is odd; (α^k, β^k) = (φ₁/σ, φ₂/σ), σ = 2/(m + n).
    HOT   X^k = recover_transport_plan() of the Halpern iterate after k steps;
          (α^k, β^k) = c_s·(ȳ[mn:2mn], ȳ[2mn:3mn]), with c_s = ‖w‖₂ + 1 the
          factor HOT divides its cost w by.

The corrected triple (X^k, α′, β′) keeps X^k and replaces the dual by the
alternating c-transform of α^k,

    β′_j = min_i (C_ij − α^k_i),    α′_i = min_j (C_ij − β′_j),

so that α′ ⊕ β′ ≤ C. Its gap g′ = ⟨C, X^k⟩ − (rᵀα′ + cᵀβ′) is signed.

BDRS runs once, with its checks at the scheduled iterations; its iterates do
not depend on where the checks are. DROT and HOT run again from scratch for
every k, exactly k iterations with a stopping tolerance of 0, so no
evaluation touches an iterate that a later check reads.

Every triple is scored in fp64, against the unit-mass fp64 marginals and the
fp64 max-1 lattice cost C. DROT iterates on their fp32 casts r₃₂, c₃₂, C₃₂.
Scoring DROT's triples against the fp32 casts instead would change r_DF by up
to ‖C₃₂ − C‖_F (5.7e-6 at n_g = 32, 2.3e-5 at n_g = 64) and the other fields
by about 1e-7 or less (max|C₃₂ − C| = 3e-8).

One JSONL per solver, ``<output-dir>/runs/<SOLVER>.jsonl``, holds three row
types per instance, appended together once the instance has finished:

    instance  instance_id, n_g, m, n, cost_type, normalization, mass_r, mass_c, C_inf
    check     iter and, at that iteration,
                native_r_P, native_r_DF, native_r_Dinf, native_gap,
                native_p_obj, native_d_obj        the native triple
                repaired_r_DF, repaired_r_Dinf, repaired_gap (= g′),
                repaired_d_obj (= ⟨C, X^k⟩ − g′)   the corrected triple
                r_P, r_DF, r_Dinf, p_obj, d_obj, gap
                                                  the triple the solver's runner
                                                  delivers: corrected for BDRS,
                                                  native for DROT and HOT
              with r_P = ‖X1 − r‖₂ + ‖Xᵀ1 − c‖₂ + ‖min(X, 0)‖_F, the same
              for both triples, r_DF = ‖(α ⊕ β − C)₊‖_F, r_Dinf =
              max(0, max_ij(α_i + β_j − C_ij)) (a diagnostic), and gap and
              native_gap = |p_obj − d_obj|; the timing fields t_solver,
              t_wall, eval_overhead and c_transform_time_s; for HOT,
              hot_kkt_res, HOT's own KKT value, and hot_kkt_res_iter, the
              iteration it was last computed at; for DROT, drot_native_res,
              its residual √(‖X1 − r‖² + ‖Xᵀ1 − c‖²) in fp32.
    result    termination_reason "fixed_iters" with iters = N, or
              "solver_failure" with the error.

Every row carries git_sha and device_uuid. When the command runs again it
skips an instance whose fixed_iters result row has iters = N and check_at
equal to the schedule of N, and retries a failed one. A fixed_iters result
row with another N or schedule makes it refuse to run (exit 2).

    python -m src.studies.dual_stall --solver {BDRS,DROT,HOT} --resolution {32,64}
        [--fixed-iters N] [--classes ...] [--pairs i,j ...] --output-dir DIR
        [--device cuda:0] [--data-dir DIR] [--no-warmup]

``--device cpu`` runs BDRS through the dense torch path and HOT in torch on
the CPU. DROT's shim runs on the process's default CUDA device, so DROT takes
only ``--device cuda`` or ``cuda:0``; CUDA_VISIBLE_DEVICES selects the GPU.
"""

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from src.benchmark import env_capture
from src.instances.dotmark_build import OUTPUT_ROOT
from src.methods.dual_cleanup import feasible_dual_and_gap
from src.methods.eta import eta_from_mean, grid_cost_mean
from src.methods.grid_cost import _cost_axis
from src.methods.params import BregmanDRParams
from src.protocol.instance_record import GRID, InstanceSpec, utc_now_iso
from src.protocol.paths import REPO_ROOT
from src.protocol.results_io import iter_rows
from src.runners.common import (
    WARMUP_SEED,
    load_marginal_unit_mass,
    mentions_oom,
    scratch_dir,
    warmup_marginal,
)
from src.scoring.kkt_residual import Candidate, Instance, evaluate
from src.scoring.plan_metrics import BdrsFactoredPlan


SOLVERS = ("BDRS", "DROT", "HOT")
NORM = "sql2"
CLASSES = ("CauchyDensity", "ClassicImages", "GRFmoderate", "GRFrough", "GRFsmooth",
           "LogGRF", "LogitGRF", "MicroscopyImages", "Shapes", "WhiteNoise")
PAIRS = ((0, 1), (0, 2), (0, 3), (0, 4), (0, 5))
FIXED_ITERS = {32: 6000, 64: 12000}         # N by n_g
FIRST_IMAGE_ID = 1001                       # image i of a DOTmark class is data<n_g>_<1001 + i>

# The check schedule: every k up to DENSE_UNTIL, PER_DECADE points per decade,
# every multiple of EVERY, and N.
PER_DECADE = 12
DENSE_UNTIL = 10
EVERY = 100

# A shim call runs at most N iterations, a few seconds at n_g = 64.
SHIM_TIMEOUT_S = 600

CHECK_FIELDS = (
    "native_r_P", "native_r_DF", "native_r_Dinf", "native_gap", "native_p_obj", "native_d_obj",
    "repaired_r_DF", "repaired_r_Dinf", "repaired_gap", "repaired_d_obj",
    "r_P", "r_DF", "r_Dinf", "p_obj", "d_obj", "gap",
)


def check_schedule(n_iters: int, *, per_decade: int = PER_DECADE,
                   dense_until: int = DENSE_UNTIL, every: int = EVERY) -> List[int]:
    """The check iterations in [1, N]: every k ≤ ``dense_until``, the points
    k = round(10^(j/``per_decade``)) above it, every multiple of ``every``,
    and N. A shorter N gives a prefix of a longer N's schedule, plus N."""
    ks = set(range(1, min(int(dense_until), int(n_iters)) + 1))
    j = 0
    while True:
        k = int(round(10 ** (j / float(per_decade))))
        if k > n_iters:
            break
        if k > dense_until:
            ks.add(k)
        j += 1
    ks.update(range(int(every), int(n_iters) + 1, int(every)))
    ks.add(int(n_iters))
    return sorted(ks)


def dotmark_spec(class_name: str, ng: int, i: int, j: int, *, data_dir: Path,
                 device: str) -> InstanceSpec:
    """The spec of DOTmark pair (i, j) of ``class_name`` at n_g with the sql2
    cost, its marginals under ``<data_dir>/marginals``. ``cost_matrix_path``
    names the stored max-1 cost when ``<data_dir>/cost_matrices`` has it; only
    DROT reads it, as fp32."""
    data_dir = Path(data_dir)
    id_i, id_j = FIRST_IMAGE_ID + int(i), FIRST_IMAGE_ID + int(j)

    def marginal(image_id: int) -> str:
        return str(data_dir / "marginals" / class_name / f"n{ng}" / f"data{ng}_{image_id}.npy")

    cost = data_dir / "cost_matrices" / f"cost_n{ng}_{NORM}.npy"
    return InstanceSpec(
        instance_id=f"{class_name}_n{ng}_i{i}_j{j}_{NORM}", dataset="dotmark", rung=GRID,
        class_name=class_name, resolution=int(ng), i=int(i), j=int(j), id_i=id_i, id_j=id_j,
        norm=NORM, marginal_i_path=marginal(id_i), marginal_j_path=marginal(id_j),
        cost_matrix_path=str(cost) if cost.is_file() else None,
        warmup=0, device=device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def _quiet():
    """Discard what a vendored solver prints (HOT's progress lines and tqdm bars)."""
    with open(os.devnull, "w") as sink, \
            contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        yield


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def score_check(X: Any, alpha: torch.Tensor, beta: torch.Tensor, inst: Instance,
                cost_axis: torch.Tensor, *, x_nonneg_structural: bool,
                delivers_corrected: bool) -> Dict[str, Any]:
    """The check row's KKT fields for the native triple (X, α, β) and the
    corrected triple (X, α′, β′).

    ``X`` is a dense plan or a ``plan_metrics`` representation;
    ``x_nonneg_structural`` declares X ≥ 0 by construction. The unprefixed
    fields copy the corrected triple when ``delivers_corrected``, else the
    native one."""
    device = inst.r.device
    native = evaluate(Candidate(X=X, alpha=alpha, beta=beta,
                                x_nonneg_structural=x_nonneg_structural),
                      inst, compute_frobenius_dual=True)
    _sync(device)
    t0 = time.perf_counter()
    # (α, β) enter as (η·log u, η·log v) with η = 1; the alternating variant
    # reads α only.
    ap, bp, g_feas = feasible_dual_and_gap(
        alpha, beta, 1.0, inst.r, inst.c, int(inst.ng), str(inst.cost_type),
        float(native["p_obj"]), variant="alternating", cost_axis=cost_axis)
    _sync(device)
    t_ct = time.perf_counter() - t0
    corrected = evaluate(Candidate(X=X, alpha=ap, beta=bp,
                                   x_nonneg_structural=x_nonneg_structural),
                         inst, compute_frobenius_dual=True)
    row: Dict[str, Any] = {
        "native_r_P": float(native["r_P"]),
        "native_r_DF": float(native["r_DF"]),
        "native_r_Dinf": float(native["r_Dinf"]),
        "native_gap": float(native["gap"]),
        "native_p_obj": float(native["p_obj"]),
        "native_d_obj": float(native["d_obj"]),
        "repaired_r_DF": float(corrected["r_DF"]),
        "repaired_r_Dinf": float(corrected["r_Dinf"]),
        "repaired_gap": float(g_feas),
        "repaired_d_obj": float(native["p_obj"]) - float(g_feas),
    }
    delivered = corrected if delivers_corrected else native
    for key in ("r_P", "r_DF", "r_Dinf", "p_obj", "d_obj", "gap"):
        row[key] = float(delivered[key])
    row["c_transform_time_s"] = float(t_ct)
    return row


# --------------------------------------------------------------------------
# BDRS
# --------------------------------------------------------------------------

def bdrs_eta(ng: int) -> float:
    """η = 0.7·mean(C) with the exact mean of the n_g lattice cost."""
    return eta_from_mean(grid_cost_mean(int(ng), NORM))


def _bdrs_session(spec: InstanceSpec, inst: Instance, schedule: Sequence[int], n_iters: int,
                  device: torch.device, cost_axis: torch.Tensor, *,
                  _ext_override: Any = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One Grid solve of exactly N iterations whose stop hook scores the iterate
    at every scheduled k and never stops it: the primal gate ``epsilon_primal``
    is +inf, so the hook runs at every check, and it returns +inf against
    ``gap_stop_tol`` = 0. ``_ext_override`` replaces the CUDA extension (tests)."""
    from src.methods.grid_cuda import bregman_dr_solve_grid_cuda

    ng = int(spec.resolution)
    eta = bdrs_eta(ng)
    params = BregmanDRParams(eta=eta, max_iters=int(n_iters), epsilon_primal=math.inf,
                             check_every=int(n_iters))
    rows: List[Dict[str, Any]] = []
    clock = {"t0": 0.0, "overhead": 0.0}

    def hook(info: Dict[str, Any]) -> float:
        # Everything here is outside the solver clock. The session's next batch
        # overwrites ξ, ζ, log u and log v in place on its own stream, so the
        # hook copies them and synchronizes before it returns.
        _sync(device)
        t_pause = time.perf_counter()
        k = int(info["iters"])
        plan = BdrsFactoredPlan(A=info["A"].to(torch.float64).clone(),
                                B=info["B"].to(torch.float64).clone(),
                                ng=ng, cost_type=NORM, eta=eta, iters=k)
        alpha = eta * info["log_u"].reshape(-1).to(torch.float64)
        beta = eta * info["log_v"].reshape(-1).to(torch.float64)
        row = score_check(plan, alpha, beta, inst, cost_axis,
                          x_nonneg_structural=True, delivers_corrected=True)
        _sync(device)
        t_back = time.perf_counter()
        row.update({"iter": k,
                    "t_solver": (t_pause - clock["t0"]) - clock["overhead"],
                    "t_wall": t_pause - clock["t0"],
                    "eval_overhead": t_back - t_pause})
        clock["overhead"] += t_back - t_pause
        rows.append(row)
        return math.inf

    _sync(device)
    clock["t0"] = time.perf_counter()
    _, info = bregman_dr_solve_grid_cuda(
        None, inst.r, inst.c, params, cost_type=NORM, grid_size=ng, return_plan=False,
        gap_stop_fn=hook, gap_stop_tol=0.0, check_at=list(schedule),
        _ext_override=_ext_override)
    _sync(device)
    t_wall = time.perf_counter() - clock["t0"]
    if info["native_termination"] != "max_iters":
        raise RuntimeError(
            f"BDRS stopped by {info['native_termination']!r}, not after N iterations")
    return rows, {"iters": int(info["iters"]), "eta": eta,
                  "t_solver": t_wall - clock["overhead"], "t_wall": t_wall}


def _bdrs_dense(spec: InstanceSpec, inst: Instance, schedule: Sequence[int], n_iters: int,
                device: torch.device,
                cost_axis: torch.Tensor) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The torch General solver on the dense lattice C, run again for each k
    with its dense plan scored. The same iteration as the Grid solver, up to
    the rounding of its log-sum-exps; for runs without CUDA."""
    from src.instances.dense_cost import build_dense_cost
    from src.methods.factored_torch import bregman_dr_solve_general_factored

    ng = int(spec.resolution)
    eta = bdrs_eta(ng)
    C = build_dense_cost(ng, NORM, device=device, dtype=torch.float64)
    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for k in schedule:
        # A check period above k: no check runs, and the solve takes k steps.
        params = BregmanDRParams(eta=eta, max_iters=int(k), check_every=int(k) + 1)
        t0 = time.perf_counter()
        X, info = bregman_dr_solve_general_factored(C, inst.r, inst.c, params, return_plan=True)
        t1 = time.perf_counter()
        alpha = eta * info["log_u"].reshape(-1).to(torch.float64)
        beta = eta * info["log_v"].reshape(-1).to(torch.float64)
        row = score_check(X, alpha, beta, inst, cost_axis,
                          x_nonneg_structural=True, delivers_corrected=True)
        row.update({"iter": int(info["iters"]), "t_solver": t1 - t0, "t_wall": t1 - t0,
                    "eval_overhead": time.perf_counter() - t1})
        rows.append(row)
        del X
    t_wall = time.perf_counter() - t_start
    return rows, {"iters": int(n_iters), "eta": eta,
                  "t_solver": rows[-1]["t_solver"] if rows else 0.0, "t_wall": t_wall}


def run_bdrs(spec: InstanceSpec, inst: Instance, schedule: Sequence[int], n_iters: int,
             device: torch.device, cost_axis: torch.Tensor, *,
             _ext_override: Any = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The BDRS arm: the CUDA Grid session on a CUDA device (or with a test
    extension), else the dense torch path."""
    if device.type == "cuda" or _ext_override is not None:
        return _bdrs_session(spec, inst, schedule, n_iters, device, cost_axis,
                             _ext_override=_ext_override)
    return _bdrs_dense(spec, inst, schedule, n_iters, device, cost_axis)


# --------------------------------------------------------------------------
# DROT
# --------------------------------------------------------------------------

def _drot_shim() -> Path:
    from src.runners import run_drot as RD
    shim = Path(os.environ.get("BDRS_DROT_SHIM", str(RD.DEFAULT_SHIM_PATH)))
    if not shim.is_file() or not os.access(shim, os.X_OK):
        raise RuntimeError(
            f"DROT shim not found at {shim}; build it with `make -C vendor/drot/shim`")
    return shim


def run_drot(spec: InstanceSpec, inst: Instance, schedule: Sequence[int], n_iters: int,
             device: torch.device,
             cost_axis: torch.Tensor) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The DROT arm: for each k one shim call of exactly k steps.

    ``--max-iters k --check-freq k`` places the shim's only residual check
    after step k, so its tolerance test cannot end the loop earlier, and
    ``--tol 0`` keeps it from firing there. The shim gets the stored marginals
    and the cost as fp32 blobs written as ``run_drot`` writes them (C
    column-major); the plan comes back column-major in fp32 and is scored in
    fp64."""
    from src.runners import run_drot as RD

    ng = int(spec.resolution)
    if ng > RD.SKIP_RESOLUTION_ABOVE:
        raise RuntimeError(f"DROT is refused above n_g = {RD.SKIP_RESOLUTION_ABOVE}")
    shim = _drot_shim()
    N = ng * ng
    scratch = Path(tempfile.mkdtemp(prefix=f"dual_stall_drot_{spec.instance_id}_",
                                    dir=scratch_dir()))
    # Basenames that differ in more than letter case (case-insensitive filesystems).
    cost_blob = scratch / "cost.bin"
    r_blob, c_blob = scratch / "marg_r.bin", scratch / "marg_c.bin"
    plan_blob, info_path = scratch / "plan.bin", scratch / "info.json"
    alpha_blob, beta_blob = scratch / "phi1.bin", scratch / "phi2.bin"
    rows: List[Dict[str, Any]] = []
    sigma = None
    t_start = time.perf_counter()
    try:
        r32, c32 = RD._load_marginals_fp32(spec)
        r32.tofile(r_blob)
        c32.tofile(c_blob)
        RD._write_cost_fp32_blob(spec, cost_blob)
        for k in schedule:
            for p in (plan_blob, info_path, alpha_blob, beta_blob):
                p.unlink(missing_ok=True)
            cmd = [str(shim), "--cost", str(cost_blob), "--r", str(r_blob), "--c", str(c_blob),
                   "--n", str(N), "--tol", "0", "--max-iters", str(int(k)),
                   "--check-freq", str(int(k)),
                   "--plan-out", str(plan_blob), "--info-out", str(info_path),
                   "--alpha-out", str(alpha_blob), "--beta-out", str(beta_blob)]
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=SHIM_TIMEOUT_S, check=False)
            if proc.returncode == 2 or mentions_oom(proc.stderr or ""):
                raise RuntimeError(f"DROT shim out of memory at k = {k}: {proc.stderr[-2000:]}")
            if proc.returncode != 0 or not info_path.is_file() or not plan_blob.is_file():
                raise RuntimeError(f"DROT shim exit {proc.returncode} at k = {k}: "
                                   f"{(proc.stderr or '')[-2000:]}")
            info = json.loads(info_path.read_text())
            if int(info["iters"]) != int(k):
                raise RuntimeError(f"DROT shim ran {info['iters']} steps, not {k}")
            sigma = float(info["stepsize"])
            X = torch.from_numpy(np.ascontiguousarray(
                np.fromfile(plan_blob, dtype=np.float32, count=N * N).reshape(N, N, order="F")
            )).to(device=device, dtype=torch.float64)
            phi1 = torch.from_numpy(np.fromfile(alpha_blob, dtype=np.float32, count=N))
            phi2 = torch.from_numpy(np.fromfile(beta_blob, dtype=np.float32, count=N))
            alpha = phi1.to(device=device, dtype=torch.float64) / sigma
            beta = phi2.to(device=device, dtype=torch.float64) / sigma
            t1 = time.perf_counter()
            row = score_check(X, alpha, beta, inst, cost_axis,
                              x_nonneg_structural=False, delivers_corrected=False)
            row.update({"iter": int(info["iters"]),
                        "drot_native_res": float(info["residual"]),
                        "t_solver": float(info["solve_seconds"]),
                        "t_wall": t1 - t0,
                        "eval_overhead": time.perf_counter() - t1})
            rows.append(row)
            del X
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return rows, {"iters": int(n_iters), "drot_stepsize": sigma,
                  "drot_cost_source": "stored" if spec.cost_matrix_path else "lattice",
                  "t_solver": rows[-1]["t_solver"] if rows else 0.0,
                  "t_wall": time.perf_counter() - t_start}


# --------------------------------------------------------------------------
# HOT
# --------------------------------------------------------------------------

def _hot_module():
    from src.runners import run_hot as RH
    if RH.HOTSolver is None:
        raise RuntimeError(f"vendor/HOT does not import: {RH._HOT_IMPORT_ERROR}")
    return RH


def run_hot(spec: InstanceSpec, inst: Instance, schedule: Sequence[int], n_iters: int,
            device: torch.device,
            cost_axis: torch.Tensor) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The HOT arm: for each k a new solver of exactly k iterations.

    ``max_iters`` = k ends ``optimize()`` after step k, and ``tolerance`` = 0
    keeps its KKT test (every ``check_freq`` = 100 iterations, with the σ
    restart) from ending it earlier. The inputs are built as ``run_hot``
    builds them. The duals are read from ȳ before ``recover_transport_plan()``,
    which rescales and projects the Halpern iterate in place and is called
    once, after the run."""
    RH = _hot_module()
    ng = int(spec.resolution)
    b, w, c_max = RH._build_b_c_cMax(spec)
    check_freq = int(RH.DEFAULT_CHECK_FREQ)
    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for k in schedule:
        _sync(device)
        t0 = time.perf_counter()
        solver = RH.HOTSolver(
            b=b, c=w, m=ng, n=ng, cMax=c_max, max_iters=int(k), tolerance=0.0,
            check_freq=check_freq, sigma=None, adjust_sigma=True, logging=True,
            dtype=torch.float64, device=str(device))
        _sync(device)
        t1 = time.perf_counter()
        with _quiet():
            solver.optimize()
        _sync(device)
        t2 = time.perf_counter()
        iters = int(solver.iter_count)
        if iters != int(k):
            raise RuntimeError(f"HOT ran {iters} iterations, not {k}")
        mn = int(solver.mn)
        c_scale = float(solver.c_scale)
        alpha = (solver.y_bar[mn:2 * mn] * c_scale).to(device=device, dtype=torch.float64)
        beta = (solver.y_bar[2 * mn:3 * mn] * c_scale).to(device=device, dtype=torch.float64)
        hot_kkt_res = float(solver.kkt_res.item())
        with _quiet():
            X = solver.recover_transport_plan()
        X = X.to(torch.float64)
        _sync(device)
        t3 = time.perf_counter()
        row = score_check(X, alpha, beta, inst, cost_axis,
                          x_nonneg_structural=False, delivers_corrected=False)
        row.update({"iter": iters,
                    "hot_kkt_res": hot_kkt_res,
                    # The KKT test runs before the first step and at every
                    # multiple of check_freq.
                    "hot_kkt_res_iter": (iters // check_freq) * check_freq,
                    "t_solver": t2 - t1,
                    "t_wall": t3 - t0,
                    "eval_overhead": time.perf_counter() - t3})
        rows.append(row)
        del solver, X
    return rows, {"iters": int(n_iters), "hot_check_freq": check_freq, "hot_tolerance": 0.0,
                  "t_solver": rows[-1]["t_solver"] if rows else 0.0,
                  "t_wall": time.perf_counter() - t_start}


ARMS: Dict[str, Callable[..., Tuple[List[Dict[str, Any]], Dict[str, Any]]]] = {
    "BDRS": run_bdrs, "DROT": run_drot, "HOT": run_hot,
}


# --------------------------------------------------------------------------
# One instance
# --------------------------------------------------------------------------

def load_instance(spec: InstanceSpec, device: torch.device) -> Instance:
    """The scoring instance: the unit-mass marginals and the Grid sql2 cost."""
    r = torch.as_tensor(load_marginal_unit_mass(spec.marginal_i_path),
                        dtype=torch.float64, device=device)
    c = torch.as_tensor(load_marginal_unit_mass(spec.marginal_j_path),
                        dtype=torch.float64, device=device)
    inst = Instance(r=r, c=c, ng=int(spec.resolution), cost_type=NORM)
    inst.assert_normalized_max1()
    return inst


def run_instance(solver: str, spec: InstanceSpec, *, n_iters: int,
                 schedule: Optional[Sequence[int]] = None,
                 device: Optional[str] = None,
                 _ext_override: Any = None) -> List[Dict[str, Any]]:
    """The instance, check and result rows of ``solver`` on ``spec``, without
    the provenance fields. Raises if the solver fails or a scheduled check is
    missing."""
    if solver not in ARMS:
        raise ValueError(f"unknown solver {solver!r}; choose from {SOLVERS}")
    schedule = list(schedule) if schedule is not None else check_schedule(n_iters)
    if not schedule or schedule[-1] != int(n_iters) or schedule != sorted(set(schedule)):
        raise ValueError("the schedule must be increasing and end at N")
    dev = torch.device(device or spec.device)
    t0 = time.perf_counter()
    inst = load_instance(spec, dev)
    cost_axis = _cost_axis(NORM, int(spec.resolution), device=dev, dtype=torch.float64)
    head = {"solver": solver, "instance_id": spec.instance_id}
    instance_row = {
        "type": "instance", **head, "n_g": int(spec.resolution),
        "m": inst.m, "n": inst.n, "cost_type": NORM, "normalization": "max1",
        "mass_r": float(inst.r.sum().item()), "mass_c": float(inst.c.sum().item()),
        "C_inf": float(inst.C_inf),
    }
    kwargs = {"_ext_override": _ext_override} if solver == "BDRS" else {}
    checks, extra = ARMS[solver](spec, inst, schedule, int(n_iters), dev, cost_axis, **kwargs)
    got = [int(row["iter"]) for row in checks]
    if got != schedule:
        raise RuntimeError(f"{solver} checked at {got[:12]}..., not at the schedule")
    check_rows = [{"type": "check", **head, "iter": row["iter"], **row} for row in checks]
    result_row = {
        "type": "result", **head, "n_g": int(spec.resolution), "norm": NORM,
        "termination_reason": "fixed_iters", "n_checks": len(check_rows),
        "check_at": schedule, **extra,
        "eval_overhead_total": float(sum(row["eval_overhead"] for row in checks)),
        "t_instance": time.perf_counter() - t0, "utc": utc_now_iso(),
    }
    return [instance_row, *check_rows, result_row]


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def completed_instances(path: Path, solver: str, n_iters: int, schedule: Sequence[int]
                        ) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
    """The instances of ``solver`` in ``path`` whose fixed_iters result row has
    iters = N and check_at = ``schedule``, and, by instance, the first
    fixed_iters result row with another N or schedule."""
    done: Set[str] = set()
    other: Dict[str, Dict[str, Any]] = {}
    if not Path(path).is_file():
        return done, other
    for _, row in iter_rows([path]):
        if (row.get("type") != "result" or row.get("solver") != solver
                or row.get("termination_reason") != "fixed_iters"):
            continue
        if row.get("iters") == int(n_iters) and row.get("check_at") == list(schedule):
            done.add(row["instance_id"])
        else:
            other.setdefault(row["instance_id"], row)
    return done, other


def append_rows(path: Path, rows: Sequence[Dict[str, Any]], stamp: Dict[str, Any]) -> None:
    """Append ``rows``, each with ``stamp``, in one write followed by fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps({**row, **stamp}) + "\n" for row in rows)
    with open(path, "a") as fh:
        fh.write(blob)
        fh.flush()
        os.fsync(fh.fileno())


def warm_up(solver: str, ng: int, device: torch.device) -> None:
    """An untimed solve on seeded dummy marginals, never on an instance: BDRS
    10 iterations at n_g, which builds the CUDA extension; HOT ``run_hot``'s
    32 × 32 warm-up. DROT has none, as upstream ships none."""
    if solver == "BDRS" and device.type == "cuda":
        from src.methods.grid_cuda import bregman_dr_solve_grid_cuda
        rng = np.random.default_rng(WARMUP_SEED)
        r = torch.as_tensor(warmup_marginal(rng, ng * ng), dtype=torch.float64, device=device)
        c = torch.as_tensor(warmup_marginal(rng, ng * ng), dtype=torch.float64, device=device)
        params = BregmanDRParams(eta=bdrs_eta(ng), max_iters=10, check_every=10)
        bregman_dr_solve_grid_cuda(None, r, c, params, cost_type=NORM, grid_size=ng,
                                   return_plan=False)
    elif solver == "HOT":
        with _quiet():
            _hot_module()._warmup(device)
    _sync(device)


def _parse_pair(token: str) -> Tuple[int, int]:
    a, b = token.replace("-", ",").split(",")
    return int(a), int(b)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Fixed-iteration trajectories of the native and corrected triples "
                    "of BDRS, DROT and HOT on DOTmark sql2 pairs.")
    p.add_argument("--solver", required=True, choices=SOLVERS)
    p.add_argument("--resolution", required=True, type=int,
                   help=f"n_g; N defaults to {FIXED_ITERS}")
    p.add_argument("--fixed-iters", type=int, default=None, metavar="N",
                   help="iterations per run (default: N of the resolution)")
    p.add_argument("--classes", nargs="+", default=list(CLASSES), choices=CLASSES,
                   metavar="CLASS",
                   help=f"DOTmark classes (default: all ten, {', '.join(CLASSES)})")
    p.add_argument("--pairs", nargs="+", default=None, metavar="I,J",
                   help="image pairs, e.g. 0,1 0,2 (default: (0,1) to (0,5))")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data-dir", type=Path, default=Path(OUTPUT_ROOT),
                   help="the DOTmark data directory (marginals/, cost_matrices/)")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args(argv)

    if args.solver == "DROT" and args.device not in ("cuda", "cuda:0"):
        p.error(f"--solver DROT takes --device cuda or cuda:0, not {args.device}: the shim "
                f"runs on the default CUDA device; select the GPU with CUDA_VISIBLE_DEVICES")
    n_iters = args.fixed_iters or FIXED_ITERS.get(args.resolution)
    if not n_iters or n_iters <= 0:
        p.error(f"--fixed-iters is required at n_g = {args.resolution}")
    pairs = [_parse_pair(t) for t in args.pairs] if args.pairs else list(PAIRS)
    specs = [dotmark_spec(cls, args.resolution, i, j, data_dir=args.data_dir, device=args.device)
             for cls in args.classes for i, j in pairs]
    missing = [s.instance_id for s in specs
               if not (Path(s.marginal_i_path).is_file() and Path(s.marginal_j_path).is_file())]
    if missing:
        p.error(f"marginals missing for {len(missing)} instance(s), e.g. {missing[0]}; "
                f"build them with `python -m src.instances.dotmark_build`")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error(f"--device {args.device} but no CUDA device is visible")
    schedule = check_schedule(n_iters)

    path = args.output_dir / "runs" / f"{args.solver}.jsonl"
    done, other = completed_instances(path, args.solver, n_iters, schedule)
    if other:
        inst, row = next(iter(other.items()))
        same_n = row.get("iters") == n_iters
        print(f"[dual_stall] refusing to run: {inst} has a fixed_iters result in {path} with "
              f"N = {row.get('iters')}{' and another check schedule' if same_n else ''}, "
              f"and this run has N = {n_iters} ({len(other)} instance(s) differ); "
              f"choose another --output-dir", file=sys.stderr)
        return 2
    todo = [s for s in specs if s.instance_id not in done]
    print(f"[dual_stall] {args.solver} n_g={args.resolution} N={n_iters}: {len(schedule)} checks "
          f"{schedule[:12]}{' ...' if len(schedule) > 12 else ''}; "
          f"{len(todo)} of {len(specs)} instances to run, {len(specs) - len(todo)} done; "
          f"rows to {path}")
    if not todo:
        return 0
    header = env_capture.write(args.output_dir / "header.json", REPO_ROOT)
    stamp = {"git_sha": header.get("git_sha"),
             "device_uuid": env_capture.cached_device_uuid(device.index)
             if device.type == "cuda" else None}
    if not args.no_warmup:
        warm_up(args.solver, args.resolution, device)

    failed = 0
    for spec in todo:
        try:
            rows = run_instance(args.solver, spec, n_iters=n_iters, schedule=schedule,
                                device=args.device)
        except Exception as e:                              # noqa: BLE001
            failed += 1
            append_rows(path, [{"type": "result", "solver": args.solver,
                                "instance_id": spec.instance_id, "n_g": args.resolution,
                                "norm": NORM, "termination_reason": "solver_failure",
                                "error": f"{e!r}\n{traceback.format_exc()[-4000:]}",
                                "utc": utc_now_iso()}], stamp)
            print(f"[dual_stall] {args.solver} {spec.instance_id} FAILED: {e!r}", file=sys.stderr)
            continue
        append_rows(path, rows, stamp)
        res = rows[-1]
        print(f"[dual_stall] {args.solver} {spec.instance_id} iters={res['iters']} "
              f"checks={res['n_checks']} t_instance={res['t_instance']:.1f}s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
