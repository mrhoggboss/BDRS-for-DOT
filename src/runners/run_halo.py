"""HALO runner: the hierarchical multiscale OT solver of Xia et al. (ICLR 2026,
https://github.com/WenzhouXia/HALO), sql2 only.

A Grid instance runs HALO's grid mode (``emd2_grid``), a Point Cloud instance
its tree mode (``emd2``). HALO's grid cost is (Δi² + Δj²)/n_g², and the sql2
grid cost here is s times that, s = n_g²/(2(n_g−1)²); so the objective and r_P
are recomputed in fp64 from HALO's sparse plan under our cost, and its duals
``dual_source = [u; v]`` are recovered as α = s·u, β = s·v. In tree mode the
coordinates are scaled by 1/√C_max before the solve, so HALO solves the max-1
cost itself and s = 1.

HALO allocates through torch, CuPy and cuPDLPx's own cudaMalloc, so the peak
GPU memory is polled from NVML rather than read from torch.
"""

import dataclasses
import math
import sys
import time
import traceback
import uuid
from typing import Optional, Tuple

import numpy as np
import torch

from src.protocol.paths import REPO_ROOT
from src.protocol.instance_record import InstanceSpec, ResultRecord, utc_now_iso
from src.scoring.plan_metrics import (
    evaluate_plan_metrics,
    sparse_plan_from_scipy_coo,
)
from src.runners.gpu_state import device_fingerprint
from src.runners.common import (
    WARMUP_SEED,
    cuda_ordinal,
    failure_record,
    is_oom_exception,
    json_safe,
    load_marginal,
    runner_main,
    warmup_blobs,
)


_HALO_BUILD = REPO_ROOT / "vendor" / "halo" / "_build"
_HALO_SRC = REPO_ROOT / "vendor" / "halo" / "upstream" / "src"
for _p in (_HALO_SRC, _HALO_BUILD):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from hierarchical_ot import GridConfig, TreeConfig, emd2, emd2_grid  # type: ignore
except ImportError as _e:                               # pragma: no cover
    GridConfig = TreeConfig = emd2 = emd2_grid = None
    _HALO_IMPORT_ERROR = repr(_e)
else:
    _HALO_IMPORT_ERROR = None


# No δ_m floor: HALO gets the stored marginals.
MARGINAL_FLOOR = 0.0

# t_solve is the ``emd2_grid`` / ``emd2`` call alone. HALO casts and
# renormalises its inputs and copies them to the device inside that call, which
# cannot be split out without editing upstream (T_SOLVE_INCLUDES, named for grid
# mode; in tree mode the cast inside the call is to fp64, and the runner's fp32
# cast of the coordinates is part of the prescale, before t0).
T_SOLVE_BOUNDARY_GRID = "halo_emd2_grid_call"
T_SOLVE_BOUNDARY_TREE = "halo_emd2_tree_call"
T_SOLVE_INCLUDES: Tuple[str, ...] = ("input_fp32_cast", "input_h2d")
T_SOLVE_EXCLUDES = ("marginals_load", "input_reshape_or_prescale",
                    "config_build", "warmup")
T_TOTAL_BOUNDARY_GRID = "halo_emd2_grid_plus_coo_assembly_plus_dual_rescale"
T_TOTAL_BOUNDARY_TREE = "halo_emd2_tree_plus_coo_assembly_plus_dual_slice"
T_TOTAL_INCLUDES: Tuple[str, ...] = ()


def _halo_obj_scale_factor(ng: int) -> float:
    """s = n_g²/(2(n_g−1)²), the ratio of our sql2 grid cost to HALO's.

    It scales HALO's duals; for the objective it is only a recorded cross-check
    (objective ≈ s·obj_halo_native)."""
    return ng * ng / (2.0 * (ng - 1) ** 2)


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str,
                         engine: Optional[str] = None) -> ResultRecord:
    return failure_record(spec, "HALO", status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, m=spec.m,
                          algo_extras={"engine": engine} if engine is not None else None)


def _load_marginals(spec: InstanceSpec) -> Tuple[np.ndarray, np.ndarray]:
    """The stored marginals. Grid mode renormalises its own fp32 copies, so these
    arrays stay unchanged and every metric is scored against them."""
    return load_marginal(spec.marginal_i_path), load_marginal(spec.marginal_j_path)


def _make_config(tol: float, engine: str, *, pareto: bool = False) -> "GridConfig":
    """HALO's grid-mode config.

    With ``pareto`` no tolerance is passed and HALO's default ``tolerance=1e-6``
    stands, so the driver's tolerance cannot change HALO's stop. Without it
    ``tol`` is passed as ``tolerance`` and as ``final_refinement_tolerance``;
    upstream carries the latter but never reads it."""
    _tol_kw = {} if pareto else {
        "tolerance": float(tol),
        "final_refinement_tolerance": {
            "objective": float(tol), "primal": float(tol), "dual": float(tol)},
    }
    return GridConfig(
        cost_type="l2^2",
        solver_engine=engine,
        **_tol_kw,
        printing={"enabled": False},
        enable_profiling=False,
        # The scipy engine gets the CPU feasibility check, but grid shielding
        # runs on CuPy with either engine, so a CUDA device is still needed.
        check_type=("cpu" if engine == "scipy" else "gpu_exact"),
    )


# The warmup pays only size-independent first-call costs (numba and cubin JIT,
# the CuPy pool, the first pycupdlpx call), so a small grid suffices. A
# same-n_g warmup at large n_g builds a large final-level LP, and a failed host
# allocation in cuPDLPx calls exit(), which Python cannot catch.
WARMUP_NG_MAX = 128


def _warmup(ng: int, engine: str) -> None:
    """Untimed solve on two dummy blobs of side min(n_g, WARMUP_NG_MAX); failures are ignored."""
    ng = min(ng, WARMUP_NG_MAX)
    a, b = warmup_blobs(ng, np.random.default_rng(WARMUP_SEED))
    try:
        cfg = _make_config(1e-2, engine)
        emd2_grid(a, b, config=cfg, log=False, return_coupling=False)
    except Exception:                                       # noqa: BLE001
        pass


from src.runners.nvml_sampler import NvmlPeakSampler as _NvmlPeakSampler


# ---------------------------------------------------------------------------
# Tree mode (Point Cloud instances)
# ---------------------------------------------------------------------------

# Only the keys are read (the norm gate): _make_tree_config sets "l2^2" itself.
# TreeConfig also accepts l1 and linf, but HALO is scheduled on sql2 only.
_TREE_COST_BY_NORM = {"sql2": "l2^2"}


def _load_clouds(spec: InstanceSpec):
    """Clouds P (m × d), Q (n × d) and the stored marginals r (``a``), c (``b``), all fp64."""
    P = np.load(spec.support_i_path).astype(np.float64)
    Q = np.load(spec.support_j_path).astype(np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    if Q.ndim == 1:
        Q = Q.reshape(-1, 1)
    a = np.load(spec.marginal_i_path).astype(np.float64).flatten()
    b = np.load(spec.marginal_j_path).astype(np.float64).flatten()
    return P, Q, a, b


def _make_tree_config(tol: float, *, pareto: bool = False) -> "TreeConfig":
    """HALO's tree-mode config, with the tolerances as in ``_make_config``.

    No ``solver_engine`` is passed, so tree mode always uses cuPDLPx."""
    _tol_kw = {} if pareto else {
        "tolerance": float(tol),
        "final_refinement_tolerance": {
            "objective": float(tol), "primal": float(tol), "dual": float(tol)},
    }
    return TreeConfig(
        cost_type="l2^2",
        **_tol_kw,
        printing={"enabled": False},
        enable_profiling=False,
    )


def _native_stop_label(cfg) -> str:
    """A label of the tolerances ``cfg`` holds, read from the config itself."""
    try:
        tol = cfg.normalized_tolerance()
        fin = cfg.normalized_final_refinement_tolerance()
    except Exception:                                       # noqa: BLE001
        return "halo_config_unknown"
    def _fmt(d):
        if d is None:
            return "none"
        if isinstance(d, dict):
            return "/".join(f"{k}={float(v):g}" for k, v in sorted(d.items())
                            if isinstance(v, (int, float)))
        return f"{float(d):g}"
    return f"halo_tol[{_fmt(tol)}]_finalref[{_fmt(fin)}]"


# Points per side of the tree-mode warm-up clouds. HALO sends a problem whose
# two clouds have at most 256 points each to a single-level exact solve that
# its tree mode does not implement, so smaller warm-up clouds would end in that
# error before the multilevel solve that the timed run uses.
TREE_WARMUP_POINTS = 1024


def _tree_warmup(d: int, *, pareto: bool = False) -> None:
    """Untimed solve on seeded dummy clouds of ``TREE_WARMUP_POINTS`` points per
    side in dimension ``d``, so that the first call of HALO's multilevel solve
    (its cuPDLPx backend, CuPy kernels and Numba functions) falls here and not
    in the timed solve; failures are ignored.

    Under ``pareto`` the config is the timed solve's, so the warm-up runs the
    same code paths. The clouds are fp32, as the timed solve's are."""
    rng = np.random.default_rng(20260427)
    Pw = rng.standard_normal((TREE_WARMUP_POINTS, d)).astype(np.float32)
    Qw = (rng.standard_normal((TREE_WARMUP_POINTS, d)) + 0.4).astype(np.float32)
    aw = np.full(TREE_WARMUP_POINTS, 1.0 / TREE_WARMUP_POINTS)
    bw = np.full(TREE_WARMUP_POINTS, 1.0 / TREE_WARMUP_POINTS)
    try:
        emd2(Pw, Qw, source_mass=aw, target_mass=bw,
             config=_make_tree_config(1e-2, pareto=pareto),
             log=False, return_coupling=False)
    except Exception:                                       # noqa: BLE001
        pass


def _geo_metrics_from_coo(coo, P: np.ndarray, Q: np.ndarray,
                          a: np.ndarray, b: np.ndarray, c_max: float):
    """⟨C, X⟩ and r_P in fp64 from a COO plan over point indices, C_ij = ‖p_i − q_j‖²/C_max.

    Returns (⟨C, X⟩, r_P, ‖X1 − a‖₂, ‖Xᵀ1 − b‖₂, ‖min(X, 0)‖₂). The sparse
    plan of ``plan_metrics`` indexes grid cells, hence this routine."""
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    vals = np.asarray(coo.data, dtype=np.float64)
    diff = P[rows] - Q[cols]
    cost_k = (diff * diff).sum(axis=1) / float(c_max)
    objective = float((vals * cost_k).sum())
    row_sum = np.bincount(rows, weights=vals, minlength=P.shape[0])
    col_sum = np.bincount(cols, weights=vals, minlength=Q.shape[0])
    row_err = float(np.linalg.norm(row_sum - a))
    col_err = float(np.linalg.norm(col_sum - b))
    neg = float(np.linalg.norm(np.minimum(vals, 0.0)))
    r_P = row_err + col_err + neg
    return objective, r_P, row_err, col_err, neg


def _halo_feaserr(row_err: float, col_err: float, r: np.ndarray, c: np.ndarray) -> float:
    """HALO's own measure of the transport constraints, ‖[X1 − r; Xᵀ1 − c]‖₂ / (1 + ‖(r; c)‖₂).

    HALO hands cuPDLPx the dual LP, so cuPDLPx reports this as its relative
    dual residual and HALO's per-level stop tests it as ``dual_feas``. Here it
    is measured on the delivered plan."""
    b_norm = math.hypot(float(np.linalg.norm(r)), float(np.linalg.norm(c)))
    return math.hypot(row_err, col_err) / (1.0 + b_norm)


def _halo_lp_form(out: dict, key: str):
    """The LP form (``primal`` or ``dual``) of HALO's last LP solve, from its log."""
    try:
        diag = (out.get(key) or [{}])[-1]
        return json_safe(diag.get("solver_lp_form", diag.get("tree_lp_form")))
    except Exception:                                       # noqa: BLE001
        return None


def _run_tree(spec: InstanceSpec, *, engine: str) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    if spec.norm not in _TREE_COST_BY_NORM:
        return _make_failure_record(
            spec, status="error", error_kind=f"halo_{spec.norm}_unsupported",
            error_msg=("HALO runs on sql2 only (squared Euclidean, l2^2); the "
                       f"Point Cloud cost {spec.norm!r} is not scheduled for it "
                       "(src/protocol/regimes.py)."),
            started_utc=started_utc, run_uid=run_uid, engine=engine)
    if engine != "cupdlpx":
        return _make_failure_record(
            spec, status="error", error_kind="halo_tree_engine_unsupported",
            error_msg=(f"HALO's tree mode always solves its LPs with cuPDLPx; "
                       f"--engine {engine} applies to grid mode only."),
            started_utc=started_utc, run_uid=run_uid, engine=engine)
    try:
        import pycupdlpx  # noqa: F401
    except ImportError as e:
        return _make_failure_record(
            spec, status="error", error_kind="halo_pycupdlpx_missing",
            error_msg=f"{e!r}; build with vendor/halo/build.sh.",
            started_utc=started_utc, run_uid=run_uid, engine=engine)
    if not torch.cuda.is_available():
        return _make_failure_record(
            spec, status="error", error_kind="no_cuda",
            error_msg="HALO tree mode needs CUDA (cupy / pycupdlpx kernels).",
            started_utc=started_utc, run_uid=run_uid, engine=engine)

    _pareto = getattr(spec, "pareto", False)
    P, Q, a, b = _load_clouds(spec)
    m, n, d = P.shape[0], Q.shape[0], P.shape[1]
    c_max = float(spec.c_max)
    # Scaled so that HALO's ‖p_i − q_j‖² is the max-1 cost and its tolerance
    # applies to the normalised problem.
    inv = 1.0 / math.sqrt(c_max)
    Ps = (P * inv).astype(np.float32)
    Qs = (Q * inv).astype(np.float32)
    cfg = _make_tree_config(spec.tolerance, pareto=_pareto)

    if spec.warmup > 0:
        _tree_warmup(d, pareto=_pareto)
        torch.cuda.synchronize()

    device_index = cuda_ordinal(spec.device)
    sampler = _NvmlPeakSampler(device_index)

    torch.cuda.synchronize()
    sampler.start()
    t0 = time.perf_counter()
    try:
        # HALO's tree builder divides ``a`` and ``b`` by their sums in place
        # (np.asarray of fp64 does not copy), so every metric below is measured
        # against r/Σr and c/Σc, bitwise what load_marginal_unit_mass returns.
        out = emd2(Ps, Qs, source_mass=a, target_mass=b,
                   config=cfg, log=True, return_coupling=True)
    except Exception as e:                                  # noqa: BLE001
        sampler.stop()
        if is_oom_exception(e):
            return _make_failure_record(
                spec, status="oom", error_kind="cuda_oom_solve",
                error_msg=f"{e!r}\n{traceback.format_exc()[-2000:]}",
                started_utc=started_utc, run_uid=run_uid, engine=engine)
        raise
    torch.cuda.synchronize()
    wall_solve = time.perf_counter() - t0
    peak_mem, baseline_mem = sampler.stop()

    if not isinstance(out, dict):
        return _make_failure_record(
            spec, status="error", error_kind="halo_invalid_result",
            error_msg=f"emd2(log=True) returned {type(out)!r}, expected dict",
            started_utc=started_utc, run_uid=run_uid, engine=engine)
    distance = out.get("distance")
    coupling = out.get("sparse_coupling")
    level_summaries = out.get("level_summaries") or []
    if (coupling is None or getattr(coupling, "nnz", 0) == 0
            or distance is None or not math.isfinite(float(distance))):
        return _make_failure_record(
            spec, status="error", error_kind="halo_invalid_result",
            error_msg=(f"distance={distance!r}, coupling="
                       f"{type(coupling).__name__} nnz={getattr(coupling,'nnz',None)!r}"),
            started_utc=started_utc, run_uid=run_uid, engine=engine)

    # Recovery of (X, α, β), timed into t_total. The duals need no rescaling:
    # HALO solved the max-1 cost.
    t2 = time.perf_counter()
    _tr0 = time.perf_counter()
    coo = coupling.tocoo()
    _du = out.get("dual_source") if _pareto else None
    _alpha_h = _beta_h = None
    if _pareto and _du is not None:
        _du = np.asarray(_du, dtype=np.float64).reshape(-1)
        if _du.size >= m + n:
            _alpha_h = torch.from_numpy(np.ascontiguousarray(_du[:m]))
            _beta_h = torch.from_numpy(np.ascontiguousarray(_du[m:m + n]))
    wall_recovery = time.perf_counter() - _tr0
    objective, r_P, row_err, col_err, negpart = _geo_metrics_from_coo(
        coo, P, Q, a, b, c_max)
    feaserr = _halo_feaserr(row_err, col_err, a, b)
    wall_eval = time.perf_counter() - t2

    pareto_block = None
    c_max_ratio = None
    if _pareto:
        _te0 = time.perf_counter()
        from src.methods import geo_cost as _geo_cost
        from src.protocol.instance_record import geo_cost_type as _gct
        from src.scoring.pareto_score import sanitize_block as _sanitize
        from src.scoring.pareto_score import score_symbolic
        _dev = torch.device(spec.device if torch.cuda.is_available() else "cpu")
        _ct = _gct(spec.norm)
        _Pt = torch.as_tensor(P, dtype=torch.float64, device=_dev)
        _Qt = torch.as_tensor(Q, dtype=torch.float64, device=_dev)
        # A ratio other than 1 means spec.c_max is not the exact maximum, so
        # HALO solved a rescaled cost.
        c_max_ratio = _geo_cost.exact_c_max(_Pt, _Qt, _ct) / c_max
        _cost_fn = _geo_cost.make_metric_cost_fn(_Pt, _Qt, _ct, c_max)
        _rt = torch.as_tensor(a, dtype=torch.float64, device=_dev)
        _ct_t = torch.as_tensor(b, dtype=torch.float64, device=_dev)
        if _alpha_h is None:
            pareto_block = _sanitize({
                "dual_absent": True, "pareto_partial": True,
                "reason": "HALO returned no/short dual_source",
                "t_solve": float(wall_solve),
                "t_total": float(wall_solve + wall_recovery), "t_eval": None,
                "t_solve_boundary": T_SOLVE_BOUNDARY_TREE,
                "t_solve_includes": list(T_SOLVE_INCLUDES),
                "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                "t_total_boundary": T_TOTAL_BOUNDARY_TREE,
                "t_total_includes": list(T_TOTAL_INCLUDES),
                "precision": "fp32", "halo_dual_scale": 1.0,
                "native_stop": _native_stop_label(cfg)})
        else:
            _blk = score_symbolic(
                p_obj=objective, r_P_row=row_err, r_P_col=col_err,
                r_P_neg=negpart, alpha=_alpha_h.to(_dev), beta=_beta_h.to(_dev),
                r=_rt, c=_ct_t, cost_fn=_cost_fn, m=m, n=n)
            pareto_block = _sanitize({
                **_blk,
                "t_solve": float(wall_solve),
                "t_total": float(wall_solve + wall_recovery),
                "t_eval": float(time.perf_counter() - _te0),
                "t_solve_boundary": T_SOLVE_BOUNDARY_TREE,
                "t_solve_includes": list(T_SOLVE_INCLUDES),
                "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                "t_total_boundary": T_TOTAL_BOUNDARY_TREE,
                "t_total_includes": list(T_TOTAL_INCLUDES),
                "pareto_partial": False, "precision": "fp32",
                "halo_dual_scale": 1.0,
                "c_max_exact_ratio": float(c_max_ratio),
                "native_stop": _native_stop_label(cfg),
            })

    iters_total = sum(int(row.get("iters") or 0) for row in level_summaries)
    native_term = (str(level_summaries[-1].get("stop_reason"))
                   if level_summaries else None)
    stripped_levels = [
        {k: json_safe(row.get(k)) for k in
         ("level", "iters", "support_final", "support_pre_lp_final", "stop_reason")
         if k in row} for row in level_summaries]
    fp = device_fingerprint(idx=device_index)

    return ResultRecord(
        instance_id=spec.instance_id, algorithm="HALO",
        norm=spec.norm, resolution=None, n=m,
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status="ok",
        wall_time_s=wall_solve + wall_eval, wall_time_solve_s=wall_solve,
        wall_time_recovery_s=wall_recovery,
        iters=iters_total, iter_cap_hit=False, native_termination=native_term,
        tolerance=spec.tolerance, feaserr=feaserr,
        native_duality_gap=None, objective=objective,
        peak_gpu_mem_bytes=peak_mem,
        peak_gpu_mem_source=("nvml_poll" if peak_mem is not None else "unknown"),
        device_name=fp.get("device_name"), device_uuid=fp.get("device_uuid"),
        dataset=spec.dataset, rung="PointCloud", m=m,
        algo_extras={
            "feaserr_measure": "halo_relative_primal_residual",
            "r_P": r_P,
            "halo_lp_form": _halo_lp_form(out, "tree_lp_diags"),
            "engine": engine, "halo_dtype": "fp32", "halo_mode": "tree",
            "halo_config": json_safe(dataclasses.asdict(cfg)),
            "obj_halo_native": float(distance),    # on the scaled coordinates, so ≈ objective
            "c_max": c_max, "m": m, "n": n, "dim": d,
            "marginal_floor": MARGINAL_FLOOR,
            "n_zero_mass_r": int((a == 0.0).sum()),
            "n_zero_mass_c": int((b == 0.0).sum()),
            **({"c_max_exact_ratio": float(c_max_ratio)}
               if c_max_ratio is not None else {}),
            "lp_solve_time_total": json_safe(out.get("lp_solve_time_total")),
            "level_summaries": stripped_levels,
            "active_support_nnz_final": int(coupling.nnz),
            "wall_time_eval_s": wall_eval,
            "nvml_baseline_bytes": baseline_mem,
            "t_solve_boundary": T_SOLVE_BOUNDARY_TREE,
            "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_total_boundary": T_TOTAL_BOUNDARY_TREE,
            "t_total_includes": list(T_TOTAL_INCLUDES),
            "row_sum_err_l2": row_err, "col_sum_err_l2": col_err,
            "plan_negpart_l2": negpart,
            **({"pareto": pareto_block} if pareto_block is not None else {}),
        },
    )


def _run_one(spec: InstanceSpec, *, engine: str = "cupdlpx") -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())

    if emd2 is None:
        return _make_failure_record(
            spec, status="error", error_kind="halo_import_failed",
            error_msg=(f"hierarchical_ot import failed ({_HALO_IMPORT_ERROR}); run "
                       "vendor/halo/fetch.sh && vendor/halo/build.sh first."),
            started_utc=started_utc, run_uid=run_uid, engine=engine)

    if spec.rung == "PointCloud":
        return _run_tree(spec, engine=engine)

    ng = spec.resolution
    n_total = ng * ng

    if GridConfig is None:
        return _make_failure_record(
            spec, status="error", error_kind="halo_import_failed",
            error_msg=(
                f"hierarchical_ot import failed ({_HALO_IMPORT_ERROR}); run "
                "vendor/halo/fetch.sh && vendor/halo/build.sh first."),
            started_utc=started_utc, run_uid=run_uid, engine=engine,
        )
    if spec.norm != "sql2":
        return _make_failure_record(
            spec, status="error", error_kind=f"halo_{spec.norm}_unsupported",
            error_msg=(
                "HALO's hierarchical grid mode hard-codes the separable "
                "squared-L₂ cost (cost_type='l2^2'); the protocol restricts "
                "HALO to sql2 (src/protocol/regimes.py)."),
            started_utc=started_utc, run_uid=run_uid, engine=engine,
        )
    if engine == "cupdlpx":
        try:
            import pycupdlpx  # noqa: F401
        except ImportError as e:
            return _make_failure_record(
                spec, status="error", error_kind="halo_pycupdlpx_missing",
                error_msg=(f"{e!r}; build the extension with "
                           "vendor/halo/build.sh."),
                started_utc=started_utc, run_uid=run_uid, engine=engine,
            )
        if not torch.cuda.is_available():
            return _make_failure_record(
                spec, status="error", error_kind="no_cuda",
                error_msg="torch.cuda.is_available() is False; "
                          "engine='cupdlpx' requires CUDA.",
                started_utc=started_utc, run_uid=run_uid, engine=engine,
            )

    r_flat, c_flat = _load_marginals(spec)
    r2d = r_flat.reshape(ng, ng)       # C order, inverting the instance builders' flatten()
    c2d = c_flat.reshape(ng, ng)
    cfg = _make_config(spec.tolerance, engine, pareto=getattr(spec, "pareto", False))

    if spec.warmup > 0:
        _warmup(ng, engine)
        if engine == "cupdlpx":
            torch.cuda.synchronize()

    device_index = cuda_ordinal(spec.device)
    sampler = _NvmlPeakSampler(device_index)

    if engine == "cupdlpx":
        torch.cuda.synchronize()
        sampler.start()
    t0 = time.perf_counter()
    try:
        out = emd2_grid(r2d, c2d, config=cfg, log=True, return_coupling=True)
    except Exception as e:                                  # noqa: BLE001
        sampler.stop()
        if is_oom_exception(e):
            return _make_failure_record(
                spec, status="oom", error_kind="cuda_oom_solve",
                error_msg=f"{e!r}\n{traceback.format_exc()[-2000:]}",
                started_utc=started_utc, run_uid=run_uid, engine=engine,
            )
        raise
    if engine == "cupdlpx":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak_mem, baseline_mem = sampler.stop()
    wall_solve = t1 - t0

    if not isinstance(out, dict):
        return _make_failure_record(
            spec, status="error", error_kind="halo_invalid_result",
            error_msg=f"emd2_grid(log=True) returned {type(out)!r}, expected dict",
            started_utc=started_utc, run_uid=run_uid, engine=engine,
        )
    distance = out.get("distance")
    coupling = out.get("sparse_coupling")
    level_summaries = out.get("level_summaries") or []
    if (coupling is None or getattr(coupling, "nnz", 0) == 0
            or distance is None or not math.isfinite(float(distance))):
        return _make_failure_record(
            spec, status="error", error_kind="halo_invalid_result",
            error_msg=(f"distance={distance!r}, coupling="
                       f"{type(coupling).__name__} nnz="
                       f"{getattr(coupling, 'nnz', None)!r}"),
            started_utc=started_utc, run_uid=run_uid, engine=engine,
        )

    # Recovery of (X, α, β), timed into t_total: the plan wrapper and, with
    # --pareto, α = s·u, β = s·v from dual_source = [u; v].
    _pareto = getattr(spec, "pareto", False)
    _du = out.get("dual_source") if _pareto else None
    _alpha_h = _beta_h = None
    _wall_recovery = 0.0
    t2 = time.perf_counter()
    _tr0 = time.perf_counter()
    rep = sparse_plan_from_scipy_coo(coupling.tocoo(), ng=ng, cost_type="sql2")
    if _pareto and _du is not None:
        _du = np.asarray(_du, dtype=np.float64).reshape(-1)
        if _du.size >= 2 * n_total:
            _s = _halo_obj_scale_factor(ng)
            _alpha_h = torch.from_numpy(_du[:n_total]).to(torch.float64) * _s
            _beta_h = torch.from_numpy(_du[n_total:2 * n_total]).to(torch.float64) * _s
    _wall_recovery = time.perf_counter() - _tr0
    r_t = torch.from_numpy(r_flat)
    c_t = torch.from_numpy(c_flat)
    metrics = evaluate_plan_metrics(rep, r_t, c_t)
    wall_eval = time.perf_counter() - t2

    pareto_block = None
    if _pareto:
        from src.scoring.pareto_score import (score as _pareto_score,
                                                        sanitize_block as _sanitize)
        if _alpha_h is None:
            pareto_block = {"dual_absent": True, "pareto_partial": True,
                            "reason": "HALO returned no/short dual_source",
                            "t_solve": float(wall_solve),
                            "t_total": float(wall_solve + _wall_recovery),
                            "t_eval": None, "precision": "fp32",
                            "t_solve_boundary": T_SOLVE_BOUNDARY_GRID,
                            "t_solve_includes": list(T_SOLVE_INCLUDES),
                            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                            "t_total_boundary": T_TOTAL_BOUNDARY_GRID,
                            "t_total_includes": list(T_TOTAL_INCLUDES),
                            "native_stop": _native_stop_label(cfg)}
        else:
            _te0 = time.perf_counter()
            _blk = _pareto_score(rep, _alpha_h, _beta_h, r_t, c_t,
                                 ng=ng, cost_type="sql2",
                                 x_nonneg_structural=False, compute_frobenius=True)
            pareto_block = _sanitize({
                **_blk,
                "t_solve": float(wall_solve),
                "t_total": float(wall_solve + _wall_recovery),
                "t_eval": float(time.perf_counter() - _te0),
                "t_solve_boundary": T_SOLVE_BOUNDARY_GRID,
                "t_solve_includes": list(T_SOLVE_INCLUDES),
                "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                "t_total_boundary": T_TOTAL_BOUNDARY_GRID,
                "t_total_includes": list(T_TOTAL_INCLUDES),
                "pareto_partial": False, "precision": "fp32",
                "halo_dual_scale": _halo_obj_scale_factor(ng),
                "native_stop": _native_stop_label(cfg),
            })

    iters_total = sum(int(row.get("iters") or 0) for row in level_summaries)
    native_term = (str(level_summaries[-1].get("stop_reason"))
                   if level_summaries else None)
    stripped_levels = [
        {k: json_safe(row.get(k))
         for k in ("level", "iters", "support_final", "support_pre_lp_final",
                   "stop_reason")
         if k in row}
        for row in level_summaries
    ]

    if engine == "cupdlpx":
        fp = device_fingerprint(idx=device_index)
        device_name = fp.get("device_name")
        device_uuid = fp.get("device_uuid")
    else:
        device_name = device_uuid = None

    return ResultRecord(
        instance_id=spec.instance_id, algorithm="HALO",
        norm=spec.norm, resolution=ng, n=n_total,
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status="ok",
        wall_time_s=wall_solve + wall_eval,
        wall_time_solve_s=wall_solve,
        wall_time_recovery_s=_wall_recovery,
        iters=iters_total,
        iter_cap_hit=False,                 # no iteration cap is imposed on HALO
        native_termination=native_term,
        tolerance=spec.tolerance,
        feaserr=_halo_feaserr(float(metrics["row_sum_err_l2"]),
                              float(metrics["col_sum_err_l2"]), r_flat, c_flat),
        native_duality_gap=None,            # HALO reports only its LPs' relative gaps
        objective=float(metrics["objective"]),
        peak_gpu_mem_bytes=peak_mem,
        peak_gpu_mem_source=("nvml_poll" if peak_mem is not None else "unknown"),
        device_name=device_name,
        device_uuid=device_uuid,
        algo_extras={
            "feaserr_measure": "halo_relative_primal_residual",
            "r_P": float(metrics["feaserr"]),
            "halo_lp_form": _halo_lp_form(out, "grid_lp_diags"),
            "engine": engine,
            "halo_dtype": "fp32",
            "halo_config": json_safe(dataclasses.asdict(cfg)),
            "obj_halo_native": float(distance),
            "halo_obj_scale_factor": _halo_obj_scale_factor(ng),
            "lp_solve_time_total": json_safe(out.get("lp_solve_time_total")),
            "level_summaries": stripped_levels,
            "active_support_nnz_final": int(coupling.nnz),
            "wall_time_eval_s": wall_eval,
            "nvml_baseline_bytes": baseline_mem,
            "t_solve_boundary": T_SOLVE_BOUNDARY_GRID,
            "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_total_boundary": T_TOTAL_BOUNDARY_GRID,
            "t_total_includes": list(T_TOTAL_INCLUDES),
            "row_sum_err_l2": float(metrics["row_sum_err_l2"]),
            "col_sum_err_l2": float(metrics["col_sum_err_l2"]),
            "r_P_negpart_assumed_zero": not metrics["negpart_measured"],
            "marginal_floor": MARGINAL_FLOOR,
            "n_zero_mass_r": int((r_flat == 0.0).sum()),
            "n_zero_mass_c": int((c_flat == 0.0).sum()),
            **({"pareto": pareto_block} if pareto_block is not None else {}),
        },
    )


def main(argv=None):
    def add_args(p):
        p.add_argument("--engine", choices=("cupdlpx", "scipy"), default="cupdlpx",
                       help="HALO LP engine; 'scipy' is the CPU debug path of grid "
                            "mode (tree mode refuses it).")
    return runner_main(
        argv, description=("HALO runner (sql2 only: grid mode on a Grid instance, "
                           "tree mode on a Point Cloud instance)."), add_args=add_args,
        run=lambda spec, args: _run_one(spec, engine=args.engine),
        failure=lambda spec, args, **kw: _make_failure_record(spec, engine=args.engine, **kw))


if __name__ == "__main__":
    sys.exit(main())
