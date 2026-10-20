"""
Runner for POT's log-domain Sinkhorn (``ot.bregman.sinkhorn_log``) on a dense
cost C, one algorithm name ``Sinkhorn_e<slug>`` per entropic weight ε.

Sinkhorn solves min ⟨C, X⟩ + ε·Σ X_ij log X_ij over the transport polytope,
not the LP. Its triple is the returned plan T and the potentials
(α, β) = (ε·log u, ε·log v), so T_ij = exp((α_i + β_j − C_ij)/ε), and the LP
gap g of the triple tends to ε·H(T) > 0 (H the Shannon entropy), not to 0.
"""

import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.protocol.instance_record import InstanceSpec, ResultRecord, utc_now_iso
from src.scoring.plan_metrics import DenseTorchPlan, evaluate_plan_metrics
from src.scoring.kkt_residual import support_dot
from src.runners.gpu_state import device_fingerprint
from src.instances.dense_cost import resolve_dense_cost
from src.protocol.regimes import (
    DENSE_ONLY_NORMS, GENERAL, GRID, SINKHORN_PREFIX, eps_slug, eps_variant_name,
)
from src.runners.common import (
    failure_record,
    load_marginal_unit_mass,
    runner_main,
    warmup_marginal,
)


# sinkhorn_log's signature defaults, the stop under --pareto (checked against
# the installed POT at solve time). POT computes err only when ii % 10 == 0
# (ii the 0-based iteration), so log["err"] has one entry per 10 iterations.
POT_DEFAULT_NUM_ITER_MAX = 1000
POT_DEFAULT_STOP_THR = 1e-9
POT_ERR_CHECK_EVERY = 10

# Preemptive OOM from N = 256² rows, tested on the marginal length so that it
# also covers instances with no lattice. An N × N fp64 array is 34 GB there,
# and M, −M/ε and POT's log-sum-exp temporaries are live together.
SKIP_RESOLUTION_GE = 256
SKIP_MARGINAL_LEN_GE = SKIP_RESOLUTION_GE * SKIP_RESOLUTION_GE

DEFAULT_SINKHORN_METHOD = "sinkhorn_log"

# t_solve is the sinkhorn_log call alone: POT forms −M/ε once, iterates and
# computes err inside it. t_total adds the potential lift
# (α, β) = (ε·log u, ε·log v), the whole recovery, since POT returns T itself.
T_SOLVE_BOUNDARY = "pot_sinkhorn_log_call"
T_SOLVE_INCLUDES: Tuple[str, ...] = ()
T_SOLVE_EXCLUDES = ("warmup", "marginals_load_h2d", "cost_matrix_load_h2d")
T_TOTAL_BOUNDARY = "pot_sinkhorn_log_plus_potential_lift"
T_TOTAL_INCLUDES: Tuple[str, ...] = ()


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str,
                         algorithm: str, epsilon: float,
                         algo_extras: Optional[Dict[str, Any]] = None) -> ResultRecord:
    extras: Dict[str, Any] = {
        "epsilon": epsilon,
        "eps_slug": eps_slug(epsilon),
        "method": DEFAULT_SINKHORN_METHOD,
    }
    if algo_extras:
        extras.update(algo_extras)
    return failure_record(spec, algorithm, status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, m=spec.dense_shape()[0],
                          algo_extras=extras)


def _load_marginals(spec: InstanceSpec, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """(r, c) divided by their sums and never floored, as fp64 tensors."""
    r = load_marginal_unit_mass(spec.marginal_i_path)
    c = load_marginal_unit_mass(spec.marginal_j_path)
    return (
        torch.as_tensor(r, dtype=torch.float64, device=device),
        torch.as_tensor(c, dtype=torch.float64, device=device),
    )


def _count_zero_mass(v: torch.Tensor) -> int:
    """Coordinates of mass ≤ 0. POT takes the log of each marginal as is, so a
    zero-mass coordinate gets a −inf potential and an all-zero plan row (or
    column)."""
    return int((v <= 0).sum().item())


def _count_neg_inf(v: Optional[torch.Tensor]) -> Optional[int]:
    if v is None:
        return None
    return int(torch.isneginf(v).sum().item())


def _pot_stop_params(spec: InstanceSpec) -> Tuple[float, int]:
    """``(stopThr, numItermax)`` the solve ran with, from which the record's
    termination fields are derived: under ``--pareto`` POT never sees
    ``spec.tolerance`` or ``spec.iter_cap``."""
    if getattr(spec, "pareto", False):
        return POT_DEFAULT_STOP_THR, POT_DEFAULT_NUM_ITER_MAX
    return float(spec.tolerance), int(spec.iter_cap)


def _iters_done(sk_log: Any, err_history: list) -> Tuple[int, Optional[int], str]:
    """``(iters_done, niter_raw_0based, source)``. POT logs ``niter`` as the
    0-based index of the last executed iteration, so iters_done = niter + 1."""
    niter_raw: Optional[int] = None
    if isinstance(sk_log, dict):
        for key in ("niter", "n_iter", "n_outer"):
            if key in sk_log and sk_log[key] is not None:
                try:
                    niter_raw = int(sk_log[key])
                    break
                except (TypeError, ValueError):
                    continue
    if niter_raw is not None:
        return niter_raw + 1, niter_raw, "log_niter"
    # Without niter: e err entries (at ii = 0, 10, ...) mean between
    # 10(e − 1) + 1 and 10e executed iterations, so 10e is an upper bound.
    return (POT_ERR_CHECK_EVERY * len(err_history) if err_history else 0,
            None, "err_history_upper_bound")


def _native_termination(err_last: Optional[float], iters_done: int,
                        stop_thr: float, num_iter_max: int) -> Tuple[str, bool]:
    """``(native_termination, iter_cap_hit)``. POT breaks on err < stopThr
    (strict), else runs all ``numItermax`` iterations; ``err_last`` is err at the
    last checked iterate."""
    cap_hit = iters_done >= int(num_iter_max)
    if err_last is not None and err_last < float(stop_thr):
        return "stop_thr_below_tol", cap_hit
    return ("max_iters" if cap_hit else "unknown"), cap_hit


def _load_cost_matrix(spec: InstanceSpec, device: torch.device) -> Tuple[torch.Tensor, str]:
    """``(M, cost_source)``: the dense cost from ``resolve_dense_cost`` and its
    source, ``"cached_npy"``, ``"procedural"`` or ``"constructed_grid"``."""
    cached = bool(spec.cost_matrix_path and Path(spec.cost_matrix_path).exists())
    M = resolve_dense_cost(spec, device=device)
    if M.ndim != 2:
        raise ValueError(f"Cost matrix has unexpected shape {tuple(M.shape)}")
    src = ("cached_npy" if cached
           else ("procedural" if spec.cost_seed is not None else "constructed_grid"))
    return M, src


def _warmup(device: torch.device, epsilon: float = 0.01) -> None:
    """An untimed solve on a 32 × 32 l1 lattice with random marginals, paying
    POT's and CUDA's first-call costs."""
    import ot  # noqa: F401
    from ot.bregman import sinkhorn_log

    n = 32
    rng = np.random.default_rng(20260429)
    a = warmup_marginal(rng, n * n)
    b = warmup_marginal(rng, n * n)

    coords = np.arange(n)
    rr, cc = np.meshgrid(coords, coords, indexing="ij")
    flat_r = rr.flatten()
    flat_c = cc.flatten()
    dr = flat_r[:, None] - flat_r[None, :]
    dc = flat_c[:, None] - flat_c[None, :]
    M_np = (np.abs(dr) + np.abs(dc)).astype(np.float64)
    M_np /= max(1.0, float(M_np.max()))

    a_t = torch.as_tensor(a, dtype=torch.float64, device=device)
    b_t = torch.as_tensor(b, dtype=torch.float64, device=device)
    M_t = torch.as_tensor(M_np, dtype=torch.float64, device=device)
    try:
        sinkhorn_log(a_t, b_t, M_t, reg=epsilon, numItermax=200,
                     stopThr=1e-3, log=False, verbose=False)
    except Exception:                                       # noqa: BLE001
        pass
    torch.cuda.synchronize(device)


def _compute_entropic_dual_gap(T: torch.Tensor, M: torch.Tensor,
                                a: torch.Tensor, b: torch.Tensor,
                                u: Optional[torch.Tensor], v: Optional[torch.Tensor],
                                epsilon: float) -> Optional[float]:
    """⟨C, T⟩ + ε·Σ T_ij(log T_ij − 1) − rᵀα − cᵀβ, a diagnostic; ``u``, ``v``
    are the potentials α, β, not POT's scalings.

    It omits the term ε·Σ T_ij of the entropic dual objective, so at a
    converged Sinkhorn point it equals −ε·Σ T_ij = −ε, not 0. None when a
    potential is missing, the value is not finite, or on any exception.
    """
    if u is None or v is None:
        return None
    try:
        # The clamp makes 0·log 0 = 0.
        T_safe = T.clamp_min(1e-300)
        H = float((T * (T_safe.log() - 1.0)).sum().item())
        primal_unreg = float((M * T).sum().item())
        primal_reg = primal_unreg + epsilon * H
        # Zero-mass coordinates carry −inf potentials: 0·(−inf) = 0, not NaN.
        dual = support_dot(u, a) + support_dot(v, b)
        gap = primal_reg - dual
        if not np.isfinite(gap):
            return None
        return gap
    except Exception:                                       # noqa: BLE001
        return None


def _run_one(spec: InstanceSpec, *, epsilon: float) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    algorithm = eps_variant_name(SINKHORN_PREFIX, epsilon)
    n_total = spec.dense_shape()[0]

    if n_total >= SKIP_MARGINAL_LEN_GE:
        return _make_failure_record(
            spec, status="oom", error_kind="sinkhorn_dense_M_too_large",
            error_msg=(
                f"Sinkhorn at N = {n_total} skipped preemptively. "
                f"POT requires a dense N×N cost matrix; that is "
                f"{(n_total ** 2 * 8) / 1e9:.0f} GB fp64, "
                f"plus another {(n_total ** 2 * 8) / 1e9:.0f} GB for the "
                f"plan T. Won't fit on 80 GB GPUs."
            ),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )

    if not torch.cuda.is_available():
        return _make_failure_record(
            spec, status="error", error_kind="no_cuda",
            error_msg="torch.cuda.is_available() is False; Sinkhorn runner requires CUDA.",
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )

    # Imported here so that a missing POT gives a record, not an import error.
    try:
        import ot                                            # noqa: F401
        from ot.bregman import sinkhorn_log
        pot_version = ot.__version__
    except ImportError as e:
        return _make_failure_record(
            spec, status="error", error_kind="pot_import_failed",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )

    device = torch.device(spec.device if spec.device.startswith("cuda") else "cuda:0")

    if spec.warmup > 0:
        try:
            _warmup(device, epsilon=0.01)
        except torch.cuda.OutOfMemoryError as e:
            return _make_failure_record(
                spec, status="oom", error_kind="cuda_oom_warmup",
                error_msg=repr(e),
                started_utc=started_utc, run_uid=run_uid,
                algorithm=algorithm, epsilon=epsilon,
            )
        torch.cuda.synchronize(device)

    try:
        a_dev, b_dev = _load_marginals(spec, device)
    except (FileNotFoundError, ValueError) as e:
        return _make_failure_record(
            spec, status="error", error_kind="marginal_load_failed",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )
    n_zero_r = _count_zero_mass(a_dev)
    n_zero_c = _count_zero_mass(b_dev)
    try:
        M_dev, cost_source = _load_cost_matrix(spec, device)
    except FileNotFoundError as e:
        return _make_failure_record(
            spec, status="error", error_kind="cost_matrix_missing",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_cost_load",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )
    except (MemoryError, ValueError) as e:
        return _make_failure_record(
            spec, status="error", error_kind="cost_matrix_load_failed",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )

    torch.cuda.reset_peak_memory_stats(device)
    mem_before = int(torch.cuda.memory_allocated(device))
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    try:
        _sk_kw = dict(reg=float(epsilon), log=True, verbose=False)
        if not getattr(spec, "pareto", False):
            _sk_kw.update(numItermax=int(spec.iter_cap),
                          stopThr=float(spec.tolerance))
        else:
            # The record's native_stop names these defaults; raise if the
            # installed POT has others.
            import inspect as _inspect
            _d = _inspect.signature(sinkhorn_log).parameters
            _ni = _d.get("numItermax"); _st = _d.get("stopThr")
            if not (_ni and _ni.default == POT_DEFAULT_NUM_ITER_MAX
                    and _st and _st.default == POT_DEFAULT_STOP_THR):
                raise RuntimeError(
                    f"POT sinkhorn_log as-shipped defaults changed: "
                    f"numItermax={_ni.default if _ni else '?'}, "
                    f"stopThr={_st.default if _st else '?'} "
                    f"(expected {POT_DEFAULT_NUM_ITER_MAX}, {POT_DEFAULT_STOP_THR})")
        T, sk_log = sinkhorn_log(a_dev, b_dev, M_dev, **_sk_kw)
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_solve",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
            algo_extras={"pot_version": pot_version},
        )
    except Exception as e:                                  # noqa: BLE001
        return _make_failure_record(
            spec, status="error", error_kind="sinkhorn_solve_failed",
            error_msg=f"{e!r}\n{traceback.format_exc()[:3500]}",
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
            algo_extras={"pot_version": pot_version},
        )
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    wall_solve = t1 - t0

    # Catch a silent CPU fallback of POT's backend (it follows the input types).
    pot_backend = "torch_cuda"
    if not isinstance(T, torch.Tensor):
        try:
            T = torch.as_tensor(np.asarray(T), dtype=torch.float64, device=device)
        except Exception as e:                              # noqa: BLE001
            return _make_failure_record(
                spec, status="error", error_kind="sinkhorn_result_not_convertible",
                error_msg=repr(e),
                started_utc=started_utc, run_uid=run_uid,
                algorithm=algorithm, epsilon=epsilon,
                algo_extras={"pot_version": pot_version, "pot_backend": "unknown"},
            )
        pot_backend = "cpu_fallback"
    elif T.device.type != "cuda":
        T = T.to(device=device, dtype=torch.float64)
        pot_backend = "cpu_fallback"
    elif T.dtype != torch.float64:
        T = T.to(dtype=torch.float64)

    err_history = sk_log.get("err") if isinstance(sk_log, dict) else None
    if err_history is None:
        err_history = []
    elif isinstance(err_history, torch.Tensor):
        err_history = err_history.detach().cpu().tolist()
    elif isinstance(err_history, np.ndarray):
        err_history = err_history.tolist()
    err_first = float(err_history[0]) if err_history else None
    err_last  = float(err_history[-1]) if err_history else None

    pot_stop_thr, pot_num_iter_max = _pot_stop_params(spec)
    niter, niter_raw, iters_source = _iters_done(sk_log, err_history)
    native_termination, iter_cap_hit = _native_termination(
        err_last, niter, pot_stop_thr, pot_num_iter_max)

    # POT reports no LP duality gap.
    native_duality_gap = None

    # (α, β) = (ε·log u, ε·log v), from the log-scalings log u, log v, or from
    # the scalings u, v when only those are logged. Timed into t_total.
    torch.cuda.synchronize(device)
    _tr0 = time.perf_counter()
    raw_lu = sk_log.get("log_u") if isinstance(sk_log, dict) else None
    raw_lv = sk_log.get("log_v") if isinstance(sk_log, dict) else None
    pot_u = pot_v = None
    if isinstance(raw_lu, torch.Tensor) and isinstance(raw_lv, torch.Tensor):
        pot_u = epsilon * raw_lu.to(torch.float64)
        pot_v = epsilon * raw_lv.to(torch.float64)
    elif isinstance(sk_log, dict) and "u" in sk_log:
        u_raw = sk_log["u"]; v_raw = sk_log.get("v")
        if isinstance(u_raw, torch.Tensor) and isinstance(v_raw, torch.Tensor):
            pot_u = epsilon * u_raw.clamp_min(1e-300).log().to(torch.float64)
            pot_v = epsilon * v_raw.clamp_min(1e-300).log().to(torch.float64)
    torch.cuda.synchronize(device)
    wall_recovery = time.perf_counter() - _tr0
    if pot_u is not None:
        entropic_gap = _compute_entropic_dual_gap(T, M_dev, a_dev, b_dev,
                                                  pot_u, pot_v, epsilon)
    else:
        entropic_gap = None

    # The last u-update makes the row sums of T equal r, so
    # α_i + β_j − C_ij = ε·log T_ij ≤ ε·log r_i ≤ 0: r_D is 0 up to rounding,
    # and g tends to ε·H(T), the entropic bias, not a scoring error.
    pareto_block = None
    if getattr(spec, "pareto", False):
        if pot_u is None:
            # A flagged block, so the row counts as unscorable instead of
            # silently lacking one.
            pareto_block = {"pareto_absent": True, "pareto_partial": True,
                            "reason": "POT returned no log_u/log_v potentials",
                            "t_solve": float(wall_solve), "t_total": None, "t_eval": None,
                            "t_solve_boundary": T_SOLVE_BOUNDARY,
                            "t_solve_includes": list(T_SOLVE_INCLUDES),
                            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                            "t_total_boundary": T_TOTAL_BOUNDARY,
                            "t_total_includes": list(T_TOTAL_INCLUDES),
                            "native_stop": "pot_default_stopThr_1e-9_numItermax_1000"}
        else:
            from src.scoring.pareto_score import (score as _pareto_score,
                                                            sanitize_block as _sanitize)
            torch.cuda.synchronize(device)
            _te0 = time.perf_counter()
            _blk = _pareto_score(T, pot_u, pot_v, a_dev, b_dev, C=M_dev,
                                 # A baseline's ‖min(X, 0)‖ is measured, not declared 0.
                                 x_nonneg_structural=False, compute_frobenius=True)
            torch.cuda.synchronize(device)
            pareto_block = _sanitize({
                **_blk,
                "t_solve": float(wall_solve),
                "t_total": float(wall_solve + wall_recovery),
                "t_eval": float(time.perf_counter() - _te0),
                "t_solve_boundary": T_SOLVE_BOUNDARY,
                "t_solve_includes": list(T_SOLVE_INCLUDES),
                "t_solve_excludes": list(T_SOLVE_EXCLUDES),
                "t_total_boundary": T_TOTAL_BOUNDARY,
                "t_total_includes": list(T_TOTAL_INCLUDES),
                "t_dual_recovery": float(wall_recovery),
                "pareto_partial": False, "precision": "fp64",
                "native_stop": "pot_default_stopThr_1e-9_numItermax_1000",
                # A nonzero count means α (resp. β) is −inf there; d_obj is
                # summed over the support.
                "n_zero_mass_r": n_zero_r, "n_zero_mass_c": n_zero_c,
                "marginals": "original_unfloored",
            })

    # ⟨C, T⟩ is traced against M_dev except on a Grid l1/sql2 instance, where C
    # is rebuilt in row blocks from (n_g, norm). A non-Grid instance's cost is
    # not the lattice cost of (resolution, norm).
    _use_C = (spec.rung != GRID) or (spec.norm in DENSE_ONLY_NORMS)
    try:
        rep = DenseTorchPlan(X=T, ng=spec.resolution, cost_type=spec.norm,
                             C=M_dev if _use_C else None)
        torch.cuda.synchronize(device)
        t2 = time.perf_counter()
        metrics = evaluate_plan_metrics(rep, a_dev, b_dev)
        torch.cuda.synchronize(device)
        t3 = time.perf_counter()
        wall_eval = t3 - t2
    except torch.cuda.OutOfMemoryError as e:
        return _make_failure_record(
            spec, status="oom", error_kind="cuda_oom_eval",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
            algo_extras={"pot_version": pot_version, "pot_backend": pot_backend},
        )

    objective = float(metrics["objective"])
    row_err_l2 = float(metrics["row_sum_err_l2"])
    col_err_l2 = float(metrics["col_sum_err_l2"])
    # feaserr is POT's stop measure err = ‖Tᵀ1 − c‖₂ (the u-update comes last,
    # so the row sums are exact up to rounding), recomputed on the delivered T:
    # the logged err can belong to an iterate up to 9 iterations earlier.
    feaserr = col_err_l2

    peak_mem = int(torch.cuda.max_memory_allocated(device))
    fp = device_fingerprint(idx=device.index or 0)

    rec = ResultRecord(
        instance_id=spec.instance_id, algorithm=algorithm, regime=GENERAL,
        norm=spec.norm, resolution=spec.resolution, n=n_total, m=n_total,
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status="ok",
        wall_time_s=wall_solve + wall_recovery + wall_eval,
        wall_time_solve_s=wall_solve,
        wall_time_recovery_s=wall_recovery,
        iters=niter,
        iter_cap_hit=iter_cap_hit,
        native_termination=native_termination,
        tolerance=spec.tolerance,
        feaserr=feaserr,
        native_duality_gap=native_duality_gap,
        objective=objective,
        peak_gpu_mem_bytes=peak_mem,
        peak_gpu_mem_source="torch_cuda_max",
        device_name=fp.get("device_name"),
        device_uuid=fp.get("device_uuid"),
        algo_extras={
            "epsilon": float(epsilon),
            "eps_slug": eps_slug(epsilon),
            "method": DEFAULT_SINKHORN_METHOD,
            "cost_source": cost_source,
            "pot_version": pot_version,
            "pot_backend": pot_backend,
            "numiter": niter,
            "pot_niter_0based": niter_raw,
            "iters_source": iters_source,
            "pot_stop_thr": float(pot_stop_thr),
            "pot_num_iter_max": int(pot_num_iter_max),
            "err_history_first": err_first,
            "err_history_last": err_last,
            "entropic_dual_gap": entropic_gap,
            "marginals": "original_unfloored",
            "n_zero_mass_r": n_zero_r,
            "n_zero_mass_c": n_zero_c,
            "n_neg_inf_alpha": _count_neg_inf(pot_u),
            "n_neg_inf_beta": _count_neg_inf(pot_v),
            **({"pareto": pareto_block} if pareto_block is not None else {}),
            "wall_time_eval_s": wall_eval,
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_total_boundary": T_TOTAL_BOUNDARY,
            "t_total_includes": list(T_TOTAL_INCLUDES),
            "row_sum_err_l2": row_err_l2,
            "col_sum_err_l2": col_err_l2,
            "mem_before_solve_bytes": mem_before,
            "feaserr_measure": "pot_sinkhorn_err",
            "r_P": float(metrics["feaserr"]),
            "r_P_negpart_assumed_zero": not metrics["negpart_measured"],
        },
    )
    return rec


def main(argv=None) -> int:
    def add_args(p):
        p.add_argument("--epsilon", required=True, type=float,
                       help="Entropic regularization parameter ε > 0.")

    def check_args(args):
        if args.epsilon <= 0:
            return f"run_sinkhorn: --epsilon must be positive; got {args.epsilon}"
        return None

    return runner_main(
        argv, description="POT log-Sinkhorn runner.", add_args=add_args, check_args=check_args,
        run=lambda spec, args: _run_one(spec, epsilon=float(args.epsilon)),
        failure=lambda spec, args, **kw: _make_failure_record(
            spec, algorithm=eps_variant_name(SINKHORN_PREFIX, float(args.epsilon)),
            epsilon=float(args.epsilon), **kw))


if __name__ == "__main__":
    sys.exit(main())
