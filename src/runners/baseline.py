"""The extension point for a new baseline runner.

A baseline subclasses ``BaselineAdapter`` with what is specific to its solver;
``run_baseline`` runs the stages (gates, warmup, load, solve, after_solve,
recover, score, diagnostics, cleanup), keeps the clocks and returns the
ResultRecord. The warm-up runs before the instance is loaded, so it cannot see
the instance, and the driver,
not the adapter, loads the original unit-mass marginals and scores the
recovered (X, α, β) against them.

A stage from gates to recover may end the run with
``raise StageFailure(status, error_kind, message)``. Any other exception there,
or in scoring, gives ``oom`` with ``cuda_oom_<stage>`` for an allocation
failure (``oom_recovery`` in recover) and ``error`` with ``<stage>_failed``
otherwise, where ``<stage>`` is ``recovery`` for recover and ``eval`` for
scoring. An exception in ``diagnostics`` is recorded and the row keeps its
status.
"""

import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from src.protocol.instance_record import GRID, InstanceSpec, ResultRecord, utc_now_iso
from src.protocol.regimes import canonical_regime, regime_for_algorithm
from src.runners.common import (
    cuda_ordinal,
    failure_record,
    finite_or_none,
    is_oom_exception,
    json_safe,
    load_marginal_unit_mass,
    runner_main,
)

# Keys the driver owns in algo_extras and in the Pareto block; an adapter that
# sets one is refused.
_RESERVED_EXTRAS = frozenset({"pareto", "timeout_flag", "certificate_valid",
                              "native_stop", "precision", "scored_marginals",
                              "feaserr_measure", "r_P"})
_DRIVER_BLOCK_KEYS = frozenset({
    "r_P", "r_P_row", "r_P_col", "r_P_neg", "r_Dinf", "r_DF", "p_obj", "d_obj", "gap",
    "C_inf", "t_solve", "t_total", "t_eval", "t_load", "t_solve_bracket",
    "t_solve_boundary", "t_solve_includes", "t_solve_excludes", "t_total_boundary",
    "t_total_includes", "pareto_partial", "pareto_absent", "dual_absent", "precision",
    "native_stop", "native_stop_is_library_default", "native_stop_is_upstream_default",
    "x_nonneg_structural", "r_P_neg_source", "dual_certified"})


@dataclass(frozen=True)
class ClockProvenance:
    """What t_solve and t_total span, recorded in the Pareto block."""
    t_solve_boundary: str
    t_solve_includes: Tuple[str, ...]
    t_solve_excludes: Tuple[str, ...]
    t_total_boundary: str
    t_total_includes: Tuple[str, ...]


@dataclass(frozen=True)
class Refusal:
    """A gate's verdict not to run the instance: status ``error``, or ``oom``
    for a structural size limit."""
    status: str
    error_kind: str
    message: str

    def __post_init__(self):
        if self.status not in ("error", "oom"):
            raise ValueError(f"a refusal's status is error or oom, not {self.status!r}")


class StageFailure(Exception):
    """Raised by an adapter in a stage from gates to recover to end the run with this row."""

    def __init__(self, status: str, error_kind: str, message: str,
                 extras: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status, self.error_kind, self.message = status, error_kind, message
        self.extras = dict(extras or {})


@dataclass
class Problem:
    """One instance as the solver receives it.

    The cost is exactly one of: a lattice (``ng``, ``cost_type``), a stored
    ``C`` (m × n), or a block source ``cost_fn(i0, i1) -> C[i0:i1, :]`` with
    ``m`` and ``n``. ``assert_max1`` makes the scorer check max C = 1 on a
    lattice or a stored C. ``timings`` holds durations of parts of load for the
    Pareto block (e.g. ``t_cost_build``); ``extras`` is recorded on every row
    from load on. The driver sets ``deadline`` to the run's start plus
    ``spec.time_limit_s``, in ``time.perf_counter`` time."""
    ng: Optional[int] = None
    cost_type: Optional[str] = None
    C: Optional[torch.Tensor] = None
    cost_fn: Optional[Callable[[int, int], torch.Tensor]] = None
    m: Optional[int] = None
    n: Optional[int] = None
    assert_max1: bool = True
    payload: Any = None
    timings: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[float] = None

    def remaining_s(self, margin_s: float = 0.0) -> Optional[float]:
        if self.deadline is None:
            return None
        return self.deadline - time.perf_counter() - margin_s


@dataclass
class SolveResult:
    """What ``solve`` returns.

    ``t_solve`` is the solver's own clock when it reports one (one longer than
    the driver's bracket around the call fails the row); None means that
    bracket. ``t_recover_in_call`` is time inside the
    call, outside that clock, spent recovering the plan or duals; it counts
    toward t_total. ``time_limit_hit`` makes the row a timeout. ``native_stop``
    overrides the adapter's label for this run. ``feaserr`` is the solver's own
    primal feasibility measure and must be named by ``feaserr_measure`` (set
    without a name, it fails the row); with neither, the record's feaserr is
    r_P of the recovered plan."""
    raw: Any
    iters: Optional[int] = None
    native_termination: Optional[str] = None
    iter_cap_hit: bool = False
    time_limit_hit: bool = False
    t_solve: Optional[float] = None
    t_recover_in_call: float = 0.0
    native_stop: Optional[str] = None
    native_duality_gap: Optional[float] = None
    feaserr: Optional[float] = None
    feaserr_measure: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)
    pareto: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recovered:
    """The primal-dual triple for the original LP.

    ``plan`` is a dense m × n tensor or a ``scoring.plan_metrics`` plan
    representation; on a ``cost_fn`` problem it must be dense, or the row is
    ``error/plan_not_dense``. Without a dual, leave
    ``alpha`` and ``beta`` None and give ``dual_absent_reason``: the row is then
    ok but not a Pareto point. ``x_nonneg_structural`` declares X ≥ 0 by
    construction for a non-dense plan; a dense plan's negative part is always
    measured."""
    plan: Any
    alpha: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None
    dual_absent_reason: Optional[str] = None
    x_nonneg_structural: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)
    pareto: Dict[str, Any] = field(default_factory=dict)


class BaselineAdapter:
    """The solver-specific half of a baseline runner; ``check_adapter`` validates
    the class attributes a subclass sets.

    ``dual_certified`` is True when the delivered dual is feasible by
    construction; False makes the aggregation add the scoring time t_eval to the
    solver's time to certificate, and None leaves that to
    ``NEEDS_CERTIFICATE_PASS`` in ``report/aggregate.py``. ``memory`` is "torch" for the caching allocator's peak,
    or "nvml" for polled whole-device memory, for a solver that allocates
    outside torch.
    """
    algorithm: str = ""
    device_kind: str = "cuda"
    precision: str = "fp64"
    native_stop: str = ""
    native_stop_is_library_default: Optional[bool] = None
    native_stop_is_upstream_default: Optional[bool] = None
    dual_certified: Optional[bool] = None
    clocks: Optional[ClockProvenance] = None
    memory: str = "torch"

    def gates(self, spec: InstanceSpec) -> Optional[Refusal]:
        return None

    def warmup(self, shape: Tuple[int, int], spec: InstanceSpec, device: torch.device) -> None:
        """One untimed run on a dummy of this shape, before the instance is loaded."""

    def load(self, spec: InstanceSpec, device: torch.device) -> Problem:
        raise NotImplementedError

    def solve(self, problem: Problem, spec: InstanceSpec) -> SolveResult:
        raise NotImplementedError

    def after_solve(self, problem: Problem, result: SolveResult) -> None:
        """Untimed work between the solve and the recovery."""

    def recover(self, problem: Problem, result: SolveResult) -> Recovered:
        raise NotImplementedError

    def diagnostics(self, problem: Problem, result: SolveResult, recovered: Recovered,
                    block: Dict[str, Any]) -> Dict[str, Any]:
        """Untimed diagnostics recorded in algo_extras."""
        return {}

    def cleanup(self, problem: Optional[Problem]) -> None:
        """Release what load or solve created. Called on every path once the gates
        and the CUDA check have passed; ``problem`` is None if load did not return one."""

    def failure_extras(self, spec: InstanceSpec) -> Dict[str, Any]:
        """Configuration known before the run, recorded on every row."""
        return {}


def check_adapter(adapter: BaselineAdapter) -> None:
    """Refuse a malformed adapter before anything runs."""
    regime_for_algorithm(adapter.algorithm)          # raises on an unregistered name
    problems = []
    if adapter.device_kind not in ("cuda", "cpu"):
        problems.append(f"device_kind {adapter.device_kind!r}")
    if adapter.precision not in ("fp64", "fp32", "mixed"):
        problems.append(f"precision {adapter.precision!r}")
    if adapter.memory not in ("torch", "nvml"):
        problems.append(f"memory {adapter.memory!r}")
    if not adapter.native_stop:
        problems.append("native_stop is empty")
    if not isinstance(adapter.clocks, ClockProvenance):
        problems.append("clocks is not a ClockProvenance")
    for flag in ("dual_certified", "native_stop_is_library_default",
                 "native_stop_is_upstream_default"):
        if getattr(adapter, flag) not in (True, False, None):
            problems.append(f"{flag} {getattr(adapter, flag)!r} is not True, False or None")
    if problems:
        raise ValueError(f"adapter {type(adapter).__name__}: " + "; ".join(problems))


def _rows(spec: InstanceSpec) -> Optional[int]:
    """The instance's row count, or None for a spec that does not state it."""
    try:
        return int(spec.dense_shape()[0])
    except ValueError:
        return None


def _sync(device: torch.device) -> None:
    """Wait for queued device work; never creates a CUDA context."""
    if device.type == "cuda" and torch.cuda.is_initialized():
        torch.cuda.synchronize(device)


def _check_problem(spec: InstanceSpec, problem: Problem) -> Optional[str]:
    """Why ``problem`` does not describe ``spec``'s instance, or None."""
    forms = [problem.ng is not None or problem.cost_type is not None,
             problem.C is not None, problem.cost_fn is not None]
    if sum(forms) != 1:
        return "a Problem sets exactly one cost form: lattice, stored C or cost_fn"
    shape = spec.dense_shape()
    if forms[0]:
        if canonical_regime(spec.rung or GRID) != GRID or problem.ng != spec.resolution \
                or problem.cost_type != spec.norm or spec.norm not in ("l1", "sql2"):
            return (f"a lattice cost (ng={problem.ng}, {problem.cost_type}) does not "
                    f"describe this instance (rung {spec.rung}, n_g {spec.resolution}, "
                    f"norm {spec.norm}); the lattice scorer takes l1/sql2 on a Grid instance")
    elif forms[1]:
        if tuple(problem.C.shape) != tuple(shape):
            return f"stored C is {tuple(problem.C.shape)}, the instance is {tuple(shape)}"
    elif (problem.m, problem.n) != tuple(shape):
        return f"cost_fn is for {(problem.m, problem.n)}, the instance is {tuple(shape)}"
    return None


class _PeakMemory:
    """Peak device memory over the timed solve: the polled whole-device peak for
    ``memory = "nvml"`` when NVML answers, else the torch allocator's peak."""

    def __init__(self, kind: str, spec: InstanceSpec, device: torch.device):
        self.kind, self.device, self.sampler = kind, device, None
        if device.type == "cuda" and kind == "nvml":
            from src.runners.nvml_sampler import NvmlPeakSampler
            self.sampler = NvmlPeakSampler(cuda_ordinal(spec.device))

    def start(self) -> None:
        if self.device.type != "cuda":
            return
        torch.cuda.reset_peak_memory_stats(self.device)
        if self.sampler is not None:
            self.sampler.start()

    def stop(self) -> Tuple[Optional[int], Optional[str], Dict[str, Any]]:
        if self.device.type != "cuda":
            return None, None, {}
        torch_peak = int(torch.cuda.max_memory_allocated(self.device))
        extras: Dict[str, Any] = {"peak_gpu_mem_window": "solve",
                                  "torch_max_memory_allocated_bytes": torch_peak}
        if self.sampler is not None:
            peak, baseline = self.sampler.stop()
            extras["nvml_baseline_bytes"] = baseline
            if peak is not None:
                return peak, "nvml_poll", extras
        return torch_peak, "torch_cuda_max", extras


def _error_text(e: BaseException) -> str:
    return f"{e!r}\n{traceback.format_exc()}"


def run_baseline(spec: InstanceSpec, adapter: BaselineAdapter) -> ResultRecord:
    """Run ``adapter`` on ``spec`` through every stage; return the record."""
    check_adapter(adapter)
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    t_run0 = time.perf_counter()
    problem: Optional[Problem] = None
    kept: Dict[str, Any] = {}          # what a later failure row keeps of the run so far

    def fail(status: str, error_kind: str, error_msg: str,
             extras: Optional[Dict[str, Any]] = None, **fields) -> ResultRecord:
        algo_extras = {**adapter.failure_extras(spec),
                       **(problem.extras if problem is not None else {}),
                       **kept, **(extras or {})}
        return failure_record(spec, adapter.algorithm, status=status, error_kind=error_kind,
                              error_msg=error_msg, started_utc=started_utc, run_uid=run_uid,
                              m=_rows(spec), algo_extras=json_safe(algo_extras, nonfinite_to_none=True),
                              timeout_flag=(status == "timeout"), **fields)

    def stage(name: str, fn, oom_status: str = "oom"):
        """Run one stage; turn its failure into the row's (status, kind, msg, extras)."""
        try:
            return fn(), None
        except StageFailure as f:
            return None, (f.status, f.error_kind, f.message, f.extras)
        except Exception as e:                                   # noqa: BLE001
            if is_oom_exception(e):
                return None, (oom_status, f"cuda_oom_{name}", _error_text(e), {})
            return None, ("error", f"{name}_failed", _error_text(e), {})

    refusal, err = stage("gates", lambda: adapter.gates(spec))
    if err:
        return fail(*err[:3], extras=err[3])
    if refusal is not None:
        return fail(refusal.status, refusal.error_kind, refusal.message)
    if adapter.device_kind == "cuda":
        if not torch.cuda.is_available() or not str(spec.device).startswith("cuda"):
            return fail("error", "no_cuda", f"a CUDA baseline; spec.device {spec.device!r}, "
                        f"torch.cuda.is_available() {torch.cuda.is_available()}")
        device = torch.device(spec.device)
    else:
        device = torch.device("cpu")

    try:
        shape = spec.dense_shape()
        for _ in range(int(spec.warmup)):
            _, err = stage("warmup", lambda: adapter.warmup(shape, spec, device))
            if err:
                return fail(*err[:3], extras=err[3])

        t_l0 = time.perf_counter()
        problem, err = stage("load", lambda: adapter.load(spec, device))
        if err:
            return fail(*err[:3], extras=err[3])
        why = _check_problem(spec, problem)
        if why:
            return fail("error", "problem_mismatch", why)
        t_load = time.perf_counter() - t_l0
        problem.deadline = t_run0 + float(spec.time_limit_s)

        peak = _PeakMemory(adapter.memory, spec, device)
        _sync(device)
        peak.start()
        t0 = time.perf_counter()
        result, err = stage("solve", lambda: adapter.solve(problem, spec))
        _sync(device)
        t_bracket = time.perf_counter() - t0
        peak_bytes, peak_source, peak_extras = peak.stop()
        kept.update(peak_extras)
        mem = {"peak_gpu_mem_bytes": peak_bytes, "peak_gpu_mem_source": peak_source}
        if err:
            return fail(*err[:3], extras=err[3], wall_time_solve_s=t_bracket, **mem)
        if result.t_solve is not None and float(result.t_solve) > t_bracket * (1 + 1e-6) + 1e-3:
            return fail("error", "solver_clock_exceeds_bracket",
                        f"reported t_solve {result.t_solve} s > the call's {t_bracket} s", **mem)
        t_solve = float(result.t_solve) if result.t_solve is not None else t_bracket
        kept.update({"t_solve": t_solve, "t_solve_bracket": t_bracket, **result.extras})
        solved = dict(wall_time_solve_s=t_solve, iters=_int(result.iters),
                      native_termination=result.native_termination, **mem)
        if result.time_limit_hit or t_solve > float(spec.time_limit_s):
            kind = "native_time_limit" if result.time_limit_hit else "solve_over_roof"
            return fail("timeout", kind,
                        f"t_solve {t_solve:.1f} s, time_limit_s {spec.time_limit_s}",
                        wall_time_s=t_solve, **solved)

        _, err = stage("after_solve", lambda: adapter.after_solve(problem, result))
        if err:
            return fail(*err[:3], extras=err[3], **solved)
        t_r0 = time.perf_counter()
        rec, err = stage("recovery", lambda: adapter.recover(problem, result),
                         oom_status="oom_recovery")
        _sync(device)
        t_recover = time.perf_counter() - t_r0
        if err:
            return fail(*err[:3], extras=err[3], **solved)
        # The point-cloud scorer reads C only through cost_fn, so it needs the
        # entries of X; a plan representation traces its own cost instead.
        if problem.cost_fn is not None and not isinstance(rec.plan, torch.Tensor):
            return fail("error", "plan_not_dense",
                        f"a cost_fn problem is scored on a dense plan, not a "
                        f"{type(rec.plan).__name__}", **solved)

        t_e0 = time.perf_counter()
        scored, err = stage("eval", lambda: _score(spec, problem, rec))
        _sync(device)
        t_eval = time.perf_counter() - t_e0
        if err:
            return fail(*err[:3], extras=err[3], **solved)
        blk, r0, c0, r_P_neg_source = scored
        return _ok_row(spec, adapter, problem, result, rec, blk, r0, c0, r_P_neg_source,
                       started_utc=started_utc, run_uid=run_uid, t_load=t_load,
                       t_solve=t_solve, t_bracket=t_bracket,
                       t_total=t_solve + float(result.t_recover_in_call) + t_recover,
                       t_recover=t_recover, t_eval=t_eval, device=device, mem=mem,
                       peak_extras=peak_extras, fail=fail)
    finally:
        try:
            adapter.cleanup(problem)
        except Exception:                                         # noqa: BLE001
            pass


def _int(x: Any) -> Optional[int]:
    return None if x is None else int(x)


def _score(spec: InstanceSpec, problem: Problem, rec: Recovered):
    """The KKT components of the recovered triple against the instance's
    original unit-mass marginals, whatever the solver was given."""
    from src.scoring.pareto_score import score, score_symbolic
    dev = _score_device(problem, rec)
    r0 = torch.as_tensor(load_marginal_unit_mass(spec.marginal_i_path), dtype=torch.float64,
                         device=dev)
    c0 = torch.as_tensor(load_marginal_unit_mass(spec.marginal_j_path), dtype=torch.float64,
                         device=dev)
    dense = isinstance(rec.plan, torch.Tensor)
    structural = bool(rec.x_nonneg_structural) and not dense
    # With no dual, the primal terms are still measured: score with zero
    # potentials and drop every component that depends on the dual.
    alpha = rec.alpha if rec.alpha is not None else torch.zeros_like(r0)
    beta = rec.beta if rec.beta is not None else torch.zeros_like(c0)
    if problem.cost_fn is not None:             # the plan is dense (run_baseline checks)
        rs = rec.plan.to(torch.float64).sum(dim=1)
        cs = rec.plan.to(torch.float64).sum(dim=0)
        blk = score_symbolic(
            p_obj=_dense_trace(rec.plan, problem),
            r_P_row=float((rs - r0).norm().item()), r_P_col=float((cs - c0).norm().item()),
            r_P_neg=float(torch.clamp_min(-rec.plan, 0).norm().item()),
            alpha=alpha, beta=beta, r=r0, c=c0, cost_fn=problem.cost_fn,
            m=int(problem.m), n=int(problem.n))
    else:
        blk = score(rec.plan, alpha, beta, r0, c0, ng=problem.ng, cost_type=problem.cost_type,
                    C=problem.C, x_nonneg_structural=structural, compute_frobenius=True,
                    assert_max1=problem.assert_max1)
    if rec.alpha is None or rec.beta is None:
        blk = {k: (None if k in ("r_Dinf", "r_DF", "d_obj", "gap") else v) for k, v in blk.items()}
    return blk, r0, c0, ("structural" if structural else "measured")


def _score_device(problem: Problem, rec: Recovered) -> torch.device:
    """Where the scoring runs: the dual's device, else the plan's (a plan
    without a dual is still scored for its primal terms), else the cost's."""
    if rec.alpha is not None:
        return rec.alpha.device
    if isinstance(rec.plan, torch.Tensor):
        return rec.plan.device
    if rec.plan is not None:
        return rec.plan.row_sums().device
    return problem.C.device if problem.C is not None else torch.device("cpu")


def _dense_trace(X: torch.Tensor, problem: Problem) -> float:
    """<C, X> for a dense plan on a point-cloud cost, row block by row block."""
    m = int(problem.m)
    step = max(1, (32 * 1024 * 1024) // max(1, int(problem.n)))
    acc = torch.zeros((), dtype=torch.float64, device=X.device)
    for i0 in range(0, m, step):
        i1 = min(m, i0 + step)
        acc = acc + (problem.cost_fn(i0, i1).to(torch.float64) * X[i0:i1].to(torch.float64)).sum()
    return float(acc.item())


def _ok_row(spec, adapter, problem, result, rec, blk, r0, c0, r_P_neg_source, *,
            started_utc, run_uid, t_load, t_solve, t_bracket, t_total, t_recover, t_eval,
            device, mem, peak_extras, fail) -> ResultRecord:
    from src.scoring.pareto_score import sanitize_block
    ck = adapter.clocks
    dual_absent = rec.alpha is None or rec.beta is None
    adapter_block = {**problem.timings, **result.pareto, **rec.pareto}
    clash = sorted(set(adapter_block) & _DRIVER_BLOCK_KEYS)
    adapter_extras = {**problem.extras, **result.extras, **rec.extras}
    clash += sorted(set(adapter_extras) & _RESERVED_EXTRAS)
    if clash:
        return fail("error", "adapter_key_clash", f"adapter keys owned by the driver: {clash}")

    native_stop = result.native_stop or adapter.native_stop
    pareto = sanitize_block({
        **blk, **adapter_block,
        "t_solve": t_solve, "t_total": None if dual_absent else t_total,
        "t_eval": t_eval, "t_load": t_load, "t_solve_bracket": t_bracket,
        "t_solve_boundary": ck.t_solve_boundary,
        "t_solve_includes": list(ck.t_solve_includes),
        "t_solve_excludes": list(ck.t_solve_excludes),
        "t_total_boundary": ck.t_total_boundary,
        "t_total_includes": list(ck.t_total_includes),
        "pareto_partial": dual_absent, "pareto_absent": dual_absent, "dual_absent": dual_absent,
        "precision": adapter.precision, "native_stop": native_stop,
        "native_stop_is_library_default": adapter.native_stop_is_library_default,
        "native_stop_is_upstream_default": adapter.native_stop_is_upstream_default,
        "x_nonneg_structural": r_P_neg_source == "structural",
        "r_P_neg_source": r_P_neg_source,
        "dual_certified": adapter.dual_certified,
    })
    if dual_absent:
        pareto["reason"] = rec.dual_absent_reason or "dual_absent"

    try:
        diag = adapter.diagnostics(problem, result, rec, pareto) or {}
    except Exception as e:                                       # noqa: BLE001
        diag = {"diagnostics_error": repr(e)}

    if result.feaserr_measure:
        feaserr, feaserr_measure = finite_or_none(result.feaserr), result.feaserr_measure
    elif result.feaserr is not None:
        return fail("error", "adapter_feaserr_unnamed",
                    "SolveResult.feaserr is set but feaserr_measure does not name it")
    else:
        feaserr, feaserr_measure = blk.get("r_P"), "r_P"
    device_name = device_uuid = None
    if device.type == "cuda":
        from src.runners.gpu_state import device_fingerprint
        fp = device_fingerprint(idx=device.index or 0)
        device_name, device_uuid = fp.get("device_name"), fp.get("device_uuid")

    algo_extras = json_safe({
        **adapter.failure_extras(spec), **adapter_extras, **peak_extras, **diag,
        "native_stop": native_stop, "precision": adapter.precision,
        "scored_marginals": "unit_mass_original",
        "feaserr_measure": feaserr_measure, "r_P": blk.get("r_P"),
        "wall_time_eval_s": t_eval,
    }, nonfinite_to_none=True)
    if spec.pareto:
        algo_extras["pareto"] = pareto
    m = int(r0.numel())
    record = ResultRecord(
        instance_id=spec.instance_id, algorithm=adapter.algorithm, norm=spec.norm,
        resolution=spec.resolution, n=m, m=m, run_uid=run_uid,
        started_utc=started_utc, ended_utc=utc_now_iso(), status="ok",
        wall_time_s=t_total + t_eval, wall_time_solve_s=t_solve,
        wall_time_recovery_s=t_total - t_solve,
        iters=_int(result.iters), iter_cap_hit=bool(result.iter_cap_hit),
        native_termination=result.native_termination, tolerance=spec.tolerance,
        feaserr=feaserr, native_duality_gap=result.native_duality_gap,
        objective=blk.get("p_obj"), device_name=device_name, device_uuid=device_uuid,
        regime=regime_for_algorithm(adapter.algorithm), dataset=spec.dataset,
        rung=canonical_regime(spec.rung) if spec.rung else None,
        algo_extras=algo_extras, **mem,
    )
    try:
        record.to_json()
    except (TypeError, ValueError) as e:
        return fail("error", "record_not_serialisable", repr(e))
    return record


def baseline_main(argv: Optional[Sequence[str]],
                  adapter_factory: Callable[[Any], BaselineAdapter], *, description: str,
                  add_args: Optional[Callable] = None,
                  check_args: Optional[Callable] = None) -> int:
    """The command-line entry of a baseline runner; ``adapter_factory(args)``
    builds the adapter once, from the parsed arguments."""
    built: Dict[str, BaselineAdapter] = {}

    def adapter(args) -> BaselineAdapter:
        if "a" not in built:
            built["a"] = adapter_factory(args)
        return built["a"]

    return runner_main(
        argv, description=description, add_args=add_args, check_args=check_args,
        run=lambda spec, args: run_baseline(spec, adapter(args)),
        failure=lambda spec, args, **kw: failure_record(
            spec, adapter(args).algorithm, m=_rows(spec),
            algo_extras=adapter(args).failure_extras(spec), **kw))
