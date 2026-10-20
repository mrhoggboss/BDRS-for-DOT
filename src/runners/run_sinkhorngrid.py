"""OTT-JAX Grid Sinkhorn runner, the entropic baseline of the Grid format (``l1``, ``sql2``).

OTT's ``Grid`` geometry applies the Gibbs kernel one axis at a time, so an
iteration costs O(n_g³) and no N × N array is formed. It solves the entropic
problem, not the LP: at a fixed point the gap of (X, f, g) is ε·H(X), with
H(X) = −Σ X_ij log X_ij.

The runner runs in the JAX environment (``$BDRS_OTT_PYTHON``) and must never
import torch, since torch and JAX must not share a CUDA context.
"""

import contextlib
import os
import sys
import time
import traceback
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Set before jax is imported. With "false" JAX allocates on demand instead of
# reserving most of the device, so the NVML peak measures the working set.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from src.methods.grid_cmax import grid_cmax
from src.protocol.instance_record import InstanceSpec, ResultRecord, utc_now_iso
from src.protocol.regimes import (
    GRID, SEPARABLE_NORMS, SINKHORNGRID_PREFIX, eps_slug, eps_variant_name,
)
from src.runners.gpu_state import device_identity
from src.runners.nvml_sampler import NvmlPeakSampler
from src.runners.common import (
    WARMUP_SEED,
    failure_record,
    is_oom_exception,
    load_marginal_unit_mass,
    runner_main,
    warmup_marginal,
)


# OTT's Sinkhorn defaults in ott-jax 0.6.0. Under --pareto they are asserted at
# solve time, so an OTT upgrade cannot silently change the native stop.
OTT_ASSHIPPED_STATIC = {"threshold": 1e-3, "max_iterations": 2000,
                        "inner_iterations": 10, "norm_error": (1,)}

# out.primal_cost goes through OTT's low-rank-cost (LRC) geometry, which for the
# l1 axis cost allocates about 31 GB at n_g = 128 (about 1 GB at n_g = 64), so
# the objective is cross-checked against it only up to n_g = 64.
PRIMAL_COST_CROSSCHECK_NG_MAX = 64

# Scaling (kernel) mode, not OTT's default log-sum-exp mode, for two reasons.
# The log-sum-exp Grid kernel (``Grid.apply_lse_kernel``) drops valid terms
# when a marginal has zero mass (see ``_plan_marginals``), which gives wrong
# costs and no convergence on such instances. And the ptxas compile time of
# that kernel grows super-linearly with n_g (about 15 s at n_g = 128, 13 min
# at n_g = 256), while scaling mode compiles in 5-8 s at n_g = 256. Without
# zero mass the two modes agree to round-off (objectives within about 1e-14
# relative on DOTmark at n_g = 32, ε = 1e-2 and 1e-3). The axis kernels are
# effectively banded, and in fp64 scaling mode did not underflow for any ε
# tested down to 1e-3; the finiteness guard in _solve_and_record catches an
# underflow below that range.
_LSE_MODE = False

# t_solve is one call of the jitted solve (_make_solve), compiled in the
# warm-up: the per-axis costs and kernels, OTT's initializer, the fixed-point
# loop and the regularized cost reg_ot_cost, all inside one XLA program.
T_SOLVE_BOUNDARY = "ott_sinkhorn_jit_call_block_until_ready"
T_SOLVE_INCLUDES = ("axis_cost_and_kernel", "ott_initializer", "reg_ot_cost")
T_SOLVE_EXCLUDES = ("jax_ott_import_and_x64_config", "solver_build",
                    "warmup_jit_trace_and_xla_compile", "marginals_load_h2d",
                    "grid_coordinates_build")
T_TOTAL_BOUNDARY = "ott_sinkhorn_plus_potential_d2h"
T_TOTAL_INCLUDES: Tuple[str, ...] = ()


def _make_failure_record(spec: InstanceSpec, *, status: str, error_kind: str,
                         error_msg: str, started_utc: str, run_uid: str,
                         algorithm: str, epsilon: float,
                         algo_extras: Optional[Dict[str, Any]] = None) -> ResultRecord:
    extras: Dict[str, Any] = {"epsilon": epsilon, "eps_slug": eps_slug(epsilon)}
    if algo_extras:
        extras.update(algo_extras)
    return failure_record(spec, algorithm, status=status, error_kind=error_kind, error_msg=error_msg,
                          started_utc=started_utc, run_uid=run_uid, algo_extras=extras)


def _load_marginals(spec: InstanceSpec) -> Tuple[np.ndarray, np.ndarray]:
    """The original (r, c), unit mass and never floored: OTT does not floor either."""
    r = load_marginal_unit_mass(spec.marginal_i_path)
    c = load_marginal_unit_mass(spec.marginal_j_path)
    return r, c


def _dot_support(mass: np.ndarray, potential: np.ndarray) -> float:
    """``⟨mass, potential⟩`` over the support, so 0·(−inf) = 0, as in OTT's own dual cost.

    OTT gives a zero-mass coordinate the potential −inf, where the plain dot
    product would be NaN."""
    sel = mass > 0.0
    if not bool(np.any(sel)):
        return 0.0
    return float(np.dot(mass[sel], potential[sel]))


def _offsupport_nonfinite(potential: np.ndarray, mass: np.ndarray) -> int:
    """Number of non-finite potentials, not counting −inf on a zero-mass
    coordinate, which is structural rather than scaling-mode underflow."""
    bad = ~np.isfinite(potential)
    structural = np.isneginf(potential) & (mass <= 0.0)
    return int(np.count_nonzero(bad & ~structural))


def _static_config(solver) -> Dict[str, Any]:
    """The static Sinkhorn fields the compiled fixed-point loop is specialised on.

    ``Sinkhorn.norm_error`` is a property returning a tuple of p-norm orders,
    not a scalar; both forms are accepted."""
    ne = solver.norm_error
    ne = tuple(ne) if isinstance(ne, (tuple, list)) else (int(ne),)
    return {"threshold": float(solver.threshold),
            "max_iterations": int(solver.max_iterations),
            "inner_iterations": int(solver.inner_iterations),
            "lse_mode": bool(solver.lse_mode),
            "norm_error": ne}


def _solver_kwargs(spec: InstanceSpec, *, pareto: bool) -> Dict[str, Any]:
    """OTT's defaults except ``lse_mode`` under --pareto; otherwise
    threshold = tol and max_iterations = K_max."""
    if pareto:
        return {"lse_mode": _LSE_MODE}
    return {"threshold": float(spec.tolerance),
            "max_iterations": int(spec.iter_cap),
            "lse_mode": _LSE_MODE}


def _make_solver(spec: InstanceSpec, *, pareto: bool):
    """The one ``Sinkhorn`` object of both the warm-up and the timed solve, with its static config."""
    from ott.solvers.linear import sinkhorn
    solver = sinkhorn.Sinkhorn(**_solver_kwargs(spec, pareto=pareto))
    cfg = _static_config(solver)
    if pareto:
        expected = dict(OTT_ASSHIPPED_STATIC, lse_mode=_LSE_MODE)
        if cfg != expected:
            raise RuntimeError(
                f"OTT Sinkhorn as-shipped defaults changed: got {cfg}, "
                f"expected {expected}")
    return solver, cfg


def _check_separable(norm: str) -> None:
    if norm not in SEPARABLE_NORMS:
        raise ValueError(f"SinkhornGrid is separable-only; got norm {norm!r} "
                         f"(valid: {SEPARABLE_NORMS})")


def _axis_coords(ng: int, norm: str):
    """The coordinates of one grid axis, scaled so that the per-axis cost of
    ``_axis_cost_fn`` summed over the two axes is the max-1 grid cost
    (``dense_cost.build_separable_cost`` up to rounding)."""
    import jax.numpy as jnp
    _check_separable(norm)
    c_max = grid_cmax(norm, ng)
    if norm == "l1":
        return jnp.arange(ng, dtype=jnp.float64) / c_max
    return jnp.arange(ng, dtype=jnp.float64) / np.sqrt(c_max)


def _axis_cost_fn(norm: str):
    """OTT's built-in cost of one axis: |x − y| for ``l1``, (x − y)² for ``sql2``."""
    from ott.geometry import costs
    _check_separable(norm)
    return costs.Euclidean() if norm == "l1" else costs.SqEuclidean()


def _grid_geometry(x, norm: str, epsilon: float):
    """OTT's ``Grid`` on the axis coordinates ``x = [x_r, x_c]`` at ε."""
    from ott.geometry import grid
    fn = _axis_cost_fn(norm)
    return grid.Grid(x=list(x), cost_fns=[fn, fn], epsilon=float(epsilon))


def _build_grid_geom(ng: int, norm: str, epsilon: float):
    coord = _axis_coords(ng, norm)
    return _grid_geometry([coord, coord], norm, epsilon)


def _make_solve(solver, norm: str, epsilon: float):
    """The solve of the warm-up and of the timed call, ``solve(x, a, b)``: one
    ``jax.jit`` program of ``solver`` on the grid with axis coordinates ``x``
    and marginals (a, b), returning OTT's ``SinkhornOutput``.

    Called directly, ``solver(prob)`` builds its fixed-point loop as a new
    closure on every call, so every call traces and compiles again. ``solve``
    is traced and compiled on its first call, the warm-up's, and a call with
    arguments of the same shapes and dtypes runs that program. The arguments
    are arrays only; the solver settings, the axis cost and ε are constants of
    the program, as they are of the loop OTT compiles when called directly.
    On CPU this gives the potentials and the iteration count of the direct
    call bit for bit."""
    import jax
    from ott.problems.linear import linear_problem

    def solve(x, a, b):
        geom = _grid_geometry(x, norm, epsilon)
        return solver(linear_problem.LinearProblem(geom, a=a, b=b))

    return jax.jit(solve)


# jax.monitoring events: a top-level trace; an XLA compilation, or a read of
# the compiled program from the persistent cache (one event either way); a
# persistent-cache read; a persistent-cache write.
_EV_TRACE = "/jax/core/compile/jaxpr_trace_duration"
_EV_COMPILE = "/jax/core/compile/backend_compile_duration"
_EV_CACHE_HIT = "/jax/compilation_cache/cache_hits"
_EV_CACHE_WRITE = "/jax/compilation_cache/cache_misses"


class _CompileEvents:
    """Counts of JAX's trace and compile events per window of the run.

    Inside ``with _CompileEvents() as ev``, events are counted into the
    window ``"other"``, or into ``name`` inside ``with ev.during(name)``."""

    def __init__(self):
        self.window = "other"
        self.counts: Dict[str, Dict[str, Any]] = {}

    def _bucket(self) -> Dict[str, Any]:
        return self.counts.setdefault(self.window, {
            "jaxpr_traces": 0, "jaxpr_trace_s": 0.0,
            "backend_compiles": 0, "backend_compile_s": 0.0,
            "cache_hits": 0, "cache_writes": 0})

    def _on_event(self, event: str, **_kw) -> None:
        if event == _EV_CACHE_HIT:
            self._bucket()["cache_hits"] += 1
        elif event == _EV_CACHE_WRITE:
            self._bucket()["cache_writes"] += 1

    def _on_duration(self, event: str, secs: float, **_kw) -> None:
        if event == _EV_TRACE:
            b = self._bucket()
            b["jaxpr_traces"] += 1
            b["jaxpr_trace_s"] += float(secs)
        elif event == _EV_COMPILE:
            b = self._bucket()
            b["backend_compiles"] += 1
            b["backend_compile_s"] += float(secs)

    @contextlib.contextmanager
    def during(self, name: str):
        prev, self.window = self.window, name
        self._bucket()
        try:
            yield
        finally:
            self.window = prev

    def compile_free(self, name: str) -> bool:
        """No trace and no compilation (or cache read) in window ``name``."""
        b = self.counts.get(name, {})
        return b.get("jaxpr_traces", 0) == 0 and b.get("backend_compiles", 0) == 0

    def __enter__(self) -> "_CompileEvents":
        import jax
        self._bucket()
        jax.monitoring.register_event_listener(self._on_event)
        jax.monitoring.register_event_duration_secs_listener(self._on_duration)
        return self

    def __exit__(self, *exc) -> None:
        import jax
        jax.monitoring.unregister_event_listener(self._on_event)
        jax.monitoring.unregister_event_duration_listener(self._on_duration)


def _structure_aware_objective(out, geom, ng: int, epsilon: float) -> Tuple[float, float, float]:
    """(⟨C, X⟩, Σ M^r, Σ M^c) of OTT's plan without forming X; each mass is 1 up to rounding.

    With a = (a_r, a_c), b = (b_r, b_c), X_ab = exp((f_a + g_b − C_ab)/ε) and
    C_ab = C^r[a_r, b_r] + C^c[a_c, b_c], so ⟨C, X⟩ = ⟨C^r, M^r⟩ + ⟨C^c, M^c⟩
    with M^r = Σ_{a_c, b_c} X and M^c = Σ_{a_r, b_r} X, each two nested
    log-sum-exps in O(n_g³) operations and memory."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    eps = float(epsilon)
    F = out.f.reshape(ng, ng)
    G = out.g.reshape(ng, ng)
    Cr = geom.geometries[0].cost_matrix.reshape(ng, ng)
    Cc = geom.geometries[1].cost_matrix.reshape(ng, ng)
    lUr = F / eps                 # [a_r, a_c]
    lVr = G / eps                 # [b_r, b_c]
    lKr = -Cr / eps               # [a_r, b_r]
    lKc = -Cc / eps               # [a_c, b_c]

    # logS[a_r,b_r] = LSE_{a_c,b_c}(lUr[a_r,a_c] + lVr[b_r,b_c] + lKc[a_c,b_c])
    P = logsumexp(lKc[:, None, :] + lVr[None, :, :], axis=2)      # (a_c, b_r)
    logS = logsumexp(lUr[:, :, None] + P[None, :, :], axis=1)     # (a_r, b_r)
    Mr = jnp.exp(lKr + logS)
    Tr = jnp.sum(Cr * Mr)

    # logSc[a_c,b_c] = LSE_{a_r,b_r}(lUr[a_r,a_c] + lVr[b_r,b_c] + lKr[a_r,b_r])
    PA = logsumexp(lUr[:, :, None] + lKr[:, None, :], axis=0)     # (a_c, b_r)
    logSc = logsumexp(PA[:, :, None] + lVr[None, :, :], axis=1)   # (a_c, b_c)
    Mc = jnp.exp(lKc + logSc)
    Tc = jnp.sum(Cc * Mc)

    obj = Tr + Tc
    return float(obj), float(jnp.sum(Mr)), float(jnp.sum(Mc))


def _plan_marginals(f, g, geom, ng: int, epsilon: float):
    """(X1, Xᵀ1) of the plan X_ab = exp((f_a + g_b − C_ab)/ε) of the potentials
    (f, g), flat, without forming X.

    With C_ab = C^r[a_r, b_r] + C^c[a_c, b_c], each is two nested log-sum-exps
    in O(n_g³) operations and memory:
    log (X1)_a  = f_a/ε + LSE_{b_r}(−C^r[a_r, b_r]/ε + LSE_{b_c}(g_b/ε − C^c[a_c, b_c]/ε)),
    log (Xᵀ1)_b = g_b/ε + LSE_{a_r}(−C^r[a_r, b_r]/ε + LSE_{a_c}(f_a/ε − C^c[a_c, b_c]/ε)).
    A potential only ever adds to an exponent, so f_a = −∞ (zero mass) removes
    row a of X and nothing else. OTT's ``Grid.marginal_from_potentials`` is not
    used: its log-domain kernel (``Grid.apply_lse_kernel``) adds a potential at
    a mixed index, such as f at (a_r, b_c), in its first one-dimensional pass and
    subtracts it after, so where that potential is −∞ the pass also drops terms
    of rows or columns with positive mass."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    eps = float(epsilon)
    lU = jnp.reshape(jnp.asarray(f), (ng, ng)) / eps                 # [a_r, a_c]
    lV = jnp.reshape(jnp.asarray(g), (ng, ng)) / eps                 # [b_r, b_c]
    lKr = -geom.geometries[0].cost_matrix.reshape(ng, ng) / eps      # [a_r, b_r]
    lKc = -geom.geometries[1].cost_matrix.reshape(ng, ng) / eps      # [a_c, b_c]

    # Q[b_r, a_c] = LSE_{b_c}(lV[b_r, b_c] + lKc[a_c, b_c])
    Q = logsumexp(lV[:, None, :] + lKc[None, :, :], axis=2)
    # log (X1)[a_r, a_c] = lU[a_r, a_c] + LSE_{b_r}(lKr[a_r, b_r] + Q[b_r, a_c])
    row = jnp.exp(lU + logsumexp(lKr[:, :, None] + Q[None, :, :], axis=1))

    # P[a_r, b_c] = LSE_{a_c}(lU[a_r, a_c] + lKc[a_c, b_c])
    P = logsumexp(lU[:, :, None] + lKc[None, :, :], axis=1)
    # log (Xᵀ1)[b_r, b_c] = lV[b_r, b_c] + LSE_{a_r}(lKr[a_r, b_r] + P[a_r, b_c])
    col = jnp.exp(lV + logsumexp(lKr[:, :, None] + P[:, None, :], axis=0))
    return row.reshape(-1), col.reshape(-1)


def _r_P(f, g, geom, ng: int, epsilon: float,
         r: np.ndarray, c: np.ndarray) -> Tuple[float, float, float]:
    """(r_P, ‖X1 − r‖₂, ‖Xᵀ1 − c‖₂) of the plan of the potentials (f, g), with
    the sums of ``_plan_marginals``; the plan is nonnegative, so r_P has no
    negative-part term."""
    import jax.numpy as jnp
    X1, XT1 = _plan_marginals(f, g, geom, ng, epsilon)
    err_r = float(jnp.linalg.norm(X1 - jnp.asarray(r)))
    err_c = float(jnp.linalg.norm(XT1 - jnp.asarray(c)))
    return err_r + err_c, err_r, err_c


def _native_feaserr(out) -> Optional[float]:
    """OTT's stopping error ‖Xᵀ1 − c‖_p (p = ``norm_error``) of the returned state, or None.

    Only the column marginal, because the u-update comes last and makes the
    row sums exact. ``out.errors`` holds one value per block of
    ``inner_iterations`` and -1 for blocks that never ran; None when nothing
    was recorded or the last value is not finite."""
    errs = np.asarray(out.errors, dtype=np.float64).reshape(-1)
    recorded = errs[errs != -1.0]
    if recorded.size == 0 or not np.isfinite(recorded[-1]):
        return None
    return float(recorded[-1])


def _warmup(solve, ng: int, norm: str, epsilon: float,
            events: Optional[_CompileEvents] = None) -> float:
    """One untimed call of the timed call's ``solve`` (``_make_solve``) on
    seeded dummy marginals of the same n_g, so that its trace and compile fall
    on no clock; the timed call passes arguments of the same shapes and dtypes
    and runs the program compiled here. Returns the seconds of that call.

    The metric passes also compile on their first call; running them here
    keeps those compiles out of wall_time_eval_s."""
    import jax

    N = ng * ng
    rng = np.random.default_rng(WARMUP_SEED)
    a = warmup_marginal(rng, N)
    b = warmup_marginal(rng, N)
    geom = _build_grid_geom(ng, norm, epsilon)
    a_j, b_j = jax.numpy.asarray(a), jax.numpy.asarray(b)
    window = events.during("warmup_solve") if events is not None else contextlib.nullcontext()
    with window:
        t0 = time.perf_counter()
        out = solve(geom.x, a_j, b_j)
        jax.block_until_ready(out)
        t_call = time.perf_counter() - t0
    _r_P(out.f, out.g, geom, ng, epsilon, a, b)
    _structure_aware_objective(out, geom, ng, epsilon)
    if ng <= PRIMAL_COST_CROSSCHECK_NG_MAX:
        try:
            float(out.primal_cost)
        except Exception:                                      # noqa: BLE001
            pass
    return t_call


def _run_one(spec: InstanceSpec, *, epsilon: float) -> ResultRecord:
    started_utc = utc_now_iso()
    run_uid = str(uuid.uuid4())
    algorithm = eps_variant_name(SINKHORNGRID_PREFIX, epsilon)

    if spec.norm not in SEPARABLE_NORMS:
        return _make_failure_record(
            spec, status="error", error_kind="non_separable_norm",
            error_msg=(f"SinkhornGrid needs a separable cost; norm {spec.norm!r} "
                       f"is not in {SEPARABLE_NORMS}."),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )

    # Imported here so a missing jax or ott gives an error record, not an
    # import crash.
    try:
        import jax
        # Before any jax array is built: without x64, a requested jnp.float64
        # array is created in fp32, with only a warning.
        jax.config.update("jax_enable_x64", True)
        # Persistent compilation cache: a later runner process with the same
        # n_g, cost, ε and solver settings reads the solve program in its
        # warm-up instead of compiling it. It holds compiled code, never
        # instance data.
        try:
            import tempfile
            cache_dir = os.environ.get(
                "JAX_COMPILATION_CACHE_DIR",
                os.path.join(tempfile.gettempdir(), "bdrs_jax_cache"))
            jax.config.update("jax_compilation_cache_dir", cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
        except Exception:                                      # noqa: BLE001
            pass
        import ott                                            # noqa: F401
        jax_version = jax.__version__
        ott_version = ott.__version__
    except ImportError as e:
        return _make_failure_record(
            spec, status="error", error_kind="jax_ott_import_failed",
            error_msg=repr(e),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
        )

    jax_platform = jax.default_backend()
    if jax_platform != "gpu":
        return _make_failure_record(
            spec, status="error", error_kind="no_gpu",
            error_msg=(f"jax.default_backend()={jax_platform!r}; SinkhornGrid is "
                       f"a GPU baseline and requires a CUDA device."),
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
            algo_extras={"jax_version": jax_version, "ott_version": ott_version,
                         "jax_platform": jax_platform},
        )

    return _solve_and_record(
        spec, epsilon=epsilon, started_utc=started_utc, run_uid=run_uid,
        env={"jax_version": jax_version, "ott_version": ott_version,
             "jax_platform": jax_platform})


def _solve_and_record(spec: InstanceSpec, *, epsilon: float, started_utc: str,
                      run_uid: str, env: Dict[str, str]) -> ResultRecord:
    """The run past the GPU gate: warm-up, timed solve, recovery of (f, g),
    evaluation and the record. ``env`` holds ``jax_version``, ``ott_version``
    and ``jax_platform``; jax must be imported with x64 enabled."""
    import jax

    algorithm = eps_variant_name(SINKHORNGRID_PREFIX, epsilon)
    n_total = spec.resolution * spec.resolution
    versions = {"jax_version": env["jax_version"], "ott_version": env["ott_version"]}
    device_index = 0
    sampler = NvmlPeakSampler(device_index)
    sampler.start()
    events = _CompileEvents()
    wall_warmup = wall_warmup_solve = None
    try:
        with events:
            solver, static_cfg = _make_solver(spec, pareto=bool(getattr(spec, "pareto", False)))
            solve = _make_solve(solver, spec.norm, epsilon)

            if spec.warmup > 0:
                _tw0 = time.perf_counter()
                wall_warmup_solve = _warmup(solve, spec.resolution, spec.norm, epsilon, events)
                wall_warmup = time.perf_counter() - _tw0

            r, c = _load_marginals(spec)
            a_j = jax.numpy.asarray(r)
            b_j = jax.numpy.asarray(c)
            geom = _build_grid_geom(spec.resolution, spec.norm, epsilon)

            # JAX dispatch is asynchronous: block before stopping the clock.
            with events.during("timed_solve"):
                t0 = time.perf_counter()
                out = solve(geom.x, a_j, b_j)
                jax.block_until_ready(out)
                t1 = time.perf_counter()
            wall_solve = t1 - t0

            # The device-to-host copy of (f, g) is the recovery of (α, β), timed
            # into t_total.
            _tr0 = time.perf_counter()
            f_np = np.asarray(out.f, dtype=np.float64).reshape(-1)
            g_np = np.asarray(out.g, dtype=np.float64).reshape(-1)
            wall_recovery = time.perf_counter() - _tr0
            n_zero_r = int(np.count_nonzero(r <= 0.0))
            n_zero_c = int(np.count_nonzero(c <= 0.0))

            # Scaling mode can underflow at ε far below the tested range.
            n_bad = (_offsupport_nonfinite(f_np, r) + _offsupport_nonfinite(g_np, c))
            if n_bad:
                sampler.stop()
                return _make_failure_record(
                    spec, status="error", error_kind="scaling_mode_nonfinite",
                    error_msg=(f"{n_bad} non-finite potential entries off the "
                               f"zero-mass support at eps={epsilon} (scaling-mode "
                               f"underflow); use a larger eps."),
                    started_utc=started_utc, run_uid=run_uid,
                    algorithm=algorithm, epsilon=epsilon,
                    algo_extras={**versions, "ott_call": "jax.jit",
                                 "lse_mode": _LSE_MODE,
                                 "n_zero_mass_r": n_zero_r, "n_zero_mass_c": n_zero_c},
                )

            t2 = time.perf_counter()
            r_P, row_err, col_err = _r_P(out.f, out.g, geom, spec.resolution, epsilon, r, c)
            feaserr = _native_feaserr(out)
            objective, mass_r, mass_c = _structure_aware_objective(
                out, geom, spec.resolution, epsilon)
            crosscheck_rel = None
            if spec.resolution <= PRIMAL_COST_CROSSCHECK_NG_MAX:
                try:
                    pc = float(out.primal_cost)
                    crosscheck_rel = abs(objective - pc) / max(abs(pc), 1e-300)
                except Exception:                              # noqa: BLE001
                    crosscheck_rel = None
            jax.block_until_ready(out.g)
            t3 = time.perf_counter()
            wall_eval = t3 - t2
            # Diagnostics on no clock.
            reg_ot_cost = float(out.reg_ot_cost)
            ent_reg_cost = float(out.ent_reg_cost)
    except Exception as e:                                     # noqa: BLE001
        is_oom = is_oom_exception(e)
        sampler.stop()
        return _make_failure_record(
            spec, status=("oom" if is_oom else "error"),
            error_kind=("jax_oom" if is_oom else type(e).__name__),
            error_msg=f"{e!r}\n{traceback.format_exc()[:3500]}",
            started_utc=started_utc, run_uid=run_uid,
            algorithm=algorithm, epsilon=epsilon,
            algo_extras={**versions, "ott_call": "jax.jit"},
        )
    peak_mem, baseline_mem = sampler.stop()

    # OTT reports f, g in cost units, so (α, β) = (f, g) with no factor ε.
    # r_Dinf and r_DF need torch: they stay None here, and
    # report.aggregate fills them from the .pareto_fg.npz sidecar, or
    # analytically when the sidecar is missing.
    pareto_block = None
    if getattr(spec, "pareto", False):
        import math as _math
        _fin = lambda v: (None if (isinstance(v, float) and not _math.isfinite(v)) else v)
        r_flat = np.asarray(r, dtype=np.float64).reshape(-1)
        c_flat = np.asarray(c, dtype=np.float64).reshape(-1)
        d_obj = _dot_support(r_flat, f_np) + _dot_support(c_flat, g_np)
        fg_path = None
        if spec.output_path:
            fg_path = str(spec.output_path) + ".pareto_fg.npz"
            np.savez(fg_path, f=f_np, g=g_np, r=r_flat, c=c_flat,
                     ng=np.int64(spec.resolution), cost_type=str(spec.norm))
        pareto_block = {k: _fin(v) for k, v in {
            "r_P_row": row_err, "r_P_col": col_err, "r_P": row_err + col_err,
            "r_P_neg": 0.0, "p_obj": objective, "d_obj": d_obj,
            "gap": objective - d_obj,
            "r_Dinf": None, "r_DF": None,
            "C_inf": 1.0,
            "t_solve": float(wall_solve),
            "t_total": float(wall_solve + wall_recovery),
            "t_eval": None, "pareto_partial": False,
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_total_boundary": T_TOTAL_BOUNDARY,
            "t_total_includes": list(T_TOTAL_INCLUDES),
            "t_dual_recovery": float(wall_recovery),
            "pareto_needs_posthoc_rDinf": True, "pareto_fg_path": fg_path,
            "precision": "fp64",
            "native_stop": "ott_default_threshold_1e-3_maxiter_2000",
            "n_zero_mass_r": n_zero_r, "n_zero_mass_c": n_zero_c,
            "marginals": "original_unfloored",
        }.items()}

    converged = bool(out.converged)
    n_iters = int(out.n_iters)
    # Under --pareto the cap is OTT's max_iterations, not spec.iter_cap.
    _binding_cap = int(static_cfg["max_iterations"])
    iter_cap_hit = (not converged) or (n_iters >= _binding_cap)
    name, uuid_s = device_identity(device_index)

    rec = ResultRecord(
        instance_id=spec.instance_id, algorithm=algorithm, regime=GRID,
        norm=spec.norm, resolution=spec.resolution, n=n_total,
        run_uid=run_uid, started_utc=started_utc, ended_utc=utc_now_iso(),
        status="ok",
        wall_time_s=wall_solve + wall_recovery + wall_eval,
        wall_time_solve_s=wall_solve,
        wall_time_recovery_s=wall_recovery,
        iters=n_iters,
        iter_cap_hit=iter_cap_hit,
        native_termination=("converged" if converged else "max_iters"),
        tolerance=spec.tolerance,
        feaserr=feaserr,
        native_duality_gap=None,            # OTT reports no LP duality gap
        objective=objective,
        peak_gpu_mem_bytes=peak_mem,
        peak_gpu_mem_source=("nvml_poll" if peak_mem is not None else "unknown"),
        device_name=name,
        device_uuid=uuid_s,
        algo_extras={
            "feaserr_measure": "ott_sinkhorn_error",
            "feaserr_norm_p": int(static_cfg["norm_error"][0]),
            "r_P": r_P,
            "epsilon": float(epsilon),
            "eps_slug": eps_slug(epsilon),
            **versions,
            "jax_platform": env["jax_platform"],
            "lse_mode": _LSE_MODE,
            "ott_static_config": {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in static_cfg.items()},
            # OTT's solve runs as one jax.jit program (_make_solve), compiled
            # in the warm-up; jax_compile_events counts JAX's traces,
            # compilations and persistent-cache reads and writes in the
            # warm-up's solve call, the timed solve and the rest of the run.
            "ott_call": "jax.jit",
            "jax_compile_events": events.counts,
            "timed_solve_compile_free": events.compile_free("timed_solve"),
            "wall_time_warmup_s": wall_warmup,
            "wall_time_warmup_solve_s": wall_warmup_solve,
            "t_solve_boundary": T_SOLVE_BOUNDARY,
            "t_solve_includes": list(T_SOLVE_INCLUDES),
            "t_solve_excludes": list(T_SOLVE_EXCLUDES),
            "t_total_boundary": T_TOTAL_BOUNDARY,
            "t_total_includes": list(T_TOTAL_INCLUDES),
            "marginals": "original_unfloored",
            "n_zero_mass_r": n_zero_r,
            "n_zero_mass_c": n_zero_c,
            "converged": converged,
            "n_iters": n_iters,
            "reg_ot_cost": reg_ot_cost,
            "ent_reg_cost": ent_reg_cost,
            "objective_crosscheck_rel": crosscheck_rel,
            "mass_residual_r": abs(mass_r - 1.0),
            "mass_residual_c": abs(mass_c - 1.0),
            "row_sum_err_l2": row_err,
            "col_sum_err_l2": col_err,
            "wall_time_eval_s": wall_eval,
            "nvml_baseline_bytes": baseline_mem,
            "nvml_poll_spans_warmup": bool(spec.warmup > 0),
            "r_P_negpart_assumed_zero": True,
            **({"pareto": pareto_block} if pareto_block is not None else {}),
        },
    )
    return rec


def main(argv=None) -> int:
    def add_args(p):
        p.add_argument("--epsilon", required=True, type=float,
                       help="Entropic regularization parameter ε > 0.")

    def check_args(args):
        if args.epsilon <= 0:
            return f"run_sinkhorngrid: --epsilon must be positive; got {args.epsilon}"
        return None

    return runner_main(
        argv, description="OTT-JAX Grid Sinkhorn runner.", add_args=add_args, check_args=check_args,
        run=lambda spec, args: _run_one(spec, epsilon=float(args.epsilon)),
        failure=lambda spec, args, **kw: _make_failure_record(
            spec, algorithm=eps_variant_name(SINKHORNGRID_PREFIX, float(args.epsilon)),
            epsilon=float(args.epsilon), **kw))


if __name__ == "__main__":
    sys.exit(main())
