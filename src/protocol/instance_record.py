"""
The orchestrator–runner contract: ``InstanceSpec`` in, ``ResultRecord`` out.

``InstanceSpec``: the orchestrator writes one per (algorithm, instance) pair to
a temporary JSON file and runs ``python -m src.runners.run_<algo> --instance
<spec.json> --output <record.json>``. PDOT's adapter instead sends the spec
as one JSON line to a long-running Julia process.

Both schemas carry ``schema_version``. A change that makes existing records
incompatible must bump ``SCHEMA_VERSION``: the aggregator drops every record
whose version is not in ``SCHEMA_COMPAT``, and ``InstanceSpec.validate``
accepts only ``SCHEMA_VERSION``. An added field with a default (such as
``ResultRecord.regime``) does not bump it: existing records still load with
the default in place, and the aggregator derives a missing ``regime`` from
the algorithm name.
"""

import datetime as _dt
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from src.protocol.regimes import PROCEDURAL_NORMS, REGIME_BY_ALGORITHM, canonical_regime


SCHEMA_VERSION = "v2"

# Schema versions the aggregator accepts. v1 records are runs on DOTmark; v2
# adds the `dataset` namespace, the instance's native `rung` and the Point
# Cloud instance block.
SCHEMA_COMPAT = frozenset({"v1", "v2"})

GRID = "Grid"
POINT_CLOUD = "PointCloud"
GENERAL = "General"
RUNGS = (GRID, POINT_CLOUD, GENERAL)

# Final statuses for each (instance, algorithm, tolerance) triple. An "error" row is retried on the next pass.
SETTLED_STATUSES = frozenset({"ok", "timeout", "oom", "oom_recovery"})

WALL_BUFFER_S = 30

# Base algorithm names a result record may carry: the keys of
# regimes.REGIME_BY_ALGORITHM.
ALL_ALGORITHMS = frozenset(REGIME_BY_ALGORITHM)


def geo_cost_type(norm: str) -> str:
    return "linf" if norm == "inf" else norm


VALID_NORMS = ("l1", "sql2", "l2", "inf") + PROCEDURAL_NORMS


def utc_now_iso() -> str:
    """The current UTC time in ISO 8601."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


@dataclass(frozen=True)
class InstanceSpec:
    """One instance and its solver controls, as handed to a runner.

    The fields that describe the instance depend on ``rung``:

    - ``Grid`` (DOTmark and other image datasets): ``resolution`` (n_g),
      ``class_name``, the pair ``i``/``j`` with image ids ``id_i``/``id_j``,
      and the marginal paths.
    - ``PointCloud``: ``support_i_path``/``support_j_path`` (P, Q), ``c_max``,
      ``m``, ``n``, ``dim`` and the marginal paths.
    - ``General``: the marginal paths and one cost source: a
      ``cost_matrix_path``, a ``cost_seed`` (procedural cost, with ``m`` and
      ``n``), or a ``resolution`` from which the lattice C is built.

    ``rung`` is the format of the spec as the runner receives it, which can
    differ from the instance's native format. A Grid instance given to our
    Point Cloud solver arrives as ``PointCloud`` with the lattice points as P
    and Q and ``resolution`` kept (HALO and M3S get the Grid spec). A Point
    Cloud instance given to a dense solver arrives as ``General`` with a
    materialized ``cost_matrix_path`` and ``resolution`` = √N, its Point
    Cloud fields kept. A Grid instance given to a dense solver arrives
    unchanged; the runner loads its stored C if it has one and otherwise
    builds C from the lattice. The orchestrator writes the native rung into
    the record.

    ``norm`` names the ground cost in every rung; fields that do not apply
    stay None. Every field after ``instance_id`` has a default, so a v1 spec
    dict constructs by keyword.
    """

    instance_id: str
    dataset: str = "dotmark"               # namespace; only DOTmark ids lack "<dataset>__"
    rung: str = GRID                       # Grid | PointCloud | General (class docstring)
    class_name: str = ""
    resolution: Optional[int] = None       # n_g; √N after a dense lift; else None
    i: Optional[int] = None
    j: Optional[int] = None
    id_i: Optional[int] = None
    id_j: Optional[int] = None
    norm: str = "l1"                       # ground cost (VALID_NORMS)
    marginal_i_path: str = ""              # r, length m, .npy
    marginal_j_path: str = ""              # c, length n, .npy
    cost_matrix_path: Optional[str] = None  # stored dense C, if any
    # Procedural General cost: with no cost_matrix_path and cost_seed set,
    # every dense runner regenerates the same C from
    # (m, n, cost_seed, cost_family) through dense_cost.resolve_dense_cost.
    cost_seed: Optional[int] = None
    cost_family: Optional[str] = None       # 'uniform' (also when None) | 'absgauss'
    support_i_path: Optional[str] = None   # P: [m, dim] .npy source points
    support_j_path: Optional[str] = None   # Q: [n, dim] .npy target points
    c_max: Optional[float] = None          # C_max: C = κ(p, q)/C_max, so max C = 1
    m: Optional[int] = None                # number of rows (length of r)
    n: Optional[int] = None                # number of columns (length of c)
    dim: Optional[int] = None              # ambient dimension d
    tolerance: float = 1e-4
    time_limit_s: int = 1800            # seconds; a solve that exceeds it is a timeout
    iter_cap: int = 1_000_000              # K_max
    output_path: str = ""                  # base path of runner side files (not the record)
    warmup: int = 1
    device: str = "cuda:0"
    # T_chk override, read by our three solvers only.
    check_every: Optional[int] = None
    # Native-stop protocol. When True, every solver stops by its own default
    # rule (baselines run with their upstream stopping parameters and ignore
    # ``tolerance``; our three solvers keep their feasible-gap stop at
    # ``tolerance``), and the runner recovers (X, α, β) for the original
    # problem and writes the raw KKT components and the clocks t_solve and
    # t_total under ``algo_extras["pareto"]`` (scoring/pareto_score.py). When
    # False no such block is written, and the baselines that accept a
    # tolerance are given ``tolerance``.
    pareto: bool = False
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rung", canonical_regime(self.rung))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_path(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def from_json(cls, s: str) -> "InstanceSpec":
        return cls(**json.loads(s))

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "InstanceSpec":
        return cls.from_json(Path(path).read_text())

    def dense_shape(self) -> Tuple[int, int]:
        """``(rows, cols)`` of the dense C and of the plan X.

        A spec with a ``resolution`` has the lattice shape (n_g², n_g²); any
        other spec has ``(m, n)``, with ``n`` defaulting to ``m``.
        """
        if self.resolution is not None:
            side = int(self.resolution) * int(self.resolution)
            return side, side
        if not self.m:
            raise ValueError(
                f"{self.instance_id}: dense shape needs a grid resolution or m/n")
        return int(self.m), int(self.n or self.m)

    def grid_equivalent_ng(self) -> Optional[int]:
        """The n_g with n_g² rows for a square instance, or None."""
        if self.resolution is not None:
            return int(self.resolution)
        rows, cols = self.dense_shape()
        if rows != cols:
            return None
        ng = math.isqrt(rows)
        return ng if ng * ng == rows else None

    def validate(self) -> None:
        """Structural checks of the fields; raises ValueError on the first failure."""
        if self.rung not in RUNGS:
            raise ValueError(f"Unknown rung: {self.rung!r}")
        if self.norm not in VALID_NORMS:
            raise ValueError(f"Unknown norm: {self.norm!r}")
        if self.norm in PROCEDURAL_NORMS:
            if self.rung != GENERAL:
                raise ValueError(
                    f"norm {self.norm!r} is General-rung only; got {self.rung!r}")
            if self.cost_seed is None:
                raise ValueError(
                    f"norm {self.norm!r} claims a procedural cost but "
                    "cost_seed is None")
        if self.rung == GRID:
            if self.resolution not in (32, 64, 128, 256, 512, 1024):
                raise ValueError(f"Unexpected grid resolution: {self.resolution}")
        elif self.rung == POINT_CLOUD:
            missing = [k for k in ("support_i_path", "support_j_path", "c_max",
                                   "m", "n", "dim") if getattr(self, k) is None]
            if missing:
                raise ValueError(
                    f"PointCloud rung requires {missing} (got None)")
            if self.m <= 0 or self.n <= 0 or self.dim <= 0:
                raise ValueError("PointCloud m/n/dim must be > 0")
            if self.c_max <= 0:
                raise ValueError(f"PointCloud c_max must be > 0; got {self.c_max}")
        elif self.rung == GENERAL:
            if (self.cost_matrix_path is None and self.cost_seed is None
                    and self.resolution is None):
                raise ValueError(
                    "dense rung needs a cost_matrix_path, a cost_seed "
                    "(procedural cost), or a grid resolution to build C on "
                    "the fly")
            if self.resolution is None and not self.m:
                raise ValueError(
                    "a non-grid dense instance needs m (and n): there is no "
                    "lattice to derive the dense shape from")
        if not (math.isfinite(self.tolerance) and self.tolerance > 0):
            raise ValueError(f"tolerance must be finite and > 0; got {self.tolerance}")
        if self.time_limit_s <= 0:
            raise ValueError(f"time_limit_s must be > 0; got {self.time_limit_s}")
        if self.iter_cap <= 0:
            raise ValueError(f"iter_cap must be > 0; got {self.iter_cap}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0; got {self.warmup}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version mismatch: spec is {self.schema_version!r}, "
                f"this build is {SCHEMA_VERSION!r}"
            )


@dataclass
class ResultRecord:
    """One run of one algorithm on one instance, as appended to the JSONL stream."""

    # Identity and provenance
    instance_id: str
    algorithm: str
    norm: str
    resolution: Optional[int]              # spec.resolution (n_g; √N after a dense lift)
    n: int                                 # length of r (n_g² or m), not the column count
    run_uid: str
    started_utc: str
    ended_utc: str

    # Status. oom_recovery: out of memory while recovering the plan after the solve.
    status: str                             # ok | timeout | oom | oom_recovery | error
    oom_flag: bool = False
    timeout_flag: bool = False
    error_kind: Optional[str] = None
    error_msg: Optional[str] = None         # cut to at most 4096 characters

    wall_time_s: Optional[float] = None
    wall_time_solve_s: Optional[float] = None
    wall_time_recovery_s: Optional[float] = None

    # Termination. tolerance is part of the resume and aggregation key: our
    # solvers run the same instance at several tolerances.
    iters: Optional[int] = None
    iter_cap_hit: bool = False
    native_termination: Optional[str] = None
    tolerance: Optional[float] = None

    # Metrics, all nullable. feaserr is a primal-feasibility measure of the
    # delivered solution, named by algo_extras["feaserr_measure"]. For Grid,
    # General and PointCloud it is the primal residual r_P. For a baseline it
    # is the measure the solver itself reports or stops on, so feaserr is not
    # comparable across solvers.
    feaserr: Optional[float] = None
    native_duality_gap: Optional[float] = None
    objective: Optional[float] = None

    peak_gpu_mem_bytes: Optional[int] = None
    peak_gpu_mem_source: Optional[str] = None
    device_name: Optional[str] = None
    device_uuid: Optional[str] = None
    # SHA of the code that produced this row; the orchestrator fills it when
    # the runner did not.
    git_sha: Optional[str] = None

    # Format the ALGORITHM consumes: "Grid" | "PointCloud" | "General"
    # (regimes.regime_for_algorithm). The orchestrator overwrites the runner's
    # value; a record without one gets it from the algorithm name when
    # aggregated.
    regime: Optional[str] = None

    # Dataset namespace and the instance's native format, set by the
    # orchestrator from the index entry.
    dataset: Optional[str] = None
    rung: Optional[str] = None
    m: Optional[int] = None

    # Per-algorithm fields (eta, drot_dtype, hot_plan_recovery, "pareto", ...)
    algo_extras: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_path(self, path: Union[str, Path]) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def from_json(cls, s: str) -> "ResultRecord":
        d = json.loads(s)
        return cls(**d)

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> "ResultRecord":
        return cls.from_json(Path(path).read_text())

