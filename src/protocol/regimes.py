"""
The three DOT formats and the rules that decide which algorithm runs which instance.

Every solver consumes exactly one format, determined by its input. Result records
store it as ``regime``:

- ``Grid``: the solver receives ``(r, c, n_g, cost_type)``. Only the separable
  ground costs ``l1`` and ``sql2`` are representable.

- ``PointCloud``: the solver receives point clouds P, Q in R^d and a ground
  cost κ, and evaluates κ from the coordinates; no dense cost matrix is
  stored.

- ``General``: the solver consumes an explicit dense cost matrix C, which may
  be arbitrary.

A DOT instance has its own native format, stored as ``rung`` in specs and
records. It can be solved in its native format or in any format more general
than it in the order Grid, Point Cloud, General (``CONSUMES``). For example,
a DOTmark instance with sql2 norm is a Grid instance, so Grid, Point Cloud
and General solvers can all solve it.
"""

from typing import (Any, Dict, FrozenSet, Iterable, List, Optional,
                    Sequence, Tuple)

GRID = "Grid"
POINT_CLOUD = "PointCloud"
GENERAL = "General"
REGIMES = (GRID, POINT_CLOUD, GENERAL)

# Alias -> current name. Stored result records and instance specs may carry
# the solver names BDRS / BDRS_geo / BDRS_dense and the lowercase format names
# grid / geometric / dense; every read boundary maps them to the current names.
_LEGACY_ALGO_ALIASES = {"BDRS": "Grid", "BDRS_geo": "PointCloud", "BDRS_dense": "General"}
_LEGACY_REGIME_ALIASES = {"grid": GRID, "geometric": POINT_CLOUD, "dense": GENERAL}


def canonical_algorithm(algo: str) -> str:
    return _LEGACY_ALGO_ALIASES.get(algo, algo)


def canonical_regime(value: str) -> str:
    return _LEGACY_REGIME_ALIASES.get(value, value)


# For each native instance format, the set of algorithm formats that can
# consume it.
CONSUMES = {
    GRID: {GRID, POINT_CLOUD, GENERAL},
    POINT_CLOUD: {POINT_CLOUD, GENERAL},
    GENERAL: {GENERAL},
}

SEPARABLE_NORMS = ("l1", "sql2")
DENSE_ONLY_NORMS = ("l2", "inf")

# Procedural costs are not ground costs: the entries are drawn i.i.d. and the
# matrix is regenerated from (m, n, cost_seed, cost_family) instead of being
# evaluated on coordinates. They correspond to random_dense cost matrices in
# General format.
PROCEDURAL_NORMS = ("rand_uniform", "rand_absgauss")

ALL_NORMS = SEPARABLE_NORMS + DENSE_ONLY_NORMS + PROCEDURAL_NORMS


def procedural_norm(cost_family: str) -> str:
    token = f"rand_{cost_family}"
    if token not in PROCEDURAL_NORMS:
        raise ValueError(
            f"Unknown procedural cost family {cost_family!r}; "
            f"valid tokens: {PROCEDURAL_NORMS}")
    return token


REGIME_NORMS = {
    GRID: ("l1", "sql2"),
    POINT_CLOUD: ("l1", "sql2", "l2", "inf"),
    GENERAL: ("l1", "l2", "inf") + PROCEDURAL_NORMS,
}

OUR_SOLVERS = frozenset({"Grid", "General", "PointCloud"})


# Format of each algorithm. Variant names are reduced to a base name by
# base_algorithm() before lookup.
REGIME_BY_ALGORITHM: Dict[str, str] = {
    # Grid: needs a regular grid and an axis-separable cost.
    "Grid": GRID,
    "HOT": GRID,
    "SinkhornGrid": GRID,
    # Point Cloud: points in a metric space.
    "PointCloud": POINT_CLOUD,
    "HALO": POINT_CLOUD,
    "M3S": POINT_CLOUD,
    "LAMPPointCloud": POINT_CLOUD,
    # General: an arbitrary stored N x N cost.
    "General": GENERAL,
    "DROT": GENERAL,
    "PDOT": GENERAL,
    "Sinkhorn": GENERAL,
    "CUPDLPX": GENERAL,
    "LAMP": GENERAL,
}

# Per-algorithm cost restrictions, set by the baselines themselves.
_NORMS_BY_ALGORITHM: Dict[str, Sequence[str]] = {
    "HOT": ("sql2",),
    "HALO": ("sql2",),
    "M3S": ("sql2",),
    # LAMP's matrix-free kernels compute Σ|d|, Σd² or max|d|: no unsquared l2.
    "LAMPPointCloud": ("l1", "sql2", "inf"),
}


# The entropic arms are scheduled once per ε. For example
# Sinkhorn_e0p01 is POT Sinkhorn at ε = 0.01.
SINKHORN_PREFIX = "Sinkhorn_e"
SINKHORNGRID_PREFIX = "SinkhornGrid_e"


def eps_slug(eps: float) -> str:
    return f"{eps:g}".replace(".", "p").replace("-", "m")


def eps_variant_name(prefix: str, eps: float) -> str:
    return prefix + eps_slug(eps)


def parse_eps_variant(algo: str, prefix: str) -> Optional[float]:
    if not algo.startswith(prefix):
        return None
    return float(algo[len(prefix):].replace("p", ".").replace("m", "-"))


def base_algorithm(algo: str) -> str:
    """Reduce a variant algorithm name to its base name."""
    if algo.startswith(SINKHORNGRID_PREFIX) or algo == "SinkhornGrid":
        return "SinkhornGrid"
    if algo.startswith(SINKHORN_PREFIX) or algo == "Sinkhorn":
        return "Sinkhorn"
    return canonical_algorithm(algo)


def regime_for_algorithm(algo: str) -> str:
    """Format ("Grid" | "PointCloud" | "General") of an algorithm or variant name."""
    base = base_algorithm(algo)
    try:
        return REGIME_BY_ALGORITHM[base]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm {algo!r} (base {base!r}); "
            f"known: {sorted(REGIME_BY_ALGORITHM)}"
        ) from None


def parse_policy_exceptions(specs: Iterable[str]) -> FrozenSet[Tuple[str, str]]:
    """Parse ``"<format>:<norm>"`` strings into (format, norm) pairs."""
    representable = {GRID: SEPARABLE_NORMS,
                     POINT_CLOUD: SEPARABLE_NORMS + DENSE_ONLY_NORMS,
                     GENERAL: ALL_NORMS}
    out = set()
    for spec in specs:
        regime, _, norm = str(spec).partition(":")
        regime = canonical_regime(regime)
        if regime not in REGIME_NORMS:
            raise ValueError(
                f"Unknown regime in policy exception {spec!r}; "
                f"valid: {tuple(REGIME_NORMS)}")
        if norm not in ALL_NORMS:
            raise ValueError(
                f"Unknown norm in policy exception {spec!r}; valid: {ALL_NORMS}")
        if norm not in representable[regime]:
            raise ValueError(
                f"Policy exception {spec!r}: a {regime} solver cannot represent "
                f"the {norm} cost; valid for {regime}: {representable[regime]}")
        out.add((regime, norm))
    return frozenset(out)


def algo_norms(algo: str,
               exceptions: "Iterable[Tuple[str, str]] | None" = None
               ) -> Sequence[str]:
    """The ground costs the benchmark runs for this algorithm."""
    base = base_algorithm(algo)
    if base in _NORMS_BY_ALGORITHM:
        return _NORMS_BY_ALGORITHM[base]
    regime = regime_for_algorithm(base)
    policy = REGIME_NORMS[regime]
    if exceptions:
        extra = tuple(n for r, n in sorted(exceptions)
                      if r == regime and n not in policy)
        if extra:
            return policy + extra
    return policy


def rung_compatible(instance_rung: str, algo: str) -> bool:
    """Whether ``algo``'s format can consume an instance of this native format."""
    instance_rung = canonical_regime(instance_rung)
    if instance_rung not in CONSUMES:
        raise ValueError(
            f"Unknown instance rung {instance_rung!r}; valid: {tuple(CONSUMES)}")
    return regime_for_algorithm(algo) in CONSUMES[instance_rung]


def algo_applies_instance(algo: str, instance: Dict[str, Any],
                          exceptions: "Iterable[Tuple[str, str]] | None" = None
                          ) -> bool:
    """Scheduler gate: whether ``algo`` runs this instance. Requires BOTH

    1. the instance's ground cost (``instance['norm']``) is in
       ``algo_norms(algo, exceptions)``; AND
    2. ``algo``'s format can consume the instance's native format
       (``rung_compatible``).
    """
    rung = instance.get("rung") or GRID
    return (instance["norm"] in algo_norms(algo, exceptions)
            and rung_compatible(rung, algo))


def synthesize_dense_norm_entries(entries: Iterable[Dict[str, Any]],
                                  norm: str) -> List[Dict[str, Any]]:
    """Derive the DOTmark instances of ``norm`` ("l2" | "inf") from the index."""
    if norm not in DENSE_ONLY_NORMS:
        raise ValueError(
            f"Not a dense-only norm: {norm!r}; valid: {DENSE_ONLY_NORMS}")
    entries = list(entries)          # two passes below; accept any iterable
    out: List[Dict[str, Any]] = []
    seen = {
        (ent["class_name"], ent["resolution"], ent["i"], ent["j"])
        for ent in entries
        if ent["norm"] == norm
    }
    for ent in entries:
        key = (ent["class_name"], ent["resolution"], ent["i"], ent["j"])
        if key in seen:
            continue
        seen.add(key)
        cls, ng, i, j = key
        out.append({
            "instance_id": f"{cls}_n{ng}_i{i}_j{j}_{norm}",
            "class_name": cls,
            "resolution": ng,
            "i": i,
            "j": j,
            "id_i": ent["id_i"],
            "id_j": ent["id_j"],
            "norm": norm,
            "marginal_i_path": ent["marginal_i_path"],
            "marginal_j_path": ent["marginal_j_path"],
            "cost_matrix_path": None,
        })
    return out
