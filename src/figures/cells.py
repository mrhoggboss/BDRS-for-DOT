"""The cells a figure draws, rebuilt from the per-run table ``pareto.csv``.

Each run is scored again from its components by the rules of
``src.report.aggregate`` (``row_verdict``, ``cert_time``): whether it is
scorable, its time limit and its t_cert are derived, never read from the
table's own columns, so a table written under other rules is drawn under
these.

A figure's co-solved set is taken over the algorithms the figure draws: the
per-run rows of those algorithms are aggregated by the same
``_aggregate_pareto_cells`` that writes ``pareto_cells.csv``, with the basis
made of those algorithms less ``BASIS_EXCLUDE``. As there, an instance counts
as scored by an algorithm when any of its tolerance arms scored it.
"""

import csv
import gzip
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from src.report.aggregate import (
    _aggregate_pareto_cells, _write_csv, cert_time, row_verdict, time_limit_of,
)

# Convolutional Sinkhorn at ε = 1e-4 fails on part of the instances; it is
# aggregated and drawn, but it does not decide the co-solved set.
BASIS_EXCLUDE = frozenset({"SinkhornGrid_e0p0001"})

CELL_COLUMNS = ["dataset", "rung", "size", "norm", "algorithm", "tolerance", "regime",
                "resolution", "n_records", "n_total", "n_ok", "n_scorable", "n_partial",
                "n_cosolved", "n_error", "n_timeout", "n_oom", "n_unscorable",
                "unscorable_reasons", "in_cosolve_basis", "n_cosolved_cell",
                "n_cosolved_if_dropped", "cosolve_shrink", "E_abs_gm", "E_abs_median",
                "n_E_abs", "n_cert_invalid", "t_solve_sgm10_s", "t_solve_median_s",
                "t_solve_iqr_s", "t_total_sgm10_s", "t_total_median_s", "t_cert_sgm10_s",
                "t_cert_median_s", "n_cert_pass_missing", "n_cell_instances"]

_BOOL = {"scorable", "pareto_partial", "certificate_valid", "dual_certified",
         "native_stop_is_library_default", "native_stop_is_upstream_default",
         "iter_cap_hit"}
_INT = {"size", "resolution"}
_STR = {"instance_id", "algorithm", "dataset", "rung", "regime", "norm", "status",
        "unscorable_reason", "native_termination", "native_stop", "precision",
        "t_solve_boundary", "t_solve_includes", "t_solve_excludes", "t_total_boundary",
        "t_total_includes", "device_name", "device_uuid", "git_sha", "run_uid",
        "started_utc", "ended_utc", "schema_version", "error_kind"}


def _coerce(k: str, v: Any) -> Any:
    """CSV text to the type ``_aggregate_pareto_cells`` reads; empty is None."""
    if v is None or v == "":
        return None
    if k in _BOOL:
        return v in ("True", "true", "1")
    if k in _INT:
        try:
            return int(float(v))
        except ValueError:
            return None
    if k in _STR:
        return v
    try:
        return float(v)
    except ValueError:
        return v


def runs_table_path(path: Path) -> Path:
    """``path`` itself, or for a directory its ``pareto.csv`` (or ``pareto.csv.gz``)."""
    path = Path(path)
    if path.is_dir():
        for name in ("pareto.csv", "pareto.csv.gz"):
            if (path / name).is_file():
                return path / name
        raise FileNotFoundError(f"no pareto.csv or pareto.csv.gz in {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def rescore(row: Dict[str, Any]) -> Dict[str, Any]:
    """``row`` with ``scorable``, ``unscorable_reason``, ``time_limit_s`` and
    ``t_cert`` derived again from its components."""
    scorable, reason = row_verdict(row)
    row["scorable"] = scorable
    row["unscorable_reason"] = reason or None
    row["time_limit_s"] = time_limit_of(row)
    row["t_cert"] = cert_time(row)[0]
    return row


def read_runs(path: Path) -> List[Dict[str, Any]]:
    """The typed and rescored rows of a ``pareto.csv`` (``.csv`` or
    ``.csv.gz``, or a directory holding one)."""
    path = runs_table_path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        return [rescore({k: _coerce(k, v) for k, v in row.items()})
                for row in csv.DictReader(fh)]


def cells_for_roster(rows: Iterable[Dict[str, Any]], *, dataset: str,
                     norms: Sequence[str], sizes: Optional[Sequence[int]] = None,
                     regimes: Optional[Sequence[str]] = None,
                     basis_exclude: Iterable[str] = BASIS_EXCLUDE,
                     ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """``(cells, roster)``: the cells of ``dataset`` on ``norms`` (and ``sizes``)
    for the algorithms of ``regimes`` (all when None), co-solved over those
    algorithms less ``basis_exclude``. ``roster`` lists every algorithm drawn."""
    norm_set = set(norms)
    size_set = {int(s) for s in sizes} if sizes else None
    regime_set = set(regimes) if regimes else None
    sel = [r for r in rows
           if r.get("dataset") == dataset and r.get("norm") in norm_set
           and (size_set is None or r.get("size") in size_set)
           and (regime_set is None or r.get("regime") in regime_set)]
    roster = sorted({r["algorithm"] for r in sel})
    basis = {a for a in roster if a not in set(basis_exclude)}
    return _aggregate_pareto_cells(sel, algo_filter=basis), roster


def cells_frame(cells: List[Dict[str, Any]]) -> pd.DataFrame:
    if not cells:
        return pd.DataFrame(columns=CELL_COLUMNS)
    df = pd.DataFrame(cells)
    for c in CELL_COLUMNS:
        if c not in df.columns:
            df[c] = None
    return df


def write_cells(cells: List[Dict[str, Any]], path: Path) -> None:
    _write_csv(cells, Path(path), extra_columns=CELL_COLUMNS)


class CellSource:
    """The cells of one figure, aggregated once per (dataset, norms, sizes,
    regimes) request; ``drawn`` collects every cell handed out."""

    def __init__(self, rows: List[Dict[str, Any]], log: Callable[[str], None] = print):
        self.rows = rows
        self.log = log
        self._cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self.drawn: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    def table(self, dataset: str, norms: Sequence[str],
              sizes: Optional[Sequence[int]] = None,
              regimes: Optional[Sequence[str]] = None) -> pd.DataFrame:
        key = (dataset, tuple(norms), tuple(int(s) for s in sizes) if sizes else None,
               tuple(regimes) if regimes else None)
        if key not in self._cache:
            cells, roster = cells_for_roster(self.rows, dataset=dataset, norms=list(norms),
                                             sizes=sizes, regimes=regimes)
            excluded = [a for a in roster if a in BASIS_EXCLUDE]
            self.log(f"  co-solve basis for {dataset}/{list(norms)}: "
                     f"{[a for a in roster if a not in BASIS_EXCLUDE]}"
                     + (f" (drawn, not in the basis: {excluded})" if excluded else ""))
            for c in sorted({(c["norm"], c["size"], c["n_cosolved_cell"],
                              c["n_cell_instances"]) for c in cells}):
                self.log(f"    {c[0]:13s} n{c[1]:6d}: co-solved {c[2]}/{c[3]}")
            self._cache[key] = cells
            for c in cells:
                self.drawn[(c["dataset"], c["rung"], c["size"], c["norm"],
                            c["algorithm"], c["tolerance"])] = c
        return cells_frame(self._cache[key])
