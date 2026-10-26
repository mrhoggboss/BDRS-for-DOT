"""Draw Figs. 1 to 7 from the benchmark's and the studies' outputs.

    python -m src.figures.make --figure FIG [FIG ...] --runs PATH --out DIR
                               [--cells-out DIR]

FIG is one of 1, 2, 3, 4a, 4b, 5, 6, 7a, 7b; ``presets()`` holds what each
figure draws. ``--runs`` is, for Figs. 1 to 5, the per-run table ``pareto.csv``
written by ``src.report.aggregate`` (or ``pareto.csv.gz``, or the directory
holding it); for Fig. 6, the directory holding ``n32/runs`` and ``n64/runs`` of
``src.studies.dual_stall``; for Figs. 7a and 7b, the ``--out`` of
``src.studies.robustness``. The figures are written to DIR as

    1   pareto_exp1_sql2_n64.pdf             4b  pareto_trajectory_22b_dotmark.pdf
    2   pareto_trajectory_21_general.pdf     5   pareto_trajectory_23_grid.pdf
    3   pareto_trajectory_21_random.pdf      6   ct_ablation_combined.pdf
    4a  pointcloud_trajectory.pdf            7a  robustness_eta.pdf
                                             7b  robustness_tchk.pdf

``--cells-out DIR`` also writes the cells a Pareto figure drew to
``DIR/<name>.csv``.

Text is set with LaTeX when it is usable and with matplotlib's bundled
Computer Modern otherwise; ``BDRS_FIG_TEX=0`` or ``1`` forces the choice.
"""

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib


@dataclasses.dataclass(frozen=True)
class CellFigure:
    dataset: str
    norm: str
    size: int


@dataclasses.dataclass(frozen=True)
class DualStallFigure:
    sizes: Tuple[int, ...]
    eps: float
    ylim: Optional[Tuple[float, float]]


@dataclasses.dataclass(frozen=True)
class RobustnessFigure:
    panel: str                              # "eta" or "tchk"


# Output names, as the LaTeX source includes the figures.
STEMS = {"1": "pareto_exp1_sql2_n64", "2": "pareto_trajectory_21_general",
         "3": "pareto_trajectory_21_random", "4a": "pointcloud_trajectory",
         "4b": "pareto_trajectory_22b_dotmark", "5": "pareto_trajectory_23_grid",
         "6": "ct_ablation_combined", "7a": "robustness_eta", "7b": "robustness_tchk"}
FIGURES = tuple(STEMS)
PARETO_FIGURES = ("1", "2", "3", "4a", "4b", "5")


def presets() -> Dict[str, Any]:
    from src.figures.pareto_trajectory import Drop, Mark, TrajectorySpec

    dotmark_sizes = (32, 64, 128, 256)
    return {
        # Every solver run on the cell, at every tolerance and ε.
        "1": CellFigure("dotmark", "sql2", 64),
        "2": TrajectorySpec(
            dataset="dotmark", regimes=("General",), norms=("l1", "l2", "inf"),
            sizes=dotmark_sizes, tols=(1e-4,), size_tags=False,
            fail_labels_for=frozenset({"General"}),
            title="General format, lifted DOTmark"),
        "3": TrajectorySpec(
            dataset="random_dense", regimes=("General",),
            norms=("rand_uniform", "rand_absgauss"), sizes=(1024, 4096, 16384, 65536),
            tols=(1e-4, 1e-6), size_tags=False, fail_labels_for=frozenset({"General"}),
            title="General format, random cost matrices"),
        # The tol = 1e-6 arm is run up to N = 2^15; the cross marks N = 2^16.
        "4a": TrajectorySpec(
            dataset="modelnet10", regimes=("PointCloud",), norms=("sql2",),
            layout="compact", tols=(1e-4, 1e-6),
            marks=(Mark("sql2", "PointCloud", 1e-6, 65536, "timeout"),),
            title="Point Cloud format, ModelNet10"),
        "4b": TrajectorySpec(
            dataset="dotmark", regimes=("PointCloud",), norms=("sql2",),
            layout="compact", sizes=dotmark_sizes, tols=(1e-4, 1e-6),
            title="Point Cloud format, lifted DOTmark"),
        # One rule decides what an arm draws at each size: a point where most
        # of its runs there finish, taken over the runs that finish within the
        # co-solved set, and a timeout cross where most of them reach the time
        # limit. Hence tol = 1e-6 at n_g = 1024 is a point with l1, where 296
        # of its 418 runs finish (the 32 pairs that reach the limit at
        # n_g = 512 are not run at 1024). With sql2 it is not run at 1024, and
        # a cross marks it there, projected from the solve times at n_g = 512;
        # rows of that cell made under another stopping rule and step size are
        # dropped. tol = 1e-8 is run up to n_g = 256 with sql2 and up to
        # n_g = 512 with l1, there on 50 pairs of which 42 finish; a cross
        # marks the next size, where its runs are projected to reach the limit.
        # Convolutional Sinkhorn at ε = 1e-4 fails on most instances at every
        # size with l1, so it has no point there.
        "5": TrajectorySpec(
            dataset="dotmark", regimes=("Grid",), norms=("sql2", "l1"),
            sizes=dotmark_sizes + (512, 1024), size_tags=False,
            drops=(Drop("sql2", "Grid", 1e-6, 1024), Drop("l1", "SinkhornGrid_e0p0001")),
            marks=(Mark("sql2", "Grid", 1e-8, 512, "timeout"),
                   Mark("l1", "Grid", 1e-8, 1024, "timeout"),
                   Mark("sql2", "Grid", 1e-6, 1024, "timeout")),
            title="Grid format, DOTmark"),
        "6": DualStallFigure(sizes=(32, 64), eps=1e-4, ylim=(1e-9, 3e3)),
        "7a": RobustnessFigure("eta"),
        "7b": RobustnessFigure("tchk"),
    }


def make(fig: str, runs: Path, out_dir: Path, cells_out: Optional[Path] = None,
         rows=None):
    """Draw figure ``fig``; returns (the PDF path, the drawing's summary).
    ``rows`` are the typed rows of ``runs`` when already read."""
    from src.figures import dual_stall, pareto_cell, pareto_trajectory, robustness
    from src.figures.cells import CellSource, read_runs, write_cells

    preset = presets()[fig]
    out = Path(out_dir) / f"{STEMS[fig]}.pdf"
    if isinstance(preset, DualStallFigure):
        return out, dual_stall.draw(runs, out, sizes=preset.sizes, eps=preset.eps,
                                    ylim=preset.ylim)
    if isinstance(preset, RobustnessFigure):
        draw = robustness.draw_eta if preset.panel == "eta" else robustness.draw_tchk
        return out, draw(runs, out)
    source = CellSource(read_runs(runs) if rows is None else rows)
    if isinstance(preset, CellFigure):
        summary = pareto_cell.draw(source, dataset=preset.dataset, norm=preset.norm,
                                   size=preset.size, out=out)
    else:
        summary = pareto_trajectory.draw(source, preset, out)
    if cells_out is not None:
        write_cells(list(source.drawn.values()), Path(cells_out) / f"{STEMS[fig]}.csv")
    return out, summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.figures.make", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", nargs="+", required=True, choices=FIGURES, metavar="FIG")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cells-out", type=Path, default=None)
    a = p.parse_args(argv)

    matplotlib.use("Agg")
    from src.figures import plotstyle
    from src.figures.cells import read_runs
    try:
        usetex, why = plotstyle.resolve_usetex()
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    plotstyle.use(usetex)
    print(f"text: {'LaTeX' if usetex else 'matplotlib Computer Modern'} ({why})")

    rows = None
    if any(f in PARETO_FIGURES for f in a.figure):
        try:
            rows = read_runs(a.runs)
        except FileNotFoundError as e:
            print(f"error: no per-run table: {e}", file=sys.stderr)
            return 1
        print(f"{len(rows)} runs from {a.runs}")
    for fig in a.figure:
        print(f"Fig. {fig}")
        try:
            out, _ = make(fig, a.runs, a.out, a.cells_out, rows=rows)
        except (ValueError, FileNotFoundError) as e:
            print(f"error: Fig. {fig}: {e}", file=sys.stderr)
            return 1
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
