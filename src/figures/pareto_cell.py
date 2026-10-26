"""Accuracy against time for one Pareto cell (one dataset, size and cost).

Each (algorithm, tolerance) arm is one point: x = t_cert (SGM10), y = the KKT
residual E_abs (geometric mean), both over the cell's co-solved instances.
Colour is the algorithm family, its shade the arm's swept parameter (lighter =
looser), and the marker the format the solver consumes. A point that
aggregates fewer instances than the cell's co-solved set (an arm outside the
co-solve basis, or one with fewer scored solves) is drawn hollow. Where t_cert exceeds
t_solve, a tick at t_solve and a segment to the point show what t_cert adds:
a baseline's recovery of (X, α, β) and, where its dual needs it, the scoring
pass that measures r_D.
"""

import io
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.figures import plotstyle
from src.figures.cells import CellSource

XCOL = "t_cert_sgm10_s"
XLABEL = "Runtime including recovery and scoring (s), SGM10"
ECOL = "E_abs_gm"
ELABEL = "KKT residual, GM"

FORMAT_MARKER = {"Grid": "s", "PointCloud": "^", "General": "o"}

# One colour per algorithm family; our three solvers are the family "BDRS",
# and LAMP's General and Point Cloud modes the family "LAMP".
FAMILY_COLOR = {
    "BDRS": "#1b7f3b",
    "HALO": "#d62728",
    "HOT": "#1f77b4",
    "M3S": "#9467bd",
    "Sinkhorn": "#ff7f0e",
    "Conv. Sinkhorn": "#bcbd22",
    "DROT": "#8c564b",
    "PDOT": "#e377c2",
    "CUPDLPX": "#17becf",
    "LAMP": "#393b79",
}
OURS = ("Grid", "PointCloud", "General")

# Legend names that differ from the algorithm name.
DISPLAY = {"PointCloud": "Point Cloud", "CUPDLPX": "cuPDLPx"}


def display(name: str) -> str:
    return DISPLAY.get(name, name)


def family(algo: str) -> str:
    """The algorithm with its ε suffix removed; our three formats are "BDRS"
    and LAMP's two are "LAMP"."""
    base = algo.split("_e0p")[0]
    if base in OURS:
        return "BDRS"
    if base == "SinkhornGrid":
        return "Conv. Sinkhorn"
    if base == "LAMPPointCloud":
        return "LAMP"
    return base


def arm_param(algo: str, tol) -> float:
    """The swept parameter of an arm: ε for an entropic arm, tol for ours."""
    if "_e0p" in str(algo):
        return float("0." + str(algo).split("_e0p")[-1])
    return float(tol)


def shade(color: str, rank: int, n: int) -> str:
    """``color`` lightened by 0.5·(n − 1 − rank)/(n − 1): rank 0 (the loosest
    parameter) is lightest, rank n − 1 the full colour."""
    if n <= 1:
        return color
    r, g, b = mc.to_rgb(color)
    f = 0.5 * (n - 1 - rank) / (n - 1)
    return mc.to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def is_partial(row) -> bool:
    """Whether a drawn cell aggregates fewer instances than its cell's co-solved
    set (``n_cosolved < n_cosolved_cell``)."""
    try:
        return int(row["n_cosolved"]) < int(row["n_cosolved_cell"])
    except (KeyError, TypeError, ValueError):
        return False


def point_style(color: str, hollow: bool, edge_width: float) -> Dict:
    """The marker colours of one point: filled with a white edge, or hollow
    with an edge in ``color``."""
    if hollow:
        return dict(color=color, markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=1.8)
    return dict(color=color, markeredgecolor="white", markeredgewidth=edge_width)


def draw_cert_segments(ax, g: pd.DataFrame, color: str) -> None:
    """A segment from t_solve to the point at t_cert, with a tick at t_solve,
    for each point whose t_cert exceeds t_solve by more than 2%."""
    for ts, tc, y in zip(g.t_solve_sgm10_s, g.t_cert_sgm10_s, g[ECOL]):
        if tc > ts * 1.02:
            ax.plot([ts, tc], [y, y], color=color, linestyle="-", linewidth=1.6,
                    solid_capstyle="butt", zorder=4)
            ax.plot([ts], [y], marker="|", color=color, markersize=9,
                    markeredgewidth=1.6, linestyle="none", zorder=4)


def _family_ranks(c: pd.DataFrame) -> Dict[str, List[float]]:
    """Each family's swept parameters, loosest first."""
    ranks = {}
    for fam, g in c.groupby("family"):
        ranks[fam] = sorted({arm_param(a, t) for a, t in zip(g.algorithm, g.tolerance)},
                            reverse=True)
    return ranks


def load_cell(source: CellSource, dataset: str, norm: str, size: int) -> pd.DataFrame:
    """The cell's scorable arms, ordered by t_solve."""
    d = source.table(dataset, [norm], sizes=[size])
    c = d[(d.dataset == dataset) & (d["size"] == size) & (d.norm == norm)]
    c = c[c[ECOL].notna() & c.t_solve_sgm10_s.notna()].copy()
    c["family"] = c.algorithm.map(family)
    if c.empty:
        raise ValueError(f"no scorable rows for {dataset} n={size} {norm}")
    return c.sort_values("t_solve_sgm10_s")


def draw_cell(ax, c: pd.DataFrame) -> None:
    """Every arm of ``c`` as one point; no legend and no title."""
    ranks = _family_ranks(c)
    for (fam, reg, algo, tol), g in c.groupby(["family", "regime", "algorithm", "tolerance"],
                                              sort=False):
        col = shade(FAMILY_COLOR.get(fam, "#444444"),
                    ranks[fam].index(arm_param(algo, tol)), len(ranks[fam]))
        for _, r in g.iterrows():
            ax.plot([r[XCOL]], [r[ECOL]], marker=FORMAT_MARKER.get(reg, "o"),
                    linestyle="none", markersize=10, zorder=5,
                    **point_style(col, is_partial(r), 1.1))
        draw_cert_segments(ax, g, col)
    ax.set_xscale("log")
    ax.set_yscale("log")


def _tex_power(v: float) -> str:
    e = int(round(math.log10(v)))
    return rf"10^{{{e}}}" if abs(v - 10.0 ** e) < 1e-12 * v else f"{v:g}"


def legend_handles(c: pd.DataFrame) -> Tuple[list, list]:
    """(format-marker handles, one colour swatch per arm) for the arms drawn;
    ours first, the entropic families last."""
    shapes = [Line2D([], [], marker=m, color="#444444", linestyle="none",
                     markersize=10, markeredgecolor="white", label=display(f))
              for f, m in FORMAT_MARKER.items() if (c.regime == f).any()]
    ranks = _family_ranks(c)
    colors = []
    for fam in sorted(ranks, key=lambda f: (f != "BDRS", f in ("Sinkhorn", "Conv. Sinkhorn"), f)):
        prm = ranks[fam]
        for i, v in enumerate(prm):
            if len(prm) == 1:
                lab = display(fam)
            elif fam == "BDRS":
                lab = rf"{fam}, $\mathrm{{tol}} = {_tex_power(v)}$"
            else:
                lab = rf"{display(fam)}, $\varepsilon = {_tex_power(v)}$"
            colors.append(Patch(facecolor=shade(FAMILY_COLOR.get(fam, "#444444"), i, len(prm)),
                                edgecolor="none", label=lab))
    return shapes, colors


def _overlaps(fig, renderer, lower, upper) -> bool:
    """Whether the top of legend ``lower`` lies above the bottom of ``upper``
    once ``fig`` is drawn."""
    fig.savefig(io.BytesIO(), format="pdf")
    return lower.get_window_extent(renderer).y1 > upper.get_window_extent(renderer).y0


def draw(source: CellSource, *, dataset: str, norm: str, size: int, out: Path) -> Dict:
    """The cell as one figure, with the legends to the right of the axes."""
    c = load_cell(source, dataset, norm, size)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    draw_cell(ax, c)
    plotstyle.label(ax, x=XLABEL, y=ELABEL)
    shapes, colors = legend_handles(c)
    leg = ax.legend(handles=shapes, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                    frameon=False, borderaxespad=0.0, fontsize=plotstyle.TICK_SIZE)
    ax.add_artist(leg)
    # The colour legend's lower edge is level with the x label's, which is only
    # placed once the figure is drawn.
    fig.savefig(io.BytesIO(), format="pdf")
    r = fig.canvas.get_renderer()
    y0 = ax.transAxes.inverted().transform(
        (0.0, ax.xaxis.label.get_window_extent(r).y0))[1]
    kw = dict(handles=colors, frameon=False, borderaxespad=0.0, borderpad=0.0,
              fontsize=plotstyle.TICK_SIZE, handlelength=1.2)
    lower = ax.legend(loc="lower left", bbox_to_anchor=(1.02, y0), **kw)
    # Where it reaches into the format legend, its rows are set closer; where
    # it still does (as in the mathtext fallback), it hangs from just below
    # the format legend instead, keeping the closer rows.
    if _overlaps(fig, r, lower, leg):
        lower.remove()
        kw["labelspacing"] = 0.3
        lower = ax.legend(loc="lower left", bbox_to_anchor=(1.02, y0), **kw)
        if _overlaps(fig, r, lower, leg):
            lower.remove()
            gap = 0.5 * plotstyle.TICK_SIZE * fig.dpi / 72.0
            floor = leg.get_window_extent(r).y0
            y1 = ax.transAxes.inverted().transform((0.0, floor - gap))[1]
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, y1), **kw)
    ax.annotate("better $\\downarrow\\leftarrow$", xy=(0.015, 0.02),
                xycoords="axes fraction", fontsize=plotstyle.TICK_SIZE, color="#555555")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return {"arms": [(a, t) for a, t in zip(c.algorithm, c.tolerance)],
            "hollow": sorted((r.algorithm, float(r.tolerance)) for _, r in c.iterrows()
                             if is_partial(r)),
            "n_cosolved": int(c.n_cosolved.max())}
