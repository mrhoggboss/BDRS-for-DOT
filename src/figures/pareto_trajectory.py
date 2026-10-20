"""Accuracy against time over a ladder of problem sizes.

Each (algorithm, tolerance) arm is a set of unconnected points in the
(t_cert, KKT residual) plane, one per size, so an arm traces its trajectory as
the problem grows. Colour is the algorithm family, its shade the swept
parameter (lighter = looser), and the marker shape the problem size, assigned
by rank over the whole figure. An arm that has no scored solve at the next
size (or, symmetrically, at a smaller one) ends in a dotted tie to a cross at
the panel edge, labelled with the cause: OOM, timeout, error or no plan. The
label sits above the cross, or below it where above it would cover a marker
or another label, with the y range lowered to make room.

Two layouts: a row of panels, one per cost, sharing both axes; or one compact
panel with the legends below it.
"""

import dataclasses
import math
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator

from src.figures import plotstyle
from src.figures.cells import CellSource
from src.figures.pareto_cell import (
    ECOL, ELABEL, FAMILY_COLOR, FORMAT_MARKER, XCOL, XLABEL, arm_param, display,
    draw_cert_segments, family, shade,
)


@dataclasses.dataclass(frozen=True)
class Drop:
    """A point not drawn: ``norm`` and ``algorithm`` may be "*", ``tol`` and
    ``size`` None for any."""
    norm: str
    algorithm: str
    tol: Optional[float] = None
    size: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class Mark:
    """A failure cross after the arm's last point, labelled "<label> at
    <size>", drawn where no recorded failure already marks the arm. ``norm``
    may be "*"."""
    norm: str
    algorithm: str
    tol: float
    size: int
    label: str


@dataclasses.dataclass(frozen=True)
class TrajectorySpec:
    """What a trajectory figure draws.

    ``regimes`` selects the algorithms by the format they consume, ``tols``
    keeps only these tolerance arms of ours (a baseline's tolerance is unused),
    and ``fail_labels_for`` restricts the failure labels to these algorithms
    (None: all), while every cross is drawn. Baseline points outside the size
    range of ours are not drawn."""
    dataset: str
    regimes: Tuple[str, ...]
    norms: Tuple[str, ...]
    title: str
    layout: str = "row"                    # "row" or "compact" (one norm)
    sizes: Optional[Tuple[int, ...]] = None
    tols: Optional[Tuple[float, ...]] = None
    size_tags: bool = True                 # "row": size labels at each arm's ends
    fail_labels_for: Optional[FrozenSet[str]] = None
    drops: Tuple[Drop, ...] = ()
    marks: Tuple[Mark, ...] = ()


SIZE_MARKERS = ["^", "s", "D", "o", "p", "h", "v", "*", "<", ">"]

NORM_TEX = {"l1": r"$\ell_1$", "l2": r"$\ell_2$",
            "inf": r"$\ell_\infty$", "sql2": r"$\ell_2^2$",
            "rand_uniform": "uniform cost",
            "rand_absgauss": "half-normal cost"}

BETTER = "better $\\downarrow\\leftarrow$"

# The row layout's colour legend: its number of columns, and the height of one
# row in inches.
LEGEND_NCOL = 4
LEGEND_ROW_IN = 0.28


def _apply_drops(d: pd.DataFrame, norm: str, drops: Sequence[Drop]) -> pd.DataFrame:
    if not drops:
        return d
    keep = pd.Series(True, index=d.index)
    for dr in drops:
        if dr.norm not in ("*", norm):
            continue
        hit = (d.algorithm == dr.algorithm) if dr.algorithm != "*" \
            else pd.Series(True, index=d.index)
        if dr.tol is not None:
            hit &= (d.tolerance - dr.tol).abs() <= 1e-12 * max(dr.tol, 1e-300)
        if dr.size is not None:
            hit &= d["size"] == dr.size
        for _, r in d[hit].sort_values("size").iterrows():
            print(f"  dropped {norm} {r.algorithm} tol={r.tolerance} n={int(r['size'])} "
                  f"(scored {int(r.n_scorable)} of its {int(r.n_records)} runs; "
                  f"{int(r.n_cell_instances)} instances in the cell)")
        keep &= ~hit
    return d[keep]


def _keep_tols(d: pd.DataFrame, tols) -> pd.DataFrame:
    if not tols:
        return d
    return d[(d.family != "BDRS") | d.tolerance.isin(list(tols))]


def load(source: CellSource, spec: TrajectorySpec, norm: str) -> pd.DataFrame:
    """The scorable arms of one panel, ordered by size."""
    regimes = list(spec.regimes)
    d = source.table(spec.dataset, [norm], sizes=spec.sizes, regimes=regimes)
    d = d[(d.dataset == spec.dataset) & (d.norm == norm) & d.regime.isin(regimes)]
    if spec.sizes:
        d = d[d["size"].isin(list(spec.sizes))]
    d = d[d[ECOL].notna() & d.t_solve_sgm10_s.notna()].copy()
    d["family"] = d.algorithm.map(family)
    d = _keep_tols(d, spec.tols)
    if d.empty:
        raise ValueError(f"no scorable {spec.dataset}/{regimes}/{norm} rows")
    ours = d[d.family == "BDRS"]["size"]
    if not ours.empty:
        lo, hi = ours.min(), ours.max()
        trimmed = int(((d["size"] < lo) | (d["size"] > hi)).sum())
        if trimmed:
            print(f"  trimmed {trimmed} baseline cells outside N in [{int(lo)}, {int(hi)}]")
        d = d[(d["size"] >= lo) & (d["size"] <= hi)]
    d = _apply_drops(d, norm, spec.drops)
    if d.empty:
        raise ValueError(f"every {spec.dataset}/{regimes}/{norm} point was dropped")
    return d.sort_values("size")


def load_failures(source: CellSource, spec: TrajectorySpec, norm: str) -> pd.DataFrame:
    """The arms' cells with no scored solve."""
    regimes = list(spec.regimes)
    d = source.table(spec.dataset, [norm], sizes=spec.sizes, regimes=regimes)
    d = d[(d.dataset == spec.dataset) & (d.norm == norm) & d.regime.isin(regimes)]
    if spec.sizes:
        d = d[d["size"].isin(list(spec.sizes))]
    d = d[d.n_scorable.fillna(0).astype(int) == 0].copy()
    d["family"] = d.algorithm.map(family)
    return _keep_tols(d, spec.tols)


def fail_label(row) -> str:
    """Why a cell has no scored solve, in one word."""
    def n(col):
        try:
            return int(float(row[col]))
        except Exception:                                         # noqa: BLE001
            return 0
    if n("n_oom") > 0:
        return "OOM"
    reasons = str(row.get("unscorable_reasons") or "")
    if n("n_timeout") > 0 or "time_limit" in reasons:
        return "timeout"
    if "pareto_partial" in reasons:
        return "no plan"
    if "error" in reasons:
        return "error"
    return "failed"


def size_marker_map(sizes) -> Dict[int, str]:
    return {n: SIZE_MARKERS[i % len(SIZE_MARKERS)]
            for i, n in enumerate(sorted(set(int(x) for x in sizes)))}


def _size_tag(dataset: str, n) -> str:
    """The grid side on DOTmark; elsewhere the marginal length, as a power of two."""
    n = int(n)
    if dataset == "dotmark":
        return rf"${n}$"
    e = int(round(math.log2(n)))
    return rf"$2^{{{e}}}$" if 2 ** e == n else rf"${n}$"


def _size_legend_label(dataset: str, n) -> str:
    n = int(n)
    if dataset == "dotmark":
        return rf"$n_g = {n}$"
    e = int(round(math.log2(n)))
    pw = rf"2^{{{e}}}" if 2 ** e == n else rf"{n}"
    if dataset == "random_dense":
        return rf"$m = n = {pw}$"
    return rf"$N = {pw}$"


def _tol_tex(tol: float) -> str:
    e = int(round(np.log10(tol)))
    return rf"10^{{{e}}}" if np.isclose(tol, 10.0 ** e) else f"{tol:g}"


def _draw_arms(ax, d, dataset, multi_format, size_mk, fails, norm, marks):
    """Draw every arm of ``d``. Returns (size tags to place, failure crosses,
    legend handles); a cross is (x, y, colour, text, side, algorithm)."""
    placed, crosses, handles = [], [], []
    max_size = max(size_mk) if size_mk else None
    fam_params = {}
    for (algo, tol) in d.groupby(["algorithm", "tolerance"], sort=False).groups:
        fam_params.setdefault(family(algo), set()).add(arm_param(algo, tol))
    fam_ranks = {f: sorted(v, reverse=True) for f, v in fam_params.items()}

    for (algo, tol), g in d.groupby(["algorithm", "tolerance"], sort=False):
        g = g.sort_values("size")
        fam = family(algo)
        reg = g.regime.iloc[0]
        ranks = fam_ranks[fam]
        col = shade(FAMILY_COLOR.get(fam, "#444444"),
                    ranks.index(arm_param(algo, tol)), len(ranks))
        if fam != "BDRS":
            eps = algo.split("_e0p")[-1] if "_e0p" in algo else None
            lab = (display(fam) if eps is None
                   else rf"{display(fam)}, $\varepsilon = {_tol_tex(float('0.' + eps))}$")
        else:
            lab = rf"{fam}, $\mathrm{{tol}} = {_tol_tex(tol)}$"
            if multi_format:
                lab = rf"{fam} {display(reg)}, $\mathrm{{tol}} = {_tol_tex(tol)}$"

        xs, ys = g[XCOL].to_numpy(), g[ECOL].to_numpy()
        for x, y, n in zip(xs, ys, g["size"]):
            mk = size_mk[int(n)] if size_mk else FORMAT_MARKER.get(reg, "o")
            # Our solvers' points are drawn above the baselines' where they meet.
            ax.plot([x], [y], marker=mk, color=col, linestyle="none",
                    markersize=14.0 if mk == "*" else 9.5,
                    markeredgecolor="white", markeredgewidth=1.0,
                    zorder=6.5 if fam == "BDRS" else 6)
        handles.append((lab, Patch(facecolor=col, edgecolor="none", label=lab)))
        draw_cert_segments(ax, g, col)

        for r, dy in ((g.iloc[0], 12), (g.iloc[-1], -17)):
            placed.append((r[XCOL], r[ECOL], dy, _size_tag(dataset, r["size"]), col))

        marked = False
        if fails is not None and max_size is not None and int(g["size"].max()) < max_size:
            nxt = fails[(fails.algorithm == algo) & (fails.tolerance == tol)
                        & (fails["size"] > g["size"].max())].sort_values("size")
            if not nxt.empty:
                f0 = nxt.iloc[0]
                crosses.append((xs[-1], ys[-1], col,
                                f"{fail_label(f0)} at {_size_tag(dataset, f0['size'])}",
                                "right", algo, float(tol)))
                marked = True
        if fails is not None and size_mk and int(g["size"].min()) > min(size_mk):
            prv = fails[(fails.algorithm == algo) & (fails.tolerance == tol)
                        & (fails["size"] < g["size"].min())].sort_values("size", ascending=False)
            if not prv.empty:
                f0 = prv.iloc[0]
                crosses.append((xs[0], ys[0], col,
                                f"{fail_label(f0)} below {_size_tag(dataset, g['size'].min())}",
                                "left", algo, float(tol)))
        if not marked:
            for um in marks:
                if (um.norm in ("*", norm) and um.algorithm == algo
                        and abs(um.tol - float(tol)) <= 1e-12 * max(1.0, um.tol)
                        and um.size > int(g["size"].max())):
                    crosses.append((xs[-1], ys[-1], col,
                                    f"{um.label} at {_size_tag(dataset, um.size)}", "right",
                                    algo, float(tol)))
                    break
    return placed, crosses, handles


# A failure cross: its marker size and edge width, and the offset in points of
# its label from it.
CROSS_SIZE, CROSS_EDGE = 11, 2.4
LABEL_DX, LABEL_DY = 9, 8


def _cross_x(ax, side: str) -> float:
    """The x of a cross at the panel's ``side`` edge."""
    lo, hi = ax.get_xlim()
    return hi / 1.35 if side == "right" else lo * 1.35


def _annotate_label(ax, cross, below: bool):
    """The label of ``cross``, above it or ``below`` it."""
    _, y, col, text, side, *_ = cross
    dx, ha = (-LABEL_DX, "right") if side == "right" else (LABEL_DX, "left")
    return ax.annotate(text, (_cross_x(ax, side), y), textcoords="offset points",
                       xytext=(dx, -LABEL_DY if below else LABEL_DY), ha=ha,
                       va="top" if below else "bottom", color=col,
                       fontsize=plotstyle.TICK_SIZE - 1, zorder=7)


def _fail_label_sides(ax, crosses, fail_labels_for
                      ) -> Tuple[List[bool], List[Tuple[float, float]]]:
    """Whether each cross's label goes below it, and, for each label below
    that the axes' bottom edge cuts, (the height of its cross above that edge,
    the points it lacks).

    A label goes above its cross unless it would overlap a marker there (an
    arm's point or a cross) or a label placed before it, and would not below.
    Decided in display space, so the axes' limits and position must be final;
    an unlabelled cross gets False."""
    below = [False] * len(crosses)
    short: List[Tuple[float, float]] = []
    canvas = ax.figure.canvas
    if not crosses or not hasattr(canvas, "get_renderer"):
        return below, short
    renderer = canvas.get_renderer()
    pt = ax.figure.dpi / 72.0
    boxes = []                                    # (x0, y0, x1, y1), display space
    for line in ax.lines:
        if line.get_marker() in (None, "", "None", " "):
            continue
        r = (line.get_markersize() + line.get_markeredgewidth()) / 2 * pt
        for mx, my in ax.transData.transform(line.get_xydata()):
            boxes.append((mx - r, my - r, mx + r, my + r))
    r = (CROSS_SIZE + CROSS_EDGE) / 2 * pt
    for c in crosses:
        cx, cy = ax.transData.transform((_cross_x(ax, c[4]), c[1]))
        boxes.append((cx - r, cy - r, cx + r, cy + r))

    def extent(c, lower):
        t = _annotate_label(ax, c, lower)
        bb = t.get_window_extent(renderer)
        t.remove()
        return (bb.x0, bb.y0, bb.x1, bb.y1)

    def clear(b):
        return not any(b[0] < q[2] and q[0] < b[2] and b[1] < q[3] and q[1] < b[3]
                       for q in boxes)

    bottom = ax.get_window_extent(renderer).y0
    for i, c in enumerate(crosses):
        if fail_labels_for is not None and c[5] not in fail_labels_for:
            continue
        b = extent(c, False)
        if not clear(b) and clear(extent(c, True)):
            below[i] = True
            b = extent(c, True)
            lack = (bottom - b[1]) / pt + 2.0
            if lack > 0.0:
                cy = ax.transData.transform((_cross_x(ax, c[4]), c[1]))[1]
                short.append(((cy - bottom) / pt, lack))
        boxes.append(b)
    return below, short


def _lower_floor(ax, short: List[Tuple[float, float]]) -> bool:
    """Lower the bottom of ``ax``'s log y range so that each (p, d) of
    ``short``, a point p points above the bottom edge, rises d points above
    it. Returns whether the range changed."""
    lo, hi = (math.log10(v) for v in ax.get_ylim())
    h = ax.get_window_extent().height / (ax.figure.dpi / 72.0)
    # On an axis h points tall spanning s decades, extending the range down by
    # δ decades moves a point p points above the bottom edge to
    # (p s / h + δ) h / (s + δ) points, which is p + d for δ = d s / (h - p - d).
    step = max((d * (hi - lo) / (h - p - d) for p, d in short if h - p - d > 0.0),
               default=0.0)
    if step <= 0.0:
        return False
    ax.set_ylim(10.0 ** (lo - step), 10.0 ** hi)
    return True


def _place_fail_labels(axes, crosses_per_ax, fail_labels_for) -> List[List[bool]]:
    """``_fail_label_sides`` for each of ``axes``, which share their y range,
    after lowering that range until every label below its cross fits; the
    sides are decided at the final range."""
    for _ in range(8):
        decided = [_fail_label_sides(ax, cr, fail_labels_for)
                   for ax, cr in zip(axes, crosses_per_ax)]
        if not _lower_floor(axes[0], [s for _, short in decided for s in short]):
            return [below for below, _ in decided]
    return [_fail_label_sides(ax, cr, fail_labels_for)[0]
            for ax, cr in zip(axes, crosses_per_ax)]


def _draw_failure_marks(ax, crosses, fail_labels_for, below=None) -> List[Tuple[str, str]]:
    """Each cross at the panel edge (it has no time of its own), tied to its arm;
    called once the limits are fixed. ``below[i]`` puts the label of
    ``crosses[i]`` below it (``_place_fail_labels``); by default labels go
    above. Returns the (algorithm, label) drawn."""
    labels = []
    for i, cross in enumerate(crosses):
        x, y, col, text, side, algo, *_ = cross
        xe = _cross_x(ax, side)
        ax.plot([x, xe], [y, y], color=col, linestyle=(0, (1.5, 2.5)),
                linewidth=1.4, alpha=0.9, zorder=4)
        ax.plot([xe], [y], marker="x", color=col, markersize=CROSS_SIZE,
                markeredgewidth=CROSS_EDGE, linestyle="none", zorder=7, clip_on=False)
        if fail_labels_for is not None and algo not in fail_labels_for:
            continue
        _annotate_label(ax, cross, bool(below[i]) if below is not None else False)
        labels.append((algo, text))
    return labels


def _finish_axes(ax, placed, headroom=2.0, set_limits=True, floorroom=1.0) -> None:
    """Log scales, room above and below, and the size tags.

    y is labelled at every decade while it spans at most eleven, whatever the
    axes' height. A tag is placed above or below its point, whichever side
    clears every other tag and every marker in display space.
    ``set_limits=False`` for panels that share an axis, whose limits are set
    once from all of them."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    if set_limits:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo / floorroom, hi * headroom)

    taken = []
    markers = []
    for line in ax.lines:
        if line.get_marker() in (None, "", "None", " "):
            continue
        for mx, my in line.get_xydata():
            markers.append(tuple(ax.transData.transform((mx, my))))
    pt = ax.figure.dpi / 72.0                 # display pixels per point
    fs = plotstyle.TICK_SIZE
    half_w, box_h, r_mk = 0.95 * fs * pt, 0.95 * fs * pt, 5.5 * pt
    for x, y, dy, text, col in (placed or []):
        px, py0 = ax.transData.transform((x, y))
        for side in (dy, 12 if dy < 0 else -17):
            py = py0 + side * pt
            clash_tag = any(abs(px - qx) < 2 * half_w and abs(py - qy) < box_h
                            for qx, qy in taken)
            clash_mk = any(abs(px - mx) < half_w + r_mk
                           and py - r_mk < my < py + box_h + r_mk
                           for mx, my in markers
                           if not (abs(mx - px) < 1 and abs(my - py0) < 1))
            if not (clash_tag or clash_mk):
                taken.append((px, py))
                ax.annotate(text, (x, y), textcoords="offset points",
                            xytext=(0, side), ha="center", color=col,
                            fontsize=plotstyle.TICK_SIZE, zorder=6)
                break


def legend_extra_height(n_labels: int) -> float:
    """The height in inches of the row layout's colour legend beyond its first
    two rows."""
    rows = -(-n_labels // LEGEND_NCOL)
    return max(0, rows - 2) * LEGEND_ROW_IN


def _merge_handles(*handle_lists):
    """The union of (label, handle) pairs over panels; ours first, then by name."""
    seen, hs, ls = set(), [], []
    for hl in handle_lists:
        for lab, h in hl:
            if lab not in seen:
                seen.add(lab)
                hs.append(h)
                ls.append(lab)
    order = sorted(range(len(ls)), key=lambda i: (not ls[i].startswith("BDRS"), ls[i].lower()))
    return [hs[i] for i in order], [ls[i] for i in order]


def _size_handles(dataset, size_mk):
    return [Line2D([], [], marker=m, color="#555555", linestyle="none",
                   markersize=13 if m == "*" else 9, markeredgecolor="white",
                   label=_size_legend_label(dataset, n))
            for n, m in sorted(size_mk.items())]


def _summary(frames, crosses, labels, below) -> Dict:
    return {"panels": {nm: {"arms": sorted({(a, float(t)) for a, t in
                                            zip(d.algorithm, d.tolerance)}),
                            "points": sorted((a, float(t), int(s)) for a, t, s in
                                             zip(d.algorithm, d.tolerance, d["size"])),
                            "crosses": [(c[5], c[3], c[4]) for c in cr],
                            "labels": lb,
                            "labels_below": [(c[5], c[6], c[3]) for c, b in zip(cr, bl) if b]}
                       for (nm, d), cr, lb, bl in zip(frames, crosses, labels, below)}}


def _draw_row(source: CellSource, spec: TrajectorySpec, out: Path) -> Dict:
    """One panel per cost, sharing both axes, with the legends below."""
    frames, failsets = [], []
    for nm in spec.norms:
        frames.append((nm, load(source, spec, nm)))
        failsets.append(load_failures(source, spec, nm))
    size_mk = size_marker_map(pd.concat([d["size"] for _, d in frames]))
    ncol = len(frames)
    figw, figh = 3.5 * ncol + 0.9, 5.4
    fig, axes = plt.subplots(1, ncol, figsize=(figw, figh),
                             sharex=True, sharey=True, squeeze=False)
    multi_format = any(d.regime.nunique() > 1 for _, d in frames)
    pending, crosses, handle_lists = [], [], []
    for j, (nm, d) in enumerate(frames):
        ax = axes[0][j]
        ax.set_xscale("log")
        ax.set_yscale("log")
        pl, cr, hd = _draw_arms(ax, d, spec.dataset, multi_format, size_mk,
                                failsets[j], nm, spec.marks)
        pending.append(pl)
        crosses.append(cr)
        handle_lists.append(hd)
        ax.set_title(plotstyle.bold(NORM_TEX.get(nm, nm)), fontsize=plotstyle.LABEL_SIZE,
                     **plotstyle.bold_kw())
    # Each colour legend row beyond two adds its height to the figure and to
    # the bottom margin, so the axes keep their size.
    h, l = _merge_handles(*handle_lists)
    extra = legend_extra_height(len(l))
    if extra:
        figh = figh + extra
        fig.set_size_inches(figw, figh)

    ys = [v for _, d in frames for v in d[ECOL]]
    xs = [v for _, d in frames for v in d[XCOL]]
    pad_hi = 3.2 if spec.size_tags else 1.6
    pad_lo = 2.5 if spec.size_tags else 1.8
    xpad = 4.0 if any(crosses) else 2.0
    axes[0][0].set_ylim(min(ys) / pad_lo, max(ys) * pad_hi)
    axes[0][0].set_xlim(min(xs) / 2.0, max(xs) * xpad)
    # The final geometry before the tags are placed: they are placed in display space.
    margins = dict(bottom=(1.75 + extra) / figh, top=1.0 - 0.75 / figh, left=0.088,
                   right=0.988, wspace=0.08)
    fig.subplots_adjust(**margins)
    below = _place_fail_labels(axes[0], crosses, spec.fail_labels_for)
    labels = []
    for j in range(ncol):
        _finish_axes(axes[0][j], pending[j] if spec.size_tags else None, set_limits=False)
        labels.append(_draw_failure_marks(axes[0][j], crosses[j], spec.fail_labels_for,
                                          below[j]))
    axes[0][0].annotate(BETTER, xy=(0.97, 0.96), xycoords="axes fraction", ha="right",
                        va="top", fontsize=plotstyle.TICK_SIZE, color="#555555")

    fig.subplots_adjust(**margins)
    fig.supxlabel(plotstyle.bold(XLABEL), fontsize=plotstyle.LABEL_SIZE,
                  y=(1.08 + extra) / figh, **plotstyle.bold_kw())
    fig.text(0.001, 0.5 + (0.30 + extra / 2) / figh, plotstyle.bold(ELABEL), rotation=90,
             va="center", ha="left", fontsize=plotstyle.LABEL_SIZE, **plotstyle.bold_kw())
    fig.suptitle(plotstyle.bold(spec.title), fontsize=plotstyle.TITLE_SIZE,
                 y=1.0 - 0.26 / figh, **plotstyle.bold_kw())
    size_h = _size_handles(spec.dataset, size_mk)
    leg1 = fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                      ncol=min(len(l), LEGEND_NCOL), frameon=False,
                      fontsize=plotstyle.TICK_SIZE, handletextpad=0.6,
                      columnspacing=1.6, handlelength=1.2)
    fig.add_artist(leg1)
    fig.legend(size_h, [x.get_label() for x in size_h], loc="lower center",
               bbox_to_anchor=(0.5, (0.62 + extra) / figh), ncol=len(size_h),
               frameon=False, fontsize=plotstyle.TICK_SIZE, handletextpad=0.4,
               columnspacing=1.4, title=None)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return _summary(frames, crosses, labels, below)


def _draw_compact(source: CellSource, spec: TrajectorySpec, out: Path) -> Dict:
    """One panel for one cost, on a narrow canvas with both legends below."""
    (norm,) = spec.norms
    d = load(source, spec, norm)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.14)
    multi_format = d.regime.nunique() > 1
    size_mk = size_marker_map(d["size"])
    fails = load_failures(source, spec, norm)
    _, crosses, handles = _draw_arms(ax, d, spec.dataset, multi_format, size_mk,
                                     fails, norm, spec.marks)
    _finish_axes(ax, None, headroom=2.5, floorroom=1.8)
    if crosses:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo / 2.5 if any(m[4] == "left" for m in crosses) else lo,
                    hi * 2.5 if any(m[4] == "right" for m in crosses) else hi)
    (below,) = _place_fail_labels([ax], [crosses], spec.fail_labels_for)
    labels = _draw_failure_marks(ax, crosses, spec.fail_labels_for, below)
    plotstyle.label(ax, x=XLABEL, y=ELABEL, title=spec.title)
    h, l = _merge_handles(handles)
    sh = _size_handles(spec.dataset, size_mk)
    # Anchored in axes coordinates below the x label (about 0.15 of the axes height).
    y = -0.24
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, y),
               bbox_transform=ax.transAxes, frameon=False, borderaxespad=0.0, ncol=2,
               fontsize=plotstyle.TICK_SIZE, handlelength=1.2)
    fig.legend(sh, [x.get_label() for x in sh], loc="upper center",
               bbox_to_anchor=(0.5, y - 0.085 * (len(l) / 2 + 1.0)),
               bbox_transform=ax.transAxes, frameon=False, borderaxespad=0.0, ncol=4,
               fontsize=plotstyle.TICK_SIZE)
    ax.annotate(BETTER, xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
                fontsize=plotstyle.TICK_SIZE, color="#555555")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return _summary([(norm, d)], [crosses], [labels], [below])


def draw(source: CellSource, spec: TrajectorySpec, out: Path) -> Dict:
    """Draw ``spec`` to ``out``; returns the arms, points and crosses of each panel."""
    if spec.layout == "row":
        return _draw_row(source, spec, out)
    if spec.layout == "compact":
        return _draw_compact(source, spec, out)
    raise ValueError(f"unknown layout {spec.layout!r}")
