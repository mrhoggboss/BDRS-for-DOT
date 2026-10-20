"""The dual stall of BDRS, DROT and HOT: one panel per (solver, n_g).

Each panel draws, against the iteration k, the medians over a cohort of
instances of the native triple's residuals r_P^k, |g^k| and r_D^k (Frobenius),
and of the KKT residual max{r_P, r_D, |g|} of the corrected triple
(X^k, α', β'), whose dual is the c-transform of α^k. The KKT residual is taken
per instance, then the median. A dotted line marks tol, and a dashed line the
median iteration at which the corrected triple first meets tol on every term
(drawn when more than half of the cohort meets it).

The cohort is the instances whose last check is at or after the median last
check, so the population is the same at every point of a curve; the curves
keep the checks at which every cohort member is finite on every field drawn.

Input: ``<runs>/n<n_g>/runs/<SOLVER>.jsonl`` with the ``instance`` and
``check`` rows of ``src.studies.dual_stall``.
"""

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

from src.figures import plotstyle

PANELS = [("BDRS", "BDRS (ours)"), ("DROT", "DROT"), ("HOT", "HOT")]

CURVE_KEYS = ("native_r_P", "native_gap", "native_r_DF")
MERGED_CURVES = [
    ("native_r_P", r"$r_P^k$", "o", "#1f77b4", 1.5),
    ("native_gap", r"$|g^k|$", "s", "#ff7f0e", 1.5),
    ("native_r_DF", r"$r_D^k$", "^", "#2ca02c", 1.5),
    ("corrected", r"$\mathrm{KKT}(X^k,\alpha',\beta')$", "D", "#c51b7d", 2.2),
]
KKT_KEYS = ["native_r_P", "native_r_DF", "native_gap", "repaired_r_DF", "repaired_gap"]

# Font and tick sizes on the page: the figure is included at the text width,
# unscaled.
COMBINED_FONTS = {
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.5,
    "font.size": 7.0,
    "lines.markersize": 3.2,
    "grid.linewidth": 0.6,
    "axes.linewidth": 0.7,
    "xtick.major.size": 2.4, "xtick.minor.size": 1.2,
    "ytick.major.size": 2.4, "ytick.minor.size": 1.2,
    "xtick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.major.width": 0.6, "ytick.minor.width": 0.4,
    "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
}
_PANEL_LETTERS = "abcdefghijkl"

# The fields of each triple; the primal residual and p_obj are the native ones.
_ARM_FIELDS = {"native": ("native_r_DF", "native_gap", "native_d_obj"),
               "repaired": ("repaired_r_DF", "repaired_gap", "repaired_d_obj")}


def kkt_of(row, arm: str) -> float:
    """max{r_P, r_D, |g|} of the native or the corrected triple at one check."""
    r_p = abs(float(row["native_r_P"]))
    if arm == "native":
        return max(r_p, abs(float(row["native_r_DF"])), abs(float(row["native_gap"])))
    return max(r_p, abs(float(row["repaired_r_DF"])), abs(float(row["repaired_gap"])))


def meets_tol(row, eps: float, arm: str) -> Optional[bool]:
    """r_P ≤ eps, r_D ≤ eps and |g| ≤ eps for the arm's triple; None when the
    row lacks a field."""
    r_df, gap, d_obj = _ARM_FIELDS[arm]
    vals = [row.get(k) for k in ("native_r_P", r_df, gap, "native_p_obj", d_obj)]
    if any(v is None for v in vals):
        return None
    return bool(vals[0] <= eps and vals[1] <= eps and abs(vals[2]) <= eps)


def load_solver(runs_dir: Path, solver: str, size: int):
    """(instance rows by id, check rows by instance at ``size`` in iteration order)."""
    path = Path(runs_dir) / f"{solver}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    insts, checks = {}, defaultdict(list)
    with path.open() as fh:
        lines = [line for line in fh if line.strip()]
    for line in lines:
        row = json.loads(line)
        if row.get("type") == "instance":
            insts[row["instance_id"]] = row
            continue
        if row.get("type") != "check":
            continue
        inst = row["instance_id"]
        if f"_n{size}_" not in inst:
            continue
        missing = [k for k in CURVE_KEYS if k not in row]
        if missing:
            raise ValueError(f"{path}: check row at iter {row.get('iter')} lacks {missing}")
        checks[inst].append(row)
    if not checks:
        raise ValueError(f"{path}: no check rows at n_g={size}")
    for rows in checks.values():
        rows.sort(key=lambda r: int(r["iter"]))
    return insts, dict(checks)


def diverges(rows) -> bool:
    """The native primal residual leaves the unit ball or is not finite."""
    vals = [float(r["native_r_P"]) for r in rows]
    finite = [v for v in vals if math.isfinite(v)]
    return len(finite) < len(vals) or (bool(finite) and max(finite) > 1.0)


def cohort(checks, keys):
    """(members, their checks by iteration, the checks kept, the number cut)."""
    lasts = sorted(int(rows[-1]["iter"]) for rows in checks.values())
    horizon = lasts[len(lasts) // 2]
    members = {i: rows for i, rows in checks.items() if int(rows[-1]["iter"]) >= horizon}
    per = {i: {int(r["iter"]): r for r in rows} for i, rows in members.items()}
    grid = [k for k in sorted(set.intersection(*(set(d) for d in per.values())))
            if k <= horizon]
    keep, dropped = [], 0
    for k in grid:
        vals = [abs(float(per[i][k][key])) for i in per for key in keys]
        if all(math.isfinite(v) for v in vals):
            keep.append(k)
        else:
            dropped += 1
    if not keep:
        raise ValueError("every cohort check has a non-finite value")
    return members, per, keep, dropped


def first_pass_iter(rows, eps: float, arm: str) -> Optional[int]:
    for row in rows:
        if meets_tol(row, eps, arm):
            return int(row["iter"])
    return None


def _decade_ticks(lo: float, hi: float, step: int):
    """Powers of ten every ``step`` decades, counted down from the top of the range."""
    e_lo, e_hi = math.ceil(math.log10(lo)), math.floor(math.log10(hi))
    return [10.0 ** e for e in sorted(range(e_hi, e_lo - 1, -step))]


def _merged_panel(ax, runs_dir: Path, solver: str, size: int, *, eps: float):
    """One solver at one size. Returns (stats, lowest and highest value drawn)."""
    _, checks = load_solver(runs_dir, solver, size)
    n_total = len(checks)
    n_div = sum(diverges(rows) for rows in checks.values())
    members, per, ks, dropped = cohort(checks, KKT_KEYS)
    n_coh = len(members)

    med = {key: [st.median([abs(float(per[i][k][key])) for i in per]) for k in ks]
           for key in CURVE_KEYS}
    med["corrected"] = [st.median([kkt_of(per[i][k], "corrected") for i in per]) for k in ks]
    med["native"] = [st.median([kkt_of(per[i][k], "native") for i in per]) for k in ks]
    zeros = {key: sum(1 for v in med[key] if v <= 0.0) for key in med}

    every = max(1, len(ks) // 9)
    for key, label, marker, color, lw in MERGED_CURVES:
        ys = [v if v > 0.0 else float("nan") for v in med[key]]
        line, = ax.loglog(ks, ys, marker=marker, label=label, color=color,
                          linewidth=lw, markevery=every)
        # A median that reaches exactly 0 leaves the log axis; say so where it does.
        if zeros[key] and not all(v <= 0.0 for v in med[key]):
            j = max(i for i, v in enumerate(med[key]) if v > 0.0)
            ax.annotate(label.rstrip("$") + r"\to 0$", (ks[j], med[key][j]),
                        textcoords="offset points", xytext=(4, -10),
                        color=line.get_color(), fontsize=5.5)

    ax.axhline(eps, color="0.25", linestyle=":", linewidth=1.0, zorder=1)
    stops = [first_pass_iter(rows, eps, "repaired") for rows in members.values()]
    rescued = sorted(v for v in stops if v is not None)
    k_stop = st.median(rescued) if len(rescued) > n_coh / 2 else None
    if k_stop is not None and ks[0] <= k_stop <= ks[-1]:
        ax.axvline(k_stop, color="0.35", linestyle="--", linewidth=1.0)
    n_nat = sum(first_pass_iter(rows, eps, "native") is not None for rows in members.values())

    vals = [v for key in ("native_r_P", "native_gap", "native_r_DF", "corrected")
            for v in med[key] if v > 0.0 and math.isfinite(v)]
    stats = dict(solver=solver, size=size, n_total=n_total, n_coh=n_coh, n_div=n_div,
                 ks=ks, dropped=dropped, zeros=zeros, k_stop=k_stop, n_nat=n_nat,
                 n_rescued=len(rescued), med=med)
    return stats, min(vals), max(vals)


def draw(runs: Path, out: Path, *, sizes: Sequence[int] = (32, 64), eps: float = 1e-4,
         ylim: Optional[Tuple[float, float]] = None, xmin: float = 1.0,
         width: float = 4.69, panel_h: float = 1.34) -> Dict:
    """Rows are the sizes, columns the solvers, from ``runs/n<size>/runs``.

    ``width`` is the text width in inches. ``ylim=None`` shares the y-axis down
    each column; ``ylim=(lo, hi)`` puts every panel on that axis (curves below
    ``lo`` are cut). Every panel spans iterations ``xmin`` to 1.15× the last check."""
    runs = Path(runs)
    nrow, ncol = len(sizes), len(PANELS)
    with plt.rc_context(COMBINED_FONTS):
        fig = plt.figure(figsize=(width, panel_h * nrow + 1.12), layout="constrained")
        subfigs = fig.subfigures(nrow, 1, hspace=0.0)
        subfigs = [subfigs] if nrow == 1 else list(subfigs)
        axes = []
        for ri, size in enumerate(sizes):
            row = subfigs[ri].subplots(1, ncol, sharex=True, squeeze=False)[0]
            subfigs[ri].suptitle(plotstyle.bold(rf"$n_g = {size}$"), fontsize=9.0,
                                 **plotstyle.bold_kw())
            axes.append(list(row))

        allstats, extent, own = [], {}, {}
        for ri, size in enumerate(sizes):
            for ci, (solver, title) in enumerate(PANELS):
                ax = axes[ri][ci]
                stats, lo, hi = _merged_panel(ax, runs / f"n{size}" / "runs", solver, size,
                                              eps=eps)
                allstats.append(stats)
                own[(ri, ci)] = (lo, hi)
                prev = extent.get(ci)
                extent[ci] = (min(lo, prev[0]), max(hi, prev[1])) if prev else (lo, hi)
                letter = _PANEL_LETTERS[ri * ncol + ci]
                name = title + (", diverges" if stats["n_div"] else "")
                plotstyle.label(ax, title=rf"({letter}) {name}")
                if ci == 0:
                    plotstyle.label(ax, y="Residual, median")
                if ri == nrow - 1:
                    plotstyle.label(ax, x=r"Iteration $k$")

        kmax = max(s["ks"][-1] for s in allstats)
        if ylim is not None:
            extent = {ci: ylim for ci in range(ncol)}
        for ci, (lo, hi) in extent.items():
            span = math.log10(hi) - math.log10(lo)
            # The same decades in every row: the log locator depends on the axis height.
            step = max(1, math.ceil((span * 1.12) / 5.0))
            for ri in range(nrow):
                ax = axes[ri][ci]
                if ylim is not None:
                    ax.set_ylim(lo, hi)
                else:
                    ax.set_ylim(10 ** (math.log10(lo) - span * 0.06),
                                10 ** (math.log10(hi) + span * 0.06))
                ax.set_xlim(xmin, kmax * 1.15)
                ax.yaxis.set_major_locator(FixedLocator(_decade_ticks(*ax.get_ylim(), step)))
                ax.yaxis.set_minor_formatter(NullFormatter())

        handles, labels = axes[0][0].get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color="0.25", linestyle=":", linewidth=1.0))
        labels.append(r"tolerance $\mathrm{tol}$")
        handles.append(plt.Line2D([], [], color="0.35", linestyle="--", linewidth=1.0))
        labels.append(r"corrected triple meets $\mathrm{tol}$")
        fig.legend(handles, labels, loc="outside lower center", ncol=3,
                   columnspacing=1.4, handlelength=1.6, handletextpad=0.35,
                   borderpad=0.35, labelspacing=0.3)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)

    for s, letter in zip(allstats, _PANEL_LETTERS):
        print(f"  ({letter}) {s['solver']:5s} n_g={s['size']:<4d} cohort "
              f"{s['n_coh']}/{s['n_total']}, {len(s['ks'])} checks "
              f"({s['ks'][0]}..{s['ks'][-1]}), diverging {s['n_div']}/{s['n_total']}; "
              f"native meets tol on {s['n_nat']}/{s['n_coh']}, corrected on "
              f"{s['n_rescued']}/{s['n_coh']}"
              + (f", corrected crossing median k={s['k_stop']:g}"
                 if s["k_stop"] is not None else ""))
        if s["dropped"]:
            print(f"        cut {s['dropped']} later checks: a cohort member is not finite")
        for key, nz in s["zeros"].items():
            if nz:
                print(f"        {key} is exactly 0 at {nz} checks (gaps on the log axis)")
    if ylim is not None:
        for (ri, ci), (p_lo, _) in sorted(own.items()):
            if p_lo < ylim[0]:
                print(f"  panel ({_PANEL_LETTERS[ri * ncol + ci]}): curves continue below "
                      f"{ylim[0]:g}, down to {p_lo:.1e}")
    return {"panels": [{k: s[k] for k in ("solver", "size", "n_coh", "n_total", "n_div",
                                          "k_stop", "n_nat", "n_rescued")}
                       for s in allstats]}
