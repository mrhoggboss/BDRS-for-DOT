"""Robustness of the Grid solver to its two constants, one figure each.

Step size: iterations to the stop against the multiplier m of η = m·0.7·mean(C),
relative to the same instance at m = 1. Check period: solve time against the
check period T_chk, relative to the same instance at its deployed T_chk. Each
curve is one (cost, n_g) and plots the median ratio over its instances; only
runs with status ok enter.

Input: ``<tree>/eta_sweep/runs/sweep.jsonl`` (rows with ``sweep == "eta"``)
and ``<tree>/tchk_sweep/runs/sweep.jsonl``, as ``src.studies.robustness``
writes them.
"""

import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.figures import plotstyle

COST_TEX = {"l1": r"$\ell_1$", "sql2": r"$\ell_2^2$"}
STYLE = {("l1", 32): ("#8c564b", "v"), ("l1", 64): ("#1b7f3b", "o"),
         ("l1", 128): ("#1f77b4", "s"),
         ("sql2", 32): ("#e377c2", "P"), ("sql2", 64): ("#d62728", "^"),
         ("sql2", 128): ("#9467bd", "D")}

# The deployed check period T_chk at n_g (src/runners/bdrs_common.CHECK_EVERY_BY_N).
PROD_T = {32: 100, 64: 200, 128: 500, 256: 1000}

TCHK_FIELD = "wall_time_solve_s"


def load(path: Path) -> List[Dict]:
    with Path(path).open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def provenance(tree: Path, rows) -> str:
    """The tree, campaign tags, commits and row count behind a figure, which
    the drawing prints."""
    shas = sorted({(r.get("git_sha") or "?")[:12] for r in rows})
    tags = sorted({r.get("campaign_tag") or "untagged" for r in rows})
    return f"tree={Path(tree).name} tag={','.join(tags)} git_sha={','.join(shas)} rows={len(rows)}"


def _ratio_axis(ax) -> None:
    """Majors on 1, 2 and 5 of each decade, labelled as plain numbers."""
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="all", numticks=100))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def draw_eta(tree: Path, out: Path) -> Dict:
    """Relative iterations against the η multiplier."""
    tree = Path(tree)
    allrows = load(tree / "eta_sweep" / "runs" / "sweep.jsonl")
    eta = [r for r in allrows if r["solver"] == "Grid" and r.get("sweep", "eta") == "eta"]
    n_not_ok = sum(r.get("status") != "ok" for r in eta)
    eta = [r for r in eta if r.get("status") == "ok"]
    fig, ax = plt.subplots(1, 1, figsize=(6.3, 5.0))
    by = defaultdict(dict)
    for r in eta:
        by[(r["norm"], r["resolution"], r["class"], r["ia"], r["ib"])][r["multiplier"]] = r["iters"]
    curves = defaultdict(lambda: defaultdict(list))
    for (norm, res, *_), d in by.items():
        base = d.get(1.0)
        if not base:
            continue
        for m, w in d.items():
            curves[(norm, res)][m].append(w / base)
    for key in sorted(curves, key=lambda k: (k[0], k[1])):
        col, mk = STYLE[key]
        ms = sorted(curves[key])
        ax.plot(ms, [st.median(curves[key][m]) for m in ms], marker=mk, color=col,
                markersize=8, linewidth=2.0, markeredgecolor="white",
                markeredgewidth=1.0, label=rf"{COST_TEX[key[0]]}, $n_g={key[1]}$")
    ax.axhline(1.0, color="0.55", linestyle="-", linewidth=1.0, zorder=1)
    ax.axvline(1.0, color="0.35", linestyle="--", linewidth=1.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    plotstyle.label(ax, x=r"$\eta$ as a multiple of the deployed value",
                    y="Relative iterations, median", title="Step size")
    ax.legend(loc="lower right", fontsize=plotstyle.TICK_SIZE - 3, handlelength=1.4,
              borderpad=0.3, labelspacing=0.3, handletextpad=0.4)
    _ratio_axis(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {provenance(tree, eta)}")
    print(f"  {len(eta)} Grid runs in {len(curves)} cells"
          + (f"; {n_not_ok} runs not ok, left out" if n_not_ok else ""))
    medians = {key: {m: st.median(v) for m, v in sorted(curves[key].items())}
               for key in sorted(curves)}
    for key, row in medians.items():
        print(f"    {key}: " + " ".join(f"{m:g}:{v:.6g}" for m, v in row.items()))
    return {"medians": medians, "n_runs": len(eta), "n_not_ok": n_not_ok}


def draw_tchk(tree: Path, out: Path) -> Dict:
    """Relative solve time against T_chk, with a ring at each deployed T_chk."""
    tree = Path(tree)
    rows = [r for r in load(tree / "tchk_sweep" / "runs" / "sweep.jsonl")
            if r["solver"] == "Grid"]
    n_not_ok = sum(r.get("status") != "ok" for r in rows)
    rows = [r for r in rows if r.get("status") == "ok"]
    fig, ax = plt.subplots(1, 1, figsize=(6.3, 5.0))
    tb = defaultdict(dict)
    for r in rows:
        k = (r["solver"], r["norm"], r["resolution"], r["class"], r["ia"], r["ib"])
        tb[k][r["T_chk"]] = r[TCHK_FIELD]
    t = defaultdict(lambda: defaultdict(list))
    for (solver, norm, res, *_), d in tb.items():
        base = d.get(PROD_T.get(res))
        if not base:
            continue
        for T, w in d.items():
            t[(solver, norm, res)][T].append(w / base)
    for key in sorted(t, key=lambda k: (k[0], k[1], k[2])):
        _, norm, res = key
        col, mk = STYLE[(norm, res)]
        ts = sorted(t[key])
        ax.plot(ts, [st.median(t[key][x]) for x in ts], marker=mk, color=col,
                markersize=7, linestyle="-", linewidth=2.0,
                markeredgecolor="white", markeredgewidth=1.0,
                label=rf"{COST_TEX[norm]}, $n_g={res}$")
    for res in sorted({k[2] for k in t}):
        ax.plot([PROD_T[res]], [1.0], marker="o", markersize=15, markerfacecolor="none",
                markeredgecolor="black", markeredgewidth=1.6, linestyle="none", zorder=6)
        tag = rf"$n_g{{=}}{res}$" if plotstyle.usetex_active() else rf"$n_g\!=\!{res}$"
        ax.annotate(tag, (PROD_T[res], 1.0), textcoords="offset points",
                    xytext=(0, -20), ha="center", va="top", fontsize=plotstyle.TICK_SIZE - 2,
                    color="black")
    ax.plot([], [], marker="o", markersize=10, markerfacecolor="none", markeredgecolor="black",
            markeredgewidth=1.6, linestyle="none", label=r"deployed $T_{\mathrm{chk}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plotstyle.label(ax, x=r"Check period $T_{\mathrm{chk}}$ (iterations)",
                    y="Relative solve time, median", title="Check period")
    ax.legend(loc="upper left", fontsize=plotstyle.TICK_SIZE - 3, ncol=2, handlelength=1.4,
              borderpad=0.3, labelspacing=0.3, handletextpad=0.4, columnspacing=0.8)
    ax.axhline(1.0, color="0.55", linestyle="-", linewidth=1.0, zorder=1)
    _ratio_axis(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  {provenance(tree, rows)}")
    print(f"  {sum(len(v) for d in t.values() for v in d.values())} runs in {len(t)} cells"
          + (f"; {n_not_ok} runs not ok, left out" if n_not_ok else ""))
    medians = {key: {T: st.median(v) for T, v in sorted(t[key].items())} for key in sorted(t)}
    for key, row in medians.items():
        print(f"    {key}: " + " ".join(f"{x:g}:{v:.6g}" for x, v in row.items()))
    return {"medians": medians, "n_runs": len(rows), "n_not_ok": n_not_ok}
