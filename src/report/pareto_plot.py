"""Accuracy-time Pareto panels from ``pareto_cells.csv``: one log-log PDF per
(dataset, rung, size, norm) cell, x = t_solve (SGM10, s), y = the KKT residual
E_abs (geometric mean).

Our solvers are drawn as one curve through their tolerance arms, each baseline
(run at its own native stop) as one point. An (algorithm, tolerance) row
without both coordinates is named, with its reason, in the panel's "absent"
note.

    python -m src.report.pareto_plot --cells CELLS_CSV --plots-dir DIR
                                     [--datasets NAME ...]
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.protocol.regimes import OUR_SOLVERS, canonical_algorithm, regime_for_algorithm

# The y-axis: its pareto_cells.csv column and its label.
Y_COLUMN = "E_abs_gm"
Y_LABEL = "KKT residual $E_{\\mathrm{abs}}$ (geometric mean)"


def _f(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def read_cells(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def panel_key(row: Dict[str, Any]) -> Tuple[str, str, Any, str]:
    """One panel per aggregation cell."""
    return (row.get("dataset") or "dotmark", row.get("rung") or "",
            row.get("size") or "", row.get("norm") or "")


def split_rows(rows: Sequence[Dict[str, Any]]
               ) -> Tuple[Dict[str, List[Dict[str, Any]]],
                          List[Dict[str, Any]], List[Tuple[str, str]]]:
    """(our curves by solver, baseline points, absences with reasons). A row
    plots only with a positive finite t_solve_sgm10_s and E_abs_gm."""
    ours: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    base: List[Dict[str, Any]] = []
    gone: List[Tuple[str, str]] = []
    for r in rows:
        algo = canonical_algorithm(r.get("algorithm") or "")
        x, y = _f(r.get("t_solve_sgm10_s")), _f(r.get(Y_COLUMN))
        if x is None or y is None or x <= 0 or y <= 0:
            why = (r.get("unscorable_reasons") or "").strip()
            if not why:
                n_oom = int(_f(r.get("n_oom")) or 0)
                n_to = int(_f(r.get("n_timeout")) or 0)
                why = ("device/host OOM" if n_oom else
                       "time limit" if n_to else "no scorable record")
            gone.append((algo, why))
            continue
        pt = {"algo": algo, "x": x, "y": y,
              "tol": _f(r.get("tolerance")), "n": int(_f(r.get("n_E_abs")) or 0)}
        base.append(pt) if _base_algorithm_of(algo) not in OUR_SOLVERS else ours[algo].append(pt)
    for a in ours:
        # The curve joins the arms in tolerance order, loosest (normally fastest) first.
        ours[a].sort(key=lambda p: (-(p["tol"] or 0.0), p["x"]))
    return dict(ours), base, sorted(set(gone))


def _base_algorithm_of(algo: str) -> str:
    try:
        regime_for_algorithm(algo)
    except ValueError:
        return algo
    from src.protocol.regimes import base_algorithm
    return base_algorithm(algo)


def draw_panel(ax, rows: Sequence[Dict[str, Any]], title: str) -> None:
    ours, base, gone = split_rows(rows)
    for p in base:
        ax.scatter(p["x"], p["y"], marker="o", s=44, zorder=3)
        ax.annotate(p["algo"], (p["x"], p["y"]), textcoords="offset points",
                    xytext=(5, 4), fontsize=7)
    for algo, pts in sorted(ours.items()):
        ax.plot([p["x"] for p in pts], [p["y"] for p in pts],
                marker="s", ms=5, lw=1.6, zorder=4, label=f"{algo} (eps sweep)")
        for p in pts:
            if p["tol"]:
                ax.annotate(f"{p['tol']:.0e}", (p["x"], p["y"]),
                            textcoords="offset points", xytext=(4, -9), fontsize=6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$t_{\\mathrm{solve}}$ (s, SGM10)")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    if ours:
        ax.legend(fontsize=7, loc="best")
    if gone:
        ax.text(0.02, 0.02, "absent: " + "; ".join(f"{a} ({w})" for a, w in gone),
                transform=ax.transAxes, fontsize=6, va="bottom", alpha=0.75)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=Path, required=True,
                   help="summaries/pareto_cells.csv")
    p.add_argument("--plots-dir", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    a = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_cells(a.cells)
    if a.datasets:
        rows = [r for r in rows if (r.get("dataset") or "dotmark") in a.datasets]
    panels: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        panels[panel_key(r)].append(r)

    a.plots_dir.mkdir(parents=True, exist_ok=True)
    wrote = 0
    for key, prs in sorted(panels.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        dataset, rung, size, norm = key
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        draw_panel(ax, prs, f"{dataset} · {rung} · N={size} · {norm}")
        fig.tight_layout()
        out = a.plots_dir / f"pareto_{dataset}_{rung}_{size}_{norm}.pdf"
        fig.savefig(out); plt.close(fig)
        print(f"pareto_plot: wrote {out}")
        wrote += 1
    if not wrote:
        print("pareto_plot: no panels (empty or filtered-out cells file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
