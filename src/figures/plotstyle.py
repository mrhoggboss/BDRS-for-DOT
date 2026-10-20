"""The figures' house style.

Text is set in Computer Modern: through LaTeX when it is usable, otherwise
through matplotlib's mathtext with its bundled Computer Modern fonts. The axes
panel is grey with a white grid on a white figure; axis labels and titles are
bold, legend text is not.

``BDRS_FIG_TEX`` selects the text renderer: ``1`` requires LaTeX, ``0`` uses
the fallback, and unset or empty uses LaTeX when ``latex`` is on PATH and a
probe figure compiles.

Under LaTeX, matplotlib ignores ``axes.labelweight``, so bold is written into
the string (``{\\boldmath\\textbf{...}}``, which also makes the math bold). In
the fallback, :func:`bold` and :func:`bold_kw` set the words in the bold face
``cmb10`` and leave math regular, since no bold Computer Modern math is
bundled.
"""

import functools
import os
import shutil
import tempfile
import warnings
from typing import Any, Dict, Optional, Tuple

import matplotlib

LABEL_SIZE = 17
TITLE_SIZE = 19
TICK_SIZE = 13
BASE_SIZE = 13

AXES_BG = "#E6E6E6"
GRID_COLOR = "#FFFFFF"
FIGURE_BG = "#FFFFFF"

TEX_ENV = "BDRS_FIG_TEX"
_PREAMBLE = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}"
_BOLD_FONT = "cmb10"


@functools.lru_cache(maxsize=1)
def latex_available() -> Tuple[bool, str]:
    """(usable, reason): ``latex`` is on PATH and a probe figure with bold text
    and math under the house preamble saves as PDF."""
    if shutil.which("latex") is None:
        return False, "latex not on PATH"
    import matplotlib.pyplot as plt
    saved = matplotlib.rcParams.copy()
    try:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["text.latex.preamble"] = _PREAMBLE
        fig = plt.figure(figsize=(0.6, 0.6))
        fig.text(0.5, 0.5, r"{\boldmath\textbf{A $x_1$}}")
        try:
            with tempfile.TemporaryDirectory() as td, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig.savefig(os.path.join(td, "probe.pdf"))
        finally:
            plt.close(fig)
        return True, ""
    except Exception as exc:                                      # noqa: BLE001
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        return False, f"probe failed: {type(exc).__name__}: {first}"[:200]
    finally:
        matplotlib.rcParams.update(saved)


def resolve_usetex(env: Optional[str] = None) -> Tuple[bool, str]:
    """(usetex, why) from ``BDRS_FIG_TEX`` (or ``env`` when given).

    Raises ``RuntimeError`` when LaTeX is required but unusable and
    ``ValueError`` for a value other than 0, 1 or empty."""
    value = (os.environ.get(TEX_ENV, "") if env is None else env).strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{TEX_ENV} must be 0, 1 or empty, got {value!r}")
    if value == "0":
        return False, f"{TEX_ENV}=0"
    ok, reason = latex_available()
    if value == "1":
        if not ok:
            raise RuntimeError(f"{TEX_ENV}=1 but LaTeX is unusable ({reason})")
        return True, f"{TEX_ENV}=1"
    return ok, ("LaTeX found" if ok else reason)


def rc_params(usetex: bool) -> Dict[str, Any]:
    rc: Dict[str, Any] = {
        "font.family": "serif",
        "font.size": BASE_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "figure.titlesize": TITLE_SIZE,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "figure.titleweight": "bold",
        "legend.fontsize": LABEL_SIZE,
        "legend.title_fontsize": LABEL_SIZE,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "figure.facecolor": FIGURE_BG,
        "savefig.facecolor": FIGURE_BG,
        "axes.facecolor": AXES_BG,
        "axes.edgecolor": "#4D4D4D",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 1.0,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#4D4D4D",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    }
    if usetex:
        rc["text.usetex"] = True
        rc["text.latex.preamble"] = _PREAMBLE
    else:
        rc["text.usetex"] = False
        rc["mathtext.fontset"] = "cm"
        rc["font.serif"] = ["cmr10"]
        # cmr10 has no U+2212; the mathtext tick formatter draws the minus.
        rc["axes.unicode_minus"] = False
        rc["axes.formatter.use_mathtext"] = True
    return rc


def use(usetex: bool) -> None:
    """Apply the house style for the given text renderer."""
    matplotlib.rcParams.update(rc_params(usetex))


def usetex_active() -> bool:
    return bool(matplotlib.rcParams.get("text.usetex", False))


def bold(text: str) -> str:
    """``text`` (words with ``$...$`` math) marked bold for the active renderer.
    In the fallback the words also need the font of :func:`bold_kw`."""
    if usetex_active():
        return rf"{{\boldmath\textbf{{{text}}}}}"
    return text


def bold_kw() -> Dict[str, Any]:
    """Text properties that go with :func:`bold`: none under LaTeX, the bold
    Computer Modern face in the fallback."""
    return {} if usetex_active() else {"fontfamily": _BOLD_FONT}


def label(ax, x: Optional[str] = None, y: Optional[str] = None,
          title: Optional[str] = None, **kwargs) -> None:
    """Bold axis labels and title."""
    kw = {**bold_kw(), **kwargs}
    if x is not None:
        ax.set_xlabel(bold(x), **kw)
    if y is not None:
        ax.set_ylabel(bold(y), **kw)
    if title is not None:
        ax.set_title(bold(title), **kw)
