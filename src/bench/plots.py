"""Plotting helpers: measured points with a theoretical curve overlaid.

The brief rewards "predicted-vs-measured" figures. Each theoretical curve is
fitted by a single constant c (least-squares on c * g(n)) so it sits on the same
axes as the measured timings -- we are checking the *shape/growth rate*, not the
constant factor, which is what Big-O describes.
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: render straight to files, no display needed
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "figures")

# Named growth functions for overlays.
GROWTH: Dict[str, Callable[[float], float]] = {
    "O(1)": lambda n: 1.0,
    "O(log n)": lambda n: math.log2(n) if n > 1 else 1.0,
    "O(n)": lambda n: n,
    "O(n log n)": lambda n: n * math.log2(n) if n > 1 else 1.0,
    "O(n^2)": lambda n: n * n,
}


def _fit_constant(ns: Sequence[float], ys: Sequence[float], g: Callable[[float], float]) -> float:
    """Least-squares constant c minimising sum (y - c*g(n))^2 -> c = <y,g>/<g,g>."""
    gs = [g(n) for n in ns]
    num = sum(y * gv for y, gv in zip(ys, gs))
    den = sum(gv * gv for gv in gs)
    return num / den if den else 0.0


def ensure_fig_dir() -> str:
    path = os.path.abspath(FIG_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def save(fig, name: str) -> str:
    """Save a figure into figures/ and return its absolute path."""
    out = os.path.join(ensure_fig_dir(), name)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")
    return out


def plot_measured_vs_theory(
    series: Dict[str, Dict[str, List[float]]],
    *,
    title: str,
    filename: str,
    theory_for: Dict[str, str],
    xlabel: str = "input size n",
    ylabel: str = "time (ms)",
    logx: bool = False,
    logy: bool = False,
) -> str:
    """Plot one or more measured series, each with its theoretical overlay.

    ``series``    : label -> {"n": [...], "ms": [...]}
    ``theory_for``: label -> key in GROWTH (the predicted growth for that series)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data in series.items():
        ns, ys = data["n"], data["ms"]
        points = ax.plot(ns, ys, "o", markersize=5, label=f"{label} (measured)")
        colour = points[0].get_color()
        gkey = theory_for.get(label)
        if gkey:
            g = GROWTH[gkey]
            c = _fit_constant(ns, ys, g)
            smooth = list(range(min(ns), max(ns) + 1, max(1, (max(ns) - min(ns)) // 200)))
            ax.plot(smooth, [c * g(n) for n in smooth], "--", color=colour,
                    alpha=0.7, label=f"{label} ~ {gkey}")
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return save(fig, filename)
