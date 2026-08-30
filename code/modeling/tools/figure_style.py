"""Shared publication styling for VLST figure panels (Harmony palette)."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ggsci Harmony qualitative palette (14 colors)
HARMONY = [
    "#DC0000",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#7AC5CD",
    "#008280",
    "#BC3C29",
    "#0072B5",
    "#E18727",
    "#20854E",
    "#7876B1",
    "#6F99AD",
    "#FFDC91",
    "#EE4C97",
]

FAMILY_COLORS = {
    "baseline": HARMONY[7],
    "advanced": HARMONY[8],
    "tabpfn": HARMONY[9],
    "qc": HARMONY[5],
    "negative": HARMONY[2],
    "positive": HARMONY[0],
    "neutral": HARMONY[11],
}

METRIC_COLORS = {
    "precision": HARMONY[7],
    "recall": HARMONY[9],
    "f1": HARMONY[8],
    "pr_auc": HARMONY[5],
    "roc_auc": HARMONY[11],
}

FONT_FAMILY = "DejaVu Sans"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    plt.style.use("seaborn-v0_8-whitegrid")


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(fig, path, *, formats=("png", "pdf")) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"))
    plt.close(fig)
