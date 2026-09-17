"""Single write location for paper artefacts.

All notebooks and hygiene scripts should write figures/CSVs only under
``paper_results/<part>/paper_figures``. Do not mirror into ``code/`` or
``data/result/``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

PARTS = {
    "01_eda": ROOT / "paper_results" / "01_eda" / "paper_figures",
    "02_ml_selectors": ROOT / "paper_results" / "02_ml_selectors" / "paper_figures",
    "03_stats_vs_ml": ROOT / "paper_results" / "03_stats_vs_ml" / "paper_figures",
    "04_tabpfn_rating": ROOT / "paper_results" / "04_tabpfn_rating" / "paper_figures",
    "05_tabpfn_interpretability": ROOT / "paper_results" / "05_tabpfn_interpretability" / "paper_figures",
}


def paper_figures(part: str) -> Path:
    path = PARTS[part]
    path.mkdir(parents=True, exist_ok=True)
    return path


def paper_figure_dirs(part: str) -> list[Path]:
    """One-element list so existing ``for d in DIRS`` writers stay valid."""
    return [paper_figures(part)]
