#!/usr/bin/env python3
"""Supplementary table: with-TSSI vs without-TSSI single-split baselines.

Numbers are taken from the stored notebook outputs (no refit):
  baseline_tssi_leakage.ipynb  and  baseline_without_tssi.ipynb
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "modeling" / "tools"))
from figure_style import HARMONY, apply_style  # noqa: E402
from paper_paths import paper_figure_dirs  # noqa: E402

OUT_DIRS = paper_figure_dirs("04_tabpfn_rating")

# Stored test-set metrics. Duplicate RF rows in the leaky notebook are collapsed.
ROWS = [
    ("Logistic Regression", 0.9846, 0.6923, 1.0000, 0.5294, 0.9990, 0.9575, 0.9354, 0.2637, 0.6667, 0.1644, 0.9171, 0.5077),
    ("Decision Tree", 0.9942, 0.8500, 0.9444, 0.7727, 0.9696, 0.7308, 0.9749, 0.1875, 0.1667, 0.2143, 0.5774, 0.0405),
    ("Random Forest", 0.9942, 0.8000, 0.6667, 1.0000, 0.9993, 0.9680, 0.9749, 0.4348, 0.5556, 0.3571, 0.9338, 0.4700),
    ("Gaussian NB", 0.1851, 0.0409, 1.0000, 0.0209, 0.5854, 0.0209, 0.1851, 0.0409, 1.0000, 0.0209, 0.5854, 0.0209),
    ("CatBoost", 0.9981, 0.9412, 0.8889, 1.0000, 0.9995, 0.9773, 0.9875, 0.5806, 0.5000, 0.6923, 0.9669, 0.6582),
    ("XGBoost", 0.9981, 0.9412, 0.8889, 1.0000, 0.9987, 0.9609, 0.9884, 0.5714, 0.4444, 0.8000, 0.9380, 0.6118),
    ("LightGBM", 0.9981, 0.9412, 0.8889, 1.0000, 0.9989, 0.9708, 0.9865, 0.5625, 0.5000, 0.6429, 0.9483, 0.6018),
]
COLS = [
    "model",
    "with_acc",
    "with_f1",
    "with_recall",
    "with_prec",
    "with_roc_auc",
    "with_pr_auc",
    "without_acc",
    "without_f1",
    "without_recall",
    "without_prec",
    "without_roc_auc",
    "without_pr_auc",
]

# GridSearchCV(scoring="f1", cv=StratifiedKFold(5, shuffle=True, random_state=42)).
# Prints from executed notebooks. Not Part 4 nested-CV hyperparameters.
BEST_PARAMS = [
    (
        "Logistic Regression",
        "C=1.0, max_iter=2000, penalty=l1, solver=liblinear",
        "C=10.0, max_iter=2000, penalty=l1, solver=liblinear",
    ),
    (
        "Decision Tree",
        "criterion=gini, max_depth=None, min_samples_leaf=1, min_samples_split=10",
        "criterion=entropy, max_depth=None, min_samples_leaf=2, min_samples_split=2",
    ),
    (
        "Random Forest",
        "max_depth=15, min_samples_leaf=1, n_estimators=200",
        "max_depth=5, min_samples_leaf=1, n_estimators=200",
    ),
    ("Gaussian NB", "var_smoothing=1e-12", "var_smoothing=1e-12"),
    (
        "CatBoost",
        "depth=6, iterations=100, l2_leaf_reg=1, learning_rate=0.1",
        "depth=4, iterations=200, l2_leaf_reg=1, learning_rate=0.1",
    ),
    (
        "XGBoost",
        "learning_rate=0.05, max_depth=5, min_child_weight=1, n_estimators=200",
        "learning_rate=0.1, max_depth=3, min_child_weight=1, n_estimators=200",
    ),
    (
        "LightGBM",
        "learning_rate=0.1, max_depth=5, min_child_samples=20, n_estimators=200",
        "learning_rate=0.1, max_depth=3, min_child_samples=20, n_estimators=200",
    ),
]


def main() -> None:
    apply_style()
    df = pd.DataFrame(ROWS, columns=COLS)
    df["delta_pr_auc"] = df["without_pr_auc"] - df["with_pr_auc"]
    df["delta_roc_auc"] = df["without_roc_auc"] - df["with_roc_auc"]

    display = pd.DataFrame(
        {
            "Model": df["model"],
            "With TSSI PR-AUC": df["with_pr_auc"].map(lambda x: f"{x:.4f}"),
            "Without TSSI PR-AUC": df["without_pr_auc"].map(lambda x: f"{x:.4f}"),
            "Δ PR-AUC": df["delta_pr_auc"].map(lambda x: f"{x:+.4f}"),
            "With TSSI ROC-AUC": df["with_roc_auc"].map(lambda x: f"{x:.4f}"),
            "Without TSSI ROC-AUC": df["without_roc_auc"].map(lambda x: f"{x:.4f}"),
            "With TSSI F1": df["with_f1"].map(lambda x: f"{x:.4f}"),
            "Without TSSI F1": df["without_f1"].map(lambda x: f"{x:.4f}"),
        }
    )

    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "paper_table_s_tssi_leakage.csv", index=False)

    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.axis("off")
    tbl = ax.table(
        cellText=display.values,
        colLabels=list(display.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.45)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HARMONY[7])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FA")
    ax.set_title(
        "Single-split 70/30 baselines with vs without Time since stent implantation",
        pad=12,
    )
    for out in OUT_DIRS:
        fig.savefig(out / "paper_table_s_tssi_leakage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = range(len(df))
    ax.barh([i + 0.18 for i in y], df["with_pr_auc"], height=0.34, color=HARMONY[0], label="With TSSI (leaky)")
    ax.barh([i - 0.18 for i in y], df["without_pr_auc"], height=0.34, color=HARMONY[7], label="Without TSSI")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("PR-AUC (single stratified 70/30 hold-out)")
    ax.axvline(0.0177, color="0.4", ls=":", lw=1, label="prevalence = 0.0177")
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Dropping the follow-up-time column collapses ranking")
    for out in OUT_DIRS:
        fig.savefig(out / "paper_fig_s_tssi_pr_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    hp = pd.DataFrame(
        BEST_PARAMS,
        columns=["model", "best_params_with_tssi", "best_params_without_tssi"],
    )
    for out in OUT_DIRS:
        hp.to_csv(out / "paper_table_s_tssi_best_params.csv", index=False)

    hp_disp = pd.DataFrame(
        {
            "Model": hp["model"],
            "With TSSI (leaky) best_params_": hp["best_params_with_tssi"],
            "Without TSSI best_params_": hp["best_params_without_tssi"],
        }
    )
    fig, ax = plt.subplots(figsize=(16.2, 3.8))
    ax.axis("off")
    tbl = ax.table(
        cellText=hp_disp.values,
        colLabels=list(hp_disp.columns),
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.55)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HARMONY[7])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FA")
    ax.set_title(
        "GridSearchCV F1 winners (70/30 notebooks; not Part 4 nested CV)",
        pad=12,
    )
    for out in OUT_DIRS:
        fig.savefig(out / "paper_table_s_tssi_best_params.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote TSSI leakage table, figure, and GridSearch best_params_.")


if __name__ == "__main__":
    main()
