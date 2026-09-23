#!/usr/bin/env python3
"""Supplementary table: with-TSSI vs without-TSSI single-split baselines.

Numbers are taken from the 2026-09-19 Kaggle dumps (no refit):
  Kaggle_baseline_tssi_leakage_results/.../test_metrics.csv
  Kaggle_baseline_without_tssi_results/.../test_metrics.csv
GridSearch best_params_ are the executed notebook prints (2026-09-19 papermill),
written to modeling_*/best_params.csv and paper_table_s_tssi_best_params.*.
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

# 2026-09-19 Kaggle test_metrics.csv (papermill 14:00–14:57 WITH, 14:02–14:22 WITHOUT).
ROWS = [
    ("Logistic Regression", 0.983933, 0.675325, 0.928571, 0.530612, 0.995372, 0.913388, 0.912596, 0.209302, 0.642857, 0.125000, 0.831830, 0.343057),
    ("Decision Tree", 0.987147, 0.705882, 0.857143, 0.600000, 0.926620, 0.752385, 0.960154, 0.279070, 0.428571, 0.206897, 0.700940, 0.137767),
    ("Random Forest", 0.994859, 0.833333, 0.714286, 1.000000, 0.997686, 0.939998, 0.982005, 0.000000, 0.000000, 0.000000, 0.928712, 0.487373),
    ("Gaussian NB", 0.437018, 0.043668, 0.714286, 0.022523, 0.742532, 0.272768, 0.197301, 0.037008, 0.857143, 0.018913, 0.749310, 0.056433),
    ("CatBoost", 0.997429, 0.923077, 0.857143, 1.000000, 0.996447, 0.959865, 0.968509, 0.409639, 0.607143, 0.309091, 0.910270, 0.494218),
    ("XGBoost", 0.997429, 0.923077, 0.857143, 1.000000, 0.996284, 0.954687, 0.988432, 0.550000, 0.392857, 0.916667, 0.918334, 0.568484),
    ("LightGBM", 0.997429, 0.923077, 0.857143, 1.000000, 0.998621, 0.968654, 0.987147, 0.444444, 0.285714, 1.000000, 0.940445, 0.667542),
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
# LIVE prints from executed 2026-09-19 notebooks (not the older .nbdump).
# ALL LEAKS ON = baseline_tssi_leakage.ipynb; ALL LEAKS OFF = baseline_without_tssi.ipynb.
# Not Part 4 nested-CV hyperparameters.
BEST_PARAMS = [
    (
        "Logistic Regression",
        "C=0.1, max_iter=2000, penalty=l2, solver=lbfgs",
        "C=100.0, max_iter=2000, penalty=l1, solver=liblinear",
    ),
    (
        "Decision Tree",
        "criterion=entropy, max_depth=10, max_features=None, min_samples_leaf=2, min_samples_split=5",
        "criterion=entropy, max_depth=15, max_features=None, min_samples_leaf=5, min_samples_split=2",
    ),
    (
        "Random Forest",
        "max_depth=20, max_features=sqrt, min_samples_leaf=1, n_estimators=800",
        "max_depth=20, max_features=sqrt, min_samples_leaf=5, n_estimators=400",
    ),
    ("Gaussian NB", "var_smoothing=1e-06", "var_smoothing=1e-06"),
    (
        "CatBoost",
        "depth=4, iterations=200, l2_leaf_reg=1, learning_rate=0.03",
        "depth=4, iterations=200, l2_leaf_reg=1, learning_rate=0.03",
    ),
    (
        "XGBoost",
        "learning_rate=0.1, max_depth=5, min_child_weight=1, n_estimators=200, subsample=0.8",
        "learning_rate=0.1, max_depth=3, min_child_weight=3, n_estimators=400, subsample=1.0",
    ),
    (
        "LightGBM",
        "learning_rate=0.03, max_depth=5, min_child_samples=10, n_estimators=400, num_leaves=15",
        "learning_rate=0.1, max_depth=5, min_child_samples=40, n_estimators=400, num_leaves=15",
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
        "70/30 GridSearch: ALL LEAKS ON vs ALL LEAKS OFF",
        pad=12,
    )
    for out in OUT_DIRS:
        fig.savefig(out / "paper_table_s_tssi_leakage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = range(len(df))
    ax.barh([i + 0.18 for i in y], df["with_pr_auc"], height=0.34, color=HARMONY[0], label="ALL LEAKS ON")
    ax.barh([i - 0.18 for i in y], df["without_pr_auc"], height=0.34, color=HARMONY[7], label="ALL LEAKS OFF")
    ax.set_yticks(list(y))
    ax.set_yticklabels(df["model"])
    ax.set_xlabel("PR-AUC (single stratified 70/30 hold-out)")
    ax.axvline(0.0177, color="0.4", ls=":", lw=1, label="prevalence = 0.0177")
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("ALL LEAKS ON vs ALL LEAKS OFF — PR-AUC on the 70/30 hold-out")
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
            "ALL LEAKS ON best_params_": hp["best_params_with_tssi"],
            "ALL LEAKS OFF best_params_": hp["best_params_without_tssi"],
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
        "GridSearchCV F1 winners from executed notebooks (not Part 4 nested CV)",
        pad=12,
    )
    for out in OUT_DIRS:
        fig.savefig(out / "paper_table_s_tssi_best_params.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    dump_on = ROOT / "code/modeling/rating/Kaggle_baseline_tssi_leakage_results/baseline_leakge_results/modeling_tssi_leakage"
    dump_off = ROOT / "code/modeling/rating/Kaggle_baseline_without_tssi_results/baseline_without_leakage/modeling_without_tssi"
    on_rows = [{"model": m, "best_params": p} for m, p, _ in BEST_PARAMS]
    off_rows = [{"model": m, "best_params": p} for m, _, p in BEST_PARAMS]
    if dump_on.is_dir():
        pd.DataFrame(on_rows).to_csv(dump_on / "best_params.csv", index=False)
    if dump_off.is_dir():
        pd.DataFrame(off_rows).to_csv(dump_off / "best_params.csv", index=False)
    print("Wrote TSSI leakage table, figure, GridSearch best_params_, and dump best_params.csv.")


if __name__ == "__main__":
    main()
