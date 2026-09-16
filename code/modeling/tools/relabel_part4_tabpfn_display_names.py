#!/usr/bin/env python3
"""Relabel Part 4 curve figures so the client arm is never just 'TabPFN'.

The nested-CV notebook exported legends as 'TabPFN' for thinking-high and
'TabPFN (local)' for the local checkpoint. Tables already name both arms.
This rebuilds the four unlabeled PNGs from committed OOF scores / pooled
counts. It does not re-fit models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "modeling" / "tools"))
from figure_style import HARMONY, apply_style  # noqa: E402

OOF_PATH = ROOT / "data" / "result" / "modeling_results" / "oof" / "oof_predictions.csv"
POOLED_PATH = (
    ROOT / "paper_results" / "04_tabpfn_rating" / "paper_figures" / "paper_table3_pooled_f1.csv"
)

P4_DIRS = [
    ROOT / "paper_results" / "04_tabpfn_rating" / "paper_figures",
    ROOT / "code" / "modeling" / "rating" / "paper_figures",
    ROOT / "data" / "result" / "modeling_results" / "paper_figures",
]

# Notebook column -> publication display name. The client column was exported as
# `tabpfn_*` with legend text "TabPFN"; that is the thinking-high arm.
PROB_COLS_V4 = [
    ("logistic_regression_prob", "Logistic Regression"),
    ("random_forest_prob", "Random Forest"),
    ("xgboost_prob", "XGBoost"),
    ("lightgbm_prob", "LightGBM"),
    ("catboost_prob", "CatBoost"),
    ("tabpfn_thinking_mode_prob", "TabPFN (thinking-high)"),
    ("tabpfn_prob", "TabPFN (local)"),
]
PROB_COLS_PREV = [
    ("logistic_regression_prob", "Logistic Regression"),
    ("random_forest_prob", "Random Forest"),
    ("xgboost_prob", "XGBoost"),
    ("lightgbm_prob", "LightGBM"),
    ("catboost_prob", "CatBoost"),
    ("tabpfn_prob", "TabPFN (thinking-high)"),
    ("tabpfn_(local)_prob", "TabPFN (local)"),
]


def resolve_prob_cols(df: pd.DataFrame) -> list[tuple[str, str]]:
    if "tabpfn_thinking_mode_prob" in df.columns:
        return PROB_COLS_V4
    return PROB_COLS_PREV

COLORS = {
    "TabPFN (thinking-high)": HARMONY[6],
    "TabPFN (local)": HARMONY[2],
    "LightGBM": HARMONY[9],
    "XGBoost": HARMONY[7],
    "CatBoost": HARMONY[8],
    "Random Forest": HARMONY[11],
    "Logistic Regression": HARMONY[3],
}


def _save(fig, filename: str) -> None:
    for d in P4_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_oof() -> pd.DataFrame:
    df = pd.read_csv(OOF_PATH)
    if "y" not in df.columns:
        raise SystemExit(f"OOF file missing y: {OOF_PATH}")
    return df


def plot_pr_roc(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    y = df["y"].to_numpy()
    prevalence = float(y.mean())
    scores = {}
    for col, name in resolve_prob_cols(df):
        p = df[col].to_numpy()
        scores[name] = {
            "ap": average_precision_score(y, p),
            "auc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p),
            "p": p,
        }

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    ax = axes[0]
    for name, s in sorted(scores.items(), key=lambda kv: kv[1]["ap"]):
        prec, rec, _ = precision_recall_curve(y, s["p"])
        ax.plot(
            rec,
            prec,
            color=COLORS[name],
            lw=1.8,
            label=f"{name} (AP={s['ap']:.3f})",
            zorder=2 + int(name.startswith("TabPFN (thinking")),
        )
    ax.axhline(prevalence, color="0.45", ls=":", lw=1.2, label=f"prevalence = {prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    handles, labels = ax.get_legend_handles_labels()
    # Prevalence last; models by AP descending.
    order = np.argsort([-scores[lab.split(" (AP=")[0]]["ap"] if " (AP=" in lab else 1e9 for lab in labels])
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", fontsize=7.5)

    ax = axes[1]
    for name, s in sorted(scores.items(), key=lambda kv: kv[1]["auc"]):
        fpr, tpr, _ = roc_curve(y, s["p"])
        ax.plot(
            fpr,
            tpr,
            color=COLORS[name],
            lw=1.8,
            label=f"{name} (AUC={s['auc']:.3f})",
            zorder=2 + int(name.startswith("TabPFN (thinking")),
        )
    ax.plot([0, 1], [0, 1], color="0.45", ls="--", lw=1.2, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort(
        [
            -scores[lab.split(" (AUC=")[0]]["auc"] if " (AUC=" in lab else 1e9
            for lab in labels
        ]
    )
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right", fontsize=7.5)

    fig.suptitle("Nested-CV out-of-fold ranking curves", y=1.02)
    _save(fig, "paper_fig1_pr_roc_curves.png")
    return scores


def plot_calibration(df: pd.DataFrame, scores: dict[str, dict[str, float]]) -> None:
    y = df["y"].to_numpy()
    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.0))
    axes = axes.ravel()
    names = [name for _, name in resolve_prob_cols(df)]
    for i, name in enumerate(names):
        ax = axes[i]
        p = scores[name]["p"]
        frac, meanp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot([0, 1], [0, 1], color="0.55", ls="--", lw=1.0, label="perfect")
        ax.plot(meanp, frac, color=HARMONY[7], marker="o", ms=4, lw=1.6, label="OOF")
        ax.set_title(f"{name} (Brier = {scores[name]['brier']:.4f})", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Calibration (nested-CV OOF, quantile bins)", y=0.995)
    fig.tight_layout()
    _save(fig, "paper_fig2_calibration_curves.png")


def plot_confusion() -> None:
    pooled = pd.read_csv(POOLED_PATH)
    apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.0))
    axes = axes.ravel()
    for i, row in pooled.iterrows():
        ax = axes[i]
        mat = np.array([[int(row["TN"]), int(row["FP"])], [int(row["FN"]), int(row["TP"])]], dtype=float)
        ax.imshow(np.log1p(mat), cmap="Blues", vmin=0, vmax=np.log1p(5200))
        for (r, c), val in np.ndenumerate(mat):
            ax.text(c, r, f"{int(val)}", ha="center", va="center", color="white" if val > 400 else "black", fontsize=11)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(f"{row['Model']} ($t_{{F1}}$ = {row['t_F1']:.3f})", fontsize=9)
        ax.grid(False)
    for j in range(len(pooled), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Confusion matrices at F1-optimal pooled OOF threshold (optimistic)", y=0.995)
    fig.tight_layout()
    _save(fig, "paper_fig3_confusion_matrices.png")


def plot_threshold_sweep(df: pd.DataFrame) -> tuple[float, float]:
    y = df["y"].to_numpy()
    th_col = next(col for col, name in resolve_prob_cols(df) if name == "TabPFN (thinking-high)")
    p = df[th_col].to_numpy()
    ap = average_precision_score(y, p)
    grid = np.linspace(0.0, 1.0, 1001)
    prec = np.zeros_like(grid)
    rec = np.zeros_like(grid)
    f1 = np.zeros_like(grid)
    f2 = np.zeros_like(grid)
    fps = np.zeros_like(grid)
    fns = np.zeros_like(grid)
    for i, t in enumerate(grid):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        prec[i] = tp / (tp + fp) if (tp + fp) else 0.0
        rec[i] = tp / (tp + fn) if (tp + fn) else 0.0
        f1[i] = f1_score(y, pred, zero_division=0)
        f2[i] = fbeta_score(y, pred, beta=2, zero_division=0)
        fps[i] = fp
        fns[i] = fn
    t_star = float(grid[int(np.argmax(f1))])

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    ax.plot(grid, prec, color=HARMONY[7], label="Precision")
    ax.plot(grid, rec, color=HARMONY[8], label="Recall")
    ax.plot(grid, f1, color=HARMONY[9], label="F1")
    ax.plot(grid, f2, color=HARMONY[0], label="F2 (recall-weighted)")
    ax.axvline(t_star, color="0.15", ls="--", lw=1.1, label=f"chosen t = {t_star:.3f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (out-of-fold)")
    ax.set_title("Threshold sweep (OOF) — strategy=f1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="center right")

    ax = axes[1]
    ax.plot(fps, fns, color=HARMONY[11], lw=1.5)
    i_star = int(np.argmax(f1))
    ax.scatter(
        [fps[i_star]],
        [fns[i_star]],
        s=55,
        color="0.1",
        zorder=3,
        label=f"t = {t_star:.3f}: FP={int(fps[i_star])}, FN={int(fns[i_star])}",
    )
    ax.invert_xaxis()
    ax.set_xlabel("False positives")
    ax.set_ylabel("False negatives")
    ax.set_title("FP vs FN along the sweep — strategy=f1")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Best model by PR-AUC: TabPFN (thinking-high) (AP={ap:.3f})",
        y=1.02,
    )
    _save(fig, "best_model_threshold_fpfn_panel.png")
    return t_star, ap


def main() -> None:
    df = load_oof()
    scores = plot_pr_roc(df)
    plot_calibration(df, scores)
    plot_confusion()
    t_star, ap = plot_threshold_sweep(df)
    print("Relabeled Part 4 figures in:")
    for d in P4_DIRS:
        print(f"  {d}")
    print("Pooled metrics from OOF (sanity):")
    for name, s in sorted(scores.items(), key=lambda kv: -kv[1]["ap"]):
        print(f"  {name:28s}  PR-AUC={s['ap']:.4f}  ROC-AUC={s['auc']:.4f}  Brier={s['brier']:.4f}")
    print(f"Thinking-high pooled F1 threshold on 1001-pt grid: {t_star:.3f} (AP={ap:.3f})")


if __name__ == "__main__":
    main()
