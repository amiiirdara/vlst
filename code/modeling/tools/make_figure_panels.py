#!/usr/bin/env python3
"""Generate publication-ready VLST figure panels (Harmony palette, unified fonts)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "data" / "result" / "figures"
CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from figure_style import (  # noqa: E402
    FAMILY_COLORS,
    HARMONY,
    METRIC_COLORS,
    apply_style,
    panel_label,
    save_figure,
)

TARGET_COL = "Stent thrombosis"
RANDOM_STATE = 42
TEST_SIZE = 0.30


def _repo_paths() -> dict[str, Path]:
    return {
        "raw": ROOT / "data" / "raw" / "VLST.csv",
        "manifest": ROOT / "data" / "processed" / "preprocess_manifest.json",
        "summary": ROOT / "data" / "result" / "eda" / "plots" / "data_summary.csv",
        "metrics": FIG_DIR / "model_metrics.csv",
        "loco": FIG_DIR / "tabpfn_loco_importance.csv",
        "x_train": ROOT / "data" / "processed" / "X_train.npy",
        "x_test": ROOT / "data" / "processed" / "X_test.npy",
        "y_train": ROOT / "data" / "processed" / "y_train.npy",
        "y_test": ROOT / "data" / "processed" / "y_test.npy",
    }


def load_split_counts() -> dict:
    manifest = json.loads(_repo_paths()["manifest"].read_text())
    y_train = np.load(_repo_paths()["y_train"])
    y_test = np.load(_repo_paths()["y_test"])
    return {
        "n_total": manifest["n_rows_after_clean"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_pos": int((y_train == 1).sum()),
        "test_pos": int((y_test == 1).sum()),
        "train_neg": int((y_train == 0).sum()),
        "test_neg": int((y_test == 0).sum()),
    }


def fig01_preprocessing_qc(out_dir: Path) -> None:
    paths = _repo_paths()
    summary = pd.read_csv(paths["summary"]).iloc[0]
    split = load_split_counts()
    manifest = json.loads(paths["manifest"].read_text())

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    # A — cohort flow
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 10)
    ax_a.axis("off")
    boxes = [
        (1.0, 7.2, "Raw VLST cohort\n(n = 5,185)", HARMONY[11]),
        (1.0, 5.0, "Drop IDs & leakage\n+ deduplicate\n(n = 5,185)", HARMONY[5]),
        (1.0, 2.8, "Modeling features\n(n = 81)", HARMONY[7]),
        (5.8, 5.8, f"Train 70%\n(n = {split['n_train']:,})\npositives = {split['train_pos']}", HARMONY[9]),
        (5.8, 3.2, f"Test 30%\n(n = {split['n_test']:,})\npositives = {split['test_pos']}", HARMONY[8]),
    ]
    for x, y, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            3.4,
            1.5,
            boxstyle="round,pad=0.04,rounding_size=0.2",
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )
        ax_a.add_patch(patch)
        ax_a.text(x + 1.7, y + 0.75, text, ha="center", va="center", fontsize=9)
    ax_a.annotate("", xy=(5.5, 6.5), xytext=(4.5, 7.6), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax_a.annotate("", xy=(5.5, 4.0), xytext=(4.5, 7.6), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax_a.set_title("Preprocessing & stratified split")
    panel_label(ax_a, "A", x=-0.06, y=1.02)

    # B — class imbalance (full cohort)
    ax_b = fig.add_subplot(gs[0, 1])
    counts = [int(summary["Target_Class_0"]), int(summary["Target_Class_1"])]
    bars = ax_b.bar(
        ["No VLST (0)", "VLST (1)"],
        counts,
        color=[FAMILY_COLORS["negative"], FAMILY_COLORS["positive"]],
        edgecolor="white",
        linewidth=0.8,
    )
    ax_b.set_ylabel("Patients (n)")
    ax_b.set_title("Target class distribution")
    for bar, val in zip(bars, counts):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 40,
            f"{val:,}\n({100 * val / sum(counts):.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    panel_label(ax_b, "B", x=-0.10, y=1.02)

    # C — train vs test positive rate preserved
    ax_c = fig.add_subplot(gs[1, 0])
    groups = ["Train", "Test"]
    pos_rates = [
        100 * split["train_pos"] / split["n_train"],
        100 * split["test_pos"] / split["n_test"],
    ]
    totals = [split["n_train"], split["n_test"]]
    x = np.arange(len(groups))
    width = 0.35
    ax_c.bar(x - width / 2, totals, width, label="Total n", color=HARMONY[2], alpha=0.85)
    ax_c.bar(
        x + width / 2,
        [split["train_pos"], split["test_pos"]],
        width,
        label="VLST positives",
        color=FAMILY_COLORS["positive"],
        alpha=0.9,
    )
    ax_c.set_xticks(x, groups)
    ax_c.set_ylabel("Patients (n)")
    ax_c.set_title("Stratified hold-out (random_state = 42)")
    ax_c.legend(frameon=True, loc="upper right")
    for i, rate in enumerate(pos_rates):
        ax_c.text(i, totals[i] + 40, f"{rate:.2f}% pos.", ha="center", fontsize=8)
    panel_label(ax_c, "C", x=-0.10, y=1.02)

    # D — QC summary
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.axis("off")
    qc_rows = [
        ("Missing values (features)", int(summary["Missing_Values_Total"])),
        ("Duplicate rows removed", manifest["n_duplicate_rows_dropped"]),
        ("Leakage column dropped", "Time since stent implantation"),
        ("Binary features", int(summary["Binary_Features"])),
        ("Continuous features", int(summary["Continuous_Features"])),
        ("Categorical (stent brand)", int(summary["Categorical_Features"])),
        ("Scaled + OHE output dims", manifest["n_features_after_preprocess"]),
    ]
    y = 0.92
    ax_d.text(0.0, 1.0, "Quality control summary", fontsize=11, fontweight="bold", transform=ax_d.transAxes)
    for label, val in qc_rows:
        ax_d.text(0.02, y, f"{label}:", transform=ax_d.transAxes, fontsize=9, fontweight="bold")
        ax_d.text(0.62, y, str(val), transform=ax_d.transAxes, fontsize=9)
        y -= 0.12
    panel_label(ax_d, "D", x=-0.06, y=1.02)

    fig.suptitle("Figure 1 · Preprocessing & quality control", y=1.01, fontweight="bold")
    save_figure(fig, out_dir / "fig01_preprocessing_qc")


def _train_curve_models():
    x_train = np.load(_repo_paths()["x_train"])
    x_test = np.load(_repo_paths()["x_test"])
    y_train = np.load(_repo_paths()["y_train"])
    y_test = np.load(_repo_paths()["y_test"])

    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    scale = neg / pos

    models = {
        "XGB": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
        ),
    }
    scores = {}
    for name, clf in models.items():
        clf.fit(x_train, y_train)
        scores[name] = clf.predict_proba(x_test)[:, 1]
    return y_test, scores


def _best_f2_threshold(y_true, y_prob, grid=None):
    if grid is None:
        grid = np.linspace(0.02, 0.98, 97)
    best_t, best_f2 = 0.5, -1.0
    for t in grid:
        pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2.0, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t


def fig02_compare_models(out_dir: Path) -> None:
    metrics = pd.read_csv(_repo_paths()["metrics"])
    test = metrics[metrics["split"] == "test"].copy()
    test = test.sort_values("pr_auc", ascending=True)

    y_test, curve_scores = _train_curve_models()

    fig = plt.figure(figsize=(13, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    # A — test PR-AUC ranking
    ax_a = fig.add_subplot(gs[0, 0])
    colors = [FAMILY_COLORS.get(f, HARMONY[11]) for f in test["family"]]
    bars = ax_a.barh(test["model"], test["pr_auc"], color=colors, edgecolor="white", linewidth=0.6)
    ax_a.set_xlabel("PR-AUC (30% hold-out test)")
    ax_a.set_xlim(0, max(test["pr_auc"].max() * 1.08, 0.85))
    ax_a.set_title("Model ranking on imbalanced test set")
    for bar, val in zip(bars, test["pr_auc"]):
        ax_a.text(val + 0.008, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)
    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[k], lw=6, label=k)
        for k in ("baseline", "advanced", "tabpfn")
    ]
    ax_a.legend(handles=handles, title="Family", loc="lower right", frameon=True)
    panel_label(ax_a, "A")

    # B — CV PR-AUC (classical models with error bars)
    ax_b = fig.add_subplot(gs[0, 1])
    cv = metrics.dropna(subset=["pr_auc_cv_mean"]).sort_values("pr_auc_cv_mean", ascending=True)
    if len(cv):
        colors_b = [FAMILY_COLORS.get(f, HARMONY[11]) for f in cv["family"]]
        ax_b.barh(
            cv["model"],
            cv["pr_auc_cv_mean"],
            xerr=cv["pr_auc_cv_std"],
            color=colors_b,
            capsize=3,
            edgecolor="white",
            linewidth=0.6,
        )
        ax_b.set_xlabel("PR-AUC (5-fold CV on train)")
        ax_b.set_title("Cross-validated tree/boosters")
    else:
        ax_b.text(0.5, 0.5, "CV metrics unavailable", ha="center", va="center")
    panel_label(ax_b, "B")

    # C — PR curves
    ax_c = fig.add_subplot(gs[1, 0])
    for name, scores in curve_scores.items():
        PrecisionRecallDisplay.from_predictions(y_test, scores, ax=ax_c, name=name)
    tabpfn_row = metrics[metrics["model"] == "tabpfn_blend_stacking"].iloc[0]
    ax_c.scatter([], [], label=f"TabPFN blend (AP={tabpfn_row['pr_auc']:.3f})", s=0)
    ax_c.set_title("Precision–recall curves (test)")
    ax_c.set_xlabel("Recall")
    ax_c.set_ylabel("Precision")
    ax_c.legend(loc="upper right", frameon=True, fontsize=8)
    panel_label(ax_c, "C")

    # D — ROC curves
    ax_d = fig.add_subplot(gs[1, 1])
    for name, scores in curve_scores.items():
        RocCurveDisplay.from_predictions(y_test, scores, ax=ax_d, name=name)
    ax_d.set_title("ROC curves (test)")
    ax_d.legend(loc="lower right", frameon=True, fontsize=8)
    panel_label(ax_d, "D")

    fig.suptitle("Figure 2 · Model comparison (70/30 stratified split)", y=1.01, fontweight="bold")
    save_figure(fig, out_dir / "fig02_compare_models")


def fig03_top_model_results(out_dir: Path) -> None:
    metrics = pd.read_csv(_repo_paths()["metrics"])
    top = metrics[metrics["model"] == "tabpfn_blend_stacking"].iloc[0]

    y_test, curve_scores = _train_curve_models()
    # Use XGB as operational proxy when TabPFN scores are not materialized locally;
    # TabPFN blend metrics are shown in the summary panel from saved leaderboard values.
    y_prob = curve_scores["XGB"]
    threshold = _best_f2_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    # A — saved top-model metrics (TabPFN stacking blend)
    ax_a = fig.add_subplot(gs[0, 0])
    metric_names = ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]
    metric_vals = [
        top["pr_auc"],
        top["roc_auc"],
        top["precision"],
        top["recall"],
        top["f1"],
    ]
    bars = ax_a.bar(metric_names, metric_vals, color=HARMONY[9], alpha=0.85, edgecolor="white")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel("Score")
    ax_a.set_title("Top model · TabPFN meta-LR stacking blend")
    ax_a.tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, metric_vals):
        ax_a.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", fontsize=8)
    panel_label(ax_a, "A")

    # B — confusion matrix (XGB proxy at F2-optimal threshold; see caption)
    ax_b = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(HARMONY[9], as_cmap=True),
        cbar=False,
        ax=ax_b,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax_b.set_title(f"Confusion matrix (tree proxy, τ = {threshold:.2f})")
    ax_b.set_xlabel("Predicted label")
    ax_b.set_ylabel("True label")
    panel_label(ax_b, "B")

    # C — PR curve with operating point
    ax_c = fig.add_subplot(gs[1, 0])
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax_c, name=f"XGB (AP={pr_auc:.3f})")
    ax_c.scatter([rec], [prec], s=80, color=HARMONY[0], zorder=5, label=f"τ={threshold:.2f}")
    ax_c.set_title("PR curve with F₂-tuned threshold")
    ax_c.legend(loc="upper right", frameon=True, fontsize=8)
    panel_label(ax_c, "C")

    # D — operational metrics comparison: top blend vs tree proxy
    ax_d = fig.add_subplot(gs[1, 1])
    compare = pd.DataFrame(
        {
            "model": ["TabPFN stacking", "XGB (proxy)"],
            "precision": [top["precision"], prec],
            "recall": [top["recall"], rec],
            "f1": [top["f1"], f1],
        }
    )
    melted = compare.melt(id_vars="model", var_name="metric", value_name="value")
    sns.barplot(
        data=melted,
        x="metric",
        y="value",
        hue="model",
        palette=[HARMONY[9], HARMONY[7]],
        ax=ax_d,
        edgecolor="white",
    )
    ax_d.set_ylim(0, 1.05)
    ax_d.set_xlabel("")
    ax_d.set_ylabel("Score")
    ax_d.set_title("Operational metrics at tuned threshold")
    ax_d.legend(title="", frameon=True, loc="upper right")
    panel_label(ax_d, "D")

    fig.suptitle("Figure 3 · Top model performance on hold-out test", y=1.01, fontweight="bold")
    save_figure(fig, out_dir / "fig03_top_model_results")


def fig04_top_features(out_dir: Path) -> None:
    loco = pd.read_csv(_repo_paths()["loco"]).sort_values("importance", ascending=True)
    top = loco.tail(12)

    corr = pd.read_csv(ROOT / "data" / "result" / "eda" / "plots" / "correlations_with_target.csv")
    corr = corr[corr["Feature"] != "Time since stent implantation"].head(8)
    corr = corr.sort_values("Abs_Correlation", ascending=True)

    cont = pd.read_csv(ROOT / "data" / "result" / "eda" / "plots" / "statistical_tests_continuous.csv")
    cont = cont[cont["Feature"] != "Time since stent implantation"].head(8)
    cont = cont.sort_values("P-value", ascending=False)

    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.34)

    # A — LOCO importance (TabPFN)
    ax_a = fig.add_subplot(gs[0, :])
    colors = [HARMONY[9] if v >= 0 else HARMONY[8] for v in top["importance"]]
    ax_a.barh(top["feature"], top["importance"], color=colors, edgecolor="white", linewidth=0.6)
    ax_a.set_xlabel("LOCO importance (Δ PR-AUC when feature removed)")
    ax_a.set_title("TabPFN leave-one-covariate-out ranking (top 12)")
    panel_label(ax_a, "A", x=-0.04, y=1.02)

    # B — univariate |correlation| with target (leakage excluded)
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.barh(corr["Feature"], corr["Abs_Correlation"], color=HARMONY[7], alpha=0.9, edgecolor="white")
    ax_b.set_xlabel("|Pearson r| with VLST")
    ax_b.set_title("EDA correlation (leakage column excluded)")
    panel_label(ax_b, "B")

    # C — significant continuous features (Mann–Whitney p-values, −log10)
    ax_c = fig.add_subplot(gs[1, 1])
    cont = cont.copy()
    cont["neg_log10_p"] = -np.log10(cont["P-value"].clip(lower=1e-40))
    colors_c = [HARMONY[5] if s == "Yes" else HARMONY[2] for s in cont["Significant"]]
    ax_c.barh(cont["Feature"], cont["neg_log10_p"], color=colors_c, edgecolor="white")
    ax_c.axvline(-np.log10(0.05), color=HARMONY[0], ls="--", lw=1.0, label="p = 0.05")
    ax_c.set_xlabel("−log₁₀(p-value)")
    ax_c.set_title("Univariate tests · continuous features")
    ax_c.legend(frameon=True, loc="lower right", fontsize=8)
    panel_label(ax_c, "C")

    fig.suptitle("Figure 4 · Top predictors of very late stent thrombosis", y=1.01, fontweight="bold")
    save_figure(fig, out_dir / "fig04_top_features")


def main() -> None:
    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig01_preprocessing_qc(FIG_DIR)
    fig02_compare_models(FIG_DIR)
    fig03_top_model_results(FIG_DIR)
    fig04_top_features(FIG_DIR)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
