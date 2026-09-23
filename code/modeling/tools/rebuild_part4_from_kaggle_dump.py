#!/usr/bin/env python3
"""Rebuild Part 4 paper figures/tables from Kaggle_baseline_plus_tabpfn_results.

Dump layout (2026-09-22 ingest):
  .../Kaggle_baseline_plus_tabpfn_results/baseline_plus_tabpfn_results/modeling_results/
Nine nested-CV arms after anti-leakage (TSSI+WBC drop, lab quantize, stent train-fold).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "modeling" / "tools"))
from figure_style import HARMONY, apply_style  # noqa: E402
from paper_paths import paper_figure_dirs  # noqa: E402

DUMP = (
    ROOT
    / "code/modeling/rating/Kaggle_baseline_plus_tabpfn_results"
    / "baseline_plus_tabpfn_results"
    / "modeling_results"
)
P4 = paper_figure_dirs("04_tabpfn_rating")
N_BOOT = 2000
SEED = 42

OOF_COLS = [
    ("Logistic Regression", "logistic_regression_prob"),
    ("Random Forest", "random_forest_prob"),
    ("XGBoost", "xgboost_prob"),
    ("LightGBM", "lightgbm_prob"),
    ("CatBoost", "catboost_prob"),
    ("TabPFN thinking v3", "tabpfn_thinking_v3_prob"),
    ("TabPFN thinking v3.5", "tabpfn_thinking_v3_5_prob"),
    ("TabPFN v3", "tabpfn_v3_prob"),
    ("TabPFN v3.5", "tabpfn_v3_5_prob"),
]


def _write_df(df: pd.DataFrame, name: str) -> None:
    for d in P4:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / name, index=False)


def save_table_png(display: pd.DataFrame, *, title: str, filename: str, figsize, fontsize=8) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    tbl = ax.table(
        cellText=display.values,
        colLabels=list(display.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.45)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HARMONY[7])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FA")
    ax.set_title(title, pad=12)
    for d in P4:
        fig.savefig(d / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_figures() -> None:
    mapping = {
        "figures/pr_roc_curves.png": "paper_fig1_pr_roc_curves.png",
        "figures/calibration_curves.png": "paper_fig2_calibration_curves.png",
        "figures/confusion_matrices.png": "paper_fig3_confusion_matrices.png",
        "figures/best_model_threshold_fpfn_panel.png": "best_model_threshold_fpfn_panel.png",
    }
    for src, dest in mapping.items():
        p = DUMP / src
        if not p.is_file():
            raise FileNotFoundError(p)
        for d in P4:
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, d / dest)


def write_tables() -> None:
    models = pd.DataFrame(
        [
            ["Logistic Regression", "Linear", "No", "L2, class_weight=balanced, max_iter=1000"],
            ["Random Forest", "Bagged trees", "No", "class_weight=balanced, random_state=42"],
            ["XGBoost", "Boosting", "Yes", "eval_metric=aucpr; scale_pos_weight from train fold"],
            ["LightGBM", "Boosting", "Yes", "metric=average_precision; class_weight=balanced"],
            ["CatBoost", "Boosting", "Yes", "auto_class_weights=Balanced; eval_metric=PRAUC"],
            [
                "TabPFN thinking v3",
                "Foundation (tabular)",
                "Kaggle T4 + client",
                "tabpfn-client==0.6.0; model v3_default; thinking_mode=True; effort=high; metric=average_precision",
            ],
            [
                "TabPFN thinking v3.5",
                "Foundation (tabular)",
                "Kaggle T4 + client",
                "tabpfn-client==0.6.0; model v3.5_default; thinking_mode=True; effort=high; metric=average_precision",
            ],
            [
                "TabPFN v3",
                "Foundation (tabular)",
                "Kaggle T4",
                "tabpfn==9.0.0; checkpoint tabpfn-v3-classifier-v3_default.ckpt; n_estimators=auto; no balance_probabilities",
            ],
            [
                "TabPFN v3.5",
                "Foundation (tabular)",
                "Kaggle T4",
                "tabpfn==9.0.0; checkpoint tabpfn-v3.5-20260909.safetensors; n_estimators=auto; no balance_probabilities",
            ],
        ],
        columns=["Model", "Family", "GPU", "Specification"],
    )
    _write_df(models, "paper_table0_models.csv")
    save_table_png(
        models,
        title="Nested-CV models (9 arms; anti-leakage ON; 9-level stent encoder)",
        filename="paper_table0_models.png",
        figsize=(15.5, 4.6),
        fontsize=7,
    )

    mc = pd.read_csv(DUMP / "tables/model_comparison.csv")
    rank = mc[["model", "pr_auc", "pr_auc_mean", "pr_auc_sd", "roc_auc", "roc_auc_mean", "roc_auc_sd", "brier"]].copy()
    rank.insert(0, "rank", range(1, len(rank) + 1))
    _write_df(rank, "paper_table1_ranking.csv")
    disp = pd.DataFrame(
        {
            "Rank": rank["rank"],
            "Model": rank["model"],
            "PR-AUC": rank["pr_auc"].map(lambda x: f"{x:.4f}"),
            "PR fold mean ± SD": [
                f"{m:.4f} ± {s:.4f}" for m, s in zip(rank["pr_auc_mean"], rank["pr_auc_sd"])
            ],
            "ROC-AUC": rank["roc_auc"].map(lambda x: f"{x:.4f}"),
            "ROC fold mean ± SD": [
                f"{m:.4f} ± {s:.4f}" for m, s in zip(rank["roc_auc_mean"], rank["roc_auc_sd"])
            ],
            "Brier": rank["brier"].map(lambda x: f"{x:.4f}"),
        }
    )
    save_table_png(
        disp,
        title="Pooled nested-CV OOF ranking after anti-leakage (n=5185, 92 events)",
        filename="paper_table1_ranking.png",
        figsize=(14.8, 4.6),
        fontsize=7,
    )

    nest = pd.read_csv(DUMP / "tables/nested_cv_operating_point.csv")
    _write_df(nest, "paper_table2_nested_operating_point.csv")
    nd = pd.DataFrame(
        {
            "Model": nest["model"],
            "Threshold (mean ± SD)": [
                f"{m:.3f} ± {s:.3f}" for m, s in zip(nest["threshold_mean"], nest["threshold_sd"])
            ],
            "Accuracy": nest["accuracy"].map(lambda x: f"{x:.4f}"),
            "Precision": nest["precision"].map(lambda x: f"{x:.4f}"),
            "Recall": nest["recall"].map(lambda x: f"{x:.4f}"),
            "Specificity": nest["specificity"].map(lambda x: f"{x:.4f}"),
            "NPV": nest["npv"].map(lambda x: f"{x:.4f}"),
            "F1": nest["f1"].map(lambda x: f"{x:.4f}"),
            "F2": nest["f2"].map(lambda x: f"{x:.4f}"),
            "TN": nest["tn"],
            "FP": nest["fp"],
            "FN": nest["fn"],
            "TP": nest["tp"],
        }
    )
    save_table_png(
        nd,
        title="Honest nested-CV F1 operating point (inner-fold threshold on unseen outer fold)",
        filename="paper_table2_nested_operating_point.png",
        figsize=(16.8, 4.8),
        fontsize=7,
    )

    pooled = mc[
        ["model", "t_f1", "acc_f1", "prec_f1", "rec_f1", "spec_f1", "npv_f1", "f1_at_t_f1", "f2_at_t_f1", "tn_f1", "fp_f1", "fn_f1", "tp_f1"]
    ].copy()
    _write_df(pooled, "paper_table3_pooled_f1.csv")
    pdsp = pd.DataFrame(
        {
            "Model": pooled["model"],
            "t_F1": pooled["t_f1"].map(lambda x: f"{x:.3f}"),
            "Accuracy": pooled["acc_f1"].map(lambda x: f"{x:.4f}"),
            "Precision": pooled["prec_f1"].map(lambda x: f"{x:.4f}"),
            "Recall": pooled["rec_f1"].map(lambda x: f"{x:.4f}"),
            "Specificity": pooled["spec_f1"].map(lambda x: f"{x:.4f}"),
            "F1": pooled["f1_at_t_f1"].map(lambda x: f"{x:.4f}"),
            "F2": pooled["f2_at_t_f1"].map(lambda x: f"{x:.4f}"),
            "TN": pooled["tn_f1"],
            "FP": pooled["fp_f1"],
            "FN": pooled["fn_f1"],
            "TP": pooled["tp_f1"],
        }
    )
    save_table_png(
        pdsp,
        title="Pooled OOF F1 cut (optimistic; do not quote instead of Table 2)",
        filename="paper_table3_pooled_f1.png",
        figsize=(15.8, 4.8),
        fontsize=7,
    )

    ece = pd.read_csv(DUMP / "tables/calibration_ece.csv")
    _write_df(ece, "paper_table_s_ece.csv")
    ed = pd.DataFrame(
        {
            "Model": ece["model"],
            "Brier": ece["brier"].map(lambda x: f"{x:.4f}"),
            "ECE (8 quantile bins)": ece["ece_quantile_8"].map(lambda x: f"{x:.4f}"),
        }
    )
    save_table_png(
        ed,
        title="Calibration ECE on pooled nested-CV OOF (quantile, 8 bins)",
        filename="paper_table_s_ece.png",
        figsize=(10.5, 4.4),
        fontsize=8,
    )

    pe = pd.read_csv(DUMP / "tables/precision_equalised_sensitivity.csv")
    _write_df(pe, "paper_table_s_precision_equalised.csv")
    ped = pe.copy()
    for c in ["ap_as_is", "ap_equalised", "auc_as_is", "auc_equalised", "ap_drop"]:
        ped[c] = ped[c].map(lambda x: f"{x:.4f}")
    save_table_png(
        ped,
        title="Precision-equalised sensitivity (recording-precision probe; not nested ranking)",
        filename="paper_table_s_precision_equalised.png",
        figsize=(13.5, 4.0),
        fontsize=7,
    )

    probe = pd.read_csv(DUMP / "tables/leakage_precision_probe.csv")
    _write_df(probe, "paper_table_s_leakage_precision_probe.csv")

    wang = pd.DataFrame(
        [
            ["Wang 2020 integer score (frozen)", 0.8013, 0.1032, "0.8005 ± 0.0607", "0.1134 ± 0.0518", "Published points; folds evaluate only"],
            ["TabPFN thinking v3.5", 0.996280, 0.921222, "0.9965 ± 0.0027", "0.9227 ± 0.0423", "Part 4 nested 5×4 CV OOF; anti-leakage ON"],
            ["TabPFN v3.5", 0.991630, 0.895729, "0.9922 ± 0.0066", "0.8995 ± 0.0528", "Part 4 nested 5×4 CV OOF; anti-leakage ON"],
            ["TabPFN thinking v3", 0.983379, 0.831894, "0.9843 ± 0.0105", "0.8335 ± 0.0658", "Part 4 nested 5×4 CV OOF; anti-leakage ON"],
            ["TabPFN v3", 0.973062, 0.715000, "0.9741 ± 0.0125", "0.7286 ± 0.0319", "Part 4 nested 5×4 CV OOF; anti-leakage ON"],
            ["XGBoost", 0.937361, 0.632154, "0.9366 ± 0.0371", "0.6448 ± 0.0880", "Part 4 nested 5×4 CV OOF; anti-leakage ON"],
            ["LightGBM", 0.944389, 0.627103, "0.9458 ± 0.0332", "0.6248 ± 0.0990", "Part 4 nested 5×4 CV OOF; anti-leakage ON"],
        ],
        columns=["Model", "ROC-AUC", "PR-AUC", "ROC fold mean ± SD", "PR fold mean ± SD", "Protocol"],
    )
    _write_df(wang, "paper_table_s_wang_vs_ml.csv")
    wd = wang.copy()
    wd["ROC-AUC"] = wd["ROC-AUC"].map(lambda x: f"{x:.4f}")
    wd["PR-AUC"] = wd["PR-AUC"].map(lambda x: f"{x:.4f}")
    save_table_png(
        wd.drop(columns=["Protocol"]),
        title="Frozen Wang integer score vs nested-CV models (anti-leakage ON)",
        filename="paper_table_s_wang_vs_ml.png",
        figsize=(14.5, 4.0),
        fontsize=7,
    )


def bootstrap() -> None:
    oof = pd.read_csv(DUMP / "oof/oof_predictions.csv")
    y = oof["y"].to_numpy(dtype=int)
    assert y.size == 5185 and int(y.sum()) == 92
    rng = np.random.default_rng(SEED)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    def metrics(yy, p):
        return (
            float(average_precision_score(yy, p)),
            float(roc_auc_score(yy, p)),
            float(brier_score_loss(yy, p)),
        )

    point = {lab: dict(zip(["pr_auc", "roc_auc", "brier"], metrics(y, oof[col].to_numpy(float)))) for lab, col in OOF_COLS}
    boot = {lab: {"pr_auc": [], "roc_auc": [], "brier": []} for lab, _ in OOF_COLS}
    p_th35 = oof["tabpfn_thinking_v3_5_prob"].to_numpy(float)
    p_loc35 = oof["tabpfn_v3_5_prob"].to_numpy(float)
    p_lgb = oof["lightgbm_prob"].to_numpy(float)
    p_xgb = oof["xgboost_prob"].to_numpy(float)
    d_th_lgb, d_loc_lgb, d_th_xgb = [], [], []

    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, size=pos.size, replace=True), rng.choice(neg, size=neg.size, replace=True)])
        yb = y[idx]
        for lab, col in OOF_COLS:
            pr, roc, br = metrics(yb, oof[col].to_numpy(float)[idx])
            boot[lab]["pr_auc"].append(pr)
            boot[lab]["roc_auc"].append(roc)
            boot[lab]["brier"].append(br)
        d_th_lgb.append(average_precision_score(yb, p_th35[idx]) - average_precision_score(yb, p_lgb[idx]))
        d_loc_lgb.append(average_precision_score(yb, p_loc35[idx]) - average_precision_score(yb, p_lgb[idx]))
        d_th_xgb.append(average_precision_score(yb, p_th35[idx]) - average_precision_score(yb, p_xgb[idx]))

    def ci(arr):
        a = np.asarray(arr, dtype=float)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    rows = []
    for lab, _ in OOF_COLS:
        pr_lo, pr_hi = ci(boot[lab]["pr_auc"])
        roc_lo, roc_hi = ci(boot[lab]["roc_auc"])
        br_lo, br_hi = ci(boot[lab]["brier"])
        rows.append(
            {
                "model": lab,
                "pr_auc": point[lab]["pr_auc"],
                "pr_auc_ci_low": pr_lo,
                "pr_auc_ci_high": pr_hi,
                "roc_auc": point[lab]["roc_auc"],
                "roc_auc_ci_low": roc_lo,
                "roc_auc_ci_high": roc_hi,
                "brier": point[lab]["brier"],
                "brier_ci_low": br_lo,
                "brier_ci_high": br_hi,
                "n_boot": N_BOOT,
                "seed": SEED,
            }
        )
    ci_df = pd.DataFrame(rows)
    _write_df(ci_df, "paper_table_s_bootstrap_ci.csv")

    def paired(name, deltas, point_delta):
        d = np.asarray(deltas, dtype=float)
        lo, hi = ci(d)
        p_le0 = float(np.mean(d <= 0))
        return {
            "contrast": name,
            "delta_pr_auc": point_delta,
            "ci_low": lo,
            "ci_high": hi,
            "p_bootstrap_one_sided_le0": p_le0,
            "n_boot": N_BOOT,
            "seed": SEED,
        }

    paired_df = pd.DataFrame(
        [
            paired(
                "TabPFN thinking v3.5 − LightGBM",
                d_th_lgb,
                point["TabPFN thinking v3.5"]["pr_auc"] - point["LightGBM"]["pr_auc"],
            ),
            paired(
                "TabPFN v3.5 − LightGBM",
                d_loc_lgb,
                point["TabPFN v3.5"]["pr_auc"] - point["LightGBM"]["pr_auc"],
            ),
            paired(
                "TabPFN thinking v3.5 − XGBoost",
                d_th_xgb,
                point["TabPFN thinking v3.5"]["pr_auc"] - point["XGBoost"]["pr_auc"],
            ),
        ]
    )
    _write_df(paired_df, "paper_table_s_paired_delta.csv")

    folds = pd.read_csv(DUMP / "tables/fold_metrics.csv")
    wide = folds.pivot(index="fold", columns="model", values="pr_auc").reset_index()
    _write_df(wide, "paper_table_s_fold_pr_wins.csv")

    disp = pd.DataFrame(
        {
            "Model": ci_df["model"],
            "PR-AUC (95% CI)": [
                f"{r.pr_auc:.4f} [{r.pr_auc_ci_low:.4f}, {r.pr_auc_ci_high:.4f}]" for r in ci_df.itertuples()
            ],
            "ROC-AUC (95% CI)": [
                f"{r.roc_auc:.4f} [{r.roc_auc_ci_low:.4f}, {r.roc_auc_ci_high:.4f}]" for r in ci_df.itertuples()
            ],
            "Brier (95% CI)": [
                f"{r.brier:.4f} [{r.brier_ci_low:.4f}, {r.brier_ci_high:.4f}]" for r in ci_df.itertuples()
            ],
        }
    )
    save_table_png(
        disp,
        title="Nested-CV OOF ranking with stratified bootstrap 95% CIs (n_boot=2000)",
        filename="paper_table_s_bootstrap_ci.png",
        figsize=(15.2, 5.2),
        fontsize=7,
    )
    pdisp = pd.DataFrame(
        {
            "Contrast": paired_df["contrast"],
            "Δ PR-AUC": [f"{v:.4f}" for v in paired_df["delta_pr_auc"]],
            "95% CI": [f"[{r.ci_low:.4f}, {r.ci_high:.4f}]" for r in paired_df.itertuples()],
            "P(Δ ≤ 0)": [f"{v:.4f}" for v in paired_df["p_bootstrap_one_sided_le0"]],
        }
    )
    save_table_png(
        pdisp,
        title="Paired bootstrap Δ PR-AUC on pooled OOF (n_boot=2000)",
        filename="paper_table_s_paired_delta.png",
        figsize=(13.8, 2.6),
        fontsize=8,
    )
    print("bootstrap CIs written")


def main() -> None:
    if not DUMP.is_dir():
        raise FileNotFoundError(DUMP)
    copy_figures()
    write_tables()
    bootstrap()
    print("Part 4 rebuilt from", DUMP)


if __name__ == "__main__":
    main()
