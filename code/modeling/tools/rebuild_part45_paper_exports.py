#!/usr/bin/env python3
"""Copy Revision-7 Part 4/5 Kaggle artefacts into the dual (and data/result) trees.

Part 4 figures come from executed cells in origin/main baseline_plus_tabpfn.ipynb.
Part 5 figures/CSVs come from the user-attached Kaggle download.
Table PNGs are rebuilt from those CSVs / the 7-arm dump numbers.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "modeling" / "tools"))
from figure_style import HARMONY, apply_style  # noqa: E402

ASSETS = Path("/home/fadia/.cursor/projects/home-fadia-Documents-vlst/assets")
ATTACH = Path(
    "/home/fadia/.cursor/projects/home-fadia-Documents-vlst/attachments"
    "/d7cb603b-f893-43a9-bf6d-e342449a26a0"
)
P4_NB_FIGS = Path("/tmp/vlst_newruns/p4_figs")

P4_DIRS = [
    ROOT / "paper_results" / "04_tabpfn_rating" / "paper_figures",
    ROOT / "code" / "modeling" / "rating" / "paper_figures",
    ROOT / "data" / "result" / "modeling_results" / "paper_figures",
]
P5_DIRS = [
    ROOT / "paper_results" / "05_tabpfn_interpretability" / "paper_figures",
    ROOT / "code" / "modeling" / "interpretability" / "paper_figures",
    ROOT / "data" / "result" / "modeling_tabpfn" / "paper_figures",
]


def _copy_to(src: Path, dest_name: str, dirs: list[Path]) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / dest_name)


def _write_df(df: pd.DataFrame, name: str, dirs: list[Path]) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / name, index=False)


def save_table_png(
    display: pd.DataFrame,
    *,
    title: str,
    filename: str,
    dirs: list[Path],
    figsize: tuple[float, float],
    fontsize: int = 8,
) -> None:
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
    for d in dirs:
        fig.savefig(d / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_part4_figures() -> None:
    mapping = {
        "cell9_out0.png": "paper_fig1_pr_roc_curves.png",
        "cell11_out0.png": "paper_fig2_calibration_curves.png",
        "cell13_out0.png": "paper_fig3_confusion_matrices.png",
        "cell17_out1.png": "best_model_threshold_fpfn_panel.png",
    }
    for src_name, dest in mapping.items():
        _copy_to(P4_NB_FIGS / src_name, dest, P4_DIRS)


def write_part4_tables() -> None:
    models = pd.DataFrame(
        [
            ["Logistic Regression", "Linear", "No", "L2, class_weight=balanced, max_iter=1000"],
            ["Random Forest", "Bagged trees", "No", "class_weight=balanced, random_state=42"],
            ["XGBoost", "Boosting", "Yes", "eval_metric=aucpr; scale_pos_weight from train fold"],
            ["LightGBM", "Boosting", "Yes", "metric=average_precision; class_weight=balanced"],
            ["CatBoost", "Boosting", "Yes", "auto_class_weights=Balanced; eval_metric=PRAUC"],
            [
                "TabPFN (thinking-high)",
                "Foundation (tabular)",
                "Kaggle T4 + client",
                "tabpfn_client; thinking_mode=True; thinking_effort=high; thinking_metric=average_precision",
            ],
            [
                "TabPFN (local)",
                "Foundation (tabular)",
                "Yes (Kaggle T4)",
                "from tabpfn import TabPFNClassifier; n_estimators=auto; balance_probabilities=True; no thinking",
            ],
        ],
        columns=["Model", "Family", "GPU", "Specification (notebook)"],
    )
    _write_df(models, "paper_table0_models.csv", P4_DIRS)
    save_table_png(
        models,
        title="Nested-CV models (7 arms; 9-level stent encoder)",
        filename="paper_table0_models.png",
        dirs=P4_DIRS,
        figsize=(13.5, 3.8),
        fontsize=7,
    )

    ranking = pd.DataFrame(
        [
            [1, "TabPFN (thinking-high)", 0.8553, 0.9905, 0.0064, "0.8488 ± 0.0861", "0.9906 ± 0.0070"],
            [2, "LightGBM", 0.6926, 0.9680, 0.0093, "0.6936 ± 0.0915", "0.9694 ± 0.0165"],
            [3, "XGBoost", 0.6815, 0.9439, 0.0088, "0.6928 ± 0.1288", "0.9431 ± 0.0418"],
            [4, "TabPFN (local)", 0.6754, 0.9845, 0.0673, "0.6739 ± 0.0812", "0.9846 ± 0.0030"],
            [5, "CatBoost", 0.6172, 0.9594, 0.0101, "0.6353 ± 0.0540", "0.9612 ± 0.0137"],
            [6, "Random Forest", 0.4865, 0.9209, 0.0143, "0.5034 ± 0.0793", "0.9206 ± 0.0423"],
            [7, "Logistic Regression", 0.3326, 0.9224, 0.0563, "0.3451 ± 0.1213", "0.9235 ± 0.0251"],
        ],
        columns=["Rank", "Model", "PR-AUC", "ROC-AUC", "Brier", "PR fold mean ± SD", "ROC fold mean ± SD"],
    )
    _write_df(ranking, "paper_table1_ranking.csv", P4_DIRS)
    disp = ranking.copy()
    for col in ["PR-AUC", "ROC-AUC", "Brier"]:
        disp[col] = disp[col].map(lambda x: f"{x:.4f}")
    save_table_png(
        disp,
        title="Pooled nested-CV OOF ranking (n=5185, 92 events; quote PR-AUC at prevalence 0.0177)",
        filename="paper_table1_ranking.png",
        dirs=P4_DIRS,
        figsize=(13.2, 3.8),
        fontsize=8,
    )

    nested = pd.DataFrame(
        [
            ["TabPFN (thinking-high)", "0.271 ± 0.067", 0.9915, 0.7927, 0.7065, 0.9967, 0.7471, 0.7222, 5076, 17, 27, 65],
            ["LightGBM", "0.117 ± 0.087", 0.9880, 0.6630, 0.6630, 0.9939, 0.6630, 0.6630, 5062, 31, 31, 61],
            ["XGBoost", "0.225 ± 0.060", 0.9875, 0.6452, 0.6522, 0.9935, 0.6486, 0.6508, 5060, 33, 32, 60],
            ["TabPFN (local)", "0.915 ± 0.012", 0.9844, 0.5478, 0.6848, 0.9898, 0.6087, 0.6522, 5041, 52, 29, 63],
            ["CatBoost", "0.167 ± 0.040", 0.9815, 0.4836, 0.6413, 0.9876, 0.5514, 0.6020, 5030, 63, 33, 59],
            ["Random Forest", "0.118 ± 0.013", 0.9840, 0.5517, 0.5217, 0.9923, 0.5363, 0.5275, 5054, 39, 44, 48],
            ["Logistic Regression", "0.947 ± 0.035", 0.9769, 0.3654, 0.4130, 0.9870, 0.3878, 0.4025, 5027, 66, 54, 38],
        ],
        columns=[
            "Model",
            "Threshold (mean ± SD)",
            "Accuracy",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "F2",
            "TN",
            "FP",
            "FN",
            "TP",
        ],
    )
    _write_df(nested, "paper_table2_nested_operating_point.csv", P4_DIRS)
    nd = nested.copy()
    for col in ["Accuracy", "Precision", "Recall", "Specificity", "F1", "F2"]:
        nd[col] = nd[col].map(lambda x: f"{x:.4f}")
    save_table_png(
        nd,
        title="Honest nested-CV F1 operating point (inner-fold threshold applied once to unseen outer fold)",
        filename="paper_table2_nested_operating_point.png",
        dirs=P4_DIRS,
        figsize=(14.5, 3.9),
        fontsize=7,
    )

    pooled = pd.DataFrame(
        [
            ["TabPFN (thinking-high)", 0.193, 0.9927, 0.7812, 0.8152, 0.9959, 0.7979, 0.8082, 5072, 21, 17, 75],
            ["LightGBM", 0.064, 0.9871, 0.6263, 0.6739, 0.9927, 0.6492, 0.6638, 5056, 37, 30, 62],
            ["XGBoost", 0.203, 0.9884, 0.6739, 0.6739, 0.9941, 0.6739, 0.6739, 5063, 30, 30, 62],
            ["TabPFN (local)", 0.886, 0.9826, 0.5067, 0.8261, 0.9855, 0.6281, 0.7336, 5019, 74, 16, 76],
            ["CatBoost", 0.416, 0.9873, 0.6806, 0.5326, 0.9955, 0.5976, 0.5568, 5070, 23, 43, 49],
            ["Random Forest", 0.104, 0.9826, 0.5098, 0.5652, 0.9902, 0.5361, 0.5532, 5043, 50, 40, 52],
            ["Logistic Regression", 0.985, 0.9819, 0.4857, 0.3696, 0.9929, 0.4198, 0.3881, 5057, 36, 58, 34],
        ],
        columns=["Model", "t_F1", "Accuracy", "Precision", "Recall", "Specificity", "F1", "F2", "TN", "FP", "FN", "TP"],
    )
    _write_df(pooled, "paper_table3_pooled_f1.csv", P4_DIRS)
    pdsp = pooled.copy()
    pdsp["t_F1"] = pdsp["t_F1"].map(lambda x: f"{x:.3f}")
    for col in ["Accuracy", "Precision", "Recall", "Specificity", "F1", "F2"]:
        pdsp[col] = pdsp[col].map(lambda x: f"{x:.4f}")
    save_table_png(
        pdsp,
        title="Optimistic pooled F1 cut (do not quote instead of nested Table 2)",
        filename="paper_table3_pooled_f1.png",
        dirs=P4_DIRS,
        figsize=(14.5, 3.9),
        fontsize=7,
    )

    counts = nested[["Model", "Threshold (mean ± SD)", "TN", "FP", "FN", "TP"]].copy()
    counts.insert(1, "Strategy", "nested inner F1")
    counts = counts.rename(columns={"Threshold (mean ± SD)": "Threshold"})
    _write_df(counts, "paper_table3_confusion_counts.csv", P4_DIRS)
    save_table_png(
        counts,
        title="Nested inner-F1 confusion counts (n=5185; 92 events)",
        filename="paper_table3_confusion_counts.png",
        dirs=P4_DIRS,
        figsize=(11.5, 3.8),
        fontsize=8,
    )

    wang = pd.DataFrame(
        [
            [
                "Wang 2020 integer score (frozen)",
                0.8013,
                0.1032,
                "0.8005 ± 0.0607",
                "0.1134 ± 0.0518",
                "Published points on all 5,185 rows; folds are evaluation only",
            ],
            [
                "TabPFN (thinking-high)",
                0.9905,
                0.8553,
                "0.9906 ± 0.0070",
                "0.8488 ± 0.0861",
                "Part 4 nested 5×4 CV OOF (this notebook)",
            ],
            [
                "LightGBM (untuned nested CV)",
                0.9680,
                0.6926,
                "0.9694 ± 0.0165",
                "0.6936 ± 0.0915",
                "Part 4 nested 5×4 CV OOF (this notebook)",
            ],
            [
                "TabPFN (local)",
                0.9845,
                0.6754,
                "0.9846 ± 0.0030",
                "0.6739 ± 0.0812",
                "Part 4 nested 5×4 CV OOF (this notebook)",
            ],
            [
                "Logistic regression (untuned nested CV)",
                0.9224,
                0.3326,
                "0.9235 ± 0.0251",
                "0.3451 ± 0.1213",
                "Part 4 nested 5×4 CV OOF (this notebook)",
            ],
        ],
        columns=["Model", "ROC-AUC", "PR-AUC", "ROC fold mean ± SD", "PR fold mean ± SD", "Protocol"],
    )
    _write_df(wang, "paper_table_s_wang_vs_ml.csv", P4_DIRS)
    wd = wang.copy()
    wd["ROC-AUC"] = wd["ROC-AUC"].map(lambda x: f"{x:.4f}")
    wd["PR-AUC"] = wd["PR-AUC"].map(lambda x: f"{x:.4f}")
    save_table_png(
        wd,
        title="Frozen Wang 2020 integer score vs nested-CV models (PR-AUC at 1.77% prevalence)",
        filename="paper_table_s_wang_vs_ml.png",
        dirs=P4_DIRS,
        figsize=(14.8, 3.4),
        fontsize=7,
    )


def copy_part5_assets() -> None:
    png_map = {
        "interpretability_pdp-4c171df0-2c0b-4d1c-b5bc-636de0f531f3.png": "paper_fig1_pdp_continuous.png",
        "interpretability_pdp_binary-f4ac13eb-2919-4e3b-826d-d455997f1db8.png": "paper_fig2_pdp_binary.png",
        "sv_interpretability_shap_summary-b8f68742-834b-4bd4-a193-5feae98b6abf.png": "paper_fig3_shap_summary.png",
        "sv_interpretability_shap_scatter_f0-d8af06b2-4087-41ae-8f8e-9771f38aaac1.png": "paper_fig4_shap_scatter_age.png",
        "sv_interpretability_shap_bar-b6f04f33-343e-4f97-87d6-9df99118eb57.png": "paper_fig5_shap_bar.png",
        "sv_interpretability_shap_beeswarm-ed955160-ab44-484d-a69c-b4f4e72f5fdf.png": "paper_fig6_shap_beeswarm.png",
        "sv_interpretability_shap_waterfall_row0-0a8dec9c-ddc9-4c19-a587-d4e0b3839f7f.png": "paper_fig7_shap_waterfall.png",
        "k_ssi_interpretability_network_top15-ea44d777-e650-427e-9b37-7e53fd06a9e0.png": "paper_fig8_ksii_network.png",
        "k_ssi_interpretability_upset-6406a018-c46f-4776-8763-1ade89eacc7c.png": "paper_fig9_ksii_upset.png",
        "k_ssi_shapiq_network_top15-b1d9c9ba-d1f0-404d-9816-4f558fa6475e.png": "paper_fig11_shapiq_network.png",
        "k_ssi_shapiq_upset-69b4cbea-fb3a-469f-948e-c593c93c9dc6.png": "paper_fig12_shapiq_upset.png",
        "interpretability_feature_importance_report-c069ba43-a767-461a-a9dc-451e37d33d25.png": "paper_fig13_consensus_ranking.png",
    }
    # Force plot is the small cell-14 native shapiq figure.
    force = ASSETS / "__results___14_7-40ffcd13-e1b7-4e49-b4d0-966492a459cc.png"
    png_map[force.name] = "paper_fig10_shapiq_force.png"
    for src_name, dest in png_map.items():
        src = ASSETS / src_name
        _copy_to(src, dest, P5_DIRS)

    csv_keep = [
        "interpretability_mutual_info_ranking.csv",
        "interpretability_feature_stability.csv",
        "interpretability_feature_stability_summary.csv",
        "interpretability_pdp_binary.csv",
        "interpretability_shap_mean_abs.csv",
        "interpretability_shap_explain_indices.csv",
        "interpretability_feature_importance_report.csv",
    ]
    for name in csv_keep:
        _copy_to(ATTACH / name, name, P5_DIRS)


def write_part5_tables() -> None:
    methods = pd.DataFrame(
        [
            [
                "mutual_info_classif",
                "Univariate association",
                "sklearn",
                "0 TabPFN calls; median fill is inert (no NaNs); all 81 scores stored",
            ],
            [
                "Stability (repeated SFS)",
                "Selection frequency",
                "local TabPFN",
                "10 resamples × top-10 forward SFS, AP scoring, full cohort",
            ],
            [
                "PDP",
                "Average predicted probability (empirical prior; not Part 4 risk)",
                "local TabPFN",
                "Full cohort n=5185; balance_probabilities=False; ranking/SHAP stay True",
            ],
            [
                "SHAP (shapiq SV)",
                "Local attributions",
                "tabpfn-client + thinking (succeeded)",
                "15 VLST=1 + 15 VLST=0; fit/background=full cohort; budget=256",
            ],
            [
                "k-SII / SHAP-IQ",
                "Pairwise interactions",
                "tabpfn-client + thinking (succeeded)",
                "One VLST=1 row (5099) from the 15+15 slice; budget=256",
            ],
            [
                "Consensus (Borda)",
                "Mean of normalized ranks",
                "aggregate",
                "MI + stability frequency + mean(|SHAP|); MI not fill-zeroed",
            ],
        ],
        columns=["Method", "Question", "Backend", "Notebook setting"],
    )
    _write_df(methods, "paper_table0_methods.csv", P5_DIRS)
    save_table_png(
        methods,
        title="Part 5 interpretability methods (Kaggle Interpretability plus Version 2)",
        filename="paper_table0_methods.png",
        dirs=P5_DIRS,
        figsize=(14.2, 3.6),
        fontsize=7,
    )

    mi = pd.read_csv(ATTACH / "interpretability_mutual_info_ranking.csv")
    _write_df(mi, "paper_table1_mutual_info.csv", P5_DIRS)
    mi15 = mi.head(15).copy()
    mi15["mutual_info"] = mi15["mutual_info"].map(lambda x: f"{x:.6f}")
    mi15 = mi15.rename(columns={"rank": "Rank", "feature": "Feature", "mutual_info": "Mutual information"})
    save_table_png(
        mi15,
        title="Top 15 mutual information (full 81-row CSV; Fast-Glu and ZES included)",
        filename="paper_table1_mutual_info.png",
        dirs=P5_DIRS,
        figsize=(8.5, 5.6),
        fontsize=8,
    )

    stab = pd.read_csv(ATTACH / "interpretability_feature_stability.csv")
    _write_df(stab, "paper_table2_stability.csv", P5_DIRS)
    stab_disp = stab.copy()
    stab_disp["Selected"] = stab_disp["times_selected"].map(lambda n: f"{int(n)}/10")
    stab_disp = stab_disp.rename(columns={"feature": "Feature", "selection_freq": "Frequency"})
    stab_disp = stab_disp[["Feature", "Selected", "Frequency"]]
    stab_disp["Frequency"] = stab_disp["Frequency"].map(lambda x: f"{x:.1f}")
    save_table_png(
        stab_disp,
        title="Stability selection frequency (10 seeds × top-10 forward SFS; full cohort)",
        filename="paper_table2_stability.png",
        dirs=P5_DIRS,
        figsize=(8.2, 12.4),
        fontsize=8,
    )

    pdp = pd.read_csv(ATTACH / "interpretability_pdp_binary.csv")
    table3 = pd.DataFrame(
        {
            "Feature": pdp["feature"],
            "P(y=1 | 0)": pdp["p_absent"].map(lambda x: round(x, 4)),
            "P(y=1 | 1)": pdp["p_present"].map(lambda x: round(x, 4)),
            "ΔP": pdp["delta"].map(lambda x: round(x, 4)),
            "scale": pdp["scale"],
            "n": pdp["n"],
            "prevalence": pdp["prevalence"].map(lambda x: round(x, 4)),
        }
    )
    _write_df(table3, "paper_table3_pdp_binary.csv", P5_DIRS)
    t3d = table3.copy()
    for col in ["P(y=1 | 0)", "P(y=1 | 1)", "ΔP", "prevalence"]:
        t3d[col] = t3d[col].map(lambda x: f"{x:.4f}")
    save_table_png(
        t3d,
        title="Binary PDP on empirical prior (full cohort; not Part 4 nested-CV risk)",
        filename="paper_table3_pdp_binary.png",
        dirs=P5_DIRS,
        figsize=(12.8, 3.4),
        fontsize=8,
    )

    shap = pd.read_csv(ATTACH / "interpretability_shap_mean_abs.csv")
    shap15 = shap.head(15).copy()
    _write_df(shap15, "paper_table4_shap_mean_abs.csv", P5_DIRS)
    s4 = shap15.rename(columns={"rank": "Rank", "feature": "Feature", "shap_mean_abs": "mean(|SHAP|)"})
    s4["mean(|SHAP|)"] = s4["mean(|SHAP|)"].map(lambda x: f"{x:.4f}")
    save_table_png(
        s4,
        title="Mean(|SHAP|) on 15 VLST=1 + 15 VLST=0 (client thinking; not global SHAP)",
        filename="paper_table4_shap_mean_abs.png",
        dirs=P5_DIRS,
        figsize=(8.2, 5.6),
        fontsize=8,
    )

    cons = pd.read_csv(ATTACH / "interpretability_feature_importance_report.csv")
    cons15 = cons.head(15).copy()
    _write_df(cons15, "paper_table5_consensus.csv", P5_DIRS)
    c5 = pd.DataFrame(
        {
            "Rank": cons15["rank"],
            "Feature": cons15["feature"],
            "Score": cons15["importance_score"].map(lambda x: f"{x:.4f}"),
            "n methods": cons15["n_methods"].map(lambda n: f"{int(n)}/3"),
            "Stability": cons15["selection_freq"].map(lambda x: f"{x:.1f}"),
            "mean(|SHAP|)": cons15["shap_mean_abs"].map(lambda x: f"{x:.4f}"),
            "MI": cons15["mutual_info"].map(lambda x: f"{x:.6f}"),
            "In MI top": cons15["in_mi_top"].map(lambda x: "yes" if bool(x) else "no"),
            "In SHAP top": cons15["in_shap_top"].map(lambda x: "yes" if bool(x) else "no"),
        }
    )
    save_table_png(
        c5,
        title="Borda consensus (MI + stability + mean|SHAP|; MI from full 81-row ranking)",
        filename="paper_table5_consensus.png",
        dirs=P5_DIRS,
        figsize=(13.6, 5.8),
        fontsize=7,
    )


def rebuild_concat() -> None:
    pr = ROOT / "paper_results"
    bundle = (pr / "paper_results.md").read_text()
    # Keep TOC header through the first --- after the numbered list.
    header, rest = bundle.split("\n---\n", 1)
    header = header.rstrip() + "\n\n---\n"

    def _load(rel: str) -> str:
        return (pr / rel).read_text().rstrip() + "\n"

    def prefix_assets(md: str, prefix: str) -> str:
        md = md.replace("](paper_figures/", f"]({prefix}/paper_figures/")
        md = md.replace("(paper_figures/", f"({prefix}/paper_figures/")
        md = md.replace("[paper_figures](", f"[{prefix}/paper_figures](")
        md = md.replace("[paper_figures/](", f"[{prefix}/paper_figures/](")
        return md

    def demote_h1(md: str, part_title: str) -> str:
        lines = md.splitlines(True)
        if lines and lines[0].startswith("# "):
            lines[0] = "### " + lines[0][2:]
        return f"# {part_title}\n\n" + "".join(lines)

    front = _load("00_front_matter.md")
    front_body = "".join(front.splitlines(True)[1:])  # drop original H1
    part0 = "# Part 0. Scope, motivation, terminology, and limitations\n" + front_body

    # Preserve Parts 1–3 from the existing concat (link rewriting already done).
    p1_start = rest.find("# Part 1. Statistical EDA")
    p4_start = rest.find("# Part 4. Nested-CV baselines plus TabPFN")
    parts_1_3 = rest[p1_start:p4_start].rstrip()
    if parts_1_3.endswith("---"):
        parts_1_3 = parts_1_3[: -3].rstrip()
    parts_1_3 += "\n"

    part4 = demote_h1(
        prefix_assets(_load("04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md"), "04_tabpfn_rating"),
        "Part 4. Nested-CV baselines plus TabPFN",
    )
    part5 = demote_h1(
        prefix_assets(
            _load("05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md"),
            "05_tabpfn_interpretability",
        ),
        "Part 5. TabPFN interpretability",
    )
    # Asset-root labels in standalone files are [paper_figures]; after prefix they
    # still need the folder name in the visible text for Part 4/5.
    part4 = part4.replace(
        "**Asset root:** [04_tabpfn_rating/paper_figures](04_tabpfn_rating/paper_figures/)",
        "**Asset root:** [paper_figures](04_tabpfn_rating/paper_figures/)",
    )
    part5 = part5.replace(
        "**Asset root:** [05_tabpfn_interpretability/paper_figures/](05_tabpfn_interpretability/paper_figures/)",
        "**Asset root:** [paper_figures/](05_tabpfn_interpretability/paper_figures/)",
    )

    out = header + "\n" + part0.rstrip() + "\n\n---\n" + parts_1_3 + "\n---\n" + part4.rstrip() + "\n\n---\n" + part5
    (pr / "paper_results.md").write_text(out)
    print("Wrote paper_results/paper_results.md")


def main() -> None:
    if not P4_NB_FIGS.exists():
        # Extract from git if the caller did not dump cells yet.
        import base64
        import subprocess

        nb_path = Path("/tmp/vlst_newruns/baseline_plus_tabpfn.ipynb")
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        if not nb_path.exists():
            blob = subprocess.check_output(
                ["git", "-C", str(ROOT), "show", "origin/main:code/modeling/rating/baseline_plus_tabpfn.ipynb"]
            )
            nb_path.write_bytes(blob)
        nb = json.loads(nb_path.read_text())
        P4_NB_FIGS.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(nb["cells"]):
            for j, o in enumerate(c.get("outputs") or []):
                png = (o.get("data") or {}).get("image/png")
                if png:
                    (P4_NB_FIGS / f"cell{i}_out{j}.png").write_bytes(base64.b64decode(png))

    copy_part4_figures()
    write_part4_tables()
    copy_part5_assets()
    write_part5_tables()
    print("Copied figures and rebuilt table PNGs/CSVs.")


if __name__ == "__main__":
    main()
    if "--concat" in sys.argv:
        rebuild_concat()
