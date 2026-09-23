#!/usr/bin/env python3
"""Rebuild Part 2 paper figures/tables from Kaggle_baseline_intrepretability_results.

Dump generated 2026-09-19 14:39:53. WBC dropped with TSSI (anti-leakage).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "modeling" / "tools"))
from figure_style import HARMONY, apply_style  # noqa: E402
from paper_paths import paper_figure_dirs, paper_figures  # noqa: E402

OUT_DIRS = paper_figure_dirs("02_ml_selectors")
DUMP = (
    ROOT
    / "code/modeling/interpretability/Kaggle_baseline_intrepretability_results"
    / "baseline_interpretability_results/model_feature_selectors_antileak"
)

FAMILY = {
    "lr": "Linear",
    "rf": "Bagged trees",
    "rf_b": "Bagged trees",
    "cat": "Boosting",
    "xgb": "Boosting",
    "xgb_b": "Boosting",
    "lgb": "Boosting",
}
FAMILY_COLOR = {
    "Linear": HARMONY[7],
    "Bagged trees": HARMONY[5],
    "Boosting": HARMONY[10],
}
MODEL_ORDER = ["lr", "rf", "rf_b", "cat", "xgb", "xgb_b", "lgb"]
ALGO_ORDER = ["LOCO", "SHAP", "FFS"]
MODEL_LABEL = {
    "lr": "Logistic regression",
    "rf": "Random forest",
    "rf_b": "Random forest (subsample)",
    "cat": "CatBoost",
    "xgb": "XGBoost",
    "xgb_b": "XGBoost (subsample)",
    "lgb": "LightGBM",
}

TABLE0 = [
    {"Code": "lr", "Classic model": "Logistic regression", "Family": "Linear", "GPU": "No",
     "Specification (notebook)": "L2-penalized log-odds (C=2, balanced); additive on scaled features"},
    {"Code": "rf", "Classic model": "Random forest", "Family": "Bagged trees", "GPU": "No",
     "Specification (notebook)": "500 deep trees, class_weight=balanced_subsample"},
    {"Code": "rf_b", "Classic model": "Random forest (subsample)", "Family": "Bagged trees", "GPU": "No",
     "Specification (notebook)": "Same as RF with max_samples=0.88"},
    {"Code": "cat", "Classic model": "CatBoost", "Family": "Boosting", "GPU": "Yes",
     "Specification (notebook)": "GPU Plain boosting (task_type=GPU; Ordered is CPU-only); eval_metric=PRAUC; balanced class weights"},
    {"Code": "xgb", "Classic model": "XGBoost", "Family": "Boosting", "GPU": "Yes",
     "Specification (notebook)": "scale_pos_weight for VLST imbalance"},
    {"Code": "xgb_b", "Classic model": "XGBoost (subsample)", "Family": "Boosting", "GPU": "Yes",
     "Specification (notebook)": "Lower subsample / colsample_bytree (0.78)"},
    {"Code": "lgb", "Classic model": "LightGBM", "Family": "Boosting", "GPU": "Yes",
     "Specification (notebook)": "Leaf-wise growth, balanced class weights"},
]


def load_dump() -> dict:
    long = pd.read_csv(DUMP / "selector_summary_long.csv")
    unique = (
        long.groupby(["model", "algorithm"])["feature"]
        .nunique()
        .unstack()
        .reindex(MODEL_ORDER)[ALGO_ORDER]
    )
    unique_counts = {m: unique.loc[m].to_dict() for m in MODEL_ORDER}

    top = long[long["rank"] <= 20]
    sets = {a: set(top.loc[top.algorithm == a, "feature"]) for a in ALGO_ORDER}
    jac = pd.DataFrame(index=ALGO_ORDER, columns=ALGO_ORDER, dtype=float)
    for a in ALGO_ORDER:
        for b in ALGO_ORDER:
            inter = len(sets[a] & sets[b])
            uni = len(sets[a] | sets[b])
            jac.loc[a, b] = (inter / uni) if uni else 0.0

    common_raw = pd.read_csv(DUMP / "selector_common_features_by_algorithm.csv")
    common = pd.DataFrame(
        [
            {
                "Algorithm": r.algorithm,
                "Metric": r.metric,
                "n common": int(r.n_common),
                "Features shared by all 7 models": "" if pd.isna(r.features) else r.features,
            }
            for r in common_raw.itertuples()
        ]
    )

    cons = pd.read_csv(DUMP / "selector_common_by_model_algorithms.csv")
    union = pd.read_csv(DUMP / "selector_union_by_model.csv")[["model", "n_union_features"]]
    prio = pd.read_csv(DUMP / "selector_priority_feature_ranks.csv").head(20)
    prio["rank"] = prio["rank"].apply(lambda x: "" if pd.isna(x) else int(x))
    glob_n = int(pd.read_csv(DUMP / "selector_all_unique_features.csv").shape[0])
    glob_names = "; ".join(
        pd.read_csv(DUMP / "selector_all_unique_features.csv").iloc[:, 0].astype(str).tolist()
        if False
        else pd.read_csv(DUMP / "selector_union_all_models_algorithms.csv")
        .iloc[0]["features"]
        .split("; ")
        if "features" in pd.read_csv(DUMP / "selector_union_all_models_algorithms.csv").columns
        else []
    )
    union_all = pd.read_csv(DUMP / "selector_union_all_models_algorithms.csv")
    if "features" in union_all.columns:
        glob_names = union_all.iloc[0]["features"]
    else:
        glob_names = "; ".join(pd.read_csv(DUMP / "selector_all_unique_features.csv").iloc[:, 0].astype(str))
    manifest = json.loads((DUMP / "split_manifest.json").read_text())
    return {
        "unique_counts": unique_counts,
        "jaccard": jac,
        "common": common,
        "cons": cons,
        "union": union,
        "prio": prio,
        "glob_n": glob_n,
        "glob_names": glob_names,
        "manifest": manifest,
    }


def _save(fig, name: str) -> None:
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, dpi=300, bbox_inches="tight", facecolor="white")


def _write_csv(df: pd.DataFrame, name: str) -> None:
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / name, index=False)


def _table_image(df: pd.DataFrame, name: str, family_col: str | None = None) -> None:
    n_rows, n_cols = df.shape
    fig_w = max(10, 0.22 * sum(len(str(c)) for c in df.columns) / 2)
    fig_h = max(2.2, 0.40 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(min(fig_w, 16), min(fig_h, 18)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HARMONY[7])
            cell.set_text_props(color="white", fontweight="bold")
        elif family_col and family_col in df.columns:
            fam = df.iloc[r - 1][family_col]
            cell.set_facecolor(FAMILY_COLOR.get(fam, "#F4F7FA") + "33")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FA")
    _save(fig, name)
    plt.close(fig)


def fig_unique_counts(unique_counts: dict) -> None:
    mat = pd.DataFrame(unique_counts).T.reindex(MODEL_ORDER)[ALGO_ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues", ax=ax, cbar_kws={"label": "Unique names"})
    ax.set_title("Unique selected feature counts (PR-AUC)")
    ax.set_ylabel("")
    _save(fig, "paper_fig1_unique_counts.png")
    plt.close(fig)


def fig_jaccard(jaccard: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(jaccard, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("Jaccard overlap of top-20 unions by algorithm")
    _save(fig, "paper_fig2_jaccard.png")
    plt.close(fig)


def fig_consensus_size(cons: pd.DataFrame) -> None:
    s = cons.set_index("model")["n_common"].reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = [FAMILY_COLOR[FAMILY[m]] for m in s.index]
    ax.bar(s.index, s.values, color=colors)
    ax.set_ylabel("n (LOCO ∩ SHAP ∩ FFS)")
    ax.set_title("Within-model three-selector consensus size (PR-AUC, top-20)")
    ax.set_ylim(0, max(s.values) + 1)
    _save(fig, "paper_fig3_consensus_size.png")
    plt.close(fig)


def fig_feature_by_model(cons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in cons.iterrows():
        feats = [f.strip() for f in str(r["features"]).split(";") if f.strip()]
        for feat in feats:
            rows.append({"model": r["model"], "feature": feat})
    hit = pd.DataFrame(rows)
    feats = sorted(hit["feature"].unique())
    mat = pd.DataFrame(0, index=feats, columns=MODEL_ORDER)
    for _, r in hit.iterrows():
        mat.loc[r["feature"], r["model"]] = 1
    fig_h = max(4.5, 0.32 * len(feats))
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Within-model three-way consensus (PR-AUC)")
    ax.set_xlabel("")
    _save(fig, "paper_fig4_feature_by_model.png")
    plt.close(fig)
    return mat


def fig_family_stacked(mat: pd.DataFrame) -> None:
    fam_map = {m: FAMILY[m] for m in MODEL_ORDER}
    stacked = pd.DataFrame(0, index=mat.index, columns=["Linear", "Bagged trees", "Boosting"])
    for m in MODEL_ORDER:
        stacked[fam_map[m]] += mat[m]
    fig, ax = plt.subplots(figsize=(8.2, max(4.0, 0.35 * len(stacked))))
    left = np.zeros(len(stacked))
    y = np.arange(len(stacked))
    for col in stacked.columns:
        ax.barh(y, stacked[col].values, left=left, color=FAMILY_COLOR[col], label=col)
        left += stacked[col].values
    ax.set_yticks(y)
    ax.set_yticklabels(stacked.index.tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of models (of 7)")
    ax.set_title("Consensus support by model family")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "paper_fig5_family_stacked.png")
    plt.close(fig)


def fig_cross_model(common: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    colors = [HARMONY[7], HARMONY[5], HARMONY[10]]
    ax.bar(common["Algorithm"], common["n common"], color=colors)
    for i, r in common.iterrows():
        label = r["Features shared by all 7 models"] or "(none)"
        ax.text(i, r["n common"] + 0.08, label, ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("n shared by all 7 models")
    ax.set_title("Cross-model intersection of top-20 (PR-AUC)")
    ax.set_ylim(0, max(common["n common"].max() + 1.4, 2))
    _save(fig, "paper_fig6_cross_model_common.png")
    plt.close(fig)


def fig_union(union: pd.DataFrame, glob_n: int) -> None:
    s = union.set_index("model")["n_union_features"].reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    colors = [FAMILY_COLOR[FAMILY[m]] for m in s.index]
    ax.bar(s.index, s.values, color=colors)
    ax.axhline(glob_n, color=HARMONY[0], ls="--", lw=1, label=f"Global unique = {glob_n}")
    ax.set_ylabel("Union of top-20 (LOCO + SHAP + FFS)")
    ax.set_title("Per-model union size")
    ax.legend(frameon=False)
    _save(fig, "paper_fig7_union_by_model.png")
    plt.close(fig)


def copy_notebook_pngs() -> None:
    mapping = {
        "selector_model_algorithm_counts.png": "selector_model_algorithm_counts.png",
        "selector_top_repeated_features.png": "selector_top_repeated_features.png",
        "selector_overlap_heatmap.png": "selector_overlap_heatmap.png",
    }
    for src_name, dest_name in mapping.items():
        p = DUMP / src_name
        if not p.is_file():
            print("missing", p)
            continue
        for out in OUT_DIRS:
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out / dest_name)


def write_manifest(manifest: dict) -> None:
    out = {
        **manifest,
        "protocol": "paper",
        "metric": "pr_auc",
        "feature_topk": 20,
        "loco_max_features": 60,
        "shap_universe": 40,
        "ffs_candidate_pool": 24,
        "ffs_max_steps": 12,
        "stent_encoding": "9-level shared encoder, then OHE drop-first",
        "generated": "2026-09-19 14:39:53",
        "dump_dir": str(DUMP.relative_to(ROOT)),
    }
    dest = paper_figures("02_ml_selectors") / "split_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
    apply_style()
    data = load_dump()
    write_manifest(data["manifest"])
    copy_notebook_pngs()

    t0 = pd.DataFrame(TABLE0)
    _write_csv(t0, "paper_table0_classic_models.csv")
    _table_image(t0, "paper_table0_classic_models.png", family_col="Family")

    common = data["common"]
    _write_csv(common, "paper_table1_common_by_algorithm.csv")
    _table_image(common, "paper_table1_common_by_algorithm.png")

    cons = data["cons"]
    cons_show = cons.copy()
    cons_show.insert(0, "Family", cons_show["model"].map(FAMILY))
    cons_show.insert(0, "Code", cons_show["model"])
    cons_show = cons_show.rename(
        columns={"n_common": "n (LOCO ∩ SHAP ∩ FFS)", "features": "Consensus features"}
    )
    _write_csv(cons, "paper_table2_consensus_by_model.csv")
    _table_image(
        cons_show[["Code", "Family", "metric", "n (LOCO ∩ SHAP ∩ FFS)", "Consensus features"]],
        "paper_table2_consensus_by_model.png",
        family_col="Family",
    )

    union = data["union"]
    union_show = union.copy()
    union_show.insert(0, "Family", union_show["model"].map(FAMILY))
    union_show.insert(0, "Classic model", union_show["model"].map(MODEL_LABEL))
    union_show = union_show.rename(columns={"model": "Code", "n_union_features": "Union size"})
    _write_csv(
        union_show[["Code", "Classic model", "Family", "Union size"]],
        "paper_table3_union_by_model.csv",
    )
    _table_image(
        union_show[["Code", "Classic model", "Family", "Union size"]],
        "paper_table3_union_by_model.png",
        family_col="Family",
    )

    glob = pd.DataFrame(
        [
            {
                "Scope": "All 7 models × LOCO, SHAP, FFS (PR-AUC top-20)",
                "n features": 0,
                "Features": "",
            },
            {
                "Scope": "Any model / selector (union of scored names)",
                "n features": data["glob_n"],
                "Features": f"{data['glob_n']} unique names (selector_all_unique_features.csv)",
            },
        ]
    )
    _write_csv(glob, "paper_table4_global_common.csv")
    _table_image(glob, "paper_table4_global_common.png")

    _write_csv(data["prio"], "paper_table5_priority_ranks_excerpt.csv")
    _table_image(data["prio"], "paper_table5_priority_ranks_excerpt.png")

    fig_unique_counts(data["unique_counts"])
    fig_jaccard(data["jaccard"])
    fig_consensus_size(cons)
    mat = fig_feature_by_model(cons)
    fig_family_stacked(mat)
    fig_cross_model(common)
    fig_union(union, data["glob_n"])
    print("Wrote Part 2 figures to:")
    for d in OUT_DIRS:
        print(" ", d)


if __name__ == "__main__":
    main()
