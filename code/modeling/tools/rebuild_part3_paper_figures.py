#!/usr/bin/env python3
"""Rebuild Part 3 overlap figures from Part 1 FDR-20 vs Part 2 dump consensus.

ML catalogue is the 2026-09-19 Kaggle dump three-way union (n=10; WBC dropped).
`stats_vs_ml_comparison.ipynb` imports this module and asserts Jaccard 5/25.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "modeling" / "tools"))
from figure_style import HARMONY, apply_style  # noqa: E402
from paper_paths import paper_figure_dirs  # noqa: E402

OUT_DIRS = paper_figure_dirs("03_stats_vs_ml")
DUMP = (
    ROOT
    / "code/modeling/interpretability/Kaggle_baseline_intrepretability_results"
    / "baseline_interpretability_results/model_feature_selectors_antileak"
)

STATS_FDR = [
    "WBC",
    "eGFR",
    "LV",
    "CKD5",
    "No.of stents per lesion",
    "HbA1c",
    "NO.of vessels",
    "Total stent length",
    "Fiberinogen",
    "1.1:1Post dilation",
    "No postdilation",
    "CKD90",
    "Previous PCI",
    "3-vessel disease",
    "Clopidogrel",
    "Diabetes",
    "PES",
    "Multi-vessel CAD",
    "Single-vessel disease",
    "Stent type-SES",
]
STATS_MULTIVAR = {
    "WBC",
    "eGFR",
    "LV",
    "CKD5",
    "1.1:1Post dilation",
    "CKD90",
    "Previous PCI",
    "Clopidogrel",
}
TIME_AT_RISK = "Time since stent implantation"

DOMAINS = {
    "WBC": "Laboratory",
    "eGFR": "Laboratory",
    "LV": "Cardiac",
    "CKD5": "Laboratory / renal",
    "No.of stents per lesion": "Procedural",
    "HbA1c": "Laboratory",
    "NO.of vessels": "Anatomy",
    "Total stent length": "Procedural",
    "Fiberinogen": "Laboratory",
    "1.1:1Post dilation": "Procedural",
    "No postdilation": "Procedural",
    "CKD90": "Laboratory / renal",
    "Previous PCI": "History",
    "3-vessel disease": "Anatomy",
    "Clopidogrel": "Medication",
    "Diabetes": "Comorbidity",
    "PES": "Stent type",
    "Multi-vessel CAD": "Anatomy",
    "Single-vessel disease": "Anatomy",
    "Stent type-SES": "Stent type",
    "Cre": "Laboratory",
    "Men": "Demographics",
    "LVEF": "Cardiac",
    "HGB": "Laboratory",
    "Platelet": "Laboratory",
    "STEMI": "ACS presentation",
    "TG": "Laboratory",
    "TCL": "Laboratory",
    "CaI": "Laboratory",
    "LDL": "Laboratory",
    "UA": "ACS presentation",
    "Aneurysm": "Anatomy",
    "stent overlap": "Procedural",
    "Age": "Demographics",
    "Max-stent diameter": "Procedural",
    "Fast-Glu": "Laboratory",
    "Stent type-SES_xiencev": "Stent type",
    "Stent type-SES_tivoli": "Stent type",
    "Initial diagnosis-AMI": "ACS presentation",
    TIME_AT_RISK: "Time-at-risk",
}

SHARED_ROWS = [
    {
        "Feature": "Clopidogrel",
        "Domain": "Medication",
        "Statistical evidence": "χ² FDR; multivariable Wald CI excludes 1",
        "ML evidence": "cat / lr three-way (LOCO ∩ SHAP ∩ FFS)",
        "Why both keep it": "Full-cohort drug association that LR and CatBoost also need for val PR-AUC",
    },
    {
        "Feature": "HbA1c",
        "Domain": "Laboratory",
        "Statistical evidence": "MW r = 0.052, q = 7e-4",
        "ML evidence": "cat / lgb three-way; LightGBM’s only consensus name",
        "Why both keep it": "Glycaemic FDR hit that boosting still needs after WBC is dropped",
    },
    {
        "Feature": "LV",
        "Domain": "Cardiac",
        "Statistical evidence": "Welch d = 1.13, q = 3.3e-16",
        "ML evidence": "cat / rf three-way",
        "Why both keep it": "Large location shift and a high-gain tree split",
    },
    {
        "Feature": "No postdilation",
        "Domain": "Procedural",
        "Statistical evidence": "χ² OR ≈ 5.4",
        "ML evidence": "CatBoost three-way",
        "Why both keep it": "Complement of 1.1:1 post-dilation; CatBoost keeps this flag on this dump",
    },
    {
        "Feature": "eGFR",
        "Domain": "Laboratory",
        "Statistical evidence": "Welch d = −0.71, q = 3.7e-19",
        "ML evidence": "lr / rf_b / xgb / xgb_b three-way; LOCO all-7",
        "Why both keep it": "Filtration: largest continuous effect; LOCO drop is costly",
    },
]

STATS_ONLY_WHY = {
    "WBC": "Strongest continuous FDR hit, but dropped from the ML matrix with TSSI on this run (anti-leakage). Selectors never see it",
    "1.1:1Post dilation": "χ² complement of No postdilation (now in consensus). Still in several unions; no model three-way on this dump",
    "CKD90": "Binary renal cutpoint; ML prefers continuous eGFR/Cre rather than the threshold",
    "CKD5": "FDR hit but collinear with eGFR; adjusted OR flips sign. Often selected, not in 3-way consensus",
    "3-vessel disease": "Anatomy binary; collinear with NO.of vessels / multi-vessel CAD",
    "Multi-vessel CAD": "Overlaps Single-vessel and 3-vessel; shared anatomical information",
    "Single-vessel disease": "Complement of multi-vessel disease (same 2×2 inverted)",
    "NO.of vessels": "Continuous vessel count; collinear with the vessel-disease binaries",
    "No.of stents per lesion": "Tiny univariate effect (MW r=0.037); not in any model three-way set",
    "Total stent length": "Small univariate effect; collinear with stent count / vessel burden; frequently selected",
    "Diabetes": "Univariate FDR; multivariable CI includes 1; trees may split on HbA1c instead",
    "PES": "Stent-polymer binary; collinear with the 9-level brand column",
    "Stent type-SES": "χ² on 9 collapsed brands. ML one-hots those 9 levels; xgb_b keeps Stent type-SES_xiencev instead",
    "Previous PCI": "Fisher OR=6.49. Frequently selected, but no model puts it in LOCO ∩ SHAP ∩ FFS on PR-AUC",
    "Fiberinogen": "Weak MW r=0.035. Frequently selected; PR-AUC three-way no longer keeps it",
}

ML_ONLY_WHY = {
    "Cre": ("ns (p=0.88)", "Redundant with eGFR marginally; lr / rf_b / xgb_b still use Cre as a renal surrogate"),
    "Men": ("ns (p=0.27)", "Univariate ns. Men×eGFR interaction is FDR-significant; LR uses sex as an additive offset"),
    "HGB": ("raw p=0.039, FDR ns", "rf / xgb / xgb_b three-way — ranking, not a location test"),
    "LDL": ("ns (p=0.33)", "cat / rf three-way lipid split on the val-slice PR-AUC"),
    "Stent type-SES_xiencev": (
        "parent Stent type-SES is FDR; dummy is not a univariate column",
        "xgb_b three-way on one 9-level brand dummy after OHE; parent name stays stats-only",
    ),
}

BUCKET = {
    "Clopidogrel": "Robust intersection",
    "HbA1c": "Robust intersection",
    "LV": "Robust intersection",
    "No postdilation": "Robust intersection",
    "eGFR": "Robust intersection",
    "WBC": "Dropped from ML view (anti-leakage)",
    "1.1:1Post dilation": "Collinear family (stats-only)",
    "CKD90": "Collinear family (stats-only)",
    "CKD5": "Collinear family (stats-only)",
    "3-vessel disease": "Collinear family (stats-only)",
    "Multi-vessel CAD": "Collinear family (stats-only)",
    "Single-vessel disease": "Collinear family (stats-only)",
    "NO.of vessels": "Collinear family (stats-only)",
    "PES": "Collinear family (stats-only)",
    "Stent type-SES": "Brand encoding (9-level OHE, not parent name)",
    "No.of stents per lesion": "Weak for top-20 / not in three-way",
    "Total stent length": "Weak for top-20 / not in three-way",
    "Diabetes": "Weak for top-20 / not in three-way",
    "Previous PCI": "Weak for top-20 / not in three-way",
    "Fiberinogen": "Weak for top-20 / not in three-way",
    "Cre": "Surrogate of an FDR hit",
    "Men": "Interaction / offset",
    "HGB": "Borderline univariate, ML split",
    "LDL": "Hold-out / metric artefact",
    "Stent type-SES_xiencev": "Brand encoding (9-level OHE, not parent name)",
    TIME_AT_RISK: "Structural time-at-risk",
}


def ml_consensus_from_dump() -> list[str]:
    cons = pd.read_csv(DUMP / "selector_common_by_model_algorithms.csv")
    names: set[str] = set()
    for feats in cons["features"].fillna(""):
        names |= {x.strip() for x in str(feats).split(";") if x.strip()}
    return sorted(names)


def ml_frequent_extra(ml: set[str]) -> list[str]:
    long = pd.read_csv(DUMP / "selector_summary_long.csv")
    cnt = long.groupby("feature").size()
    extra = [f for f, n in cnt.sort_values(ascending=False).items() if n >= 14 and f not in ml]
    return extra


def _save(fig, name: str) -> None:
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, dpi=300, bbox_inches="tight")


def _write_csv(df: pd.DataFrame, name: str) -> None:
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / name, index=False)


def _table_image(df: pd.DataFrame, name: str) -> None:
    n_rows, n_cols = df.shape
    fig_w = max(10, 0.22 * sum(len(str(c)) for c in df.columns) / 2)
    fig_h = max(2.2, 0.38 * (n_rows + 2))
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
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HARMONY[7])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FA")
    _save(fig, name)
    plt.close(fig)


def _member_row(feat, fdr, multi, cons, freq, label):
    return {
        "Feature": feat,
        "Stats FDR": "Yes" if fdr else "No",
        "Stats multivariable": "Yes" if multi else "No",
        "ML consensus": "Yes" if cons else "No",
        "ML frequent": "Yes" if freq else "No",
        "Set": label,
    }


def build_membership(ml: set[str], freq_extra: list[str]) -> pd.DataFrame:
    stats, ml_freq = set(STATS_FDR), ml | set(freq_extra)
    rows = []
    for feat in sorted(stats & ml):
        rows.append(_member_row(feat, True, feat in STATS_MULTIVAR, True, True, "Both (FDR ∩ ML consensus)"))
    for feat in sorted(ml - stats):
        rows.append(_member_row(feat, False, False, True, True, "ML consensus only"))
    for feat in sorted(set(freq_extra) - stats - ml):
        rows.append(_member_row(feat, False, False, False, True, "ML frequent only"))
    for feat in sorted(stats - ml):
        rows.append(
            _member_row(
                feat,
                True,
                feat in STATS_MULTIVAR,
                False,
                feat in ml_freq,
                "Stats FDR only",
            )
        )
    rows.append(
        {
            "Feature": TIME_AT_RISK,
            "Stats FDR": "Yes",
            "Stats multivariable": "structural",
            "ML consensus": "No",
            "ML frequent": "No",
            "Set": "Structural (dropped from ML)",
        }
    )
    return pd.DataFrame(rows)


def fig_venn(stats: set[str], ml: set[str]) -> None:
    inter = stats & ml
    only_s, only_m = stats - ml, ml - stats
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_aspect("equal")
    ax.axis("off")
    c1 = Circle((-0.55, 0), 1.05, facecolor=HARMONY[7], alpha=0.35, edgecolor=HARMONY[7], lw=2)
    c2 = Circle((0.55, 0), 1.05, facecolor=HARMONY[0], alpha=0.28, edgecolor=HARMONY[0], lw=2)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.text(-1.15, 1.15, "Statistical FDR", fontsize=11, fontweight="bold", color=HARMONY[7], ha="center")
    ax.text(1.15, 1.15, "ML consensus", fontsize=11, fontweight="bold", color=HARMONY[0], ha="center")
    ax.text(-0.85, 0.15, f"n = {len(only_s)}", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.85, 0.15, f"n = {len(only_m)}", ha="center", fontsize=13, fontweight="bold")
    ax.text(0, 0.08, f"n = {len(inter)}", ha="center", fontsize=13, fontweight="bold")
    ax.text(0, -0.28, "\n".join(sorted(inter)), ha="center", va="top", fontsize=8)
    union = len(stats | ml)
    jacc = len(inter) / union
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.45, 1.45)
    ax.set_title(
        f"Overlap of extraction catalogues   Jaccard = {jacc:.2f}  ({len(inter)} / {union})",
        pad=8,
    )
    _save(fig, "fig1_venn_overlap.png")
    plt.close(fig)


def fig_presence(member: pd.DataFrame) -> None:
    plot = member.copy()
    cols = ["Stats FDR", "Stats multivariable", "ML consensus", "ML frequent"]
    mat = []
    for _, r in plot.iterrows():
        row = []
        for c in cols:
            val = r[c]
            if val == "Yes":
                row.append(1)
            elif val == "structural":
                row.append(0.5)
            else:
                row.append(0)
        mat.append(row)
    mat = np.array(mat)
    fig_h = max(8, 0.28 * len(plot))
    fig, ax = plt.subplots(figsize=(7.2, fig_h))
    cmap = plt.cm.colors.ListedColormap(["#F4F7FA", HARMONY[8], HARMONY[7]])
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["Stats FDR", "Stats multivariable", "ML consensus", "ML frequent"], rotation=25, ha="right")
    ax.set_yticks(range(len(plot)))
    ax.set_yticklabels(plot["Feature"].tolist(), fontsize=7)
    ax.set_title("Feature presence by extractor")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
    _save(fig, "fig2_presence_heatmap.png")
    plt.close(fig)


def fig_buckets(ml: list[str]) -> None:
    names = list(STATS_FDR) + [n for n in ml if n not in STATS_FDR] + [TIME_AT_RISK]
    counts: dict[str, int] = {}
    for n in names:
        b = BUCKET[n]
        counts[b] = counts.get(b, 0) + 1
    order = [
        "Robust intersection",
        "Dropped from ML view (anti-leakage)",
        "Collinear family (stats-only)",
        "Brand encoding (9-level OHE, not parent name)",
        "Weak for top-20 / not in three-way",
        "Surrogate of an FDR hit",
        "Interaction / offset",
        "Borderline univariate, ML split",
        "Hold-out / metric artefact",
        "Structural time-at-risk",
    ]
    order = [k for k in order if k in counts]
    vals = [counts[k] for k in order]
    colors = [HARMONY[i % len(HARMONY)] for i in range(len(order))]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(order[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("Number of compared names")
    ax.set_title("Primary methodological bucket (one bucket per feature)")
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.15, y, str(v), va="center", fontsize=9)
    _save(fig, "fig3_reason_buckets.png")
    plt.close(fig)


def fig_domains(stats: set[str], ml: set[str]) -> None:
    domain_order = [
        "Laboratory",
        "Laboratory / renal",
        "Cardiac",
        "Procedural",
        "Anatomy",
        "Stent type",
        "History",
        "Comorbidity",
        "Medication",
        "Demographics",
        "ACS presentation",
    ]
    s_counts, m_counts = [], []
    for d in domain_order:
        s_counts.append(sum(1 for x in stats if DOMAINS.get(x) == d))
        m_counts.append(sum(1 for x in ml if DOMAINS.get(x) == d))
    keep = [i for i, (a, b) in enumerate(zip(s_counts, m_counts)) if a or b]
    labels = [domain_order[i] for i in keep]
    s_counts = [s_counts[i] for i in keep]
    m_counts = [m_counts[i] for i in keep]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.barh(y - 0.18, s_counts, height=0.35, color=HARMONY[7], label="Statistical FDR")
    ax.barh(y + 0.18, m_counts, height=0.35, color=HARMONY[0], label="ML consensus")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Count of extracted names")
    ax.set_title("Extracted features by clinical domain")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig4_domain_counts.png")
    plt.close(fig)


def main() -> None:
    apply_style()
    ml_list = ml_consensus_from_dump()
    stats, ml = set(STATS_FDR), set(ml_list)
    inter, union = stats & ml, stats | ml
    freq_extra = ml_frequent_extra(ml)
    print(f"stats={len(stats)} ml={len(ml)} intersection={len(inter)} union={len(union)} "
          f"Jaccard={len(inter)/len(union):.4f}")
    print("ML consensus:", "; ".join(ml_list))
    print("Intersection:", "; ".join(sorted(inter)))
    print("Frequent extra:", "; ".join(freq_extra))

    member = build_membership(ml, freq_extra)
    _write_csv(member, "table_feature_by_method.csv")
    _table_image(member, "table_feature_by_method.png")

    shared = pd.DataFrame(SHARED_ROWS)
    _write_csv(shared, "table_shared_features.csv")
    _table_image(shared, "table_shared_features.png")

    stats_only = pd.DataFrame(
        [
            {
                "Feature": f,
                "Domain": DOMAINS[f],
                "Why statistical methods keep it and ML top-20 does not": STATS_ONLY_WHY[f],
            }
            for f in STATS_FDR
            if f not in ml
        ]
    )
    _write_csv(stats_only, "table_stats_only.csv")
    _table_image(stats_only, "table_stats_only.png")

    ml_only = pd.DataFrame(
        [
            {
                "Feature": f,
                "Univariate vs VLST": ML_ONLY_WHY[f][0],
                "Why ML consensus keeps it and FDR does not": ML_ONLY_WHY[f][1],
            }
            for f in ml_list
            if f not in stats
        ]
    )
    _write_csv(ml_only, "table_ml_only.csv")
    _table_image(ml_only, "table_ml_only.png")

    fig_venn(stats, ml)
    fig_presence(member)
    fig_buckets(ml_list)
    fig_domains(stats, ml)
    print("Wrote Part 3 figures to:")
    for d in OUT_DIRS:
        print(" ", d)


if __name__ == "__main__":
    main()
