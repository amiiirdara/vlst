#!/usr/bin/env python3
"""Rebuild Part 3 (stats vs ML overlap) from the verified catalogues.

Inputs are the FDR set from eda.ipynb (§8.1 of the evidence map) and the
LOCO ∩ SHAP ∩ FFS consensus union from baseline_feature_selections.ipynb (§8.6).
This script is the generating code that Part 3 previously lacked.
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

OUT_DIRS = [
    ROOT / "paper_results" / "03_stats_vs_ml" / "paper_figures",
    ROOT / "code" / "analyzes" / "stats_vs_ml" / "paper_figures",
]

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
ML_CONSENSUS = [
    "WBC",
    "eGFR",
    "LV",
    "Cre",
    "Men",
    "LVEF",
    "Previous PCI",
    "Fiberinogen",
    "HGB",
    "Platelet",
    "HL",
    "STEMI",
    "Hypertension",
    "Fast-Glu",
    "TG",
    "TCL",
    "CaI",
    "Min-stent diameter",
    "Current drinking",
    "History of HF",
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
ML_FREQUENT_EXTRA = [
    "Chronic renal insufficiency",
    "HDL",
    "History of peripheral vascular disease",
    "Initial diagnosis-AMI",
    "LDL",
    "NSTEMI",
    "Previous CABG",
    "Previous MI",
    "Stroke/TIA",
    "UA",
]
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
    "HL": "Comorbidity",
    "STEMI": "ACS presentation",
    "Hypertension": "Comorbidity",
    "Fast-Glu": "Laboratory",
    "TG": "Laboratory",
    "TCL": "Laboratory",
    "CaI": "Laboratory",
    "Min-stent diameter": "Procedural",
    "Current drinking": "Demographics",
    "History of HF": "History",
    TIME_AT_RISK: "Time-at-risk",
    "Chronic renal insufficiency": "Comorbidity",
    "HDL": "Laboratory",
    "History of peripheral vascular disease": "History",
    "Initial diagnosis-AMI": "ACS presentation",
    "LDL": "Laboratory",
    "NSTEMI": "ACS presentation",
    "Previous CABG": "History",
    "Previous MI": "History",
    "Stroke/TIA": "History",
    "UA": "ACS presentation",
}

SHARED_ROWS = [
    {
        "Feature": "WBC",
        "Domain": "Laboratory",
        "Statistical evidence": "MW r=0.13, q=9.5e-20",
        "ML evidence": "Global ML intersection; CatBoost/XGB/RF",
        "Why both methods keep it": "Mean shift and ranking feature (inflammation)",
    },
    {
        "Feature": "eGFR",
        "Domain": "Laboratory",
        "Statistical evidence": "Welch d=-0.71, q=3.7e-19",
        "ML evidence": "Global ML intersection; LOCO/SHAP core",
        "Why both methods keep it": "Renal filtration: models cannot compensate if dropped",
    },
    {
        "Feature": "LV",
        "Domain": "Cardiac",
        "Statistical evidence": "Welch d=1.13, q=3.3e-16",
        "ML evidence": "LOCO/SHAP cross-model; CatBoost/XGB",
        "Why both methods keep it": "Large location shift and a high-gain tree split",
    },
    {
        "Feature": "Fiberinogen",
        "Domain": "Laboratory",
        "Statistical evidence": "MW r=0.035, q=0.029",
        "ML evidence": "RF F2 consensus",
        "Why both methods keep it": "Weak haemostasis effect still used at recall-heavy F2",
    },
    {
        "Feature": "Previous PCI",
        "Domain": "History",
        "Statistical evidence": "Fisher OR=6.5, q=2e-4",
        "ML evidence": "RF F1 consensus",
        "Why both methods keep it": "Rare high-OR flag: 2x2 discovery and a pure binary split",
    },
]

STATS_ONLY_WHY = {
    "1.1:1Post dilation": "Complement of No postdilation; strong 2x2, collinear pair, may not enter the LOCO pool of 40",
    "No postdilation": "Strong univariate OR; multivariable OR attenuates to ~1 after the complement is in the model",
    "CKD90": "Binary renal cutpoint; ML prefers continuous eGFR/Cre rather than the threshold",
    "CKD5": "FDR hit but collinear with eGFR; adjusted OR flips sign. In ML union prefix, not in 3-way consensus",
    "3-vessel disease": "Anatomy binary; collinear with NO.of vessels / multi-vessel CAD",
    "Multi-vessel CAD": "Overlaps Single-vessel and 3-vessel; shared anatomical information",
    "Single-vessel disease": "Protective complement of multi-vessel disease",
    "NO.of vessels": "Continuous vessel count; collinear with the vessel-disease binaries",
    "No.of stents per lesion": "Tiny univariate effect (MW r=0.037); dropped from a 12-feature predictive shortlist",
    "Total stent length": "Small univariate effect; collinear with stent count / vessel burden",
    "HbA1c": "FDR hit that attenuates after adjustment; Diabetes/Fast-Glu compete",
    "Clopidogrel": "Full-cohort medication association; weak on a 28-event hold-out ranking task",
    "Diabetes": "Univariate FDR; multivariable CI includes 1; trees may split on labs instead",
    "PES": "Stent-polymer binary; collinear with Stent type-SES",
    "Stent type-SES": "9-level brand (shared encoder). Older ML runs one-hot the raw 106 strings and fragment the χ² signal",
}

ML_ONLY_WHY = {
    "Cre": ("ns (p=0.88)", "Univariate p=0.88 (redundant with eGFR). Models still use Cre as a renal surrogate"),
    "Men": ("ns (p=0.27)", "Univariate ns. Men×eGFR interaction is FDR-significant; LR uses sex as an additive offset"),
    "LVEF": ("raw p=0.033, FDR ns", "Borderline univariate (q=0.067). Joint logistic reverses sign (0.851 to 1.65) with LV in the model; trees still split on systolic function"),
    "HGB": ("raw p=0.039, FDR ns", "Raw p=0.039, FDR ns. CatBoost/XGB F-score consensus — threshold metrics, not mean tests"),
    "Fast-Glu": ("raw p=0.025, FDR ns", "Raw p=0.025, FDR ns. LR F2 consensus; correlated with HbA1c/diabetes"),
    "Platelet": ("ns", "Univariate ns. RF/CatBoost haemostasis panel with Fiberinogen"),
    "HL": ("ns", "Univariate ns. Lipid split that helps rare-event ranking in boosting/RF"),
    "STEMI": ("ns", "Univariate ns. Presentation split on hold-out PR-AUC/F2, not a 2x2 discovery"),
    "Current drinking": ("ns", "Univariate ns. LightGBM F1; unstable lifestyle split with 28 test events"),
    "History of HF": ("ns", "Univariate ns. LightGBM F2; sparse history indicator"),
    "Hypertension": ("ns", "Univariate ns. CatBoost F2; common comorbidity, weak marginal association"),
    "TG": ("ns", "Univariate ns. LR PR-AUC additive lipid term"),
    "TCL": ("ns", "Univariate ns. XGB PR-AUC lipid surrogate"),
    "Min-stent diameter": ("ns", "Univariate ns. LR PR-AUC geometric term after scaling"),
    "CaI": ("raw p=0.051, FDR ns", "Raw p=0.051, FDR ns. RF_b PR-AUC; near the FDR boundary"),
}

# One primary methodological bucket per compared name (Figure 3).
BUCKET = {
    "WBC": "Robust intersection",
    "eGFR": "Robust intersection",
    "LV": "Robust intersection",
    "Fiberinogen": "Robust intersection",
    "Previous PCI": "Robust intersection",
    "1.1:1Post dilation": "Collinear family (stats-only)",
    "No postdilation": "Collinear family (stats-only)",
    "CKD90": "Collinear family (stats-only)",
    "CKD5": "Collinear family (stats-only)",
    "3-vessel disease": "Collinear family (stats-only)",
    "Multi-vessel CAD": "Collinear family (stats-only)",
    "Single-vessel disease": "Collinear family (stats-only)",
    "NO.of vessels": "Collinear family (stats-only)",
    "PES": "Collinear family (stats-only)",
    "Stent type-SES": "Brand encoding (9-level vs raw 106 one-hot on older ML runs)",
    "No.of stents per lesion": "Not in LOCO pool / weak for top-12",
    "Total stent length": "Not in LOCO pool / weak for top-12",
    "HbA1c": "Not in LOCO pool / weak for top-12",
    "Clopidogrel": "Not in LOCO pool / weak for top-12",
    "Diabetes": "Not in LOCO pool / weak for top-12",
    "Cre": "Surrogate of an FDR hit",
    "Men": "Interaction / offset",
    "LVEF": "Borderline univariate, ML split",
    "HGB": "Borderline univariate, ML split",
    "Fast-Glu": "Borderline univariate, ML split",
    "CaI": "Borderline univariate, ML split",
    "Platelet": "Hold-out / metric artefact",
    "HL": "Hold-out / metric artefact",
    "STEMI": "Hold-out / metric artefact",
    "Current drinking": "Hold-out / metric artefact",
    "History of HF": "Hold-out / metric artefact",
    "Hypertension": "Hold-out / metric artefact",
    "TG": "Hold-out / metric artefact",
    "TCL": "Hold-out / metric artefact",
    "Min-stent diameter": "Hold-out / metric artefact",
    TIME_AT_RISK: "Structural time-at-risk",
}


def _save(fig, name: str) -> None:
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / name, dpi=300, bbox_inches="tight")


def _write_csv(df: pd.DataFrame, name: str) -> None:
    for out in OUT_DIRS:
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / name, index=False)


def _table_image(df: pd.DataFrame, name: str, col_widths: list[float] | None = None) -> None:
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
    if col_widths:
        for i, w in enumerate(col_widths):
            for r in range(n_rows + 1):
                tbl[(r, i)].set_width(w)
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(HARMONY[7])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F4F7FA")
    _save(fig, name)
    plt.close(fig)


def build_membership() -> pd.DataFrame:
    stats, ml = set(STATS_FDR), set(ML_CONSENSUS)
    ml_freq = ml | set(ML_FREQUENT_EXTRA)
    rows = []
    for feat in sorted(stats & ml):
        rows.append(_member_row(feat, True, feat in STATS_MULTIVAR, True, feat in ml_freq, "Both (FDR ∩ ML consensus)"))
    for feat in sorted(ml - stats):
        rows.append(_member_row(feat, False, False, True, True, "ML consensus only"))
    for feat in sorted(set(ML_FREQUENT_EXTRA)):
        rows.append(_member_row(feat, False, False, False, True, "ML frequent only"))
    for feat in sorted(stats - ml):
        rows.append(
            _member_row(
                feat,
                True,
                feat in STATS_MULTIVAR,
                False,
                False,
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


def _member_row(feat, fdr, multi, cons, freq, label):
    return {
        "Feature": feat,
        "Stats FDR": "Yes" if fdr else "No",
        "Stats multivariable": "Yes" if multi else "No",
        "ML consensus": "Yes" if cons else "No",
        "ML frequent": "Yes" if freq else "No",
        "Set": label,
    }


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
    ax.set_title(f"Overlap of extraction catalogues   Jaccard = {jacc:.2f}  (5 / {union})", pad=8)
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


def fig_buckets() -> None:
    names = list(STATS_FDR) + [n for n in ML_CONSENSUS if n not in STATS_FDR] + [TIME_AT_RISK]
    counts = {}
    for n in names:
        b = BUCKET[n]
        counts[b] = counts.get(b, 0) + 1
    order = [
        "Robust intersection",
        "Collinear family (stats-only)",
        "One-hot fragmentation",
        "Not in LOCO pool / weak for top-12",
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
    stats, ml = set(STATS_FDR), set(ML_CONSENSUS)
    inter = stats & ml
    union = stats | ml
    assert len(STATS_FDR) == 20
    assert len(ML_CONSENSUS) == 20
    assert inter == {"WBC", "eGFR", "LV", "Fiberinogen", "Previous PCI"}
    assert len(inter) == 5
    assert len(union) == 35
    jacc = len(inter) / len(union)
    print(f"stats={len(stats)} ml={len(ml)} intersection={len(inter)} union={len(union)} Jaccard={jacc:.4f}")

    member = build_membership()
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
                "Why statistical methods keep it and ML top-12 does not": STATS_ONLY_WHY[f],
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
            for f in ML_CONSENSUS
            if f not in stats
        ]
    )
    _write_csv(ml_only, "table_ml_only.csv")
    _table_image(ml_only, "table_ml_only.png")

    fig_venn(stats, ml)
    fig_presence(member)
    fig_buckets()
    fig_domains(stats, ml)
    print("Wrote figures and tables to:")
    for d in OUT_DIRS:
        print(" ", d)


if __name__ == "__main__":
    main()
