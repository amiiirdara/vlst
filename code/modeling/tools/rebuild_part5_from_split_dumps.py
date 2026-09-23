#!/usr/bin/env python3
"""Regenerate Part 5 paper figures/tables from the two split-notebook dumps.

Sources (Kaggle papermill 2026-09-20):
  tabpfn_interpretability_fs_pdp.ipynb  → Kaggle_tabpfn_intrepretebility_results/fs_pdp_MI/
  tabpfn_interpretability_shap.ipynb    → Kaggle_tabpfn_intrepretebility_results/shap/

Consensus Table 5 / Figure 13 merge SHAP mean(|value|) from the shap dump
into the fs_pdp ranking (the fs dump CSV has shap_mean_abs=0). Formula matches
the notebooks: Borda-style mean of rank(ascending=True) normalized to [0, 1];
n_methods counts {MI top-15, SFS freq ≥ 0.5, SHAP top-15}.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "modeling" / "tools"))
from figure_style import HARMONY, apply_style  # noqa: E402
from paper_paths import paper_figure_dirs  # noqa: E402

FS_DIR = (
    ROOT
    / "code/modeling/interpretability/Kaggle_tabpfn_intrepretebility_results"
    / "fs_pdp_MI/modeling_tabpfn"
)
SHAP_DIR = (
    ROOT
    / "code/modeling/interpretability/Kaggle_tabpfn_intrepretebility_results"
    / "shap/modeling_tabpfn"
)
OUT_DIRS = paper_figure_dirs("05_tabpfn_interpretability")
TOP_K = 15
N_SEEDS = 8


def _copy_to(src: Path, dest_name: str) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d / dest_name)


def _write_df(df: pd.DataFrame, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / name, index=False)


def save_table_png(
    display: pd.DataFrame,
    *,
    title: str,
    filename: str,
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
    for d in OUT_DIRS:
        fig.savefig(d / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def copy_dump_assets() -> None:
    png_map = {
        FS_DIR / "interpretability_pdp.png": "paper_fig1_pdp_continuous.png",
        FS_DIR / "interpretability_pdp_binary.png": "paper_fig2_pdp_binary.png",
        SHAP_DIR / "sv_interpretability_shap_summary.png": "paper_fig3_shap_summary.png",
        SHAP_DIR / "sv_interpretability_shap_scatter_f0.png": "paper_fig4_shap_scatter_age.png",
        SHAP_DIR / "sv_interpretability_shap_bar.png": "paper_fig5_shap_bar.png",
        SHAP_DIR / "sv_interpretability_shap_beeswarm.png": "paper_fig6_shap_beeswarm.png",
        SHAP_DIR / "sv_interpretability_shap_waterfall_row0.png": "paper_fig7_shap_waterfall.png",
        SHAP_DIR / "k_ssi_interpretability_network_top15.png": "paper_fig8_ksii_network.png",
        SHAP_DIR / "k_ssi_interpretability_upset.png": "paper_fig9_ksii_upset.png",
        SHAP_DIR / "k_ssi_shapiq_network_top15.png": "paper_fig11_shapiq_network.png",
        SHAP_DIR / "k_ssi_shapiq_upset.png": "paper_fig12_shapiq_upset.png",
    }
    for src, dest in png_map.items():
        _copy_to(src, dest)

    csvs = [
        (FS_DIR / "interpretability_mutual_info_ranking.csv", "interpretability_mutual_info_ranking.csv"),
        (FS_DIR / "interpretability_feature_stability.csv", "interpretability_feature_stability.csv"),
        (FS_DIR / "interpretability_feature_stability_summary.csv", "interpretability_feature_stability_summary.csv"),
        (FS_DIR / "interpretability_pdp_binary.csv", "interpretability_pdp_binary.csv"),
        (FS_DIR / "interpretability_train_indices.csv", "interpretability_train_indices.csv"),
        (FS_DIR / "interpretability_heldout_indices.csv", "interpretability_heldout_indices.csv"),
        (SHAP_DIR / "interpretability_shap_mean_abs.csv", "interpretability_shap_mean_abs.csv"),
        (SHAP_DIR / "interpretability_shap_explain_indices.csv", "interpretability_shap_explain_indices.csv"),
    ]
    for src, dest in csvs:
        _copy_to(src, dest)


def merge_consensus() -> pd.DataFrame:
    mi = pd.read_csv(FS_DIR / "interpretability_mutual_info_ranking.csv")
    stab = pd.read_csv(FS_DIR / "interpretability_feature_stability.csv")
    shap = pd.read_csv(SHAP_DIR / "interpretability_shap_mean_abs.csv")

    report = mi[["feature", "mutual_info"]].copy()
    mi_top = set(mi.sort_values("mutual_info", ascending=False).head(TOP_K)["feature"])
    report["in_mi_top"] = report["feature"].isin(mi_top)

    report = report.merge(stab[["feature", "selection_freq"]], on="feature", how="left")
    report["selection_freq"] = report["selection_freq"].fillna(0.0)

    report = report.merge(shap[["feature", "shap_mean_abs"]], on="feature", how="left")
    report["shap_mean_abs"] = report["shap_mean_abs"].fillna(0.0)
    shap_top = set(shap.sort_values("shap_mean_abs", ascending=False).head(TOP_K)["feature"])
    report["in_shap_top"] = report["feature"].isin(shap_top)

    report["n_methods"] = (
        report["in_mi_top"].astype(int)
        + (report["selection_freq"] >= 0.5).astype(int)
        + report["in_shap_top"].astype(int)
    )

    signals = ["mutual_info", "selection_freq", "shap_mean_abs"]
    norm = []
    for col in signals:
        ranks = report[col].rank(method="average", ascending=True)
        norm.append((ranks - 1) / max(len(report) - 1, 1))
    report["importance_score"] = np.mean(norm, axis=0)

    report = report.sort_values(
        ["importance_score", "n_methods"], ascending=[False, False]
    ).reset_index(drop=True)
    report.insert(0, "rank", range(1, len(report) + 1))
    cols = [
        "rank",
        "feature",
        "importance_score",
        "n_methods",
        "selection_freq",
        "shap_mean_abs",
        "mutual_info",
        "in_mi_top",
        "in_shap_top",
    ]
    report = report[cols]
    _write_df(report, "interpretability_feature_importance_report.csv")
    return report


def write_tables(consensus: pd.DataFrame) -> None:
    methods = pd.DataFrame(
        [
            [
                "mutual_info_classif",
                "Univariate association",
                "sklearn",
                "0 TabPFN calls; train n=3629; 80 scores; fs_pdp notebook",
            ],
            [
                "Stability (repeated SFS)",
                "Selection frequency",
                "local TabPFN (v3.5, FS_THINKING_MODE=False)",
                f"{N_SEEDS}/{N_SEEDS} resamples × top-10 forward SFS, AP scoring, train",
            ],
            [
                "PDP",
                "Average predicted probability (empirical prior)",
                "local TabPFN (PDP_USE_CLIENT=False)",
                "Train n=3629; y-axis empirical prior / not Part 4 nested-CV risk",
            ],
            [
                "SHAP (shapiq SV)",
                "Local attributions",
                "client thinking-high (INTERP_THINKING_MODE=True)",
                "All 1,556 held-out rows; fit/background = train; budget=256; shap notebook",
            ],
            [
                "k-SII / SHAP-IQ",
                "Pairwise interactions",
                "client thinking-high",
                "One held-out VLST=1 (pos 20, cohort 5176); budget=256; shap notebook",
            ],
            [
                "Consensus (Borda)",
                "Mean of normalized ranks",
                "aggregate",
                "Train MI + train stability + held-out mean(|SHAP|); SHAP merged from shap dump",
            ],
        ],
        columns=["Method", "Question", "Backend", "Notebook setting"],
    )
    _write_df(methods, "paper_table0_methods.csv")
    save_table_png(
        methods,
        title="Part 5 interpretability methods (fs_pdp + shap split notebooks; 80 columns)",
        filename="paper_table0_methods.png",
        figsize=(14.2, 3.6),
        fontsize=7,
    )

    mi = pd.read_csv(FS_DIR / "interpretability_mutual_info_ranking.csv")
    mi15 = mi.head(TOP_K).copy()
    _write_df(mi15, "paper_table1_mutual_info.csv")
    mi_disp = mi15.copy()
    mi_disp["mutual_info"] = mi_disp["mutual_info"].map(lambda x: f"{x:.6f}")
    mi_disp = mi_disp.rename(
        columns={"rank": "Rank", "feature": "Feature", "mutual_info": "Mutual information"}
    )
    save_table_png(
        mi_disp,
        title="Top 15 mutual information (train n=3629; 80-column dump; WBC dropped)",
        filename="paper_table1_mutual_info.png",
        figsize=(8.5, 5.6),
        fontsize=8,
    )

    stab = pd.read_csv(FS_DIR / "interpretability_feature_stability.csv")
    _write_df(stab, "paper_table2_stability.csv")
    stab_disp = stab.copy()
    stab_disp["Selected"] = stab_disp["times_selected"].map(lambda n: f"{int(n)}/{N_SEEDS}")
    stab_disp = stab_disp.rename(columns={"feature": "Feature", "selection_freq": "Frequency"})
    stab_disp = stab_disp[["Feature", "Selected", "Frequency"]]
    stab_disp["Frequency"] = stab_disp["Frequency"].map(lambda x: f"{x:.3f}")
    save_table_png(
        stab_disp,
        title=f"Stability selection frequency ({N_SEEDS}/{N_SEEDS} seeds × top-10 forward SFS; train)",
        filename="paper_table2_stability.png",
        figsize=(8.2, 6.4),
        fontsize=8,
    )

    pdp = pd.read_csv(FS_DIR / "interpretability_pdp_binary.csv")
    table3 = pd.DataFrame(
        {
            "Feature": pdp["feature"],
            "P(y=1 | 0)": pdp["p_absent"].map(lambda x: round(float(x), 6)),
            "P(y=1 | 1)": pdp["p_present"].map(lambda x: round(float(x), 6)),
            "ΔP": pdp["delta"].map(lambda x: round(float(x), 6)),
        }
    )
    _write_df(table3, "paper_table3_pdp_binary.csv")
    t3d = pd.DataFrame(
        {
            "Feature": pdp["feature"],
            "P(y=1 | 0)": pdp["p_absent"].map(lambda x: f"{x:.6f}"),
            "P(y=1 | 1)": pdp["p_present"].map(lambda x: f"{x:.6f}"),
            "ΔP": pdp["delta"].map(lambda x: f"{x:+.6f}"),
        }
    )
    save_table_png(
        t3d,
        title="Binary PDP on empirical prior (train n=3629; not Part 4 nested-CV risk)",
        filename="paper_table3_pdp_binary.png",
        figsize=(11.2, 3.4),
        fontsize=8,
    )

    shap = pd.read_csv(SHAP_DIR / "interpretability_shap_mean_abs.csv")
    shap15 = shap.head(TOP_K).copy()
    _write_df(shap15, "paper_table4_shap_mean_abs.csv")
    s4 = shap15.rename(columns={"rank": "Rank", "feature": "Feature", "shap_mean_abs": "mean(|SHAP|)"})
    s4["mean(|SHAP|)"] = s4["mean(|SHAP|)"].map(lambda x: f"{x:.4f}")
    save_table_png(
        s4,
        title="Mean(|SHAP|) top 15 of 80 (1,556 held-out rows; client thinking-high)",
        filename="paper_table4_shap_mean_abs.png",
        figsize=(8.2, 5.6),
        fontsize=8,
    )

    cons15 = consensus.head(TOP_K).copy()
    _write_df(cons15, "paper_table5_consensus.csv")
    c5 = pd.DataFrame(
        {
            "Rank": cons15["rank"],
            "Feature": cons15["feature"],
            "Score": cons15["importance_score"].map(lambda x: f"{x:.4f}"),
            "n methods": cons15["n_methods"].map(lambda n: f"{int(n)}/3"),
            "Stability": cons15["selection_freq"].map(lambda x: f"{x:.3f}"),
            "mean(|SHAP|)": cons15["shap_mean_abs"].map(lambda x: f"{x:.4f}"),
            "MI": cons15["mutual_info"].map(lambda x: f"{x:.6f}"),
            "In MI top": cons15["in_mi_top"].map(lambda x: "yes" if bool(x) else "no"),
            "In SHAP top": cons15["in_shap_top"].map(lambda x: "yes" if bool(x) else "no"),
        }
    )
    save_table_png(
        c5,
        title="Borda consensus (train MI + train SFS + held-out mean|SHAP|; 80 columns)",
        filename="paper_table5_consensus.png",
        figsize=(13.6, 5.8),
        fontsize=7,
    )

    apply_style()
    bars = cons15.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(bars) + 1))
    ax.barh(bars["feature"], bars["importance_score"], color=HARMONY[7])
    for i, (score, n) in enumerate(zip(bars["importance_score"], bars["n_methods"])):
        ax.text(score, i, f"  {int(n)}/3 methods", va="center", fontsize=8, color="#333")
    ax.set_xlabel("Aggregated importance (mean normalized rank, 1 = strongest)")
    ax.set_title("VLST — top 15 features by consensus ranking (fs_pdp + shap dumps)")
    ax.margins(x=0.15)
    fig.tight_layout()
    for d in OUT_DIRS:
        fig.savefig(d / "paper_fig13_consensus_ranking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not FS_DIR.is_dir() or not SHAP_DIR.is_dir():
        raise SystemExit(f"Missing dump dir: {FS_DIR} or {SHAP_DIR}")
    copy_dump_assets()
    consensus = merge_consensus()
    write_tables(consensus)
    print("Regenerated Part 5 paper_figures from fs_pdp_MI + shap dumps.")
    print("  SFS 8/8:", list(consensus.loc[consensus["selection_freq"] == 1.0, "feature"]))
    print("  SHAP #1:", consensus.sort_values("shap_mean_abs", ascending=False).iloc[0]["feature"])
    print("  Consensus #1:", consensus.iloc[0]["feature"], f"{consensus.iloc[0]['importance_score']:.4f}")


if __name__ == "__main__":
    main()
