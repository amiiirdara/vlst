"""Rebuild the interpretability consensus report from two downloaded Kaggle outputs.

``tabpfn_interpretability_fs_pdp.ipynb`` writes mutual information and stability
selection; ``tabpfn_interpretability_shap.ipynb`` writes SHAP mean(|value|). They
run as separate Kaggle notebooks, so the consensus cell of the first one only sees
SHAP when the second one's output is attached. This script joins the two folders
locally with the same rules as that cell:

- top-k membership for MI and SHAP, selection frequency >= 0.5 for stability;
- importance score = mean normalised rank over the signals present (1 = strongest).

Both folders must come from the same analysis variant (same folder name, e.g.
``modeling_tabpfn_r3_residual``); a mismatch stops the script.

Usage::

    python code/modeling/tools/rebuild_consensus_report.py FS_DIR SHAP_DIR [OUT_DIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOP_K = 15


def build_report(fs_dir: Path, shap_dir: Path) -> pd.DataFrame:
    mi = pd.read_csv(fs_dir / "interpretability_mutual_info_ranking.csv")[["feature", "mutual_info"]]
    stab = pd.read_csv(fs_dir / "interpretability_feature_stability.csv")[["feature", "selection_freq"]]
    shap = pd.read_csv(shap_dir / "interpretability_shap_mean_abs.csv")[["feature", "shap_mean_abs"]]

    features = sorted(set(mi["feature"]) | set(stab["feature"]) | set(shap["feature"]))
    report = pd.DataFrame({"feature": features})
    report = report.merge(mi, on="feature", how="left")
    report = report.merge(stab, on="feature", how="left")
    report = report.merge(shap, on="feature", how="left")
    # A feature never kept by SFS is absent from the stability CSV: frequency 0.
    report["selection_freq"] = report["selection_freq"].fillna(0.0)
    report["shap_mean_abs"] = report["shap_mean_abs"].fillna(0.0)

    report["in_mi_top"] = report["feature"].isin(mi.nlargest(TOP_K, "mutual_info")["feature"])
    report["in_shap_top"] = report["feature"].isin(shap.nlargest(TOP_K, "shap_mean_abs")["feature"])
    report["n_methods"] = (
        report["in_mi_top"].astype(int)
        + (report["selection_freq"] >= 0.5).astype(int)
        + report["in_shap_top"].astype(int)
    )

    signals = ["mutual_info", "selection_freq", "shap_mean_abs"]
    norm = [
        (report[c].rank(method="average", ascending=True) - 1) / max(len(report) - 1, 1)
        for c in signals
    ]
    report["importance_score"] = np.nanmean(norm, axis=0)

    report = report.sort_values(["importance_score", "n_methods"], ascending=[False, False]).reset_index(drop=True)
    report.insert(0, "rank", range(1, len(report) + 1))
    return report[
        ["rank", "feature", "importance_score", "n_methods", "selection_freq",
         "shap_mean_abs", "mutual_info", "in_mi_top", "in_shap_top"]
    ]


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    fs_dir, shap_dir = Path(sys.argv[1]), Path(sys.argv[2])
    if fs_dir.name != shap_dir.name:
        sys.exit(f"Variant mismatch: {fs_dir.name} vs {shap_dir.name}")
    out_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else fs_dir
    table = build_report(fs_dir, shap_dir)
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table.head(TOP_K).to_string(index=False))
    out = out_dir / "interpretability_feature_importance_report.csv"
    table.to_csv(out, index=False)
    print("Saved", out)
