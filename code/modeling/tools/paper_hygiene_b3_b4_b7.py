#!/usr/bin/env python3
"""Paper hygiene that does not touch the TabPFN notebooks.

B3  Stratified bootstrap CIs and paired PR-AUC tests on committed OOF CSVs.
B4  Table 4b: one representative per collinear block, VIF, EPV, N_BOOT=2000.
B7  Clinical cohort-characteristics table from VLST.csv (not a photocopy of Wang Table 1).

B11 is data-access and is not computed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "modeling" / "tools"))
from figure_style import HARMONY, apply_style  # noqa: E402

N_BOOT = 2000
SEED = 42
TARGET = "Stent thrombosis"

P4_DIRS = [
    ROOT / "paper_results" / "04_tabpfn_rating" / "paper_figures",
    ROOT / "code" / "modeling" / "rating" / "paper_figures",
    ROOT / "data" / "result" / "modeling_results" / "paper_figures",
]
P4_TABLES = ROOT / "data" / "result" / "modeling_results" / "tables"
EDA_DIRS = [
    ROOT / "paper_results" / "01_eda" / "paper_figures",
    ROOT / "code" / "analyzes" / "paper_figures",
    ROOT / "data" / "result" / "eda" / "paper_figures",
]

OOF_PATH = ROOT / "data" / "result" / "modeling_results" / "oof" / "oof_predictions.csv"
VLST_PATH = ROOT / "data" / "raw" / "VLST.csv"

# Table 4 original 17 names (unidentified: complements + renal encodings).
TABLE4_FULL = [
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
]
CONTINUOUS = {
    "WBC",
    "eGFR",
    "LV",
    "CKD5",
    "No.of stents per lesion",
    "HbA1c",
    "NO.of vessels",
    "Total stent length",
    "Fiberinogen",
}

# One representative per collinear block.
# Dropped: No postdilation (complement of 1.1:1Post dilation);
#          CKD5, CKD90 (deterministic encodings of eGFR);
#          3-vessel disease (vessel-count family; keep NO.of vessels).
TABLE4B = [
    "WBC",
    "eGFR",
    "LV",
    "No.of stents per lesion",
    "HbA1c",
    "NO.of vessels",
    "Total stent length",
    "Fiberinogen",
    "1.1:1Post dilation",
    "Previous PCI",
    "Clopidogrel",
    "Diabetes",
    "PES",
]

OOF_MODELS = [
    ("TabPFN (thinking-high)", "tabpfn_prob"),
    ("LightGBM", "lightgbm_prob"),
    ("XGBoost", "xgboost_prob"),
    ("TabPFN (local)", "tabpfn_(local)_prob"),
    ("CatBoost", "catboost_prob"),
    ("Random Forest", "random_forest_prob"),
    ("Logistic Regression", "logistic_regression_prob"),
]


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
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.3f}"


def fmt_ci(lo: float, hi: float, nd: int = 4) -> str:
    return f"[{lo:.{nd}f}, {hi:.{nd}f}]"


# ---------------------------------------------------------------------------
# Unweighted logit (Newton) + VIF
# ---------------------------------------------------------------------------


def fit_logit(X: np.ndarray, y: np.ndarray, max_iter: int = 80, tol: float = 1e-8):
    """Unweighted Bernoulli logit. X has no intercept; one is prepended."""
    n = X.shape[0]
    Z = np.column_stack([np.ones(n), X])
    pdim = Z.shape[1]
    beta = np.zeros(pdim)
    ok = False
    H = np.eye(pdim)
    for _ in range(max_iter):
        eta = np.clip(Z @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        H = (Z * w[:, None]).T @ Z
        g = Z.T @ (y - mu)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(H, g, rcond=None)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            ok = True
            break
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
        ok = False
    return beta, cov, ok


def vif_series(X: np.ndarray, names: list[str]) -> pd.Series:
    out = {}
    for j, name in enumerate(names):
        yj = X[:, j]
        others = np.delete(X, j, axis=1)
        if others.shape[1] == 0:
            out[name] = 1.0
            continue
        lr = LinearRegression().fit(others, yj)
        r2 = lr.score(others, yj)
        if r2 >= 1.0 - 1e-12:
            out[name] = np.inf
        else:
            out[name] = float(1.0 / (1.0 - r2))
    return pd.Series(out)


def scale_design(df: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, list[str]]:
    cols = []
    for name in names:
        x = df[name].to_numpy(dtype=float)
        if name in CONTINUOUS:
            sd = x.std(ddof=0)
            if sd == 0:
                raise ValueError(f"{name} has zero SD")
            x = (x - x.mean()) / sd
        cols.append(x)
    return np.column_stack(cols), names


def logit_or_table(df: pd.DataFrame, names: list[str], y: np.ndarray, n_boot: int) -> pd.DataFrame:
    X, names = scale_design(df, names)
    beta, cov, ok = fit_logit(X, y)
    if not ok:
        print("WARNING: full-sample logit did not report a clean Newton stop")
    se = np.sqrt(np.clip(np.diag(cov)[1:], 0, None))
    z = beta[1:] / np.where(se == 0, np.nan, se)
    p_wald = 2.0 * stats.norm.sf(np.abs(z))
    or_adj = np.exp(beta[1:])
    wald_lo = np.exp(beta[1:] - 1.96 * se)
    wald_hi = np.exp(beta[1:] + 1.96 * se)

    uni_or = []
    for j in range(X.shape[1]):
        b, _, _ = fit_logit(X[:, [j]], y)
        uni_or.append(float(np.exp(b[1])))

    rng = np.random.default_rng(SEED)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    boots = []
    n_fail = 0
    for _ in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True),
                rng.choice(neg, size=neg.size, replace=True),
            ]
        )
        b, _, fit_ok = fit_logit(X[idx], y[idx])
        if (not fit_ok) or (not np.all(np.isfinite(b))):
            n_fail += 1
            continue
        boots.append(np.exp(b[1:]))
    boots = np.asarray(boots)
    print(f"Table 4b bootstrap: {len(boots)}/{n_boot} replicates kept ({n_fail} failed)")
    boot_lo = np.percentile(boots, 2.5, axis=0)
    boot_hi = np.percentile(boots, 97.5, axis=0)

    vif = vif_series(X, names)
    rows = []
    for j, name in enumerate(names):
        v = vif[name]
        rows.append(
            {
                "Feature": name,
                "Type": "continuous (per 1 SD)" if name in CONTINUOUS else "binary",
                "VIF": v,
                "Univariate OR": uni_or[j],
                "Adjusted OR": float(or_adj[j]),
                "Wald SE (log-OR)": float(se[j]),
                "Wald z": float(z[j]),
                "Wald p": float(p_wald[j]),
                "Wald CI low": float(wald_lo[j]),
                "Wald CI high": float(wald_hi[j]),
                "Boot CI low": float(boot_lo[j]),
                "Boot CI high": float(boot_hi[j]),
                "n_boot_ok": int(len(boots)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B3
# ---------------------------------------------------------------------------


def _metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    return (
        float(average_precision_score(y, p)),
        float(roc_auc_score(y, p)),
        float(brier_score_loss(y, p)),
    )


def run_b3() -> dict:
    oof = pd.read_csv(OOF_PATH)
    y = oof["y"].to_numpy(dtype=int)
    assert y.size == 5185 and int(y.sum()) == 92

    point = {}
    for label, col in OOF_MODELS:
        point[label] = dict(zip(["pr_auc", "roc_auc", "brier"], _metrics(y, oof[col].to_numpy(dtype=float))))

    rng = np.random.default_rng(SEED)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    boot = {label: {"pr_auc": [], "roc_auc": [], "brier": []} for label, _ in OOF_MODELS}
    d_th_lgb = []
    d_loc_lgb = []
    p_th = oof["tabpfn_prob"].to_numpy(dtype=float)
    p_lgb = oof["lightgbm_prob"].to_numpy(dtype=float)
    p_loc = oof["tabpfn_(local)_prob"].to_numpy(dtype=float)

    for _ in range(N_BOOT):
        idx = np.concatenate(
            [
                rng.choice(pos, size=pos.size, replace=True),
                rng.choice(neg, size=neg.size, replace=True),
            ]
        )
        yb = y[idx]
        for label, col in OOF_MODELS:
            pr, roc, br = _metrics(yb, oof[col].to_numpy(dtype=float)[idx])
            boot[label]["pr_auc"].append(pr)
            boot[label]["roc_auc"].append(roc)
            boot[label]["brier"].append(br)
        d_th_lgb.append(average_precision_score(yb, p_th[idx]) - average_precision_score(yb, p_lgb[idx]))
        d_loc_lgb.append(average_precision_score(yb, p_loc[idx]) - average_precision_score(yb, p_lgb[idx]))

    def ci(arr):
        a = np.asarray(arr, dtype=float)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    rows = []
    for label, _ in OOF_MODELS:
        pr_lo, pr_hi = ci(boot[label]["pr_auc"])
        roc_lo, roc_hi = ci(boot[label]["roc_auc"])
        br_lo, br_hi = ci(boot[label]["brier"])
        rows.append(
            {
                "model": label,
                "pr_auc": point[label]["pr_auc"],
                "pr_auc_ci_low": pr_lo,
                "pr_auc_ci_high": pr_hi,
                "roc_auc": point[label]["roc_auc"],
                "roc_auc_ci_low": roc_lo,
                "roc_auc_ci_high": roc_hi,
                "brier": point[label]["brier"],
                "brier_ci_low": br_lo,
                "brier_ci_high": br_hi,
                "n_boot": N_BOOT,
                "seed": SEED,
            }
        )
    ci_df = pd.DataFrame(rows)

    def paired(name, deltas, point_delta):
        d = np.asarray(deltas, dtype=float)
        lo, hi = ci(d)
        p_le0 = float(np.mean(d <= 0))
        p_ge0 = float(np.mean(d >= 0))
        p_two = float(min(1.0, 2.0 * min(p_le0, p_ge0)))
        return {
            "contrast": name,
            "delta_pr_auc": point_delta,
            "ci_low": lo,
            "ci_high": hi,
            "p_bootstrap_one_sided_le0": p_le0,
            "p_bootstrap_two_sided": p_two,
            "n_boot": N_BOOT,
            "seed": SEED,
        }

    point_th = point["TabPFN (thinking-high)"]["pr_auc"] - point["LightGBM"]["pr_auc"]
    point_loc = point["TabPFN (local)"]["pr_auc"] - point["LightGBM"]["pr_auc"]
    paired_df = pd.DataFrame(
        [
            paired("TabPFN (thinking-high) − LightGBM", d_th_lgb, point_th),
            paired("TabPFN (local) − LightGBM", d_loc_lgb, point_loc),
        ]
    )

    # Fold wins from the OOF fold column (not bootstrap).
    fold_rows = []
    for fold, g in oof.groupby("fold"):
        yy = g["y"].to_numpy(dtype=int)
        fold_rows.append(
            {
                "fold": int(fold),
                "n": int(len(g)),
                "n_pos": int(yy.sum()),
                "tabpfn_thinking_high_pr_auc": average_precision_score(yy, g["tabpfn_prob"]),
                "lightgbm_pr_auc": average_precision_score(yy, g["lightgbm_prob"]),
                "tabpfn_local_pr_auc": average_precision_score(yy, g["tabpfn_(local)_prob"]),
            }
        )
    fold_df = pd.DataFrame(fold_rows).sort_values("fold")
    fold_df["thinking_high_gt_lgb"] = fold_df["tabpfn_thinking_high_pr_auc"] > fold_df["lightgbm_pr_auc"]
    fold_df["local_gt_lgb"] = fold_df["tabpfn_local_pr_auc"] > fold_df["lightgbm_pr_auc"]

    _write_df(ci_df, "paper_table_s_bootstrap_ci.csv", P4_DIRS)
    _write_df(paired_df, "paper_table_s_paired_delta.csv", P4_DIRS)
    _write_df(fold_df, "paper_table_s_fold_pr_wins.csv", P4_DIRS)
    P4_TABLES.mkdir(parents=True, exist_ok=True)
    ci_df.to_csv(P4_TABLES / "bootstrap_ci.csv", index=False)
    paired_df.to_csv(P4_TABLES / "paired_pr_auc_delta.csv", index=False)

    disp = pd.DataFrame(
        {
            "Model": ci_df["model"],
            "PR-AUC (95% CI)": [
                f"{r.pr_auc:.4f} {fmt_ci(r.pr_auc_ci_low, r.pr_auc_ci_high)}"
                for r in ci_df.itertuples()
            ],
            "ROC-AUC (95% CI)": [
                f"{r.roc_auc:.4f} {fmt_ci(r.roc_auc_ci_low, r.roc_auc_ci_high)}"
                for r in ci_df.itertuples()
            ],
            "Brier (95% CI)": [
                f"{r.brier:.4f} {fmt_ci(r.brier_ci_low, r.brier_ci_high)}"
                for r in ci_df.itertuples()
            ],
        }
    )
    save_table_png(
        disp,
        title="Nested-CV OOF ranking with stratified bootstrap 95% CIs (n_boot=2000)",
        filename="paper_table_s_bootstrap_ci.png",
        dirs=P4_DIRS,
        figsize=(14.2, 4.6),
        fontsize=8,
    )

    pdisp = pd.DataFrame(
        {
            "Contrast": paired_df["contrast"],
            "Δ PR-AUC": [f"{v:.4f}" for v in paired_df["delta_pr_auc"]],
            "95% CI": [fmt_ci(r.ci_low, r.ci_high) for r in paired_df.itertuples()],
            "P(Δ ≤ 0)": [f"{v:.4f}" for v in paired_df["p_bootstrap_one_sided_le0"]],
            "Two-sided p": [f"{v:.4f}" for v in paired_df["p_bootstrap_two_sided"]],
        }
    )
    save_table_png(
        pdisp,
        title="Paired bootstrap Δ PR-AUC vs LightGBM (same resampled OOF rows)",
        filename="paper_table_s_paired_delta.png",
        dirs=P4_DIRS,
        figsize=(12.4, 2.6),
        fontsize=8,
    )

    fdisp = pd.DataFrame(
        {
            "Fold": fold_df["fold"].astype(int),
            "Events": fold_df["n_pos"].astype(int),
            "Thinking-high": fold_df["tabpfn_thinking_high_pr_auc"].map(lambda v: f"{v:.4f}"),
            "LightGBM": fold_df["lightgbm_pr_auc"].map(lambda v: f"{v:.4f}"),
            "Local": fold_df["tabpfn_local_pr_auc"].map(lambda v: f"{v:.4f}"),
            "TH > LGB": fold_df["thinking_high_gt_lgb"].map(lambda v: "yes" if v else "no"),
            "Local > LGB": fold_df["local_gt_lgb"].map(lambda v: "yes" if v else "no"),
        }
    )
    save_table_png(
        fdisp,
        title="Outer-fold PR-AUC (thinking-high vs LightGBM vs local)",
        filename="paper_table_s_fold_pr_wins.png",
        dirs=P4_DIRS,
        figsize=(11.2, 3.4),
        fontsize=8,
    )

    print("B3 thinking-high − LightGBM Δ PR-AUC", point_th, "P(Δ≤0)", paired_df.iloc[0]["p_bootstrap_one_sided_le0"])
    print("B3 local − LightGBM Δ PR-AUC", point_loc, "P(Δ≤0)", paired_df.iloc[1]["p_bootstrap_one_sided_le0"])
    print("B3 fold wins thinking-high", int(fold_df["thinking_high_gt_lgb"].sum()), "/5")
    print("B3 fold wins local", int(fold_df["local_gt_lgb"].sum()), "/5")
    return {"ci": ci_df, "paired": paired_df, "folds": fold_df}


# ---------------------------------------------------------------------------
# B4
# ---------------------------------------------------------------------------


def run_b4() -> dict:
    raw = pd.read_csv(VLST_PATH)
    y = raw[TARGET].to_numpy(dtype=float)
    assert int(y.sum()) == 92

    X_full, names_full = scale_design(raw, TABLE4_FULL)
    vif_full = vif_series(X_full, names_full)
    vif_full_df = vif_full.rename("VIF").reset_index().rename(columns={"index": "Feature"})
    vif_full_df["spec"] = "Table 4 (17 covariates, unidentified)"

    or_df = logit_or_table(raw, TABLE4B, y, N_BOOT)
    or_df["EPV"] = 92 / len(TABLE4B)
    or_df["n_covariates"] = len(TABLE4B)
    or_df["n_events"] = 92
    or_df["n_boot"] = N_BOOT
    or_df["seed"] = SEED

    vif_b = or_df[["Feature", "VIF"]].copy()
    vif_b["spec"] = "Table 4b (13 covariates, one per block)"
    vif_cmp = pd.concat([vif_full_df, vif_b], ignore_index=True)

    _write_df(or_df, "paper_table4b_reduced_or.csv", EDA_DIRS)
    _write_df(vif_cmp, "paper_table4b_vif_comparison.csv", EDA_DIRS)

    def vif_cell(v: float) -> str:
        if not np.isfinite(v) or v > 100:
            return "∞"
        return f"{v:.2f}"

    disp = pd.DataFrame(
        {
            "Feature": or_df["Feature"],
            "Type": or_df["Type"],
            "VIF": [vif_cell(v) for v in or_df["VIF"]],
            "Uni. OR": or_df["Univariate OR"].map(lambda v: f"{v:.3f}"),
            "Adj. OR": or_df["Adjusted OR"].map(lambda v: f"{v:.3f}"),
            "Wald 95% CI": [
                fmt_ci(lo, hi, nd=3)
                for lo, hi in zip(or_df["Wald CI low"], or_df["Wald CI high"])
            ],
            "Boot 95% CI": [
                fmt_ci(lo, hi, nd=3)
                for lo, hi in zip(or_df["Boot CI low"], or_df["Boot CI high"])
            ],
        }
    )
    save_table_png(
        disp,
        title="Table 4b. Reduced unweighted logit (one representative per collinear block; EPV=92/13≈7.1)",
        filename="paper_table4b_reduced_or.png",
        dirs=EDA_DIRS,
        figsize=(14.8, 6.4),
        fontsize=7,
    )

    vmax = vif_cmp.copy()
    vmax["VIF_display"] = vmax["VIF"].map(vif_cell)
    vdisp = vmax.pivot(index="Feature", columns="spec", values="VIF_display").reset_index()
    # Keep Table 4 column first.
    cols = ["Feature"] + [c for c in vdisp.columns if c != "Feature"]
    vdisp = vdisp[cols]
    save_table_png(
        vdisp,
        title="VIF: unidentified 17-covariate Table 4 vs reduced Table 4b",
        filename="paper_table4b_vif_comparison.png",
        dirs=EDA_DIRS,
        figsize=(12.8, 7.2),
        fontsize=7,
    )
    print("B4 EPV", 92 / len(TABLE4B))
    print(or_df[["Feature", "VIF", "Adjusted OR", "Wald CI low", "Wald CI high"]].to_string(index=False))
    return {"or": or_df, "vif": vif_cmp}


# ---------------------------------------------------------------------------
# B7
# ---------------------------------------------------------------------------

CLINICAL_CONTINUOUS = [
    ("Age", "Age, years"),
    ("LVEF", "LVEF, %"),
    ("LV", "LV (unnamed; not in Wang Table 1)"),
    ("WBC", "WBC, 10^9/L"),
    ("HGB", "Haemoglobin"),
    ("Platelet", "Platelet count"),
    ("Cre", "Creatinine"),
    ("eGFR", "eGFR"),
    ("CaI", "CaI (unnamed; not in Wang Table 1)"),
    ("TCL", "Total cholesterol, mmol/L"),
    ("LDL", "LDL, mmol/L"),
    ("HDL", "HDL, mmol/L"),
    ("TG", "Triglycerides, mmol/L"),
    ("Fast-Glu", "Fasting glucose"),
    ("HbA1c", "HbA1c, %"),
    ("Fiberinogen", "Fibrinogen, g/L"),
    ("NO.of vessels", "Number of vessels"),
    ("No.of stents per lesion", "Stents per lesion"),
    ("Total stent length", "Total stent length, mm"),
    ("Min-stent diameter", "Min. stent diameter, mm"),
    ("Stent release pressure", "Stent release pressure, atm"),
]

CLINICAL_BINARY = [
    ("Men", "Men"),
    ("Diabetes", "Diabetes"),
    ("Hypertension", "Hypertension"),
    ("HL", "Dyslipidaemia (HL)"),
    ("Current smoker", "Current smoker"),
    ("Current drinking", "Current drinking"),
    ("Stroke/TIA", "Stroke / TIA"),
    ("Previous PCI", "Previous PCI"),
    ("Previous MI", "Previous MI"),
    ("Previous CABG", "Previous CABG"),
    ("History of HF", "History of HF"),
    ("Chronic renal insufficiency", "Chronic renal insufficiency"),
    ("History of peripheral vascualr disease", "History of PVD"),
    ("Initial diagnosis-AMI", "Admitting diagnosis AMI"),
    ("STEMI", "STEMI"),
    ("NSTEMI", "NSTEMI"),
    ("UA", "Unstable angina"),
    ("Cardiogenic shock", "Cardiogenic shock"),
    ("3-vessel disease", "3-vessel disease"),
    ("Multi-vessel CAD", "Multi-vessel CAD"),
    ("PES", "SES (PES column; matches Wang Table 1 SES)"),
    ("1.1:1Post dilation", "1.1:1 post-dilation (CSV as stored)"),
    ("No postdilation", "No postdilation (CSV as stored)"),
    ("CKD90", "eGFR < 90 (CKD90)"),
    ("Aspirin", "Aspirin during follow-up (not index PCI)"),
    ("Clopidogrel", "Clopidogrel during follow-up (not index PCI)"),
    ("Ticagrelor", "Ticagrelor during follow-up (not index PCI)"),
    ("DAPT", "DAPT during follow-up (not index PCI)"),
]

# Compact paper-facing subset (Wang-comparable + LV / CaI + CSV post-dilation).
PAPER_ROWS = {
    "Age, years",
    "Men",
    "Diabetes",
    "Hypertension",
    "Dyslipidaemia (HL)",
    "Current smoker",
    "Previous PCI",
    "Previous MI",
    "Admitting diagnosis AMI",
    "3-vessel disease",
    "LVEF, %",
    "LV (unnamed; not in Wang Table 1)",
    "WBC, 10^9/L",
    "Creatinine",
    "eGFR",
    "eGFR < 90 (CKD90)",
    "CaI (unnamed; not in Wang Table 1)",
    "Fibrinogen, g/L",
    "Stents per lesion",
    "Total stent length, mm",
    "SES (PES column; matches Wang Table 1 SES)",
    "1.1:1 post-dilation (CSV as stored)",
    "No postdilation (CSV as stored)",
    "DAPT during follow-up (not index PCI)",
}


def _choose_cont_test(a: np.ndarray, b: np.ndarray) -> tuple[str, float]:
    x = np.concatenate([a, b])
    sk = float(stats.skew(x, bias=False))
    kt = float(stats.kurtosis(x, fisher=True, bias=False))
    if abs(sk) <= 1.0 and kt <= 3.0:
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return "Welch t", float(p)
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return "Mann–Whitney U", float(p)


def _binary_p(x: np.ndarray, y: np.ndarray) -> tuple[str, float]:
    tab = pd.crosstab(x, y)
    tab = tab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    chi2, p_chi, _, expected = stats.chi2_contingency(tab.values, correction=False)
    if (expected < 5).any():
        _, p_f = stats.fisher_exact(tab.values)
        return "Fisher", float(p_f)
    return "Chi-square", float(p_chi)


def run_b7() -> pd.DataFrame:
    raw = pd.read_csv(VLST_PATH)
    y = raw[TARGET].to_numpy(dtype=int)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    assert n0 == 5093 and n1 == 92

    rows = []
    for col, label in CLINICAL_CONTINUOUS:
        a = raw.loc[y == 0, col].to_numpy(dtype=float)
        b = raw.loc[y == 1, col].to_numpy(dtype=float)
        test, p = _choose_cont_test(a, b)
        rows.append(
            {
                "section": "continuous",
                "variable": col,
                "label": label,
                "non_vlst": f"{a.mean():.2f} ({a.std(ddof=1):.2f})",
                "vlst": f"{b.mean():.2f} ({b.std(ddof=1):.2f})",
                "non_vlst_mean": float(a.mean()),
                "non_vlst_sd": float(a.std(ddof=1)),
                "vlst_mean": float(b.mean()),
                "vlst_sd": float(b.std(ddof=1)),
                "test": test,
                "p": p,
                "in_paper_table": label in PAPER_ROWS,
            }
        )
    for col, label in CLINICAL_BINARY:
        x = raw[col].to_numpy(dtype=float)
        a = int(((y == 0) & (x == 1)).sum())
        b = int(((y == 1) & (x == 1)).sum())
        test, p = _binary_p(x, y)
        rows.append(
            {
                "section": "binary",
                "variable": col,
                "label": label,
                "non_vlst": f"{a} ({100 * a / n0:.2f}%)",
                "vlst": f"{b} ({100 * b / n1:.2f}%)",
                "non_vlst_n": a,
                "non_vlst_pct": 100 * a / n0,
                "vlst_n": b,
                "vlst_pct": 100 * b / n1,
                "test": test,
                "p": p,
                "in_paper_table": label in PAPER_ROWS,
            }
        )
    full = pd.DataFrame(rows)
    full["p_fmt"] = full["p"].map(fmt_p)
    _write_df(full, "paper_table_c_cohort_characteristics.csv", EDA_DIRS)

    paper = full.loc[full["in_paper_table"]].copy()
    # Preserve PAPER_ROWS order.
    order = {lab: i for i, lab in enumerate(PAPER_ROWS)}
    # PAPER_ROWS is a set; use list order from the two CLINICAL lists.
    paper_order = [lab for _, lab in CLINICAL_CONTINUOUS + CLINICAL_BINARY if lab in PAPER_ROWS]
    paper["_ord"] = paper["label"].map({lab: i for i, lab in enumerate(paper_order)})
    paper = paper.sort_values("_ord")

    disp = pd.DataFrame(
        {
            "Variable": paper["label"],
            f"No VLST (n={n0})": paper["non_vlst"],
            f"VLST (n={n1})": paper["vlst"],
            "Test": paper["test"],
            "p": paper["p_fmt"],
        }
    )
    save_table_png(
        disp,
        title="Cohort characteristics from VLST.csv (Wang 2020 derivation: 6,038 → 5,185)",
        filename="paper_table_c_cohort_characteristics.png",
        dirs=EDA_DIRS,
        figsize=(14.6, 10.8),
        fontsize=7,
    )
    print("B7 post-dilation VLST on 1.1:1:", int(raw.loc[y == 1, "1.1:1Post dilation"].sum()))
    print("B7 post-dilation VLST on No postdilation:", int(raw.loc[y == 1, "No postdilation"].sum()))
    print("B7 WBC means", raw.loc[y == 0, "WBC"].mean(), raw.loc[y == 1, "WBC"].mean())
    print("B7 HL", int(((y == 0) & (raw["HL"] == 1)).sum()), int(((y == 1) & (raw["HL"] == 1)).sum()))
    print("B7 PES/SES", int(((y == 0) & (raw["PES"] == 1)).sum()), int(((y == 1) & (raw["PES"] == 1)).sum()))
    return full


def main() -> None:
    print("== B3 ==")
    run_b3()
    print("== B7 ==")
    run_b7()
    print("== B4 ==")
    run_b4()
    print("done")


if __name__ == "__main__":
    main()
