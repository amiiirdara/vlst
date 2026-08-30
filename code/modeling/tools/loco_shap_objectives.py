#!/usr/bin/env python3
"""
Multi-objective LOCO & SHAP feature importance for TabPFN.

Runs Leave-One-Covariate-Out (LOCO) and coalition-metric SHAP under three
objectives — pr_auc, f1, f2 — mirroring the FFS scoring loop in
code/modeling/rating/tabpfn_playground.ipynb §10.

Outputs ranked feature lists, overlap analysis, optional subset evaluation,
figures, and a markdown report under data/result/loco_shap/ (override with --out).

Requires: tabpfn, torch (GPU recommended). LOCO is O(n_features) refits;
SHAP is O(n_perm × |universe|) forward passes on a fixed context model.

Example:
    python code/modeling/tools/loco_shap_objectives.py --topk 20
    python code/modeling/tools/loco_shap_objectives.py --report-only --out data/result/loco_shap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[3]
RAW_PATH = ROOT / "data" / "raw" / "VLST.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUT = ROOT / "data" / "result" / "loco_shap"

TARGET_COL = "Stent thrombosis"
DROP_FEATURES = ["Time since stent implantation"]
ID_COLS = ["NO.", "Name"]
RANDOM_STATE = 42
TEST_SIZE = 0.30

SCORINGS = ("pr_auc", "f1", "f2")
SCORING_COLORS = {"pr_auc": "#2c7fb8", "f1": "#41ab5d", "f2": "#d95f0e"}


# ---------------------------------------------------------------------------
# Data & scoring (aligned with tabpfn_playground.ipynb)
# ---------------------------------------------------------------------------


def load_raw() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(RAW_PATH)
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    y = df[TARGET_COL].astype(int).to_numpy()
    drop = [TARGET_COL] + [c for c in DROP_FEATURES if c in df.columns]
    x_df = df.drop(columns=drop)
    for c in x_df.columns:
        if x_df[c].dtype == object:
            coerced = pd.to_numeric(x_df[c].astype(str).str.strip(), errors="coerce")
            if coerced.notna().mean() >= 0.5:
                x_df[c] = coerced
            else:
                codes = x_df[c].astype("category").cat.codes.astype(float)
                x_df[c] = codes.where(codes >= 0, np.nan)
    return x_df.to_numpy(dtype=float), y, list(x_df.columns)


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    strategy: str = "f2",
    beta: float = 2.0,
    grid_points: int = 199,
) -> tuple[float, dict[str, float]]:
    grid = np.linspace(0.01, 0.99, grid_points)
    best_i, best_val = 0, -np.inf
    info: dict[str, float] = {}
    for i, t in enumerate(grid):
        pred = (scores >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        fb = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        if strategy == "f1":
            val = f1
        elif strategy in ("f2", "fbeta"):
            val = fb
        else:
            raise ValueError(strategy)
        if val > best_val:
            best_i, best_val = i, val
            info = {"precision": float(p), "recall": float(r), "f1": float(f1), f"f{beta:g}": float(fb)}
    return float(grid[best_i]), info


def metric_score(y_true: np.ndarray, probas: np.ndarray, metric: str) -> float:
    """Test-set score under a named objective (threshold tuned on same y/probas for f1/f2)."""
    if metric == "pr_auc":
        return float(average_precision_score(y_true, probas))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, probas))
    if metric == "f1":
        t, _ = select_threshold(y_true, probas, strategy="f1", beta=1.0)
        pred = (probas >= t).astype(int)
        return float(f1_score(y_true, pred, zero_division=0))
    if metric == "f2":
        t, _ = select_threshold(y_true, probas, strategy="f2", beta=2.0)
        pred = (probas >= t).astype(int)
        return float(fbeta_score(y_true, pred, beta=2.0, zero_division=0))
    raise ValueError(f"Unknown metric: {metric}")


def pos_proba(clf, x: np.ndarray) -> np.ndarray:
    classes = list(clf.classes_)
    idx = classes.index(1) if 1 in classes else 1
    return np.asarray(clf.predict_proba(x)[:, idx], dtype=float)


def make_tabpfn(seed: int = RANDOM_STATE, device: str = "auto"):
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier(
        device=device,
        n_estimators=8,
        balance_probabilities=True,
        ignore_pretraining_limits=True,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# LOCO — refit without each feature; importance = baseline − score_without
# ---------------------------------------------------------------------------


def loco_importance(
    make_clf,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    names: list[str],
    *,
    metric: str = "pr_auc",
    max_features: int | None = None,
) -> pd.DataFrame:
    base = make_clf()
    base.fit(x_tr, y_tr)
    baseline = metric_score(y_te, pos_proba(base, x_te), metric)
    print(f"  LOCO[{metric}] baseline={baseline:.4f} (all {x_tr.shape[1]} features)")

    n = x_tr.shape[1]
    order = list(range(n))
    if max_features is not None:
        order = order[: int(max_features)]
        print(f"  LOCO capped to first {len(order)} of {n} features")

    rows = []
    for rank, j in enumerate(order, 1):
        keep = [k for k in range(n) if k != j]
        m = make_clf()
        m.fit(x_tr[:, keep], y_tr)
        s = metric_score(y_te, pos_proba(m, x_te[:, keep]), metric)
        rows.append(
            {
                "feature": names[j],
                "importance": baseline - s,
                "score_without": s,
                "baseline": baseline,
                "metric": metric,
            }
        )
        if rank % 10 == 0 or rank == len(order):
            print(f"    LOCO[{metric}] {rank}/{len(order)}")

    return (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda d: np.arange(1, len(d) + 1))
    )


# ---------------------------------------------------------------------------
# SHAP — coalition Shapley on a global metric (objective-aligned ranking)
# ---------------------------------------------------------------------------


def _coalition_probas(
    clf,
    x_bg_row: np.ndarray,
    x_exp: np.ndarray,
    feat_subset: np.ndarray,
    revealed: set[int],
) -> np.ndarray:
    """Mask features in feat_subset \\ revealed to background; keep others at true values."""
    cur = x_exp.copy()
    for pos, f in enumerate(feat_subset):
        if pos not in revealed:
            cur[:, f] = x_bg_row[f]
    return pos_proba(clf, cur)


def coalition_metric_shap(
    clf,
    x_bg: np.ndarray,
    x_exp: np.ndarray,
    y_exp: np.ndarray,
    feat_subset: list[int],
    *,
    metric: str,
    n_perm: int = 40,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Shapley values for a *global* metric on the explained cohort.

    Unlike probability SHAP (local, threshold-free), this ranks features by how
    much each one moves pr_auc / f1 / f2 when revealed vs background-masked.
    """
    rng = np.random.default_rng(seed)
    k = len(feat_subset)
    feat_arr = np.asarray(feat_subset)
    phi = np.zeros(k, dtype=float)

    def v(revealed: set[int]) -> float:
        # Average metric over a few background rows for stability
        vals = []
        for _ in range(min(3, len(x_bg))):
            bg = x_bg[rng.integers(0, len(x_bg))]
            probas = _coalition_probas(clf, bg, x_exp, feat_arr, revealed)
            vals.append(metric_score(y_exp, probas, metric))
        return float(np.mean(vals))

    for p in range(n_perm):
        perm = rng.permutation(k)
        revealed: set[int] = set()
        prev = v(revealed)
        for pos in perm:
            revealed.add(int(pos))
            now = v(revealed)
            phi[pos] += now - prev
            prev = now
        if (p + 1) % 10 == 0 or p + 1 == n_perm:
            print(f"    SHAP[{metric}] permutation {p + 1}/{n_perm}")

    return pd.DataFrame(
        {
            "feature_index": feat_arr,
            "shap_value": phi / n_perm,
            "mean_abs_shap": np.abs(phi / n_perm),
            "metric": metric,
        }
    ).sort_values("mean_abs_shap", ascending=False)


def shap_ranking(
    clf,
    loco_df: pd.DataFrame,
    feature_names: list[str],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    metric: str,
    topk: int,
    universe: int,
    n_perm: int,
    n_instances: int,
    n_background: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    universe_names = loco_df.head(min(universe, len(loco_df)))["feature"].tolist()
    feat_idx = [feature_names.index(f) for f in universe_names]

    rng = np.random.default_rng(seed)
    pos_rows = np.flatnonzero(y_test == 1)
    neg_rows = np.flatnonzero(y_test == 0)
    n_neg = max(0, n_instances - len(pos_rows))
    sel = np.concatenate(
        [pos_rows, rng.choice(neg_rows, size=min(n_neg, len(neg_rows)), replace=False)]
    )[:n_instances]
    x_exp = x_test[sel]
    y_exp = y_test[sel]
    bg_idx = rng.choice(len(x_train), size=min(n_background, len(x_train)), replace=False)
    x_bg = x_train[bg_idx]

    print(f"  SHAP[{metric}] universe={len(feat_idx)} instances={len(sel)} perm={n_perm}")
    shap_df = coalition_metric_shap(
        clf, x_bg, x_exp, y_exp, feat_idx, metric=metric, n_perm=n_perm, seed=seed
    )
    shap_df["feature"] = shap_df["feature_index"].map(lambda i: feature_names[int(i)])
    shap_df = shap_df.reset_index(drop=True)
    shap_df.insert(0, "rank", np.arange(1, len(shap_df) + 1))
    top_names = shap_df.head(topk)["feature"].tolist()
    return top_names, shap_df


# ---------------------------------------------------------------------------
# Subset evaluation (optional refit per top-K list)
# ---------------------------------------------------------------------------


def oof_pos_proba(make_clf, x: np.ndarray, y: np.ndarray, n_splits: int = 10) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in skf.split(x, y):
        m = make_clf()
        m.fit(x[tr], y[tr])
        oof[va] = pos_proba(m, x[va])
    return oof


def evaluate_feature_subset(
    make_clf,
    name: str,
    feat_names: list[str],
    feature_names: list[str],
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    feat_idx = [feature_names.index(f) for f in feat_names]
    xtr = x_train[:, feat_idx]
    xte = x_test[:, feat_idx]

    oof = oof_pos_proba(make_clf, xtr, y_train)
    thr, _ = select_threshold(y_train, oof, strategy="f2", beta=2.0)
    m = make_clf()
    m.fit(xtr, y_train)
    p_test = pos_proba(m, xte)
    pred = (p_test >= thr).astype(int)
    return {
        "variant": name,
        "n_features": len(feat_names),
        "threshold": round(thr, 4),
        "test_pr_auc": round(float(average_precision_score(y_test, p_test)), 4),
        "test_f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "test_f2": round(float(fbeta_score(y_test, pred, beta=2.0, zero_division=0)), 4),
        "objective": name.split("_")[1] if "_" in name else "all",
        "method": name.split("_")[0] if "_" in name else name,
    }


# ---------------------------------------------------------------------------
# Report & figures
# ---------------------------------------------------------------------------


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def overlap_matrix(
    loco_topk: dict[str, list[str]],
    shap_topk: dict[str, list[str]],
) -> pd.DataFrame:
    labels = []
    sets: dict[str, set[str]] = {}
    for metric in SCORINGS:
        if loco_topk.get(metric):
            key = f"LOCO_{metric}"
            labels.append(key)
            sets[key] = set(loco_topk[metric])
        if shap_topk.get(metric):
            key = f"SHAP_{metric}"
            labels.append(key)
            sets[key] = set(shap_topk[metric])

    mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
    for a in labels:
        for b in labels:
            mat.loc[a, b] = jaccard(sets[a], sets[b])
    return mat


def plot_loco_panels(loco_dfs: dict[str, pd.DataFrame], out_dir: Path, top_n: int = 12) -> None:
    metrics = [m for m in SCORINGS if m in loco_dfs]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        df = loco_dfs[metric].head(top_n).iloc[::-1]
        colors = ["#2c7fb8" if v >= 0 else "#d95f0e" for v in df["importance"]]
        ax.barh(df["feature"], df["importance"], color=colors, alpha=0.9)
        ax.set_xlabel(f"LOCO importance (Δ {metric})")
        ax.set_title(f"LOCO — {metric} (top {len(df)})")
    plt.suptitle("Leave-one-covariate-out importance by objective")
    plt.tight_layout()
    plt.savefig(out_dir / "loco_by_objective.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_overlap_heatmap(mat: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat.values.astype(float), vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Jaccard similarity")
    ax.set_title("Top-K feature-set overlap (LOCO vs SHAP × objective)")
    plt.tight_layout()
    plt.savefig(out_dir / "overlap_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def write_report(
    out_dir: Path,
    *,
    loco_dfs: dict[str, pd.DataFrame],
    shap_dfs: dict[str, pd.DataFrame],
    loco_topk: dict[str, list[str]],
    shap_topk: dict[str, list[str]],
    eval_df: pd.DataFrame | None,
    topk: int,
    meta: dict,
) -> Path:
    report_path = out_dir / "report.md"
    overlap = overlap_matrix(loco_topk, shap_topk)

    lines = [
        "# LOCO & SHAP multi-objective feature importance report",
        "",
        f"Generated: {meta.get('timestamp', '—')}  ",
        f"TabPFN context: train {meta.get('n_train', '—')} rows, test {meta.get('n_test', '—')} rows, "
        f"{meta.get('n_features', '—')} features  ",
        f"Top-K = {topk} per method × objective (`pr_auc`, `f1`, `f2`)",
        "",
        "## Executive summary",
        "",
    ]

    for metric in SCORINGS:
        if metric not in loco_dfs:
            continue
        top3 = loco_dfs[metric].head(3)["feature"].tolist()
        base = loco_dfs[metric]["baseline"].iloc[0]
        lines.append(
            f"- **LOCO ({metric})** — baseline test {metric} = **{base:.3f}**. "
            f"Top drops: {', '.join(f'`{f}`' for f in top3)}."
        )

    lines.extend(["", "## Methods", ""])
    lines.extend(
        [
            "| Method | Objective | What is measured | Cost |",
            "|--------|-----------|------------------|------|",
            "| **LOCO** | pr_auc / f1 / f2 | Drop in test metric when one column is removed and TabPFN is **refit** | O(d) fits |",
            "| **SHAP** | pr_auc / f1 / f2 | Coalition Shapley value of the **global** metric on explained test rows "
            "(mask to background; no refit) | O(n_perm × k) forward passes |",
            "| **FFS** (tabpfn_playground.ipynb §10) | pr_auc / f1 / f2 | Greedy forward add on holdout | O(K × d) fits |",
            "",
            "**Threshold policy:** For `f1` and `f2`, the metric is computed after tuning the decision "
            "threshold on the same scores used for evaluation (test set for LOCO; explained cohort for SHAP). "
            "This matches the FFS holdout logic in `tabpfn_playground.ipynb` but is optimistic if you iterate on test — "
            "use for exploratory ranking, not final model selection.",
            "",
            "## Top features by objective",
            "",
        ]
    )

    for metric in SCORINGS:
        lines.append(f"### {metric.upper()}")
        lines.append("")
        if metric in loco_dfs:
            lines.append("**LOCO (drop-one, refit):**")
            lines.append("")
            lines.append("| Rank | Feature | Importance | Score without |")
            lines.append("|------|---------|------------|---------------|")
            for _, row in loco_dfs[metric].head(topk).iterrows():
                lines.append(
                    f"| {int(row['rank'])} | {row['feature']} | {row['importance']:.4f} | "
                    f"{row['score_without']:.4f} |"
                )
            lines.append("")
        if metric in shap_dfs:
            lines.append("**SHAP (coalition metric, fixed context):**")
            lines.append("")
            lines.append("| Rank | Feature | Shapley value | |SHAP| |")
            lines.append("|------|---------|---------------|--------|")
            for _, row in shap_dfs[metric].head(topk).iterrows():
                lines.append(
                    f"| {int(row['rank'])} | {row['feature']} | {row['shap_value']:.4f} | "
                    f"{row['mean_abs_shap']:.4f} |"
                )
            lines.append("")

    lines.extend(["## Pairwise overlap (Jaccard, top-K sets)", ""])
    if len(overlap.index) > 1:
        lines.append("```")
        lines.append(overlap.round(2).to_string())
        lines.append("```")
    else:
        lines.append("_Run full pipeline to populate SHAP and multi-objective LOCO overlap._")
    lines.append("")

    # Cross-objective within method
    lines.append("### Same method, different objectives")
    lines.append("")
    for method, topk_dict in [("LOCO", loco_topk), ("SHAP", shap_topk)]:
        pairs = []
        for m1, m2 in combinations(SCORINGS, 2):
            if m1 in topk_dict and m2 in topk_dict:
                pairs.append(f"{m1}∩{m2}={len(set(topk_dict[m1]) & set(topk_dict[m2]))}/{topk}")
        if pairs:
            lines.append(f"- **{method}:** " + ", ".join(pairs))
    lines.append("")

    if eval_df is not None and len(eval_df):
        lines.extend(["## Subset refit evaluation (OOF F₂ threshold)", ""])
        lines.append("| Variant | n | test PR-AUC | test F1 | test F2 |")
        lines.append("|---------|---|-------------|---------|---------|")
        for _, row in eval_df.sort_values("test_pr_auc", ascending=False).iterrows():
            lines.append(
                f"| {row['variant']} | {int(row['n_features'])} | {row['test_pr_auc']:.4f} | "
                f"{row['test_f1']:.4f} | {row['test_f2']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## How to read this",
            "",
            "1. **LOCO** answers: *If I delete this column and retrain TabPFN, how much does metric X fall?* "
            "High bars = features the model cannot compensate for. Correlated twins may both look weak.",
            "2. **SHAP** answers: *When this feature is revealed vs a background patient, how much does metric X "
            "rise on the explained cohort?* Uses the §5 context model (no refit). Correlated features can share credit.",
            "3. **Objective matters:** `pr_auc` favours ranking power; `f2` favours recall-heavy operating points; "
            "`f1` balances precision and recall. The same feature can rank high under one objective and low under another.",
            "4. **Clinical top signals (pr_auc LOCO, typical run):** renal function (`eGFR`, `Cre`), "
            "angiographic thrombus/tortuosity/calcification, and left-main involvement — consistent with VLST biology.",
            "",
            "## Artifacts",
            "",
            f"- LOCO CSVs: `data/result/loco_shap/loco_{{metric}}.csv`",
            f"- SHAP CSVs: `data/result/loco_shap/shap_{{metric}}.csv`",
            f"- Overlap matrix: `data/result/loco_shap/overlap_matrix.csv`",
            f"- Figures: `data/result/loco_shap/loco_by_objective.png`, `overlap_heatmap.png`",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def load_legacy_pr_auc_loco(out_dir: Path) -> pd.DataFrame | None:
    """Import existing pr_auc LOCO from prior figure or TabPFN result dirs if present."""
    legacy = ROOT / "data" / "result" / "modeling_tabpfn" / "tabpfn_loco_importance.csv"
    if not legacy.is_file():
        legacy = ROOT / "data" / "figures" / "tabpfn_loco_importance.csv"
    if not legacy.is_file():
        return None
    df = pd.read_csv(legacy)
    if "baseline" not in df.columns and "score_without" in df.columns:
        df["baseline"] = df["score_without"] + df["importance"]
    df["metric"] = "pr_auc"
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(out_dir / "loco_pr_auc.csv", index=False)
    print(f"Imported legacy LOCO pr_auc from {legacy} ({len(df)} features)")
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        loco_dfs, shap_dfs, loco_topk, shap_topk = {}, {}, {}, {}
        for metric in SCORINGS:
            p = out_dir / f"loco_{metric}.csv"
            if p.is_file():
                loco_dfs[metric] = pd.read_csv(p)
                loco_topk[metric] = loco_dfs[metric].head(args.topk)["feature"].tolist()
            sp = out_dir / f"shap_{metric}.csv"
            if sp.is_file():
                shap_dfs[metric] = pd.read_csv(sp)
                shap_topk[metric] = shap_dfs[metric].head(args.topk)["feature"].tolist()
        if "pr_auc" not in loco_dfs:
            legacy = load_legacy_pr_auc_loco(out_dir)
            if legacy is not None:
                loco_dfs["pr_auc"] = legacy
                loco_topk["pr_auc"] = legacy.head(args.topk)["feature"].tolist()

        eval_path = out_dir / "subset_evaluation.csv"
        eval_df = pd.read_csv(eval_path) if eval_path.is_file() else None
        if loco_dfs:
            plot_loco_panels(loco_dfs, out_dir, top_n=min(12, args.topk))
        if loco_topk and shap_topk and any(shap_topk.values()):
            mat = overlap_matrix(loco_topk, shap_topk)
            if len(mat) > 1:
                mat.to_csv(out_dir / "overlap_matrix.csv")
                plot_overlap_heatmap(mat, out_dir)
        meta = json.loads((out_dir / "run_meta.json").read_text()) if (out_dir / "run_meta.json").is_file() else {}
        rp = write_report(
            out_dir,
            loco_dfs=loco_dfs,
            shap_dfs=shap_dfs,
            loco_topk=loco_topk,
            shap_topk=shap_topk,
            eval_df=eval_df,
            topk=args.topk,
            meta=meta,
        )
        print(f"Report: {rp}")
        return

    # --- Full run (needs tabpfn) ---
    if not RAW_PATH.is_file():
        sys.exit(f"Raw data not found: {RAW_PATH}")

    os.environ.setdefault("TABPFN_TOKEN", os.environ.get("TABPFN_TOKEN", ""))

    x_all, y_all, feature_names = load_raw()
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    print(f"Train {x_train.shape} | Test {x_test.shape} | Features {len(feature_names)}")

    make_clf = lambda seed=RANDOM_STATE: make_tabpfn(seed=seed, device=args.device)

    loco_dfs: dict[str, pd.DataFrame] = {}
    shap_dfs: dict[str, pd.DataFrame] = {}
    loco_topk: dict[str, list[str]] = {}
    shap_topk: dict[str, list[str]] = {}

    # Shared context model for SHAP (§5 style — fit once on all features)
    print("\nFitting baseline TabPFN context for SHAP ...")
    context_clf = make_clf()
    t0 = time.time()
    context_clf.fit(x_train, y_train)
    print(f"  Context stored in {time.time() - t0:.1f}s")

    for metric in args.scorings:
        print(f"\n=== LOCO objective: {metric} ===")
        loco_df = loco_importance(
            make_clf,
            x_train,
            y_train,
            x_test,
            y_test,
            feature_names,
            metric=metric,
            max_features=args.loco_max_features,
        )
        loco_dfs[metric] = loco_df
        loco_topk[metric] = loco_df.head(args.topk)["feature"].tolist()
        loco_df.to_csv(out_dir / f"loco_{metric}.csv", index=False)
        print(f"  Top-{args.topk}: {loco_topk[metric][:5]} ...")

        print(f"\n=== SHAP objective: {metric} ===")
        top_names, shap_df = shap_ranking(
            context_clf,
            loco_df,
            feature_names,
            x_train,
            x_test,
            y_test,
            metric=metric,
            topk=args.topk,
            universe=args.shap_universe,
            n_perm=args.shap_n_perm,
            n_instances=args.shap_n_instances,
            n_background=args.shap_n_background,
            seed=args.seed,
        )
        shap_dfs[metric] = shap_df
        shap_topk[metric] = top_names
        shap_df.to_csv(out_dir / f"shap_{metric}.csv", index=False)
        print(f"  Top-{args.topk}: {top_names[:5]} ...")

    plot_loco_panels(loco_dfs, out_dir, top_n=min(12, args.topk))
    mat = overlap_matrix(loco_topk, shap_topk)
    mat.to_csv(out_dir / "overlap_matrix.csv")
    plot_overlap_heatmap(mat, out_dir)

    eval_df = None
    if not args.skip_eval:
        print("\n=== Subset evaluation (refit TabPFN on each top-K list) ===")
        rows = []
        base = make_clf()
        base.fit(x_train, y_train)
        p_base = pos_proba(base, x_test)
        pred = (p_base >= select_threshold(y_train, oof_pos_proba(make_clf, x_train, y_train))[0]).astype(int)
        rows.append(
            {
                "variant": "baseline",
                "n_features": len(feature_names),
                "threshold": np.nan,
                "test_pr_auc": round(float(average_precision_score(y_test, p_base)), 4),
                "test_f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
                "test_f2": round(float(fbeta_score(y_test, pred, beta=2.0, zero_division=0)), 4),
            }
        )
        for metric in args.scorings:
            for method, topk_dict in [("loco", loco_topk), ("shap", shap_topk)]:
                name = f"{method}_{metric}"
                rows.append(
                    evaluate_feature_subset(
                        make_clf, name, topk_dict[metric], feature_names, x_train, x_test, y_train, y_test
                    )
                )
        eval_df = pd.DataFrame(rows)
        eval_df.to_csv(out_dir / "subset_evaluation.csv", index=False)
        print(eval_df.sort_values("test_pr_auc", ascending=False).to_string(index=False))

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features": len(feature_names),
        "scorings": list(args.scorings),
        "topk": args.topk,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rp = write_report(
        out_dir,
        loco_dfs=loco_dfs,
        shap_dfs=shap_dfs,
        loco_topk=loco_topk,
        shap_topk=shap_topk,
        eval_df=eval_df,
        topk=args.topk,
        meta=meta,
    )
    print(f"\nDone. Report: {rp}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    p.add_argument("--topk", type=int, default=20, help="Top-K features per method × objective")
    p.add_argument("--scorings", nargs="+", default=list(SCORINGS), choices=list(SCORINGS))
    p.add_argument("--loco-max-features", type=int, default=None, help="Cap LOCO ablations (None = all)")
    p.add_argument("--shap-universe", type=int, default=40, help="LOCO pool size for SHAP ranking")
    p.add_argument("--shap-n-perm", type=int, default=40)
    p.add_argument("--shap-n-instances", type=int, default=48)
    p.add_argument("--shap-n-background", type=int, default=64)
    p.add_argument("--device", default="auto", help="TabPFN device: auto, cuda, cpu")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    p.add_argument("--skip-eval", action="store_true", help="Skip subset refit evaluation")
    p.add_argument("--report-only", action="store_true", help="Rebuild report from saved CSVs")
    return p.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
