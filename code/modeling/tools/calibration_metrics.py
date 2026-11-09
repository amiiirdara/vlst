"""Calibration summary from a nested-CV ``oof_predictions.csv`` dump.

Same quantities as the calibration cell of ``baseline_plus_tabpfn.ipynb``, for
dumps produced before that cell reported them:

- Brier, the null Brier of predicting the prevalence for everyone, and the
  scaled Brier ``1 - Brier / null`` (0 = no better than the prevalence).
- O/E: observed events / sum of predicted risks.
- Calibration intercept (logit(p) as an offset) and slope (y on logit(p)).
- ECE on 8 quantile bins, kept as a secondary figure.

Usage::

    python code/modeling/tools/calibration_metrics.py path/to/oof_predictions.csv [out.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression


def expected_calibration_error(y_true, probs, n_bins=8):
    """Weighted |observed - predicted| over quantile bins."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        return float("nan")
    idx = np.digitize(probs, edges[1:-1], right=False)
    ece = 0.0
    for b in range(edges.size - 1):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - probs[m].mean())
    return float(ece)


def calibration_summary(y_true, probs):
    """Brier vs the prevalence-only Brier, O/E, calibration intercept and slope."""
    y_arr = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    prevalence = float(y_arr.mean())
    brier = float(np.mean((p - y_arr) ** 2))
    null_brier = prevalence * (1.0 - prevalence)
    lp = logit(p)
    citl = float(brentq(lambda a: expit(a + lp).mean() - prevalence, -20.0, 20.0))
    slope = float(LogisticRegression(C=np.inf, max_iter=1000).fit(lp.reshape(-1, 1), y_arr).coef_[0][0])
    return {
        "brier": brier,
        "null_brier": null_brier,
        "scaled_brier": 1.0 - brier / null_brier,
        "o_e": float(y_arr.sum() / p.sum()),
        "calibration_intercept": citl,
        "calibration_slope": slope,
        "mean_predicted": float(p.mean()),
        "prevalence": prevalence,
        "ece_quantile_8": expected_calibration_error(y_arr, p),
    }


def summarise_dump(oof_csv: Path) -> pd.DataFrame:
    oof = pd.read_csv(oof_csv)
    rows = []
    for column in [c for c in oof.columns if c.endswith("_prob")]:
        rows.append({"model": column[: -len("_prob")], **calibration_summary(oof["y"], oof[column])})
    return pd.DataFrame(rows).sort_values("scaled_brier", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    table = summarise_dump(Path(sys.argv[1]))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))
    if len(sys.argv) == 3:
        table.to_csv(sys.argv[2], index=False)
        print("Saved", sys.argv[2])
