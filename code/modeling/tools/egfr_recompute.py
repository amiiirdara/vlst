"""Recalculate eGFR (and the CKD flags derived from it) from creatinine.

The recorded ``eGFR`` in ``VLST.csv`` is not a function of the recorded ``Cre``
for most controls: CKD-EPI 2009 from ``Cre`` / ``Age`` / ``Men`` reproduces the
stored value (within 1 unit) for 2,243/5,093 controls and 50/92 cases, and most
remaining controls carry values 1.3-1.9x higher. With recalculated eGFR the
case/control means are 91.5 vs 91.9 instead of 95.9 vs 120.0. ``CKD5``
(stage 1-5), ``CKD60`` and ``CKD90`` are exact functions of the recorded value,
so they are rebuilt from the recalculated one.

Assumes ``Cre`` is in umol/L (pending confirmation by the data custodian). The
transform reads only each patient's own row and never the outcome, so applying
it to the whole file before splitting leaks nothing (same status as the
laboratory quantization rule).

Kaggle notebooks carry an in-notebook copy of :func:`recompute_egfr_from_creatinine`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CREATININE_UMOL_PER_MG_DL = 88.4
EGFR_DERIVED_FLAGS = ("CKD5", "CKD60", "CKD90")


def ckd_epi_2009(cre_umol_l, age, male):
    """CKD-EPI 2009 creatinine equation (no race coefficient), mL/min/1.73 m2."""
    scr = np.asarray(cre_umol_l, dtype=float) / CREATININE_UMOL_PER_MG_DL
    age = np.asarray(age, dtype=float)
    male = np.asarray(male, dtype=bool)
    kappa = np.where(male, 0.9, 0.7)
    alpha = np.where(male, -0.411, -0.329)
    ratio = scr / kappa
    return (
        141.0
        * np.minimum(ratio, 1.0) ** alpha
        * np.maximum(ratio, 1.0) ** -1.209
        * 0.993 ** age
        * np.where(male, 1.0, 1.018)
    )


def ckd_stage(egfr):
    """KDIGO G-stage 1-5 from eGFR (G3a/G3b merged as 3, as in ``CKD5``)."""
    egfr = np.asarray(egfr, dtype=float)
    return np.select(
        [egfr >= 90, egfr >= 60, egfr >= 30, egfr >= 15],
        [1, 2, 3, 4],
        default=5,
    )


def recompute_egfr_from_creatinine(frame: pd.DataFrame) -> dict:
    """Overwrite ``eGFR`` and its derived flags in place; return a summary."""
    recorded = pd.to_numeric(frame["eGFR"], errors="coerce").to_numpy(dtype=float)
    egfr = ckd_epi_2009(
        pd.to_numeric(frame["Cre"], errors="coerce"),
        pd.to_numeric(frame["Age"], errors="coerce"),
        pd.to_numeric(frame["Men"], errors="coerce").eq(1),
    )
    egfr = np.round(egfr, 1)
    frame["eGFR"] = egfr
    if "CKD5" in frame.columns:
        frame["CKD5"] = ckd_stage(egfr)
    if "CKD60" in frame.columns:
        frame["CKD60"] = (egfr < 60).astype(int)
    if "CKD90" in frame.columns:
        frame["CKD90"] = (egfr < 90).astype(int)
    return {
        "equation": "CKD-EPI 2009 (no race term), Cre umol/L / 88.4",
        "n_rows": int(len(frame)),
        "n_recorded_within_1": int(np.sum(np.abs(recorded - egfr) <= 1.0)),
        "recorded_mean": float(np.nanmean(recorded)),
        "recomputed_mean": float(np.nanmean(egfr)),
        "flags_rebuilt": [c for c in EGFR_DERIVED_FLAGS if c in frame.columns],
    }
