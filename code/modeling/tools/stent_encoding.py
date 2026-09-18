"""Shared VLST stent-brand encoding.

``Stent type-SES`` in ``VLST.csv`` is free-text product names (106 raw strings),
not Wang's SES class flag. ``PES`` / ``ZES`` / ``EVS`` already partition the
cohort (mutually exclusive, cover every row). Wang 2020's published SES rates
match the ``PES`` column exactly — do not invent a second SES bit.

EDA / association notebooks may call :func:`encode_stent_brand_column` on the
full frame so they share one codebook. Nested-CV prediction must fit the
collapse on the **training fold only** (:func:`fit_stent_brand_encoder` +
:func:`transform_stent_brand_column`); held-out brand frequencies must not
decide which strings map to ``other``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

STENT_BRAND_RAW_COL = "Stent type-SES"
STENT_BRAND_COL = "Stent type-SES"  # keep the historical name; values are brand
STENT_CLASS_FLAG_COLS = ("PES", "ZES", "EVS")
STENT_BRAND_MIN_COUNT = 30

_BRAND_ALIASES = {
    "xiencex": "xiencev",
    "resolut": "resolute",
    "parnter": "partner",
    "endeavor": "endeavor",
    "cypher": "cypher",
}


def canonicalize_stent_brand(value: Any) -> str:
    """Normalize one free-text stent product name."""
    if pd.isna(value) or str(value).strip() == "":
        return "missing"
    s = str(value).strip().replace("：", ":").lower()
    s = re.sub(r"\s+", "", s)
    if ":" in s:
        s = s.split(":")[-1]
    for sep in ("，", ",", "/"):
        if sep in s:
            s = s.split(sep)[0]
    return _BRAND_ALIASES.get(s, s)


def collapse_rare_brands(series: pd.Series, min_count: int = STENT_BRAND_MIN_COUNT) -> pd.Series:
    counts = series.value_counts()
    rare = set(counts[counts < min_count].index)
    if not rare:
        return series
    return series.where(~series.isin(rare), "other")


def fit_stent_brand_encoder(
    series: pd.Series,
    *,
    min_count: int = STENT_BRAND_MIN_COUNT,
) -> dict[str, Any]:
    """Learn kept brand levels from one training series (no held-out rows)."""
    meta: dict[str, Any] = {
        "min_count": min_count,
        "numeric": False,
        "kept": set(),
        "n_raw": 0,
        "n_levels": 0,
        "applied": False,
        "value_counts": {},
    }
    if pd.api.types.is_numeric_dtype(series):
        n = int(series.nunique(dropna=True))
        meta.update({"numeric": True, "n_raw": n, "n_levels": n})
        return meta
    canon = series.map(canonicalize_stent_brand)
    counts = canon.value_counts()
    kept = set(counts[counts >= min_count].index)
    n_levels = int(len(kept) + int((counts < min_count).any()))
    meta.update(
        {
            "kept": kept,
            "n_raw": int(series.nunique(dropna=True)),
            "n_levels": n_levels,
            "applied": True,
            "value_counts": counts.to_dict(),
        }
    )
    return meta


def transform_stent_brand_column(
    df: pd.DataFrame,
    codebook: dict[str, Any],
    *,
    raw_col: str = STENT_BRAND_RAW_COL,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply a training-fold codebook. Unseen / rare brands become ``other``."""
    out = df if inplace else df.copy()
    if raw_col not in out.columns or codebook.get("numeric") or not codebook.get("applied"):
        return out
    canon = out[raw_col].map(canonicalize_stent_brand)
    kept = codebook["kept"]
    out[raw_col] = canon.where(canon.isin(kept), "other").astype("object")
    return out


def encode_stent_brand_column(
    df: pd.DataFrame,
    *,
    raw_col: str = STENT_BRAND_RAW_COL,
    min_count: int = STENT_BRAND_MIN_COUNT,
    inplace: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace the raw brand string with canonical levels (n < 30 → other).

    The column name is unchanged so reports still say ``Stent type-SES``.
    Frequency collapse uses the supplied frame only. For nested CV, fit on
    the training fold with :func:`fit_stent_brand_encoder` instead of calling
    this on the full cohort.
    """
    out = df if inplace else df.copy()
    empty: dict[str, Any] = {
        "column": raw_col,
        "n_raw": 0,
        "n_levels": 0,
        "min_count": min_count,
        "applied": False,
        "value_counts": {},
    }
    if raw_col not in out.columns:
        return out, empty
    codebook = fit_stent_brand_encoder(out[raw_col], min_count=min_count)
    out = transform_stent_brand_column(out, codebook, raw_col=raw_col, inplace=True)
    codebook = {**codebook, "column": raw_col}
    return out, codebook


def coerce_stent_class_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Force PES / ZES / EVS to {0, 1}. They already form a partition of the cohort."""
    out = df
    for col in STENT_CLASS_FLAG_COLS:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        bad = ~s.isin([0, 1])
        if bad.any():
            raise ValueError(f"{col}: expected 0/1 after coercion, got {out.loc[bad, col].unique()[:10]}")
        out[col] = s
    return out


def ensure_stent_encoding_on_path() -> Path:
    """Put this directory on ``sys.path`` so notebooks can ``import stent_encoding``."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    return here
