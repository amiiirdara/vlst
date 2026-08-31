"""Shared VLST stent-brand encoding.

``Stent type-SES`` in ``VLST.csv`` is free-text product names (106 raw strings),
not Wang's SES class flag. ``PES`` / ``ZES`` / ``EVS`` already partition the
cohort (mutually exclusive, cover every row). Wang 2020's published SES rates
match the ``PES`` column exactly — do not invent a second SES bit.

Every in-scope notebook should call :func:`encode_stent_brand_column` on the
raw frame so EDA, selectors, nested CV, and TabPFN all see the same 9-level
nominal brand column.
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


def encode_stent_brand_column(
    df: pd.DataFrame,
    *,
    raw_col: str = STENT_BRAND_RAW_COL,
    min_count: int = STENT_BRAND_MIN_COUNT,
    inplace: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace the raw brand string with 9 canonical levels (n < 30 → other).

    The column name is unchanged so reports still say ``Stent type-SES``.
    Frequency collapse uses the supplied frame only (no label, no test fold).
    Full-cohort application is intentional so every notebook shares one codebook.
    """
    out = df if inplace else df.copy()
    meta: dict[str, Any] = {
        "column": raw_col,
        "n_raw": 0,
        "n_levels": 0,
        "min_count": min_count,
        "applied": False,
        "value_counts": {},
    }
    if raw_col not in out.columns:
        return out, meta

    raw = out[raw_col]
    if pd.api.types.is_numeric_dtype(raw):
        # Already integer codes from a previous loader — leave as-is.
        meta["n_raw"] = int(raw.nunique(dropna=True))
        meta["n_levels"] = meta["n_raw"]
        return out, meta

    n_raw = int(raw.nunique(dropna=True))
    encoded = collapse_rare_brands(raw.map(canonicalize_stent_brand), min_count)
    out[raw_col] = encoded.astype("object")
    meta.update(
        {
            "n_raw": n_raw,
            "n_levels": int(encoded.nunique(dropna=True)),
            "applied": True,
            "value_counts": encoded.value_counts().to_dict(),
        }
    )
    return out, meta


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
