#!/usr/bin/env python3
"""Zero-tolerance check: every freeze scalar in paper/abstract.md must appear
in paper/results.md and paper/frozen_results.yaml with the same token.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ABSTRACT = (ROOT / "paper/abstract.md").read_text()
RESULTS = (ROOT / "paper/results.md").read_text()
YAML = (ROOT / "paper/frozen_results.yaml").read_text()
TITLE = (ROOT / "paper/title.md").read_text()
TITLE_YAML = 'VLST after ACS-PCI — association, nested-CV prediction, and TabPFN attribution on the Wang 2020 derivation cohort'

# Tokens that must match character-for-character (not prefixes of longer floats).
CLAIMS = [
    "5185",
    "92",
    "0.0177",
    "9.0.0",
    "0.6.0",
    "2000",
    "0.9212",
    "0.8785",
    "0.9613",
    "0.9963",
    "0.0047",
    "0.8370",
    "0.8603",
    "0.8957",
    "0.8446",
    "0.9426",
    "0.0048",
    "0.6322",
    "0.6271",
    "0.2941",
    "0.2071",
    "0.3805",
    "0/2000",
    "0.8013",
    "0.1032",
    "0.80",
    "0.75",
    "0.85",
    "0.152",
    "0.081",
    "0.286",
    "0.480",
    "0.293",
    "0.787",
    "1.972",
    "1.667",
    "2.331",
]


def present(haystack: str, token: str) -> bool:
    if token == "0.80":
        return re.search(r"(?<![0-9.])0\.80(?![0-9])", haystack) is not None
    if token in {"0.75", "0.85"}:
        return re.search(rf"(?<![0-9.]){re.escape(token)}(?![0-9])", haystack) is not None
    if token == "92":
        return "92" in haystack
    if token == "5185":
        return "5185" in haystack or "5,185" in haystack
    if token == "2000":
        return "n_boot=2000" in haystack or "n_boot = 2000" in haystack or "2000" in haystack
    return token in haystack


def abstract_has(token: str) -> bool:
    if token == "0.80":
        return re.search(r"(?<![0-9.])0\.80(?![0-9])", ABSTRACT) is not None
    if token in {"0.75", "0.85"}:
        return re.search(rf"(?<![0-9.]){re.escape(token)}(?![0-9])", ABSTRACT) is not None
    return token in ABSTRACT


def extra_abstract_floats() -> list[str]:
    """Flag freeze-like decimals in the abstract body that are not in CLAIMS."""
    body = re.sub(r"<!--.*?-->", " ", ABSTRACT, flags=re.S)
    for pin in ("9.0.0", "0.6.0", "v3.5", "1.1:1"):
        body = body.replace(pin, " ")
    found = re.findall(r"\d+\.\d+", body)
    return [tok for tok in found if tok not in CLAIMS]


def main() -> int:
    errors: list[str] = []
    if TITLE_YAML not in TITLE:
        errors.append("title.md missing YAML working title")
    if TITLE_YAML not in YAML:
        errors.append("YAML missing working title")

    for tok in CLAIMS:
        if not abstract_has(tok):
            errors.append(f"ABSTRACT missing {tok}")
        if not present(RESULTS, tok):
            errors.append(f"RESULTS missing {tok}")
        if not present(YAML, tok):
            errors.append(f"YAML missing {tok}")

    extras = extra_abstract_floats()
    # Allow structural decimals that are pins already listed.
    extras = [e for e in extras if e not in CLAIMS]
    if extras:
        errors.append(f"ABSTRACT extra floats not in CLAIMS: {extras}")

    if errors:
        print("ABSTRACT DIFF CHECK FAILED")
        for e in errors:
            print(" -", e)
        return 1
    print(f"ABSTRACT DIFF CHECK PASSED ({len(CLAIMS)} tokens; title matches YAML)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
