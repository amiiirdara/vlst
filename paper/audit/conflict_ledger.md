# Conflict ledger

Classification key:

- **blocking numerical conflict** — two sources in the inspect set state different values for the same quantity, and a manuscript Methods/Results sentence cannot choose without a freeze decision.
- **non-blocking formatting discrepancy** — rounding, concatenation lag, or display-only difference that does not change the scientific claim if the canonical source is followed.
- **methodological discrepancy** — protocols differ, or a caption claims a protocol the notebook does not implement.
- **terminology discrepancy** — wording that would violate the scientific rules even if numbers agree.
- **unresolved provenance issue** — origin of a number or figure is incomplete, historical, or not in the inspect set.

Canonical rule already written in `00_front_matter.md`: where a report and a notebook disagree, the notebook is authoritative (D4). Historical revision numbers in `docs/paper_evidence_map.md` are not live results.

---

## Blocking numerical conflicts

### C1. Raw stent-brand string count: 99 vs 106 — **resolved in reports (2026-09-17)**

These were **not** two measurements of the same quantity.

| Number | What it is | Where it came from |
| --- | --- | --- |
| **106** | `pandas` `nunique` on the raw `Stent type-SES` column in `VLST.csv` **before** canonicalize. Same as `encode_stent_brand_column` `meta["n_raw"]`. | Live `stent_encoding.py` on current CSV; Kaggle Part 4/5 print `Stent brand: 106 raw strings -> 9 levels (min_count=30)` (`.nbdump` / inserted notebook). `paper_domain_feature_map.csv` nunique=106. |
| **9** | Levels after alias canonicalize + collapse n < 30 → `other`. | Same encoder; χ² df = 8 in Table 3. |
| **99** | `cleaned.nunique()` inside the EDA categorical χ² helper (stored as “Raw levels” in `categorical_target_association_tests.csv`). **Not** encoder `n_raw`. Current `canonicalize_stent_brand` on this CSV yields **30** unique names before the n<30 collapse, so 99 is an older helper counter, not the live encoder. | `.nbdump/code__analyzes__eda.txt`: `Raw levels: 99 \| Used after collapse: 9`. |

**Quote for Methods:** 106 raw strings → 9 levels (`min_count=30`), matching the last Kaggle nested-CV / interpretability print. Do not quote 99 as `n_raw`. Part 1 Figure 5 caption was updated accordingly.

### C2. Concatenated `paper_results.md` Part 5 vs standalone Part 5 — **rebuild 2026-09-17**

Standalone Part 5 now states that the Version 5 Kaggle CSVs (`interpretability_mutual_info_ranking.csv`, SHAP mean-abs) are in `paper_figures/` after the notebook insert. `rebuild_concat()` was run so `paper_results.md` is regenerated from Parts 0–1 and 4–5 (Parts 2–3 preserved from the prior concat). Re-check the concat after that rebuild; do not quote the old “stayed on Kaggle” sentences.

### C3. Front matter “adjusted OR” examples — **resolved in `00_front_matter.md`**

Terminology now quotes **Table 4b** (`1.1:1Post dilation` 0.152; `Clopidogrel` 0.480) and labels 0.144 / 0.464 as the unidentified Table 4 pair, not the screen.

---

## Methodological discrepancies

### M1. TSSI “same protocol” vs SMOTE flags

| Side A | Side B |
| --- | --- |
| Part 4 §6: “Same stratified 70/30 split and tuning protocol.” Table S-TSSI presented as with vs without the column. | `baseline_tssi_leakage.ipynb` cell 4: `USE_SMOTE = True` (global SMOTE on train). `baseline_without_tssi.ipynb` cell 6: `USE_SMOTE = False` (print: original imbalanced training set). |

The PR-AUC collapse is still directionally a TSSI effect (Gaussian NB unchanged; LR 0.9575 → 0.5077 is too large to be SMOTE alone), but the table is **not** a pure ceteris-paribus TSSI contrast. Nested-CV Part 4 does not use SMOTE and is unaffected.

The with-TSSI notebook applied SMOTE on train; the without-TSSI notebook did not. Part 4 Table S-TSSI caption now states that. Nested-CV Part 4 does not use SMOTE.

**Class:** closed in reports (2026-09-17). Do not write “identical protocol except TSSI.”

### M2. Part 2 88 columns (drop-first) vs Part 4 ~89 (no drop-first)

Documented in Part 1 header and Part 4 methods. Same 9-level encoder; different OHE. **Not a bug** if Methods names both views. Class: methodological discrepancy (documented).

### M3. Stent encoder fitted before the split (Parts 4 and 5)

`min_count=30` collapse uses full-cohort brand frequencies, then CV / 70/30 proceeds. Imputer/scaler/OHE for classics are inside folds (Part 4). Brand-frequency information from held-out rows can in principle affect which strings map to `other`.

**Class:** methodological discrepancy / possible leak. Not quantified in the inspect set. Do not invent a sensitivity analysis.

### M4. Table 4 vs Table 4b

Two different logits (17 unidentified vs 13 reduced). EPV 5.4 vs 7.1. **Not a conflict** if Table 4 is labeled unidentified and Table 4b is quoted. Becomes a conflict only if both are called “the” multivariable model (see C3).

### M5. Nested F1 vs pooled F1

Table 2 vs Table 3. Documented optimistic bias. **Not a conflict** if Table 2 is quoted for operating points.

### M6. Evidence map Revision 7 vs live Part 4

`docs/paper_evidence_map.md` header still narrates local PR **0.6754**, Brier **0.0673**, LightGBM nested 5062/31/31/61. Later revisions in the same file correct this to Version 4 **0.6742 / 0.0102** and Table 2 **5060/33/30/62**.

**Class:** unresolved provenance issue if Revision 7 is quoted; non-blocking if the map is read as a dated ledger.

---

## Terminology discrepancies

### T1. Root README slogan

`README.md` line 2: “Personalized Risk prediction … Machin Learning.” Front matter explicitly forbids “personalised” as a result claim. **Class:** terminology discrepancy. Not a results table.

### T2. Part 3 “external-validation bar”

`feature_extraction_comparison.md` § practical reading: ML-only names are hypothesis-generating “until they pass a pre-specified association or **external-validation bar**.”

Replaced with “held-out / external cohort that this pack does not contain.” Closed in Part 3 and concat (2026-09-17).

### T3. Wang “externally tested on Shantou”

Allowed **only** as a description of Wang 2020’s own score. Front matter and Part 4 state nested CV is not external validation. Evidence map and front matter are aligned on this if Shantou is never attached to TabPFN.

### T4. Banned phrases in live reports

Grep of `paper_results/*.md` did not find “independent predictor,” “clinically useful,” or causal claims presented as findings. Front matter **prohibits** “risk factor”, “causal”, “protective”, “independent predictor”, “clinically useful”, “validated” for this pack. Figure 6 caption correctly uses “association” and “not a causal … claim.”

Part 3 “true” risk markers is in scare quotes as something **not** to claim. Acceptable if the scare quotes stay.

---

## Non-blocking formatting / rounding

| ID | Difference | Class |
| --- | --- | --- |
| F1 | TSSI LR PR-AUC 0.9575 (script) vs 0.958 (Part 4 prose); 0.5077 vs 0.508; CatBoost 0.9773 vs 0.977 | non-blocking formatting |
| F2 | Jaccard 0.1786 (notebook print) vs 5/28 ≈ 0.18 (Part 3 prose) | non-blocking formatting |
| F3 | Prevalence 0.0177 vs “~2%” / “1.8%” in interpretive sentences | non-blocking formatting |
| F4 | Table 4b univariate Previous PCI 6.485 vs Table 2 6.49 vs Table 4 6.46 vs S4 6.73 — **different estimators**, labeled in Part 1 | not a conflict if estimator is named |
| F5 | Dual-tree report copies under `code/**` were absent on disk; glob index may still list them | unresolved provenance for search tools only |

---

## Unresolved provenance issues

| ID | Issue | Class |
| --- | --- | --- |
| P1 | Part 2 per-selector CSVs “not in this repo”; tables reconstructed from notebook displays; XGB 7-name HTML truncated then completed as `WBC; eGFR` | unresolved provenance issue |
| P2 | Table S-CI / S-Δ generated by `run_b3()` in `paper_hygiene_b3_b4_b7.py`, **not** in `baseline_plus_tabpfn.ipynb` | provenance detail (documented); not a numerical conflict |
| P3 | Table S-TSSI rebuilt from hard-coded `ROWS` in `rebuild_tssi_leakage_table.py`, not computed inside either TSSI notebook | documented; matches stored LR/CatBoost prints to 4 dp |
| P4 | TSSI notebooks load `preprocessing.ipynb` arrays; that notebook was **not inspected** | gap (see unsupported_claims) |
| P5 | Complete 81-name predictor list not printed in inspect set | gap |
| P6 | Client/server TabPFN versions “remain unrecorded” (front matter limitation 5) | provenance detail |
| P7 | `paper_results.md` concat lag (C2) | blocking for concat freeze |
| P8 | Evidence map is a multi-revision diary; quoting the header without the later correction reintroduces 0.0673 | unresolved if misquoted |

---

## Conflicts that are **not** conflicts if labeled

| Topic | Why it is OK |
| --- | --- |
| Thinking-high vs local TabPFN | Different constructors; different metrics; reports keep them separate |
| Part 4 nested CV vs Part 5 70/30 | Different questions (prediction vs attribution) |
| Wang 0.8013 vs nested-CV PR-AUC 0.8553 | Frozen integer score vs fitted ML; same cohort, not external validation |
| Wang intermediate bin n = 1577 vs published 1837 | Documented; file n = 5,185 vs Wang printed 5,445 |
| Wang Table 1 14 “no post-dilation” vs CSV `No postdilation` 78/92 | Documented encoding; flipped polarity ROC 0.5084 |
| Class-weighted Part 4 LR vs unweighted Table 4b | Different inferential vs predictive objects |
| SMOTE in leakage notebook vs no SMOTE in Part 4 | Different notebooks |

---

## Scientific-rule checks

| Rule | Status in inspect set |
| --- | --- |
| Wang only as historical comparator on derivation cohort | Part 4 §7 and front matter comply. Part 3 “external-validation bar” is the exception (T2). |
| TSSI not a baseline predictor | Complied in Parts 1–5 nested/selector/attribution. Used only in leakage pair. |
| No causal language as findings | Complied in live part reports. |
| Association vs prediction | Front matter table is the rule; Parts follow it. |
| TabPFN arms separate | Complied in Part 4 tables. |
| No invented CIs / n / hyperparameters | Gaps listed in `unsupported_claims.md` rather than filled. |
