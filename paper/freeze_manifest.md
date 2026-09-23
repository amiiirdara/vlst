# Freeze manifest — VLST manuscript numerical registry

**Date:** 2026-09-22  
**Registry:** [`paper/frozen_results.yaml`](frozen_results.yaml)  
**Status:** nested TabPFN anti-leakage-ON is **frozen** (`nested_cv_v35_antileakage_on`): thinking v3.5 PR-AUC **0.9212**, TabPFN v3.5 **0.8957**. Unlabeled 0.9771 / 0.9635 excluded. Leakage-contrast + Part 2/5 dumps are current.

Notebooks were not rerun for nested CV. Leakage `best_params_` were copied from executed notebook prints into dump `best_params.csv` and Table S-TSSI-HP.

The requested filenames with suffixes `-(2)` and `-(3)` were **not on disk**. Canonical copies below were used instead.

---

## 1. Files used

### Numerical source of truth (Markdown reports)

| Requested name | File used |
| --- | --- |
| `EDA_paper_figures_and_tables-(3).md` | `paper_results/01_eda/EDA_paper_figures_and_tables.md` |
| `baseline_feature_selections_paper_figures_and_tables-(3).md` | `paper_results/02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md` |
| `feature_extraction_comparison-(3).md` | `paper_results/03_stats_vs_ml/feature_extraction_comparison.md` |
| `baseline_plus_tabpfn_paper_figures_and_tables-(3).md` | `paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md` |
| `tabpfn_interpretability_paper_figures_and_tables-(3).md` | `paper_results/05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md` |
| `paper_results-(2).md` | `paper_results/paper_results.md` |
| `00_front_matter.md` | `paper_results/00_front_matter.md` |
| `paper_evidence_map-(2).md` | `docs/paper_evidence_map.md` (dated ledger only; Revision 7 dump numbers **excluded**) |

### Audit companions (status, not competing numbers)

- `paper/audit/file_inventory.md`
- `paper/audit/methods_inventory.md`
- `paper/audit/numerical_claims_inventory.md`
- `paper/audit/conflict_ledger.md`

### Provenance only (not a second numerical source)

- `code/analyzes/eda.ipynb`
- `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb` (Jaccard print 0.1786)
- `code/modeling/interpretability/baseline_feature_selections.ipynb`
- `code/modeling/interpretability/tabpfn_interpretability.ipynb` (`e356bb1`, Version 5)
- `code/modeling/rating/baseline_plus_tabpfn.ipynb` (papermill 2026-09-17; `tabpfn==9.0.0` / v3.5)
- `code/modeling/rating/baseline_tssi_leakage.ipynb` / `baseline_without_tssi.ipynb`
- `code/modeling/rating/wang_vlst_score.ipynb`
- `code/modeling/rating/rebuild_tssi_leakage_table.py`
- `code/modeling/tools/paper_hygiene_b3_b4_b7.py`
- Table CSVs named in the Part 1–5 reports (e.g. `paper_table_s_tssi_leakage.csv`)

---

## 2. Frozen values (safe to quote)

Draft from `status: frozen` in the YAML. Headline scalars:

### Cohort

- n = **5,185**; events = **92**; non-events = **5,093**; prevalence = **0.0177** (1.77%).
- Flow cited from Wang 2020: 6,038 eligible → 236 in-hospital deaths, 413 refused follow-up, 204 lost.
- Median follow-up **1,502** days; median PCI → VLST **697** days.
- Outcome: ARC 2007 **definite** stent thrombosis >1 year after implantation.

### Predictors / leakage

- **81** non-ID non-TSSI columns after dropping `NO.`, `Name`, and `Time since stent implantation` (names: `paper/table_s0_baseline_predictors.md`).
- Stent encoder: **106** raw `Stent type-SES` strings → **9** levels (`min_count=30`). Do not quote 99 as `n_raw`.
- **TSSI is not a baseline predictor.** Nested-CV, Part 2, and Part 5 drop it.
- EDA: no missing values printed.

### Association (Part 1) — not prediction, not causal

- Quote **Table 4b** (13 covariates, unweighted logit, EPV ≈ **7.1**). Example adjusted ORs: `1.1:1Post dilation` **0.152** [0.081, 0.286]; `Clopidogrel` **0.480** [0.293, 0.787]; WBC **1.972** [1.667, 2.331].
- Do **not** quote Table 4 (17 covariates, unidentified, EPV ≈ 5.4; adj OR 0.042 / 0.527; Wald CI undefined).
- Supplementary Table S2: 16 LR interaction tests; q < 0.05 for LV × eGFR and Men × eGFR only. Hypothesis-generating.

### Prediction (Part 4 nested 5×4 CV)

**Live anti-leakage-ON nested TabPFN** is the 9-arm dump `Kaggle_baseline_plus_tabpfn_results/baseline_plus_tabpfn_results/`. Grep freeze YAML for the **top-level** key `nested_cv_v35_antileakage_on`. Do not quote unlabeled 0.9771 / 0.9635.

| Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Brier (95% CI) | ECE (8 q-bins) | Cite |
| --- | --- | --- | --- | --- | --- |
| TabPFN thinking v3.5 | 0.9212 [0.8785, 0.9613] | 0.9963 [0.9932, 0.9986] | 0.0047 [0.0037, 0.0058] | 0.0028 | frozen |
| TabPFN v3.5 | 0.8957 [0.8446, 0.9426] | 0.9916 [0.9828, 0.9978] | 0.0048 [0.0038, 0.0060] | 0.0003 | frozen |
| TabPFN thinking v3 | 0.8319 [0.7626, 0.8945] | 0.9834 [0.9709, 0.9939] | 0.0066 [0.0053, 0.0080] | 0.0036 | frozen |
| TabPFN v3 | 0.7150 [0.6260, 0.8074] | 0.9731 [0.9551, 0.9883] | 0.0099 [0.0089, 0.0111] | 0.0059 | frozen |
| XGBoost | 0.6322 [0.5331, 0.7247] | 0.9374 [0.8967, 0.9699] | 0.0100 [0.0084, 0.0116] | 0.0066 | frozen |
| LightGBM | 0.6271 [0.5313, 0.7202] | 0.9444 [0.9135, 0.9698] | 0.0106 [0.0090, 0.0123] | 0.0090 | frozen |
| CatBoost | 0.5707 [0.4750, 0.6729] | 0.9404 [0.9142, 0.9632] | 0.0108 [0.0091, 0.0125] | 0.0031 | frozen |
| Random forest | 0.3506 [0.2660, 0.4633] | 0.8883 [0.8413, 0.9296] | 0.0150 [0.0144, 0.0155] | 0.0090 | frozen |
| Logistic regression | 0.2596 [0.1834, 0.3588] | 0.8651 [0.8220, 0.9044] | 0.0788 [0.0730, 0.0844] | 0.1156 | frozen |

Nested operating point (Table 2), selected:

- Thinking v3.5: t = 0.318 ± 0.059; PPV 0.8851; recall 0.8370; NPV 0.9971; F1 0.8603; 5083/10/15/77.
- TabPFN v3.5: t = 0.386 ± 0.061; recall 0.7826; NPV 0.9961; 5082/11/20/72.
- LightGBM: t = 0.076 ± 0.037; recall 0.5870; 5056/37/38/54.
- XGBoost: t = 0.238 ± 0.104; recall 0.5217; 5068/25/44/48.

CIs: stratified bootstrap of pooled OOF, n_boot = 2,000, seed 42; models not re-fit.

Δ PR-AUC vs LightGBM: thinking v3.5 **+0.2941** (0.2071–0.3805), P(Δ ≤ 0) = 0/2000; TabPFN v3.5 **+0.2686** (0.1822–0.3559), P(Δ ≤ 0) = 0/2000. Thinking v3.5 vs XGBoost **+0.2891** (0.2038–0.3791), P(Δ ≤ 0) = 0/2000. Fold wins vs LightGBM: all four TabPFN arms 5/5.

The unlabeled two-arm dump (thinking 0.9771 / local 0.9635) is excluded history.

### TSSI leakage demonstration (70/30; not nested CV)

SMOTE: with-TSSI notebook `True`, without `False`. Quote as leakage, not a matched experiment.

- Logistic regression PR-AUC **0.9134 → 0.3431**.
- CatBoost **0.9599 → 0.4942**.
- Remaining models: CSV `paper_table_s_tssi_leakage.csv`.
- `best_params_`: Table S-TSSI-HP / `paper_table_s_tssi_best_params.csv` (both notebooks).

### Wang 2020 integer score (historical comparator, same 5,185 rows)

- Frozen published points: full-cohort ROC-AUC **0.8013**, PR-AUC **0.1032**.
- Wang published derivation c = **0.80** (0.75–0.85), cited.
- Wang Shantou c = **0.82** (n = 2,058) is **Wang’s** external test of **Wang’s** score. File not in this repository. **Not** this pack’s external validation. **Not** a validation cohort for the ML models.
- Encoding: SES point on `PES`; four post-dilation points on `No postdilation=1`. Alternate encoding using Wang Table 1’s 14 “No post-dilation” VLST cases: ROC-AUC **0.5084** (do not use).

### Attribution (Part 5, 70/30 seed 42; not nested-CV prediction)

- Train 3,629 / 64 events; held-out 1,556 / 28 events.
- SHAP: all 1,556 held-out rows; top mean |SHAP| eGFR **1.2288**, CaI **1.0867**, Cre **0.8093**, LV **0.4828**; WBC absent.
- Consensus 3/3: CaI, eGFR, LV.
- k-SII / waterfall: one held-out VLST=1 row, cohort index **5176**.
- Train MI top: CaI **0.022005**; Cre train MI **0.000000**.
- Binary PDP ΔP (train empirical prior): Previous PCI **+0.0137**; `1.1:1Post dilation` **−0.0093**. Not Part 4 risk.

### Part 2 / Part 3 catalogues (attribution / methods; not Part 4 masks)

- Part 2: fit 4,148 (74 events) / val 1,037 (18 events); **87** scaled columns (TSSI+WBC dropped); FFS path lengths as in YAML; 7×3 intersection **0**; scored union **86**.
- Part 3: FDR n=20 vs ML consensus n=10; intersection 5 names (`Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR`); Jaccard **5/25 = 0.20**. Dual-label: WBC is FDR-only because it is not in the ML matrix.

---

## 3. Methods-only items

These may appear in Methods. They are not Results claims.

- Analysis setting: derivation-cohort nested CV only; no temporal/ML external test.
- Inclusion/exclusion flow cited from Wang 2020, not reconstructed from extra eligibility code.
- Post-index DAPT / Aspirin / Clopidogrel / Ticagrelor persistence after the mandated year.
- Imputation (inert), encoding (Part 2 drop-first 88 vs Part 4 ~89 no drop-first vs Part 5 native), scaling inside CV splits, stent encoder applied on the full frame before Part 4 split.
- Inner 4-fold CV tunes **F1 threshold only**, not hyperparameters and not feature sets.
- Classic nested-CV models: explicit constructor kwargs in `RUN_MODELS` plus library defaults for unspecified arguments; inner loop tunes F1 threshold only; no unpublished hyperparameter grid.
- TabPFN thinking-high: `tabpfn_client.TabPFNClassifier`, thinking high, metric average_precision.
- TabPFN local: `from tabpfn import TabPFNClassifier`, `n_estimators="auto"`, **no** `balance_probabilities`; restore prior skipped.
- Bootstrap design (OOF resampling, not model re-fit).
- Part 5 HTTP 429 → local SHAP/k-SII; PDP y-axis is train empirical prior.
- Wang’s Shantou and Dangas c=0.66 as **literature** about Wang’s paper, not results computed here.
- Ethics 2013-256; NCT03491891; cite Wang for consent/data availability.

---

## 4. Excluded (do not quote as this freeze)

- Table 4 unidentified 17-covariate ORs (0.042 / 0.527; Wald undefined).
- Table 3 pooled F1 / pooled TabPFN recall 0.8152 / 0.8478.
- EDA χ² helper “Raw levels” = **99**.
- Evidence-map Revision 7 / old dump: local Brier **0.0673**, local PR-AUC **0.6754**, local nested t **0.915**, SHAP 15+15, Cre |SHAP| 0.158, k-SII rows 5099/5093.
- Historical thinking-high Brier 0.0060 / 0.0360 (other dumps).
- Treating nested CV, Part 5 70/30, or Wang’s Shantou as **external validation of this pack**.
- Causal / “risk factor” / “protective” / “clinically useful” / “validated” language.

---

## 5. Unresolved vs found in code

**Still unresolved (leave blank):**

- `preprocessing.ipynb` details for TSSI `.npy` arrays.
- TabPFN client/server **pip versions** recorded this run: `tabpfn==9.0.0`, `tabpfn_client==0.6.0`; weights `tabpfn-v3.5-20260909.safetensors`.
- ECE (8 quantile bins) printed this Part 4 run (`paper_table_s_ece.csv`). Calibration slope / intercept still not computed.
- This-run `oof_predictions.csv` is in `code/modeling/rating/baseline_plus_tabpfn_results/` and `data/result/modeling_results/oof/`. Table S-CI / S-Δ recomputed.

**Found / now implemented:**

- **NPV** = TN/(TN+FN) from frozen Table 2 2×2. `classification_report` does **not** label it; class-0 precision is NPV at 2 decimals. Part 4 `metrics_at_threshold` and TSSI `evaluate_model` now print `npv`.
- **TSSI GridSearch `best_params_`** (not Part 4 nested-CV HPs). With TSSI: LR `C=1` L1; RF `max_depth=15`; CatBoost depth 6 / 100 iter; XGB lr 0.05 `max_depth=5`. Without TSSI: LR `C=10` L1; RF `max_depth=5`; CatBoost depth 4 / 200 iter; XGB/LGB `max_depth=3`, lr 0.1. Full table: Part 4 Table S-TSSI-HP.
- **Firth:** Table 4b 13-covariate association sensitivity (EDA / `run_b4`). Not nested CV.

**Found in code / frozen counts (not invented grids or new ECE):**

- **NPV** = TN/(TN+FN); this Part 4 notebook prints Nested NPV matching Table 2.
- **Classic constructors:** explicit `RUN_MODELS` kwargs (LR `max_iter=1000` balanced; RF balanced `n_jobs=-1`; XGB `aucpr` + `tree_method=hist` + fold `scale_pos_weight`; LGB `average_precision` balanced; CatBoost Balanced PRAUC GPU). Unspecified = library default. TSSI GridSearch winners are not Part 4 HPs.
- This run prints local `balance_probabilities=False`. Thinking-high constructor unchanged.

---

## 6. Remaining conflicts

**No blocking conflict on a single frozen quantity.** The YAML `freeze.blocked` flag is `false`.

Ledger items that look like conflicts are **different quantities** (keep both labelled; do not merge):

| ID | Pair | Freeze rule |
| --- | --- | --- |
| C1 | 106 vs 99 vs 9 | Quote 106 → 9. Exclude 99. |
| Table 4 vs 4b | 0.042/0.527 vs 0.152/0.480 | Quote 4b. Exclude Table 4 as the clinical model. |
| Nested vs pooled F1 | Table 2 vs Table 3 | Quote Table 2. |
| Thinking-high vs local | two models | Never collapse. |
| TSSI SMOTE mismatch | with vs without notebooks | Frozen as leakage demo with mismatch named. |
| Evidence-map Rev 7 | 0.0673 vs Version 4 0.0102 | Exclude Rev 7. |
| Previous PCI OR estimators | 6.49 / 6.46 / 6.73 / 6.485 | Different tables; do not mix. |
| Wang published intermediate n 1,837 vs file 1,577 | bin counts | Freeze file counts; note Wang’s printed n. |

---

## 7. Drafting rules encoded in the registry

1. Do not merge versions of a result.
2. TSSI is never a valid baseline feature.
3. Do not label any result in this pack as external validation.
4. Do not describe Wang 2020 as a validation cohort.
5. Do not convert association estimates into causal or risk-factor claims.
6. Preserve PR-AUC vs ROC-AUC; point estimate vs CI; nested vs apparent; thinking-high vs local; with-TSSI vs without-TSSI.

---

## 8. Is the registry safe for manuscript drafting?

**Yes.** `paper/frozen_results.yaml` is **READY FOR MANUSCRIPT DRAFTING**.

Draft Results from frozen nested-CV metrics, Table 4b association ORs, the frozen Wang integer-score ranking, the TSSI leakage demonstration, and Part 5 attribution ranks. Leave unresolved items blank. Do not treat `docs/paper_evidence_map.md` header diary as live numbers.

If a later notebook dump disagrees with these reports, stop and re-freeze; do not silently overwrite.
