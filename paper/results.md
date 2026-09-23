# Results

Requested `-(3)` names were not on disk. This draft uses the canonical unsuffixed files (same mapping as `paper/frozen_results.yaml`).

| Requested | On disk |
| --- | --- |
| `EDA_paper_figures_and_tables-(3).md` | `paper_results/01_eda/EDA_paper_figures_and_tables.md` |
| `baseline_feature_selections_paper_figures_and_tables-(3).md` | `paper_results/02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md` |
| `feature_extraction_comparison-(3).md` | `paper_results/03_stats_vs_ml/feature_extraction_comparison.md` |
| `baseline_plus_tabpfn_paper_figures_and_tables-(3).md` | `paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md` |
| `wang_vlst_score-(3).ipynb` | `code/modeling/rating/wang_vlst_score.ipynb` |

Language follows **W5**: association (Part 1), attribution (Parts 2–3), prediction (nested CV only). The 70/30 leakage twins are a hold-out demonstration, not nested-CV ranking and not external validation. Odds ratios and overlap catalogues are not treatment effects.

**[RE-SOURCE] nested with vs without anti-leakage.** There is no live 9-arm nested dump with ALL LEAKS ON. The unlabeled two-arm nested dump (thinking **0.9771** / local **0.9635**) is excluded and is not a v3/v3.5-tagged OFF scoreboard. Classic ON vs OFF is section 4. Nested TabPFN ranking is anti-leakage ON only (section 5).

Provenance tags on every prediction number: **checkpoint** (v3 / v3.5 / n/a classic), **thinking** (thinking-high / local / n/a), **leak status** (ON / OFF).

---

## 1. Cohort flow and EDA summary

Source: `paper_results/01_eda/EDA_paper_figures_and_tables.md` (`eda.ipynb`; logic unchanged from the audit). Numbers: YAML `study.*`.

The analysed file is Wang 2020’s derivation cohort: **5,185** patients, **92** definite VLST events, **5,093** non-events, prevalence **0.0177**. Wang’s recruitment flow (cited, not re-derived): 6,038 eligible → 5,185 analysed (236 in-hospital deaths, 413 refused follow-up, 204 lost). Median follow-up 1,502 days; median PCI → VLST 697 days. The notebook printed **no missing values**. `Stent type-SES`: **106** raw strings → **9** levels (`min_count=30`). `Time since stent implantation` is omitted from Table C (time-at-risk, not a baseline covariate).

**Table C (manuscript Table 1)** — association, not prediction.

| Contrast | No VLST (n = 5,093) | VLST (n = 92) | Test | p |
| --- | --- | --- | --- | --- |
| Previous PCI | 94 (1.85%) | 10 (10.87%) | Fisher | 1.25e-05 |
| Diabetes | 1293 (25.39%) | 36 (39.13%) | χ² | 0.003 |
| 3-vessel disease | 1422 (27.92%) | 42 (45.65%) | χ² | 0.000 |
| WBC, 10⁹/L | 8.75 (3.24) | 12.49 (3.92) | MW | 7.90e-21 |
| eGFR | 120.03 (34.10) | 95.88 (19.63) | Welch | 4.64e-20 |
| LV (unnamed) | 44.55 (4.04) | 49.11 (4.23) | Welch | 5.44e-17 |
| 1.1:1 post-dilation (as stored) | 2496 (49.01%) | 14 (15.22%) | χ² | 1.30e-10 |
| No postdilation (complement) | 2597 (50.99%) | 78 (84.78%) | χ² | 1.30e-10 |
| SES (`PES` column) | 3502 (68.76%) | 76 (82.61%) | χ² | 0.004 |
| DAPT during follow-up | 2260 (44.37%) | 35 (38.04%) | χ² | 0.226 |

DAPT / Aspirin / Clopidogrel / Ticagrelor are follow-up persistence after the mandated year, not index-PCI prescriptions. `LV` and `CaI` remain unnamed (`CaI` means match Wang peak troponin I). Do not photocopy Wang Table 1’s post-dilation row: the 14 VLST cases sit on `1.1:1Post dilation` = 1.

**Univariate FDR.** Strongest continuous q-values include `Time since stent implantation` (q = 4.07e-33; excluded from baseline interpretation), `WBC` (q = 9.48e-20), `eGFR` (q = 3.71e-19), `LV` (q = 3.26e-16). The FDR name list used later (n = 20, TSSI excluded) is in Part 3.

**Identified multivariable screen (Table 4b, 13 covariates, unweighted Bernoulli logit, EPV ≈ 7.1).** Quote this, not unidentified Table 4 (17 covariates, EPV ≈ 5.4). Example adjusted ORs (Wald 95% CI): `1.1:1Post dilation` **0.152** [0.081, 0.286]; `Clopidogrel` **0.480** [0.293, 0.787]; `WBC` **1.972** [1.667, 2.331]; `Previous PCI` **6.710** [2.884, 15.610]; `eGFR` **0.568** [0.449, 0.717]; `LV` **1.832** [1.539, 2.181]. OR < 1 is lower modelled odds of recorded VLST, not a treatment benefit. Firth on the same 13 covariates is EDA-only (Methods).

**Artifacts.** Table C `paper_results/01_eda/paper_figures/paper_table_c_cohort_characteristics.csv` / `.png`; Table R `paper_table_test_rationale.csv`; Table 4b YAML `statistical_analysis.association_estimates.table4b_adjusted_or`.

---

## 2. Feature-selection consensus (Part 2)

Source: `paper_results/02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md`. Live dump: `code/modeling/interpretability/Kaggle_baseline_intrepretability_results/baseline_interpretability_results/model_feature_selectors_antileak/`.

**This 2026-09-19 dump is the anti-leakage selector run** (Kaggle papermill `2026-09-19T14:39:53Z`). `split_manifest.json`: TSSI and WBC dropped; labs quantized; stent encoder train-only; scaled **87** columns; fit **4,148** (74 events) / val **1,037** (18 events); no parked 70/30 test. It is the newest Part 2 Kaggle folder on disk.

**Superseded — do not resume.** Pre-antileak cache `data/result/model_feature_selectors/` (2026-08-31): `scaled_n_features=88`, WBC still in the matrix, ML consensus n = 13. `USE_CACHE=False`. Attribution catalogues do not feed Part 4.

**How much is kept.** LOCO saturates its 60-name cap on every model; SHAP universe = 40; FFS path lengths: lr 12, rf 8, rf_b 6, cat 8, xgb 10, xgb_b 12, lgb 4. Top-20 unions per model 31–34. Scored union **86**. Strict 7 × 3 intersection **0**.

**Cross-model intersection (top-20, all 7 models).** LOCO: `Cre`, `eGFR` (n = 2). SHAP: `Cre`, `HGB`, `LDL`, `eGFR` (n = 4). FFS: none. Selector-union Jaccard: LOCO–SHAP **0.60**, SHAP–FFS **0.49**, LOCO–FFS **0.37**.

**Within-model three-way union = ML consensus n = 10:** `Clopidogrel`, `Cre`, `HGB`, `HbA1c`, `LDL`, `LV`, `Men`, `No postdilation`, `Stent type-SES_xiencev`, `eGFR`. Per-model n: cat 5, lgb 1, lr 4, rf 3, rf_b 2, xgb 2, xgb_b 4. `WBC` is not a column.

**Artifacts.** Part 2 Figures 1–2, 6–7 and Tables 0–3 in `paper_results/02_ml_selectors/paper_figures/`; dump `selector_common_by_model_algorithms.csv`, `selector_report.md`, `split_manifest.json`. YAML `feature_selection_models`.

---

## 3. Stats-versus-ML feature overlap (Part 3)

Source: `paper_results/03_stats_vs_ml/feature_extraction_comparison.md`. Catalogues: Part 1 FDR-20 vs the **2026-09-19 anti-leak** ML-10 dump. `stats_vs_ml_comparison.ipynb` loads that dump and asserts Jaccard 5/25; figures from `rebuild_part3_paper_figures.py`.

| Catalogue | n | Names / note |
| --- | ---: | --- |
| Statistical FDR (TSSI excluded) | 20 | `eda.ipynb` |
| ML three-way union (PR-AUC) | 10 | 2026-09-19 antileak dump |
| Intersection | 5 | `Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR` |
| Union | 25 | |
| Jaccard | **5/25 = 0.20** | YAML `feature_extraction_models.results` |

Dual-label: `WBC` is FDR-only because it was dropped from the ML matrix. If WBC is stripped from the FDR side as well, name-count Jaccard is 5/24. **ARCHIVED:** ML-13 / Jaccard 5/28 with intersection `{WBC, eGFR, LV, HbA1c, 1.1:1Post dilation}`.

ML-only consensus names (fail FDR): `Cre`, `Men`, `HGB`, `LDL`, `Stent type-SES_xiencev`. Stats-only includes anatomy/stent collinear families plus `WBC`. This is a methods comparison of two extractors, not a ranking of markers.

**Artifacts.** `paper_results/03_stats_vs_ml/paper_figures/fig1_venn_overlap.png`; `table_shared_features.csv`; `table_stats_only.csv`; `table_ml_only.csv`; `table_feature_by_method.csv`.

---

## 4. Leakage contrast (headline finding)

Same seven classic models, same stratified 70/30 (train 3,629 / test 1,556, seed 42), `GridSearchCV` (`GRID_SCORING=average_precision`). **Not nested CV. Not TabPFN.** Five flags flip together; SMOTE is ON only in the leaks-on arm. Do not write “identical except TSSI.” Incentives: TSSI = mixed time-to-event vs completed follow-up; WBC = recording-precision / batch marker; unquantized labs = decimal-grid fingerprint (signature-only probe AP **0.4270**, 243 indicators); full-cohort stent codebook = test-brand leak; train SMOTE = unmatched inflation. Canonical protocol: `paper_results/anti_leakage_protocol.md`.

| Tag | Notebook | Flags |
| --- | --- | --- |
| **ALL LEAKS ON** | `baseline_tssi_leakage.ipynb` | `KEEP_TSSI=True`, `DROP_WBC=False`, `QUANTIZE_CLINICAL=False`, `STENT_ENCODER_TRAIN_ONLY=False`, `USE_SMOTE=True` |
| **ALL LEAKS OFF** | `baseline_without_tssi.ipynb` | inverse flags, `USE_SMOTE=False` |

Hold-out metrics from dump `test_metrics.csv` (YAML `kaggle_tssi_leakage` / `kaggle_without_tssi`). Δ = ON − OFF. Positive Δ is optimistic bias of the leaks-on pipeline on this 1,556-row hold-out.

### 4.1 PR-AUC (primary ranking metric at prevalence 0.0177)

| Model | ON | OFF | Δ PR-AUC |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.9134 | 0.3431 | **+0.5703** |
| Decision tree | 0.7524 | 0.1378 | **+0.6146** |
| Random forest | 0.9400 | 0.4874 | **+0.4526** |
| Gaussian NB | 0.2728 | 0.0564 | **+0.2163** |
| CatBoost | 0.9599 | 0.4942 | **+0.4656** |
| XGBoost | 0.9547 | 0.5685 | **+0.3862** |
| LightGBM | 0.9687 | 0.6675 | **+0.3011** |

Every model’s hold-out PR-AUC is higher on ALL LEAKS ON. Inflation ranges **+0.30 to +0.61**. LightGBM is the least inflated booster and still drops by 0.30.

### 4.2 ROC-AUC

| Model | ON | OFF | Δ ROC-AUC |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.9954 | 0.8318 | **+0.1635** |
| Decision tree | 0.9266 | 0.7009 | **+0.2257** |
| Random forest | 0.9977 | 0.9287 | **+0.0690** |
| Gaussian NB | 0.7425 | 0.7493 | **−0.0068** |
| CatBoost | 0.9964 | 0.9103 | **+0.0862** |
| XGBoost | 0.9963 | 0.9183 | **+0.0779** |
| LightGBM | 0.9986 | 0.9404 | **+0.0582** |

ROC-AUC compression is smaller than PR-AUC because negatives dominate. Gaussian NB is the exception: ROC-AUC is slightly *higher* on ALL LEAKS OFF while PR-AUC still falls (0.2728 → 0.0564). Dual-label that row; do not summarise GNB as unchanged.

### 4.3 F1, recall, precision, accuracy (notebook operating point)

| Model | F1 ON→OFF (Δ) | Recall ON→OFF | Precision ON→OFF | Acc ON→OFF |
| --- | --- | --- | --- | --- |
| LR | 0.6753 → 0.2093 (**+0.4660**) | 0.9286 → 0.6429 | 0.5306 → 0.1250 | 0.9839 → 0.9126 |
| DT | 0.7059 → 0.2791 (**+0.4268**) | 0.8571 → 0.4286 | 0.6000 → 0.2069 | 0.9871 → 0.9602 |
| RF | 0.8333 → **0.0000** (**+0.8333**) | 0.7143 → **0.0000** | 1.0000 → 0.0000 | 0.9949 → 0.9820 |
| GNB | 0.0437 → 0.0370 (+0.0067) | 0.7143 → 0.8571 | 0.0225 → 0.0189 | 0.4370 → 0.1973 |
| CatBoost | 0.9231 → 0.4096 (**+0.5134**) | 0.8571 → 0.6071 | 1.0000 → 0.3091 | 0.9974 → 0.9685 |
| XGBoost | 0.9231 → 0.5500 (**+0.3731**) | 0.8571 → 0.3929 | 1.0000 → 0.9167 | 0.9974 → 0.9884 |
| LightGBM | 0.9231 → 0.4444 (**+0.4786**) | 0.8571 → 0.2857 | 1.0000 → 1.0000 | 0.9974 → 0.9871 |

Random forest F1 on ALL LEAKS OFF is **0** (no predicted events at the notebook cut) while ROC-AUC remains 0.9287. Boosters that look near-perfect on ON (F1 0.9231, precision 1.0) drop to F1 0.41–0.55 once the leak flags are off. Accuracy stays high on both arms except GNB because the hold-out is 1,528/1,556 non-events; accuracy is not the ranking metric.

**Reading.** A 70/30 GridSearch pipeline that retains TSSI (and WBC, unquantized labs, a full-cohort stent encoder, and train SMOTE) **overstates hold-out ranking on this derivation file**. That is why nested CV, Part 2, and Part 5 implement **all five OFF flags** (not TSSI+WBC only). This subsection does not transfer to TabPFN and is not nested-CV performance.

**ARCHIVED — do not insert:** LR PR-AUC 0.9575 → 0.5077; CatBoost 0.9773 → 0.6582.

**Artifacts.**  
ON: `code/modeling/rating/Kaggle_baseline_tssi_leakage_results/baseline_leakge_results/modeling_tssi_leakage/test_metrics.csv`  
OFF: `code/modeling/rating/Kaggle_baseline_without_tssi_results/baseline_without_leakage/modeling_without_tssi/test_metrics.csv`  
Paper restyle: `paper_results/04_tabpfn_rating/paper_figures/paper_table_s_tssi_leakage.csv`; dump `roc_pr_curves.png` / `confusion_matrices.png`; sub-report `paper_results/04_tabpfn_rating/leakage_contrast_paper_figures_and_tables.md`; Part 4 pointer Table S-TSSI / Figure S-TSSI.

---

## 5. Nested ranking (TabPFN checkpoints; library-default classic panel)

Source: `paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md`. Dump: `code/modeling/rating/Kaggle_baseline_plus_tabpfn_results/baseline_plus_tabpfn_results/modeling_results/` (`run_manifest.json`: Tesla T4, `tabpfn==9.0.0`, `tabpfn-client==0.6.0`, 9 arms). Nested 5×4, seed 42, inner loop = F1 threshold only, classics = library defaults + class weighting (**untuned reference panel**), GridSearch winners **not** imported, no SMOTE.

**Leak status for this entire table: ON** (ALL LEAKS OFF: TSSI+WBC dropped, labs quantized pre-split, stent encoder train-fold only, no SMOTE, scaler/OHE inside splits). **[RE-SOURCE]** for a matched nested OFF 9-arm dump — see header. Classic ON vs OFF remains section 4.

Pooled nested OOF ranking (threshold-independent). Bootstrap 95% CIs: stratified patient-level resample of OOF, `n_boot=2000`, seed 42; models not re-fit.

| Rank | Model | Checkpoint | Thinking | Leak | PR-AUC | 95% CI | ROC-AUC | Brier |
| ---: | --- | --- | --- | --- | ---: | --- | ---: | ---: |
| 1 | TabPFN thinking v3.5 | hosted `v3.5_default` | thinking-high (`effort=high`, metric `average_precision`) | ON | **0.9212** | [0.8785, 0.9613] | 0.9963 | 0.0047 |
| 2 | TabPFN v3.5 | local `tabpfn-v3.5-20260909.safetensors` | local / no-thinking | ON | **0.8957** | [0.8446, 0.9426] | 0.9916 | 0.0048 |
| 3 | TabPFN thinking v3 | hosted `v3_default` | thinking-high | ON | 0.8319 | [0.7626, 0.8945] | 0.9834 | 0.0066 |
| 4 | TabPFN v3 | local `tabpfn-v3-classifier-v3_default.ckpt` | local / no-thinking | ON | 0.7150 | [0.6260, 0.8074] | 0.9731 | 0.0099 |
| 5 | XGBoost | n/a classic | n/a | ON | 0.6322 | [0.5331, 0.7247] | 0.9374 | 0.0100 |
| 6 | LightGBM | n/a classic | n/a | ON | 0.6271 | [0.5313, 0.7202] | 0.9444 | 0.0106 |
| 7 | CatBoost | n/a classic | n/a | ON | 0.5707 | [0.4750, 0.6729] | 0.9404 | 0.0108 |
| 8 | Random forest | n/a classic | n/a | ON | 0.3506 | [0.2660, 0.4633] | 0.8883 | 0.0150 |
| 9 | Logistic regression | n/a classic | n/a | ON | 0.2596 | [0.1834, 0.3588] | 0.8651 | 0.0788 |

**TabPFN vs library-default reference panel (leak ON).** Highest default-classic PR-AUC is XGBoost 0.6322, then LightGBM 0.6271. These are not GridSearch winners. Paired bootstrap Δ PR-AUC: thinking v3.5 − LightGBM **0.2941** (0.2071–0.3805), P(Δ ≤ 0) = 0/2000; TabPFN v3.5 − LightGBM **0.2686** (0.1822–0.3559), P = 0/2000; thinking v3.5 − XGBoost **0.2891** (0.2038–0.3791), P = 0/2000. The Δ is a contrast of stored OOF vectors, not a claim that nested inner GridSearch would leave the same gap.

**v3 vs v3.5 (same thinking status, leak ON).** Local: v3.5 0.8957 vs v3 0.7150. Thinking: v3.5 0.9212 vs v3 0.8319. Do not collapse checkpoints.

**Thinking vs non-thinking (same checkpoint family, leak ON).** v3.5: thinking 0.9212 vs local 0.8957. v3: thinking 0.8319 vs local 0.7150. Thinking v3.5 is **not** higher than local v3.5 in every fold (fold 1: local 0.9365 vs thinking 0.8802). Fold PR-AUC thinking v3.5: 0.8802, 0.9078, 0.9434, 0.9858, 0.8961.

**With vs without anti-leakage.** Nested live dump = ON only (**[RE-SOURCE]** for nested OFF). Do not quote excluded unlabeled 0.9771 / 0.9635 as nested OFF or as untagged v3/v3.5. The quantified leak contrast is section 4 (classics, 70/30).

Nested-CV discrimination on the derivation file is not external validation.

**Nested operating points (Table 2; inner-fold F1 thresholds — not pooled Table 3).** Thinking v3.5: t 0.318 ± 0.059; 5083 / 10 / 15 / 77; recall 0.8370; precision 0.8851; F1 0.8603. TabPFN v3.5: 5082 / 11 / 20 / 72; recall 0.7826; F1 0.8229. LightGBM: 5056 / 37 / 38 / 54; recall 0.5870; F1 0.5902.

**Artifacts.** Part 4 Tables 1, 2, S-CI, S-Δ, S-folds; dump `tables/model_comparison.csv`, `tables/nested_cv_operating_point.csv`, `oof/oof_predictions.csv`; Figures 1–2 `paper_results/04_tabpfn_rating/paper_figures/`. YAML `nested_cv_v35_antileakage_on`.

---

## 6. Comparator: Wang 2020 integer score

Source: `code/modeling/rating/wang_vlst_score.ipynb`; restyle Part 4 report §7. YAML `wang_2020`. Frozen **published Table 2 integer points**, weights **not re-fit**. Not a nested-CV arm, not a Cox linear-predictor re-fit, not this pack’s external validation. Wang’s Shantou c = 0.82 describes **Wang’s score only** (file absent).

**Label-mapping caveats** (notebook cell 2):

1. The 4-point “no post-dilation” item is `No postdilation` = 1 (78/92 events). Using Wang Table 1’s 14 VLST “No post-dilation” cases (scoring `1.1:1Post dilation` = 1) yields ROC-AUC **0.5084** — anti-predictive; **rejected**.
2. The SES point is the CSV **`PES`** flag (rates match Wang Table 1 SES 82.61% / 68.76%), not the 106-level `Stent type-SES` strings.

**Discrimination of the accepted encoding.** Full-cohort ROC-AUC **0.8013**, PR-AUC **0.1032** (Wang published derivation c = 0.80 [0.75, 0.85]). Same five outer folds as Part 4 nested CV, score not refit: fold-mean ROC-AUC **0.8005 ± 0.0607**, PR-AUC **0.1134 ± 0.0518**. That is the split used for the frozen comparator — not the 70/30 leakage twins.

Risk bins (low ≤7 / intermediate 8–9 / high ≥10): n = 3135 / 1577 / 473; observed rates 0.0051 / 0.0222 / 0.0867. Wang’s published n’s sum to 5,445 ≠ 5,185; intermediate count does not match.

Nested thinking v3.5 PR-AUC 0.9212 versus this frozen score’s PR-AUC 0.1032 is derivation-cohort nested CV versus an integer Cox score on a binary label — not external validation.

---

<!--
SECTION → FILE TRACEABILITY

Requested -(3) names: not on disk; unsuffixed canonical paths used.

1. Cohort + EDA
   paper_results/01_eda/EDA_paper_figures_and_tables.md §0 Table C, Table R, Table 4b
   paper_results/01_eda/paper_figures/paper_table_c_cohort_characteristics.csv
   paper/frozen_results.yaml study.*; table4b_adjusted_or; epv
   code/analyzes/eda.ipynb

2. Part 2
   paper_results/02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md
   Kaggle_baseline_intrepretability_results/.../model_feature_selectors_antileak/
     (papermill 2026-09-19 = anti-leakage ON; 87 cols)
   data/result/model_feature_selectors/ = ARCHIVED 88-col 2026-08-31
   YAML feature_selection_models

3. Part 3
   paper_results/03_stats_vs_ml/feature_extraction_comparison.md
   paper_results/03_stats_vs_ml/paper_figures/
   stats_vs_ml_comparison.ipynb (dump-backed 5/25)
   rebuild_part3_paper_figures.py
   YAML feature_extraction_models.results

4. Leakage
   YAML kaggle_tssi_leakage.test_metrics / kaggle_without_tssi.test_metrics
   modeling_tssi_leakage/test_metrics.csv
   modeling_without_tssi/test_metrics.csv
   paper_table_s_tssi_leakage.csv
   Δ = ON−OFF from frozen scalars

5. Nested benchmark
   YAML nested_cv_v35_antileakage_on; kaggle_baseline_plus_tabpfn
   dump model_comparison.csv; nested_cv_operating_point.csv; run_manifest.json
   Part 4 Tables 1, 2, S-CI, S-Δ, S-folds
   [RE-SOURCE] nested ON vs OFF TabPFN: no matched dump
   excluded unlabeled 0.9771 / 0.9635

6. Wang
   code/modeling/rating/wang_vlst_score.ipynb
   YAML wang_2020
   Part 4 report §7
-->
