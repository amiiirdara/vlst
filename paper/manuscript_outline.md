# Manuscript outline (blueprint only — no drafted prose)

**Freeze:** `paper/frozen_results.yaml` (2026-09-17), `freeze.status = READY_FOR_MANUSCRIPT_DRAFTING`, `freeze.blocked = false`.  
**Do not draft Results/Methods sentences here.** Each subsection lists allowed claims, YAML insert paths, sources, planned display items, and completeness.

**Completeness codes**

| Code | Meaning |
| --- | --- |
| **complete** | Frozen values exist; a paragraph can be written without invention |
| **incomplete** | Allowed claim exists, but at least one insert is `unresolved`, `methods_only` without a scalar, or a named TODO |
| **blocked** | Would require merging conflicting quantities, unsupported wording, or an excluded number |

No subsection is **blocked** if the freeze rules below are followed. Several subsections are **incomplete** because of documented gaps (TabPFN pip versions, Firth, ECE, TSSI `.npy` pipeline). NPV may be quoted only as TN/(TN+FN) from Table 2.

**Global restrictions (every section)**

- Nested CV ≠ external validation. Derivation file ≠ external validation cohort.
- Wang 2020 = **historical comparator** on the same 5,185 rows. Shantou c = 0.82 describes **Wang’s score only**.
- Associations: “associated with.” Never “independent risk factor,” “causal,” “protective,” “clinically useful,” “validated,” “personalised.”
- Prediction: “model discrimination,” “ranking performance,” “calibration,” “predictive performance.”
- TabPFN thinking-high ≠ TabPFN local. Never collapse Brier/PR-AUC/ROC-AUC.
- TSSI (`Time since stent implantation`) is leakage-sensitive; excluded from primary nested-CV, Part 2, and Part 5 models.
- Quote **Part 4 Table 2** nested operating points, not **Part 4 Table 3** pooled F1.
- Quote **Part 1 Table 4b**, not **Part 1 Table 4**, as the identified association logit.

**Numbering aliases (always prefix)**

| Manuscript ID (this blueprint) | Report asset | Never confuse with |
| --- | --- | --- |
| MS Table 1 | Part 1 Table C | — |
| MS Table 2 | Part 1 Table 4b | Part 1 Table 4 (excluded); Part 4 Table 2 |
| MS Table 3 | Part 4 Table 1 + S-CI | Part 4 Table 3 (pooled F1, excluded from headline) |
| MS Table 4 | Part 4 Table 2 (nested F1) | Part 1 Table 4 (17-cov logit) |
| MS Table 5 | Part 5 Table 5 (attribution consensus) | Part 5 Table 4 (SHAP list; MS Table S13); former Wang table (text only) |
| MS Figure 3 | Part 1 Figure 6 | Part 3 Figure 1 (MS Figure 4); Part 4 Figure 3 (pooled 2×2, exclude) |

**Author decisions (2026-09-17):** Wang integer-score comparison is **text only** (no dedicated table). TSSI leakage is **Methods 4.4 only** (no Results 5.3). The 81 predictor names are **Table S0**.
- Encoder: 106 raw strings → 9 levels. Never quote 99 as `n_raw`.

---

## 1. Title

- **Purpose:** Name the analysis without implying external validation or clinical utility.
- **Exact claims allowed:** Working title only. Study is association + nested-CV prediction + TabPFN attribution on the Wang 2020 **derivation** cohort.
- **Exact values to insert:** none required.
- **YAML path:** `study.title_or_working_title` (status `methods_only`).
- **Source:** `paper_results/00_front_matter.md` — Clinical motivation / What this pack adds.
- **Planned table/figure:** none.
- **Paragraph status:** **incomplete** — authors must finalise wording; YAML title is working-only.

**TODO:** Final journal title. Do not use “validated,” “personalised,” or “external.”

---

## 2. Abstract

### 2.1 Background

- **Purpose:** State the clinical problem and that a published score already exists.
- **Exact claims allowed:** VLST is rare definite stent thrombosis >1 year after PCI. Wang 2020 published an 8-variable Cox integer score on this derivation cohort (published derivation c = 0.80). This analysis uses the same derivation file; it does not replace Wang’s Shantou test.
- **Exact values:** Wang published derivation c `0.80` CI `[0.75, 0.85]`; optional Dangas c `0.66` as **Wang’s published comparison**, not computed here.
- **YAML:** `wang_2020.reported_performance.wang_published_derivation_c`; `wang_2020.reported_performance.dangas_in_wang_comparison` (`methods_only`).
- **Source:** `paper_results/00_front_matter.md` — A score already exists.
- **Planned table/figure:** none in abstract.
- **Status:** **complete** if Dangas is clearly cited-not-computed; else omit Dangas.

### 2.2 Objective

- **Purpose:** State three aims: association catalogue; nested-CV predictive performance without TSSI; TabPFN attribution. Name Wang integer score as historical comparator.
- **Exact claims allowed:** Compare classic ML and two TabPFN arms under nested CV on the derivation cohort; report association (Table 4b); demonstrate TSSI leakage; score frozen Wang points on the same rows.
- **Exact values:** none.
- **YAML:** `study.analysis_setting`; `wang_2020.role_in_current_study`.
- **Source:** `paper_results/00_front_matter.md` — What this pack adds.
- **Planned table/figure:** none.
- **Status:** **complete**.

### 2.3 Methods

- **Purpose:** One-sentence each: cohort, outcome, TSSI drop, nested 5×4 CV, two TabPFN constructors, PR-AUC primary, bootstrap on OOF, Table 4b unweighted logit, attribution 70/30.
- **Exact claims allowed:** n=5185 / 92 events; nested 5 outer / 4 inner, stratified, seed 42; inner loop = F1 threshold only; thinking-high = client; local = `tabpfn`, no `balance_probabilities`; TSSI excluded from primary models (70/30 leakage demonstration: LR PR-AUC 0.9575→0.5077; CatBoost 0.9773→0.6582; SMOTE mismatch named); Wang points frozen (not re-fit).
- **Exact values:** `n_total=5185`; `n_vlst=92`; `event_rate=0.0177`; `validation.outer_cv.n_splits=5`; `inner_cv.n_splits=4`; `bootstrap_method.n_boot=2000`.
- **YAML:** `study.*`; `validation.*`; `model_results.tabpfn_thinking_high.model_definition`; `model_results.tabpfn_local.model_definition`.
- **Source:** `00_front_matter.md`; Part 4 protocol.
- **Planned table/figure:** none.
- **Status:** **incomplete** — TabPFN client/server versions unresolved. Do not invent them. The 81-name list is Table S0.

### 2.4 Results

- **Purpose:** Headline nested-CV ranking + Table 4b examples + Wang frozen PR/ROC as **text**. Keep arms separate. Do **not** put TSSI leakage numbers in the Abstract Results (they belong in Methods).
- **Exact claims allowed:** Thinking-high first on PR-AUC/ROC-AUC/Brier among seven nested-CV models. LightGBM second on PR-AUC. Local fourth on PR-AUC, second on ROC-AUC, Brier in booster band. Nested (not pooled) recall/F1 if space. Table 4b association ORs (not risk factors). One sentence: frozen Wang integer score ROC-AUC 0.8013 / PR-AUC 0.1032 on the same rows (historical comparator, no table).
- **Exact values:**
  - Thinking-high PR-AUC `0.8553` [0.7957, 0.9131]; ROC-AUC `0.9905`; Brier `0.0064`.
  - LightGBM PR-AUC `0.6942` [0.6065, 0.7782].
  - Local PR-AUC `0.6742` [0.5864, 0.7657]; Brier `0.0102`.
  - Δ thinking−LGB PR-AUC `0.1611` (0.0984–0.2289), P(Δ≤0)=0/2000.
  - Nested thinking-high recall `0.7065`, F1 `0.7471` (optional).
  - Wang frozen ROC-AUC `0.8013`, PR-AUC `0.1032` (text only).
  - Table 4b: post-dilation adj OR `0.152` [0.081, 0.286]; Clopidogrel `0.480` [0.293, 0.787]; WBC `1.972` [1.667, 2.331] — “associated with” only.
- **YAML:** `model_results.tabpfn_thinking_high.metrics`; `model_results.tabpfn_local.metrics`; `model_results.baseline_models_without_tssi.lightgbm`; `wang_2020.reported_performance.this_pack_frozen_integer_score`; `statistical_analysis.association_estimates.table4b_adjusted_or`.
- **Source:** Part 4 Tables 1, 2, S-CI, S-Δ; Part 1 Table 4b; Part 4 §7 prose.
- **Planned table/figure:** none in abstract.
- **Status:** **complete**. Do not insert Part 4 Table 3 pooled recall. TSSI scalars: Abstract **Methods**, not Results.

### 2.5 Conclusions

- **Purpose:** Limit to derivation-cohort ranking/association/attribution. State that external or temporal testing of the ML models was not performed.
- **Exact claims allowed:** On this derivation cohort, thinking-high had the highest nested-CV PR-AUC among the seven models; local did not outperform LightGBM on PR-AUC (Δ compatible with no difference). TSSI must not enter binary classifiers. Association findings are not treatment effects. ML models were not externally tested.
- **Exact values:** optional repeat of Δ local−LGB `−0.0201` (−0.0974–0.0566).
- **YAML:** `model_results.tabpfn_local.metrics.delta_pr_auc_vs_lightgbm`; `validation.external_validation.ml_models`.
- **Source:** `00_front_matter.md` limitations 1–2, 4; Part 4 Table S-Δ.
- **Planned table/figure:** none.
- **Status:** **incomplete** — conclusion wording must be authored; values are frozen. **TODO:** do not write “clinically useful,” “validated,” or “ready for use.”

---

## 3. Introduction

- **Purpose:** Motivate VLST after ACS-PCI; cite Wang 2020 as the existing score and this file as its derivation cohort; state what this pack adds (association catalogue, nested-CV comparison of classics + two TabPFN arms after dropping TSSI, attribution, leakage control); state what it does not add (external/temporal ML test; Cox re-fit; decision-curve).
- **Exact claims allowed:** Cohort identity and flow **cited from Wang**, not reconstructed de novo. Wang Shantou c=0.82 / n=2058 is Wang’s test of Wang’s score; file absent. WBC ranking here vs Wang’s exclusion of WBC is a discrepancy to report, not a validated marker. DAPT columns are follow-up persistence.
- **Exact values:** n=5185; 92 events; 1.77%; 6038→5185 (236 / 413 / 204); Wang c=0.80 and Shantou 0.82 (literature); this pack Wang integer ROC-AUC 0.8013 / PR-AUC 0.1032 (preview only, or defer to Results 5.10).
- **YAML:** `study.*`; `wang_2020.*`; `data_and_predictors.post_index_or_time_dependent_variables`.
- **Source:** `00_front_matter.md` Clinical motivation, limitations 1, 6, 7.
- **Planned table/figure:** none required. Optional: none of Part 4 figures in Introduction.
- **Status:** **incomplete** — clinical-burden sentences beyond Wang citation are **TODO** (do not invent incidence outside the freeze). Ethics/NCT may be Methods not Introduction.

---

## 4. Methods

### 4.1 Study design and cohort

- **Purpose:** Retrospective derivation-cohort analysis of `VLST.csv`. Name site, dates, n, flow (cited), ethics (cite Wang), no separate pre-registration.
- **Exact claims allowed:** Consecutive ACS ≥18 years, PCI at First Hospital of Jilin University, 1 Jan 2014 – 1 Jun 2015. Analysed n=5185. Eligible 6038; 236 in-hospital deaths; 413 refused follow-up; 204 lost — **cited from Wang 2020**. This repository analysis was not separately pre-registered. Ethics 2013-256; NCT03491891 — cite Wang. All later analyses use these 5,185 rows; **splits differ**.
- **Exact values:** 5185; 92; 5093; 0.0177; 6038; 236; 413; 204; median follow-up 1502 days; median PCI→VLST 697 days.
- **YAML:** `study.cohort`; `study.n_*`; `study.event_rate`; `study.exclusion_criteria`; `study.median_*`; `study.ethics_and_registration`.
- **Source:** `00_front_matter.md` Outcome; Part 1 Table C caption.
- **Planned table/figure:** manuscript **Table 1** = Part 1 Table C (placed in Results 5.1). No de-novo CONSORT from this repo alone.
- **Status:** **complete** for cited flow and n. **TODO:** do not add extra inclusion rules.

### 4.2 Outcome definition

- **Purpose:** Define binary VLST.
- **Exact claims allowed:** ARC 2007 **definite** stent thrombosis >1 year after implantation, angiographically confirmed. Probable/possible not counted. Column `Stent thrombosis`. Binary classification vs Wang’s Cox time-to-event (limitation, not a result).
- **Exact values:** none beyond event counts already in 4.1.
- **YAML:** `study.outcome`; `study.outcome_definition`.
- **Source:** `00_front_matter.md` Outcome; Part 4 protocol.
- **Planned table/figure:** none.
- **Status:** **complete**.

### 4.3 Candidate predictors

- **Purpose:** Define the 81-column baseline view; dropped IDs and TSSI; stent encoder; post-index drugs; unnamed `LV`/`CaI`.
- **Exact claims allowed:** After dropping `NO.`, `Name`, and TSSI: **81** columns, listed in **Table S0** (CSV header order). Stent `Stent type-SES`: 106 raw strings → 9 levels, `min_count=30`. Part 2 then OHE drop-first → 88 columns. Part 4 classics OHE no drop-first → ~89, fitted in CV. Both TabPFN arms: 9-level native, no scaling. `Aspirin`/`Clopidogrel`/`Ticagrelor`/`DAPT` = follow-up persistence after mandated DAPT year. `LV` unnamed; `CaI` unnamed (means match Wang peak troponin I — Results Table 1, not a new marker).
- **Exact values:** 81 names from `complete_81_name_list`; 106→9; min_count=30; 88; ~89.
- **YAML:** `data_and_predictors.baseline_predictors`; `data_and_predictors.excluded_predictors`; `data_and_predictors.post_index_or_time_dependent_variables`.
- **Source:** `data/raw/VLST.csv` headers; Part 4 protocol; Part 2 header; `paper/table_s0_baseline_predictors.md`.
- **Planned table/figure:** Methods **Table M1** = Part 4 Table 0. **Table S0** = 81 names. Optional Methods box of analysis views (81 vs 88 vs ~89 vs Wang 8 points).
- **Status:** **complete** for the 81-name listing.

### 4.4 TSSI and leakage control

- **Purpose:** Define TSSI as time-at-risk; exclude from primary models; report the 70/30 leakage demonstration **here in Methods** (no Results subsection). Name the SMOTE mismatch.
- **Exact claims allowed:** VLST=1: time to thrombosis, min 380 days (Wang median 697). VLST=0: completed follow-up, min 1241, max 1605; cohort median 1502. Used only in the leakage pair of notebooks. Nested-CV Part 4, Part 2, Part 5 **drop** TSSI. With-TSSI `USE_SMOTE=True`; without `USE_SMOTE=False`. Nested-CV does not use SMOTE. Demonstration PR-AUC: LR 0.9575→0.5077; CatBoost 0.9773→0.6582 (full model list in YAML). Hold-out n=1556.
- **Exact values:** min 380 / 1241 / max 1605; median 1502; 1556; leakage table from YAML.
- **YAML:** `data_and_predictors.tssi_status`; `model_results.tssi_leakage_analysis`.
- **Source:** Part 4 §6; conflict ledger M1.
- **Planned table/figure:** **Table M2** (Methods box) = Part 4 Table S-TSSI. Optional Figure S-TSSI as Methods figure, not Results.
- **Status:** **complete** if SMOTE mismatch is stated. **Blocked** if written as “identical protocol except TSSI” or as nested-CV performance.

### 4.5 Data preprocessing

- **Purpose:** Missingness, imputation (inert), encoding, scaling, fold-wise transformers, stent encoder before split.
- **Exact claims allowed:** EDA printed no missing values; Part 2/4 imputers inert. Part 5 TabPFN-native keeps NaNs; CSV has none. Stent encoder on full frame before Part 4/5 split (`min_count` uses full-cohort frequencies — possible brand-frequency leak, **not quantified**). Classics: imputer/scaler/OHE cloned inside each CV split. Outlier handling: Tukey IQR 1.5 screen in `eda.ipynb` for display only; **no** row drop, winsorisation, or modeling exclusion.
- **Exact values:** none beyond encoder 106→9. Do not quote the EDA IQR count table unless a new exhibit is added from that cell.
- **YAML:** `preprocessing.*`; `data_and_predictors.missingness_summary`.
- **Source:** Part 1 header; Part 4 methods note; conflict ledger M3; `eda.ipynb` IQR cell.
- **Planned table/figure:** optional Methods box. Do not invent a missingness heatmap.
- **Status:** **complete** for outlier *policy* (detection-only). **incomplete** for TSSI `.npy` pipeline (`preprocessing.ipynb`).

### 4.6 Statistical analysis

- **Purpose:** Univariate tests, FDR, unweighted Table 4 vs 4b, EPV, Wald CIs, exploratory interactions.
- **Exact claims allowed:** Welch if abs(skew)≤1 and excess kurtosis≤3, else Mann–Whitney; χ²/Fisher; BH-FDR. Table 4 = 17-cov unidentified, EPV≈5.4, **do not publish as the clinical model**. Table 4b = 13-cov identified unweighted Bernoulli logit, continuous per 1 SD, Wald 95% CI primary, EPV≈7.1 still <10. No `class_weight` on association logits (different from Part 4 LR). Firth = **same 13 covariates**, supplementary; do not replace Table 4b MLE. Table S2: 16 LR interaction tests vs main effects; hypothesis-generating; q<0.05 for LV×eGFR and Men×eGFR only.
- **Exact values:** EPV 92/81≈1.14; 92/17≈5.4; 92/13≈7.1; Table S2 n_pairs=16; LV×eGFR LR 9.81 q=0.0277; Men×eGFR LR 8.53 q=0.028. Firth adj OR: WBC 1.952 [1.657, 2.300]; post-dilation 0.160 [0.087, 0.293]; Clopidogrel 0.491 [0.304, 0.792].
- **YAML:** `statistical_analysis.*`; `firth_logistic_regression`.
- **Source:** Part 1 Tables R, 4, 4b, 4b Firth, S2; front matter W4.
- **Planned table/figure:** Results Table 4b; optional Supp Table 4b Firth and Figure 6.
- **Status:** **complete** for Table 4b/S2/Firth sensitivity. Do not mix Previous PCI OR estimators.

### 4.7 Machine-learning models

- **Purpose:** Name the seven nested-CV arms and constructors. Separate thinking-high and local. Classics = explicit `RUN_MODELS` kwargs + library defaults for unspecified arguments. No unpublished grid.
- **Exact claims allowed:** LR `class_weight=balanced`, `max_iter=1000`, `random_state=42`. RF balanced, `n_jobs=-1`. XGB `eval_metric=aucpr`, `tree_method=hist`, fold `scale_pos_weight=n_neg/n_pos`, GPU `device=cuda` on Tesla T4. LGB `class_weight=balanced`, `metric=average_precision`, `device=gpu`. CatBoost `auto_class_weights=Balanced`, `eval_metric=PRAUC`, `task_type=GPU`. Thinking-high: `tabpfn_client.TabPFNClassifier`, thinking high, metric average_precision. Local Version 4: `from tabpfn import TabPFNClassifier`, `n_estimators=auto`, **no** `balance_probabilities`; restore prior skipped. Do not quote nbdump `balance_probabilities=True` as Version 4. No Part 2/5 mask. Decision Tree / Gaussian NB only in TSSI notebooks, not Part 4. Do not import TSSI GridSearch `max_depth` etc.
- **Exact values:** seeds 42; constructors as YAML `part4_constructors`. Pip versions **unrecorded**.
- **YAML:** `model_results.part4_constructors`; `tabpfn_thinking_high.model_definition`; `tabpfn_local.model_definition`.
- **Source:** Part 4 Table 0; nbdump `RUN_MODELS`; methods inventory F.
- **Planned table/figure:** manuscript Methods **Table M1** = Part 4 Table 0.
- **Status:** **incomplete** — client/server package versions unrecorded. Constructors beyond Table 0 are now methods_only.

### 4.8 Feature selection and feature extraction

- **Purpose:** Part 2 selectors and Part 3 FDR vs ML catalogues **do not feed Part 4**.
- **Exact claims allowed:** Part 2: INNER_VAL 0.2 seed 42; fit 4148 (74 events) / val 1037 (18 events); PR-AUC only; LOCO cap 60; SHAP universe 40; FFS pool 24, max 12, min_gain 0; SMOTE not used; 7×3 intersection 0; scored union 86. Part 3: FDR n=20 (TSSI excluded) vs ML consensus n=13; intersection 5 names; Jaccard 5/28=0.1786. Attribution, not prediction.
- **Exact values:** as above; FFS path lengths lr12 / rf12 / rf_b8 / cat4 / xgb12 / xgb_b11 / lgb5.
- **YAML:** `model_results.feature_selection_models`; `model_results.feature_extraction_models`.
- **Source:** Part 2 header/Figure 1; Part 3 §1–2.
- **Planned table/figure:** Results overlap figure (Part 3 Fig 1); optional Part 2 Fig 1.
- **Status:** **incomplete** — Part 2 long CSVs not in repo (conflict P1); do not reconstruct truncated names beyond freeze.

### 4.9 Nested cross-validation

- **Purpose:** Define the **only** prediction evaluation.
- **Exact claims allowed:** Stratified nested CV, 5 outer / 4 inner, outer shuffle true, `random_state=42`. Events per outer fold 18,18,18,19,19. Inner CV tunes **F1 threshold only**. Single nested CV (not repeated). Preprocessing of classics inside splits; stent encoder before split. This is **not** external validation and **not** a temporal test.
- **Exact values:** 5; 4; 42; fold events.
- **YAML:** `validation.*`.
- **Source:** Part 4 protocol; front matter limitation 1; methods inventory E.
- **Planned table/figure:** none (protocol). Fold PR-AUC = Supplementary Table S-folds in Results.
- **Status:** **complete**.

### 4.10 Performance metrics and uncertainty

- **Purpose:** Primary PR-AUC at 1.77% prevalence; ROC-AUC; Brier; nested Table 2 thresholded metrics; bootstrap CIs; F2 β=2.0. Optional NPV only as TN/(TN+FN) from Table 2 counts.
- **Exact claims allowed:** Ranking = pooled nested OOF (threshold-independent). Quote Table 2 for PPV/recall/spec/F1/F2/accuracy. Table 3 pooled F1 is optimistic — methods contrast only, not headline. Stratified bootstrap of pooled OOF, n_boot=2000, seed 42, keep 92 events and 5093 non-events; models **not** re-fit. ECE / slope / intercept **not reported**. Calibration: quantile-bin reliability curves + Brier. If NPV is mentioned: thinking-high 5076/5103=0.9947 (same identity for other arms in YAML).
- **Exact values:** n_boot=2000; seed 42; F2 beta=2.0; optional `metrics.npv`.
- **YAML:** `metrics.*`; `validation.bootstrap_method`.
- **Source:** Part 4 §3–5; Table 2 2×2.
- **Planned table/figure:** MS Table 3 (ranking) + S-CI; MS Figure 2 calibration. Part 4 Table 3 pooled F1 excluded from headline. No ECE table.
- **Status:** **complete** for named metrics and derived NPV. **incomplete** for ECE.

### 4.11 Model interpretation

- **Purpose:** Part 5 70/30 attribution is **not** nested-CV prediction and **not** a feature mask. SHAP vs k-SII scope.
- **Exact claims allowed:** Split test_size=0.3 seed 42; train 3629/64; held-out 1556/28. MI/SFS/PDP on **train**. SHAP all 1556 held-out rows; client thinking then HTTP 429 ~row 550; finished local + KV cache. k-SII/waterfall/SHAP-IQ: one held-out VLST=1, cohort row 5176, budget 256 — **not** a cohort interaction screen. PDP = train empirical prior near prevalence, **not** Part 4 risk. Local constructors omit `balance_probabilities`.
- **Exact values:** 3629/64; 1556/28; 5176; 256.
- **YAML:** `interpretability.*`; `model_results` does not apply.
- **Source:** Part 5 protocol; front matter limitation 11.
- **Planned table/figure:** Results 5.9 figures 3–13 / Tables 1–5 as selected in the figure plan.
- **Status:** **complete** for protocol. Do not describe as 15+15 SHAP or k-SII rows 5099/5093.

### 4.12 Wang score as a historical comparator

- **Purpose:** Frozen published integer points on the same 5,185 rows. Not a nested-CV arm. Not a Cox re-fit. Not this pack’s external validation.
- **Exact claims allowed:** Eight Wang Table 2 variables/points as in YAML. SES point on `PES`. Four post-dilation points on `No postdilation=1`. Alternate encoding using Wang Table 1’s 14 “No post-dilation” VLST cases yields ROC-AUC 0.5084 — **do not use** as the comparator. Same five outer folds used only to evaluate the frozen score. Shantou is Wang’s external test of Wang’s score; file not in repository.
- **Exact values:** encoding 0.5084 as a **rejected** polarity check; published derivation c 0.80 [0.75, 0.85] cited.
- **YAML:** `wang_2020.*`.
- **Source:** Part 4 §7; `wang_vlst_score.ipynb` provenance only.
- **Planned table/figure:** none (text in Results 5.10 and/or Discussion 6.4). Do not add a dedicated Wang table.
- **Status:** **complete**. **Blocked** if called a validation cohort or an eighth nested-CV model.

---

## 5. Results

### 5.1 Cohort and baseline characteristics

- **Purpose:** Clinical Table 1 (association, not prediction). Cite Wang flow. Do not photocopy Wang’s post-dilation row.
- **Exact claims allowed:** n=5093 vs 92. Table C cells **as printed** (copy, do not recalculate). TSSI omitted from Table 1. DAPT/antiplatelet flags = follow-up persistence. `LV`/`CaI` unnamed. `PES` recovers Wang SES percents. Complementary post-dilation columns shown as stored (14/92 with `1.1:1Post dilation`=1; 78/92 `No postdilation`).
- **Exact values from YAML:** 5185; 92; 5093; 0.0177; 6038; 236; 413; 204; 1502; 697. Table C cell values: insert from Part 1 Table C (YAML `figures_and_tables` Part1-TableC; not re-derived).
- **YAML:** `study.*`; `figures_and_tables` id `Part1-TableC`.
- **Source:** `EDA_paper_figures_and_tables.md` §0 Table C.
- **Planned table/figure:** **Table 1** (Table C). Optional Fig 5 stent 9-level rates (descriptive).
- **Status:** **complete** if copied from Table C. **TODO:** do not invent a repo-only eligibility diagram.

### 5.2 Missingness and preprocessing

- **Purpose:** Report no missing values; encoder 106→9; imputers inert.
- **Exact claims allowed:** No missing values printed in any column. 106 raw stent strings → 9 levels. Figure 5 is the 9-level chart, not 99 or 106 bars.
- **Exact values:** 106; 9; min_count=30. Figure 5 rates as printed in Part 1 caption (`other` 5.5% n=311 … `xx` 0% n=249) — descriptive, from the report caption (also freeze encoder note).
- **YAML:** `data_and_predictors.missingness_summary`; `baseline_predictors.stent_encoder`; exclude `eda_chi2_helper_raw_levels=99`.
- **Source:** Part 1 cohort context; Figure 5.
- **Planned table/figure:** optional **Figure S-stent** = Part 1 Fig 5.
- **Status:** **complete**. **Blocked** if 99 is quoted as `n_raw`.

### 5.3 TSSI leakage analysis

- **Purpose:** **Omitted from Results** (author decision 2026-09-17). All TSSI protocol and demonstration numbers live in **Methods 4.4**.
- **Exact claims allowed:** none in Results. Do not present 70/30 TSSI metrics as nested-CV performance anywhere.
- **Exact values:** none here; see 4.4 / YAML `tssi_leakage_analysis`.
- **YAML:** n/a for this subsection.
- **Source:** Methods 4.4.
- **Planned table/figure:** none in Results (**Table M2** in Methods).
- **Status:** **complete** as an intentional omission.

### 5.4 Statistical associations

- **Purpose:** Univariate FDR + identified Table 4b. Association only.
- **Exact claims allowed:** Quote Table 4b adjusted ORs and Wald CIs (all 13 rows in YAML). Univariate FDR catalogues in Part 1 Tables 1–3 as **associated with** VLST. OR<1 is lower modelled odds of recorded VLST, not treatment benefit. Table 4 unidentified — do not quote 0.144/0.464. S2 interactions hypothesis-generating. EPV 7.1 still below 10.
- **Exact values:** full `table4b_adjusted_or` list; EPV 7.1; S2 two q<0.05 pairs.
- **YAML:** `statistical_analysis.association_estimates`; `epv.table4b`; `likelihood_ratio_tests`.
- **Source:** Part 1 Tables 1–3, 4b, S2; Figure 6.
- **Planned table/figure:** **MS Table 2** = Part 1 Table 4b; **MS Figure 3** uni vs adjusted OR (Part 1 Fig 6). Univariate FDR tables → supplementary.
- **Status:** **complete**. **Blocked** if Table 4 is “the” multivariable model or if ORs are called independent risk factors.

### 5.5 Baseline machine-learning performance

- **Purpose:** Nested-CV classics **without TSSI**, without collapsing in TabPFN. Ranking Table 1 + nested Table 2 + CIs. Primary metric PR-AUC.
- **Exact claims allowed:** LightGBM highest classic PR-AUC 0.6942 [0.6065, 0.7782]; XGBoost 0.6815; CatBoost 0.6172; RF 0.4865; LR 0.3326. ROC-AUC and Brier as frozen. Nested operating points from Table 2 only. Calibration: Brier + Figure 2; do not invent ECE.
- **Exact values:** all five classic `nested_cv_oof` and `nested_operating_point_table2` blocks in YAML.
- **YAML:** `model_results.baseline_models_without_tssi.{logistic_regression,xgboost,lightgbm,catboost,random_forest}`.
- **Source:** Part 4 Tables 1, 2, S-CI; Figures 1–2.
- **Planned table/figure:** **MS Table 3** ranking (all seven models; name arms separately); **MS Figure 1** PR/ROC; **MS Figure 2** calibration; **MS Table 4** nested operating points.
- **Status:** **complete**. Do not use **Part 4 Table 3** pooled F1. NPV optional only as Table 2 identity.

### 5.6 Feature selection and feature extraction

- **Purpose:** Methods-comparison results, not a predictor mask.
- **Exact claims allowed:** Intersection names WBC, eGFR, LV, HbA1c, 1.1:1Post dilation. Jaccard 0.1786 (5/28 ≈ 0.18). ML consensus 13 names as listed. FDR 20 names. 7×3 intersection 0; union 86. FFS path lengths. Do not call this external validation of features.
- **Exact values:** as YAML `feature_extraction_models.results` and `feature_selection_models`.
- **YAML:** those paths.
- **Source:** Part 3 Figure 1 / Table; Part 2 Figure 1, Tables 2–4.
- **Planned table/figure:** **MS Figure 4** overlap (Part 3 Fig 1); optional selector count figure.
- **Status:** **complete** for frozen overlap numbers. **incomplete** for Part 2 truncated HTML names (P1).

### 5.7 TabPFN thinking-high results

- **Purpose:** Nested-CV prediction performance of the **client thinking-high** arm only.
- **Exact claims allowed:** First of seven on PR-AUC 0.8553 [0.7957, 0.9131], ROC-AUC 0.9905 [0.9834, 0.9964], Brier 0.0064 [0.0052, 0.0077] (best of seven). Fold PR-AUC 0.8640, 0.7837, 0.7407, 0.9497, 0.9061; higher than LightGBM in 5/5 folds. Nested t 0.271±0.067; PPV 0.7927; recall 0.7065; spec 0.9967; F1 0.7471; F2 0.7222; 5076/17/27/65. Δ vs LightGBM +0.1611 (0.0984–0.2289), P=0/2000. Not external validation. Not clinical utility.
- **Exact values:** YAML `tabpfn_thinking_high.metrics` (exclude Part 4 Table 3 pooled F1).
- **YAML:** that block; `metrics.pr_auc`.
- **Source:** Part 4 Tables 1, 2, S-CI, S-Δ, S-folds.
- **Planned table/figure:** same ranking/operating-point tables as 5.5, **row labelled thinking-high**.
- **Status:** **complete**. **Blocked** if pooled recall 0.8152 is quoted as nested, or if collapsed with local.

### 5.8 TabPFN local results

- **Purpose:** Nested-CV performance of **local** `tabpfn` (no thinking, no `balance_probabilities`).
- **Exact claims allowed:** PR-AUC 0.6742 [0.5864, 0.7657] (fourth); ROC-AUC 0.9845 [0.9760, 0.9917] (second); Brier 0.0102 [0.0092, 0.0113] (booster band: XGB 0.0088, LGB 0.0093, Cat 0.0101). “Worst Brier” is **false**. Fold PR-AUC 0.6384, 0.6353, 0.5829, 0.7274, 0.7855; higher than LGB in 2/5 (folds 3 and 5). Nested t 0.166±0.020; PPV 0.5478; recall 0.6848; spec 0.9898; F1 0.6087; 5041/52/29/63. Δ vs LGB −0.0201 (−0.0974–0.0566), compatible with no difference. Exclude historical Brier 0.0673 / t 0.915.
- **Exact values:** YAML `tabpfn_local.metrics` (exclude `historical_excluded` and Part 4 Table 3 pooled F1).
- **YAML:** that block.
- **Source:** Part 4 Tables 1, 2, S-CI, S-Δ.
- **Planned table/figure:** same as 5.7, **row labelled local**.
- **Status:** **complete**. **Blocked** if merged with thinking-high or if 0.0673 is quoted as this run.

### 5.9 Model interpretation

- **Purpose:** Attribution ranks and plots. Not prediction. Not a Part 4 mask.
- **Exact claims allowed:** Train MI top CaI 0.022005; Cre train MI 0.000000. Stability: WBC 10/10; Cre and LV 8/10; eGFR 7/10. Held-out mean |SHAP| top: eGFR 1.0439, WBC 1.0202, LV 0.8695. Consensus 3/3: WBC, LV, eGFR. Binary PDP ΔP Previous PCI +0.0137; post-dilation −0.0093 (train empirical prior, not Part 4 risk). k-SII = one row 5176.
- **Exact values:** as YAML `interpretability.*`, including `cre_heldout_mean_abs_shap=0.2449` (Part 5 Table 4 rank 7). Do not use 0.158.
- **YAML:** `interpretability.shap`; `feature_rankings`; `shapiq`; `ffs`.
- **Source:** Part 5 Tables 1–5, Figures 1–13.
- **Planned table/figure:** **MS Table 5** consensus; **MS Figure 5** SHAP bar; optional PDP; k-SII as **supplementary one-row illustration**.
- **Status:** **complete**. Do not treat consensus as a Part 4 mask.

### 5.10 Historical comparison with Wang 2020

- **Purpose:** Frozen integer score vs nested-CV ranking on the **same derivation rows**, **in prose only** (no dedicated table).
- **Exact claims allowed:** Full-cohort ROC-AUC 0.8013, PR-AUC 0.1032. Fold-mean ROC 0.8005±0.0607, PR 0.1134±0.0518 if needed in one sentence. Published derivation c 0.80 cited. Nested-CV thinking-high / LightGBM / local PR-AUC still higher than frozen-score PR-AUC **on this file** — derivation-cohort ranking comparison only. Optional one sentence on bins: intermediate n=**1577** on this file vs Wang printed 1837. Do not include S-Wang as an exhibit.
- **Exact values:** YAML `wang_2020.reported_performance`.
- **YAML:** that block; `external_validation.this_pack_ml: false`.
- **Source:** Part 4 §7 prose (tables exist in the report pack but are **not** manuscript exhibits).
- **Planned table/figure:** **none**.
- **Status:** **complete**. **Blocked** if called external validation or if intermediate n is 1837 for **this CSV**.

---

## 6. Discussion

### 6.1 Principal findings

- **Purpose:** Restate frozen headlines with correct scope.
- **Exact claims allowed:** Thinking-high highest nested-CV PR-AUC/Brier among seven; local not superior to LightGBM on PR-AUC; TSSI inflates 70/30 ranking and was excluded; Table 4b associations (WBC, eGFR, LV, post-dilation, Previous PCI, Clopidogrel) are adjusted associations, not risk factors; Wang frozen score recovers published derivation c on ROC-AUC but PR-AUC is low at 1.77% prevalence; attribution consensus WBC/LV/eGFR.
- **Exact values:** reuse Results inserts; do not add new calculations.
- **YAML:** same as Results 5.4–5.10.
- **Source:** freeze_manifest §2; front matter “What this pack adds.”
- **Planned table/figure:** none new.
- **Status:** **complete** if wording stays within freeze. **Blocked** if “TabPFN was validated” or arms collapsed.

### 6.2 Interpretation of predictive performance

- **Purpose:** PR-AUC vs ROC-AUC at 1.77% prevalence; nested vs apparent F1; unmatched tuning; two TabPFN objects; thin events per fold (18–19).
- **Exact claims allowed:** ROC-AUC is less informative than PR-AUC here. Nested F1 < pooled F1 (optimistic). Classics untuned defaults vs thinking-high vs local — unmatched effort (limitation 8). Accuracy high because negatives dominate. Local Brier in booster band, not worst.
- **Exact values:** prevalence 0.0177; fold events 18–19; nested vs pooled recalls only if clearly labelled as methods contrast (pooled excluded from headline).
- **YAML:** `metrics.*`; `study.event_rate`; `validation.outer_cv.events_per_fold`.
- **Source:** Part 4 methods notes; front matter limitations 4, 8, 9.
- **Planned table/figure:** none.
- **Status:** **complete**.

### 6.3 TSSI leakage and time-at-risk

- **Purpose:** Explain why binary classification cannot use follow-up time as a covariate; contrast with Wang Cox time axis.
- **Exact claims allowed:** Mixed definitions (time-to-event vs completed follow-up). Leakage demo with SMOTE caveat. Primary models drop TSSI.
- **Exact values:** optional LR/CatBoost PR-AUC pair.
- **YAML:** `tssi_status`; `tssi_leakage_analysis`.
- **Source:** Part 4 §6; front matter limitation 2.
- **Planned table/figure:** refer to 5.3.
- **Status:** **complete**.

### 6.4 Relationship to prior work

- **Purpose:** Position against Wang 2020 and Dangas **as cited**.
- **Exact claims allowed:** This pack does not re-fit the Cox linear predictor and does not score Shantou. Integer points on the derivation file recover c≈0.80. Dangas c=0.66 is Wang’s published comparison, not computed here. WBC discrepancy vs Wang’s exclusion. “No VLST score exists” is false.
- **Exact values:** 0.8013 vs 0.80; Shantou 0.82 n=2058 (literature); Dangas 0.66 (literature).
- **YAML:** `wang_2020`.
- **Source:** `00_front_matter.md`; `unsupported_claims.md` “No VLST score exists.”
- **Planned table/figure:** none.
- **Status:** **incomplete** — additional literature beyond freeze/reports is **TODO** (do not invent citations/numbers).

### 6.5 Strengths

- **Purpose:** Only strengths the files support.
- **Exact claims allowed:** Explicit TSSI leakage control; nested CV with honest nested threshold; two TabPFN arms reported separately; identified Table 4b vs unidentified Table 4; frozen Wang comparator; bootstrap CIs on stored OOF; no missing data in file.
- **Exact values:** none required.
- **YAML:** methods_only / frozen protocol fields.
- **Source:** front matter W1–W5; freeze_manifest.
- **Planned table/figure:** none.
- **Status:** **complete**. Do not list “external validation” or “clinical utility” as strengths.

### 6.6 Limitations

- **Purpose:** Copy freeze limitations; do not soften.
- **Exact claims allowed (must include):**
  1. No external or temporal test of ML models (nested CV on 5185 rows).
  2. Binary label vs Wang Cox.
  3. EPV 7.1 / 5.4; Table 4 unidentified.
  4. Two TabPFN calibrations; client non-determinism (historical Brier 0.0060/0.0360 vs this 0.0064 — other dumps, not this freeze’s local 0.0102).
  5. Unrecorded TabPFN versions.
  6. DAPT columns post-baseline.
  7. WBC vs Wang exclusion.
  8. Unequal tuning; stent encoder before split (possible brand-frequency leak, unquantified).
  9. Bootstrap does not re-fit models.
  10. `LV`/`CaI` unnamed.
  11. Part 5 ≠ Part 4 predictor; k-SII one row; PDP scale.
  12. SMOTE mismatch in TSSI pair.
  13. Single nested CV, not repeated.
  14. Firth unused; ECE uncomputed; TabPFN pip versions unrecorded; NPV only as Table 2 identity if quoted.
- **Exact values:** as listed in YAML/front matter.
- **YAML:** `study.analysis_setting`; `validation.external_validation`; `preprocessing.unknown_or_unreported_items`; `provenance.unresolved_items`.
- **Source:** `00_front_matter.md` limitations; conflict ledger M1, M3; `unsupported_claims.md`.
- **Planned table/figure:** none.
- **Status:** **complete** as a checklist. Prose still to be authored.

### 6.7 Implications for future validation

- **Purpose:** State what would be required next. Do not claim it was done.
- **Exact claims allowed:** A held-out temporal or geographically distinct cohort **that this pack does not contain** is required before any transportability claim. Re-fit or externally test the Cox linear predictor and decision-curve vs Dangas were not performed. Do not treat nested CV as a substitute.
- **Exact values:** none.
- **YAML:** `validation.external_validation.ml_models.value = false`.
- **Source:** front matter limitation 1; Part 3 practical reading (held-out/external cohort this pack does not contain).
- **Planned table/figure:** none.
- **Status:** **complete** as implication-only. **Blocked** if written as if validation were completed.

---

## 7. Conclusion

- **Purpose:** Three frozen facts + one negative: no external ML validation.
- **Exact claims allowed:** On the Wang 2020 derivation cohort (n=5185, 92 events), nested-CV TabPFN thinking-high had the highest PR-AUC among the seven models; TabPFN local did not outperform LightGBM on PR-AUC; TSSI is not a legitimate baseline covariate for this binary task; Table 4b reports associations, not causal effects; Wang’s integer score is a historical comparator, not a validation of these models.
- **Exact values:** optional 0.8553 / 0.6942 / 0.6742 / 0.8013 / 0.1032.
- **YAML:** same headline paths as Abstract 2.4–2.5.
- **Source:** freeze_manifest §8.
- **Planned table/figure:** none.
- **Status:** **incomplete** — sentence craft is TODO; numbers are frozen. Prohibit clinical-utility close.

---

## Cross-cutting TODOs (do not fill from assumptions)

| ID | Gap | Sections affected |
| --- | --- | --- |
| TODO-NPV | Optional: NPV as TN/(TN+FN) from Table 2; not a notebook print | 4.10, 5.5 |
| TODO-FIRTH | Firth is Table 4b sensitivity — do not put it in nested CV | 4.6 |
| TODO-OUTLIER | IQR screen is detection-only; no exclusion table unless new exhibit | 4.5 |
| TODO-HP | Explicit RUN_MODELS kwargs are methods_only; do not invent RF depth etc. | 4.7 |
| TODO-PREPROC-NPY | TSSI `preprocessing.ipynb` | 4.4–4.5 |
| TODO-TABPFN-VER | Client/server pip versions | 4.7, 6.6 |
| TODO-ECE | Calibration slope / ECE — leave unreported | 4.10, 5.5 |
| TODO-TITLE | Final title | 1 |
| TODO-LIT | Extra citations beyond Wang 2020 | 3, 6.4 |

**Registry safety:** Safe for drafting if this outline is followed. Not safe if excluded dumps, **Part 4 Table 3** pooled F1, **Part 1 Table 4** unidentified ORs, encoder 99, or external-validation wording are used.
