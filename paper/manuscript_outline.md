# Manuscript outline (blueprint only — no drafted prose)

**Audit re-sync (2026-09-22).** File inventory and section→notebook mapping only. **Do not draft Results/Methods sentences here.**  
**Freeze:** `paper/frozen_results.yaml` (`freeze.date = 2026-09-22`), `freeze.status = READY_FOR_MANUSCRIPT_DRAFTING`, `freeze.blocked = false`.

**Completeness codes**

| Code | Meaning |
| --- | --- |
| **complete** | Frozen values exist; a paragraph can be written without invention |
| **incomplete** | Allowed claim exists, but at least one insert is `unresolved`, `methods_only` without a scalar, or a named TODO |
| **blocked** | Would require merging conflicting quantities, unsupported wording, or an excluded number |

No subsection is **blocked** if the freeze rules below are followed. Several subsections are **incomplete** because of documented gaps (Firth, ECE, TSSI `.npy` pipeline, unlabeled v3 vs v3.5 in the 2026-09-18 nested dump). NPV may be quoted only as TN/(TN+FN) from Table 2.

**Global restrictions (every section)**

- Nested CV ≠ external validation. Derivation file ≠ external validation cohort.
- Wang 2020 = **historical comparator** on the same 5,185 rows. Shantou c = 0.82 describes **Wang’s score only**.
- Associations: “associated with.” Never “independent risk factor,” “causal,” “protective,” “clinically useful,” “validated,” “personalised.”
- Prediction: “model discrimination,” “ranking performance,” “calibration,” “predictive performance.”
- TabPFN thinking ≠ TabPFN local (no-thinking). Never collapse Brier/PR-AUC/ROC-AUC. Never collapse **v3** vs **v3.5**.
- TSSI (`Time since stent implantation`) is leakage-sensitive. Nested-CV / Part 2 / Part 5 drop TSSI (and, on the live dumps, `WBC`). The **70/30 leakage contrast** is Results **5.3**, not nested-CV performance.
- Quote **Part 4 Table 2** nested operating points, not **Part 4 Table 3** pooled F1.
- Quote **Part 1 Table 4b**, not **Part 1 Table 4**, as the identified association logit.
- **Do not cite ARCHIVED artifacts** (inventory below).

**Numbering aliases (always prefix)**

| Manuscript ID (this blueprint) | Report asset | Never confuse with |
| --- | --- | --- |
| MS Table 1 | Part 1 Table C | — |
| MS Table 2 | Part 1 Table 4b | Part 1 Table 4 (excluded); Part 4 Table 2 |
| MS Table 3 | Part 4 Table 1 + S-CI | Part 4 Table 3 (pooled F1, excluded from headline) |
| MS Table 4 | Part 4 Table 2 (nested F1) | Part 1 Table 4 (17-cov logit) |
| MS Table 5 | Part 5 Table 5 (attribution consensus) | Part 5 Table 4 (SHAP list; MS Table S13); former Wang table (text only) |
| MS Figure 3 | Part 1 Figure 6 | Part 3 Figure 1 (MS Figure 4); Part 4 Figure 3 (pooled 2×2, exclude) |
| MS Table M2 / Results 5.3 | 70/30 leakage pair `test_metrics.csv` | Nested-CV Part 4 tables |

**Author decisions (2026-09-17, patched 2026-09-22):** Wang integer-score comparison is **text only** (no dedicated table). The 81-name baseline list is **Table S0**. Encoder: 106 raw strings → 9 levels. Never quote 99 as `n_raw`.  
**Superseded 2026-09-17 decision:** “TSSI leakage is Methods 4.4 only (no Results 5.3)” — **withdrawn**. Leakage contrast is a **Results 5.3** axis (twin 70/30 GridSearch notebooks). Methods 4.4 keeps protocol only.

---

## 0. File inventory (live vs ARCHIVED)

Numerical inserts: `paper/frozen_results.yaml` + the Kaggle dump CSVs named in the mapping. Protocol: the **live** notebooks. Part-folder Markdown under `paper_results/0{1-5}_*/` is a restyle of those dumps, not a second freeze.

### 0.1 ARCHIVED — do not cite

| Artifact | Why archived |
| --- | --- |
| `paper_results/paper_results.md` (concat, **no** `-(2)` suffix) | Unsuffixed concat pack. Do not cite. |
| `docs/paper_evidence_map.md` (**no** `-(2)` suffix) | Unsuffixed evidence map. Do not cite. |
| `code/modeling/interpretability/tabpfn_interpretability.ipynb` | Monolithic parent. Split into `tabpfn_interpretability_fs_pdp.ipynb` + `tabpfn_interpretability_shap.ipynb`. |
| Prior-freeze **citations of** `baseline_plus_tabpfn.ipynb` (thinking PR-AUC **0.8553**, local **0.6742** / Brier **0.0102**, unlabeled checkpoint) | Superseded by `Kaggle_baseline_plus_tabpfn_results/` (papermill 2026-09-18). On-disk notebook is the **live protocol**; those old numbers are not. |
| Prior-freeze **citations of** `baseline_feature_selections.ipynb` (parked 70/30 test, 88 columns, WBC-in-consensus n=13, Jaccard 5/28) | Superseded by `Kaggle_baseline_intrepretability_results/` (`model_feature_selectors_antileak/`, 2026-09-19). On-disk notebook is the **live protocol**; those old catalogues are not. Cached CSVs from the test-scored export are **leaky — do not resume**. |

The two notebooks in the last rows remain **live protocol files** (see 0.2). What is ARCHIVED is citing them the 2026-09-17 way (unsuffixed reports / unlabeled dumps).

### 0.2 Live notebooks

| Notebook | Role | Dump / outputs |
| --- | --- | --- |
| `code/analyzes/eda.ipynb` | Univariate FDR, Table C, Table 4/4b, stent χ². Logic **unchanged**; local venv vs Kaggle is infrastructure only. | `paper_results/01_eda/` (keep this blueprint section as-is) |
| `code/modeling/rating/baseline_tssi_leakage.ipynb` | Twin **ALL LEAKS ON**: `KEEP_TSSI=True`, `DROP_WBC=False`, `QUANTIZE_CLINICAL=False`, `STENT_ENCODER_TRAIN_ONLY=False`. 7-model GridSearchCV. `USE_SMOTE=True`. Stratified 70/30, 3629/1556, seed 42. | `code/modeling/rating/Kaggle_baseline_tssi_leakage_results/` |
| `code/modeling/rating/baseline_without_tssi.ipynb` | Twin **ALL LEAKS OFF**: `KEEP_TSSI=False`, `DROP_WBC=True`, `QUANTIZE_CLINICAL=True`, `STENT_ENCODER_TRAIN_ONLY=True`. Same 7 models, same split. `USE_SMOTE=False`. | `code/modeling/rating/Kaggle_baseline_without_tssi_results/` |
| `code/modeling/rating/baseline_plus_tabpfn.ipynb` | Canonical **nested 5×4** modeling. Pins `TABPFN_PIN="9.0.0"` (first release exposing v3.5 weights) and `TABPFN_CLIENT_PIN="0.6.0"` (hosted v3.5). Classics = library defaults + class weighting; **GridSearchCV winners from the twin notebooks are NOT imported**; inner loop = **F1 threshold only**. Drops `NO.`, `Name`, TSSI, `WBC`. Notebook `RUN_MODELS` requests local v3 + local v3.5 + thinking v3 + thinking v3.5. | `code/modeling/rating/Kaggle_baseline_plus_tabpfn_results/` (papermill 2026-09-18; CSV arms labeled only `TabPFN thinking mode` and `TabPFN`) |
| `code/modeling/interpretability/tabpfn_interpretability_fs_pdp.ipynb` | Part 5 FS half: MI, stability SFS, PDP, consensus. **0 client fits.** `FS_THINKING_MODE=False`, `PDP_USE_CLIENT=False`, local `TABPFN_MODEL_VERSION="v3.5"`. `STABILITY_N_SEEDS=8` (10-seed plan abandoned; Kaggle 9h limit). | `.../Kaggle_tabpfn_intrepretebility_results/fs_pdp_MI/` |
| `code/modeling/interpretability/tabpfn_interpretability_shap.ipynb` | Part 5 SHAP half: held-out SHAP (shapiq), SHAP-IQ, k-SII/waterfall. Client thinking fits (`INTERP_THINKING_MODE=True`, effort high). Same 70/30, v3.5, anti-leakage as fs_pdp. Source still prints `STABILITY_N_SEEDS=10` but **does not run SFS**. | `.../Kaggle_tabpfn_intrepretebility_results/shap/` |
| `code/modeling/interpretability/baseline_feature_selections.ipynb` | Part 2 selectors. Never reads a parked 70/30 test. Fit on train / score on `INNER_VAL_SIZE=0.2`. Writes `model_feature_selectors_antileak/`. `USE_CACHE=False`. Drops TSSI+WBC. | `.../Kaggle_baseline_intrepretability_results/baseline_interpretability_results/model_feature_selectors_antileak/` |
| `code/modeling/rating/wang_vlst_score.ipynb` | Frozen Wang integer score (historical comparator). | Part 4 S-Wang restyle |
| `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb` | Part 3 overlap **code**. Still hardcodes ML-13 / Jaccard 5/28 — **do not execute for numbers**. Live overlap = dump + `rebuild_part3_paper_figures.py`. | — |

Also present, not a paper results axis: `code/modeling/preprocessing/preprocessing.ipynb` (TSSI `.npy`; TODO-PREPROC-NPY). D1 `failed_hypothesis/**` out of scope.

### 0.3 Section → notebook mapping

| Blueprint section | Live notebook(s) | Notes |
| --- | --- | --- |
| 4.1–4.2, 4.5–4.6, 5.1–5.2, 5.4 | `eda.ipynb` | **Unchanged** (venv vs Kaggle only). |
| 4.3 (81-name list / encoder) | `eda.ipynb` + CSV headers | Part 2/4/5 widths differ after anti-leakage. |
| 4.4 protocol; **5.3 Results leakage contrast** | `baseline_tssi_leakage.ipynb` + `baseline_without_tssi.ipynb` | NEW results axis. Not nested CV. |
| 4.7, 4.9–4.10, 5.5, 5.7, 5.8 | `baseline_plus_tabpfn.ipynb` | Nested CV. See TabPFN tags. |
| 4.8, 5.6 | `baseline_feature_selections.ipynb` (+ Part 3 rebuild, not the stats notebook asserts) | Independent LOCO/SHAP/FFS. |
| 4.11, 5.9 MI/SFS/PDP | `tabpfn_interpretability_fs_pdp.ipynb` | Local v3.5; 8/8 SFS. |
| 4.11, 5.9 SHAP / k-SII | `tabpfn_interpretability_shap.ipynb` | Client thinking v3.5. |
| 4.12, 5.10 | `wang_vlst_score.ipynb` | Historical comparator. |

### 0.4 TabPFN number tags

Every TabPFN scalar in this outline must carry **checkpoint** (v3 / v3.5 / AMBIGUOUS), **thinking** (thinking high / local no-thinking), **anti-leakage** (ON = TSSI+WBC dropped + quantize + stent train-only or fold-only / OFF = twin ALL LEAKS ON / AMBIGUOUS).

| Scalar (this freeze / dump) | Checkpoint | Thinking | Anti-leakage |
| --- | --- | --- | --- |
| Nested thinking v3.5 PR-AUC **0.9212** / ROC 0.9963 / Brier 0.0047 / nested 5083/10/15/77 | v3.5_default hosted | thinking high (`thinking_effort=high`, metric average_precision) | ON |
| Nested TabPFN v3.5 PR-AUC **0.8957** / ROC 0.9916 / Brier 0.0048 / nested 5082/11/20/72 | local `tabpfn-v3.5-20260909.safetensors` | local / no-thinking | ON |
| Nested thinking v3 **0.8319** / TabPFN v3 **0.7150** | v3_default / v3 ckpt | thinking vs local | ON |
| Unlabeled dump thinking **0.9771** / local **0.9635** | unlabeled two-arm dump | thinking vs local | excluded (not the 9-arm dump) |
| Outline 2026-09-17 thinking **0.8553** / local **0.6742** / Brier **0.0102** | **AMBIGUOUS — re-source** (unlabeled pre-pin dump; ARCHIVED) | thinking vs local as labeled then | AMBIGUOUS |
| Part 5 SFS 8/8 `{CaI, LV, eGFR}`; train MI CaI **0.020536** | v3.5 local (`FS_THINKING_MODE=False`) | no-thinking (0 client fits) | ON |
| Part 5 held-out mean \|SHAP\| eGFR **1.2288** / CaI **1.0867** | v3.5 | thinking high (client) | ON |
| Part 5 10/10 WBC; CaI MI 0.022005; SHAP eGFR 1.0439 / WBC 1.0202 | **AMBIGUOUS — re-source** (monolithic notebook era; ARCHIVED) | mixed | WBC still in matrix |

Pin on live interp + nested notebooks: `tabpfn==9.0.0`, `tabpfn-client==0.6.0`.

---

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

- **Purpose:** State three aims: association catalogue; nested-CV predictive performance without TSSI (classics vs TabPFN, v3 vs v3.5, thinking vs local); 70/30 leakage contrast (ALL LEAKS ON vs OFF); TabPFN attribution on the two interpretability notebooks. Name Wang integer score as historical comparator.
- **Exact claims allowed:** Nested CV on the derivation cohort after anti-leakage drops; 70/30 GridSearch twins are a **separate** leakage axis, not nested-CV performance; score frozen Wang points on the same rows.
- **Exact values:** none.
- **YAML:** `study.analysis_setting`; `wang_2020.role_in_current_study`.
- **Source:** live notebooks in §0.2 (not unsuffixed `paper_results.md`).
- **Planned table/figure:** none.
- **Status:** **complete**.

### 2.3 Methods

- **Purpose:** One-sentence each: cohort, outcome, TSSI/WBC drop in nested CV, nested 5×4 CV, TabPFN pins, PR-AUC primary, bootstrap on OOF, Table 4b unweighted logit, attribution 70/30 split across **two** interpretability notebooks.
- **Exact claims allowed:** n=5185 / 92 events; nested 5 outer / 4 inner, stratified, seed 42; inner loop = F1 threshold only; classics = library defaults + class weighting (GridSearchCV winners **not** imported from the twin notebooks); thinking = `tabpfn_client` effort high; local = `tabpfn`, no thinking; pins `tabpfn==9.0.0` / `tabpfn-client==0.6.0`; nested-CV anti-leakage ON (TSSI+WBC dropped). Leakage demonstration is the 70/30 twins (Results 5.3), not nested CV.
- **Exact values:** `n_total=5185`; `n_vlst=92`; `event_rate=0.0177`; `validation.outer_cv.n_splits=5`; `inner_cv.n_splits=4`; `bootstrap_method.n_boot=2000`; `TABPFN_PIN=9.0.0`; `TABPFN_CLIENT_PIN=0.6.0`.
- **YAML:** `study.*`; `validation.*`; `model_results.tabpfn_thinking_high.model_definition`; `model_results.tabpfn_local.model_definition`.
- **Source:** `baseline_plus_tabpfn.ipynb` protocol cells; dump `Kaggle_baseline_plus_tabpfn_results/`.
- **Planned table/figure:** none.
- **Status:** **complete** for 9-arm dump (v3 vs v3.5 split). The 81-name list is Table S0.

### 2.4 Results

- **Purpose:** Headline nested-CV ranking + Table 4b examples + Wang frozen PR/ROC as **text**. Keep thinking vs local separate. Leakage scalars belong in Results **5.3**, not Abstract Results.
- **Exact claims allowed:** On the 9-arm anti-leakage nested dump, thinking v3.5 first on PR-AUC among **nine** CSV arms. XGBoost highest classic PR-AUC (0.6322), then LightGBM (0.6271). TabPFN v3.5 second overall. Nested (not pooled) recall/F1 if space. Table 4b association ORs (not risk factors). One sentence: frozen Wang integer score ROC-AUC 0.8013 / PR-AUC 0.1032 (historical comparator, no table).
- **Exact values:**
  - Thinking v3.5 PR-AUC `0.9212` [0.8785, 0.9613]; ROC-AUC `0.9963`; Brier `0.0047`. **Tags:** thinking high; anti-leakage ON; hosted v3.5_default.
  - TabPFN v3.5 PR-AUC `0.8957` [0.8446, 0.9426]; Brier `0.0048`. **Tags:** local no-thinking; anti-leakage ON; v3.5 safetensors.
  - XGBoost PR-AUC `0.6322`; LightGBM `0.6271` (nested dump; not TabPFN).
  - Δ thinking v3.5−LGB PR-AUC `0.2941` (0.2071–0.3805), P(Δ≤0)=0/2000.
  - Nested thinking v3.5 recall `0.8370`, F1 `0.8603`, 5083/10/15/77 (optional).
  - **ARCHIVED — do not insert:** unlabeled 0.9771 / 0.9635; thinking 0.8553; local 0.6742 / Brier 0.0102; Δ 0.1611.
  - Wang frozen ROC-AUC `0.8013`, PR-AUC `0.1032` (text only; not TabPFN).
  - Table 4b: post-dilation adj OR `0.152` [0.081, 0.286]; Clopidogrel `0.480` [0.293, 0.787]; WBC `1.972` [1.667, 2.331] — “associated with” only (EDA; keep).
- **YAML:** `nested_cv_v35_antileakage_on`; `kaggle_baseline_plus_tabpfn`.
- **Source:** `Kaggle_baseline_plus_tabpfn_results/baseline_plus_tabpfn_results/.../model_comparison.csv` + nested operating-point CSV.
- **Planned table/figure:** none in abstract.
- **Status:** **complete** for 9-arm numbers. Do not insert Part 4 Table 3 pooled recall.

### 2.5 Conclusions

- **Purpose:** Limit to derivation-cohort ranking/association/attribution. State that external or temporal testing of the ML models was not performed.
- **Exact claims allowed:** On this derivation cohort, thinking v3.5 had the highest nested-CV PR-AUC among the nine CSV arms (0.9212); TabPFN v3.5 0.8957 also exceeds XGBoost 0.6322 / LightGBM 0.6271. TSSI must not enter nested-CV binary classifiers. Association findings are not treatment effects. ML models were not externally tested.
- **Exact values:** optional repeat of Δ local−LGB from YAML `tabpfn_local.metrics.delta_pr_auc_vs_lightgbm` (dump, anti-leakage ON; checkpoint **AMBIGUOUS — re-source**).
- **YAML:** `model_results.tabpfn_local.metrics.delta_pr_auc_vs_lightgbm`; `validation.external_validation.ml_models`.
- **Source:** `Kaggle_baseline_plus_tabpfn_results/` (not unsuffixed concat).
- **Planned table/figure:** none.
- **Status:** **incomplete** — conclusion wording must be authored; dump values are frozen with tags. **TODO:** do not write “clinically useful,” “validated,” or “ready for use.” **ARCHIVED:** Δ `−0.0201` from the 0.6742 local run.

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
- **Exact claims allowed:** After dropping `NO.`, `Name`, and TSSI: **81** columns in the EDA / Table S0 view (CSV header order). Live nested-CV / Part 2 / Part 5 dumps **also drop `WBC`**. Stent `Stent type-SES`: 106 raw strings → 9 levels, `min_count=30`. Part 2 antileak dump: OHE drop-first → **87** columns (80 raw − 1 brand + 8 dummies). Nested-CV TabPFN: 9-level native after TSSI+WBC drop. `Aspirin`/`Clopidogrel`/`Ticagrelor`/`DAPT` = follow-up persistence after mandated DAPT year. `LV` unnamed; `CaI` unnamed (means match Wang peak troponin I — Results Table 1, not a new marker).
- **Exact values:** 81 names from `complete_81_name_list` (EDA view); 106→9; min_count=30; Part 2 scaled **87**; **ARCHIVED:** Part 2 **88**.
- **YAML:** `data_and_predictors.baseline_predictors`; `data_and_predictors.excluded_predictors`; `kaggle_feature_selectors.provenance.anti_leakage`.
- **Source:** `data/raw/VLST.csv` headers; `baseline_plus_tabpfn.ipynb` `DROP_FEATURES`; `split_manifest.json` (Part 2 dump); `paper/table_s0_baseline_predictors.md`.
- **Planned table/figure:** Methods **Table M1** = Part 4 Table 0. **Table S0** = 81 names. Optional Methods box of analysis views (81 EDA vs 87 Part 2 vs nested drop-WBC vs Wang 8 points).
- **Status:** **complete** for the 81-name listing; flag the WBC drop as dump-specific.

### 4.4 TSSI and leakage control

- **Purpose:** Define TSSI as time-at-risk. Protocol of the **twin 70/30 GridSearch notebooks**. Demonstration **numbers** live in Results **5.3**, not here.
- **Exact claims allowed:** VLST=1: time to thrombosis, min 380 days (Wang median 697). VLST=0: completed follow-up, min 1241, max 1605; cohort median 1502. **ALL LEAKS ON** (`baseline_tssi_leakage.ipynb`): `KEEP_TSSI=True`, `DROP_WBC=False`, `QUANTIZE_CLINICAL=False`, `STENT_ENCODER_TRAIN_ONLY=False`, `USE_SMOTE=True`. **ALL LEAKS OFF** (`baseline_without_tssi.ipynb`): inverse flags, `USE_SMOTE=False`. Same 7 models (LR, DT, RF, GNB, CatBoost, XGB, LGBM), same stratified 70/30 (3629/1556, seed 42). Nested-CV / Part 2 / Part 5 **do not** use this GridSearch and **do not** import its `best_params_`. Nested-CV does not use SMOTE. Do not write “identical protocol except TSSI” (WBC, quantize, stent, SMOTE also flip).
- **Exact values:** min 380 / 1241 / max 1605; median 1502; n_train=3629; n_test=1556. PR-AUC pair → Results 5.3.
- **YAML:** `data_and_predictors.tssi_status`; `model_results.tssi_leakage_analysis`; `kaggle_tssi_leakage`; `kaggle_without_tssi`.
- **Source:** the two twin notebooks + `Kaggle_baseline_*_tssi_*_results/` `test_metrics.csv`.
- **Planned table/figure:** protocol box only. **Table of numbers = Results 5.3**.
- **Status:** **complete** if SMOTE and the four flags are stated. **Blocked** if written as nested-CV performance.

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

- **Purpose:** Name nested-CV arms in `baseline_plus_tabpfn.ipynb`. Separate thinking vs local and v3 vs v3.5. Classics = library defaults + class weighting. **GridSearchCV winners from `baseline_without_tssi.ipynb` are NOT imported.** Inner loop = F1 threshold only.
- **Exact claims allowed:** LR `class_weight=balanced`, `max_iter=1000`, `random_state=42`. RF balanced, `n_jobs=-1`. XGB `eval_metric=aucpr`, `tree_method=hist`, fold `scale_pos_weight=n_neg/n_pos`, GPU `device=cuda`. LGB `class_weight=balanced`, `metric=average_precision`, `device=gpu`. CatBoost `auto_class_weights=Balanced`, `eval_metric=PRAUC`, `task_type=GPU`. Pins: `TABPFN_PIN="9.0.0"`, `TABPFN_CLIENT_PIN="0.6.0"`. Notebook `RUN_MODELS`: `TabPFN v3`, `TabPFN v3.5` (local, no thinking) and `TabPFN thinking v3`, `TabPFN thinking v3.5` (`thinking_mode=True`, `thinking_effort=high`, `thinking_metric=average_precision`). 2026-09-18 dump CSV has only `TabPFN thinking mode` and `TabPFN` — **do not invent nested v3 rows**. No Part 2/5 mask. DT / GNB only in the twin 70/30 notebooks.
- **Exact values:** seeds 42; pins 9.0.0 / 0.6.0; `THINKING_KWARGS` as in the notebook.
- **YAML:** `model_results.part4_constructors`; `tabpfn_thinking_high.model_definition`; `tabpfn_local.model_definition`; `kaggle_baseline_plus_tabpfn`.
- **Source:** `baseline_plus_tabpfn.ipynb` (Fit and Eval markdown + `RUN_MODELS`). Dump for fitted arms.
- **Planned table/figure:** manuscript Methods **Table M1** = nested-CV constructors.
- **Status:** **incomplete** for v3 vs v3.5 nested metrics (dump unlabeled). Pins are recorded.

### 4.8 Feature selection and feature extraction

- **Purpose:** Part 2 selectors and Part 3 FDR vs ML catalogues **do not feed Part 4**.
- **Exact claims allowed:** `baseline_feature_selections.ipynb` never reads a parked 70/30 test. Fit on `1-INNER_VAL_SIZE`, score on `INNER_VAL_SIZE=0.2`, seed 42; fit 4148 (74 events) / val 1037 (18 events); PR-AUC only; LOCO cap 60; SHAP universe 40; FFS pool 24, max 12, min_gain 0; SMOTE not used; writes `model_feature_selectors_antileak/`; `USE_CACHE=False`; old cached CSVs **leaky — do not resume**. Selectors independent of each other’s selected names (shared cheap ranking; prefixes 60/40/24). Live dump: TSSI+WBC dropped; scaled 87; 7×3 intersection 0; scored union 86; FFS paths lr12 / rf8 / rf_b6 / cat8 / xgb10 / xgb_b12 / lgb4. Part 3: FDR n=20 vs ML three-way union n=10; intersection 5; Jaccard 5/25=0.20. **ARCHIVED:** n=13, Jaccard 5/28, FFS 12/12/8/4/12/11/5, 88 columns.
- **Exact values:** as above (YAML `feature_selection_models` / `feature_extraction_models` after 2026-09-22 ingest).
- **YAML:** those paths; `kaggle_feature_selectors`.
- **Source:** `baseline_feature_selections.ipynb`; dump `Kaggle_baseline_intrepretability_results/.../model_feature_selectors_antileak/`. Not `stats_vs_ml_comparison.ipynb` asserts. Not unsuffixed evidence map.
- **Planned table/figure:** Results overlap figure (Part 3 Fig 1); optional Part 2 Fig 1.
- **Status:** **complete** for dump catalogues. `stats_vs_ml_comparison.ipynb` remains stale (do not cite).

### 4.9 Nested cross-validation

- **Purpose:** Define the **only** prediction evaluation.
- **Exact claims allowed:** Stratified nested CV, 5 outer / 4 inner, outer shuffle true, `random_state=42`. Events per outer fold 18,18,18,19,19. Inner CV tunes **F1 threshold only**. Single nested CV (not repeated). Classics: imputer/scaler/OHE cloned inside splits; stent-brand collapse fit on each **training fold only** (`baseline_plus_tabpfn.ipynb`). This is **not** external validation and **not** a temporal test. GridSearch from the twin notebooks is not this evaluation.
- **Exact values:** 5; 4; 42; fold events.
- **YAML:** `validation.*`.
- **Source:** `baseline_plus_tabpfn.ipynb` nested cell.
- **Planned table/figure:** none (protocol). Fold PR-AUC = Supplementary Table S-folds in Results.
- **Status:** **complete**.
- **Planned table/figure:** none (protocol). Fold PR-AUC = Supplementary Table S-folds in Results.
- **Status:** **complete**.

### 4.10 Performance metrics and uncertainty

- **Purpose:** Primary PR-AUC at 1.77% prevalence; ROC-AUC; Brier; nested Table 2 thresholded metrics; bootstrap CIs; F2 β=2.0. Optional NPV only as TN/(TN+FN) from Table 2 counts.
- **Exact claims allowed:** Ranking = pooled nested OOF (threshold-independent). Quote Table 2 for PPV/recall/spec/F1/F2/accuracy. Table 3 pooled F1 is optimistic — methods contrast only, not headline. Stratified bootstrap of pooled OOF, n_boot=2000, seed 42, keep 92 events and 5093 non-events; models **not** re-fit. ECE is reported (8 quantile bins); slope / intercept are not. Calibration: quantile-bin reliability curves + Brier. If NPV is mentioned: derive TN/(TN+FN) from **this dump’s** Table 2 (thinking v3.5 5083/5098 ≈ 0.9971). **ARCHIVED:** 5088/5095=0.9986 from the unlabeled 0.9771 dump; 5076/5103=0.9947 from the 0.8553 run.
- **Exact values:** n_boot=2000; seed 42; F2 beta=2.0; optional `metrics.npv`.
- **YAML:** `metrics.*`; `validation.bootstrap_method`.
- **Source:** Part 4 §3–5; Table 2 2×2.
- **Planned table/figure:** MS Table 3 (ranking) + S-CI; MS Figure 2 calibration. Part 4 Table 3 pooled F1 excluded from headline. No ECE table.
- **Status:** **complete** for named metrics and derived NPV. **incomplete** for ECE.

### 4.11 Model interpretation

- **Purpose:** Part 5 70/30 attribution is **not** nested-CV prediction and **not** a feature mask. Two notebooks.
- **Exact claims allowed:** Split test_size=0.3 seed 42; train 3629/64; held-out 1556/28. Same anti-leakage as nested (TSSI+WBC dropped). Pins 9.0.0 / 0.6.0, v3.5.
  - `tabpfn_interpretability_fs_pdp.ipynb`: MI, stability SFS (`STABILITY_N_SEEDS=8`; 10-seed plan abandoned), PDP, consensus ranking. **0 client fits.** `FS_THINKING_MODE=False`, `PDP_USE_CLIENT=False`, local `TABPFN_MODEL_VERSION="v3.5"`.
  - `tabpfn_interpretability_shap.ipynb`: held-out SHAP via shapiq, native SHAP-IQ, k-SII/waterfall. Client thinking (`INTERP_THINKING_MODE=True`, effort high). SHAP all 1556 held-out rows. k-SII/waterfall/SHAP-IQ: one held-out VLST=1, cohort row 5176, budget 256 — **not** a cohort interaction screen.
  - Do **not** cite `tabpfn_interpretability.ipynb`. Shap notebook still prints `STABILITY_N_SEEDS=10` but does not run SFS — ignore that print.
  - PDP = train empirical prior near prevalence, **not** Part 4 risk.
- **Exact values:** 3629/64; 1556/28; 5176; 256; STABILITY_N_SEEDS=8.
- **YAML:** `interpretability.*`; `kaggle_interpretability`.
- **Source:** the two split notebooks + `Kaggle_tabpfn_intrepretebility_results/{fs_pdp_MI,shap}/`.
- **Planned table/figure:** Results 5.9 figures / Tables 1–5 from those dumps.
- **Status:** **complete** for protocol. Do not describe as 15+15 SHAP, 10 SFS seeds, or k-SII rows 5099/5093.

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

### 5.3 Leakage contrast (70/30 GridSearch twins)

- **Purpose:** **Results axis.** Compare ALL LEAKS ON vs ALL LEAKS OFF on the same 7-model GridSearchCV pipeline and the same 3629/1556 split. Not nested CV. Not a TabPFN result.
- **Exact claims allowed:** Quote `test_metrics.csv` from both dumps, paired by model. SMOTE is ON only in the leaks-on arm. Do not call this nested-CV performance. Do not say “identical except TSSI.”
- **Exact values (PR-AUC, 2026-09-19 dumps):**
  - ON (`KEEP_TSSI=True`, `DROP_WBC=False`, quantize off, stent not train-only, SMOTE on): LR 0.913388, DT 0.752385, RF 0.939998, GNB 0.272768, Cat 0.959865, XGB 0.954687, LGB 0.968654.
  - OFF (inverse flags, SMOTE off): LR 0.343057, DT 0.137767, RF 0.487373, GNB 0.056433, Cat 0.494218, XGB 0.568484, LGB 0.667542.
  - **ARCHIVED — do not insert:** LR 0.9575→0.5077; Cat 0.9773→0.6582; GNB “unchanged.”
- **YAML:** `model_results.tssi_leakage_analysis`; `kaggle_tssi_leakage`; `kaggle_without_tssi`.
- **Source:** `baseline_tssi_leakage.ipynb` + `baseline_without_tssi.ipynb`; `Kaggle_baseline_tssi_leakage_results/` and `Kaggle_baseline_without_tssi_results/` `test_metrics.csv`.
- **Planned table/figure:** **MS Results table** = paired 7× metrics. Optional PR bar. Not a Methods-only box.
- **Status:** **complete** for dump PR-AUC. No `best_params_` CSV in either dump (prints live in notebooks only).

### 5.4 Statistical associations

- **Purpose:** Univariate FDR + identified Table 4b. Association only.
- **Exact claims allowed:** Quote Table 4b adjusted ORs and Wald CIs (all 13 rows in YAML). Univariate FDR catalogues in Part 1 Tables 1–3 as **associated with** VLST. OR<1 is lower modelled odds of recorded VLST, not treatment benefit. Table 4 unidentified — do not quote 0.144/0.464. S2 interactions hypothesis-generating. EPV 7.1 still below 10.
- **Exact values:** full `table4b_adjusted_or` list; EPV 7.1; S2 two q<0.05 pairs.
- **YAML:** `statistical_analysis.association_estimates`; `epv.table4b`; `likelihood_ratio_tests`.
- **Source:** Part 1 Tables 1–3, 4b, S2; Figure 6.
- **Planned table/figure:** **MS Table 2** = Part 1 Table 4b; **MS Figure 3** uni vs adjusted OR (Part 1 Fig 6). Univariate FDR tables → supplementary.
- **Status:** **complete**. **Blocked** if Table 4 is “the” multivariable model or if ORs are called independent risk factors.

### 5.5 Baseline machine-learning performance

- **Purpose:** Nested-CV classics **anti-leakage ON** (TSSI+WBC dropped in `baseline_plus_tabpfn.ipynb`), without collapsing in TabPFN. Ranking + nested Table 2 + CIs. Primary metric PR-AUC. Not the 70/30 GridSearch twins (those are 5.3).
- **Exact claims allowed:** XGBoost highest classic PR-AUC 0.6322 [0.5331, 0.7247]; LightGBM 0.6271 [0.5313, 0.7202]; CatBoost 0.5707; RF 0.3506; LR 0.2596 (9-arm anti-leakage dump). Nested operating points from Table 2 only. Calibration: Brier + ECE (8 quantile bins) + Figure 2. Classics are **not** GridSearch winners.
- **Exact values:** five classic rows in `model_comparison.csv` / YAML `baseline_models_without_tssi`.
- **YAML:** `model_results.baseline_models_without_tssi.{logistic_regression,xgboost,lightgbm,catboost,random_forest}`; `kaggle_baseline_plus_tabpfn`.
- **Source:** `Kaggle_baseline_plus_tabpfn_results/.../model_comparison.csv`.
- **Planned table/figure:** **MS Table 3** ranking (name TabPFN arms separately + tags); **MS Figure 1** PR/ROC; **MS Figure 2** calibration; **MS Table 4** nested operating points.
- **Status:** **complete** for classics. Do not use **Part 4 Table 3** pooled F1.

### 5.6 Feature selection and feature extraction

- **Purpose:** Methods-comparison results, not a predictor mask.
- **Exact claims allowed:** Intersection names Clopidogrel, HbA1c, LV, No postdilation, eGFR. Jaccard 5/25 = 0.20. ML consensus n=10 as listed in YAML (no WBC). FDR 20 names. 7×3 intersection 0; union 86. FFS paths lr12/rf8/rf_b6/cat8/xgb10/xgb_b12/lgb4. Dual label: WBC is FDR-only because it is dropped from the ML matrix. **ARCHIVED:** intersection {WBC, eGFR, LV, HbA1c, 1.1:1Post dilation}; Jaccard 5/28; n=13.
- **Exact values:** YAML `feature_extraction_models.results` (2026-09-22 ingest).
- **YAML:** those paths; `kaggle_feature_selectors`.
- **Source:** Part 2 dump CSVs; `rebuild_part3_paper_figures.py`; `stats_vs_ml_comparison.ipynb` (dump-backed, Jaccard 5/25).
- **Planned table/figure:** **MS Figure 4** overlap (Part 3 Fig 1); optional selector count figure.
- **Status:** **complete** for dump overlap.

### 5.7 TabPFN thinking-high results

- **Purpose:** Nested-CV of the **client thinking** arm only. Do not merge with local. Do not invent a v3 thinking row from this dump.
- **Exact claims allowed:** 9-arm dump `TabPFN thinking v3.5`: PR-AUC 0.9212 [0.8785, 0.9613], ROC-AUC 0.9963, Brier 0.0047; nested 5083/10/15/77; recall 0.8370; F1 0.8603; Δ vs LightGBM +0.2941 (0.2071–0.3805), P=0/2000. Fold PR-AUC 0.8802, 0.9078, 0.9434, 0.9858, 0.8961; wins vs LGB 5/5. **Tags:** thinking high; anti-leakage ON; hosted v3.5_default. Not external validation. Not clinical utility.
- **Exact values:** YAML `tabpfn_thinking_high.metrics` / dump (exclude Table 3 pooled F1).
- **YAML:** that block; `kaggle_baseline_plus_tabpfn`.
- **Source:** `Kaggle_baseline_plus_tabpfn_results/.../model_comparison.csv` + nested operating-point CSV.
- **Planned table/figure:** same ranking/operating-point tables as 5.5, **row labelled thinking** + tags.
- **Status:** **complete** for thinking v3.5. Do not quote unlabeled 0.9771. **ARCHIVED:** 0.8553 / 0.9905 / 0.0064 / 5076/17/27/65 / Δ 0.1611.

### 5.8 TabPFN local results

- **Purpose:** Nested-CV of **local** `tabpfn` (no thinking). Do not merge with thinking. Do not invent a v3 local row from this dump.
- **Exact claims allowed:** 9-arm dump `TabPFN v3.5`: PR-AUC 0.8957, ROC-AUC 0.9916, Brier 0.0048; nested 5082/11/20/72; recall 0.7826; F1 0.8229. **Tags:** local no-thinking; anti-leakage ON; v3.5 safetensors. Exclude unlabeled 0.9635; historical Brier 0.0673 / t 0.915 and ARCHIVED 0.6742 / 0.0102 / Δ −0.0201.
- **Exact values:** YAML `tabpfn_local.metrics` (exclude `historical_excluded` and Table 3 pooled F1).
- **YAML:** that block; `kaggle_baseline_plus_tabpfn`.
- **Source:** same dump CSVs as 5.7.
- **Planned table/figure:** same as 5.7, **row labelled local** + tags.
- **Status:** **complete** for TabPFN v3.5. **Blocked** if merged with thinking or if 0.0673 / 0.6742 / 0.9635 is quoted as this run.

### 5.9 Model interpretation

- **Purpose:** Attribution ranks and plots from **two** notebooks. Not prediction. Not a Part 4 mask.
- **Exact claims allowed:**
  - FS (`tabpfn_interpretability_fs_pdp.ipynb`): train MI CaI **0.020536** (80 cols); SFS **8/8** `{CaI, LV, eGFR}`. **Tags:** v3.5 local; no-thinking; anti-leakage ON. Binary PDP Previous PCI Δ **+0.001294** (train prior, not Part 4 risk).
  - SHAP (`tabpfn_interpretability_shap.ipynb`): held-out mean |SHAP| eGFR **1.228766**, CaI **1.086730**, Cre **0.809273**, LV **0.482762**. **Tags:** v3.5; thinking high (client); anti-leakage ON. k-SII = one row **5176**.
  - Consensus 3/3 `{CaI, eGFR, LV}` after merging SHAP into the FS report.
  - **ARCHIVED — do not insert:** CaI MI 0.022005; WBC 10/10; SHAP eGFR 1.0439 / WBC 1.0202 / LV 0.8695; consensus 3/3 {WBC, LV, eGFR}; PDP Previous PCI +0.0137. Checkpoint **AMBIGUOUS — re-source** for those old numbers (monolithic notebook).
- **Exact values:** YAML `interpretability.*` / `kaggle_interpretability` after 2026-09-22 ingest. Do not use 0.158.
- **YAML:** `interpretability.shap`; `feature_rankings`; `shapiq`; `ffs`; `kaggle_interpretability`.
- **Source:** `Kaggle_tabpfn_intrepretebility_results/fs_pdp_MI/` and `.../shap/`. Not `tabpfn_interpretability.ipynb`.
- **Planned table/figure:** **MS Table 5** consensus; **MS Figure 5** SHAP bar; optional PDP; k-SII as **supplementary one-row illustration**.
- **Status:** **complete** for dump. Do not treat consensus as a Part 4 mask.

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
- **Exact claims allowed:** Thinking v3.5 highest nested-CV PR-AUC/Brier among nine CSV arms; TabPFN v3.5 PR-AUC 0.8957 also above XGBoost 0.6322 / LightGBM 0.6271; 70/30 ALL LEAKS ON inflates GridSearch ranking vs ALL LEAKS OFF (Results 5.3); Table 4b associations are adjusted associations, not risk factors; Wang frozen score recovers published derivation c on ROC-AUC but PR-AUC is low at 1.77% prevalence; attribution consensus `{CaI, eGFR, LV}` (v3.5; WBC dropped). **ARCHIVED claims:** unlabeled 0.9771/0.9635; thinking 0.8553; consensus WBC/LV/eGFR.
- **Exact values:** reuse Results inserts; do not add new calculations.
- **YAML:** same as Results 5.3–5.10.
- **Source:** YAML + dumps in §0.2. Not unsuffixed `paper_results.md` / `paper_evidence_map.md`.
- **Planned table/figure:** none new.
- **Status:** **complete** if wording stays within freeze. **Blocked** if “TabPFN was validated” or arms collapsed.

### 6.2 Interpretation of predictive performance

- **Purpose:** PR-AUC vs ROC-AUC at 1.77% prevalence; nested vs apparent F1; unmatched tuning; thinking vs local and (notebook) v3 vs v3.5; thin events per fold (18–19).
- **Exact claims allowed:** ROC-AUC is less informative than PR-AUC here. Nested F1 < pooled F1 (optimistic). Classics = library defaults + class weighting vs thinking vs local — unmatched effort. Accuracy high because negatives dominate. Dump local Brier 0.0025 is next to thinking 0.0023, not in the classic booster band. **ARCHIVED:** “local Brier in booster band” (0.0102 era).
- **Exact values:** prevalence 0.0177; fold events 18–19; nested vs pooled recalls only if clearly labelled as methods contrast (pooled excluded from headline).
- **YAML:** `metrics.*`; `study.event_rate`; `validation.outer_cv.events_per_fold`.
- **Source:** `baseline_plus_tabpfn.ipynb`; dump CSVs.
- **Planned table/figure:** none.
- **Status:** **complete**.

### 6.3 TSSI leakage and time-at-risk

- **Purpose:** Explain why binary classification cannot use follow-up time as a covariate; contrast with Wang Cox time axis. Numbers from Results **5.3**.
- **Exact claims allowed:** Mixed definitions (time-to-event vs completed follow-up). Twin notebooks flip TSSI, WBC, quantize, stent train-only, and SMOTE together. Nested models drop TSSI (and WBC on live dumps).
- **Exact values:** optional dump PR-AUC pair from 5.3 (LR 0.9134→0.3431; Cat 0.9599→0.4942). **ARCHIVED:** 0.9575→0.5077.
- **YAML:** `tssi_status`; `tssi_leakage_analysis`.
- **Source:** twin notebooks + dumps. Refer to 5.3.
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
- **Exact claims allowed:** Explicit leakage contrast (Results 5.3); nested CV with honest nested threshold; thinking vs local reported separately; identified Table 4b vs unidentified Table 4; frozen Wang comparator; bootstrap CIs on stored OOF; no missing data in file. Do not list a completed v3 vs v3.5 nested comparison until the dump splits those arms.
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
  4. Two TabPFN objects (thinking vs local); dump Brier thinking 0.0023 / local 0.0025. **ARCHIVED:** 0.0064 / 0.0102. Historical Brier 0.0060/0.0360/0.0673 excluded.
  5. Nested dump CSV does not label v3 vs v3.5 — **AMBIGUOUS — re-source**. Pins `tabpfn==9.0.0` / `tabpfn-client==0.6.0` are recorded on the live notebooks.
  6. DAPT columns post-baseline.
  7. WBC vs Wang exclusion; live dumps drop WBC from ML views while FDR still lists it.
  8. Unequal tuning (GridSearch not imported into nested CV); stent collapse is train-fold only on the live nested notebook (brand-frequency leak **not quantified**).
  9. Bootstrap does not re-fit models.
  10. `LV`/`CaI` unnamed.
  11. Part 5 ≠ Part 4 predictor; k-SII one row; PDP scale; two interpretability notebooks.
  12. SMOTE ON only in ALL LEAKS ON twin.
  13. Single nested CV, not repeated.
  14. Firth unused; ECE uncomputed; NPV only as Table 2 identity if quoted.
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
- **Exact claims allowed:** On the Wang 2020 derivation cohort (n=5185, 92 events), nested-CV TabPFN thinking v3.5 had the highest PR-AUC among the nine CSV arms; TabPFN v3.5 also exceeds XGBoost and LightGBM; TSSI is not a legitimate nested-CV covariate; Table 4b reports associations, not causal effects; Wang’s integer score is a historical comparator, not a validation of these models.
- **Exact values:** optional tagged dump 0.9212 / 0.6322 / 0.8957 / 0.8013 / 0.1032. **ARCHIVED:** 0.9771 / 0.9635 / 0.8553 / 0.6942 / 0.6742.
- **YAML:** same headline paths as Abstract 2.4–2.5.
- **Source:** dumps + YAML. Not unsuffixed concat.
- **Planned table/figure:** none.
- **Status:** **incomplete** — sentence craft is TODO; numbers are frozen with tags. Prohibit clinical-utility close.

---

## Cross-cutting TODOs (do not fill from assumptions)

| ID | Gap | Sections affected |
| --- | --- | --- |
| TODO-NPV | Optional: NPV as TN/(TN+FN) from **this dump’s** Table 2 | 4.10, 5.5 |
| TODO-FIRTH | Firth is Table 4b sensitivity — do not put it in nested CV | 4.6 |
| TODO-OUTLIER | IQR screen is detection-only; no exclusion table unless new exhibit | 4.5 |
| TODO-HP | Classics = library defaults + class weighting; do not import GridSearch winners | 4.7 |
| TODO-PREPROC-NPY | TSSI `preprocessing.ipynb` | 4.4–4.5 |
| TODO-TABPFN-V3V35 | Nested dump unlabeled for v3 vs v3.5 — **AMBIGUOUS — re-source** | 0.4, 4.7, 5.7, 5.8 |
| TODO-ECE | Calibration slope / ECE — leave unreported | 4.10, 5.5 |
| TODO-TITLE | Final title | 1 |
| TODO-LIT | Extra citations beyond Wang 2020 | 3, 6.4 |

**Registry safety:** Safe for drafting if this outline is followed. Not safe if ARCHIVED artifacts, excluded dumps, **Part 4 Table 3** pooled F1, **Part 1 Table 4** unidentified ORs, encoder 99, unlabeled TabPFN checkpoint, or external-validation wording are used.

**Closed:** TODO-TABPFN-VER (pins `9.0.0` / `0.6.0` on live notebooks). Do not cite unsuffixed `paper_results.md` or `paper_evidence_map.md`.
