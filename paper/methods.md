# Methods

Language in this section follows the pack terminology (**W5**): **association** for full-cohort univariate and multivariable screens; **prediction** only for nested-CV out-of-fold ranking; **interpretation / attribution** for selector catalogues and TabPFN explanations. Nested-CV discrimination on the derivation file is not external validation. Adjusted odds ratios and attribution maps are not treatment effects. Wang 2020 is a historical comparator on the same rows.

## 1. Cohort and outcome

This is a retrospective derivation-cohort analysis of `data/raw/VLST.csv`, which is the Wang 2020 derivation file: consecutive ACS patients aged ≥18 years undergoing PCI at The First Hospital of Jilin University, 1 January 2014 – 1 June 2015. Wang reported 6,038 eligible patients and 5,185 analysed (236 in-hospital deaths, 413 refused follow-up, 204 lost). Those flow counts are cited from Wang 2020; they are not reconstructed from extra eligibility code in this repository. The analysed file has **5,185** patients and **92** definite VLST events (5,093 non-events; prevalence **0.0177** / 1.77%). Median follow-up is 1,502 days; median PCI → VLST among events is 697 days. Ethics NO. 2013-256, written informed consent, and NCT03491891 are cited from Wang 2020. This repository analysis was not separately pre-registered.

The outcome is binary **very late stent thrombosis** (column `Stent thrombosis`): Academic Research Consortium 2007 *definite* stent thrombosis more than one year after implantation, angiographically confirmed. Probable and possible stent thrombosis are not counted. Wang analysed time-to-event; this pack uses the stored 0/1 label. That difference is a limitation (**W3**), not a result.

Exploratory data analysis (univariate tests, clinical Table C, Table 4 / 4b, stent χ²) is `code/analyzes/eda.ipynb` (user `eda-(4).ipynb`). The EDA logic is unchanged from the audit phase; local venv versus Kaggle is infrastructure only. All later parts use these same 5,185 rows; **splits differ** (full-cohort association; inner 80/20 selectors; nested 5×4 prediction; 70/30 leakage twins and attribution).

## 2. Anti-leakage protocol

The operational definition is **not** “drop TSSI and WBC.” It is the five named flags in the twin 70/30 notebooks, implemented as the **ALL LEAKS OFF** state in nested CV, Part 2 selectors, and Part 5 attribution, plus companion controls (identifiers, train-only preprocessors, no selector mask into nested CV, no imported GridSearch winners). Pack copy: `paper_results/anti_leakage_protocol.md`. Freeze string: drop `NO.` / `Name` / TSSI / WBC; quantize `Cre` / `CaI` / `Fiberinogen` / `Fast-Glu`; stent encoder train-fold only; no SMOTE.

**Why a protocol is needed.** The raw file is sorted by outcome (all 92 cases are the last 92 rows). Cases were transcribed under a different numeric-precision convention from controls, so recording precision is a case/control **batch marker**. A nested diagnostic that keeps only “which decimal grid does this float lie on?” (no magnitudes) yields **243** indicators, AP **0.4270**, ROC-AUC **0.9522** (`leakage_precision_probe`). Separately, recoding Wang’s Cox *time axis* as a binary covariate leaks the label through mixed time definitions. Those two artefacts, plus brand-codebook and SMOTE inflation, are the incentives below.

| Flag | OFF (prediction / attribution / selectors) | ON (leakage demonstration only) | Incentive |
| --- | --- | --- | --- |
| `KEEP_TSSI` | `False` — drop `Time since stent implantation` | `True` | Mixed time definitions: VLST=1 = time to thrombosis (min 380 d); VLST=0 = completed follow-up (min 1,241, max 1,605 d). A rule “time < 1,241 → event” has zero false positives among controls. |
| `DROP_WBC` | `True` — drop `WBC` | `False` | Recording-precision / batch marker (no case has a whole-number WBC; many controls do). Wang also excluded WBC from the Cox score (infection not ruled out). FDR still ranks it (dual-label). |
| `QUANTIZE_CLINICAL` | `True` | `False` | Spurious decimal precision fingerprints batch/source. Not a *y*-fit: applied **before** any split. After OFF cleaning, equalising leftover precision still drops TabPFN v3.5 nested AP by **0.0518** (0.8957 → 0.8439). |
| `STENT_ENCODER_TRAIN_ONLY` | `True` | `False` | Rare `Stent type-SES` strings must not define levels from held-out rows. Unseen → `Other`. Nested: `encode_stent_on_fold` on each outer-train fold (`min_count=30`). |
| `USE_SMOTE` | `False` in nested CV, Part 2, Part 5, and the OFF twin | `True` in the ON twin only | SMOTE on the leaks-on train set inflates hold-out ranking and is unmatched vs TabPFN / nested CV. |

**Clinical quantization** (`QUANTIZE_CLINICAL`) is a file-level cleaning rule applied **before splitting**, not a parameter fit on *y*:

```text
CLINICAL_QUANTIZE_PLACES = {Cre: 0, CaI: 2, Fiberinogen: 1, Fast-Glu: 1}
```

`Fiberinogen` is the CSV column name (not a second fibrinogen variable). Grid rationale in the nested notebook: `Cre` 0 dp (µmol/L integer); `CaI` 2 dp (conventional TnI — do not round to 0); `Fiberinogen` / `Fast-Glu` 1 dp.

**Stent-brand encoding.** Column `STENT_BRAND_COL = "Stent type-SES"`; brands with count `< min_count=30` collapse to `"Other"` (106 raw strings → 9 levels). Under `STENT_ENCODER_TRAIN_ONLY=True` the codebook is fit on **train only**. Nested CV uses `encode_stent_on_fold` on each outer-training fold — **not** a full-frame codebook before the split. The 70/30 twins and Part 5 fit it on the 3,629-row train. Classic models then clone imputer / scaler / one-hot **inside each CV split**; TabPFN arms see the 9-level frame natively.

**Identifiers.** `NO.` and `Name` are dropped (not predictors).

**Follow-up drugs stay in.** `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT` are **not** leak-flag drops. They are post-baseline persistence after the mandated DAPT year (limitation), not index-PCI prescriptions.

**Selectors never touch a parked 70/30 test.** `baseline_feature_selections.ipynb` splits the full 5,185 rows once: fit on `1 - INNER_VAL_SIZE`, score on `INNER_VAL_SIZE=0.2` (`RANDOM_STATE=42`) → 4,148 rows / 74 events versus 1,037 rows / 18 events. Cached CSVs from an older test-scored export are not reused (`USE_CACHE=False`). Selector catalogues do not feed nested-CV prediction.

The complementary **ALL LEAKS ON** twin (`KEEP_TSSI=True`, `DROP_WBC=False`, `QUANTIZE_CLINICAL=False`, `STENT_ENCODER_TRAIN_ONLY=False`, `USE_SMOTE=True`) exists only to demonstrate leakage on the same 70/30 GridSearch family. It is not a nested-CV arm, not TabPFN, and not “the same notebook with TSSI restored.” Nested TabPFN ON vs OFF is **[RE-SOURCE]** (no matched 9-arm nested OFF dump).

## 3. Single stratified 70/30 split

Where a hold-out is required (leakage twins and Part 5 attribution), the pack uses one stratified `train_test_split`: `TEST_SIZE=0.30`, `RANDOM_STATE=42`, stratify on `Stent thrombosis`.

| Split | *n* | Events |
| --- | ---: | ---: |
| Train | 3,629 | 64 |
| Test / held-out | 1,556 | 28 |

This split is **not** the prediction evaluation. Nested stratified CV (section 7) uses all 5,185 rows as outer out-of-fold scores. Treating 3,629/1,556 as external validation would be incorrect (**W3**, **W5**).

## 4. Classic models — two regimes

Seven classic families appear in the twin notebooks: logistic regression (LR), decision tree (DT), random forest (RF), Gaussian naïve Bayes (GNB), CatBoost, XGBoost (XGB), and LightGBM (LGBM). They are run under two **non-interchangeable** regimes.

**(a) GridSearchCV regime (leakage twins only).** `baseline_tssi_leakage.ipynb` (ALL LEAKS ON) and `baseline_without_tssi.ipynb` (ALL LEAKS OFF) share the 3,629/1,556 split and `GridSearchCV` with `GRID_SCORING="average_precision"` and `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. Class weights are used; SMOTE is applied to the ON-twin training set only. Executed `best_params_` are archived as a leakage-axis supplement, not as nested-CV hyperparameters.

**(b) Main benchmark regime (nested CV).** `baseline_plus_tabpfn.ipynb` fits five classics — LR, RF, XGB, LGBM, CatBoost — plus four TabPFN arms (nine `RUN_MODELS` arms). Constructors are library defaults plus class weighting (`class_weight="balanced"` or CatBoost `auto_class_weights="Balanced"`; XGBoost sets `scale_pos_weight` from the outer-train fold). The inner loop tunes the **F1 decision threshold only**. DT and GNB are not nested-CV arms. **GridSearch winners from the twin notebooks are not imported.** The five classics are an **untuned reference panel**. The nine-arm table ranks these fitted objects; it is not a nested inner GridSearch of boosting versus TabPFN.

**Why (b) is the primary analysis.** Under **W5**, prediction is nested-CV out-of-fold ranking only. Regime (a) is a single 70/30 GridSearch whose job is to show that TSSI (and the other ON flags) inflate hold-out ranking; it is not a nested scoreboard and it is not matched on SMOTE. Importing those winners into (b) would leak the 70/30 split into later outer folds. TabPFN is used without a per-dataset grid; classics in (b) stay at library defaults plus class weights, with a shared inner F1 threshold. Regime (a) remains Methods/Results for leakage control, not for nested ranking.

## 5. TabPFN

Pins, recorded in `run_manifest.json` on the nested dump (Tesla T4): **`tabpfn==9.0.0`** (first release exposing v3.5 weights) and **`tabpfn-client==0.6.0`** (hosted v3.5). `model_path="auto"` still resolves to v3 in `tabpfn` 9.0.0, so checkpoints are named explicitly.

| Arm | Runtime | Checkpoint | Thinking |
| --- | --- | --- | --- |
| TabPFN v3 | local `tabpfn` | `tabpfn-v3-classifier-v3_default.ckpt` | off (`n_estimators="auto"`) |
| TabPFN v3.5 | local `tabpfn` | `tabpfn-v3.5-20260909.safetensors` | off |
| TabPFN thinking v3 | hosted `tabpfn_client` | `v3_default` | `thinking_mode=True`, `thinking_effort="high"`, `thinking_metric="average_precision"` |
| TabPFN thinking v3.5 | hosted `tabpfn_client` | `v3.5_default` | same `THINKING_KWARGS` |

v3 versus v3.5 are different weights; thinking versus non-thinking are different objects (client API versus local inference). They are never collapsed. Nested CV requests all four arms (`TABPFN_MODEL_VERSIONS` / `TABPFN_THINKING_VERSIONS`). Thinking constructors are unchanged from the freeze rule.

**Design constraint — client quota** (tabpfn-client free tier comments in the interpretability env cell, as of 2026): **20 thinking fits**, **50 million prediction cells per day**, 200 million per month. Nested thinking therefore costs outer × (inner + 1) hosted fits per thinking arm. Interpretability keeps mutual information, stability SFS, and PDP on local v3.5 (`FS_THINKING_MODE=False`, `PDP_USE_CLIENT=False`; 0 client fits) so the quota is reserved for held-out SHAP / SHAP-IQ.

## 6. Interpretability (two-notebook protocol)

A full interpretability run exceeds the **Kaggle session limit**. The parent `tabpfn_interpretability.ipynb` is archived as protocol only; the live split is:

1. **`tabpfn_interpretability_fs_pdp.ipynb`** — mutual information (`mutual_info_classif` on train), stability forward SFS (`STABILITY_N_SEEDS=8`; a 10-seed plan was abandoned under the same session limit, recorded as the 9-hour Kaggle cap), partial dependence, and a consensus ranking. Local TabPFN v3.5 only; **0 client fits**. Rankings use the 3,629-row train; the held-out set is unused for selection. These catalogues are not a nested-CV feature mask.

2. **`tabpfn_interpretability_shap.ipynb`** — held-out SHAP via **shapiq** and native **SHAP-IQ** (*k*-SII / waterfall). Intended constructor: client thinking (`INTERP_THINKING_MODE=True`, effort high, metric `average_precision`, version v3.5). SHAP is computed on the **held-out test set** (1,556 rows; `SHAP_EXPLAIN_HELDOUT=True`). *k*-SII / waterfall use the first VLST=1 row in that held-out split (cohort index 5,176; budget 256) and are not a cohort interaction screen. The SHAP notebook copies install, env knobs, leakage cleaning, 70/30 split, and v3.5 constructors from the parent; it does not re-run SFS (ignore a leftover `STABILITY_N_SEEDS=10` print). The stored dump records a client quota interruption (HTTP 429) after which remaining SHAP rows were finished locally; that is an execution note, not a change of the intended constructor.

Both notebooks apply the same ALL LEAKS OFF cleaning as nested CV (IDs + TSSI + WBC dropped; `CLINICAL_QUANTIZE_PLACES`; stent codebook train-only; 80 columns; no SMOTE). PDP is a train empirical curve near prevalence, not a nested-CV probability.

## 7. Statistics

**Shared ranking metric.** Average precision (**PR-AUC**) is the common ranking metric across nested CV, GridSearch scoring, booster `eval_metric` / `metric` / `eval_metric`, and thinking `thinking_metric`. With 92 / 5,185 events (prevalence 0.0177), ROC-AUC is dominated by true negatives; PR-AUC is the informative ranking scale. ROC-AUC and Brier are reported alongside PR-AUC and are not collapsed across TabPFN arms. Thresholded precision, recall, F1, F2 (β = 2.0), and nested operating points use inner-fold F1 thresholds (nested Table 2), not a single pooled F1 cut.

**Nested cross-validation (only prediction evaluation).** Stratified nested CV, 5 outer / 4 inner folds, outer `shuffle=True`, `random_state=42`. Events per outer fold: 18, 18, 18, 19, 19 (**W4**). Inner CV tunes the F1 threshold only — not hyperparameters and not feature sets. Single nested CV (not repeated). Preprocessors are cloned inside splits; the stent encoder is fit on each training fold. This is derivation-cohort nested CV, not a temporal test and not Wang’s Shantou evaluation.

**Bootstrap CIs.** Patient-level stratified bootstrap of the pooled nested OOF rows (`n_boot=2000`, seed 42), keeping 92 events and 5,093 non-events. Classifiers are not re-fit; the interval is sampling variability of the stored scores. Paired Δ PR-AUC uses the same resamples.

**Events per variable (**W4**).** Events / 81 candidate features ≈ **1.14**; events / 17-covariate Table 4 ≈ **5.4** (unidentified; not quoted as the association model); events / 13-covariate Table 4b ≈ **7.1** (identified unweighted Bernoulli logit; still below EPV ≥ 10). Association logits do not use `class_weight` (different object from nested LR).

**Firth logistic regression** is an association-analysis sensitivity on the same 13 Table 4b covariates (Heinze/Kosmidis half-correction in `paper_hygiene_b3_b4_b7.fit_firth_logit`). It is **EDA only**. It is not a nested-CV arm and is **never fused into predictive GridSearch**. Table 4b MLE remains the primary association fit.

Univariate tests (Welch or Mann–Whitney by skew/kurtosis; χ² / Fisher; BH-FDR) and Table 4b Wald 95% CIs are association screens. Exploratory LR interactions (16 pairs) are hypothesis-generating. The published Wang 8-variable integer score is scored on the same 5,185 rows as a historical comparator; it is not a nested-CV arm and is not a Cox re-fit.

## 8. Reproducibility

Numerical source of truth is **`paper/frozen_results.yaml`** (live nested key `nested_cv_v35_antileakage_on`). Notebooks and Kaggle dumps are provenance; Markdown reports restyle those dumps. Per-run manifests include nested `modeling_results/run_manifest.json` (pins, device, named arms, thinking settings, CSV SHA-256), Part 2 `split_manifest.json`, and Part 5 train / held-out / SHAP-explain index CSVs.

Kaggle runtime for the nested dump is Tesla T4. The interpretability parent was split into FS/PDP and SHAP notebooks because a combined run exceeds the Kaggle session limit (protocol blueprint: 9-hour cap; stability SFS reduced from 10 seeds to 8). Quota comments in the interpretability env cell (20 thinking fits; 50M cells/day) are the other hard constraint on thinking-mode design.

---

<!--
SECTION → FILE TRACEABILITY
(protocol/numbers only; notebooks are live code; YAML is freeze SoT)

1. Cohort & outcome
   paper_results/00_front_matter.md — Outcome; W3/W4/W5; ethics/NCT
   paper/frozen_results.yaml — study.n_total=5185; n_vlst=92; n_non_vlst=5093; event_rate=0.0177;
     median_follow_up_days=1502; median_pci_to_vlst_days=697; exclusion_criteria (6038/236/413/204);
     study.outcome / outcome_definition
   code/analyzes/eda.ipynb — EDA (user eda-(4).ipynb); logic unchanged vs audit
   paper/manuscript_outline.md — §0.2 eda.ipynb "Logic unchanged"; §4.1–4.2

2. Anti-leakage protocol (flags = operational definition)
   code/modeling/rating/baseline_without_tssi.ipynb cell 2 —
     KEEP_TSSI=False; DROP_WBC=True; QUANTIZE_CLINICAL=True; STENT_ENCODER_TRAIN_ONLY=True;
     CLINICAL_QUANTIZE_PLACES; TEST_SIZE=0.30; RANDOM_STATE=42
   code/modeling/rating/baseline_tssi_leakage.ipynb cell 2 — inverse flags (ALL LEAKS ON)
   paper/frozen_results.yaml — leakage_contrast.all_leaks_on/off.flags; nested anti_leakage_flow
   code/modeling/rating/baseline_plus_tabpfn.ipynb cell 5 —
     DROP_FEATURES=["NO.","Name","Time since stent implantation","WBC"];
     CLINICAL_QUANTIZE_PLACES (file-level before split); encode_stent_on_fold(min_count=30);
     STENT_BRAND_COL="Stent type-SES"
   code/modeling/interpretability/baseline_feature_selections.ipynb —
     DROP_FEATURES TSSI+WBC; INNER_VAL_SIZE=0.2; no parked 70/30; USE_CACHE=False
   paper/frozen_results.yaml — feature_selection_models.protocol (4148/74 vs 1037/18)

3. Single stratified 70/30
   paper/frozen_results.yaml — leakage_contrast.protocol; kaggle_interpretability.split
     train 3629/64; test 1556/28
   tabpfn_interpretability_fs_pdp.ipynb load cell — TEST_SIZE=0.3 RANDOM_STATE=42;
     executed print "train n=3629 (events=64) | held-out n=1556 (events=28)"
   interpretability_train_indices.csv / interpretability_heldout_indices.csv
   validation.validation_framework — nested CV is the only prediction evaluation

4. Classic models, two regimes
   (a) twins: baseline_tssi_leakage.ipynb + baseline_without_tssi.ipynb
       GRID_SCORING="average_precision"; 7 models LR/DT/RF/GNB/CatBoost/XGB/LGBM;
       USE_SMOTE True/False; StratifiedKFold n_splits=5 seed 42
   (b) baseline_plus_tabpfn.ipynb RUN_MODELS — LR/RF/XGB/LGBM/CatBoost + 4 TabPFN;
       class_weight balanced; inner F1 threshold only; GridSearch winners NOT imported
   paper/frozen_results.yaml — model_results.part4_constructors; validation.inner_cv.tunes
   paper_results/00_front_matter.md — W3 item 8 unequal tuning; W5 prediction = nested OOF only
   run_manifest.json arms list (9)

5. TabPFN
   baseline_plus_tabpfn.ipynb — TABPFN_PIN / TABPFN_CLIENT_PIN; THINKING_KWARGS;
     TABPFN_MODEL_VERSIONS; TABPFN_THINKING_VERSIONS; RUN_MODELS.update
   run_manifest.json — tabpfn 9.0.0; tabpfn-client 0.6.0; Tesla T4;
     local ckpt / safetensors; thinking v3_default / v3.5_default
   paper/frozen_results.yaml — nested_cv_v35_antileakage_on.pins / checkpoints / thinking_mode
   paper_results/00_front_matter.md — W3 items 4–5, 8 (v3 vs v3.5; thinking vs local)
   tabpfn_interpretability_fs_pdp.ipynb env cell — quota comments:
     20 thinking fits; 50M cells/day; 200M cells/month

6. Interpretability two-notebook protocol
   tabpfn_interpretability.ipynb cell 0 — "full run exceeds the Kaggle session limit"
   paper/manuscript_outline.md §0.2 — STABILITY_N_SEEDS=8; 10-seed abandoned; Kaggle 9h limit
   tabpfn_interpretability_fs_pdp.ipynb — FS_THINKING_MODE=False; PDP_USE_CLIENT=False;
     STABILITY_N_SEEDS=8; 0 client fits; MI/SFS/PDP/consensus
   tabpfn_interpretability_shap.ipynb — INTERP_THINKING_MODE=True; SHAP_EXPLAIN_HELDOUT=True;
     shapiq + native SHAP-IQ; k-SII first held-out VLST=1
   paper/frozen_results.yaml — interpretability.shap.part5 (1556 rows);
     interpretability.shapiq k_sii_row_cohort_index=5176 budget 256;
     shap backend HTTP 429 note; kaggle_interpretability.thinking_mode

7. Statistics
   paper_results/00_front_matter.md — W4 EPV table; W5 terminology; W3 limitation 1
   paper/frozen_results.yaml — statistical_analysis.epv 1.14 / 5.4 / 7.1;
     firth_logistic_regression.role; validation.outer_cv (5, events 18–19);
     inner_cv n_splits=4; bootstrap_method n_boot=2000 seed 42
   paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md —
     PR-AUC at 1.77% prevalence; nested Table 2 vs pooled Table 3
   code/modeling/tools/paper_hygiene_b3_b4_b7.py — bootstrap run_b3; fit_firth_logit
   wang_vlst_score.ipynb — historical comparator (not nested arm)

8. Reproducibility
   paper/frozen_results.yaml — freeze.status READY_FOR_MANUSCRIPT_DRAFTING;
     nested_cv_v35_antileakage_on; provenance.kaggle_ingest_manifest
   docs/kaggle_outputs_manifest.md
   .../modeling_results/run_manifest.json
   paper/manuscript_outline.md §0.2 live notebook inventory
-->
