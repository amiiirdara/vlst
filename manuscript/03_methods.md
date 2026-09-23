# Methods

Language follows pack terminology (**W5**): **association** for full-cohort univariate and multivariable screens; **prediction** only for nested-CV out-of-fold ranking; **interpretation / attribution** for selector catalogues and TabPFN explanations. Nested-CV discrimination on the derivation file is not external validation. Adjusted odds ratios and attribution maps are not treatment effects. Wang 2020 is a historical comparator on the same rows.

<!-- TRACE: W5; Methods paper/methods.md; YAML validation.*; nested_cv_v35_antileakage_on -->

---

## 3.1 Study cohort and data preprocessing

This is a retrospective derivation-cohort analysis of `data/raw/VLST.csv` (Wang 2020 derivation file): consecutive ACS patients aged ≥18 years undergoing PCI at The First Hospital of Jilin University, 1 January 2014 – 1 June 2015. Wang reported 6,038 eligible and 5,185 analysed (236 in-hospital deaths, 413 refused follow-up, 204 lost). Those flow counts are cited from Wang 2020. Analysed n = **5,185**; **92** definite VLST events; 5,093 non-events; prevalence **0.0177**. Median follow-up 1,502 days; median PCI → VLST 697 days. Ethics NO. 2013-256, written informed consent, and NCT03491891 are cited from Wang. This repository analysis was not separately pre-registered. <!-- TRACE: YAML study.*; W2 ethics -->

Outcome: binary **very late stent thrombosis** (column `Stent thrombosis`) — ARC 2007 definite stent thrombosis more than one year after implantation, angiographically confirmed. Probable and possible stent thrombosis are not counted. Wang analysed time-to-event; this pack uses the stored 0/1 label (limitation **W3.2**, not a result).

Exploratory data analysis (`code/analyzes/eda.ipynb`) found **no missing values**. Identifiers `NO.` and `Name` are dropped. All later parts use these 5,185 rows; **splits differ** (full-cohort association; inner 80/20 selectors; nested 5×4 prediction; 70/30 leakage twins and attribution).

**Quantization is file-level and pre-split.** Rounding is a cleaning rule on named labs, not a parameter fit on *y*, and is applied **before** any train/test or CV split. Incentive: recording precision in `VLST.csv` is a case/control batch marker (raw file sorted by outcome; cases transcribed at a different numeric convention). A signature-only probe (243 grid indicators, no magnitudes) reaches AP **0.4270** / ROC-AUC **0.9522**. Grid:

```text
CLINICAL_QUANTIZE_PLACES = {Cre: 0, CaI: 2, Fiberinogen: 1, Fast-Glu: 1}
```

`Fiberinogen` is the CSV spelling (`Cre` integer µmol/L; `CaI` 2 dp conventional TnI — do not round to 0; `Fiberinogen` / `Fast-Glu` 1 dp). This is not post-split leakage. <!-- TRACE: Methods §2 CLINICAL_QUANTIZE_PLACES; B15 anti_leakage_flow; leakage_precision_probe -->

---

## 3.2 Strict anti-leakage protocol

Operational definition = **five flags plus companion controls**, not “drop TSSI and WBC.” Nested CV, Part 2 selectors, and Part 5 attribution use the **ALL LEAKS OFF** state. Canonical pack copy: `paper_results/anti_leakage_protocol.md`. Freeze flow: drop `NO.` / `Name` / TSSI / WBC; quantize the four labs; stent codebook train-fold only; no SMOTE.

| Flag | OFF (prediction / attribution / selectors) | ON (leakage demonstration only) | Incentive |
| --- | --- | --- | --- |
| `KEEP_TSSI` | `False` — drop `Time since stent implantation` | `True` | Mixed time definitions (time-to-event vs completed follow-up). Rule “time < 1,241 → event” has zero control false positives. |
| `DROP_WBC` | `True` — drop `WBC` | `False` | Recording-precision / batch marker; Wang excluded WBC from the Cox score (infection). FDR still ranks it (dual-label). |
| `QUANTIZE_CLINICAL` | `True` (pre-split, §3.1) | `False` | Spurious decimals fingerprint batch/source. After OFF cleaning, equalising leftover precision still drops TabPFN v3.5 nested AP by **0.0518**. |
| `STENT_ENCODER_TRAIN_ONLY` | `True` — `encode_stent_on_fold` | `False` (full-cohort codebook) | Rare brands must not define levels from held-out rows. Unseen → `Other`. |
| `USE_SMOTE` | `False` | `True` in the ON twin only | Unmatched train synthesis; nested CV never uses SMOTE. |

**TSSI** is time-at-risk / completed follow-up, not a baseline covariate (VLST=1: time to thrombosis, min 380 days; VLST=0: event-free follow-up, min 1,241, max 1,605 days). **WBC** is dropped from the ML matrix as a leaky post-event / recording-precision variable. Identifiers `NO.` and `Name` are dropped. Follow-up drugs (`Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT`) **stay in** with a post-baseline caveat — they are not leak-flag drops. <!-- TRACE: W1; §4.1; W3.6–7 -->

**Stent-brand encoding.** `STENT_BRAND_COL = "Stent type-SES"`; brands with count `< min_count=30` collapse to `"Other"` (106 raw strings → 9 levels). Nested CV uses `encode_stent_on_fold` on each outer-training fold (**not** a full-frame codebook before the split). Classic models then clone imputer / scaler / one-hot **inside each CV split**; TabPFN arms see the 9-level frame natively. EDA found no missing values, so imputers are inert.

**Selectors never touch a parked 70/30 test.** `baseline_feature_selections.ipynb` splits once: fit `1 - INNER_VAL_SIZE`, score `INNER_VAL_SIZE=0.2` (`RANDOM_STATE=42`) → 4,148 / 74 events vs 1,037 / 18 events. `USE_CACHE=False`. Selector catalogues do not feed nested-CV prediction. Twin GridSearch `best_params_` are **not** imported into nested CV.

The complementary **ALL LEAKS ON** twin exists only to demonstrate leakage on the same 70/30 GridSearch family. It is not a nested-CV arm, not TabPFN, and not “identical except TSSI.” Nested TabPFN ON vs OFF is **[RE-SOURCE]**. <!-- TRACE: §4.6; YAML leakage_contrast -->

Where a hold-out is required (leakage twins; Part 5), one stratified `train_test_split` is used: `TEST_SIZE=0.30`, `RANDOM_STATE=42` → train **3,629 / 64 events**, held-out **1,556 / 28 events**. That split is **not** the prediction evaluation.

---

## 3.3 Model zoo

### Classic models — two non-interchangeable regimes

**(a) GridSearchCV (leakage twins only).** Seven families: logistic regression, decision tree, random forest, Gaussian naïve Bayes, CatBoost, XGBoost, LightGBM. Shared 3,629/1,556 split; `GRID_SCORING="average_precision"`; `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. SMOTE on the ON-twin training set only. Executed `best_params_` are a leakage-axis supplement, **not** nested-CV hyperparameters.

**(b) Main benchmark (nested CV).** Five classics — LR, RF, XGB, LGBM, CatBoost — plus four TabPFN arms (nine `RUN_MODELS`). Constructors: library defaults plus class weighting (`class_weight="balanced"`; CatBoost `auto_class_weights="Balanced"`; XGBoost `scale_pos_weight` from the outer-train fold). DT and GNB are not nested-CV arms. **GridSearch winners from (a) are not imported.** No SMOTE. The five classics are an **untuned reference panel**, not a nested inner GridSearch. <!-- TRACE: W3.8; YAML part4_constructors -->

### TabPFN v3.5 (local) and API thinking mode

Pins (`run_manifest.json`, Tesla T4): **`tabpfn==9.0.0`**, **`tabpfn-client==0.6.0`**. `model_path="auto"` still resolves to v3 in `tabpfn` 9.0.0, so checkpoints are named explicitly.

| Arm | Runtime | Checkpoint | Thinking |
| --- | --- | --- | --- |
| TabPFN v3 | local `tabpfn` | `tabpfn-v3-classifier-v3_default.ckpt` | off (`n_estimators="auto"`) |
| TabPFN v3.5 | local `tabpfn` | `tabpfn-v3.5-20260909.safetensors` | off |
| TabPFN thinking v3 | hosted `tabpfn_client` | `v3_default` | `thinking_mode=True`, `thinking_effort="high"`, `thinking_metric="average_precision"` |
| TabPFN thinking v3.5 | hosted `tabpfn_client` | `v3.5_default` | same `THINKING_KWARGS` |

v3 versus v3.5 are different weights; thinking versus local are different objects. They are never collapsed. Thinking constructors are unchanged from the freeze rule. Client quota comments (as of 2026, free tier): **20 thinking fits**, **50 million prediction cells per day**. Interpretability keeps MI, SFS, and PDP on local v3.5 (`FS_THINKING_MODE=False`; 0 client fits) so quota is reserved for held-out SHAP. <!-- TRACE: nested_cv_v35_antileakage_on.pins; W3.4–5 -->

The published Wang 8-variable **integer score** is scored on all 5,185 rows with published Table 2 points **frozen** (weights not re-fit). It is not a nested-CV arm and not a Cox re-fit. <!-- TRACE: wang_2020; B10 -->

---

## 3.4 Evaluation protocol (honest tuning budget)

**Primary ranking metric.** Average precision (**PR-AUC**) at prevalence 0.0177. ROC-AUC and Brier are reported alongside and are not collapsed across TabPFN arms.

**Nested cross-validation — the only prediction evaluation.** Stratified nested CV, **5 outer / 4 inner** folds, outer `shuffle=True`, `random_state=42`. Events per outer fold: 18, 18, 18, 19, 19 (**W4**). Inner CV tunes the **F1 decision threshold only** — not hyperparameters and not feature sets. **Single nested CV, not repeated nested CV.** “Repeated stratified splits” in this pack means: (i) four inner stratified folds per outer fold; (ii) eight shuffled seeds for stability SFS; (iii) 2,000 stratified bootstrap resamples of **stored** OOF scores (classifiers not re-fit). It does **not** mean repeated nested CV of the whole 5×4 scheme. <!-- TRACE: YAML validation.repetitions; W3.9; do not invent repeated nested CV -->

**Honest nested operating point (quote Table 2).** Per-fold inner-CV F1 threshold applied once to the unseen outer fold. Precision, recall (sensitivity), F1, F2 (β = 2.0), and 2×2 counts in Results 4.2 use this cut. A single pooled F1 cut on concatenated OOF labels is **optimistically biased** and is not the headline.

**Bootstrap CIs.** Patient-level stratified bootstrap of pooled nested OOF (`n_boot=2000`, seed 42), keeping 92 events and 5,093 non-events. Paired Δ PR-AUC uses the same resamples. <!-- TRACE: B3; YAML bootstrap_method -->

**Events per variable (W4).** 92/81 ≈ **1.14**; unidentified Table 4 92/17 ≈ **5.4** (not quoted as the association model); identified Table 4b 92/13 ≈ **7.1** (still below EPV ≥ 10). Association logits do not use `class_weight`. Firth on the same 13 covariates is **EDA-only** — not a nested-CV arm, never fused into GridSearch.

**Unequal tuning (stated in the ranking claim).** TabPFN has no per-dataset grid. Classics in nested CV are defaults plus class weights. The nine-arm table ranks those objects; it is not a hyperparameter-matched contest (**W3.8**).

---

## 3.5 Interpretability: 8-seed SFS, MI, PDP, held-out SHAP / *k*-SII

A full interpretability run exceeds the Kaggle session limit. Parent `tabpfn_interpretability.ipynb` is archived; live split:

1. **`tabpfn_interpretability_fs_pdp.ipynb`** — `mutual_info_classif` on **train**; stability forward SFS (`STABILITY_N_SEEDS=8`; a 10-seed plan was abandoned under the session limit); PDP; consensus inputs. Local TabPFN v3.5; **0 client fits**. Rankings use the 3,629-row train. Not a nested-CV feature mask. Live dump: 8/8 seeds completed; 3/10 provisional tables are **archived**.

2. **`tabpfn_interpretability_shap.ipynb`** — held-out SHAP (shapiq SV) and native SHAP-IQ (*k*-SII / waterfall). Intended constructor: client thinking (`INTERP_THINKING_MODE=True`, effort high, metric average_precision, v3.5). SHAP on **all 1,556 held-out rows**. *k*-SII / waterfall: first held-out VLST=1 row (cohort index **5176**; budget 256) — one-row interaction view, not a cohort screen. Stored dump: HTTP 429 then local finish of remaining SHAP rows (execution note, not a change of intended constructor).

Both notebooks apply the same ALL LEAKS OFF cleaning as nested CV (IDs + TSSI + WBC dropped; `CLINICAL_QUANTIZE_PLACES`; stent codebook train-only; 80 columns; no SMOTE). PDP is a train empirical curve near prevalence (`PDP_USE_CLIENT=False`), not nested-CV risk. Consensus ranking is a Borda-style mean of normalized ranks across train MI, train SFS frequency, and held-out mean(|SHAP|). <!-- TRACE: §5.9; YAML kaggle_interpretability; W3.11 -->

Numerical SoT remains `paper/frozen_results.yaml`. Notebooks and Kaggle dumps are provenance.
