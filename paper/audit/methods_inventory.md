# Methods inventory

Facts extracted only from the inspect list. Item class is stated on each block. Cell numbers are 0-based notebook JSON indices. Do not invent missing hyperparameters, imputers, or sample sizes.

Scientific language used here: **association** for inferential/statistical relationships; **prediction performance** for nested-CV machine-learning results. TabPFN thinking-high and TabPFN local are never collapsed. Wang 2020 is a **historical comparator** on the derivation cohort, not external validation of this pack.

---

## A. Study design and cohort

| Fact | Value as written | File | Location | Source type | Item class |
| --- | --- | --- | --- | --- | --- |
| Cohort identity | `data/raw/VLST.csv` **is** Wang 2020’s derivation cohort: consecutive ACS patients ≥ 18 years undergoing PCI at The First Hospital of Jilin University, 1 January 2014 – 1 June 2015 | `00_front_matter.md` | Clinical motivation, paragraph **Outcome** | report | methodological detail |
| Derivation / analysis setting | Derivation-cohort analysis only. Nested CV is on these same 5,185 rows. No temporal test of the ML models. Shantou rows are not in the repository | `00_front_matter.md` | Limitations item 1 | report | methodological detail |
| Sample size | n = 5,185 | Multiple: front matter; Part 1 header; Part 4 header; `wang_vlst_score.ipynb` cell 2 `assert len(df) == 5185` | see files | report / notebook | numerical result |
| Outcome | Very late stent thrombosis (VLST) = Academic Research Consortium 2007 *definite* stent thrombosis more than one year after implantation, angiographically confirmed. Probable and possible stent thrombosis are not counted. Target column = `Stent thrombosis` | `00_front_matter.md` Outcome paragraph; Part 4 protocol | report | methodological detail |
| Events / non-events | 92 VLST; 5,093 non-events; prevalence 0.0177 (1.77%) | Front matter; Part 1 Table C; Part 4 protocol | report | numerical result |
| Inclusion / exclusion (from Wang, cited not re-derived) | 6,038 eligible → 5,185 analysed: 236 in-hospital deaths, 413 refused follow-up, 204 lost | `00_front_matter.md` Outcome; Part 1 Table C caption | report | methodological detail |
| Ethics / registration | Ethics NO. 2013-256; written informed consent; NCT03491891 — cite Wang 2020; this repository analysis was not separately pre-registered | `00_front_matter.md` Data, ethics, consent | report | provenance detail |
| Same cohort throughout | Parts 1–5 and Wang score all use the same 5,185-row file. Splits differ (full-cohort association; 80/20 selector val; nested 5×4; Part 5 70/30; TSSI 70/30) | Front matter “What this pack adds”; Part 4/5 protocols | report | methodological detail |
| Train/test terminology | **Prediction:** nested 5 outer / 4 inner stratified CV only (Part 4). **Part 2:** no parked 70/30; `INNER_VAL_SIZE=0.2` fit/val. **Part 5:** stratified `train_test_split(test_size=0.3, random_state=42)` for attribution, not nested-CV prediction. **TSSI notebooks:** single 70/30 from `data/processed/` | Front matter Terminology table; Part 2 header; Part 5 protocol; TSSI cell 0 | report / notebook | methodological detail |
| Follow-up (descriptive, Wang) | Median follow-up 1,502 days; median PCI → VLST 697 days | `00_front_matter.md` Outcome | report | numerical result |
| Follow-up (TSSI column, as stored) | VLST=1: time to thrombosis, min 380 days. VLST=0: event-free follow-up, min 1,241, max 1,605; cohort median 1,502 | Part 4 §6 | report | numerical result |

**Derivation vs validation wording.** Reports correctly restrict “external” to Wang’s **Shantou** test of *their* Cox score. Nested CV on the derivation file is not external validation. Part 3 still contains the phrase “external-validation bar” as a hypothetical future bar for ML-only names (terminology issue; see conflict ledger).

---

## B. Predictor variables

### Complete baseline list

The inspected reports state **81 features** after dropping identifiers (`NO.`, `Name`) and `Time since stent implantation`. **No inspected report or notebook prints the complete 81-name list in one table.** Part 1 Table C is a clinical subset. Part 5 Table 1 is a top-15 MI excerpt. This is a gap, not an invented list (see `unsupported_claims.md`).

| Fact | File | Location | Source type | Item class |
| --- | --- | --- | --- | --- |
| 81 columns after dropping IDs + TSSI | Part 4 protocol; Part 5 protocol; front matter EPV table | report | methodological detail |
| Identifiers dropped | `NO.`, `Name` | Part 4 header; `baseline_plus_tabpfn.ipynb` cell 5 `X = df.drop(columns=["NO.", "Name", "Time since stent implantation", TARGET], errors="ignore")`; Part 5 cell 6 `ID_COLS` | notebook / report | methodological detail |
| Candidate removed before modeling (all ML parts) | `Time since stent implantation` | Part 2 `DROP_FEATURES` cell 1; Part 4 cell 5; Part 5 cell 6; `baseline_without_tssi.ipynb` cells 3–4 | notebook | methodological detail |
| Variable types (as used) | Mixed tabular: binary flags, continuous labs/echo, categorical `Stent type-SES` (9-level encoder), integer counts | Part 1 test-selection; Part 4 ColumnTransformer; Part 5 TabPFN-native integer categoricals | report | methodological detail |

### Variables retained by family

| Family | Feature view | File | Location |
| --- | --- | --- | --- |
| Part 1 association | Raw clinical columns; stent collapsed to 9 levels for χ²; TSSI excluded from interpretation as baseline | Part 1 header | report |
| Part 1 Table 4 | 17-covariate unweighted logit (unidentified) | Part 1 §5 Table 4 | report |
| Part 1 Table 4b | 13-covariate reduced unweighted logit (quote this) | Part 1 Table 4b | report |
| Part 2 selectors | Shared 9-level encoder then OHE drop-first + `StandardScaler` → **88 columns** | Part 2 header; notebook cell 1 | report / notebook |
| Part 4 classics | Same encoder, then OHE **without** drop-first inside each CV split → **~89 columns** | Part 4 methods note | report |
| Part 4 both TabPFN arms | Same 9-level frame natively (no scaling / one-hot) | Part 4 methods; cell 5 | report / notebook |
| Part 5 TabPFN | 81 TabPFN-native columns (9 brand codes, other text coded as integers) | Part 5 protocol; cell 6 | report / notebook |
| Wang integer score | Eight published point variables only (see §I) | `wang_vlst_score.ipynb` cell 2 | notebook |

### Post-index / time-dependent columns (must not be described as index-PCI covariates without caveat)

| Variable | What the files say | File | Location | Item class |
| --- | --- | --- | --- | --- |
| `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT` | Follow-up persistence after the mandated DAPT year, not index-PCI prescriptions | Front matter limitation 6; Part 1 Table C caption | report | methodological detail |
| `Time since stent implantation` | Time-to-event for cases; completed follow-up for non-cases. Not a baseline covariate | Part 4 §6; TSSI notebooks | report / notebook | methodological detail |

`CaI` means match Wang Table 1 peak troponin I but the CSV does not expand the name. `LV` is unnamed in the file. Neither is treated as a newly named marker (front matter items 10; Part 1 Table C).

Part 2 / Part 5 catalogues **do not feed** the Part 4 predictor (explicit in Part 2 header, Part 4 protocol, Part 5 protocol).

---

## C. TSSI and leakage

Every inspected occurrence of follow-up time / TSSI:

| Occurrence | Role | File | Location | Source type |
| --- | --- | --- | --- | --- |
| Named as time-at-risk / follow-up, excluded from baseline association | Descriptive / methods | Part 1 header; Table C caption (omitted from Table C) | report |
| Strongest univariate hit but excluded from ML | Association catalogue, not a predictor | Part 3 §1 statistical catalogue | report |
| Dropped before Part 2 selectors | `DROP_FEATURES = ["Time since stent implantation"]` | `baseline_feature_selections.ipynb` cell 1 | notebook |
| Dropped before Part 4 nested CV | `df.drop(..., "Time since stent implantation", ...)` | `baseline_plus_tabpfn.ipynb` cell 5 | notebook |
| Dropped before Part 5 | `DROP_FEATURES` cell 6 | `tabpfn_interpretability.ipynb` | notebook |
| **Used as a model feature** | Leakage demonstration only | `baseline_tssi_leakage.ipynb` (loads processed arrays that still contain the column) | notebook |
| Removed after load | Leakage control | `baseline_without_tssi.ipynb` cells 3–4 `DROP_FEATURES = ["Time since stent implantation"]` | notebook |
| Supplementary Table S-TSSI / Figure S-TSSI | With vs without metrics | Part 4 §6; `rebuild_tssi_leakage_table.py` `ROWS` | report / script |
| Front matter W1 | Already covered in Part 4 | `00_front_matter.md` line 5 | report |

**Is TSSI in any final baseline / nested-CV model?** No. Part 4 nested-CV models use the without-TSSI view. Part 2 and Part 5 drop it. Wang score does not include it.

**Is it used only for descriptive analysis or leakage demonstration?** Descriptive: Part 1/3 mention it as a univariate hit and exclude it. Leakage demonstration: TSSI pair of 70/30 notebooks + Table S-TSSI. It is **not** a legitimate baseline predictor in this pack.

**Results with vs without TSSI** (hard-coded in the rebuild script; rounded in Part 4 prose):

| Model | With TSSI PR-AUC | Without TSSI PR-AUC | File |
| --- | ---: | ---: | --- |
| Logistic Regression | 0.9575 | 0.5077 | `rebuild_tssi_leakage_table.py` lines 27–33 |
| Decision Tree | 0.7308 | 0.0405 | same |
| Random Forest | 0.9680 | 0.4700 | same |
| Gaussian NB | 0.0209 | 0.0209 | same (unaffected) |
| CatBoost | 0.9773 | 0.6582 | same |
| XGBoost | 0.9609 | 0.6118 | same |
| LightGBM | 0.9708 | 0.6018 | same |

Part 4 prose: LR PR-AUC 0.958 → 0.508; CatBoost 0.977 → 0.658 (rounding of the script values).

**Protocol caveat (methods, not a new number).** `baseline_tssi_leakage.ipynb` cell 4 sets `USE_SMOTE = True` (global SMOTE on train). `baseline_without_tssi.ipynb` cell 6 sets `USE_SMOTE = False`. Part 4 claims “same stratified 70/30 split and tuning protocol.” That SMOTE mismatch is a methodological discrepancy (conflict ledger). Nested-CV Part 4 does **not** use SMOTE.

---

## D. Preprocessing

| Topic | What the files support | File | Location | Item class |
| --- | --- | --- | --- | --- |
| Missing data | EDA printed **no missing values** in any column. Median / most-frequent imputers in Part 2/4 transformers are therefore **inert** | Part 1 header; Part 2 header; Part 4 methods note | report | methodological detail |
| Imputation | Part 2/4: `SimpleImputer(median)` on numeric; most-frequent on encoded stent. Part 5: “keep NaNs” in TabPFN-native path; CSV has none | Part 4 methods; Part 5 cell 6 docstring | report / notebook | methodological detail |
| Categorical encoding | Shared 9-level stent encoder (`min_count=30`, `STENT_BRAND_RAW_COL = "Stent type-SES"`). Part 2: OneHotEncoder **drop-first**. Part 4 classics: OneHotEncoder `handle_unknown="ignore"` **no drop-first**. Part 5: integer codes, no one-hot | Part 1/2/4/5 headers; Part 4 cell 5 | report / notebook | methodological detail |
| Scaling | Part 2 and Part 4 classics: `StandardScaler` on numeric. Both TabPFN arms: **no scaling** | Part 2 header; Part 4 methods | report | methodological detail |
| Outlier handling | Tukey IQR 1.5 in `eda.ipynb`; printed; rows not dropped; unused in Part 4 | eda.ipynb | provenance | **methods_only** |
| Other transforms | Continuous logits: per 1 SD in Table 4/4b. Welch vs Mann–Whitney by skew/kurtosis for univariate tests | Part 1 Tables R, 4b | report | methodological detail |
| Fit within training fold? | Part 4: `ColumnTransformer` **cloned and fitted inside every CV split**. Part 2: encoder then transformer on the fit slice of the 80/20 split (full cohort used once). Part 5: encoder on the loaded frame **before** `train_test_split` (stent collapse uses full-cohort counts — potential leakage of brand frequencies into the held-out split) | Part 4 methods; Part 5 cell 6 | report / notebook | methodological detail |
| Cross-fold leak risk | Part 4 imputers/scaler/OHE inside folds: designed not to leak. Stent encoder **before** the Part 4 split (full-cohort min_count). Part 5 same. Part 2 `INNER_VAL` is a single split, not nested CV | Part 4 methods note “feature views”; Part 5 cell 6 | report / notebook | methodological detail |
| TSSI 70/30 preprocessing | Loads `X_train.npy` / `X_test.npy` from `data/processed/` produced by `preprocessing.ipynb` (**not inspected**). Optional SMOTE on train only | TSSI notebooks cells 2, 4/6 | notebook | methodological detail / gap |

---

## E. Data splitting and validation

| Topic | Specification | File | Location | Item class |
| --- | --- | --- | --- | --- |
| Part 4 split | Nested stratified CV: `OUTER_SPLITS = 5`, `INNER_SPLITS = 4`, outer `StratifiedKFold(..., shuffle=True, random_state=42)`, inner `random_state=10_000 + outer_fold` | `baseline_plus_tabpfn.ipynb` cell 5 | notebook | methodological detail |
| Stratification | Yes, on the binary target | cell 5 | notebook | methodological detail |
| Repetitions | Single nested CV (not repeated k-fold). Random seeds: 42 (outer, models); inner fold-dependent | cell 5 | notebook | methodological detail |
| Feature selection inside folds? | **No** Part 2/5 mask. Inner loop tunes **only the F1 threshold**, not hyperparameters and not feature sets | Part 4 methods notes | report | methodological detail |
| Preprocessing inside folds? | Classics: yes (imputer/scaler/OHE). Stent 9-level encoder: before split. TabPFN: native frame | Part 4 | report | methodological detail |
| Bootstrap CIs | Patient-level **stratified** bootstrap of pooled Version 4 OOF; `n_boot = 2000`, seed 42; classifiers **not** re-fit. Code: `run_b3()` (not in the nested-CV notebook) | Part 4 §3; front matter limitation 9 | report | methodological detail |
| External validation of ML | **Absent.** Explicitly stated | Front matter limitations 1, 42 | report | methodological detail |
| Part 2 split | `test_size=INNER_VAL_SIZE` (0.2), `random_state=42` → fit 4,148 (74 events) / val 1,037 (18 events). No unused outer test | Part 2 header; notebook cell 3 | report / notebook | numerical result + methods |
| Part 5 split | `TEST_SIZE = 0.3`, `RANDOM_STATE = 42` → train 3,629 / 64 events; test 1,556 / 28 events | Part 5 protocol; notebook cell 6 | report / notebook | numerical result + methods |
| TSSI split | Pre-saved 70/30 arrays (same counts as Part 5: 1,556-row hold-out in Part 4 Figure S-TSSI caption) | Part 4 §6; Part 5 protocol | report | methodological detail |
| Wang fold evaluation | Frozen score evaluated on the same five outer folds as Part 4 (`StratifiedKFold(5, shuffle=True, random_state=42)`); weights not re-fit | Part 4 §7; wang notebook | report / notebook | methodological detail |

Events per Part 4 outer fold: **18, 18, 18, 19, 19** (front matter EPV table).

---

## F. Models

Firth logistic regression: **Table 4b sensitivity** (`fit_firth_logit` in `paper_hygiene_b3_b4_b7.py`). Same 13 covariates. Not nested CV.

### Inferential / statistical (association, not prediction)

| Model | Spec | File | Location |
| --- | --- | --- | --- |
| Univariate tests | Welch or Mann–Whitney by skew/kurtosis; χ² / Fisher for binary; BH-FDR | Part 1 header, Figure 1, Table R | report |
| Unweighted multivariable logistic | `statsmodels` Bernoulli logit; Table 4 (17 cov, unidentified); Table 4b (13 cov, quote); **no** `class_weight="balanced"` | Part 1 Table 4 caption | report |
| Domain-specific sparse logits | Supplementary Figures S3–S4 | Part 1 §7 | report |

### Prediction models (Part 4 nested CV)

From `baseline_plus_tabpfn.ipynb` / nbdump `RUN_MODELS`. SMOTE **not** used. Inner CV selects F1 threshold only. Classics are **explicit constructor kwargs** plus library defaults for unspecified arguments — **not** GridSearch winners from the TSSI notebooks.

| Model | Constructor / notes | Class weighting |
| --- | --- | --- |
| Logistic regression | `LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)` | balanced |
| Random forest | `RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)` | balanced |
| XGBoost | `eval_metric=aucpr`; `tree_method=hist`; `device=cuda` on Tesla T4; `scale_pos_weight` from train fold; `random_state=42` | scale_pos_weight |
| LightGBM | `metric=average_precision`; `class_weight="balanced"`; `device=gpu`; `verbose=-1` | balanced |
| CatBoost | `auto_class_weights="Balanced"`; `eval_metric=PRAUC`; `task_type=GPU`, `devices=0` | Balanced |
| **TabPFN (thinking-high)** | `tabpfn_client.TabPFNClassifier`, `thinking_mode=True`, `thinking_effort="high"`, `thinking_metric="average_precision"`, `random_state=42` | not class_weight; client thinking |
| **TabPFN (local)** | `from tabpfn import TabPFNClassifier`; `n_estimators="auto"`; Version 4 **omits** `balance_probabilities`; `restore_tabpfn_empirical_prior` skipped. nbdump still shows `balance_probabilities=True` — **not** Version 4. | none of the thinking-high flags |

Do not collapse the two TabPFN arms. Historical local Brier 0.0673 used `balance_probabilities=True` and is **not** Version 4.

### Other models (TSSI 70/30 notebooks only — not Part 4)

Logistic regression, Decision Tree, Random Forest, Gaussian NB, CatBoost, XGBoost, LightGBM, with GridSearchCV where applicable. These are the leakage demonstration, not the nested-CV headline.

### Part 2 selector factories (not Part 4)

`C=2.0` logistic; RF 500 trees; boosting 400 rounds; `lr=0.05`; CatBoost GPU Plain; **no TabPFN** in Part 2. SMOTE not used.

### Part 5 TabPFN for attribution

MI: sklearn, 0 TabPFN. Stability SFS: **local** `tabpfn`, `FS_THINKING_MODE=False`, 10 seeds, `n_estimators=1` in SFS cell. PDP: local (`PDP_USE_CLIENT=False`). SHAP / SHAP-IQ: intended client thinking (`INTERP_THINKING_MODE=True`) then **fell back to local** after HTTP 429. Local constructors omit `balance_probabilities`.

### Model selection procedure

- Part 4 ranking: pooled OOF PR-AUC (primary at 1.77% prevalence), ROC-AUC, Brier. Operating point: honest nested inner-F1 thresholds (Table 2), not pooled F1 (Table 3).
- Part 2: PR-AUC on the 20% val slice only.
- No hyperparameter grid in Part 4.

---

## G. Feature selection and extraction

| Method | Where | What it is | Feeds Part 4? |
| --- | --- | --- | --- |
| Univariate FDR (Welch / MW / χ² / Fisher + BH) | Part 1; 20 names excluding TSSI | Association | No |
| Multivariable unweighted logit | Part 1 Table 4b (13 names) | Association | No |
| LOCO | Part 2: drop-one PR-AUC on cheap-importance prefix cap 60 | Attribution | No |
| Coalition SHAP | Part 2: universe 40; independent of LOCO | Attribution | No |
| FFS (forward) | Part 2: pool 24, max 12 steps, `FFS_MIN_GAIN=0`; independent of LOCO | Attribution | No |
| Mutual information | Part 5 train split, all 81 scores | Attribution | No |
| Repeated forward SFS (stability) | Part 5 train, 10 seeds, k=10 | Attribution | No |
| PDP | Part 5 train average | Attribution | No |
| SHAP (shapiq SV) | Part 5 all 1,556 held-out rows | Attribution | No |
| k-SII / SHAP-IQ | Part 5 **one** held-out VLST=1 row (cohort 5176), budget 256 | Illustration, not a cohort interaction screen | No |
| Borda consensus | Part 5 train MI + train stability + held-out mean\|SHAP\| | Attribution | No |
| Stats vs ML Jaccard | Part 3: FDR-20 ∩ ML three-way-13 = 5 names; 5/28 ≈ 0.18 | Methods comparison | No |

**Exact feature sets for prediction models (Part 4):** all 81 non-ID, non-TSSI columns after the 9-level stent encoder (classics then ~89 OHE columns). No Part 2 three-way mask. No Part 5 consensus mask.

**ML consensus catalogue (n = 13, Part 3):** `1.1:1Post dilation`, `Aneurysm`, `CaI`, `Cre`, `HGB`, `HbA1c`, `LDL`, `LV`, `LVEF`, `Men`, `UA`, `WBC`, `eGFR`.

**Statistical FDR catalogue (n = 20, TSSI excluded):** listed in Part 3 §1 and `stats_vs_ml_comparison.ipynb` cell 2.

**Intersection (5):** `WBC`, `eGFR`, `LV`, `HbA1c`, `1.1:1Post dilation`. Notebook cell 4 print: `Jaccard=0.1786`.

---

## H. Metrics and inference (procedure)

| Metric / test | Used for | File |
| --- | --- | --- |
| PR-AUC (average precision) | Part 4 ranking (primary); Part 2 selector objective; TSSI comparison; Wang frozen score | Parts 2–4 |
| ROC-AUC | Part 4; TSSI; Wang | Part 4 |
| Brier | Part 4 calibration | Part 4 Table 1 / Figure 2 |
| Accuracy, precision (PPV), recall (sensitivity), specificity, F1, F2 (`beta=2.0`) | Part 4 Tables 2–3; TSSI script also stores F1/recall/prec | Part 4; rebuild script |
| NPV | **Not printed**; identity from Table 2 TN/(TN+FN) | Part 4 Table 2 counts |
| Calibration curves | Quantile-bin reliability, Figure 2 | Part 4 |
| Stratified bootstrap percentile 95% CI | Pooled OOF PR/ROC/Brier; paired Δ vs LightGBM | Part 4 Table S-CI / S-Δ |
| Wald 95% CI | Table 4 / 4b / S4 unweighted logit | Part 1 |
| Likelihood-ratio tests | Part 1 Supplementary Table S2 (16 interaction pairs vs main-effects-only). **Not** used for Table 4/4b CIs (those are Wald) | Part 1 |
| p-values | Univariate tests, FDR q, Table C | Part 1 |
| EPV | 92/81 ≈ 1.14; Table 4 92/17 ≈ 5.4; Table 4b 92/13 ≈ 7.1 | Front matter W4 |
| Class-weighted vs unweighted logit | Table 4/4b **unweighted**. Part 4 LR **balanced** for prediction — different objects | Part 1 vs Part 4 |

---

## I. Wang et al. 2020

See also `numerical_claims_inventory.md` §I. Methods-only:

- Use: **frozen 8-variable integer points** (Wang Table 2) scored on all 5,185 derivation rows. Not nested-CV. Not a re-fit of the Cox linear predictor.
- Formula: `wang_vlst_score.ipynb` cell 2 `WANG_POINTS`.
- Cohort: same `VLST.csv` derivation file.
- Historical comparator only. Wang’s Shantou c = 0.82 is **their** external test; that file is not here.
- This pack has **no** external validation of the ML models.

---

## J. Figures and tables (readiness)

Manuscript-facing items are listed with source file. “Ready” means the inspected report states D4 alignment with the executed notebook / CSV. “Verify” means a known caption, concat, or encoder-count issue.

### Part 1 (`EDA_paper_figures_and_tables.md`)

| ID | Content | Ready? |
| --- | --- | --- |
| Table C | Derivation-cohort characteristics | Ready for association Table 1; `LV`/`CaI` unnamed |
| Figure 1, Table R | Test selection | Ready |
| Tables 1–3, Figures 2–5 | Univariate FDR | Ready; encoder is **106 raw → 9 levels** (Kaggle/code). Do not quote EDA helper 99 as `n_raw`. |
| Table 4 | 17-cov unidentified logit | **Do not publish** as the clinical model |
| Table 4b | 13-cov reduced logit | Quote this; EPV still < 10 |
| Figure 6, S1–S4 | OR forest / domain | Ready as association |

### Part 2

| ID | Content | Ready? |
| --- | --- | --- |
| Tables 0–5, Figures 1–7 | Selector catalogues | Ready as attribution; Kaggle long CSVs not in repo; XGB 7-name HTML truncation reconstructed |

### Part 3

| ID | Content | Ready? |
| --- | --- | --- |
| Figures 1–2, Table 1 | Jaccard / membership | Ready as methods comparison. Wording is “held-out / external cohort that this pack does not contain.” |

### Part 4

| ID | Content | Ready? |
| --- | --- | --- |
| Tables 0–3, Figures 1–3 | Nested-CV prediction | Ready if Table 2 (nested) is quoted, not Table 3 (pooled F1) |
| Table S-CI, S-Δ, S-folds | Bootstrap / fold wins | Ready; provenance is hygiene script, not the Kaggle notebook |
| Table S-TSSI, Figure S-TSSI | Leakage demo | Ready as demonstration; SMOTE mismatch vs “same protocol” (verify) |
| Table S-Wang, S-Wang-bins | Frozen score | Ready as historical comparator; intermediate n 1577 vs Wang 1837 is documented |

### Part 5 (standalone file)

| ID | Content | Ready? |
| --- | --- | --- |
| Tables 0–5, Figures 1–13 | Attribution Version 5 | Ready. Concat rebuilt 2026-09-17 from standalone Parts 0–5. Quote Part 5 Table 4 Cre mean\|SHAP\| **0.2449**, not 0.158. |
