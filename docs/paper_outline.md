# VLST paper — proposed structure

**Companion document:** [`paper_evidence_map.md`](paper_evidence_map.md). Every number cited below is traced there;
section references of the form "EM §7.1" point into it.

**Source Markdown reports** (canonical copies under `paper_results/`):

| Short name | Path |
| --- | --- |
| Part 1 | `paper_results/01_eda/EDA_paper_figures_and_tables.md` |
| Part 2 | `paper_results/02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md` |
| Part 3 | `paper_results/03_stats_vs_ml/feature_extraction_comparison.md` |
| Part 4 | `paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md` |
| Part 5 | `paper_results/05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md` |

---

## 0. Two structural decisions before drafting

### 0.1 Beat 5 cannot be written as stated until one number is resolved

The requested story says *"TabPFN provides stronger discrimination than the classical baselines but has weaker
calibration."* The first half is solid. **The second half is the one finding in this repository that the
repository contradicts itself on.**

| Source | TabPFN Brier |
| --- | --- |
| `paper_fig2_calibration_curves.png` panel title (= Part 4 Table 1, = Part 4 narrative) | **0.0360** — 5th of 6 |
| `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L737, same notebook | **0.0060** — **1st of 6** |

All five classic models agree to four decimals across both artefacts; only TabPFN differs, because the
`tabpfn_client` thinking mode is not reproducible across runs (EM §12.2). If 0.0060 is correct, TabPFN is the
**best**-calibrated model in the study and beat 5 inverts to *"stronger discrimination and better calibration,
but the probabilities are not transportable because prevalence is a sampling artefact."*

**Consequence for this outline.** Results §6 and the Discussion are written with two mutually exclusive drafting
branches, **A** and **B**. Do not draft either until the notebook is re-run once and a single Brier value is
fixed. Everything else in the paper is branch-independent.

### 0.2 One beat should be added to the story

The requested seven beats do not include the leakage analysis, which is the strongest methodological material in
the repository and appears in **no** report (EM §4.1, §7.5). A reviewer's first question about a 0.99 ROC-AUC on a
1.8% outcome will be "what leaked?", and the repository already answers it decisively: every control has
`Time since stent implantation` ≥ 1,241 days while cases reach down to 380, and removing that column drops
logistic-regression PR-AUC from 0.9575 to 0.5077 and CatBoost from 0.9773 to 0.6582.

**Proposed beat 2.5:** *A structurally leaky follow-up variable was identified and excluded, and the excluded
model is the one reported.* This becomes a Methods subsection (§2.10) and a short Results subsection (§3.2.5)
feeding one supplementary table. It costs no main-text figure slot and it pre-empts the most likely rejection.

---

## 1. Main-text figure and table budget

**Nine items: four tables, five figures.** Two optional swaps are noted. Everything else goes to the supplement.
Rationale for each cut is in §6.

| ID | Type | Working title | Story beat | Built from | Status |
| --- | --- | --- | --- | --- | --- |
| **Table 1** | Table | Baseline characteristics of the cohort, VLST vs no VLST | 1 | — | **MUST BE BUILT — does not exist** |
| **Figure 1** | Figure | Cohort derivation and analysis flow | 1 | — | **MUST BE BUILT — does not exist** |
| **Table 2** | Table | Univariate associations with VLST after FDR control | 2 | Part 1 Tables 1 + 2 + 3, merged | Rebuild (merge) |
| **Figure 2** | Figure | Univariate vs multivariable-adjusted associations (forest) | 2 | Part 1 Figure 6 | **Rebuild — model must be respecified first** |
| **Table 3** | Table | Nested cross-validation performance of six models | 3, 5 | Part 4 Table 1 + notebook §7.3/§7.4 | **Rebuild — current table is stale** |
| **Figure 3** | Figure | Precision–recall and ROC curves, nested-CV out-of-fold | 3, 5 | Part 4 Figure 1 | Reuse as-is |
| **Figure 4** | Figure | Calibration (reliability curves), nested-CV out-of-fold | 5 | Part 4 Figure 2 | **Conditional — see §0.1** |
| **Table 4** | Table | Feature membership across all four extraction methods | 4, 7 | Part 3 Table 1 + Part 2 Table 2 + Part 5 Table 5 | Rebuild (merge) |
| **Figure 5** | Figure | TabPFN interpretability: partial dependence + selection stability (2 panels) | 6 | Part 5 Figure 1 + Part 5 Table 2 | Rebuild (merge) |

**Optional swaps if a 10th/11th slot is wanted:** Part 3 Figure 1 (Venn) — but it duplicates Table 4;
Part 5 Figure 13 (Borda consensus) — but it duplicates Table 4's TabPFN column. Prefer to keep 9.

---

## 2. Section-by-section plan

---

## TITLE

**Purpose.** Commit to *what was done* and *on what*, without promising clinical utility that the evidence
cannot support.

**Candidates (ranked):**

1. *Prediction and interpretation of very late stent thrombosis after percutaneous coronary intervention: a
   comparison of classical machine learning, a tabular foundation model, and statistical feature extraction*
2. *A tabular foundation model for very late stent thrombosis: discrimination, calibration, and interpretability
   in a rare-event cohort*
3. *Statistical and machine-learning feature extraction for very late stent thrombosis: overlapping but
   non-identical evidence*

**Safe.** "Prediction", "interpretation", "comparison", "rare-event", "discrimination", "calibration".

**Requires confirmation.** Any word implying deployment ("risk score", "clinical decision support",
"personalised"). Note the root `README.md` uses "Personalized Risk prediction", but nothing in the repository
supports personalisation: there is a single global model, no individual-level validation, and no external cohort
(EM §6.6). Also confirm whether "cohort" is defensible at all — see §2.1 below.

---

## ABSTRACT

**Purpose.** Structured abstract carrying the seven beats in ~300 words. The one place where over-claiming is
most costly.

**Exact evidence to use.**

| Slot | Value | Source |
| --- | --- | --- |
| Sample | n = 5,185; 92 VLST events; prevalence 1.77% | EM §2.1 |
| Design | 5 outer × 4 inner stratified nested CV; inner loop selects the decision threshold only | EM §6.1–6.2 |
| Primary metric | PR-AUC (average precision), prevalence reference 0.0177 | Part 4 L5, L52 |
| Best model | TabPFN, PR-AUC 0.853 (fold mean 0.850 ± 0.075), ROC-AUC 0.988 (0.988 ± 0.006) | EM §7.1 |
| Best classical | CatBoost, PR-AUC 0.697 (0.701 ± 0.068), ROC-AUC 0.970 | EM §7.1 |
| Weakest | Logistic regression, PR-AUC 0.342 (0.357 ± 0.115) | EM §7.1 |
| Operating point | Use the **honest nested** figures: TabPFN precision 0.835, recall 0.717, F1 0.772 | EM §7.3 |
| Feature overlap | 5 of 20 statistical and 5 of 20 ML-consensus names shared; Jaccard 0.14 | Part 3 L55, L61 |
| Recurring across all methods | `WBC`, `eGFR`, `LV` | EM §8.8 |

**Safe claims.** All of the above, provided PR-AUC is reported against the 0.0177 prevalence line and the
operating-point numbers are the nested ones.

**Requires confirmation.**
- Any Brier or calibration sentence (§0.1).
- The word "cohort" if the design is case-control (§2.1).
- Do **not** write "TabPFN could identify patients at risk" or similar. There is no decision-curve analysis, no
  net benefit, no external validation (EM §12.3).
- Do **not** put the pooled-threshold recall of 0.837 in the abstract; it is optimistically biased (EM §6.5).

---

## INTRODUCTION

Four paragraphs. **The repository supplies almost nothing here** — the entire motivation content is two lines of
`README.md` and there are zero citations anywhere (EM §1). Everything below must be written from external
literature.

### Paragraph 1 — Clinical problem
**Purpose.** Establish that VLST is rare, late, and consequential.
**Evidence.** External only: ARC definitions, reported VLST incidence, associated mortality and reinfarction.
**Source.** **[GAP]** Nothing in the repository.
**Requires confirmation.** The ARC category used in *this* dataset (definite / probable / possible) and the
timing threshold that makes an event "very late" — neither is documented anywhere (EM §2.3). The introduction
cannot define the outcome until the author supplies this.

### Paragraph 2 — Why prediction is hard here
**Purpose.** Motivate the methodological choices: 1.77% prevalence, 92 events against 81 candidate features
(EPV ≈ 1.14 unfiltered, ≈ 5.4 for the 17-covariate model), and the resulting need for PR-AUC over ROC-AUC.
**Evidence.** EM §2.2.
**Safe.** The class-imbalance argument for PR-AUC; the events-per-variable arithmetic.

### Paragraph 3 — Why a tabular foundation model
**Purpose.** Motivate TabPFN specifically: in-context learning on small tabular data, native categorical
handling, no per-dataset hyperparameter search.
**Evidence.** External (TabPFN literature) plus the configuration actually used:
`thinking_mode=True, thinking_effort="high", thinking_metric="average_precision"` (EM §5.8).
**Requires confirmation.** Do not claim TabPFN needs no tuning as an *advantage* in this paper, because the
classical baselines were also left untuned (EM §6.3) — that turns a stated advantage into an admission that the
comparison is unmatched. Frame it neutrally.

### Paragraph 4 — Objectives
**Purpose.** State the three aims that the evidence actually supports.
**Proposed wording of aims:**
1. Characterise marginal and mutually adjusted associations between baseline variables and VLST.
2. Compare the discrimination and calibration of five classical models and one tabular foundation model under a
   single nested cross-validation protocol.
3. Compare the feature sets recovered by statistical hypothesis testing against those recovered by
   model-based selection and by TabPFN interpretability, and characterise why they differ.
**Safe.** All three, as *descriptive/comparative* aims.
**Requires confirmation.** Do not add a fourth aim about clinical implementation.

---

## METHODS

### 2.1 Study population and outcome

**Purpose.** Define who is in the dataset, what the outcome is, and — critically — how controls were sampled.

**Evidence available.**
- n = 5,185 rows, 92 events, prevalence 0.0177, zero missing values in all 82 columns
  (`domain_feature_map.csv`; EM §2.1).
- Identifier columns `NO.` and `Name` dropped (EM §2.1).
- Distribution of `Time since stent implantation`: controls min 1,241 / max 1,605 days; cases min 380
  (`.nbdump/code__modeling__rating__baseline_tssi_leakage.txt` L869–877).

**Source files.** `data/raw/VLST.csv`; Part 1 L5; Part 4 L5.

**Cites.** Table 1, Figure 1.

**Safe claims.** The counts, the prevalence, the completeness of the data.

**Requires author confirmation — blocking (EM CONFIRM #1, #2, #3):**
1. **Target definition.** ARC definite/probable/possible? Angiographic confirmation? What time threshold defines
   "very late" in this dataset? Could a patient contribute more than one event?
2. **Control sampling.** Every control has ≥ 1,241 days of follow-up. If controls were *selected* for surviving
   event-free past a fixed horizon, this is a **case-control design, not a cohort**, the 1.77% figure is a
   sampling artefact rather than an incidence, and PR-AUC / precision / PPV / Brier become non-transportable.
   The word "cohort" must be removed throughout if so.
3. **Provenance.** Recruitment period, number of centres, country, consecutive vs selected enrolment,
   inclusion/exclusion criteria, ethics approval, consent, data-availability statement. None exists in the
   repository.

### 2.2 Data preprocessing

**Purpose.** Describe the encoding pipeline — and disclose that the six compared models did **not** all receive
the same representation.

**Evidence.**

| Arm | Pipeline |
| --- | --- |
| LR, RF, XGBoost, LightGBM, CatBoost | `ColumnTransformer`: median imputation + `StandardScaler` (numeric); most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` (categorical). Cloned and refit **inside every CV split**. |
| TabPFN | Raw frame, native categorical handling, no scaling, 81 columns. |

Source: `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L179–190, L235–241.

Because there are no missing values, imputation is inert. The one-hot step is not: `Stent type-SES` contains
**106 distinct free-text brand strings**, so the classical arm receives ~106 near-empty dummy columns that
TabPFN never sees (EM §6.4, §12.5).

**Cites.** Methods text only; the encoding table goes to **Supplementary Table S1**.

**Safe claims.** The pipeline description; "the preprocessor was refit within each fold, so no scaling or
encoding information crossed the fold boundary."

**Requires confirmation (EM CONFIRM #9, #14).**
- Disclose the unequal representation explicitly. It is a genuine between-arm difference.
- **Choose one canonical `Stent type-SES` encoding.** It is currently handled six different ways across the
  repository (9 collapsed levels in EDA; 9 canonicalised categories in `preprocessing.ipynb`; raw 106-level
  one-hot in Part 2 giving 185 columns; raw one-hot in Part 4's classical arm; native categorical in Part 4's
  TabPFN arm; integer-coded-as-numeric in Part 5). Nine collapsed levels is the defensible choice.
- Note that `preprocessing.ipynb`'s artefacts (88 features, `preprocessor.joblib`) are **used by none** of the
  five reported analyses and should either be removed or explained.

### 2.3 Exploratory analysis

**Purpose.** State the test-selection rule and the multiplicity strategy.

**Evidence.**
- Continuous (24 variables): Welch t-test if |skew| ≤ 1 **and** excess kurtosis ≤ 3, else Mann–Whitney U.
  Verified: every row of `paper_table_test_rationale.csv` obeys the rule (EM §5.2).
- Binary (58 variables): χ² or Fisher exact; OR, RR, φ reported.
- Categorical (1 variable): χ² on `Stent type-SES` after collapsing levels with n < 30 to `other`, leaving 9.
- Multiplicity: Benjamini–Hochberg **within each of the three families separately**, not globally.

**Source.** Part 1 L5; `paper_table_test_rationale.csv`.

**Cites.** **Supplementary Figure S1** (test-selection map, Part 1 Fig 1) and **Supplementary Table S2**
(Table R, all 24 variables).

**Safe claims.** The rule and its consistent application.

**Requires confirmation.**
- Disclose that BH was applied **three times** (24, 58, 1 tests), not once across 83.
- Disclose that `Time since stent implantation` was **inside** the continuous FDR family, which shifts every
  other continuous q-value.
- Several results sit on the FDR knife edge: `Fast-Glu` q = 0.0536, `LVEF` q = 0.0666, `HGB` q = 0.0716,
  `CaI` q = 0.0869. Confirm that the q < 0.05 cut is reported as a threshold, not a dichotomy of truth.

### 2.4 Statistical feature extraction

**Purpose.** Define the univariate screen and the multivariable model.

**Evidence — univariate.** As §2.3; the FDR-significant set is 20 names excluding time-at-risk (EM §8.1).

**Evidence — multivariable, exactly as implemented** (`.nbdump/code__analyzes__eda.txt`):

| Element | Value | Line |
| --- | --- | --- |
| Estimator | `LogisticRegression(solver="lbfgs", max_iter=2000)` | L3643 |
| Penalty | `C=1e6` — near-unpenalised **ridge**, not unpenalised | L3611, L3643 |
| Weighting | `class_weight="balanced"` | L3610 |
| Entry rule | all FDR-significant continuous + FDR-significant binaries **truncated to the first 8** | L3586 |
| Inference | percentile bootstrap, `N_BOOT = 200` | L4625, L4666 |

**Cites.** Table 2, Figure 2.

**Safe claims.** The univariate screen as specified.

**Requires confirmation — blocking (EM CONFIRM #5).** The multivariable model **as currently fitted is not
identified** and cannot be published:
- It contains `1.1:1Post dilation` together with `No postdilation`, which are **exact complements** (verified
  cross-tab 2510/0/0/2675). Two parameters for one bit; the split between adjusted OR 0.144 and 0.895 is set by
  the ridge penalty, not by the data.
- It contains `eGFR`, `CKD5` (its stage) and `CKD90` (its dichotomisation at 90) simultaneously — three
  deterministic functions of one measurement. This produces `CKD5`'s sign flip (1.391 → 0.156) and `CKD90`'s
  bootstrap CI of **2.708 – 639.506**.
- It contains `3-vessel disease` and `NO.of vessels`.
- No VIFs; EPV = 92/17 ≈ 5.4; the 8-binary cap is undisclosed; 200 bootstrap replicates is low.
- `class_weight="balanced"` in an inferential model needs justification or removal.

**Required action:** refit with one representative per collinear block (recommended: `eGFR` continuous only;
`1.1:1Post dilation` only; `NO.of vessels` only), report VIFs and EPV, raise bootstrap replicates, and disclose
the entry rule. **Figure 2 and Table 2 cannot be finalised until this is done.**

### 2.5 Classical machine-learning models

**Purpose.** Specify the five baselines and disclose the tuning budget.

**Evidence** (`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L203–234):

| Model | Configuration | Tuned? |
| --- | --- | --- |
| Logistic regression | L2, `class_weight="balanced"`, `max_iter=1000` | **No** |
| Random forest | `class_weight="balanced"`, `random_state=42` | **No** |
| XGBoost | `eval_metric="aucpr"`, `scale_pos_weight` from the training fold | **No** |
| LightGBM | `metric="average_precision"`, `class_weight="balanced"` | **No** |
| CatBoost | `auto_class_weights="Balanced"`, `eval_metric="PRAUC"` | **No** |

**Cites.** Table 3; full specification to **Supplementary Table S3** (Part 4 Table 0, corrected).

**Safe claims.** The specifications; that class imbalance was addressed by weighting in every model.

**Requires confirmation — important (EM CONFIRM #8).** **No hyperparameter search was performed for any
classical model.** The inner CV loop tunes the decision threshold only (§2.9). Comparing untuned library
defaults against a high-effort TabPFN and concluding that TabPFN "dominates" is not a matched comparison. Either
tune the baselines or state this plainly in Methods and again in Limitations. Part 4's current phrasing —
"Six classifiers compared under the same nested-CV protocol" (Part 4 L31) — is true about the protocol and
misleading about the effort.

### 2.6 Feature-selection methods

**Purpose.** Describe LOCO, coalition SHAP and forward feature selection — and be explicit that these produced a
**discovery** list, not a validated one.

**Evidence** (`.nbdump/code__modeling__interpretability__baseline_feature_selections.txt`):

| Selector | Mechanism | Scored on | Line |
| --- | --- | --- | --- |
| LOCO | drop one column, refit, measure loss | **the 1,556-row / 28-event test set** | L521, L532 |
| Coalition SHAP | Shapley decomposition of a metric, 12 permutations, 3 background draws | **32 test rows, of which 28 are cases (87.5% prevalence)** | L556, L561–567, L584 |
| FFS | greedy forward addition | inner 20% hold-out of train (≈ 726 rows, ≈ 13 events), but confined to the LOCO top-30 pool | L621–634, L647 |

Run configuration: `RUN_MODE = "smoke"`, `FEATURE_TOPK = 12`, `SHAP_UNIVERSE = 24`, `LOCO_MAX_FEATURES = 40`,
`FFS_CANDIDATE_POOL = 30`, seven models, three objectives (`pr_auc`, `f1`, `f2`), 185 encoded columns.

**Cites.** Table 4; Part 2's own tables to **Supplementary Tables S6–S8**.

**Safe claims.** The mechanism of each selector; the within-model and cross-model intersections as *descriptive*
results.

**Requires confirmation — blocking (EM CONFIRM #6).** The **stored smoke figures** still have three defects
(Part 2/3 markdowns now disclose them). The **notebook code** has been changed but not re-run:
1. **Stored LOCO pool is not a ranking.** The executed run used the first 40 `ColumnTransformer` columns
   (23 continuous, then the first 17 binaries). Current code ranks that cap by train importance.
2. **Stored LOCO/SHAP scored on the test set.** Current code scores on an inner hold-out of train.
3. **Stored SHAP value function used an 87.5%-positive sample** (28 test events + 4 controls). Current code
   draws a stratified inner-val sample at train prevalence.

Until Kaggle re-export, report Parts 2 and 3 as exploratory discovery on the old smoke run.
**The paper must not describe these features as "validated" or "required by the model".**

Also: disclose `RUN_MODE = "smoke"` and its cost — 12 Shapley permutations with 3 background draws, Monte-Carlo
error unquantified, no seed-stability check (EM §12.8). And correct Part 2 Table 0, which states CatBoost used
"Ordered boosting"; the factory sets `task_type="GPU"`, where CatBoost supports only **Plain** (EM §12.7).

### 2.7 TabPFN

**Purpose.** Specify the foundation model and the reproducibility caveat.

**Evidence.** `tabpfn_client.TabPFNClassifier(thinking_mode=True, thinking_effort="high",
thinking_metric="average_precision", random_state=42)`, evaluated in the identical nested-CV loop as the five
baselines — 25 fits total (5 outer × [4 inner + 1 outer]); the first inner fit took 24 min 38 s
(`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L235–241, L327–330).

**Cites.** Table 3, Figure 3, Figure 4.

**Safe claims.** The configuration; that TabPFN used the same folds and the same pooled out-of-fold evaluation as
every baseline.

**Requires confirmation — blocking (EM CONFIRM #4, #18).**
- **TabPFN is not reproducible here despite `random_state=42`.** Two runs of the same notebook produced
  Brier 0.0360 vs 0.0060, F1 threshold 0.901 vs 0.173, and confusion counts 5066/27/15/77 vs 5072/21/19/73. Fix
  one run and report it.
- Record the `tabpfn_client` package version **and** the server-side model version. The headline result depends
  on a remote service that can change.
- `oof_predictions.csv` and `fold_thresholds.csv` were written to `/kaggle/working/` and never committed. They
  must be regenerated and deposited, or no reviewer can recompute a single Part 4 number or add a confidence
  interval.

### 2.8 Interpretability

**Purpose.** Specify the five TabPFN interpretability signals and, for each, the sample it was computed on.

**Evidence** (`.nbdump/code__modeling__interpretability__tabpfn_interpretability.txt`):

| Signal | Sample | Backend | Note |
| --- | --- | --- | --- |
| Mutual information | full cohort, 5,185 | sklearn, 0 TabPFN calls | screening only |
| Stability selection | full cohort; forward SFS keeping 10 of 81, 5-fold CV, AP scoring, **10 seeds**, ~8.6 h | local TabPFN | **the most defensible signal here** |
| PDP | fit on 70% train; 4 continuous + 6 binary | local TabPFN | `balance_probabilities=True` |
| SHAP (shapiq SV) | **15 rows, all VLST cases, zero controls** | local TabPFN after client failure | budget 256 |
| k-SII / SHAP-IQ | **one** positive-class patient | local TabPFN after client failure | budget 256 |

**Cites.** Figure 5; SHAP and k-SII to **Supplementary Figures S6–S8**.

**Safe claims.** Mutual information and stability selection as full-cohort screens; PDP as a model-average
response shape.

**Requires confirmation — critical (EM CONFIRM #11).** Four disclosures are mandatory and none currently
appears:
1. **All 15 SHAP rows are VLST cases.** The explained set is
   `np.concatenate([_pos_idx, _neg_idx])[:15]` on a stratified 30% split holding 28 positives (L1056–1060). The
   statement "high `LV`/`WBC` raise predicted risk" derived from an all-case sample is close to circular. Part 5
   says "15 explained rows" (L151) but never that they are all cases.
2. **k-SII is one patient.** Part 5 already says this (L223) — keep it and strengthen it. It is **not** a cohort
   interaction screen. The only cohort-level interaction evidence in the study is the 16-pair LR screen (§2.4).
3. **Every TabPFN call in Part 5 uses `balance_probabilities=True`** (L610, L774, L1067, …), a uniform-prior
   rescaling. That is why the binary PDP table shows baseline "P(y=1 | 0)" of 0.24 against a true prevalence of
   0.0177, and why Figure 1 is described as rising "toward ~0.6". **These are not absolute risks** and must be
   relabelled everywhere.
4. **The zeros in Part 5 Table 5's mutual-information column are imputed**, not measured — `Cre` and
   `No.of stents per lesion` simply fell outside the MI top-15. Also, `Fast-Glu` and `ZES` appear in Part 5
   Table 1 with blank values; recover them or drop the rows.

Also: the Part 5 PDP treats `Stent type-SES` as a numeric axis over an arbitrary brand ordering. Drop that panel.

### 2.9 Validation and evaluation metrics

**Purpose.** Specify the nested CV precisely, and — the key methodological point — that the reported operating
point is the honest nested one.

**Evidence.**
- Outer: `StratifiedKFold(5, shuffle=True, random_state=42)`. Inner:
  `StratifiedKFold(4, shuffle=True, random_state=10_000 + outer_fold)`.
- **The inner loop tunes the decision threshold only, not hyperparameters.** The F1-optimal threshold is
  estimated on inner out-of-fold predictions from the training portion and applied once to that outer fold's
  unseen cases (L293–309). This is a correct design and should be described as *nested threshold selection*.
- Primary metric **PR-AUC**, reference prevalence 0.0177. Secondary: ROC-AUC, Brier, and — at the operating
  point — precision, recall, specificity, F1, F2.
- Fold sizes: n = 1,037 each; events 18, 18, 18, 19, 19.

**Cites.** Table 3.

**Safe claims.** All of the above.

**Requires confirmation — important (EM CONFIRM #7, #10).**
1. **Report the nested operating point, not the pooled one.** Part 4 Tables 2–3 threshold at
   `best_fbeta_threshold(y_true_oof, probs)` — a grid search maximising F1 **against the same labels being
   scored** (L831). The honest nested numbers exist in the same notebook (L1010–1026) and are lower:
   TabPFN recall **0.717** not 0.837, F1 **0.772** not 0.786, TP 66 not 77.
2. **No confidence intervals exist on any metric, and no paired model comparison was run.** "TabPFN dominates,
   CatBoost is next" is a point-estimate ordering. Fold SDs are large relative to some gaps (TabPFN PR-AUC
   0.850 ± 0.075; CatBoost 0.701 ± 0.068). Add bootstrap CIs on PR-AUC/ROC-AUC/Brier and a paired test
   (DeLong for ROC-AUC, bootstrap for PR-AUC) before any superiority language.
3. State that there is **no external, temporal, or geographic validation** (EM §6.6).

### 2.10 [PROPOSED — new] Leakage assessment and sensitivity analysis

**Purpose.** Pre-empt the "why is your AUC 0.99?" question, and turn the repository's strongest unreported
material into a contribution.

**Evidence** (`.nbdump/code__modeling__rating__baseline_tssi_leakage.txt` L846–902;
`.nbdump/code__modeling__rating__baseline_without_tssi.txt` L906–924):
- `Time since stent implantation`: controls min 1,241 / max 1,605 days; cases min 380. The rule
  "time < 1,241 → predict thrombosis" gives **zero** false positives among controls.
- The column is defined differently for the two classes: follow-up window for controls, time-to-event for cases.
- With/without contrast on one 70/30 split — logistic regression PR-AUC 0.9575 → 0.5077; random forest
  0.9680 → 0.4700; CatBoost 0.9773 → 0.6582; XGBoost 0.9609 → 0.6118; LightGBM 0.9708 → 0.6018.
- The variable is excluded from every reported model (`.nbdump/…baseline_plus_tabpfn.txt` L167).

**Cites.** **Supplementary Table S4** (the with/without contrast).

**Safe claims.** That the variable is structurally confounded with the outcome definition; that it was excluded
from all reported models; the with/without numbers.

**Requires confirmation (EM CONFIRM #17).** Confirm this analysis enters the paper. Also confirm §2.1 item 2 —
the same evidence that identifies the leak also implies the design is case-control, and the paper must say so.

---

## RESULTS

### 3.1 Cohort and outcome — *beat 1*

**Purpose.** Establish rarity and describe who these patients are.

**Evidence.** n = 5,185; 92 events (1.77%); complete data; 81 analysis features across eight clinical domains
(4 demographic, 10 comorbidity, 5 presentation, 23 anatomical, 16 procedural, 2 cardiac function, 16 laboratory,
4 medication) per `domain_feature_map.csv`.

**Source.** `data/raw/VLST.csv`; `paper_results/01_eda/paper_figures/domain_feature_map.csv`.

**Cites.** **Table 1**, **Figure 1**.

**Safe claims.** Counts, prevalence, completeness, domain structure.

**Requires confirmation.**
- **Table 1 does not exist and must be built** (EM CONFIRM #19): cases vs controls, n (%) for binaries,
  mean ± SD or median [IQR] for continuous, standardised differences. This is mandatory for any clinical
  journal and its absence would be the first reviewer comment.
- **Figure 1 does not exist and must be built**: cohort derivation (how the 5,185 were assembled, how controls
  were sampled) plus the analysis flow. Cannot be drawn until §2.1 item 2 is answered.
- Do not report 1.77% as an *incidence* until control sampling is confirmed.

### 3.2 EDA findings — *beat 2*

**Purpose.** Present the marginal associations and the clinically interpretable pattern behind them.

#### 3.2.1 Univariate associations

**Evidence — the ten FDR-significant continuous variables** (`paper_table1_continuous_univariate.csv`):

| Variable | Test | Effect | Δ mean | q |
| --- | --- | --- | --- | --- |
| WBC | Mann–Whitney | r = 0.130 | +3.75 | 9.48e-20 |
| eGFR | Welch | d = −0.712 | −24.15 | 3.71e-19 |
| LV | Welch | d = 1.127 | +4.56 | 3.26e-16 |
| CKD5 | Mann–Whitney | r = 0.040 | +0.18 | 5.69e-05 |
| No.of stents per lesion | Mann–Whitney | r = 0.037 | +0.21 | 6.00e-04 |
| HbA1c | Mann–Whitney | r = 0.052 | +0.47 | 7.00e-04 |
| NO.of vessels | Welch | d = 0.388 | +0.32 | 0.0014 |
| Total stent length | Mann–Whitney | r = 0.045 | +6.76 | 0.0030 |
| Fiberinogen | Mann–Whitney | r = 0.035 | +0.20 | 0.0286 |
| *(Time since stent implantation)* | Mann–Whitney | r = −0.170 | −572.05 | 4.07e-33 | → §3.2.5 only |

**Evidence — the ten FDR-significant binary variables** (`paper_table2_binary_univariate.csv`):
`1.1:1Post dilation` OR 0.187 (q 3.69e-09), `No postdilation` OR 5.355 (same test), `CKD90` OR 2.625
(q 1e-04), `Previous PCI` OR 6.485 (q 2e-04), `3-vessel disease` OR 2.169 (q 0.0021), `Clopidogrel` OR 0.503
(q 0.0209), `Diabetes` OR 1.889 (q 0.0226), `PES` OR 2.158 (q 0.0315), `Multi-vessel CAD` OR 1.890 and
`Single-vessel disease` OR 0.529 (both q 0.0351).

**Evidence — categorical:** `Stent type-SES`, χ² = 44.90, df = 8, Cramér's V = 0.093, p = 3.85e-07.

**Cites.** **Table 2** (all three families merged).

**Safe claims.** Every p, q, OR and effect size above, described as **marginal associations**.

**Requires confirmation (EM CONFIRM #12).**
- **Do not present Cohen's d and Mann–Whitney r on one axis or in one column.** `WBC` has the second-smallest
  p-value in the whole study (7.9e-21) but r = 0.130, next to `LV` at d = 1.127. MW r = Z/√N is severely
  attenuated at 1.77% prevalence. Split the metrics into separate panels/columns or convert to a common scale.
  This is why Part 1 Figure 3 should **not** go to the main text.
- **Report the 20 "discoveries" as roughly 12 distinct constructs.** At least 8 slots are redundant
  re-encodings: the post-dilation complements (2 slots, 1 bit), the vessel-disease family (`3-vessel`,
  `Multi-vessel`, `Single-vessel`, `NO.of vessels` — 4 slots, 1 construct), and the renal family (`CKD5`,
  `CKD90` alongside continuous `eGFR` — 3 slots, 1 construct). Verified from the raw data (EM §3.4).
- No confidence intervals accompany any effect size or OR. Add them.

#### 3.2.2 Multivariable adjustment

**Evidence.** The respecified model from §2.4. From the current (non-identified) fit, the estimates that are
likely to survive respecification: `WBC` adjusted OR 3.00 per SD [2.42–4.23], `eGFR` 0.113 per SD [0.060–0.158],
`LV` 3.28 per SD [2.34–5.25], `Previous PCI` 8.98 [3.23–28.62], `Clopidogrel` 0.464 [0.195–0.817],
`1.1:1Post dilation` 0.144 [0.040–0.245].

**Cites.** **Figure 2** (univariate vs adjusted forest), **Table 2** (adjusted-OR column).

**Safe claims.** *After respecification only.* The direction and rough magnitude for `WBC`, `eGFR`, `LV` and
`Previous PCI` are stable across both the 17-covariate model and the independent 12-covariate joint domain model
(`domain_joint_multivariable_or.csv`: WBC 2.94, eGFR 0.203, LV 2.95, Previous PCI 9.58) — that concordance across
two specifications is worth stating.

**Requires confirmation — blocking.**
- Nothing from this model can be reported until it is respecified (§2.4).
- `CKD5`'s sign flip and `CKD90`'s CI of 2.708–639.506 must not appear in the paper.
- `LVEF` in the joint model **reverses sign** (univariate 0.851 → adjusted 1.650 [1.301–2.260]) with `LV` in the
  same model. Part 3 L164 describes this as the OR "persist[ing]" — that is wrong and must be corrected
  (EM CONFIRM #21).
- `Men` goes from a null univariate (OR 1.286, p = 0.27) to adjusted OR 3.28 [1.58–7.90]. Report as exploratory.
- **Never write "protective."** `1.1:1Post dilation` (OR 0.14) and `Clopidogrel` (OR 0.46) are subject to
  confounding by indication in an observational dataset.

#### 3.2.3 Clinically interpretable pattern

**Purpose.** The "risk pattern" half of beat 2 — group the associations into a coherent clinical story.

**Evidence.** `domain_strength_summary.csv`: Laboratory 16 features / 6 global FDR hits; Cardiac function 2 / 1;
Procedural-stent 16 / 5; Comorbidities 10 / 2; Anatomy 23 / 4; Medications 4 / 1; Presentation 5 / 0;
Demographics 4 / 0. The signal concentrates in **laboratory, cardiac-function and procedural** blocks and is
absent from presentation and demographics.

**Cites.** **Supplementary Figure S2** (domain top hits, Part 1 Fig S1) and **Supplementary Table S5** (domain
summary).

**Safe claims.** The domain-level concentration of signal.

**Requires confirmation.** This paragraph is where the reader will form a biological narrative
(inflammation + renal impairment + LV remodelling + suboptimal stent deployment). That narrative is only
legitimate if §2.1 item 1 confirms the labs and echo are **baseline** measurements. See §3.2.4.

#### 3.2.4 [CRITICAL — must be resolved before §3.2.3 can be written]

`WBC` 12.49 vs 8.75, `LV` 49.11 vs 44.55 mm, `eGFR` 95.88 vs 120.03. Univariate ROC-AUCs of 0.784, 0.802 and
0.711; a plain 5-fold logistic on those three variables alone reaches ROC-AUC 0.894 / PR-AUC 0.116, against
full-model ROC-AUCs of 0.925–0.988.

A +3.75 ×10⁹/L white-cell difference and a +4.6 mm LV difference are the magnitudes one expects when comparing an
**acute event against a stable outpatient**, not baseline labs predicting an event years later. The repository
contains **no** record of when bloods were drawn or the echocardiogram performed (EM §4.3).

**Requires confirmation — blocking (EM CONFIRM #1).** If these are event-time values for cases, this is the same
class of structural leakage as `Time since stent implantation`, the paper is not a prediction study, and every
performance number in §3.3–3.4 is uninterpretable. **Answer this before drafting any Results prose.**

#### 3.2.5 Interactions (brief)

**Evidence.** 16 hand-picked pairs, LR test, BH within those 16. Two survive: `LV × eGFR` (OR 1.245, q 0.0277)
and `Men × eGFR` (OR 0.342, q 0.0280).

**Cites.** **Supplementary Table S9** — and it must show **all 16 rows**; Part 1's Table S2 currently shows only
8 with no truncation note (EM §10).

**Safe claims.** Two interactions reached q < 0.05 among 16 pre-selected pairs.

**Requires confirmation.** Both q-values sit just under 0.05; the pairs were chosen by hand, not systematically.
Frame as hypothesis-generating. Note also that the LR tests use **unweighted** fits while the ORs in the same
section use `class_weight="balanced"` — an internal inconsistency to disclose or fix.

### 3.3 Classical model performance — *beat 3*

**Purpose.** Establish what conventional methods achieve, as the reference against which TabPFN is judged.

**Evidence** (pooled nested-CV out-of-fold, n = 5,185, 92 events; EM §7.1, §7.4):

| Model | PR-AUC (pooled) | PR-AUC (fold mean ± SD) | ROC-AUC (pooled) | ROC-AUC (fold mean ± SD) |
| --- | --- | --- | --- | --- |
| CatBoost | 0.6967 | 0.7007 ± 0.0684 | 0.9704 | 0.9712 ± 0.0117 |
| LightGBM | 0.6770 | 0.6841 ± 0.0937 | 0.9613 | 0.9633 ± 0.0198 |
| XGBoost | 0.6647 | 0.6792 ± 0.0883 | 0.9493 | 0.9492 ± 0.0343 |
| Random forest | 0.4563 | 0.4740 ± 0.0319 | 0.9313 | 0.9304 ± 0.0186 |
| Logistic regression | 0.3418 | 0.3568 ± 0.1153 | 0.9246 | 0.9256 ± 0.0225 |

Prevalence reference for PR-AUC: **0.0177**.

**Source.** `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L974–990, L1030–1060 — **not** Part 4
Table 1, which is stale.

**Cites.** **Table 3**, **Figure 3**.

**Safe claims.**
- Gradient boosting (0.66–0.70 PR-AUC) clearly outperforms bagging (0.46) and the linear model (0.34).
- **The key rhetorical point of this subsection:** all five models exceed ROC-AUC 0.92, yet PR-AUC ranges from
  0.34 to 0.70. ROC-AUC is nearly uninformative here; PR-AUC separates the models. Part 4 L52 makes this point
  and it should be elevated to a headline methodological observation.

**Requires confirmation.**
- Baselines are untuned (§2.5).
- No CIs, no paired tests (§2.9 item 2). Note that logistic regression's fold SD is 0.115 — its 0.34 point
  estimate is unstable.
- Do **not** claim clinical usefulness from any ROC-AUC (EM §12.3).

### 3.4 TabPFN versus baselines — *beats 3 and 5 (discrimination half)*

**Purpose.** The paper's headline result.

**Evidence.**

| Metric | TabPFN | Best classical (CatBoost) | Margin |
| --- | --- | --- | --- |
| PR-AUC pooled | **0.8534** | 0.6967 | +0.157 |
| PR-AUC fold mean ± SD | **0.8503 ± 0.0746** | 0.7007 ± 0.0684 | — |
| ROC-AUC pooled | **0.9883** | 0.9704 | +0.018 |
| ROC-AUC fold mean ± SD | 0.9884 ± 0.0061 | 0.9712 ± 0.0117 | — |
| Per-fold PR-AUC | 0.824, 0.778, 0.793, 0.948, 0.909 | 0.699, 0.675, 0.609, 0.795, 0.726 | TabPFN higher in **5/5 folds** |

**Honest nested operating point** (EM §7.3): TabPFN precision 0.8354, recall 0.7174, specificity 0.9974,
F1 0.7719, F2 0.7383, TN/FP/FN/TP = 5080/13/26/66, threshold 0.297 ± 0.053. CatBoost: precision 0.6211,
recall 0.6413, F1 0.6310, 5057/36/33/59.

**Cites.** **Table 3**, **Figure 3**.

**Safe claims.**
- TabPFN ranked highest on PR-AUC in **every one of the five outer folds** — this is the strongest statement
  available without a formal test, and it should be made explicitly.
- At the nested operating point TabPFN identified 66 of 92 events with 13 false positives, versus CatBoost's 59
  events with 36 false positives.

**Requires confirmation.**
- Report the nested operating point, not the pooled one (EM CONFIRM #7).
- Add a paired test before the word "superior" (EM CONFIRM #10).
- Disclose unequal tuning (§2.5) and unequal feature representation (§2.2) in the same paragraph, not buried in
  Limitations.
- Do not report accuracy (0.99 for every model; it is prevalence, not skill). Part 4 L103 already says this —
  act on it by removing the column rather than printing it with a caveat.

### 3.5 Statistical versus ML feature-set comparison — *beats 4 and 7*

**Purpose.** Show that the four extraction routes recover overlapping but non-identical sets, and — beat 7 — that
this is **complementarity, not contradiction**.

**Evidence.**

| Quantity | Value | Source |
| --- | --- | --- |
| Statistical FDR catalogue | 20 names | Part 3 L43 |
| Classical-ML consensus catalogue | 20 names | Part 3 L45 |
| Intersection | **5**: `WBC`, `eGFR`, `LV`, `Fiberinogen`, `Previous PCI` | Part 3 L61 |
| Jaccard | 5/35 ≈ **0.14** | Part 3 L55 |
| Strictest ML intersection (7 models × 3 selectors) | `WBC`, `eGFR` | Part 2 Table 4 |
| TabPFN consensus at 3/3 signals | `LV`, `WBC`, `eGFR`, `Stent type-SES`, `No postdilation`, `HbA1c` | Part 5 Table 5 |
| **Recovered by all four routes** | **`WBC`, `eGFR`, `LV`** | EM §8.8 |

**The mechanisms behind the disagreement**, which are the actual scientific content of this subsection:

| Mechanism | Example | Assessment |
| --- | --- | --- |
| Collinear families — tests score every member, models keep one | `3-vessel` / `Multi-vessel` / `Single-vessel` / `NO.of vessels`; the post-dilation complements; `CKD5` / `CKD90` / `eGFR` | **Real and generalisable** |
| Surrogates — a null univariate that a model still uses | `Cre` (p = 0.88) standing in for `eGFR` | **Real** |
| Interactions invisible to marginal tests | `Men` (p = 0.27) but `Men × eGFR` q = 0.028 | **Real, exploratory** |
| Encoding | `Stent type-SES`: one χ² on 9 levels vs 106 sparse dummies | **Real but self-inflicted** — an inconsistency between notebooks, not a property of the variable |
| Candidate-pool truncation | `HbA1c`, `Clopidogrel`, `Diabetes`, `Total stent length`, `No.of stents per lesion` may simply lie beyond column 40 | **Artefact** — absence is not evidence |
| Metric and sample | FDR on 92 events vs PR-AUC/F2 on 28 | **Real** |

**Cites.** **Table 4** (membership matrix across statistical FDR / multivariable survivor / classical-ML
consensus / TabPFN consensus). Part 3's Venn (Fig 1), presence heatmap (Fig 2), reason buckets (Fig 3) and domain
counts (Fig 4) → **Supplementary Figures S3–S5**.

**Safe claims.**
- The membership counts and the intersection, stated as a **methods-comparison** result.
- The first three mechanisms in the table above.
- Beat 7's thesis: the routes ask different questions — marginal distributional difference, versus change in a
  fitted model's held-out score — so non-identical answers are the expected outcome, not a contradiction.
  Part 3's own framing table (L33–40) already makes this argument well.

**Requires confirmation — important.**
- **Jaccard = 0.14 must not be presented as a biological finding.** It is largely a function of differing power
  (92 vs 28 events), differing encodings, and an arbitrary candidate pool. Rewrite as a methods result.
- Every sentence about a statistics-only feature must carry the pool-truncation caveat (§2.6 item 1). Part 3
  states it at L145 and L226; it must not stay buried.
- Correct the `LVEF` claim: the adjusted OR **reverses**, it does not persist (§3.2.2).
- Reconcile the three different values reported for the same "univariate OR" — `Previous PCI` appears as 6.485
  (2×2), 6.465 (weighted logistic) and 6.733 (unweighted logistic) in three tables, all labelled identically
  (EM CONFIRM #13).
- Part 2 Table 5 (priority ranks) should be **cut entirely**: its own caption says most rows are string-matching
  failures (`Age, years` vs `Age`). That is a bug report, not a result.

### 3.6 TabPFN interpretability — *beat 6*

**Purpose.** Show nonlinear response shapes and the variables that recur across independent signals.

#### 3.6.1 Nonlinearity

**Evidence.** Continuous PDP (Part 5 Figure 1): `LV` response is flat until the mid-40s and then rises steeply;
`eGFR` runs the other way, high predicted output at low filtration falling as eGFR rises; `Age` is essentially
flat despite being selected in 9/10 stability runs.

**Cites.** **Figure 5, panel A**.

**Safe claims.** The **shape** of the model's average response — that `LV` shows a threshold-like rise rather
than a linear trend, and that `Age`'s flat PDP contrasts with its high selection frequency.

**Requires confirmation — critical.** The y-axis is a **balanced-prior model output, not absolute risk**
(§2.8 item 3). "Rises toward ~0.6" against a true prevalence of 0.0177 will be read as a 60% event probability.
Relabel the axis and every caption. Drop the `Stent type-SES` PDP panel (numeric sweep over an arbitrary brand
ordering).

#### 3.6.2 Recurring variables

**Evidence.** Stability selection, 10 seeds of forward SFS keeping 10 of 81 (Part 5 Table 2):
selection frequency **1.0** for `LV`, `Stent type-SES`, `eGFR`; **0.9** for `Age`, `Cre`, `WBC`; 0.7
`No postdilation`; 0.6 `STEMI`; 0.5 `HbA1c` and `LVEF`. A long tail appears once or twice and is not robust.

Mutual information top 5 (full cohort): `CaI` 0.0192, `WBC` 0.0184, `LV` 0.0152, `Stent type-SES` 0.0114,
`eGFR` 0.0099.

**Cites.** **Figure 5, panel B** (stability frequencies). Mutual information → **Supplementary Table S10**.

**Safe claims.** `LV`, `eGFR` and `WBC` recur across mutual information, stability selection, the classical-ML
consensus **and** the statistical FDR screen. This four-way convergence is the most robust finding in the study
and is the right anchor for beat 6.

**Requires confirmation.**
- Stability selection ran on the **full cohort**, so it cannot serve as a leakage-free feature mask. Part 5 L339
  says this; keep it.
- The Borda consensus score mixes three incommensurable scales and includes imputed zeros (§2.8 item 4). Prefer
  reporting the three signals separately, as Figure 5 does, over the composite score.

#### 3.6.3 SHAP and interactions — supplementary only

**Evidence.** Mean |SHAP| over the 15 explained rows: `LV` 1.2368, `WBC` 1.1648, `LDL` 0.6408, `eGFR` 0.4713.
The residual bundle ("sum of 72 other features") is **1.41** — larger than any single feature.

**Cites.** **Supplementary Figures S6–S8**, **Supplementary Table S11**.

**Safe claims.** Only with the full qualifier: *"attributions for 15 individual VLST cases; no control patients
were explained."* Note that the residual bundle exceeding every named feature means importance is **not**
concentrated in the top four.

**Requires confirmation — critical (EM CONFIRM #11).** These figures must not appear in the main text and must
never be described as global importance. Similarly the k-SII network and UpSet plots are **one patient** and must
be labelled as such in every caption. Part 5's own text is careful (L151, L223, L245) — carry that care forward
rather than losing it in translation to the manuscript.

### 3.7 Calibration — *beat 5 (calibration half)*

**Purpose.** Separate ranking skill from probability quality.

**Evidence — undisputed (all five classical models agree across both runs):** Brier — CatBoost 0.0090,
XGBoost 0.0093, LightGBM 0.0096, random forest 0.0147, logistic regression 0.0543.

**Evidence — disputed:** TabPFN Brier is **0.0360** in the stored figures and **0.0060** in the stored notebook
text (§0.1).

**Cites.** **Figure 4** — conditional.

> **DRAFTING BRANCH A — if Brier = 0.0360 is confirmed.**
> Beat 5 stands as requested. Safe claims: TabPFN ranks best but calibrates worse than the tree ensembles;
> logistic regression is worst on both counts; ranking skill and probability quality come apart. Recommend
> reporting a post-hoc recalibration (Platt or isotonic on inner folds) as an obvious remedy.

> **DRAFTING BRANCH B — if Brier = 0.0060 is confirmed.**
> Beat 5 **inverts**. TabPFN is best on discrimination *and* on Brier. The paper's calibration message becomes:
> apparent calibration is excellent, but because control sampling makes the 1.77% prevalence a design artefact
> (§2.1), the absolute probabilities are **not transportable** to a real PCI population regardless of the Brier
> score. That is a more interesting and more honest message than Branch A. Part 4's current narrative sentences
> ("overestimate event probability", "not a well-calibrated risk engine", "worse than the tree ensembles") must
> be **deleted**.

**Requires confirmation — blocking (EM CONFIRM #4, #21).** Resolve the Brier value. Do not draft either branch
before then. In both branches, add calibration slope and intercept — the repository reports only Brier and a
visual reliability curve.

---

## DISCUSSION

Six paragraphs, in this order.

### D1 — Principal findings
Restate the four anchors: (i) VLST is rare at 1.77%, so PR-AUC and not ROC-AUC is the discriminating metric —
every model exceeded ROC-AUC 0.92 while PR-AUC ranged 0.34–0.85; (ii) TabPFN led PR-AUC in 5/5 folds;
(iii) `WBC`, `eGFR`, `LV` recur across all four extraction routes; (iv) the routes disagree elsewhere for
identifiable structural reasons.
**Safe.** All four. **Confirm.** Nothing new beyond the Results confirmations.

### D2 — Why PR-AUC, and why ROC-AUC misleads here
The strongest transferable methodological message in the paper. Anchor on the 0.92-vs-0.34 contrast.
**Safe.** Fully supported by Table 3.

### D3 — Why the feature sets differ (beat 7)
Develop the mechanism table from §3.5. The argument is that marginal testing and model-based selection answer
different questions, so complementarity is the expected result.
**Safe.** The collinearity, surrogate and interaction mechanisms.
**Confirm.** Must concede that the encoding and pool-truncation mechanisms are artefacts of this implementation,
not general properties.

### D4 — Interpretability: what a foundation model can and cannot tell you
Nonlinear PDP shapes and stable recurrence are informative; 15-case SHAP and one-patient k-SII are not
population evidence. This is a genuine contribution — a worked example of interpretability claims that outrun
their sample.
**Safe.** With the §3.6 qualifiers intact.

### D5 — Leakage as a first-class methodological result
Develop §2.10: a variable that looks like an ordinary covariate but is defined differently for cases and
controls; a single-rule check that exposes it; a with/without sensitivity analysis. Generalise to the point that
rare-event tabular studies should routinely audit for class-dependent variable definitions.
**Safe.** With the numbers from EM §7.5.
**Confirm.** This paragraph is where the case-control design must be acknowledged (§2.1 item 2).

### D6 — Comparison with prior work
**[GAP]** The repository has **zero citations**. Entirely external. Compare against published VLST risk factors
and against prior TabPFN evaluations on clinical tabular data.

---

## LIMITATIONS

Ordered by severity. Each is a confirmed defect from the evidence map, not a boilerplate hedge.

| # | Limitation | Evidence-map reference |
| --- | --- | --- |
| 1 | **Measurement timing of laboratory and echocardiographic variables is undocumented.** If any are event-time values for cases, the study describes presentation rather than prediction. | EM §4.3 |
| 2 | **Control sampling implies a case-control frame**; the 1.77% prevalence is not an incidence, and prevalence-dependent metrics are not transportable. | EM §4.2 |
| 3 | **No external, temporal, or geographic validation.** Every number comes from the same 5,185 rows. | EM §6.6 |
| 4 | **TabPFN is not reproducible** across runs despite a fixed seed, and depends on a remote service whose version is unrecorded. | EM §12.2, §12.10 |
| 5 | **Classical baselines were untuned**; TabPFN ran at high effort. The comparison is unmatched. | EM §6.3 |
| 6 | **Models did not receive identical feature representations.** | EM §6.4 |
| 7 | **No confidence intervals and no paired model comparisons.** | EM §12.9 |
| 8 | **Feature-selection results are exploratory:** test-set scoring, an arbitrary 40-column candidate pool, and a coalition value function computed on an 87.5%-positive sample. | EM §4.4, §5.7 |
| 9 | **SHAP covers 15 VLST cases and no controls; k-SII covers one patient.** | EM §5.9 |
| 10 | **Low events-per-variable** (≈ 5.4 for the multivariable model), so adjusted estimates are unstable. | EM §2.2 |
| 11 | **Observational design:** confounding by indication precludes causal or protective language for `1.1:1Post dilation`, `Clopidogrel`, and every procedural variable. | EM §12.12 |
| 12 | **Selective reporting risk:** nineteen notebooks exist, five analyses are reported. | EM §12.11 |

**Requires confirmation (EM CONFIRM #16).** Item 12 obliges a declaration of the unreported work — the
`failed_hypothesis/` directory (blending, oversampling, synthesis, false-positive mining, anomaly detection,
LLM-tabular), the causal analysis, and the TabPFN playground. Without it, the TabPFN result cannot be
distinguished from selection over many attempts.

---

## CONCLUSION

Three sentences, no more.

**Safe template.**
> In a dataset of 5,185 patients with 92 very late stent thromboses (1.77%), a tabular foundation model achieved
> higher out-of-fold precision–recall performance than five conventional classifiers in every cross-validation
> fold (PR-AUC 0.85 vs 0.34–0.70), while all models exceeded ROC-AUC 0.92 — showing that ROC-AUC is
> uninformative at this event rate. Leukocyte count, estimated glomerular filtration rate and left-ventricular
> dimension were recovered independently by marginal hypothesis testing, by model-based feature selection and by
> foundation-model interpretability, whereas the remaining feature sets diverged for identifiable structural
> reasons. These findings are exploratory and require confirmation of measurement timing and external validation
> before any clinical application.

**Requires confirmation.** The third sentence is not optional. Do not replace it with a softer hedge, and do not
add any sentence proposing clinical use.

---

## 4. Supplementary material

| ID | Content | Origin |
| --- | --- | --- |
| S1 | Preprocessing and encoding by model arm | §2.2 |
| S2 | Test-selection rationale, all 24 continuous variables | Part 1 Table R |
| S3 | Full model specifications | Part 4 Table 0 (CatBoost row corrected) |
| S4 | **Leakage sensitivity: with vs without `Time since stent implantation`** | EM §7.5 — currently unreported |
| S5 | Domain strength summary | Part 1 Table S1 |
| S6–S8 | SHAP summary/bar/waterfall; k-SII network; k-SII UpSet | Part 5 Figs 3, 5, 7, 8, 9 |
| S9 | Interaction screen, **all 16 rows** | `domain_interaction_screen.csv` |
| S10 | Mutual information top 15 | Part 5 Table 1 |
| S11 | Mean \|SHAP\|, 15 cases | Part 5 Table 4 |
| S12 | Per-fold performance metrics | EM §7.4 |
| S13 | Nested-CV operating points, all six models | EM §7.3 |
| S14 | Feature-selection detail: LOCO/SHAP/FFS by model and metric | Part 2 Tables 1–4 |
| S15 | Venn, presence heatmap, reason buckets, domain counts | Part 3 Figs 1–4 |
| S16 | Correlation clustermaps | Part 1 Figs S2a–d |
| S17 | Per-domain and joint multivariable models | Part 1 Figs S3, S4 |
| **Data** | `oof_predictions.csv`, `fold_thresholds.csv`, `fold_metrics.csv`, `nested_cv_operating_point.csv` | **Must be regenerated — never committed** |

---

## 5. Items cut, and why

| Cut | Reason |
| --- | --- |
| Part 4 Table 3 (confusion counts) | Identical to Table 2's last four columns |
| Part 4 Figure 3 (confusion heatmaps) | Same counts again, as a picture |
| Part 2 Figure 7 | Its own caption: "same numbers as Table 3" (L94) |
| Part 2 Figures S1, S3 | Own captions: "Paper restyle: Figure 1 / Figure 2" (L280, L292) |
| Part 2 Figure 6 | A nine-row table drawn as bars (L132) |
| Part 2 Table 5 (priority ranks) | Own caption says most rows are string-matching failures (L247) |
| Part 5 Figures 10–12 | Own caption: "a second view of the **same one-row explanation**" (L245) |
| Part 5 Figure 6 | "Same 15-row attributions as Figure 3" (L181) |
| Part 5 Figure 4 (Age scatter) | Documents a null: "almost all points sit at SHAP = 0" (L165) |
| Part 1 Figure 3 (effect sizes) | Mixes Cohen's d and Mann–Whitney r on one axis — replaced by Table 2 with separated metrics |
| Part 5 Figure 13 (Borda) | Duplicates Table 4's TabPFN column; the composite score mixes incommensurable scales |
| `paper_domain_feature_map.csv` | Byte-identical to `domain_feature_map.csv` (verified) |

---

## 6. Drafting order

1. **Answer EM CONFIRM #1 (measurement timing) and #2 (control sampling).** Nothing else is worth writing first.
2. **Re-run `baseline_plus_tabpfn.ipynb` once**, commit the OOF predictions, and fix the TabPFN Brier — this
   selects Branch A or B for §3.7 and D1.
3. **Respecify the multivariable model** (§2.4). Table 2 and Figure 2 depend on it.
4. **Add bootstrap CIs and a paired model test** to Table 3.
5. **Build Table 1 and Figure 1.**
6. Draft Methods, then Results, then Discussion.
7. Draft the Abstract last, from the finished Results.
