# VLST paper — proposed structure

**Companion document:** [`paper_evidence_map.md`](paper_evidence_map.md). Every number cited below is traced there;
section references of the form "EM §7.1" point into it.

**Source Markdown reports** (canonical copies under `paper_results/`):

| Short name | Path |
| --- | --- |
| Part 0 | `paper_results/00_front_matter.md` (W2–W5: motivation, EPV, limitations, terminology) |
| Part 1 | `paper_results/01_eda/EDA_paper_figures_and_tables.md` |
| Part 2 | `paper_results/02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md` |
| Part 3 | `paper_results/03_stats_vs_ml/feature_extraction_comparison.md` |
| Part 4 | `paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md` |
| Part 5 | `paper_results/05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md` |

---

## 0. Two structural decisions before drafting

### 0.1 Beat 5 is now writeable (local TabPFN Brier is settled)

The requested story said *"TabPFN provides stronger discrimination than the classical baselines but has weaker
calibration."* **This Kaggle snapshot does not support the first half on PR-AUC.** LightGBM is first
(PR-AUC **0.6937**); TabPFN (local) is third (**0.6754**) and first on ROC-AUC (**0.9845**). The second half
**is** true for this run: TabPFN (local) Brier **0.0673** is the **worst** of the six (XGBoost 0.0088 is best).
Notebook, embedded figures, and `paper_figures/` agree (EM §12.2). Historical thinking-high client Brier 0.0060
vs PNG 0.0360 is a different arm and must not be quoted here.

**Consequence for this outline.** Draft beat 5 as: *LightGBM leads rare-event ranking; local TabPFN leads ROC-AUC
and is the worst-calibrated of the six.* Do not restore "TabPFN wins every fold" or "best Brier."

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
| **Table 3** | Table | Nested cross-validation performance of six models | 3, 5 | Part 4 Table 1 + notebook §7.3/§7.4 | **Current** — LightGBM PR-AUC 0.6937 |
| **Figure 3** | Figure | Precision–recall and ROC curves, nested-CV out-of-fold | 3, 5 | Part 4 Figure 1 | Reuse as-is |
| **Figure 4** | Figure | Calibration (reliability curves), nested-CV out-of-fold | 5 | Part 4 Figure 2 | Current — TabPFN (local) Brier 0.0673 |
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

Drafted in `paper_results/00_front_matter.md` (W2). Four beats: VLST as ARC definite ST > 1 year (Wang 2020);
why prediction is hard (1.77%, EPV ≈ 5.4); why TabPFN is in the comparison (with the unequal-tuning disclosure);
objectives as association / nested-CV prediction / catalogue comparison — not implementation. Do not write as if
no VLST score exists; Wang’s 8-variable integer points are now scored frozen on these rows (Part 4 S-Wang; ROC-AUC 0.8013, PR-AUC 0.1032). The Cox linear predictor, Dangas DCA, and Shantou file remain absent (B11).
The README word “Personalized” is disallowed as a result claim (single global model).

### Paragraph 1 — Clinical problem
**Purpose.** Establish that VLST is rare, late, and consequential.
**Evidence.** Wang 2020 (ARC 2007 definite ST > 1 year; incidence 1.77% on this cohort; Dangas LST c = 0.66 vs Wang Cox c = 0.80 / 0.82).
**Source.** `paper_results/00_front_matter.md`. The old “[GAP] nothing in the repository” is closed.

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

**Stored export** (figures still shown): top-12, SHAP universe 24, LOCO cap 40, FFS pool 30, three metrics.
**Paper protocol** (current notebook, no smoke switch): top-20, SHAP 40/40/3, LOCO 60, FFS 24×12, PR-AUC only,
400 rounds, seven models, 185 encoded columns.

**Cites.** Table 4; Part 2's own tables to **Supplementary Tables S6–S8**.

**Safe claims.** The mechanism of each selector; the within-model and cross-model intersections as *descriptive*
results.

**Requires confirmation — blocking (EM CONFIRM #6).** The **stored figures** are a prior reduced export and
still have three defects (Part 2/3 markdowns now disclose them). The **notebook** is the paper protocol but
has not been re-run:
1. **Stored LOCO pool is not a ranking.** The executed run used the first 40 `ColumnTransformer` columns
   (23 continuous, then the first 17 binaries). Current code ranks that cap by train importance.
2. **Stored LOCO/SHAP scored on the test set.** Current code scores on a val slice of the full cohort (no unused outer test).
3. **Stored SHAP value function used an 87.5%-positive sample** (28 test events + 4 controls). Current code
   draws a stratified val sample at cohort prevalence.

Until Kaggle re-export, do not quote Parts 2 and 3 as the paper result.
**The paper must not describe these features as "validated" or "required by the model".**

Do not ship a smoke switch in the methods notebook (removed). After the paper-protocol run, replace the
figures. Correct Part 2 Table 0 if it still says CatBoost used "Ordered boosting"; the factory sets
`task_type="GPU"`, where CatBoost supports only **Plain** (EM §12.7).

### 2.7 TabPFN

**Purpose.** Specify the foundation model and the reproducibility caveat.

**Evidence.** Local TabPFN: `from tabpfn import TabPFNClassifier`, `n_estimators="auto"`,
`balance_probabilities=True`, Kaggle Tesla T4, checkpoint `tabpfn-v3-classifier-v3_default.ckpt`. Evaluated in
the identical nested-CV loop as the five baselines. Shared 9-level stent encoder before the split. The client
thinking-high constructor remains in the notebook (`RUN_MODELS["TabPFN"]=False`) and was not fit
(`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt`).

**Cites.** Table 3, Figure 3, Figure 4.

**Safe claims.** The configuration; that TabPFN (local) used the same folds and the same pooled out-of-fold
evaluation as every baseline; that it is **not** thinking-high.

**Requires confirmation.**
- Record the local `tabpfn` package / checkpoint version (partially done: `tabpfn-v3-classifier-v3_default.ckpt`).
- `oof_predictions.csv` and `fold_thresholds.csv` were written to `/kaggle/working/` and never committed (EM B2).
- Do not quote historical thinking-high Brier 0.0060 vs 0.0360 as this run.

### 2.8 Interpretability

**Purpose.** Specify the five TabPFN interpretability signals and, for each, the sample it was computed on.

**Evidence** (`.nbdump/code__modeling__interpretability__tabpfn_interpretability.txt`):

| Signal | Sample | Backend | Note |
| --- | --- | --- | --- |
| Mutual information | full cohort, 5,185 | sklearn, 0 TabPFN calls | screening only; CSV will store all 81 scores on next `[1a]` |
| Stability selection | full cohort; forward SFS keeping 10 of 81, 5-fold CV, AP scoring, **10 seeds**, ~8.6 h | local TabPFN | **the most defensible signal here**; `balance_probabilities=True` |
| PDP | **full cohort** fit and average; 4 continuous + 6 binary | local TabPFN | **`balance_probabilities=False`** (empirical prior / not Part 4 risk). Stored PNGs still the old True + 70/30 run |
| SHAP (shapiq SV) | **15 VLST=1 + 15 VLST=0** in code; stored PNGs are 15 VLST cases only | local TabPFN after client failure | budget 256; ranking scale `True` |
| k-SII / SHAP-IQ | **one** VLST=1 row from that 15+15 slice | local TabPFN after client failure | budget 256 |

**Cites.** Figure 5; SHAP and k-SII to **Supplementary Figures S6–S8**.

**Safe claims.** Mutual information and stability selection as full-cohort screens; PDP as a model-average
response shape on the empirical-prior scale after re-run.

**Requires confirmation — critical (EM CONFIRM #11).** Disclosures for the **stored** figures, and what **code now** does:

1. **Stored SHAP PNGs are 15 VLST cases only.** Code now samples 15 VLST=1 + 15 VLST=0. Re-run `[3/5]`. Until then captions are **[STALE]**.
2. **k-SII is one patient** (a VLST=1 row from the 15+15 slice). Part 5 already says this — keep it. It is **not** a cohort
   interaction screen. The only cohort-level interaction evidence in the study is the 16-pair LR screen (§2.4).
3. **PDP vs ranking scales are split.** Ranking / SHAP / stability keep `balance_probabilities=True` so a 1.8%
   outcome is visible. PDP only uses `False` (empirical prior; y-axis near ~2%; labeled **not Part 4 risk**).
   Do not mix True and False on one axis. Do **not** quote stored Table 3 0.24 or Figure 1 ~0.6 as clinical risk.
4. **The zeros in Part 5 Table 5's mutual-information column are imputed** in the stored export — `Cre` and
   `No.of stents per lesion` fell outside the MI top-15. Code now writes all 81 MI scores and does not `fillna(0)`.
   `Fast-Glu` and `ZES` blanks close on the next `[1a]` sklearn re-run.

Also: stored Figure 1 still sweeps `Stent type-SES` as a numeric axis. Code now drops that panel.

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
1. **Report the nested operating point, not the pooled one.** **[closed in Part 4 reports]** Table 2 is the honest nested print (LightGBM recall **0.6630**, F1 0.6667, TP 61; TabPFN local recall **0.6848**, TP 63). Figure 3 / Table 3 are the pooled cut, labelled as optimistically biased; do not quote TabPFN local pooled recall 0.8261 as nested.
2. **No confidence intervals exist on any metric, and no paired model comparison was run.** "LightGBM is first,
   TabPFN (local) is third on PR-AUC" is a point-estimate ordering. Fold SDs: LightGBM PR-AUC 0.6941 ± 0.0917;
   TabPFN (local) 0.6739 ± 0.0812; LightGBM higher in 3 of 5 folds. Add bootstrap CIs and a paired test
   before any superiority language.
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

**Published clinical baseline (frozen Wang 2020 integer score).** Same 5,185 rows, published Table 2 points, not re-fit. Full-cohort ROC-AUC **0.8013** (Wang published c = 0.80), PR-AUC **0.1032**; fold-mean ROC-AUC 0.8005 ± 0.0607 on the Part 4 outer folds (evaluation only). Source: `wang_vlst_score.ipynb` / Part 4 Table S-Wang. This is the missing comparator: nested-CV models vs the score already published on these patients. Encoding traps (SES → `PES`; 4 points on `No postdilation`) are disclosed there. Not Shantou, not a Cox linear predictor.

**Evidence** (pooled nested-CV out-of-fold, n = 5,185, 92 events; EM §7.1, §7.4):

| Model | PR-AUC (pooled) | PR-AUC (fold mean ± SD) | ROC-AUC (pooled) | ROC-AUC (fold mean ± SD) |
| --- | --- | --- | --- | --- |
| LightGBM | **0.6937** | 0.6941 ± 0.0917 | 0.9681 | 0.9695 ± 0.0164 |
| XGBoost | 0.6815 | 0.6928 ± 0.1288 | 0.9439 | 0.9431 ± 0.0418 |
| CatBoost | 0.6172 | 0.6353 ± 0.0540 | 0.9594 | 0.9612 ± 0.0137 |
| Random forest | 0.4865 | 0.5034 ± 0.0793 | 0.9209 | 0.9206 ± 0.0423 |
| Logistic regression | 0.3326 | 0.3451 ± 0.1213 | 0.9224 | 0.9235 ± 0.0251 |

Prevalence reference for PR-AUC: **0.0177**.

**Source.** `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L1071–1086, L1122–1153 — and Part 4
Table 1, which matches that dump.

**Cites.** **Table 1**, **Figure 1**.

**Safe claims.**
- LightGBM (0.69 PR-AUC) and XGBoost (0.68) outperform bagging (0.49) and the linear model (0.33). CatBoost is fourth on this 9-level-encoder run (0.62), not second.
- **The key rhetorical point of this subsection:** all five classic models exceed ROC-AUC 0.92, yet PR-AUC ranges from
  0.33 to 0.69. ROC-AUC is nearly uninformative here; PR-AUC separates the models.

**Requires confirmation.**
- Baselines are untuned (§2.5).
- No CIs, no paired tests (§2.9 item 2). LightGBM fold SD is 0.092 — the 0.69 point estimate is noisy.
- Do **not** claim clinical usefulness from any ROC-AUC (EM §12.3).

### 3.4 TabPFN versus baselines — *beats 3 and 5 (discrimination half)*

**Purpose.** The paper's nested-CV comparison. On this run TabPFN (local) does **not** lead PR-AUC.

**Evidence.**

| Metric | LightGBM (best PR-AUC) | TabPFN (local) | Wang 2020 integer score (frozen) |
| --- | --- | --- | --- |
| PR-AUC pooled | **0.6937** | 0.6754 | 0.1032 |
| PR-AUC fold mean ± SD | 0.6941 ± 0.0917 | 0.6739 ± 0.0812 | 0.1134 ± 0.0518 |
| ROC-AUC pooled | 0.9681 | **0.9845** | 0.8013 |
| ROC-AUC fold mean ± SD | 0.9695 ± 0.0164 | 0.9846 ± 0.0030 | 0.8005 ± 0.0607 |
| Per-fold PR-AUC | 0.753, 0.714, 0.540, 0.772, 0.692 | 0.638, 0.635, 0.583, 0.727, 0.786 | LightGBM higher than TabPFN local in **3/5** folds |

**Honest nested operating point** (EM §7.3): LightGBM precision 0.6703, recall 0.6630, F1 0.6667,
TN/FP/FN/TP = 5063/30/31/61, threshold 0.121 ± 0.085. TabPFN (local): precision 0.5478, recall **0.6848**,
F1 0.6087, 5041/52/29/63, threshold 0.915 ± 0.012.

**Cites.** **Table 2** (honest nested operating point), **Figure 1** (ranking), **Supplementary Table S-Wang**.

**Safe claims.**
- LightGBM ranked highest on pooled PR-AUC; higher than TabPFN (local) in **3 of 5** outer folds. Do not write
  "TabPFN wins every fold."
- TabPFN (local) ranked highest on ROC-AUC (0.9845) and worst on Brier (0.0673).
- On the same derivation rows, nested-CV LightGBM PR-AUC **0.6937** versus the frozen published Wang integer score **0.1032** (TabPFN local PR-AUC 0.6754; ROC-AUC 0.9845 vs Wang 0.8013). That is not Shantou and not a Cox re-fit.
- At the nested operating point LightGBM identified 61 of 92 events with 30 false positives; TabPFN (local)
  identified 63 events with 52 false positives.

**Requires confirmation.**
- Add a paired test before the word "superior" (EM CONFIRM #10).
- Disclose untuned classics (§2.5) and that this TabPFN is **local, not thinking-high**.
- Do not report accuracy (0.99 for every model; it is prevalence, not skill).
- Nested vs pooled operating points are labelled in Part 4; quote Table 2 (LightGBM recall 0.6630 / TabPFN local 0.6848), not Figure 3 pooled TabPFN recall 0.8261.

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

**Requires confirmation — critical.** The **stored** y-axis is a **balanced-prior model output, not absolute risk**.
Code now uses `balance_probabilities=False` on PDP only, labeled **empirical prior / not Part 4 risk**. Do not
quote "rises toward ~0.6" as a 60% event probability. Relabel after re-run. Drop the `Stent type-SES` PDP panel
(already dropped in code; stored Figure 1 still has it).

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

**Evidence (this Kaggle local-TabPFN run; notebook and figures agree):** Brier — XGBoost **0.0088**, LightGBM 0.0093,
CatBoost 0.0101, random forest 0.0143, logistic regression 0.0563, TabPFN (local) **0.0673** (worst of six).

**Cites.** **Figure 4**.

Beat 5: TabPFN (local) does **not** lead PR-AUC and **is** the worst-calibrated of the six. Historical
thinking-high Brier 0.0060 is not this run. Absolute probabilities remain non-transportable because prevalence
is a property of this derivation cohort (§2.1), regardless of Brier.

**Requires confirmation.** Add calibration slope and intercept — the repository reports only Brier and a
visual reliability curve. Do not draft the old Branch A/B (0.0360 vs 0.0060) against this snapshot.

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

Drafted in `paper_results/00_front_matter.md` (W3). Do **not** restore the withdrawn selective-reporting item
(`failed_hypothesis/` and `tabpfn_playground.ipynb` are out of scope: D1–D2). Do **not** call this a case-control
sample: Wang 2020 establishes a consecutive complete-follow-up cohort; 1.77% is published incidence (EM §2.3, §4.2).

| # | Limitation | Evidence-map reference |
| --- | --- | --- |
| 1 | **No external or temporal test of the ML models.** Wang’s Cox score was tested on Shantou; those rows are not here. | EM §6.6 |
| 2 | **Binary classification vs published Cox analysis.** Follow-up time is the Cox axis; as a covariate it leaks (Part 4 S-TSSI). | EM §4.1–4.2 |
| 3 | **EPV ≈ 5.4** on the 17-covariate logit; collinear blocks remain. | EM §2.2, W4 |
| 4 | **Unused client thinking-high arm** is non-deterministic; this snapshot is local TabPFN Brier **0.0673**. | EM §12.2, §12.10 |
| 5 | **Classical baselines untuned**; TabPFN is local (no thinking). Unequal model class, not thinking-high vs defaults. | EM §6.3 |
| 6 | **Classics still one-hot** the 9-level brand column (~89); TabPFN (local) sees it natively. | EM §6.4 |
| 7 | **No CIs and no paired test** of LightGBM vs TabPFN (local). LightGBM PR-AUC higher in 3/5 folds. | EM §12.9 |
| 8 | **Part 2 catalogues are discovery on an 18-event val slice**, not a Part 4 mask. | EM §4.4 |
| 9 | **DAPT columns are post-baseline**; **WBC** was excluded by Wang; **`LV` unnamed**. | EM §3, A1 |
| 10 | **Observational design:** no causal / “protective” language for post-dilation or clopidogrel. | EM §12.12 |

---

## CONCLUSION

Three sentences, no more.

**Safe template.**
> In a dataset of 5,185 patients with 92 very late stent thromboses (1.77%), nested cross-validation of six
> classifiers found LightGBM first on precision–recall (PR-AUC 0.69) and local TabPFN first on ROC-AUC (0.98)
> but worst on Brier (0.067); all models exceeded ROC-AUC 0.92 — showing that ROC-AUC is
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
2. **Part 4 nested CV is the Kaggle local-TabPFN run** (LightGBM PR-AUC 0.6937; TabPFN local Brier 0.0673).
   Commit OOF predictions (EM B2) so CIs can be added; do not re-enable thinking-high.
3. **Respecify the multivariable model** (§2.4). Table 2 and Figure 2 depend on it.
4. **Add bootstrap CIs and a paired model test** to Table 3.
5. **Build Table 1 and Figure 1.**
6. Draft Methods, then Results, then Discussion.
7. Draft the Abstract last, from the finished Results.
