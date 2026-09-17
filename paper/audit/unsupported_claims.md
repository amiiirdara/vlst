# Unsupported claims

Items the inspect set does **not** support. Do not invent values. Do not use these phrases in the manuscript unless a later, in-scope source is added.

Scientific rules applied: Wang 2020 is a historical comparator on the derivation cohort only. TSSI is not a baseline predictor. Association ≠ prediction. TabPFN thinking-high ≠ TabPFN local.

---

## 1. Phrases that are not supported for this pack

| Phrase | Why unsupported | What the files allow instead |
| --- | --- | --- |
| External validation (of the ML models, TabPFN, or nested-CV metrics) | Front matter limitation 1: every Part 4 number is nested CV on 5,185 derivation rows. Shantou is not in the repository. | Nested-CV **prediction performance** on the derivation cohort. Wang’s Shantou c = 0.82 describes **Wang’s score only**. |
| Clinically validated / clinical utility / clinically useful | Explicitly banned in `00_front_matter.md`. No decision-curve analysis. No net benefit. | Discrimination, calibration (Brier / reliability curves), and association on this file. |
| Independent risk factor / independent predictor | Banned. Table 4b is a sparse unweighted logit with EPV ≈ 7.1, not an independence claim. | **Association** after adjustment in Table 4b; name the specification. |
| Causal factor / causal language | Banned. Figure 6 caption already refuses a causal claim. | Association; confounding / shared information. |
| Protective effect of post-dilation or clopidogrel | Front matter: OR < 1 is not a treatment benefit (confounding by indication). `Clopidogrel` is follow-up persistence, not an index prescription. | Lower modelled odds of recorded VLST when the flag is 1, with the DAPT-timing caveat. |
| Personalised / personalized risk | Root `README.md` slogan only. Front matter: not an individual-level model. | Global classifier / global logit. |
| Validated inflammatory marker (WBC) | Front matter item 7: Wang dropped WBC because infection could not be ruled out. Ranking WBC is a discrepancy to report. | Univariate association and selector/SHAP attribution; not validation. |

---

## 2. Methods that were requested for extraction but are absent

| Topic | Status | Notes |
| --- | --- | --- |
| Firth logistic regression | **Implemented** as Table 4b 13-covariate association sensitivity | Do not describe it as nested-CV or GridSearch. Primary association model remains Table 4b MLE. |
| Likelihood-ratio tests (Table 4 / 4b main-effects logits) | **Not used** for Table 4/4b intervals | Table 4/4b intervals are **Wald**; Table 4b CSV also stores a 2,000-replicate percentile bootstrap. |
| Likelihood-ratio tests (exploratory interactions) | **Present** in Part 1 Supplementary Table S2 | 16 pair × VLST LR tests vs main-effects-only; freeze `statistical_analysis.likelihood_ratio_tests`. Hypothesis-generating only. |
| NPV | **Not printed**; identity TN/(TN+FN) from frozen Table 2 2×2 if quoted | Do not present as a separately computed notebook metric. |
| Outlier handling | **IQR 1.5 screen in `eda.ipynb`**; rows not dropped; unused in Part 4 / Table 4b | Methods: detection-only. Do not invent winsorisation. Do not quote the printed count table unless a new exhibit. |
| Repeated nested CV / extra random seeds beyond those named | **Not present** | One nested 5×4 CV; outer `random_state=42`; models `random_state=42`; bootstrap seed 42. |
| Hyperparameter grids for Part 4 | **Not used** | Classics = library defaults + class weighting. Inner loop = F1 threshold only. TSSI GridSearch `best_params_` are **not** imported. |
| `balance_probabilities` on Version 4 local TabPFN | **Omitted** | Do not describe Version 4 local as balanced-prior. |
| Complete 81-name baseline predictor list | **Frozen** as CSV header listing | `paper/table_s0_baseline_predictors.md`; YAML `complete_81_name_list`. Spellings as stored. |
| `preprocessing.ipynb` scaler / imputer / encoding for TSSI arrays | **Not inspected** (file not in scope) | TSSI notebooks load `data/processed/*.npy`. Do not invent those steps. |
| Cox linear predictor re-fit | **Absent** (front matter B11) | Frozen integer points only. |
| Dangas decision-curve analysis | **Absent** | Cited c = 0.66 is Wang’s published comparison, not computed here. |
| Shantou scoring in this repo | **Absent** | Cannot be obtained this cycle (front matter). |
| Separate ethics approval for this repository analysis | **Not claimed** | Cite Wang 2020 for NCT03491891 / ethics 2013-256. |
| Pre-registration of this analysis | **Not done** | Front matter: not separately pre-registered. |

---

## 3. Numbers that must not be filled in

Do not invent:

- Sample sizes other than 5,185 / 92 / 5,093 and the split counts already printed (Part 2 4148/1037; Part 5 3629/1556; fold events 18–19).
- Confidence intervals other than: Wang published derivation c 0.75–0.85 (cited); Table 4/4b Wald intervals; Table 4b bootstrap in CSV; Part 4 stratified OOF bootstrap (Table S-CI / S-Δ).
- Hyperparameters not in `RUN_MODELS` / Table 0 (e.g. RF `max_depth`, LightGBM `num_leaves`, CatBoost `depth`, TabPFN pip versions).
- A raw stent-string count of 99 as encoder `n_raw` (that is the EDA χ² helper only). Quote **106 → 9**.
- Calibration slope/intercept, ECE, or decision-curve net benefit. NPV only as Table 2 identity.
- External-test n, events, or c-statistics for TabPFN or the classic nested-CV models.

---

## 4. Claims that look supported but are not, if mis-scoped

| Unsafe claim | Why it is unsupported |
| --- | --- |
| “TabPFN was validated” because Wang’s score was tested on Shantou | Contagion of the word “validation.” Nested CV ≠ external validation; Shantou ≠ this pack’s models. |
| “TabPFN Brier is poor / best” without naming the arm | Thinking-high Brier 0.0064 (best of seven); local 0.0102 (booster band). Collapsing arms is false. |
| “Local TabPFN has the worst Brier” | Explicitly false on Version 4 (Part 4 models paragraph). |
| Nested-CV recall 0.8152 (thinking-high) or 0.8478 (local) | Those are **pooled F1** (Table 3), optimistic. Nested is 0.7065 and 0.6848 (Table 2). |
| Part 5 consensus as the Part 4 feature set | Both reports forbid using attribution as a leakage-free mask. |
| k-SII network as cohort interactions | One held-out VLST=1 row (5176), budget 256. |
| Table 4 as the clinical multivariable model | Unidentified (complementary post-dilation; eGFR with CKD flags); EPV 5.4. |
| TSSI 70/30 metrics as nested-CV performance | Different split, GridSearch, optional SMOTE; leakage demonstration only. |
| Integer Wang score as a nested-CV eighth arm | Frozen points; weights not re-fit. |
| Intermediate-bin n = 1,837 on this CSV | This file has n = 1,577 in that bin; Wang printed 1,837. |
| “No VLST score exists” | Front matter: false. Wang 8-variable Cox score exists. |
| PDP y-axis as Part 4 nested-CV risk | Part 5 PDP is train-split empirical prior, near prevalence ~0.018. |
| SHAP 15+15 or k-SII row 5099 / 5093 | Superseded dumps; live Version 5 is 1,556 held-out rows and row 5176. |
| Cre mean\|SHAP\| 0.158 as Table 5 | Old dump; live Cre mean\|SHAP\| 0.2449 with train MI 0.000000. |
| Jaccard 5/35 | Live 5/28 ≈ 0.18 (print 0.1786). |

---

## 5. Inclusion / exclusion and “same cohort”

Supported: Wang 2020 flow 6,038 → 5,185, cited in front matter and Table C caption; `VLST.csv` is that derivation file.

Unsupported: a de-novo eligibility diagram reconstructed from this repository alone (the CSV is the analysed n). Do not invent additional inclusion rules beyond Wang as cited.

Same cohort: supported as the same 5,185 rows. Unsupported: that every analysis used the same **split**. They did not (full-cohort association; 80/20 selectors; nested 5×4; Part 5 70/30; TSSI 70/30).

---

## 6. Train/test language to avoid

| Avoid | Use |
| --- | --- |
| “Test set performance” for Part 4 without saying nested OOF | Nested-CV out-of-fold prediction performance |
| “External test” for the Part 5 1,556 rows | Held-out split of the **same** derivation file, for attribution |
| “Validation of TabPFN” for Wang fold-mean c | Frozen historical comparator evaluated on the same folds |

---

## Freeze status

**READY FOR FREEZE**

Live Part 4 table PNGs/CSVs now match Version 4 OOF (`model_comparison.csv` / `nested_cv_operating_point.csv`): local PR-AUC **0.6742**, Brier **0.0102**, nested LightGBM 5060/33/30/62, nested local *t* **0.166**. Encoder quote is **106 → 9**. Front matter quotes Table 4b (0.152 / 0.480). Part 3 no longer says “external-validation bar.” Table S-TSSI names the SMOTE mismatch. Concat rebuilds Parts 0–5 from the standalone files. Use `rebuild_part45_paper_exports.py --concat-only` so `main()` cannot restore an old dump.

Still not result claims: root README slogan; evidence-map Revision 7 diary numbers; EDA sidecar “Raw levels=99” (χ² remains on 9 levels). `frozen_results.yaml` is the next step.
