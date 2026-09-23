# Title, authors, abstract, keywords

Numerical source of truth: `paper/frozen_results.yaml`. Trace tags: `docs/paper_evidence_map.md` (W1–W5, B3, B11, §4.6, §5.8–5.9, §7.3). Live nested key: `nested_cv_v35_antileakage_on`. Do not quote excluded unlabeled nested PR-AUC 0.9771 / 0.9635, Version 4 0.8553 / 0.6742, or evidence-map §7.1 historical rows.

---

## Title

VLST after ACS-PCI — association, nested-CV prediction, and TabPFN attribution on the Wang 2020 derivation cohort

<!-- TRACE: YAML study.title_or_working_title -->

---

## Authors

Authors are **not listed** in `paper/frozen_results.yaml`. Complete investigator names, affiliations, and corresponding author here before journal submission. This placeholder is not a freeze scalar.

---

## Structured abstract

### Background

Very late stent thrombosis (VLST) is Academic Research Consortium 2007 definite stent thrombosis more than one year after percutaneous coronary intervention (PCI). It is rare (92 events among 5,185 patients in this file; prevalence 0.0177) and analysed here as a binary label on Wang 2020’s derivation cohort. Wang published an 8-variable Cox integer score on these same rows (published derivation c-statistic 0.80 [0.75, 0.85]; Shantou external c-statistic 0.82 for **Wang’s score**, file not in this repository). Classic tabular classifiers on this file are easy to inflate if follow-up time remains in the matrix. Tabular prior-fitted networks (TabPFN) offer in-context learning on mixed-type tables without a per-dataset hyperparameter grid; this paper ranks four TabPFN arms and five class-weighted **library-default** tree/linear models under a shared nested-CV protocol after anti-leakage drops. The nine-arm table is that ranking, not a tuned-boosting contest. <!-- TRACE: W2; YAML study.*; wang_2020.reported_performance; W5 -->

### Methods

Retrospective derivation-cohort analysis of `VLST.csv` (The First Hospital of Jilin University; ACS, age ≥18 years, PCI 1 January 2014 – 1 June 2015; n = 5185). Outcome = binary `Stent thrombosis`. Nested evaluation implements ALL LEAKS OFF: drop identifiers, TSSI (mixed time-to-event vs completed follow-up), and WBC (recording-precision / batch marker); quantize named labs (`Cre`/`CaI`/`Fiberinogen`/`Fast-Glu`) **before** splitting; fit the stent-brand codebook on the **train fold only**; no SMOTE; clone scaler/OHE inside splits. Prediction uses stratified nested 5 outer × 4 inner CV (seed 42); the inner loop tunes the F1 threshold only. Five classic arms (logistic regression, random forest, XGBoost, LightGBM, CatBoost) are an **untuned reference panel** (library defaults plus class weighting); GridSearchCV winners from 70/30 leakage twins are **not** imported. Four TabPFN arms: local v3 / v3.5 (`tabpfn==9.0.0`) and hosted thinking-high v3 / v3.5 (`tabpfn-client==0.6.0`; effort high; metric average_precision). Primary ranking metric is PR-AUC at prevalence 0.0177. Uncertainty on pooled out-of-fold scores: stratified patient-level bootstrap (`n_boot=2000`, seed 42). Attribution uses two notebooks: 8-seed forward SFS, mutual information, and PDP on train; held-out SHAP / *k*-SII (same OFF cleaning). <!-- TRACE: Methods flags; nested_cv_v35_antileakage_on.pins; W3.8; §5.9 STABILITY_N_SEEDS=8 -->

### Results

Among nine nested-CV arms (anti-leakage ON), TabPFN thinking v3.5 had PR-AUC 0.9212 [0.8785, 0.9613], ROC-AUC 0.9963, Brier 0.0047; nested recall (sensitivity) 0.8370, precision 0.8851, F1 0.8603 (5083/10/15/77). Local TabPFN v3.5 had PR-AUC 0.8957 [0.8446, 0.9426], Brier 0.0048; nested recall 0.7826, F1 0.8229. Highest library-default classic PR-AUC was XGBoost 0.6322 [0.5331, 0.7247], then LightGBM 0.6271 [0.5313, 0.7202] (nested F1 0.5818 and 0.5902). Paired bootstrap Δ PR-AUC thinking v3.5 − LightGBM was 0.2941 (0.2071–0.3805), P(Δ ≤ 0) = 0/2000. The frozen Wang integer score on the same 5,185 rows had ROC-AUC 0.8013 and PR-AUC 0.1032 (weights not re-fit). Table 4b adjusted ORs (Wald 95% CI): `1.1:1Post dilation` 0.152 [0.081, 0.286]; Clopidogrel 0.480 [0.293, 0.787]; WBC 1.972 [1.667, 2.331] — associated with recorded VLST, not treatment effects. <!-- TRACE: nested_cv_v35_antileakage_on; nested_table2; wang_2020; table4b_adjusted_or; B3; W5 -->

### Conclusions

On this derivation cohort, thinking v3.5 had the highest nested-CV PR-AUC among nine fitted arms (0.9212); local TabPFN v3.5 (0.8957) also exceeded library-default XGBoost (0.6322) and LightGBM (0.6271). That ranking is not a hyperparameter-matched contest against tuned boosting. TSSI, WBC, raw lab decimals, a full-cohort stent codebook, and train SMOTE must not enter nested-CV binary classifiers as if they were baseline covariates. Association findings are not treatment effects. Nested-CV discrimination on the derivation file is not external validation; the ML models were not externally or temporally tested. <!-- TRACE: W3.1; B11; W5 reserved word validated -->

---

## Keywords

very late stent thrombosis; TabPFN; nested cross-validation; data leakage; class imbalance; acute coronary syndrome; percutaneous coronary intervention; SHAP

Keywords are topical, not freeze scalars.

<!--
TRACE INDEX
Title: YAML study.title_or_working_title
Abstract numbers: nested_cv_v35_antileakage_on; study.n_total/n_vlst/event_rate;
  wang_2020.reported_performance; table4b_adjusted_or; pins 9.0.0/0.6.0
Evidence map: W1–W5, B3, B11, §4.6 (leakage twins not in abstract Results),
  §5.8 (live numbers from YAML, not §7.1 historical 0.8553), §7.3 nested Table 2
-->
