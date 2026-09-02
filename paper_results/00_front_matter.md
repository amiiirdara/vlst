# Scope, motivation, terminology, and limitations

This note is the manuscript front matter that Parts 1–5 previously lacked. It is written so it does **not** clash with the three scope decisions: `code/failed_hypothesis/` is unused (D1); TabPFN numbers come only from `baseline_plus_tabpfn.ipynb` and `tabpfn_interpretability.ipynb` (D2); where a report and a notebook disagree, the notebook is authoritative (D4).

**W1 (leakage)** is already in Part 4 (Table S-TSSI / Figure S-TSSI and the follow-up-time methods paragraph). This file covers **W2–W5**.

---

## Terminology (W5)

| Term | Use for |
| --- | --- |
| **Association** | Part 1 univariate and multivariable results; Part 5 mutual information. Full-cohort, no held-out evaluation. |
| **Prediction** | Part 4 nested-CV out-of-fold results **only**. That is the only out-of-sample evaluation in this pack. |
| **Interpretation / attribution** | Part 2 selectors; Part 5 SHAP, k-SII, PDP, stability. Model-explanatory, not evidence about patients. |

Do **not** use: “risk factor”, “causal”, “protective”, “independent predictor”, “clinically useful”, or “validated” for any result in this pack. Wang’s 8-variable Cox score *was* externally tested on Shantou data; that word is reserved for **their** score. Nested-CV discrimination on the derivation cohort is not external validation, and it does not transfer to TabPFN by contagion.

An adjusted OR < 1 (`1.1:1Post dilation` 0.144; `Clopidogrel` 0.464) or a negative PDP shift is a lower modelled odds / probability of recorded VLST, not a treatment benefit (confounding by indication).

---

## Clinical motivation and what this analysis adds (W2)

**Outcome.** Very late stent thrombosis (VLST) is Academic Research Consortium 2007 *definite* stent thrombosis more than one year after implantation, angiographically confirmed (Wang et al., *Sci Rep* 2020;10:6378; hereafter **Wang 2020**). Probable and possible stent thrombosis are not counted. The file `data/raw/VLST.csv` **is** Wang 2020’s derivation cohort: consecutive ACS patients ≥ 18 years undergoing PCI at The First Hospital of Jilin University, 1 January 2014 – 1 June 2015; 6,038 eligible → 5,185 analysed (236 in-hospital deaths, 413 refused follow-up, 204 lost); **92** definite VLST events (**1.77%**). Median follow-up 1,502 days; median PCI → VLST 697 days. Ethics NO. 2013-256; written informed consent; registered NCT03491891.

**Why it matters (clinical, via Wang).** VLST is rare and late. Wang reviews that stent thrombosis accounts for a substantial share of new myocardial infarction after index PCI and carries several-fold higher adjusted mortality than infarction unrelated to a previously stented site. The intended decision is risk-stratification **more than one year** after PCI (monitoring and therapy after the mandated DAPT year).

**A score already exists.** This is not an empty clinical-prediction field. The Dangas late stent thrombosis score (also used for VLST) had c-statistic 0.66 in Wang’s comparison. Wang derived an **8-variable Cox** VLST score on these same 5,185 rows (diabetes, previous PCI, AMI as admitting diagnosis, eGFR < 90, 3-vessel disease, stents per lesion, SES, no post-dilation) with derivation c-statistic **0.80** and **Shantou** external c-statistic **0.82** (n = 2,058; that file is **not** in this repository). Any claim that “no VLST score exists” is false. This pack scores the published **integer points** as a frozen comparator on the same 5,185 rows (Part 4 Table S-Wang): ROC-AUC **0.8013**, PR-AUC **0.1032**. The Cox linear predictor, decision-curve analysis vs Dangas, and the Shantou file are still absent (B11).

**What “personalised” does *not* mean.** The repository README says “Personalized Risk prediction.” Nothing here is an individual-level model, a patient-specific fine-tune, or a decision-curve analysis. The artefact is a **single global classifier** (or a single global logit for association). Nested-CV probabilities are not portable personalised risks: prevalence, calibration, and PPV are properties of this derivation cohort. We do not use “personalised” as a result claim.

**Why TabPFN is in the comparison.** VLST here is a small-n, mixed-type tabular problem (92 events, 81 raw columns). TabPFN is a tabular foundation model that does in-context learning, handles categoricals natively, and does not run a per-dataset hyperparameter grid. Part 4 now reports **both** arms from the same nested CV (Kaggle Tesla T4, `de46f92`): **TabPFN (thinking-high)** (`tabpfn_client.TabPFNClassifier`, constructor unchanged) and **TabPFN (local)** (`from tabpfn import TabPFNClassifier`, `n_estimators="auto"`, `balance_probabilities=True`, no thinking; checkpoint `tabpfn-v3-classifier-v3_default.ckpt`). Classics in the same nested CV use **library defaults** plus class weighting. On this run thinking-high is first on PR-AUC (**0.8553**), ROC-AUC (**0.9905**), and Brier (**0.0064**). LightGBM is second on PR-AUC (**0.6926**). TabPFN (local) is fourth on PR-AUC (**0.6754**), second on ROC-AUC (**0.9845**), and worst on Brier (**0.0673**). Name the two TabPFN Briers separately.

**What this pack adds on the *same* derivation cohort, beyond Wang’s score:**

1. **Association catalogue** (Part 1) — FDR-controlled univariate tests and an exploratory 17-covariate logit (not a Cox model; not Wang’s eight variables).
2. **Interpretation catalogues** (Parts 2–3, 5) — classic-model LOCO / SHAP / FFS versus FDR names; TabPFN attributions. These do not feed the predictor.
3. **Prediction comparison** (Part 4) — nested 5×4 stratified CV of five classic classifiers and **both** TabPFN arms after dropping the leaky follow-up-time column (W1), plus the frozen Wang integer score on the same rows (Table S-Wang).
4. **Leakage control** (Part 4 supplement) — with-TSSI vs without-TSSI on a 70/30 split, showing why binary-ified survival time must not be a covariate.

It does **not** add: external or temporal testing of the ML models; a re-fit of Wang’s Cox linear predictor or a Dangas decision curve; a statement that TabPFN is ready for clinical use.

**Data, ethics, consent.** Cite Wang 2020 for NCT03491891, ethics 2013-256, written consent, and their data-availability statement. This repository’s analysis of the derivation file was not separately pre-registered.

---

## Events per variable (W4)

| Comparison | Value | Consequence |
| --- | --- | --- |
| Events / candidate features (81) | 92 / 81 ≈ **1.14** | Far below any conventional EPV rule |
| Events / multivariable logit covariates (17) | 92 / 17 ≈ **5.4** | Below the conventional EPV ≥ 10 rule |
| Events per Part 4 outer fold | 18, 18, 18, 19, 19 | Nested-CV scoreboard is thin |
| Events on the Part 2 val slice | 18 of 1,037 | Selector catalogues are not prediction |

Every **adjusted odds ratio** in Part 1 Table 4 (and the joint-domain supplement) is from the 17-covariate unweighted logit: **EPV ≈ 5.4**. Quote that number next to the OR. The model is for screening / confounding context, not prediction.

---

## Limitations (W3)

1. **No external or temporal test of the ML models.** Every Part 4 number is nested CV on the 5,185 derivation rows. Wang’s Cox score **was** tested on Shantou (n = 2,058, 1.70% VLST); those rows are not here. Nested CV is not a substitute.

2. **Binary classification vs published Cox analysis.** Wang used time-to-event on the follow-up axis. This pack uses a 0/1 label and drops `Time since stent implantation` because, as a covariate, it leaks (Part 4 S-TSSI). The frozen integer score on that binary label recovers Wang’s derivation c-statistic (ROC-AUC 0.8013 vs published 0.80; Part 4 S-Wang). That is not a re-fit of the Cox linear predictor, and it is not Shantou. Nested-CV TabPFN (thinking-high) PR-AUC **0.8553** (LightGBM **0.6926**; TabPFN local **0.6754**) vs the frozen score **0.1032** is a derivation-cohort ranking comparison only.

3. **EPV ≈ 5.4** on the 17-covariate logit (W4). Collinear blocks remain (`1.1:1Post dilation` beside `No postdilation`; `eGFR` beside `CKD5` / `CKD90`). `CKD90`’s Wald interval is extremely wide. Do not read Table 4 as an identified clinical model.

4. **Two TabPFN calibrations.** Nested-CV TabPFN (thinking-high) Brier is **0.0064**, the **best** of the seven. TabPFN (local) Brier is **0.0673**, the **worst**. Client thinking-high is non-deterministic across dumps (historical Brier 0.0060 / 0.0360 vs this dump 0.0064). Do not collapse the arms.

5. **Unequal TabPFN objects.** Thinking-high is the client API; local is `tabpfn` on Kaggle T4 (`tabpfn-v3-classifier-v3_default.ckpt`). Client and server-side versions remain unrecorded. Classics are untuned defaults.

6. **DAPT columns are post-baseline.** All patients had DAPT for ≥ 1 year; continuation after year 1 was at the treating physician’s discretion. Wang Table 1 “DAPT” is persistence during follow-up, not a discharge prescription. `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT` must not be described as index-PCI covariates without that caveat.

7. **WBC was excluded by the original investigators.** Wang dropped WBC from the Cox score because infection could not be ruled out. Our FDR screen and several selectors rank `WBC` at the top. That is a discrepancy to report, not a new “validated” inflammatory marker.

8. **Unequal tuning (Part 4).** A shared 9-level stent encoder is applied before the split. Classics then scale + one-hot that column inside each CV split (~89 columns). Both TabPFN arms see the same 9-level frame natively. Classics are untuned defaults; local TabPFN is not thinking-high; the client arm is thinking-high. Part 2/5 catalogues are discovery / attribution, not a mask for Part 4.

9. **No interval on PR-AUC, ROC-AUC, or Brier; no paired test** of thinking-high vs LightGBM (or local vs LightGBM). Thinking-high is first on PR-AUC as a point estimate (higher in **5 of 5** outer folds). Local is higher than LightGBM in **2 of 5**. OOF CSVs were written on Kaggle but **not committed** (B2).

10. **`LV` (and `CaI`) are not in Wang Table 1.** Until the column is named, timed, and unit-defined, do not treat `LV` as a novel echo marker.

11. **Part 5 is not the Part 4 predictor.** Ranking / SHAP / stability use `balance_probabilities=True` (stretched 1.8% prior so shapes are visible). PDP uses `False` (empirical prior, labeled not Part 4 risk; binary P(y=1) ≈ 0.017–0.023). Do not mix those scales on one axis. SHAP is **15 VLST=1 + 15 VLST=0** with client thinking; k-SII is one VLST=1 row (5099). Do not treat k-SII as cohort interactions.

---

## Sources for this note

Wang X, et al. A novel risk model for predicting very late stent thrombosis after percutaneous coronary intervention: a derivation and validation study. *Sci Rep*. 2020;10:6378. doi:10.1038/s41598-020-63455-0.

TabPFN configuration actually used: `code/modeling/rating/baseline_plus_tabpfn.ipynb` (performance) and `code/modeling/interpretability/tabpfn_interpretability.ipynb` (attribution). No other TabPFN notebook is in scope.
