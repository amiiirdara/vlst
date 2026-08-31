# VLST paper results

Self-contained bundle of the paper-style markdown reports (front matter, statistical EDA, classic-model feature selection, stats-vs-ML comparison, nested-CV TabPFN rating, and TabPFN interpretability). All figures and tables live next to these files. Zip this `paper_results/` folder to send it elsewhere; keep the folder layout unchanged so image links keep working.

**How to view.** Open this file in a Markdown previewer (VS Code / Cursor: Markdown Preview). Individual parts also open on their own:

0. [Part 0 — Scope, motivation, terminology, limitations](00_front_matter.md)
1. [Part 1 — Statistical EDA](01_eda/EDA_paper_figures_and_tables.md)
2. [Part 2 — Classic-model feature selection](02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md)
3. [Part 3 — Statistical vs ML feature extraction](03_stats_vs_ml/feature_extraction_comparison.md)
4. [Part 4 — Nested-CV baselines plus TabPFN](04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md)
5. [Part 5 — TabPFN interpretability](05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md)

Notebooks are not included in this pack.

---
# Part 0. Scope, motivation, terminology, and limitations

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

**Why TabPFN is in the comparison.** VLST here is a small-n, mixed-type tabular problem (92 events, 81 raw columns). TabPFN is a tabular foundation model that does in-context learning, handles categoricals natively, and does not run a per-dataset hyperparameter grid. The **stored Part 4 arm** is the Prior Labs **client** with `thinking_mode=True`, `thinking_effort="high"`, `thinking_metric="average_precision"` (`baseline_plus_tabpfn.ipynb` only). That is not an argument that TabPFN “needs no tuning therefore it wins”: the five classic models in the same nested CV also use **library defaults** plus class weighting. The comparison is unequal search budget, disclosed in Part 4. A local (no-thinking) TabPFN arm exists in that notebook for a later run; it is not in the stored six-model numbers.

**What this pack adds on the *same* derivation cohort, beyond Wang’s score:**

1. **Association catalogue** (Part 1) — FDR-controlled univariate tests and an exploratory 17-covariate logit (not a Cox model; not Wang’s eight variables).
2. **Interpretation catalogues** (Parts 2–3, 5) — classic-model LOCO / SHAP / FFS versus FDR names; TabPFN attributions. These do not feed the predictor.
3. **Prediction comparison** (Part 4) — nested 5×4 stratified CV of five classic classifiers and client TabPFN after dropping the leaky follow-up-time column (W1), plus the frozen Wang integer score on the same rows (Table S-Wang).
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

2. **Binary classification vs published Cox analysis.** Wang used time-to-event on the follow-up axis. This pack uses a 0/1 label and drops `Time since stent implantation` because, as a covariate, it leaks (Part 4 S-TSSI). The frozen integer score on that binary label recovers Wang’s derivation c-statistic (ROC-AUC 0.8013 vs published 0.80; Part 4 S-Wang). That is not a re-fit of the Cox linear predictor, and it is not Shantou. Nested-CV TabPFN PR-AUC **0.8534** vs the frozen score **0.1032** is a derivation-cohort ranking comparison only.

3. **EPV ≈ 5.4** on the 17-covariate logit (W4). Collinear blocks remain (`1.1:1Post dilation` beside `No postdilation`; `eGFR` beside `CKD5` / `CKD90`). `CKD90`’s Wald interval is extremely wide. Do not read Table 4 as an identified clinical model.

4. **TabPFN client non-determinism.** The stored nested-CV run used `random_state=42` and still produced two client probability sets (Brier **0.0060** in the executed notebook vs **0.0360** on some exported PNGs). D4: quote the notebook (0.0060, t_F1 0.173). A re-run can move the headline again.

5. **Remote service, version unrecorded.** Stored Part 4 TabPFN is `tabpfn_client`, not a pinned local checkpoint. Client and server-side model versions are not in the repo. The local no-thinking arm is a different object and is not in the stored six-model table.

6. **DAPT columns are post-baseline.** All patients had DAPT for ≥ 1 year; continuation after year 1 was at the treating physician’s discretion. Wang Table 1 “DAPT” is persistence during follow-up, not a discharge prescription. `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT` must not be described as index-PCI covariates without that caveat.

7. **WBC was excluded by the original investigators.** Wang dropped WBC from the Cox score because infection could not be ruled out. Our FDR screen and several selectors rank `WBC` at the top. That is a discrepancy to report, not a new “validated” inflammatory marker.

8. **Unequal feature views and unequal tuning (Part 4).** Classics see scaled one-hot input; TabPFN sees the raw 81-column frame. Classics are untuned defaults; TabPFN is thinking-high. Part 2/5 catalogues are discovery / attribution, not a mask for Part 4.

9. **No interval on PR-AUC, ROC-AUC, or Brier; no paired test** of TabPFN vs CatBoost. “TabPFN is first” is a point-estimate ordering (it wins PR-AUC in 5 of 5 outer folds). OOF CSVs were not committed.

10. **`LV` (and `CaI`) are not in Wang Table 1.** Until the column is named, timed, and unit-defined, do not treat `LV` as a novel echo marker.

---

## Sources for this note

Wang X, et al. A novel risk model for predicting very late stent thrombosis after percutaneous coronary intervention: a derivation and validation study. *Sci Rep*. 2020;10:6378. doi:10.1038/s41598-020-63455-0.

TabPFN configuration actually used: `code/modeling/rating/baseline_plus_tabpfn.ipynb` (performance) and `code/modeling/interpretability/tabpfn_interpretability.ipynb` (attribution). No other TabPFN notebook is in scope.

---
# Part 1. Statistical EDA

### VLST EDA — Paper Figures and Tables

This document gathers publication-oriented figures and tables from the exploratory data analysis of very late stent thrombosis (VLST) in `eda.ipynb`.

**Cohort context.** Analyses use the VLST dataset (n = 5,185; 92 VLST events; prevalence 0.0177). The notebook printed **no missing values** in any column — univariate screens do not impute. Univariate continuous tests use Welch t-test when abs(skew) ≤ 1 and excess kurtosis ≤ 3, otherwise Mann–Whitney U. Binary associations use recommended 2×2 tests (chi-square / Fisher / related). Multiplicity is controlled with Benjamini–Hochberg FDR unless noted. Multivariable models are exploratory and sparse given the limited number of events. `Stent type-SES` is collapsed to levels with n ≥ 30 plus `other` (**9 levels**) for the χ² screen via the shared encoder (`code/modeling/tools/stent_encoding.py`). Part 2 now uses that same 9-level column and one-hots it (drop-first → **88** scaled columns). Part 4 still needs a re-run before its stored one-hot width matches. `Time since stent implantation` is treated as a **time-at-risk / follow-up** variable and is **not** interpreted as a baseline clinical association.

**Asset root:** [01_eda/paper_figures/](01_eda/paper_figures/)

> Preview note: large table images are linked (not inlined) so the Markdown preview stays responsive. Figure PNGs are inlined below.

---

## Contents

1. [Test selection](#1-test-selection)
2. [Univariate continuous associations](#2-univariate-continuous-associations)
3. [Univariate binary associations](#3-univariate-binary-associations)
4. [Categorical associations](#4-categorical-associations)
5. [Multivariable adjustment](#5-multivariable-adjustment)
6. [Pairwise / bivariate structure](#6-pairwise--bivariate-structure)
7. [Medical-domain analysis (supplementary)](#7-medical-domain-analysis-supplementary)
8. [File index](#8-file-index)

---

## 1. Test selection

### Figure 1. Continuous test-selection map

![Figure 1](01_eda/paper_figures/paper_fig1_test_selection_map.png)

**Figure 1.** Map of continuous predictors in the abs(skewness)–excess-kurtosis plane used to choose the primary univariate test for association with VLST. Green markers indicate Welch t-test (abs(skew) ≤ 1 and excess kurtosis ≤ 3); orange markers indicate Mann–Whitney U. Panel A uses a linear kurtosis axis (high-kurtosis features annotated if clipped); panel B uses a log kurtosis scale to show the full range; panel C lists feature IDs, chosen test, and shape statistics. Dashed lines mark the selection thresholds.

**Source file:** [paper_figures/paper_fig1_test_selection_map.png](01_eda/paper_figures/paper_fig1_test_selection_map.png)

### Table R. Continuous variables: chosen univariate test and rationale

Rendered table image (open separately if needed): [paper_figures/paper_table_test_rationale.png](01_eda/paper_figures/paper_table_test_rationale.png)

**Table R.** For each continuous predictor, the selected univariate test, distributional rationale (skewness, excess kurtosis, variance ratio), raw p, FDR q, and significance code (`p<0.05` / `p<0.01` / `p<0.001`).

| ID | Feature | Chosen test | Skewness | Kurtosis | Var. ratio | Why this test | p (raw) | q (FDR) | Sig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | CKD5 | Mann-Whitney U | 2.62 | 7.51 | 1.32 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 1.19e-05 | 5.69e-05 | p<0.001 |
| 2 | CaI | Mann-Whitney U | 2.83 | 9.43 | 1.37 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.0507 | 0.0869 | ns |
| 3 | Cre | Mann-Whitney U | 7.48 | 161.34 | 1.7 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.878 | 0.938 | ns |
| 4 | Fast-Glu | Mann-Whitney U | 2.42 | 7.84 | 2.13 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.0246 | 0.0536 | ns |
| 5 | Fiberinogen | Mann-Whitney U | 1.62 | 5.65 | 1.31 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.0119 | 0.0286 | p<0.05 |
| 6 | HbA1c | Mann-Whitney U | 1.4 | 1.92 | 1.11 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 2.00e-04 | 7.00e-04 | p<0.001 |
| 7 | LDL | Mann-Whitney U | 1.06 | 4.67 | 1.78 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.327 | 0.491 | ns |
| 8 | LVEF | Mann-Whitney U | -1.48 | 7.42 | 1.51 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.0333 | 0.0666 | ns |
| 9 | No.of stents per lesion | Mann-Whitney U | 2.13 | 4.7 | 2.03 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 1.00e-04 | 6.00e-04 | p<0.001 |
| 10 | Platelet | Mann-Whitney U | 1.32 | 6.92 | 1.24 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.816 | 0.938 | ns |
| 11 | TG | Mann-Whitney U | 2.13 | 8.84 | 1.52 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.558 | 0.705 | ns |
| 12 | Time since stent implantation | Mann-Whitney U | -2.36 | 13.2 | 12.7 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 1.70e-34 | 4.07e-33 | p<0.001 |
| 13 | Total stent length | Mann-Whitney U | 1.74 | 4.15 | 1.76 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 0.0011 | 0.003 | p<0.01 |
| 14 | WBC | Mann-Whitney U | 1.21 | 2.12 | 1.47 | Heavy skew/tails (/skew/>1 or kurt>3) -> MW | 7.90e-21 | 9.48e-20 | p<0.001 |
| 15 | Age | Welch t-test | -0.07 | -0.28 | 1.3 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 0.463 | 0.618 | ns |
| 16 | HDL | Welch t-test | 0.91 | 1.73 | 1.58 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 0.899 | 0.938 | ns |
| 17 | HGB | Welch t-test | -0.41 | 0.52 | 1.54 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 0.0388 | 0.0716 | ns |
| 18 | LV | Welch t-test | 0.97 | 2.47 | 1.1 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 5.44e-17 | 3.26e-16 | p<0.001 |
| 19 | Max-stent diameter | Welch t-test | 0.59 | 0.3 | 1.17 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 0.885 | 0.938 | ns |
| 20 | Min-stent diameter | Welch t-test | 0.76 | 0.72 | 1.09 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 0.106 | 0.169 | ns |
| 21 | NO.of vessels | Welch t-test | 0.24 | -1.48 | 1.04 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 5.00e-04 | 0.0014 | p<0.01 |
| 22 | Stent release pressure | Welch t-test | 0.07 | -0.56 | 1.13 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 0.949 | 0.949 | ns |
| 23 | TCL | Welch t-test | 0.87 | 2.75 | 1.57 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 0.381 | 0.537 | ns |
| 24 | eGFR | Welch t-test | 0.06 | -0.58 | 3.02 | Approx. symmetric (/skew/≤1 & kurt≤3) -> Welch | 4.64e-20 | 3.71e-19 | p<0.001 |

**Source files:** [paper_figures/paper_table_test_rationale.png](01_eda/paper_figures/paper_table_test_rationale.png), [paper_figures/paper_table_test_rationale.csv](01_eda/paper_figures/paper_table_test_rationale.csv)

---

## 2. Univariate continuous associations

### Figure 2. Univariate continuous significance overview

![Figure 2](01_eda/paper_figures/paper_fig2_univariate_significance.png)

**Figure 2.** Horizontal ranking of continuous features by -log10(p) for association with VLST. Bar color denotes the chosen test (Welch vs Mann–Whitney). The dotted line marks nominal p = 0.05; the dashed line approximates the FDR discovery boundary among continuous tests. FDR marks label features with FDR q < 0.05.

**Source file:** [paper_figures/paper_fig2_univariate_significance.png](01_eda/paper_figures/paper_fig2_univariate_significance.png)

### Table 1. Continuous features with FDR q < 0.05 (univariate)

Rendered table image: [paper_figures/paper_table1_continuous_fdr.png](01_eda/paper_figures/paper_table1_continuous_fdr.png)

**Table 1.** FDR-significant continuous associations with VLST, including test type, effect-size metric (Cohen d or Mann–Whitney r), mean/median differences (VLST − no VLST), raw p, and FDR q. Features annotated as time-at-risk should not be interpreted as baseline clinical associations.

| Feature | Test | Direction | Effect size | ES | Δ mean | Δ median | p | q (FDR) | Sig | Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time since stent implantation | Mann-Whitney U | Negative | Mann-Whitney r | -0.17 | -572.05 | -683.00 | 1.70e-34 | 4.07e-33 | p<0.001 | Time-at-risk / structural |
| WBC | Mann-Whitney U | Positive | Mann-Whitney r | 0.13 | 3.75 | 3.82 | 7.90e-21 | 9.48e-20 | p<0.001 | — |
| eGFR | Welch t-test | Negative | Cohen's d | -0.712 | -24.1 | -17.6 | 4.64e-20 | 3.71e-19 | p<0.001 | — |
| LV | Welch t-test | Positive | Cohen's d | 1.13 | 4.56 | 4 | 5.44e-17 | 3.26e-16 | p<0.001 | — |
| CKD5 | Mann-Whitney U | Positive | Mann-Whitney r | 0.04 | 0.18 | 0 | 1.19e-05 | 5.69e-05 | p<0.001 | — |
| No.of stents per lesion | Mann-Whitney U | Positive | Mann-Whitney r | 0.037 | 0.21 | 0 | 1.00e-04 | 6.00e-04 | p<0.001 | — |
| HbA1c | Mann-Whitney U | Positive | Mann-Whitney r | 0.052 | 0.47 | 1.05 | 2.00e-04 | 7.00e-04 | p<0.001 | — |
| NO.of vessels | Welch t-test | Positive | Cohen's d | 0.388 | 0.32 | 0 | 5.00e-04 | 0.0014 | p<0.01 | — |
| Total stent length | Mann-Whitney U | Positive | Mann-Whitney r | 0.045 | 6.76 | 2 | 0.0011 | 0.003 | p<0.01 | — |
| Fiberinogen | Mann-Whitney U | Positive | Mann-Whitney r | 0.035 | 0.2 | 0.17 | 0.0119 | 0.0286 | p<0.05 | — |

**Source files:** [paper_figures/paper_table1_continuous_fdr.png](01_eda/paper_figures/paper_table1_continuous_fdr.png), [paper_figures/paper_table1_continuous_univariate.csv](01_eda/paper_figures/paper_table1_continuous_univariate.csv)

### Figure 3. Effect sizes for FDR-significant continuous features

![Figure 3](01_eda/paper_figures/paper_fig3_continuous_effect_sizes.png)

**Figure 3.** Primary effect sizes for continuous features with FDR q < 0.05. **Cohen's d (Welch) and Mann–Whitney r are different metrics and are not on one scale** — `WBC` r = 0.13 is not “smaller than” `LV` d = 1.13. The stored PNG plots them on one axis; read the `Effect size` column in Table 1, not bar length across tests. Grey highlighting (when present) marks the structural time-at-risk variable (`Time since stent implantation`).

**Source file:** [paper_figures/paper_fig3_continuous_effect_sizes.png](01_eda/paper_figures/paper_fig3_continuous_effect_sizes.png)

---

## 3. Univariate binary associations

### Figure 4. Odds ratios for binary features (FDR q < 0.05)

![Figure 4](01_eda/paper_figures/paper_fig4_binary_odds_ratios.png)

**Figure 4.** Univariate 2×2 odds ratios for binary indicators with FDR q < 0.05. OR > 1: higher odds of VLST when the flag is present. OR < 1: lower odds in this table, **not** a treatment benefit (confounding by indication).

**Source file:** [paper_figures/paper_fig4_binary_odds_ratios.png](01_eda/paper_figures/paper_fig4_binary_odds_ratios.png)

### Table 2. Binary features associated with VLST

Rendered table image: [paper_figures/paper_table2_binary_fdr.png](01_eda/paper_figures/paper_table2_binary_fdr.png)

**Table 2.** Binary associations with FDR q < 0.05. **OR is the 2×2 cross-product** (Fisher exact when the notebook chose Fisher; chi-square otherwise) — for `Previous PCI` this is **6.49**. This is not the univariate logistic OR in Table 4 (6.46) or the joint-domain univariate OR in Table S4 (6.73). Rates, RR, and phi use the same 2×2. OR < 1 is a **lower odds of recorded VLST** when the flag is 1, not a treatment benefit (confounding by indication).

| Feature | Test | OR | RR | Phi | VLST% (1) | VLST% (0) | p | q (FDR) | Sig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.1:1Post dilation | Chi-square | 0.187 | 0.191 | -0.089 | 0.56 | 2.92 | 1.30e-10 | 3.69e-09 | p<0.001 |
| No postdilation | Chi-square | 5.36 | 5.23 | 0.089 | 2.92 | 0.56 | 1.30e-10 | 3.69e-09 | p<0.001 |
| CKD90 | Chi-square | 2.62 | 2.57 | 0.063 | 3.59 | 1.4 | 6.55e-06 | 1.00e-04 | p<0.001 |
| Previous PCI | Fisher exact | 6.49 | 5.96 | 0.085 | 9.62 | 1.61 | 1.25e-05 | 2.00e-04 | p<0.001 |
| 3-vessel disease | Chi-square | 2.17 | 2.13 | 0.052 | 2.87 | 1.34 | 2.00e-04 | 0.0021 | p<0.01 |
| Clopidogrel | Chi-square | 0.503 | 0.509 | -0.043 | 1.17 | 2.29 | 0.0022 | 0.0209 | p<0.05 |
| Diabetes | Chi-square | 1.89 | 1.86 | 0.042 | 2.71 | 1.45 | 0.0028 | 0.0226 | p<0.05 |
| PES | Chi-square | 2.16 | 2.13 | 0.04 | 2.12 | 1 | 0.0044 | 0.0315 | p<0.05 |
| Multi-vessel CAD | Chi-square | 1.89 | 1.87 | 0.038 | 2.19 | 1.17 | 0.0062 | 0.0351 | p<0.05 |
| Single-vessel disease | Chi-square | 0.529 | 0.535 | -0.038 | 1.17 | 2.19 | 0.0062 | 0.0351 | p<0.05 |

**Source files:** [paper_figures/paper_table2_binary_fdr.png](01_eda/paper_figures/paper_table2_binary_fdr.png), [paper_figures/paper_table2_binary_univariate.csv](01_eda/paper_figures/paper_table2_binary_univariate.csv)

---

## 4. Categorical associations

### Figure 5. VLST rate by stent type

![Figure 5](01_eda/paper_figures/paper_fig5_categorical_rates_Stent_type-SES.png)

**Figure 5.** Observed VLST rate (%) across categories of `Stent type-SES` after the shared encoder (canonicalize aliases, collapse n < 30 → `other`, **9 levels**). Raw distinct strings in this run: **99**. Rates: `other` 5.5%, `xiencev` 3.3%, `partner` 2.3%, `excel` 2.1%, `firebird` 1.6%, `tivoli` 0.83%, `resolute` 0.82%, `xv` 0%, `xx` 0%. Part 2 one-hots these 9 levels. Wang 2020 used a **binary SES class flag**. Rates are descriptive; formal association testing is summarized in Table 3.

**Source file:** [paper_figures/paper_fig5_categorical_rates_Stent_type-SES.png](01_eda/paper_figures/paper_fig5_categorical_rates_Stent_type-SES.png)

### Table 3. Categorical feature association with VLST

Rendered table image: [paper_figures/paper_table3_categorical.png](01_eda/paper_figures/paper_table3_categorical.png)

**Table 3.** Chi-square association between stent-type category and VLST, with degrees of freedom, Cramer V, raw p, and FDR q.

| Feature | Test | Levels used | Chi-square | df | Cramér's V | p | q (FDR) | Sig |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stent type-SES | Chi-square | 9 | 44.9 | 8 | 0.093 | 3.85e-07 | 3.85e-07 | p<0.001 |

**Source files:** [paper_figures/paper_table3_categorical.png](01_eda/paper_figures/paper_table3_categorical.png), [paper_figures/paper_table3_categorical.csv](01_eda/paper_figures/paper_table3_categorical.csv)

**Methods note — 20 FDR names ≈ 12 constructs.** Pooling Tables 1–3 (and excluding `Time since stent implantation`) gives **20** FDR q < 0.05 names. At least 8 are redundant re-encodings: post-dilation complements (2 slots for 1 bit), the vessel-disease family (`3-vessel`, `Multi-vessel`, `Single-vessel`, `NO.of vessels` — 4 slots for 1 construct), and the renal family (`CKD5`, `CKD90` with continuous `eGFR` — 3 slots for 1 construct). Report the headline as roughly **12 distinct clinical constructs**, not 20 independent discoveries.

---

## 5. Multivariable adjustment

### Table 4. Exploratory multivariable logistic model (adjusted ORs)

Rendered table image: [paper_figures/paper_table4_multivariable_or.png](01_eda/paper_figures/paper_table4_multivariable_or.png)

**Table 4.** Exploratory multivariable logistic regression for VLST (**unweighted MLE**, `statsmodels.Logit`). Continuous predictors are scaled per 1 SD. `Time since stent implantation` is excluded. Primary interval is the **Wald 95% CI**; SE (log-OR) and Wald p are reported. A 2,000-replicate percentile bootstrap of the same unweighted fit is stored in the numeric CSV as a robustness check. `class_weight="balanced"` is **not** used here (it is a predictive device; it distorts the likelihood used for Wald/LR tests). **EPV = 92 / 17 ≈ 5.4** (92 events, 17 covariates), below the conventional EPV ≥ 10 rule. Adjusted ORs are for screening/confounding context, **not prediction**. Re-run `eda.ipynb` cell 10f to refresh the numbers below.

**OR estimators (do not mix).** Table 4 “Univariate OR” is from this unweighted logit (one covariate at a time, same scaling). Table 2 OR is the **2×2 / Fisher** estimator. Supplementary Figure S4 “Univariate OR” is from the joint-domain specification. For `Previous PCI` those three numbers are **6.46 / 6.49 / 6.73**. OR < 1 for `1.1:1Post dilation` or `Clopidogrel` is not a treatment benefit.

| Feature | Type | Univariate OR | Adjusted OR | 95% CI |
| --- | --- | --- | --- | --- |
| WBC | continuous (per 1 SD) | 2.69 | 3 | [2.424, 4.225] |
| eGFR | continuous (per 1 SD) | 0.334 | 0.113 | [0.060, 0.158] |
| LV | continuous (per 1 SD) | 3.12 | 3.28 | [2.335, 5.254] |
| CKD5 | continuous (per 1 SD) | 1.39 | 0.156 | [0.023, 0.285] |
| No.of stents per lesion | continuous (per 1 SD) | 1.38 | 1.3 | [0.693, 2.177] |
| HbA1c | continuous (per 1 SD) | 1.37 | 0.87 | [0.569, 1.280] |
| NO.of vessels | continuous (per 1 SD) | 1.46 | 1.5 | [0.774, 3.392] |
| Total stent length | continuous (per 1 SD) | 1.39 | 1.13 | [0.658, 2.269] |
| Fiberinogen | continuous (per 1 SD) | 1.22 | 1.04 | [0.804, 1.348] |
| 1.1:1Post dilation | binary | 0.187 | 0.144 | [0.040, 0.245] |
| No postdilation | binary | 5.35 | 0.895 | [0.429, 1.358] |
| CKD90 | binary | 2.62 | 12.5 | [2.708, 639.506] |
| Previous PCI | binary | 6.46 | 8.98 | [3.226, 28.618] |
| 3-vessel disease | binary | 2.17 | 0.605 | [0.135, 2.936] |
| Clopidogrel | binary | 0.504 | 0.464 | [0.195, 0.817] |
| Diabetes | binary | 1.89 | 1.88 | [0.678, 4.331] |
| PES | binary | 2.16 | 1.24 | [0.585, 4.056] |

**Source files:** [paper_figures/paper_table4_multivariable_or.png](01_eda/paper_figures/paper_table4_multivariable_or.png), [paper_figures/paper_table4_multivariable_or.csv](01_eda/paper_figures/paper_table4_multivariable_or.csv)

### Figure 6. Univariate versus multivariable associations

![Figure 6](01_eda/paper_figures/paper_fig6_uni_vs_multivariable_or.png)

**Figure 6.** Comparison of univariate ORs (diamonds) with multivariable adjusted ORs and 95% CIs (circles/whiskers) for features entering the exploratory joint model. Attenuation toward the null suggests confounding or shared information; persistence of association after adjustment means the name still moves log-odds in this sparse specification, not a causal or standalone clinical claim.

**Source file:** [paper_figures/paper_fig6_uni_vs_multivariable_or.png](01_eda/paper_figures/paper_fig6_uni_vs_multivariable_or.png)

---

## 6. Pairwise / bivariate structure

A separate “bivariate feature extraction” step is **not** required. The correlation heatmap *is* the feature–feature bivariate analysis; predictor-versus-outcome bivariate work is already the FDR screens in sections 2–4 (the notebook calls those tests “univariate” because each model has one predictor). The three layers already in `eda.ipynb` answer different questions:

| Layer | What is paired | Role in the paper | Where |
| --- | --- | --- | --- |
| Predictor × VLST | Each column vs the outcome | Discovery catalogue (FDR q < 0.05) | Sections 2–4 |
| Feature × feature | Numeric columns vs each other | Multicollinearity / clustering before sparse models | Heatmaps below; Supplementary Figure S2 |
| Pair × VLST | Chosen interactions vs the outcome | Hypothesis-generating likelihood-ratio tests | Supplementary Table S2 |

Heatmaps that include `Stent thrombosis` also show predictor–outcome Pearson/Spearman *r*. That is a linear association measure and is **not** a substitute for the Welch / Mann–Whitney / Fisher screens, which allow non-Gaussian, rank, and 2×2 associations and apply FDR. The heatmap does not produce a second “bivariate FDR set.”

### Supplementary Figure S5. Pearson and Spearman heatmaps (top-42 vs next-41, with target)

![Figure S5a](01_eda/paper_figures/03_correlation_heatmap_top42_vs_next41_with_target.png)

**Figure S5a.** Pearson correlation among the 42 numeric columns with strongest |r| versus the outcome, against the next 41, including `Stent thrombosis`. Produced by the 2026-08-31 `eda.ipynb` re-run. Off-diagonal blocks are feature–feature structure; the target row/column is the linear predictor–outcome slice.

![Figure S5b](01_eda/paper_figures/03b_spearman_correlation_heatmap_top42_vs_next41_with_target.png)

**Figure S5b.** Spearman rank correlation for the same layout. Prefer this panel when tails or monotonic-but-nonlinear associations matter.

Publication clustering of the same numeric block is Supplementary Figure S2 (global and per-domain Spearman clustermaps). Pairwise *with outcome* beyond linear *r* is Table S2.

**Source files:** [paper_figures/03_correlation_heatmap_top42_vs_next41_with_target.png](01_eda/paper_figures/03_correlation_heatmap_top42_vs_next41_with_target.png), [paper_figures/03b_spearman_correlation_heatmap_top42_vs_next41_with_target.png](01_eda/paper_figures/03b_spearman_correlation_heatmap_top42_vs_next41_with_target.png)

---

## 7. Medical-domain analysis (supplementary)

Clinical-block analysis (section 10g): predictors grouped by medical domain; correlation clustering used to drop redundant mates before sparse domain-wise and joint models.

### Supplementary Figure S1. Domain univariate top associations

![Figure S1](01_eda/paper_figures/domain_univariate_top_hits.png)

**Figure S1.** Within each clinical domain (demographics/lifestyle, comorbidities, presentation, anatomy/lesion, procedural/stent, cardiac function, laboratory, medications), the strongest univariate associations with VLST ranked by -log10(p). Time-at-risk is excluded from domain ranking. Color encodes test family (Welch / Mann–Whitney / binary).

### Supplementary Table S1. Domain strength summary

**Table S1.** Domain-level summary of univariate screening: number of features, count of raw p < 0.05, global FDR hits, within-domain FDR hits, minimum p, peak -log10(p), and top feature.

| Domain | n_features | n_raw_p05 | n_fdr_global | n_fdr_domain | min_p | max_neglog10p | top_feature |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Laboratory | 16 | 8 | 6 | 6 | 7.90e-21 | 20.1 | WBC |
| Cardiac function | 2 | 2 | 1 | 2 | 5.44e-17 | 16.3 | LV |
| Procedural / stent | 16 | 6 | 5 | 6 | 1.30e-10 | 9.89 | 1.1:1Post dilation |
| Comorbidities / history | 10 | 3 | 2 | 2 | 1.25e-05 | 4.9 | Previous PCI |
| Anatomy / lesion | 23 | 6 | 4 | 4 | 1.81e-04 | 3.74 | 3-vessel disease |
| Medications (1y) | 4 | 2 | 1 | 2 | 0.0022 | 2.66 | Clopidogrel |
| Presentation / ACS | 5 | 0 | 0 | 0 | 0.0542 | 1.27 | Initial diagnosis-AMI |
| Demographics / lifestyle | 4 | 0 | 0 | 0 | 0.268 | 0.572 | Men |

**Source files:** [paper_figures/domain_strength_summary.csv](01_eda/paper_figures/domain_strength_summary.csv), [paper_figures/domain_univariate_summary.csv](01_eda/paper_figures/domain_univariate_summary.csv)

### Supplementary Figure S2. Feature correlation clustermaps

![Figure S2a](01_eda/paper_figures/domain_clustermap_global.png)

**Figure S2a.** Global Spearman correlation clustermap of numeric predictors (near-constant columns removed). Hierarchical clustering uses average linkage on distance 1 − abs(r).

![Figure S2b](01_eda/paper_figures/domain_clustermap_lab.png)

**Figure S2b.** Laboratory-domain Spearman clustermap.

![Figure S2c](01_eda/paper_figures/domain_clustermap_anatomy.png)

**Figure S2c.** Anatomy/lesion-domain Spearman clustermap.

![Figure S2d](01_eda/paper_figures/domain_clustermap_procedural.png)

**Figure S2d.** Procedural/stent-domain Spearman clustermap.

**Source files:** `domain_clustermap_*.png`, `feature_correlation_clusters.csv`, `feature_correlation_cluster_reps.csv`

### Supplementary Figure S3. Per-domain multivariable odds ratios

![Figure S3](01_eda/paper_figures/domain_multivariable_or_panels.png)

**Figure S3.** Domain-specific sparse logistic models (core demographics plus up to five non-redundant domain representatives). Primary interval is the unweighted Wald 95% CI from `statsmodels.Logit` (2,000-replicate percentile bootstrap stored as robustness). The dashed line marks OR = 1.

### Supplementary Figure S4. Joint cross-domain model (uni vs adjusted OR)

![Figure S4](01_eda/paper_figures/domain_joint_uni_vs_multi_or.png)

**Figure S4.** Joint sparse cross-domain logistic model comparing univariate ORs with adjusted ORs (**EPV = 92 / 17 ≈ 5.4** on the parent 17-covariate screen; this joint-domain fit is a separate specification). Continuous covariates are per 1 SD; time-since-stent is excluded. Unweighted Wald 95% CIs are primary. `LVEF`’s adjusted OR **reverses sign** versus its univariate OR when `LV` is in the model. The “Univariate OR” column below is from **this joint-domain specification** (`Previous PCI` 6.73), not Table 2’s 2×2 OR (6.49) or Table 4’s univariate logit (6.46).

| Feature | Domain | Univariate OR | Adjusted OR | OR lower | OR upper |
| --- | --- | --- | --- | --- | --- |
| Age | Demographics / lifestyle | 1.08 | 1.1 | 0.836 | 1.34 |
| WBC | Laboratory | 2.69 | 2.94 | 2.34 | 4.4 |
| eGFR | Laboratory | 0.334 | 0.203 | 0.127 | 0.299 |
| LV | Cardiac function | 3.12 | 2.95 | 2.2 | 5.04 |
| LVEF | Cardiac function | 0.851 | 1.65 | 1.3 | 2.26 |
| No.of stents per lesion | Procedural / stent | 1.38 | 1.55 | 1.17 | 2.27 |
| Men | Demographics / lifestyle | 1.29 | 3.28 | 1.58 | 7.9 |
| Current smoker | Demographics / lifestyle | 1.11 | 0.976 | 0.541 | 2.07 |
| Current drinking | Demographics / lifestyle | 0.981 | 1.07 | 0.423 | 2.11 |
| 1.1:1Post dilation | Procedural / stent | 0.192 | 0.191 | 0.0444 | 0.382 |
| Previous PCI | Comorbidities / history | 6.73 | 9.58 | 3.31 | 23.6 |
| Diabetes | Comorbidities / history | 1.9 | 1.46 | 0.789 | 2.6 |

### Supplementary Table S2. Exploratory interaction screen

**Table S2.** All **16** pairs from `domain_interaction_screen.csv` (likelihood-ratio test vs main-effects-only). FDR q is among these 16 tests only. With ~92 events, interactions are hypothesis-generating. Two pairs (LV × eGFR, Men × eGFR) have q < 0.05.

| Pair | LR statistic | Interaction p | Interaction OR | q (FDR) |
| --- | --- | --- | --- | --- |
| LV x eGFR | 9.81 | 0.00173 | 1.24 | 0.0277 |
| Men x eGFR | 8.53 | 0.0035 | 0.342 | 0.028 |
| WBC x eGFR | 2.65 | 0.104 | 1.13 | 0.553 |
| Current smoker x DAPT | 2.07 | 0.15 | 1.86 | 0.599 |
| Aspirin x Clopidogrel | 1.49 | 0.222 | 2.97 | 0.623 |
| Diabetes x HbA1c | 1.25 | 0.264 | 0.787 | 0.623 |
| Men x Previous PCI | 1.19 | 0.274 | 0.442 | 0.623 |
| LV x Previous PCI | 0.964 | 0.326 | 0.786 | 0.623 |
| LV x Men | 0.873 | 0.350 | 1.17 | 0.623 |
| DAPT x Diabetes | 0.428 | 0.513 | 1.49 | 0.821 |
| No.of stents per lesion x Total stent length | 0.297 | 0.586 | 0.977 | 0.823 |
| Previous PCI x eGFR | 0.186 | 0.667 | 1.19 | 0.823 |
| 1.1:1Post dilation x Men | 0.131 | 0.718 | 0.786 | 0.823 |
| 1.1:1Post dilation x Previous PCI | 0.129 | 0.720 | 1.46 | 0.823 |
| 1.1:1Post dilation x LV | 0.020 | 0.888 | 0.974 | 0.912 |
| 1.1:1Post dilation x eGFR | 0.012 | 0.912 | 0.979 | 0.912 |

**Source file:** [paper_figures/domain_interaction_screen.csv](01_eda/paper_figures/domain_interaction_screen.csv)

---

## 8. File index

| ID | Type | File |
| --- | --- | --- |
| Fig 1 | Figure | [paper_fig1_test_selection_map.png](01_eda/paper_figures/paper_fig1_test_selection_map.png) |
| Table R | Table | [paper_table_test_rationale.png](01_eda/paper_figures/paper_table_test_rationale.png) |
| Fig 2 | Figure | [paper_fig2_univariate_significance.png](01_eda/paper_figures/paper_fig2_univariate_significance.png) |
| Table 1 | Table | [paper_table1_continuous_fdr.png](01_eda/paper_figures/paper_table1_continuous_fdr.png) |
| Fig 3 | Figure | [paper_fig3_continuous_effect_sizes.png](01_eda/paper_figures/paper_fig3_continuous_effect_sizes.png) |
| Fig 4 | Figure | [paper_fig4_binary_odds_ratios.png](01_eda/paper_figures/paper_fig4_binary_odds_ratios.png) |
| Table 2 | Table | [paper_table2_binary_fdr.png](01_eda/paper_figures/paper_table2_binary_fdr.png) |
| Fig 5 | Figure | [paper_fig5_categorical_rates_Stent_type-SES.png](01_eda/paper_figures/paper_fig5_categorical_rates_Stent_type-SES.png) |
| Table 3 | Table | [paper_table3_categorical.png](01_eda/paper_figures/paper_table3_categorical.png) |
| Table 4 | Table | [paper_table4_multivariable_or.png](01_eda/paper_figures/paper_table4_multivariable_or.png) |
| Fig 6 | Figure | [paper_fig6_uni_vs_multivariable_or.png](01_eda/paper_figures/paper_fig6_uni_vs_multivariable_or.png) |
| Fig S5a | Supp. figure | [03_correlation_heatmap_top42_vs_next41_with_target.png](01_eda/paper_figures/03_correlation_heatmap_top42_vs_next41_with_target.png) |
| Fig S5b | Supp. figure | [03b_spearman_correlation_heatmap_top42_vs_next41_with_target.png](01_eda/paper_figures/03b_spearman_correlation_heatmap_top42_vs_next41_with_target.png) |
| Fig S1 | Supp. figure | [domain_univariate_top_hits.png](01_eda/paper_figures/domain_univariate_top_hits.png)
| Table S1 | Supp. table | [domain_strength_summary.csv](01_eda/paper_figures/domain_strength_summary.csv) |
| Fig S2a | Supp. figure | [domain_clustermap_global.png](01_eda/paper_figures/domain_clustermap_global.png) |
| Fig S2b | Supp. figure | [domain_clustermap_lab.png](01_eda/paper_figures/domain_clustermap_lab.png) |
| Fig S2c | Supp. figure | [domain_clustermap_anatomy.png](01_eda/paper_figures/domain_clustermap_anatomy.png) |
| Fig S2d | Supp. figure | [domain_clustermap_procedural.png](01_eda/paper_figures/domain_clustermap_procedural.png) |
| Fig S3 | Supp. figure | [domain_multivariable_or_panels.png](01_eda/paper_figures/domain_multivariable_or_panels.png) |
| Fig S4 | Supp. figure | [domain_joint_uni_vs_multi_or.png](01_eda/paper_figures/domain_joint_uni_vs_multi_or.png) |
| Table S2 | Supp. table | [domain_interaction_screen.csv](01_eda/paper_figures/domain_interaction_screen.csv) |

---

*Generated from EDA notebook outputs. Close and reopen this file (or refresh Markdown preview) after updates.*

---

# Part 2. Classic-model feature selection

### Classic-model feature selection — paper figures and tables

This document gathers publication-oriented figures and tables from the multi-model feature selectors in `baseline_feature_selections.ipynb`.

**Cohort / protocol (2026-08-31 Kaggle run).** Full VLST cohort, n = 5,185. Target = `Stent thrombosis`. `Time since stent implantation` is dropped. This is **not** the TabPFN playground notebook (out of scope). **Paper protocol:** no parked 70/30 test — every row is split once into fit / val (`INNER_VAL_SIZE=0.2`, `random_state=42`): **fit = 4,148 rows (74 events)** / **val = 1,037 rows (18 events)**. **PR-AUC only.** LOCO / SHAP / FFS are **independent** (each takes its own cheap fit-slice importance pool). Budget: top-20; SHAP universe 40; LOCO cap 60; FFS pool 24 × 12 steps with early stop (`FFS_MIN_GAIN=0`); boosting 400 rounds. `USE_CACHE=False`. GPU: Tesla T4. Models use the **scaled** view: shared 9-level stent-brand encoder, then `ColumnTransformer` one-hot (drop-first) + `StandardScaler` → **88 columns** (81 raw − 1 brand + 8 dummies). Median / most-frequent imputers sit in that transformer; the CSV has **no missing values**, so they are inert. Selector hyperparameters are the notebook’s own factories (`C=2`, RF 500 trees, `lr=0.05`) — **not** `GridSearchCV` winners from `baseline_without_tssi.ipynb`.

**Kaggle note.** Per-selector CSVs (`selector_summary_long.csv`, `loco_*.csv`, …) were written to `/kaggle/working/model_feature_selectors` and are **not** in this repo. Tables below are reconstructed from the notebook’s displayed frames and the three compact PNGs embedded in the report cell. XGBoost’s 7-name three-way list was truncated in HTML as `… LV; WB…`; the alphabetically sorted completion is `WBC; eGFR`.

**Selectors.** LOCO = drop-one and refit on the val slice (cheap-importance prefix of 60). Coalition SHAP = permutation coalitions on a cheap-importance universe of 40 (not LOCO’s names). FFS = greedy forward search on its own 24-name pool, stop at 12 steps or when PR-AUC stops rising. Objective: **`pr_auc` only**. These catalogues are **interpretation / attribution**, not prediction, and do **not** feed Part 4. SMOTE is not used.

**Asset root:** [02_ml_selectors/paper_figures/](02_ml_selectors/paper_figures/)

---

## Contents

1. [Classic models (Table 0)](#1-classic-models)
2. [How much each selector keeps (Figure 1, Table 3, Figure 7)](#2-how-much-each-selector-keeps)
3. [Cross-model consensus (Table 1, Figure 6, Figure 2)](#3-cross-model-consensus)
4. [Within-model consensus, by classic model (Table 2, Figures 3–5)](#4-within-model-consensus-by-classic-model)
5. [Global intersection (Table 4)](#5-global-intersection)
6. [Priority-feature ranks (Table 5)](#6-priority-feature-ranks)
7. [Notebook compact plots (supplementary)](#7-notebook-compact-plots-supplementary)
8. [File index](#8-file-index)

---

## 1. Classic models

### Table 0. Models used in the selector notebook

![Table 0](02_ml_selectors/paper_figures/paper_table0_classic_models.png)

**Table 0.** Seven sklearn-style classifiers from the notebook `MODEL_SPECS` (TabPFN omitted). Row colour encodes family: linear (navy), bagged trees (teal), boosting (violet). All seven share the same 88-column scaled matrix; only the inductive bias changes.

| Code | Classic model | Family | GPU | Specification (notebook) |
| --- | --- | --- | --- | --- |
| lr | Logistic regression | Linear | No | L2-penalized log-odds (C=2, balanced); additive on scaled features |
| rf | Random forest | Bagged trees | No | 500 deep trees, class_weight=balanced_subsample |
| rf_b | Random forest (subsample) | Bagged trees | No | Same as RF with max_samples=0.88 |
| cat | CatBoost | Boosting | Yes | GPU **Plain** boosting (`task_type=GPU`; Ordered is CPU-only); `eval_metric=PRAUC`; balanced class weights |
| xgb | XGBoost | Boosting | Yes | scale_pos_weight for VLST imbalance |
| xgb_b | XGBoost (subsample) | Boosting | Yes | Lower subsample / colsample_bytree (0.78) |
| lgb | LightGBM | Boosting | Yes | Leaf-wise growth, balanced class weights |

**How to read later tables through this lens.**

- **Logistic regression** can only use additive log-odds. Consensus sits on renal labs (`Cre`, `eGFR`), inflammation (`WBC`), sex (`Men`), ACS presentation (`UA`), and `LV`.
- **Random forests** split on interactions and keep both a lab and its clinical twin (`LVEF` beside `LV`).
- **Boosting** recovers post-dilation and `WBC` most often; LightGBM’s three-way set is `HbA1c; LV` only.

**Source files:** [paper_figures/paper_table0_classic_models.png](02_ml_selectors/paper_figures/paper_table0_classic_models.png), [paper_figures/paper_table0_classic_models.csv](02_ml_selectors/paper_figures/paper_table0_classic_models.csv)

---

## 2. How much each selector keeps

LOCO scores a 60-name cheap-importance prefix, so every model reports **60** unique LOCO names in `selector_summary_long`. SHAP’s universe is **40** by construction. FFS is the path length after early stop (4–12). These are **not** top-20 counts; the consensus tables below use top-20.

### Figure 1. Unique selected features by classic model and selector

![Figure 1](02_ml_selectors/paper_figures/paper_fig1_unique_counts.png)

**Figure 1.** Unique feature counts in the selector log (PR-AUC). LOCO saturates the 60-feature cap for every model because 60 columns were scored, not because 60 were independently important. FFS is sparse because it stops when PR-AUC stops rising.

| Model | Family | LOCO | SHAP | FFS |
| --- | --- | ---: | ---: | ---: |
| lr | Linear | 60 | 40 | 12 |
| rf | Bagged trees | 60 | 40 | 12 |
| rf_b | Bagged trees | 60 | 40 | 8 |
| cat | Boosting | 60 | 40 | 4 |
| xgb | Boosting | 60 | 40 | 12 |
| xgb_b | Boosting | 60 | 40 | 11 |
| lgb | Boosting | 60 | 40 | 5 |

**Source file:** [paper_figures/paper_fig1_unique_counts.png](02_ml_selectors/paper_figures/paper_fig1_unique_counts.png)

### Table 3. Union size per classic model

![Table 3](02_ml_selectors/paper_figures/paper_table3_union_by_model.png)

**Table 3.** Size of the union of **top-20** sets across LOCO, SHAP, and FFS (PR-AUC only). Feature-name lists were truncated in the notebook HTML and are not reconstructed here.

| Code | Classic model | Family | Union size |
| --- | --- | --- | ---: |
| lr | Logistic regression | Linear | 32 |
| rf | Random forest | Bagged trees | 35 |
| rf_b | Random forest (subsample) | Bagged trees | 34 |
| cat | CatBoost | Boosting | 32 |
| xgb | XGBoost | Boosting | 31 |
| xgb_b | XGBoost (subsample) | Boosting | 30 |
| lgb | LightGBM | Boosting | 30 |

**Source files:** [paper_figures/paper_table3_union_by_model.png](02_ml_selectors/paper_figures/paper_table3_union_by_model.png), [paper_figures/paper_table3_union_by_model.csv](02_ml_selectors/paper_figures/paper_table3_union_by_model.csv)

### Figure 7. Union size (same numbers as Table 3)

![Figure 7](02_ml_selectors/paper_figures/paper_fig7_union_by_model.png)

**Figure 7.** Per-model top-20 unions relative to the global unique count of **86** scored names (dashed line). No classic model recovers the full 86-name union on its own.

**Source file:** [paper_figures/paper_fig7_union_by_model.png](02_ml_selectors/paper_figures/paper_fig7_union_by_model.png)

---

## 3. Cross-model consensus

A feature is “shared by all 7 models” only if it appears in every classic model’s **top-20** for that selector (PR-AUC).

### Table 1. Features shared by all classic models

![Table 1](02_ml_selectors/paper_figures/paper_table1_common_by_algorithm.png)

**Table 1.** Cross-model intersection (row colour = selector). LOCO agrees on five labs/cardiac names. SHAP agrees only on `HGB` and `WBC`. FFS agrees on **nothing** — greedy paths diverge once each model’s own 24-name pool is searched independently.

| Algorithm | Metric | n common | Features shared by all 7 models |
| --- | --- | ---: | --- |
| LOCO | pr_auc | 5 | Cre; LV; LVEF; WBC; eGFR |
| SHAP | pr_auc | 2 | HGB; WBC |
| FFS | pr_auc | 0 | — |

**Source files:** [paper_figures/paper_table1_common_by_algorithm.png](02_ml_selectors/paper_figures/paper_table1_common_by_algorithm.png), [paper_figures/paper_table1_common_by_algorithm.csv](02_ml_selectors/paper_figures/paper_table1_common_by_algorithm.csv)

### Figure 6. Same intersection as bars

![Figure 6](02_ml_selectors/paper_figures/paper_fig6_cross_model_common.png)

**Figure 6.** Bar height is `n common` from Table 1; labels are the shared names.

**Source file:** [paper_figures/paper_fig6_cross_model_common.png](02_ml_selectors/paper_figures/paper_fig6_cross_model_common.png)

### Figure 2. Jaccard overlap of selector unions

![Figure 2](02_ml_selectors/paper_figures/paper_fig2_jaccard.png)

**Figure 2.** Jaccard index between the unions of **top-20** sets (all seven models pooled). LOCO vs SHAP = **0.62**; SHAP vs FFS = **0.48**; LOCO vs FFS = **0.43**. These are moderate because the three selectors are **independent**. The previous 0.95–0.97 figure was an artefact of nesting SHAP/FFS inside one LOCO pool.

**Source file:** [paper_figures/paper_fig2_jaccard.png](02_ml_selectors/paper_figures/paper_fig2_jaccard.png)

---

## 4. Within-model consensus, by classic model

Here the intersection is inside one model: names that LOCO, SHAP, and FFS all put in that model’s top-20 for PR-AUC.

### Table 2. LOCO ∩ SHAP ∩ FFS per model (PR-AUC)

![Table 2](02_ml_selectors/paper_figures/paper_table2_consensus_by_model.png)

**Table 2.** Within-model three-selector consensus. Row colour = family. XGBoost has the largest set (7 names); LightGBM the smallest (2).

| Code | Classic model | Family | Metric | n (LOCO ∩ SHAP ∩ FFS) | Consensus features |
| --- | --- | --- | --- | ---: | --- |
| lr | Logistic regression | Linear | pr_auc | 6 | Cre; LV; Men; UA; WBC; eGFR |
| rf | Random forest | Bagged trees | pr_auc | 6 | HGB; LDL; LVEF; Men; WBC; eGFR |
| rf_b | Random forest (subsample) | Bagged trees | pr_auc | 4 | CaI; HGB; LVEF; WBC |
| cat | CatBoost | Boosting | pr_auc | 3 | 1.1:1Post dilation; HGB; WBC |
| xgb | XGBoost | Boosting | pr_auc | 7 | 1.1:1Post dilation; Aneurysm; Cre; HGB; LV; WBC; eGFR |
| xgb_b | XGBoost (subsample) | Boosting | pr_auc | 5 | 1.1:1Post dilation; LV; LVEF; WBC; eGFR |
| lgb | LightGBM | Boosting | pr_auc | 2 | HbA1c; LV |

**Source files:** [paper_figures/paper_table2_consensus_by_model.png](02_ml_selectors/paper_figures/paper_table2_consensus_by_model.png), [paper_figures/paper_table2_consensus_by_model.csv](02_ml_selectors/paper_figures/paper_table2_consensus_by_model.csv)

**ML consensus catalogue (union of Table 2, n = 13):** `1.1:1Post dilation`, `Aneurysm`, `CaI`, `Cre`, `HGB`, `HbA1c`, `LDL`, `LV`, `LVEF`, `Men`, `UA`, `WBC`, `eGFR`. This is the set compared with statistical FDR in Part 3.

### Figure 3. Consensus-set size

![Figure 3](02_ml_selectors/paper_figures/paper_fig3_consensus_size.png)

**Figure 3.** Bar height is `n (LOCO ∩ SHAP ∩ FFS)` from Table 2. Colour = model family.

**Source file:** [paper_figures/paper_fig3_consensus_size.png](02_ml_selectors/paper_figures/paper_fig3_consensus_size.png)

### Figure 4. Which features each classic model agrees on

![Figure 4](02_ml_selectors/paper_figures/paper_fig4_feature_by_model.png)

**Figure 4.** Cell = 1 if the feature is in that model’s LOCO ∩ SHAP ∩ FFS set (PR-AUC). `WBC` appears in six of seven models; `eGFR` and `LV` in four; `1.1:1Post dilation` in the three boosting variants except LightGBM.

**Source file:** [paper_figures/paper_fig4_feature_by_model.png](02_ml_selectors/paper_figures/paper_fig4_feature_by_model.png)

### Figure 5. Family stacked counts

![Figure 5](02_ml_selectors/paper_figures/paper_fig5_family_stacked.png)

**Figure 5.** For each consensus feature, how many models in each family include it. `WBC` has support in all three families. `Men` is linear + bagged. `1.1:1Post dilation` is boosting-only. `Aneurysm` is XGBoost-only.

**Source file:** [paper_figures/paper_fig5_family_stacked.png](02_ml_selectors/paper_figures/paper_fig5_family_stacked.png)

### Reading Table 2 / Figures 3–5 by classic model

**Logistic regression (`lr`).** Linear three-way: `Cre`, `LV`, `Men`, `UA`, `WBC`, `eGFR`. Sex and unstable-angina are almost unique to LR among the consensus names.

**Random forest (`rf`).** `HGB`, `LDL`, `LVEF`, `Men`, `WBC`, `eGFR`. Keeps haemoglobin and LDL that the linear model does not.

**Random forest, subsampled (`rf_b`).** `CaI`, `HGB`, `LVEF`, `WBC`. Treat `rf_b` as a sensitivity check on `rf`; `CaI` is unique to this variant.

**CatBoost (`cat`).** `1.1:1Post dilation`, `HGB`, `WBC`. Does not put `eGFR` or `LV` in the three-way set on this run.

**XGBoost (`xgb` / `xgb_b`).** Both recover `1.1:1Post dilation`, `LV`, `WBC`; the full XGB run also keeps `Aneurysm`, `Cre`, `HGB`, `eGFR`.

**LightGBM (`lgb`).** Smallest three-way: `HbA1c; LV`. This is the only model that puts `HbA1c` in the intersection — and that is enough to put `HbA1c` in the Part 3 consensus union.

---

## 5. Global intersection

### Table 4. Strictest intersection vs global union

![Table 4](02_ml_selectors/paper_figures/paper_table4_global_common.png)

**Table 4.** Features that appear in **every** model × selector top-20: **none**. The complementary union of all scored names is **86**.

| Scope | n features | Features |
| --- | ---: | --- |
| All 7 models × LOCO, SHAP, FFS (PR-AUC top-20) | 0 | — |
| Any model / selector (union of scored names) | 86 | 86 unique names (full string not downloaded from Kaggle) |

**Source files:** [paper_figures/paper_table4_global_common.png](02_ml_selectors/paper_figures/paper_table4_global_common.png), [paper_figures/paper_table4_global_common.csv](02_ml_selectors/paper_figures/paper_table4_global_common.csv)

---

## 6. Priority-feature ranks

The notebook scores a hand-specified `PRIORITY_FEATURES` list (Wang Table 1 English labels) against each model × selector ranking. Most labels **do not match** the dataset column names (`Age, years` vs `Age`, `Male sex` vs `Men`, `aspirin` vs `Aspirin`). The display is the first 20 rows: CatBoost × LOCO then SHAP, PR-AUC only.

### Table 5. Priority ranks (display excerpt)

![Table 5](02_ml_selectors/paper_figures/paper_table5_priority_ranks_excerpt.png)

**Table 5.** Hits under CatBoost LOCO: Current smoker (rank 24), Clopidogrel (50), Current drinking (51), Hypertension (55). SHAP hit: Hypertension (35). The rest miss because of the alias mismatch, not because the clinical variables were unscored.

**Source files:** [paper_figures/paper_table5_priority_ranks_excerpt.png](02_ml_selectors/paper_figures/paper_table5_priority_ranks_excerpt.png), [paper_figures/paper_table5_priority_ranks_excerpt.csv](02_ml_selectors/paper_figures/paper_table5_priority_ranks_excerpt.csv)

---

## 7. Notebook compact plots (supplementary)

![Figure S1](02_ml_selectors/paper_figures/selector_model_algorithm_counts.png)

**Supplementary Figure S1.** Notebook heatmap of unique selected-feature counts. Paper restyle: Figure 1.

**Source file:** [paper_figures/selector_model_algorithm_counts.png](02_ml_selectors/paper_figures/selector_model_algorithm_counts.png)

![Figure S2](02_ml_selectors/paper_figures/selector_top_repeated_features.png)

**Supplementary Figure S2.** Features most often written into `selector_summary_long` (max 21 = 7 models × 3 selectors). `WBC` leads; `HGB` / `LV` / post-dilation complements / `eGFR` follow. `Stent type-SES_resolute` (a 9-level dummy) appears in the top 25 — the brand signal is now a named level, not 106 fragments.

**Source file:** [paper_figures/selector_top_repeated_features.png](02_ml_selectors/paper_figures/selector_top_repeated_features.png)

![Figure S3](02_ml_selectors/paper_figures/selector_overlap_heatmap.png)

**Supplementary Figure S3.** Notebook Jaccard heatmap of selector unions. Paper restyle: Figure 2.

**Source file:** [paper_figures/selector_overlap_heatmap.png](02_ml_selectors/paper_figures/selector_overlap_heatmap.png)

---

## 8. File index

| ID | Type | File |
| --- | --- | --- |
| Table 0 | Table | [paper_table0_classic_models.png](02_ml_selectors/paper_figures/paper_table0_classic_models.png) |
| Fig 1 | Figure | [paper_fig1_unique_counts.png](02_ml_selectors/paper_figures/paper_fig1_unique_counts.png) |
| Table 1 | Table | [paper_table1_common_by_algorithm.png](02_ml_selectors/paper_figures/paper_table1_common_by_algorithm.png) |
| Fig 2 | Figure | [paper_fig2_jaccard.png](02_ml_selectors/paper_figures/paper_fig2_jaccard.png) |
| Table 2 | Table | [paper_table2_consensus_by_model.png](02_ml_selectors/paper_figures/paper_table2_consensus_by_model.png) |
| Fig 3 | Figure | [paper_fig3_consensus_size.png](02_ml_selectors/paper_figures/paper_fig3_consensus_size.png) |
| Fig 4 | Figure | [paper_fig4_feature_by_model.png](02_ml_selectors/paper_figures/paper_fig4_feature_by_model.png) |
| Fig 5 | Figure | [paper_fig5_family_stacked.png](02_ml_selectors/paper_figures/paper_fig5_family_stacked.png) |
| Fig 6 | Figure | [paper_fig6_cross_model_common.png](02_ml_selectors/paper_figures/paper_fig6_cross_model_common.png) |
| Table 3 | Table | [paper_table3_union_by_model.png](02_ml_selectors/paper_figures/paper_table3_union_by_model.png) |
| Fig 7 | Figure | [paper_fig7_union_by_model.png](02_ml_selectors/paper_figures/paper_fig7_union_by_model.png) |
| Table 4 | Table | [paper_table4_global_common.png](02_ml_selectors/paper_figures/paper_table4_global_common.png) |
| Table 5 | Table | [paper_table5_priority_ranks_excerpt.png](02_ml_selectors/paper_figures/paper_table5_priority_ranks_excerpt.png) |
| Fig S1 | Supp. figure | [selector_model_algorithm_counts.png](02_ml_selectors/paper_figures/selector_model_algorithm_counts.png) |
| Fig S2 | Supp. figure | [selector_top_repeated_features.png](02_ml_selectors/paper_figures/selector_top_repeated_features.png) |
| Fig S3 | Supp. figure | [selector_overlap_heatmap.png](02_ml_selectors/paper_figures/selector_overlap_heatmap.png) |

---

*Numbers from the 2026-08-31 paper-protocol Kaggle run of* `baseline_feature_selections.ipynb` *(seven classic models, PR-AUC, independent selectors, 9-level stent encoder → 88 columns, fit/val 4148/1037). Regenerated by* `code/modeling/tools/rebuild_part2_paper_figures.py`.

---

# Part 3. Statistical vs ML feature extraction

### Statistical vs machine-learning feature extraction in VLST

This note compares **what was extracted** from the same VLST cohort by (i) classical statistical association tests and (ii) classic-model feature selectors, then explains **why the two catalogues only partly overlap**. This is a methods comparison of two association / attribution catalogues, not a prediction result.

Sources: [EDA_paper_figures_and_tables.md](01_eda/EDA_paper_figures_and_tables.md) (`eda.ipynb`) and [baseline_feature_selections_paper_figures_and_tables.md](02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md) (`baseline_feature_selections.ipynb`). Overlap arithmetic and figures are produced by [`stats_vs_ml_comparison.ipynb`](../code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb).

**Asset root:** [03_stats_vs_ml/paper_figures/](03_stats_vs_ml/paper_figures/)

---

## Contents

1. [What each approach is asking](#1-what-each-approach-is-asking)
2. [How common are the extracted features?](#2-how-common-are-the-extracted-features)
3. [Features found by both approaches](#3-features-found-by-both-approaches)
4. [Statistical-only features](#4-statistical-only-features)
5. [Machine-learning-only features](#5-machine-learning-only-features)
6. [Domain pattern](#6-domain-pattern)
7. [Methodological reasons for disagreement](#7-methodological-reasons-for-disagreement)
8. [File index](#8-file-index)

---

## 1. What each approach is asking

The two pipelines are not two estimates of the same quantity. They optimize different questions on slightly different feature views.


|                              | Statistical EDA                                                                       | Classic-model selectors                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Question**                 | Does this column’s *marginal* distribution differ by VLST after multiplicity control? | If I train `lr` / `rf` / boosting, which columns does the *fitted model* need for hold-out PR-AUC? |
| **Unit of evidence**         | One test per feature (Welch, Mann–Whitney, χ² / Fisher) plus FDR                      | LOCO (refit without the column), coalition SHAP, greedy FFS                                                  |
| **Sample**                   | Full cohort, n = 5,185, 92 events                                                     | Full-cohort fit/val: 4,148 / 1,037 rows (74 / 18 events). No unused outer test. |
| **Feature view**             | Raw clinical columns; `Stent type-SES` collapsed to 9 levels for χ²                  | Scaled matrix (**88 columns**): shared 9-level brand encoder, then OHE drop-first + `StandardScaler`. Imputers are inert (no NaNs). |
| **Discovery rule used here** | Univariate FDR q < 0.05 (plus a sparse multivariable logistic check)                  | Names in **LOCO ∩ SHAP ∩ FFS** top-20 for at least one model (PR-AUC)                                        |
| **Multiplicity**             | Benjamini–Hochberg across the tested family                                           | Top-20 of independent cheap-importance pools (LOCO 60 / SHAP 40 / FFS 24)                     |


**Statistical catalogue (n = 20 names, excluding time-at-risk).** Continuous FDR: `WBC`, `eGFR`, `LV`, `CKD5`, `No.of stents per lesion`, `HbA1c`, `NO.of vessels`, `Total stent length`, `Fiberinogen`. Binary FDR: `1.1:1Post dilation`, `No postdilation`, `CKD90`, `Previous PCI`, `3-vessel disease`, `Clopidogrel`, `Diabetes`, `PES`, `Multi-vessel CAD`, `Single-vessel disease`. Categorical: `Stent type-SES`. `Time since stent implantation` is the strongest univariate hit but is a follow-up / time-at-risk variable, not a baseline covariate, and is excluded from ML.

Of these 20 names, at least 8 are redundant re-encodings: the post-dilation complements (`1.1:1Post dilation`, `No postdilation` — 2 slots for 1 bit), the vessel-disease family (`3-vessel disease`, `Multi-vessel CAD`, `Single-vessel disease`, `NO.of vessels` — 4 slots for 1 construct), and the renal family (`CKD5`, `CKD90` alongside continuous `eGFR` — 3 slots for 1 construct). The “20 statistical discoveries” headline is a **name count**. Report it as roughly **12 distinct clinical constructs**. Jaccard 5/28 still uses the name lists (that is what the selectors and FDR tests emit); do not rewrite the Venn as 12 vs 13.

**ML consensus catalogue (n = 13).** Union of LOCO ∩ SHAP ∩ FFS names across logistic regression, random forests, CatBoost, XGBoost, and LightGBM (PR-AUC, top-20): `1.1:1Post dilation`, `Aneurysm`, `CaI`, `Cre`, `HGB`, `HbA1c`, `LDL`, `LV`, `LVEF`, `Men`, `UA`, `WBC`, `eGFR`.

A looser ML set (**frequent selection**, top-repeated names in the selector log) additionally includes `No postdilation`, `Previous PCI`, `STEMI`, `stent overlap`, lipids, and `Stent type-SES_resolute` (a 9-level dummy) that are selected often but rarely survive the three-selector intersection.

---

## 2. How common are the extracted features?

Only **5 of 20** statistical FDR *names* also sit in the ML three-selector consensus. Conversely, **8 of 13** ML-consensus names fail univariate FDR. Jaccard overlap of the 20-name and 13-name sets is 5 / 28 ≈ **0.18**. On the construct reading (~12 statistical constructs), several of the 15 “stats-only” names are the extra slots in those three families, not 15 independent missed discoveries.

### Figure 1. Overlap of the two extraction catalogues

![Figure 1](03_stats_vs_ml/paper_figures/fig1_venn_overlap.png)

**Figure 1.** Left circle: univariate FDR q < 0.05 (time-since-stent excluded). Right circle: features in LOCO ∩ SHAP ∩ FFS top-20 for at least one classic model (PR-AUC). The intersection is `WBC`, `eGFR`, `LV`, `HbA1c`, `1.1:1Post dilation`.

**Source file:** [paper_figures/fig1_venn_overlap.png](03_stats_vs_ml/paper_figures/fig1_venn_overlap.png)

### Figure 2. Presence by extractor

![Figure 2](03_stats_vs_ml/paper_figures/fig2_presence_heatmap.png)

**Figure 2.** Navy cells mark membership. Columns: statistical FDR; statistical multivariable (Wald CI excluding 1 in the sparse logistic); ML consensus; ML frequent (top-repeated selector log). `Time since stent implantation` is statistical-only by construction (dropped before ML). `LVEF` and `Men` are ML-side even though they fail FDR. `Previous PCI` is FDR + frequent, but not three-way consensus.

**Source file:** [paper_figures/fig2_presence_heatmap.png](03_stats_vs_ml/paper_figures/fig2_presence_heatmap.png)

### Table 1. Membership of every compared name

![Table 1](03_stats_vs_ml/paper_figures/table_feature_by_method.png)

**Table 1.** Green = both catalogues; navy tint = stats FDR only; violet = ML consensus only; pale violet = ML frequent only; orange = structural time-at-risk.

**Source files:** [paper_figures/table_feature_by_method.png](03_stats_vs_ml/paper_figures/table_feature_by_method.png), [paper_figures/table_feature_by_method.csv](03_stats_vs_ml/paper_figures/table_feature_by_method.csv)

---

## 3. Features found by both approaches

These five names are the only ones that are both a **full-cohort association discovery** and a **predictive-model necessity** under the paper-protocol selectors.

### Table 2. Shared features

![Table 2](03_stats_vs_ml/paper_figures/table_shared_features.png)


| Feature | Domain | Statistical evidence | ML evidence | Why both keep it |
| --- | --- | --- | --- | --- |
| WBC | Laboratory | MW r = 0.13, q = 9.5e-20 | Cross-model LOCO and SHAP; in 6/7 model three-way sets | Inflammation is a mean shift *and* a column models cannot replace |
| eGFR | Laboratory | Welch d = −0.71, q = 3.7e-19 | Cross-model LOCO; lr/rf/xgb_b three-way | Filtration: largest continuous effect; LOCO drop is costly |
| LV | Cardiac | Welch d = 1.13, q = 3.3e-16 | Cross-model LOCO; lr/lgb/xgb/xgb_b three-way | Large location shift and a high-gain tree split |
| HbA1c | Laboratory | MW r = 0.052, q = 7e-4 | LightGBM LOCO ∩ SHAP ∩ FFS | Glycaemic FDR hit that LightGBM also needs for PR-AUC |
| 1.1:1Post dilation | Procedural | χ² OR = 0.187, q = 3.7e-9 | CatBoost / XGBoost / XGB_b three-way | Strong 2×2 and a split boosting models cannot replace |


**Source files:** [paper_figures/table_shared_features.png](03_stats_vs_ml/paper_figures/table_shared_features.png), [paper_figures/table_shared_features.csv](03_stats_vs_ml/paper_figures/table_shared_features.csv)

`WBC` is the closest thing to a global ML intersection (six of seven models). There is **no** name in all 7 × 3 selector top-20s. `Fiberinogen` and `Previous PCI` were shared hits on the old F1/F2 test-scored run; they are now stats-only (Previous PCI remains frequently selected).

---

## 4. Statistical-only features

Fifteen FDR discoveries never enter ML LOCO ∩ SHAP ∩ FFS. They are not “false”; they fail a *different* filter: a 20-column predictive shortlist on an 88-column encoded matrix, scored on an 18-event val slice.

### Table 3. FDR hits missing from ML consensus

![Table 3](03_stats_vs_ml/paper_figures/table_stats_only.png)


| Feature | Domain | Why statistics keeps it and ML top-20 does not |
| --- | --- | --- |
| No postdilation | Procedural | Univariate OR 5.4; multivariable OR collapses toward 1 once the complement is modelled. Boosting keeps the 1.1:1 flag instead |
| CKD90 | Renal cutpoint | Binary threshold on the same axis as `eGFR`. ML keeps the continuous lab, not the cut |
| CKD5 | Renal cutpoint | FDR hit; adjusted OR *flips sign* (collinear with eGFR). Often selected, not in 3-way consensus |
| 3-vessel disease | Anatomy | χ² discovery; collinear with `NO.of vessels` / multi-vessel CAD |
| Multi-vessel CAD | Anatomy | Same information as single-vessel (complements) |
| Single-vessel disease | Anatomy | Complement of multi-vessel disease (same 2×2 inverted) |
| NO.of vessels | Anatomy | Continuous count of the same anatomy cluster |
| No.of stents per lesion | Procedural | Tiny effect (MW r = 0.037); not in any model three-way set |
| Total stent length | Procedural | Small effect; collinear with stent count / vessel burden |
| Clopidogrel | Medication | Full-cohort drug association; trees split on labs/procedure instead |
| Diabetes | Comorbidity | Univariate FDR; multivariable CI includes 1; trees may split on HbA1c |
| PES | Stent type | Polymer binary; collinear with the 9-level brand column |
| Stent type-SES | Stent type | χ² on 9 collapsed brands. ML one-hots those 9 levels; the parent name never enters a 3-way set |
| Previous PCI | History | Fisher OR 6.49. Frequently selected, but no model puts it in LOCO ∩ SHAP ∩ FFS on PR-AUC |
| Fiberinogen | Laboratory | Weak MW r = 0.035. Was RF F2 consensus on the old run; PR-AUC three-way no longer keeps it |


**Source files:** [paper_figures/table_stats_only.png](03_stats_vs_ml/paper_figures/table_stats_only.png), [paper_figures/table_stats_only.csv](03_stats_vs_ml/paper_figures/table_stats_only.csv)

Three recurring mechanisms:

1. **Collinear families.** Univariate tests score *every* member of a redundant block (vessel-disease binaries, postdilation complements, CKD cutpoints vs eGFR, PES vs stent type). FDR can declare several of them significant. A fitted model only needs one representative.
2. **Encoding.** `Stent type-SES` is one χ² test on 9 levels. In the scaled ML view it becomes 8 sparse dummies; `Stent type-SES_resolute` is frequently selected, but the parent name is not in any three-way set.
3. **Different filter, not a missing pool.** Selectors now rank their own cheap-importance prefixes (60 / 40 / 24). Absence from consensus means “not in LOCO ∩ SHAP ∩ FFS top-20,” not “never scored.”

---

## 5. Machine-learning-only features

Eight consensus names fail univariate FDR. ML is not “finding associations the tests missed” in the NHST sense; it is finding **columns that change a model’s hold-out PR-AUC**, including surrogates, interactions, and weak splits.

### Table 4. ML consensus names that fail FDR

![Table 4](03_stats_vs_ml/paper_figures/table_ml_only.png)


| Feature | Univariate vs VLST | Why ML consensus keeps it and FDR does not |
| --- | --- | --- |
| Cre | ns (p = 0.88) | Redundant with eGFR marginally; still a renal surrogate when eGFR is noisy or left out |
| Men | ns (p = 0.27) | `Men × eGFR` interaction is FDR-significant in the EDA screen; LR uses sex as an additive offset |
| LVEF | raw p = 0.033, FDR ns | Borderline mean test. Domain joint logistic **reverses sign** (uni OR 0.851 → adj 1.65) when `LV` is in the same model. Trees still split on systolic function |
| HGB | raw p = 0.039, FDR ns | CatBoost/RF/XGB three-way: ranking, not a location test |
| CaI | raw p = 0.051, FDR ns | RF_b three-way; sits on the FDR boundary |
| LDL | ns (p = 0.33) | RF three-way lipid split on the val-slice PR-AUC |
| UA | ns (p = 0.17) | LR three-way ACS-presentation offset |
| Aneurysm | ns (p = 0.40) | Rare anatomy flag; XGB three-way only — treat as unstable |


**Source files:** [paper_figures/table_ml_only.png](03_stats_vs_ml/paper_figures/table_ml_only.png), [paper_figures/table_ml_only.csv](03_stats_vs_ml/paper_figures/table_ml_only.csv)

Three recurring mechanisms:

1. **Surrogates of a stronger FDR hit.** `Cre` carries almost no univariate VLST signal because `eGFR` already does. A linear or tree model that cannot use eGFR (or that splits on creatinine first) will still list Cre.
2. **Interactions and offsets that univariate tests do not see.** `Men` is not associated with VLST on its own (p = 0.27), but `Men × eGFR` is an FDR-significant interaction in the EDA screen, and the domain joint logistic gives Men an adjusted OR of 3.3.
3. **Different error and sample.** FDR is a full-cohort mean/2×2 statement with 92 events. LOCO/SHAP/FFS optimize PR-AUC on 18 val events. Weak ACS/lipid/anatomy splits can move that metric without moving a χ² p-value across the FDR line. `Aneurysm` (XGB only) is the clearest example.

The old F1/F2-only names (`Platelet`, `HL`, `STEMI`, `Current drinking`, `History of HF`, `Hypertension`, `TG`, `TCL`, `Min-stent diameter`, `Fast-Glu`) are **no longer in the consensus**. They were operating-point artefacts of the prior three-metric export.

---

## 6. Domain pattern

### Figure 4. Extracted counts by clinical domain

![Figure 4](03_stats_vs_ml/paper_figures/fig4_domain_counts.png)

**Figure 4.** Statistical FDR is concentrated in laboratory, procedural/stent, and anatomy blocks (the EDA domain screen). ML consensus is heavier on laboratory *plus* cardiac function and demographics, and thinner on anatomy binaries and medications. Post-dilation now appears on both sides (the 1.1:1 flag, not its complement).

**Source file:** [paper_figures/fig4_domain_counts.png](03_stats_vs_ml/paper_figures/fig4_domain_counts.png)

Statistics therefore still “owns” **anatomy coding and most stent-technique flags**. Machine learning “owns” **cardiac function twins** (`LVEF` next to `LV`), **sex**, and **labs that are collinear with FDR hits** (`Cre`, `HGB`, `LDL`). Both own **WBC, eGFR, LV, HbA1c, and 1.1:1 post-dilation**.

---

## 7. Methodological reasons for disagreement

### Figure 3. Buckets

![Figure 3](03_stats_vs_ml/paper_figures/fig3_reason_buckets.png)

**Figure 3.** Counts of names in this comparison assigned to a primary methodological bucket (one bucket per feature; the anatomy/stent collinear family is grouped).

**Source file:** [paper_figures/fig3_reason_buckets.png](03_stats_vs_ml/paper_figures/fig3_reason_buckets.png)

**Why a feature can appear in statistics and not in ML**

- Univariate tests do not penalize redundancy. FDR will list `3-vessel disease`, `Multi-vessel CAD`, `Single-vessel disease`, and `NO.of vessels` if each 2×2/t-test is small. A model only needs one of them.
- Complements are two encodings of one bit (`1.1:1Post dilation` vs `No postdilation`). χ² sees both; boosting kept the 1.1:1 flag and dropped the complement from the three-way set.
- Categorical χ² on `Stent type-SES` does not survive as the parent name after 9-level one-hot (the `resolute` dummy is frequently selected instead).
- Hold-out PR-AUC with 18 val events is under-powered for moderate ORs (Clopidogrel 0.50, Diabetes 1.89, Previous PCI 6.49) that FDR can still detect on 92 events.

**Why a feature can appear in ML and not in statistics**

- Predictive importance is not a marginal p-value. LOCO asks whether the *rest of the model* can compensate after a refit. SHAP asks for coalition credit. FFS asks for greedy hold-out gain. None of these is a two-sample test.
- Correlated twins: the univariate test of `Cre` is null because `eGFR` already captures renal function; the model may still split on Cre.
- Interactions: `Men × eGFR` is an EDA FDR hit; univariate `Men` is not. LR consensus includes `Men`.
- Independent selectors disagree. LightGBM’s three-way set is `HbA1c; LV` only. Some ML-only names (`Aneurysm`) are **algorithm artefacts**, not cohort discoveries.

**Practical reading.** Treat the 5-name intersection (`WBC`, `eGFR`, `LV`, `HbA1c`, `1.1:1Post dilation`) as the robust extraction set: associated in the cohort *and* used by fitted classic models under PR-AUC. Treat statistical-only anatomy/stent/drug names as **association findings that need a non-redundant representative** before they enter a predictor. Treat ML-only names as **hypothesis-generating predictive correlates** until they pass a pre-specified association or external-validation bar. This comparison is a **methods result**, not a biological ranking of “true” risk markers.

---

## 8. File index


| ID      | Type   | File                                                                     |
| ------- | ------ | ------------------------------------------------------------------------ |
| Fig 1   | Figure | [fig1_venn_overlap.png](03_stats_vs_ml/paper_figures/fig1_venn_overlap.png)             |
| Fig 2   | Figure | [fig2_presence_heatmap.png](03_stats_vs_ml/paper_figures/fig2_presence_heatmap.png)     |
| Table 1 | Table  | [table_feature_by_method.png](03_stats_vs_ml/paper_figures/table_feature_by_method.png) |
| Table 2 | Table  | [table_shared_features.png](03_stats_vs_ml/paper_figures/table_shared_features.png)     |
| Table 3 | Table  | [table_stats_only.png](03_stats_vs_ml/paper_figures/table_stats_only.png)               |
| Table 4 | Table  | [table_ml_only.png](03_stats_vs_ml/paper_figures/table_ml_only.png)                     |
| Fig 3   | Figure | [fig3_reason_buckets.png](03_stats_vs_ml/paper_figures/fig3_reason_buckets.png)         |
| Fig 4   | Figure | [fig4_domain_counts.png](03_stats_vs_ml/paper_figures/fig4_domain_counts.png)           |


---

*Statistical names: univariate FDR q < 0.05 from* `eda.ipynb` *(time-since-stent excluded from the overlap count). ML names: LOCO ∩ SHAP ∩ FFS top-20, PR-AUC only, 2026-08-31 paper-protocol run of* `baseline_feature_selections.ipynb` *(seven classic models; 9-level stent encoder; fit/val 4148/1037). Figures and CSVs regenerated by* `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb`.

---

# Part 4. Nested-CV baselines plus TabPFN

### Nested-CV baselines plus TabPFN — paper figures and tables

This document gathers publication-oriented figures and tables from the nested cross-validation comparison in `baseline_plus_tabpfn.ipynb`.

**Cohort / protocol.** Full VLST cohort, n = 5,185 (92 events; prevalence = 0.0177). Target = `Stent thrombosis`. Identifiers (`NO.`, `Name`) and `Time since stent implantation` are dropped; the latter is treated as a time-at-risk / follow-up column, not a baseline covariate. **No Part 2 / Part 5 feature mask is applied.** Evaluation is nested stratified CV: **5 outer folds / 4 inner folds** (outer `random_state=42`). Ranking metrics (PR-AUC, ROC-AUC, Brier) use pooled outer out-of-fold probabilities and are threshold-independent. For precision / recall / F1 / F2, **quote the nested inner-fold thresholds** (Table 2): each outer fold’s cut is chosen on inner OOF scores and applied once to that fold’s unseen cases. Figure 3 / Table 3 additionally show a single pooled F1 cut; that cut is **optimistically biased** (methods note below). These nested-CV metrics are this pack’s only **prediction** results.

**Methods note — feature views are not the same.** The five classic models sit in an sklearn `Pipeline` with a `ColumnTransformer` that is **cloned and fitted inside every CV split**: numeric columns get `SimpleImputer(median)` + `StandardScaler`; `Stent type-SES` gets most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`. EDA found **no missing values**, so both imputers are inert. What actually changes the comparison is the rest of the transformer: classics see **scaled + one-hot input (~186 columns)** because the 106 raw brand strings become 106 dummy columns. TabPFN is **not** in that pipeline — it receives the **raw 81-column frame** and handles the brand column natively. That unequal input must be read into every “same protocol” claim.

**Methods note — GridSearch is a different notebook.** `baseline_without_tssi.ipynb` / `baseline_tssi_leakage.ipynb` tune hyperparameters on a single 70/30 split. Those `best_params_` are **not** imported here. Classics in this nested CV use library defaults plus class weighting. The inner loop tunes only the F1 **threshold**. A seventh arm, `TabPFN (local)` (`from tabpfn import TabPFNClassifier`, no thinking), is now in the notebook via `RUN_MODELS`; it is **not** in the stored six-model run below.

**Methods note — why the follow-up-time column is dropped.** Wang 2020 analysed this cohort with Cox regression, in which follow-up duration is the *time axis*, not a covariate. Recoded as a binary classifier, the same column (`Time since stent implantation`) mixes two definitions: time-to-event for the 92 VLST cases (min 380 days) and event-free follow-up length for the 5,093 non-events (min 1,241 days). A rule “time < 1,241 → event” has zero false positives among controls. `baseline_tssi_leakage.ipynb` (same 70/30 split, GridSearchCV) shows the resulting inflation; `baseline_without_tssi.ipynb` is the identical protocol with the column removed. Nested-CV results in this document use the without-TSSI feature view. See Supplementary Table S-TSSI.

**Models.** Logistic regression, random forest, XGBoost, LightGBM, CatBoost, and TabPFN (client, `thinking_mode=True`, `thinking_effort="high"`, `thinking_metric="average_precision"`). Average precision (PR-AUC) is the common ranking metric.

**Methods note — published clinical baseline.** Wang 2020’s 8-variable integer score is scored as a **frozen** comparator in `code/modeling/rating/wang_vlst_score.ipynb` (published Table 2 points; weights not re-fit). It is not a seventh nested-CV arm. See Supplementary Table S-Wang.

**Methods note — two F1 operating points.** Ranking metrics do not use a threshold. Precision, recall, F1, and F2 do. The executed notebook prints both. **Honest nested** (Table 2): inner-CV OOF F1 threshold applied once to the unseen outer fold. **Optimistic pooled** (Figure 3, Table 3): one F1-maximising cut on the concatenated OOF labels that are then scored. Reusing the evaluation labels to pick the cut **optimistically biases** precision, recall, F1, and F2. Quote Table 2. TabPFN nested recall is **0.7174**; pooled-notebook recall is 0.7935; the exported PNG still shows a stale run (recall **0.8370** at t = 0.901). Do not quote 0.837.

**Methods note — imbalance, SMOTE, and tuning.** Prevalence is 1.77%. Class weighting (`class_weight="balanced"`, `scale_pos_weight`, `auto_class_weights="Balanced"`) is used for *prediction* so the 92 events are not ignored. SMOTE is **not** used: synthetic minority rows would change the prevalence that PR-AUC, PPV, Brier and calibration depend on. The five classic models use library defaults plus class weighting and a PR-AUC / PRAUC eval metric; they are **not** grid-searched. TabPFN uses client thinking (`effort=high`). That is an unequal search budget and is disclosed here. Inner nested CV selects only the F1 **threshold**, not hyperparameters. Wald tests / GLM standard errors are not used for these classifiers — they are not inferential logit models.

**Asset root:** [04_tabpfn_rating/paper_figures/](04_tabpfn_rating/paper_figures/)

> The **executed notebook** does print fold-wise mean ± SD, nested operating points, and the comparison table (see `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt`). Those CSVs were not committed under `data/result/`. The **exported PNGs** below are a mix: classic-model panels match the notebook; TabPFN Brier / confusion panels are from an earlier client run (stale). Quote the notebook for TabPFN.

---

## Contents

1. [Models (Table 0)](#1-models)
2. [Ranking curves (Figure 1, Table 1)](#2-ranking-curves)
3. [Calibration (Figure 2)](#3-calibration)
4. [F1 operating point (Table 2 nested; Figure 3 / Table 3 pooled)](#4-f1-operating-point)
5. [Supplementary: follow-up-time leakage](#5-supplementary-follow-up-time-leakage)
6. [Supplementary: Wang 2020 integer score](#6-supplementary-wang-2020-integer-score)
7. [File index](#7-file-index)

---

## 1. Models

### Table 0. Nested-CV models

![Table 0](04_tabpfn_rating/paper_figures/paper_table0_models.png)

**Table 0.** Six classifiers compared under the same nested-CV *split and threshold* protocol. They do **not** see the same columns: classics get the scaled one-hot matrix; TabPFN gets raw 81 features (see Methods note above). Tree boosters use average-precision / PR-AUC as their internal metric; TabPFN is the Prior Labs client with thinking mode aimed at average precision. Classics are not grid-searched.

| Model | Family | GPU | Specification (notebook) |
| --- | --- | --- | --- |
| Logistic Regression | Linear | No | L2, class_weight=balanced, max_iter=1000 |
| Random Forest | Bagged trees | No | class_weight=balanced, random_state=42 |
| XGBoost | Boosting | Yes | eval_metric=aucpr; scale_pos_weight from train fold |
| LightGBM | Boosting | Yes | metric=average_precision; class_weight=balanced |
| CatBoost | Boosting | Yes | auto_class_weights=Balanced; eval_metric=PRAUC |
| TabPFN | Foundation (tabular) | Client GPU | thinking=True, effort=high, metric=average_precision |

**Source files:** [paper_figures/paper_table0_models.png](04_tabpfn_rating/paper_figures/paper_table0_models.png), [paper_figures/paper_table0_models.csv](04_tabpfn_rating/paper_figures/paper_table0_models.csv)

---

## 2. Ranking curves

### Figure 1. Nested-CV out-of-fold PR and ROC curves

![Figure 1](04_tabpfn_rating/paper_figures/paper_fig1_pr_roc_curves.png)

**Figure 1.** Precision–recall (left) and ROC (right) curves from pooled nested-CV out-of-fold probabilities. The dotted line on the PR panel is the positive-class prevalence (0.018). TabPFN dominates ranking (AP = 0.852, AUC = 0.990). Among classic models, CatBoost is next (AP = 0.697, AUC = 0.970), then LightGBM and XGBoost; random forest and logistic regression trail on PR-AUC even though all ROC-AUCs remain above 0.92. On a 1.8% prevalence outcome, PR-AUC is the more informative ranking metric.

**Source file:** [paper_figures/paper_fig1_pr_roc_curves.png](04_tabpfn_rating/paper_figures/paper_fig1_pr_roc_curves.png)

### Table 1. Pooled OOF ranking metrics

![Table 1](04_tabpfn_rating/paper_figures/paper_table1_ranking.png)

**Table 1.** Threshold-independent metrics as in the **exported PNG** (reconstructed earlier from Figure 1 legends). **Do not quote the TabPFN Brier cell.** The executed notebook print is TabPFN Brier = **0.0060** (best of the six); CatBoost 0.0090. The PNG / table image still show the stale 0.0360 value. Ranking order on PR-AUC / ROC-AUC is unchanged (TabPFN first).

| Rank | Model | PR-AUC | ROC-AUC | Brier |
| ---: | --- | ---: | ---: | ---: |
| 1 | TabPFN | 0.852 | 0.990 | 0.0360 |
| 2 | CatBoost | 0.697 | 0.970 | 0.0090 |
| 3 | LightGBM | 0.677 | 0.961 | 0.0096 |
| 4 | XGBoost | 0.665 | 0.949 | 0.0093 |
| 5 | Random Forest | 0.456 | 0.931 | 0.0147 |
| 6 | Logistic Regression | 0.342 | 0.925 | 0.0543 |

**Source files:** [paper_figures/paper_table1_ranking.png](04_tabpfn_rating/paper_figures/paper_table1_ranking.png), [paper_figures/paper_table1_ranking.csv](04_tabpfn_rating/paper_figures/paper_table1_ranking.csv)

---

## 3. Calibration

### Figure 2. Reliability curves (quantile bins)

![Figure 2](04_tabpfn_rating/paper_figures/paper_fig2_calibration_curves.png)

**Figure 2.** Calibration plots from nested-CV out-of-fold probabilities using quantile bins (appropriate because VLST is rare). The dashed diagonal is perfect calibration. Classic-model Brier scores in the panel titles match the notebook (CatBoost 0.0090, XGBoost 0.0093, LightGBM 0.0096, RF 0.0147, LR 0.0543). **The TabPFN panel title in this PNG is stale (0.0360).** The notebook print is **Brier = 0.0060**, the lowest of the six — calibration *supports* TabPFN on this run, it does not contradict ranking. Re-export the figure before publication.

**Source file:** [paper_figures/paper_fig2_calibration_curves.png](04_tabpfn_rating/paper_figures/paper_fig2_calibration_curves.png)

---

## 4. F1 operating point

Two cuts exist in the executed notebook. **Quote Table 2 (honest nested).** Figure 3 and Table 3 are the pooled F1 cut: the same concatenated OOF labels are used to *pick* and *score* the threshold, so precision, recall, F1, and F2 are **optimistically biased**. Counts sum to n = 5,185 with 92 events.

TabPFN recall: nested **0.7174** (quote) | pooled notebook 0.7935 | stale PNG **0.8370**. Do not quote 0.837.

### Table 2. Honest nested-CV operating point (quote this)

**Table 2.** Per-fold inner-CV F1 thresholds applied once to the unseen outer fold (notebook print, D4). TabPFN mean threshold 0.297 ± 0.053, precision 0.8354, recall **0.7174**, F1 0.7719, TN/FP/FN/TP = 5080/13/26/66. This is the protocol in the header. There is no exported PNG for this table; the Figure 3 / `paper_table2_f1_operating_point.png` images are the *pooled* cut below.

| Model | Threshold (mean ± SD) | Accuracy | Precision | Recall | Specificity | F1 | F2 | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabPFN | 0.297 ± 0.053 | 0.9925 | 0.8354 | **0.7174** | 0.9974 | 0.7719 | 0.7383 | 5080 | 13 | 26 | 66 |
| CatBoost | 0.317 ± 0.068 | 0.9867 | 0.6211 | 0.6413 | 0.9929 | 0.6310 | 0.6371 | 5057 | 36 | 33 | 59 |
| XGBoost | 0.237 ± 0.072 | 0.9882 | 0.6782 | 0.6413 | 0.9945 | 0.6592 | 0.6484 | 5065 | 28 | 33 | 59 |
| LightGBM | 0.099 ± 0.048 | 0.9863 | 0.6154 | 0.6087 | 0.9931 | 0.6120 | 0.6100 | 5058 | 35 | 36 | 56 |
| Random Forest | 0.114 ± 0.009 | 0.9834 | 0.5357 | 0.4891 | 0.9923 | 0.5114 | 0.4978 | 5054 | 39 | 47 | 45 |
| Logistic Regression | 0.953 ± 0.035 | 0.9757 | 0.3365 | 0.3804 | 0.9865 | 0.3571 | 0.3708 | 5024 | 69 | 57 | 35 |

**Source.** Executed `baseline_plus_tabpfn.ipynb` print (`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L1010–1026). Ranking PR-AUC / ROC-AUC in this print match Table 1’s notebook values and are threshold-independent.

### Figure 3. Confusion matrices at the pooled F1 threshold (optimistic; TabPFN panel stale)

![Figure 3](04_tabpfn_rating/paper_figures/paper_fig3_confusion_matrices.png)

**Figure 3.** 2×2 counts at the F1-maximising **pooled** OOF threshold (`t_F1` in each panel title). This is **not** Table 2. The cut is chosen on the same labels being scored, so recall is inflated relative to the nested point (TabPFN 0.7935 pooled-notebook vs **0.7174** nested). Classic-model panels match the notebook pooled cut. The **TabPFN panel is stale** (shown: t = 0.901, TP = 77, FN = 15, recall 0.837). Notebook pooled point: t = 0.173, TP = 73, FN = 19. Do not quote 0.837. LightGBM and XGBoost remain more conservative on false positives in this pooled figure (FP = 14 and 17). Logistic regression needs a very high threshold (0.970) and still misses 54 events. Random forest’s F1 point sits at a low probability (0.084), producing the most false positives (75). Accuracy is uniformly high because negatives dominate and is not a useful ranking criterion here.

**Source file:** [paper_figures/paper_fig3_confusion_matrices.png](04_tabpfn_rating/paper_figures/paper_fig3_confusion_matrices.png)

### Table 3. Optimistic pooled F1 metrics (do not quote instead of Table 2)

![Table 3](04_tabpfn_rating/paper_figures/paper_table2_f1_operating_point.png)

**Table 3.** Same pooled F1 cut as Figure 3. Markdown rows are the **notebook** pooled print (D4), not the stale PNG. TabPFN: t = 0.173, precision 0.7766, recall 0.7935, F1 0.7849, TN/FP/FN/TP = 5072/21/19/73. The PNG / CSV still show the superseded TabPFN row (t = 0.901, recall 0.8370, 5066/27/15/77). Classic-model rows match the notebook. Precision / recall / F1 / F2 here are **optimistically biased** versus Table 2.

| Model | t_F1 | Accuracy | Precision | Recall | Specificity | F1 | F2 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabPFN | 0.173 | 0.9923 | 0.7766 | 0.7935 | 0.9959 | 0.7849 | 0.7900 | 5072 | 21 | 19 | 73 |
| XGBoost | 0.381 | 0.9896 | 0.7639 | 0.5978 | 0.9967 | 0.6707 | 0.6250 | 5076 | 17 | 37 | 55 |
| CatBoost | 0.347 | 0.9882 | 0.6703 | 0.6630 | 0.9941 | 0.6667 | 0.6645 | 5063 | 30 | 31 | 61 |
| LightGBM | 0.228 | 0.9892 | 0.7812 | 0.5435 | 0.9973 | 0.6410 | 0.5787 | 5079 | 14 | 42 | 50 |
| Random Forest | 0.084 | 0.9786 | 0.4275 | 0.6087 | 0.9853 | 0.5022 | 0.5611 | 5018 | 75 | 36 | 56 |
| Logistic Regression | 0.970 | 0.9799 | 0.4318 | 0.4130 | 0.9902 | 0.4222 | 0.4167 | 5043 | 50 | 54 | 38 |

**Source files:** [paper_figures/paper_table2_f1_operating_point.png](04_tabpfn_rating/paper_figures/paper_table2_f1_operating_point.png) (stale TabPFN row), [paper_figures/paper_table2_f1_operating_point.csv](04_tabpfn_rating/paper_figures/paper_table2_f1_operating_point.csv), notebook L992–1008. Compact PNG [paper_table3_confusion_counts.png](04_tabpfn_rating/paper_figures/paper_table3_confusion_counts.png) is the same pooled cut and is likewise stale for TabPFN.

---

## 5. Supplementary: follow-up-time leakage

These numbers are **not** the nested-CV headline. They come from the two single-split (70/30, GridSearchCV) notebooks that diagnosed why `Time since stent implantation` cannot enter a classifier. Nothing was re-run; values are the stored test-set metrics.

**What the column is.** For VLST = 1 it is time from index PCI to angiographic thrombosis (min 380 days, Wang median 697). For VLST = 0 it is completed event-free follow-up (min 1,241, max 1,605 days; cohort median follow-up 1,502). That is binary-ified survival time, not a baseline covariate.

### Supplementary Table S-TSSI. Single-split metrics with vs without the column

![Table S-TSSI](04_tabpfn_rating/paper_figures/paper_table_s_tssi_leakage.png)

**Table S-TSSI.** Same stratified 70/30 split and tuning protocol. Logistic regression PR-AUC falls from 0.958 to 0.508 when the column is dropped; CatBoost from 0.977 to 0.658. Gaussian NB is unaffected (it never used the column). Nested-CV TabPFN in the main tables is the *without-TSSI* protocol.

**Source files:** [paper_figures/paper_table_s_tssi_leakage.png](04_tabpfn_rating/paper_figures/paper_table_s_tssi_leakage.png), [paper_figures/paper_table_s_tssi_leakage.csv](04_tabpfn_rating/paper_figures/paper_table_s_tssi_leakage.csv)

### Supplementary Figure S-TSSI. PR-AUC collapse

![Figure S-TSSI](04_tabpfn_rating/paper_figures/paper_fig_s_tssi_pr_auc.png)

**Figure S-TSSI.** PR-AUC on the 1,556-row hold-out. The dotted line is class prevalence (0.0177). The leaky column produces near-perfect ranking; removing it returns models to a rare-event scale.

**Source file:** [paper_figures/paper_fig_s_tssi_pr_auc.png](04_tabpfn_rating/paper_figures/paper_fig_s_tssi_pr_auc.png)

Notebooks: `code/modeling/rating/baseline_tssi_leakage.ipynb`, `code/modeling/rating/baseline_without_tssi.ipynb`. Table rebuilt by `code/modeling/rating/rebuild_tssi_leakage_table.py`.

---

## 6. Supplementary: Wang 2020 integer score

These numbers are **not** a nested-CV fit. They come from `code/modeling/rating/wang_vlst_score.ipynb`, which scores Wang 2020 Table 2 **integer points** on all 5,185 rows with the published weights frozen. The same five outer folds as Part 4 (`StratifiedKFold(5, shuffle=True, random_state=42)`) are used only to evaluate that frozen score.

**Headline.** Full-cohort ROC-AUC **0.8013** (Wang published derivation c-statistic 0.80, 95% CI 0.75–0.85) and PR-AUC **0.1032**. Fold-mean ROC-AUC **0.8005 ± 0.0607**, PR-AUC **0.1134 ± 0.0518**. Nested-CV TabPFN (executed Part 4 notebook) is ROC-AUC **0.9883** / PR-AUC **0.8534**. This is the missing clinical comparator: TabPFN vs the score already published on these patients, not only vs untuned sklearn defaults. It is **not** external validation (Wang’s c = 0.82 was Shantou; that file is still absent).

**Encoding (do not photocopy Wang Table 1).** The SES point is on **`PES`** (Wang Table 1 SES rates 82.61% / 68.76% match `PES`, not the 106-level `Stent type-SES` strings). The 4 post-dilation points go to **`No postdilation` = 1** (78/92 events). Using Wang Table 1’s 14 VLST “No post-dilation” cases (`1.1:1Post dilation` = 1) as the 4-point group yields ROC-AUC **0.5084**. Table 2’s β = 1.93 / 4 points vs printed HR 0.145 is the same polarity clash; this notebook follows the points plus the CSV risk direction.

**Risk bins.** Low ≤7: n = 3,135 (60.5%), rate 0.51%. Intermediate 8–9: n = 1,577 (30.4%), rate 2.22%. High ≥10: n = 473 (9.1%), rate 8.67%. Wang’s published n’s 3,135 / 1,837 / 473 sum to 5,445 ≠ 5,185; low and high n match this file exactly; the intermediate remainder here is 1,577 with the same 2.2% rate.

The Cox linear predictor, Dangas decision-curve analysis, and Shantou scoring are **not** in this notebook.

### Supplementary Table S-Wang-bins. Observed VLST rate by published risk category

![Table S-Wang-bins](04_tabpfn_rating/paper_figures/paper_table_s_wang_score_bins.png)

**Table S-Wang-bins.** Frozen integer score, cut at Wang’s published thresholds (≤7 / 8–9 / ≥10). Observed rates match Wang’s 0.5% / 2.2% / 8.7%. The intermediate *count* does not: Wang printed n = 1,837 for that bin.

**Source files:** [paper_figures/paper_table_s_wang_score_bins.png](04_tabpfn_rating/paper_figures/paper_table_s_wang_score_bins.png), [paper_figures/paper_table_s_wang_score_bins.csv](04_tabpfn_rating/paper_figures/paper_table_s_wang_score_bins.csv)

| Risk category | n | % of cohort | VLST events | Observed rate | Wang published n | Wang published rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low (≤7) | 3135 | 60.5 | 16 | 0.0051 | 3135 | 0.005 |
| intermediate (8–9) | 1577 | 30.4 | 35 | 0.0222 | 1837 | 0.022 |
| high (≥10) | 473 | 9.1 | 41 | 0.0867 | 473 | 0.087 |

### Supplementary Table S-Wang. Frozen score vs nested-CV models

![Table S-Wang](04_tabpfn_rating/paper_figures/paper_table_s_wang_vs_ml.png)

**Table S-Wang.** Wang integer score: full-cohort ranking plus the same five outer folds, score not refit. TabPFN / CatBoost / logistic regression: nested 5×4 CV out-of-fold from the executed Part 4 notebook (D4). PR-AUC is the informative metric at 1.77% prevalence.

**Source files:** [paper_figures/paper_table_s_wang_vs_ml.png](04_tabpfn_rating/paper_figures/paper_table_s_wang_vs_ml.png), [paper_figures/paper_table_s_wang_vs_ml.csv](04_tabpfn_rating/paper_figures/paper_table_s_wang_vs_ml.csv)

| Model | ROC-AUC | PR-AUC | ROC fold mean ± SD | PR fold mean ± SD | Protocol |
| --- | ---: | ---: | --- | --- | --- |
| Wang 2020 integer score (frozen) | 0.8013 | 0.1032 | 0.8005 ± 0.0607 | 0.1134 ± 0.0518 | Published points; folds evaluate only |
| TabPFN client thinking-high | 0.9883 | 0.8534 | 0.9884 ± 0.0061 | 0.8503 ± 0.0746 | Part 4 nested 5×4 CV OOF |
| CatBoost (untuned nested CV) | 0.9704 | 0.6967 | 0.9712 ± 0.0117 | 0.7007 ± 0.0684 | Part 4 nested 5×4 CV OOF |
| Logistic regression (untuned nested CV) | 0.9246 | 0.3418 | 0.9256 ± 0.0225 | 0.3568 ± 0.1153 | Part 4 nested 5×4 CV OOF |

Fold-level frozen-score metrics: [paper_table_s_wang_score_folds.csv](04_tabpfn_rating/paper_figures/paper_table_s_wang_score_folds.csv).

### Supplementary Figure S-Wang. Observed VLST rate by integer score

![Figure S-Wang](04_tabpfn_rating/paper_figures/paper_fig_s_wang_score_rate.png)

**Figure S-Wang.** Observed VLST rate at each integer total. Bar labels are cell n (shown when n ≥ 20). The dashed line is cohort prevalence (0.0177). The score is a ranker, not a calibrated probability.

**Source file:** [paper_figures/paper_fig_s_wang_score_rate.png](04_tabpfn_rating/paper_figures/paper_fig_s_wang_score_rate.png)

Notebook: `code/modeling/rating/wang_vlst_score.ipynb`.

---

## 7. File index

| ID | Type | File |
| --- | --- | --- |
| Table 0 | Table | [paper_table0_models.png](04_tabpfn_rating/paper_figures/paper_table0_models.png) |
| Fig 1 | Figure | [paper_fig1_pr_roc_curves.png](04_tabpfn_rating/paper_figures/paper_fig1_pr_roc_curves.png) |
| Table 1 | Table | [paper_table1_ranking.png](04_tabpfn_rating/paper_figures/paper_table1_ranking.png) |
| Fig 2 | Figure | [paper_fig2_calibration_curves.png](04_tabpfn_rating/paper_figures/paper_fig2_calibration_curves.png) |
| Fig 3 | Figure | [paper_fig3_confusion_matrices.png](04_tabpfn_rating/paper_figures/paper_fig3_confusion_matrices.png) |
| Table 2 | Table | Honest nested operating point (notebook print; no PNG) |
| Table 3 | Table | [paper_table2_f1_operating_point.png](04_tabpfn_rating/paper_figures/paper_table2_f1_operating_point.png) (pooled; TabPFN row stale) |
| Table S-TSSI | Table | [paper_table_s_tssi_leakage.png](04_tabpfn_rating/paper_figures/paper_table_s_tssi_leakage.png) |
| Fig S-TSSI | Figure | [paper_fig_s_tssi_pr_auc.png](04_tabpfn_rating/paper_figures/paper_fig_s_tssi_pr_auc.png) |
| Table S-Wang-bins | Table | [paper_table_s_wang_score_bins.png](04_tabpfn_rating/paper_figures/paper_table_s_wang_score_bins.png) |
| Table S-Wang | Table | [paper_table_s_wang_vs_ml.png](04_tabpfn_rating/paper_figures/paper_table_s_wang_vs_ml.png) |
| Fig S-Wang | Figure | [paper_fig_s_wang_score_rate.png](04_tabpfn_rating/paper_figures/paper_fig_s_wang_score_rate.png) |

---

*Figures 1–3 are the executed outputs stored in `baseline_plus_tabpfn.ipynb` (Kaggle nested-CV run). Table 1 Brier / Figure 2–3 TabPFN panels remain stale versus the notebook. Table 2 is the honest nested operating point from the notebook print; Table 3 markdown uses the notebook pooled F1 cut. Fold-wise mean ± SD and nested inner-threshold operating points **were** printed in the notebook; the CSVs under `/kaggle/working/` were not committed.*

---

# Part 5. TabPFN interpretability

### TabPFN interpretability — paper figures and tables

This document gathers publication-oriented figures and tables from the TabPFN interpretability notebook `tabpfn_interpretability.ipynb`.

**Cohort / protocol.** Raw VLST.csv, n = 5,185, 81 features after dropping identifiers (`NO.`, `Name`) and `Time since stent implantation` (time-at-risk / follow-up, not a baseline covariate). Target = `Stent thrombosis`. EDA found **no missing values** — there is no missingness to “keep.” Text columns are coded as integer categoricals (no scaling / one-hot). That is the TabPFN-native representation; `Stent type-SES` is treated as a numeric code, so a PDP sweep across brand integers is not a meaningful nominal contrast. Feature ranking, PDP, and SHAP are **interpretation / attribution** on the full pool — not prediction, not external validation, and not a locked-in feature mask for Part 4.

**Methods note — selection vs explanation.** Mutual information, stability (repeated forward SFS), and SHAP all use the **full cohort** (`X_all`, `y_all`): 92 events and 5,093 controls. There is no `SHAP_N_EXPLAIN` cap and no case-only slice. k-SII / SHAP-IQ force and network plots remain **one illustrative row** (row 0 of that same cohort), not a VLST = 1 pick.

**Backends.** Mutual information, stability selection, and PDP use **local** `tabpfn` (0 client thinking fits). SHAP and SHAP-IQ try tabpfn-client with thinking (`effort=high`, `metric=average_precision`) and fall back to local TabPFN + KV cache. The shapiq `imputer="baseline"` is **not** a missing-value fill: it replaces *hidden* features with a baseline value while attributing. **[STALE]** The PNGs and Table 4 numbers below are still the old 15-case run until this notebook is re-executed.

**Asset root:** [05_tabpfn_interpretability/paper_figures/](05_tabpfn_interpretability/paper_figures/)

---

## Contents

1. [Methods (Table 0)](#1-methods)
2. [Univariate and stability screens (Tables 1–2)](#2-univariate-and-stability-screens)
3. [Partial dependence (Figures 1–2, Table 3)](#3-partial-dependence)
4. [SHAP attributions (Figures 3–7, Table 4)](#4-shap-attributions)
5. [Pairwise interactions — k-SII (Figures 8–9)](#5-pairwise-interactions--k-sii)
6. [SHAP-IQ native plots (Figures 10–12)](#6-shap-iq-native-plots)
7. [Consensus ranking (Figure 13, Table 5)](#7-consensus-ranking)
8. [File index](#8-file-index)

---

## 1. Methods

### Table 0. Interpretability methods

![Table 0](05_tabpfn_interpretability/paper_figures/paper_table0_methods.png)

**Table 0.** Five signals plus a Borda-style consensus. No single method is trusted alone. Stability frequency is the reliability signal (how often forward SFS keeps a feature across 10 resamples). MI, SFS, and SHAP use the **full cohort**. Pairwise k-SII is a one-row interaction view (row 0), not a global interaction ranking.

| Method | Question | Backend | Notebook setting |
| --- | --- | --- | --- |
| mutual_info_classif | Univariate association | sklearn | 0 TabPFN calls; median fill is inert (no NaNs) |
| Stability (repeated SFS) | Selection frequency | local TabPFN | 10 resamples × top-10 forward SFS, AP scoring |
| PDP | Average predicted risk | local TabPFN | Continuous grid + binary 0 vs 1 bars |
| SHAP (shapiq SV) | Local attributions | local TabPFN (client fallback) | Full cohort (same as FFS/MI); budget=256 |
| k-SII / SHAP-IQ | Pairwise interactions | local TabPFN (client fallback) | One illustrative row (row 0 of the full cohort); budget=256 |
| Consensus (Borda) | Mean of normalized ranks | aggregate | MI + stability frequency + mean(\|SHAP\|) |

**Source files:** [paper_figures/paper_table0_methods.png](05_tabpfn_interpretability/paper_figures/paper_table0_methods.png), [paper_figures/paper_table0_methods.csv](05_tabpfn_interpretability/paper_figures/paper_table0_methods.csv)

---

## 2. Univariate and stability screens

### Table 1. Top 15 by mutual information

![Table 1](05_tabpfn_interpretability/paper_figures/paper_table1_mutual_info.png)

**Table 1.** `mutual_info_classif` ranking of the 81-column raw matrix on the **full cohort**. The code applies a column-median fill before MI; the CSV has no missing values, so that fill does nothing. Calcium index (`CaI`), `WBC`, and `LV` lead. `Stent type-SES` and `eGFR` follow. Mutual-information values for `Fast-Glu` and `ZES` were not stored in the consensus table; those two names still appear in the notebook’s printed top-15 list. This is a marginal association screen, not a model attribution.

| Rank | Feature | Mutual information |
| ---: | --- | ---: |
| 1 | CaI | 0.019224 |
| 2 | WBC | 0.018360 |
| 3 | LV | 0.015178 |
| 4 | Stent type-SES | 0.011357 |
| 5 | eGFR | 0.009931 |
| 6 | 1.1:1Post dilation | 0.007932 |
| 7 | LDL | 0.007693 |
| 8 | No postdilation | 0.007535 |
| 9 | HbA1c | 0.007439 |
| 10 | HGB | 0.004606 |
| 11 | Fast-Glu | — |
| 12 | TCL | 0.004257 |
| 13 | Fiberinogen | 0.004040 |
| 14 | ZES | — |
| 15 | Visual thrombus | 0.003707 |

**Source files:** [paper_figures/paper_table1_mutual_info.png](05_tabpfn_interpretability/paper_figures/paper_table1_mutual_info.png), [paper_figures/paper_table1_mutual_info.csv](05_tabpfn_interpretability/paper_figures/paper_table1_mutual_info.csv)

### Table 2. Stability selection frequency

![Table 2](05_tabpfn_interpretability/paper_figures/paper_table2_stability.png)

**Table 2.** Forward sequential feature selection (keep 10 of 81, 5-fold CV, average precision) repeated over 10 shuffled seeds. Features selected in every run: `LV`, `Stent type-SES`, `eGFR`. Selected in 9/10: `Age`, `Cre`, `WBC`. `No postdilation` (7/10) and `STEMI` (6/10) are moderately stable. A long tail of names appears only once or twice and should not be treated as robust TabPFN features. Wall time for this block in the stored run was ~8.6 h (local TabPFN, 0 client API calls).

| Feature | Selected | Frequency |
| --- | --- | ---: |
| LV | 10/10 | 1.0 |
| Stent type-SES | 10/10 | 1.0 |
| eGFR | 10/10 | 1.0 |
| Age | 9/10 | 0.9 |
| Cre | 9/10 | 0.9 |
| WBC | 9/10 | 0.9 |
| No postdilation | 7/10 | 0.7 |
| STEMI | 6/10 | 0.6 |
| HbA1c | 5/10 | 0.5 |
| LVEF | 5/10 | 0.5 |
| No.of stents per lesion | 3/10 | 0.3 |
| Staged PCI | 3/10 | 0.3 |
| 1.1:1Post dilation | 2/10 | 0.2 |
| CKD5 | 2/10 | 0.2 |
| CKD60 | 2/10 | 0.2 |
| EVS | 2/10 | 0.2 |
| Current drinking | 1/10 | 0.1 |
| History of peripheral vascualr disease | 1/10 | 0.1 |
| Initial diagnosis-AMI | 1/10 | 0.1 |
| PES | 1/10 | 0.1 |
| Single-vessel disease | 1/10 | 0.1 |
| Visual thrombus | 1/10 | 0.1 |

**Source files:** [paper_figures/paper_table2_stability.png](05_tabpfn_interpretability/paper_figures/paper_table2_stability.png), [paper_figures/paper_table2_stability.csv](05_tabpfn_interpretability/paper_figures/paper_table2_stability.csv)

---

## 3. Partial dependence

PDP candidates were taken from the stability / MI screens. Continuous PDP uses grid resolution 30. Binary PDP forces each flag to 0 vs 1 and reports the change in average predicted P[Stent thrombosis].

**Methods note — these are not absolute risks.** Every TabPFN fit in this notebook uses `balance_probabilities=True`, which rescales outputs toward a uniform class prior. True prevalence is 0.0177. Table 3 baselines around 0.13–0.26 and Figure 1’s LV curve “toward ~0.6” are **balanced-prior model output**, not predicted event probabilities a clinician can read as 24% or 60% risk.

### Figure 1. Continuous partial dependence

![Figure 1](05_tabpfn_interpretability/paper_figures/paper_fig1_pdp_continuous.png)

**Figure 1.** Average TabPFN **balanced-prior** output while sweeping one feature (rug = empirical distribution). `LV`: output stays low until the mid-40s then rises toward ~0.6 on that scale, **not** a 60% absolute risk. `eGFR`: high output at low filtration. `Stent type-SES` is integer-coded here; a sweep across brand codes is **not** a nominal brand contrast (EDA uses 9 collapsed levels; Part 4 one-hots 106 strings). `Age` is essentially flat (~0.14). The LV × SES contour is dominated by vertical (LV) bands.

**Source file:** [paper_figures/paper_fig1_pdp_continuous.png](05_tabpfn_interpretability/paper_figures/paper_fig1_pdp_continuous.png)

### Figure 2. Binary partial dependence

![Figure 2](05_tabpfn_interpretability/paper_figures/paper_fig2_pdp_binary.png)

**Figure 2.** Average balanced-prior P[Stent thrombosis] when each binary feature is forced to absent (0, blue) vs present (1, orange). The largest shift is `1.1:1Post dilation` (ΔP = −0.086): the flag is associated with **lower model output**, not a proven treatment effect. `No postdilation` also lowers output (ΔP = −0.036) from a higher baseline. `STEMI`, `Staged PCI`, `CKD60`, and `EVS` have small average effects (|ΔP| ≤ 0.013).

**Source file:** [paper_figures/paper_fig2_pdp_binary.png](05_tabpfn_interpretability/paper_figures/paper_fig2_pdp_binary.png)

### Table 3. Binary PDP numeric values

![Table 3](05_tabpfn_interpretability/paper_figures/paper_table3_pdp_binary.png)

**Table 3.** Printed PDP probabilities from the notebook. ΔP = P(y=1 | feature=1) − P(y=1 | feature=0). These are **balanced-prior** model-average effects, not absolute risks and not causal estimates.

| Feature | P(y=1 \| 0) | P(y=1 \| 1) | ΔP |
| --- | ---: | ---: | ---: |
| No postdilation | 0.2430 | 0.2067 | −0.0363 |
| STEMI | 0.1401 | 0.1271 | −0.0130 |
| Staged PCI | 0.1334 | 0.1303 | −0.0031 |
| 1.1:1Post dilation | 0.2643 | 0.1781 | −0.0862 |
| CKD60 | 0.1359 | 0.1234 | −0.0124 |
| EVS | 0.1328 | 0.1354 | +0.0026 |

**Source files:** [paper_figures/paper_table3_pdp_binary.png](05_tabpfn_interpretability/paper_figures/paper_table3_pdp_binary.png), [paper_figures/paper_table3_pdp_binary.csv](05_tabpfn_interpretability/paper_figures/paper_table3_pdp_binary.csv)

---

## 4. SHAP attributions

**Code now:** SHAP fits and explains the **full cohort** (same `X_all`, `y_all` as FFS/MI). **[STALE]** The PNGs and numbers in this section are still the previous 15-case run. Do not read them as the current protocol. Re-run `tabpfn_interpretability.ipynb` [3/5] before using these figures.

### Figure 3. SHAP summary

![Figure 3](05_tabpfn_interpretability/paper_figures/paper_fig3_shap_summary.png)

**Figure 3.** **[STALE PNG]** Stored beeswarm is from the deleted 15-case slice. After re-run this is a full-cohort beeswarm (colour = feature value).

**Source file:** [paper_figures/paper_fig3_shap_summary.png](05_tabpfn_interpretability/paper_figures/paper_fig3_shap_summary.png)

### Figure 4. SHAP scatter for Age

![Figure 4](05_tabpfn_interpretability/paper_figures/paper_fig4_shap_scatter_age.png)

**Figure 4.** **[STALE PNG]** Age versus SHAP from the old 15-case slice. Re-run before citing.

**Source file:** [paper_figures/paper_fig4_shap_scatter_age.png](05_tabpfn_interpretability/paper_figures/paper_fig4_shap_scatter_age.png)

### Figure 5. Mean absolute SHAP (global bar)

![Figure 5](05_tabpfn_interpretability/paper_figures/paper_fig5_shap_bar.png)

**Figure 5.** **[STALE PNG / numbers]** Mean(|SHAP|) from the old 15-case slice (`LV` 1.24, `WBC` 1.16, …). After re-run this is mean(|SHAP|) on the full cohort.

**Source file:** [paper_figures/paper_fig5_shap_bar.png](05_tabpfn_interpretability/paper_figures/paper_fig5_shap_bar.png)

### Figure 6. Compact SHAP beeswarm

![Figure 6](05_tabpfn_interpretability/paper_figures/paper_fig6_shap_beeswarm.png)

**Figure 6.** **[STALE PNG]** Compact beeswarm from the old 15-case slice.

**Source file:** [paper_figures/paper_fig6_shap_beeswarm.png](05_tabpfn_interpretability/paper_figures/paper_fig6_shap_beeswarm.png)

### Figure 7. One-row SHAP waterfall

![Figure 7](05_tabpfn_interpretability/paper_figures/paper_fig7_shap_waterfall.png)

**Figure 7.** **[STALE PNG]** Old waterfall for one VLST case. Code now uses row 0 of the full cohort (not a case-enriched pick). Local explanation for one patient, not a global ranking.

**Source file:** [paper_figures/paper_fig7_shap_waterfall.png](05_tabpfn_interpretability/paper_figures/paper_fig7_shap_waterfall.png)

### Table 4. Mean(|SHAP|) ranking

![Table 4](05_tabpfn_interpretability/paper_figures/paper_table4_shap_mean_abs.png)

**Table 4.** **[STALE]** Mean absolute SHAP from the old 15-case slice. After re-run this is the full-cohort ranking used in Table 5.

| Rank | Feature | mean(\|SHAP\|) |
| ---: | --- | ---: |
| 1 | LV | 1.2368 |
| 2 | WBC | 1.1648 |
| 3 | LDL | 0.6408 |
| 4 | eGFR | 0.4713 |
| 5 | 1.1:1Post dilation | 0.2392 |
| 6 | Stent type-SES | 0.2296 |
| 7 | No postdilation | 0.1779 |
| 8 | CaI | 0.1779 |
| 9 | HbA1c | 0.1602 |
| 10 | Cre | 0.1539 |
| 11 | Fiberinogen | 0.0942 |
| 12 | HGB | 0.0760 |
| 13 | TCL | 0.0500 |
| 14 | No.of stents per lesion | 0.0424 |
| 15 | Visual thrombus | 0.0384 |

**Source files:** [paper_figures/paper_table4_shap_mean_abs.png](05_tabpfn_interpretability/paper_figures/paper_table4_shap_mean_abs.png), [paper_figures/paper_table4_shap_mean_abs.csv](05_tabpfn_interpretability/paper_figures/paper_table4_shap_mean_abs.csv)

---

## 5. Pairwise interactions — k-SII

k-SII plots use **one illustrative row** (row 0 of the full cohort; budget = 256). Node size is the main effect; edge width is the pairwise interaction. They illustrate how TabPFN combines features for that row; they are not a cohort interaction screen. **[STALE]** Stored figures 8–12 are still the old VLST-case row.

### Figure 8. k-SII network (SHAP section)

![Figure 8](05_tabpfn_interpretability/paper_figures/paper_fig8_ksii_network.png)

**Figure 8.** Circular k-SII network for the top 20 features by |Shapley value| on **one VLST case**. Large red nodes (`LV`, `WBC`) are strong positive main effects; `1.1:1Post dilation` is a large **negative** (blue) main effect on this row — not a cohort-level benefit. Thick edges among `LV`, `WBC`, `eGFR`, `LDL`, and post-dilation are the dominant pairwise terms **for that patient**.

**Source file:** [paper_figures/paper_fig8_ksii_network.png](05_tabpfn_interpretability/paper_figures/paper_fig8_ksii_network.png)

### Figure 9. k-SII UpSet plot (SHAP section)

![Figure 9](05_tabpfn_interpretability/paper_figures/paper_fig9_ksii_upset.png)

**Figure 9.** UpSet-style listing of the largest main effects and pairwise k-SII values for the same row. The leftmost bar is the large negative base / intercept term. The largest positive main effects are `LV` then `WBC`; the largest negative main effect among named features is `1.1:1Post dilation`. Pairwise terms involving `LV`, `WBC`, `eGFR`, and `LDL` fill most of the remaining top-20 slots.

**Source file:** [paper_figures/paper_fig9_ksii_upset.png](05_tabpfn_interpretability/paper_figures/paper_fig9_ksii_upset.png)

---

## 6. SHAP-IQ native plots

Section [4/5] of the notebook recomputes imputation-based Shapley values and k-SII with shapiq’s native plotting API, again after the client backend failed. Figures 10–12 are a second view of the **same one-row explanation**, not an independent replication on new rows.

### Figure 10. SHAP-IQ force plot (one row)

![Figure 10](05_tabpfn_interpretability/paper_figures/paper_fig10_shapiq_force.png)

**Figure 10.** Force / additive layout for the same explained row (f(x) ≈ −3.09). Red segments (`LV`, `WBC`, `LDL`) raise the output from the base value; blue segments (post-dilation, eGFR, SES) lower it. This is the compact counterpart of the waterfall in Figure 7.

**Source file:** [paper_figures/paper_fig10_shapiq_force.png](05_tabpfn_interpretability/paper_figures/paper_fig10_shapiq_force.png)

### Figure 11. SHAP-IQ k-SII network

![Figure 11](05_tabpfn_interpretability/paper_figures/paper_fig11_shapiq_network.png)

**Figure 11.** Native shapiq network for the same top-20 features by |SV|. Layout and the `LV` / `WBC` / post-dilation / `LDL` / `eGFR` core match Figure 8.

**Source file:** [paper_figures/paper_fig11_shapiq_network.png](05_tabpfn_interpretability/paper_figures/paper_fig11_shapiq_network.png)

### Figure 12. SHAP-IQ k-SII UpSet plot

![Figure 12](05_tabpfn_interpretability/paper_figures/paper_fig12_shapiq_upset.png)

**Figure 12.** Native shapiq UpSet plot of top-20 main effects and pairwise interactions for the same row. Read it as a restyle of Figure 9, not as a new sample of patients.

**Source file:** [paper_figures/paper_fig12_shapiq_upset.png](05_tabpfn_interpretability/paper_figures/paper_fig12_shapiq_upset.png)

---

## 7. Consensus ranking

Ranking uses a **Borda-style mean of normalized ranks** across mutual information, stability frequency, and mean(|SHAP|), with `n_methods` (out of 3) as a consensus count. The notebook reports the top 15 as *associations* with stent thrombosis under TabPFN — exploratory, not causal, on a ~2% prevalence cohort.

### Figure 13. Top 15 by consensus

![Figure 13](05_tabpfn_interpretability/paper_figures/paper_fig13_consensus_ranking.png)

**Figure 13.** Aggregated importance (1 = strongest mean normalized rank). Annotations give how many of the three signals placed the feature in their top set. `LV`, `WBC`, `eGFR`, `Stent type-SES`, `No postdilation`, and `HbA1c` are 3/3. `1.1:1Post dilation` is high on MI and SHAP but only 2/10 in stability. `No.of stents per lesion` ranks 14th on the Borda score despite 0/3 top-set membership — a reminder that a middling rank on all three lists can still enter the top 15.

**Source file:** [paper_figures/paper_fig13_consensus_ranking.png](05_tabpfn_interpretability/paper_figures/paper_fig13_consensus_ranking.png)

### Table 5. Consensus feature report

![Table 5](05_tabpfn_interpretability/paper_figures/paper_table5_consensus.png)

**Table 5.** The notebook’s `interpretability_feature_importance_report` top 15. `importance_score` is the Borda aggregate. `n_methods` counts how many of {MI top, stability, SHAP top} contributed. The six names with n_methods = 3 and high stability (`LV`, `WBC`, `eGFR`, `Stent type-SES`, plus `No postdilation` and `HbA1c`) are the most honest TabPFN associations in this run. `Cre` is stable (0.9) and in the SHAP top set but not the MI top set. `CaI` and `LDL` are strong on MI/SHAP but never selected by repeated SFS (frequency 0.0). **`Cre` and `No.of stents per lesion` show `mutual_info = 0.000000` because they sit outside the stored MI top-15 and were filled with zeros — those are not measured zeros.**

| Rank | Feature | Score | n methods | Stability | mean(\|SHAP\|) | MI | In MI top | In SHAP top |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | LV | 0.9875 | 3/3 | 1.0 | 1.2368 | 0.015178 | yes | yes |
| 2 | WBC | 0.9750 | 3/3 | 0.9 | 1.1648 | 0.018360 | yes | yes |
| 3 | eGFR | 0.9667 | 3/3 | 1.0 | 0.4713 | 0.009931 | yes | yes |
| 4 | Stent type-SES | 0.9625 | 3/3 | 1.0 | 0.2296 | 0.011357 | yes | yes |
| 5 | No postdilation | 0.9208 | 3/3 | 0.7 | 0.1779 | 0.007535 | yes | yes |
| 6 | 1.1:1Post dilation | 0.9062 | 2/3 | 0.2 | 0.2392 | 0.007932 | yes | yes |
| 7 | HbA1c | 0.8979 | 3/3 | 0.5 | 0.1602 | 0.007439 | yes | yes |
| 8 | Visual thrombus | 0.7854 | 1/3 | 0.1 | 0.0384 | 0.003707 | yes | no |
| 9 | CaI | 0.7583 | 2/3 | 0.0 | 0.1779 | 0.019224 | yes | yes |
| 10 | LDL | 0.7542 | 2/3 | 0.0 | 0.6408 | 0.007693 | yes | yes |
| 11 | Cre | 0.7479 | 2/3 | 0.9 | 0.1539 | 0.000000 | no | yes |
| 12 | HGB | 0.6958 | 2/3 | 0.0 | 0.0760 | 0.004606 | yes | yes |
| 13 | Fiberinogen | 0.6875 | 2/3 | 0.0 | 0.0942 | 0.004040 | yes | yes |
| 14 | No.of stents per lesion | 0.6833 | 0/3 | 0.3 | 0.0424 | 0.000000 | no | no |
| 15 | TCL | 0.6792 | 1/3 | 0.0 | 0.0500 | 0.004257 | yes | no |

**Source files:** [paper_figures/paper_table5_consensus.png](05_tabpfn_interpretability/paper_figures/paper_table5_consensus.png), [paper_figures/paper_table5_consensus.csv](05_tabpfn_interpretability/paper_figures/paper_table5_consensus.csv)

---

## 8. File index

| ID | Type | File |
| --- | --- | --- |
| Table 0 | Table | [paper_table0_methods.png](05_tabpfn_interpretability/paper_figures/paper_table0_methods.png) |
| Table 1 | Table | [paper_table1_mutual_info.png](05_tabpfn_interpretability/paper_figures/paper_table1_mutual_info.png) |
| Table 2 | Table | [paper_table2_stability.png](05_tabpfn_interpretability/paper_figures/paper_table2_stability.png) |
| Fig 1 | Figure | [paper_fig1_pdp_continuous.png](05_tabpfn_interpretability/paper_figures/paper_fig1_pdp_continuous.png) |
| Fig 2 | Figure | [paper_fig2_pdp_binary.png](05_tabpfn_interpretability/paper_figures/paper_fig2_pdp_binary.png) |
| Table 3 | Table | [paper_table3_pdp_binary.png](05_tabpfn_interpretability/paper_figures/paper_table3_pdp_binary.png) |
| Fig 3 | Figure | [paper_fig3_shap_summary.png](05_tabpfn_interpretability/paper_figures/paper_fig3_shap_summary.png) |
| Fig 4 | Figure | [paper_fig4_shap_scatter_age.png](05_tabpfn_interpretability/paper_figures/paper_fig4_shap_scatter_age.png) |
| Fig 5 | Figure | [paper_fig5_shap_bar.png](05_tabpfn_interpretability/paper_figures/paper_fig5_shap_bar.png) |
| Fig 6 | Figure | [paper_fig6_shap_beeswarm.png](05_tabpfn_interpretability/paper_figures/paper_fig6_shap_beeswarm.png) |
| Fig 7 | Figure | [paper_fig7_shap_waterfall.png](05_tabpfn_interpretability/paper_figures/paper_fig7_shap_waterfall.png) |
| Table 4 | Table | [paper_table4_shap_mean_abs.png](05_tabpfn_interpretability/paper_figures/paper_table4_shap_mean_abs.png) |
| Fig 8 | Figure | [paper_fig8_ksii_network.png](05_tabpfn_interpretability/paper_figures/paper_fig8_ksii_network.png) |
| Fig 9 | Figure | [paper_fig9_ksii_upset.png](05_tabpfn_interpretability/paper_figures/paper_fig9_ksii_upset.png) |
| Fig 10 | Figure | [paper_fig10_shapiq_force.png](05_tabpfn_interpretability/paper_figures/paper_fig10_shapiq_force.png) |
| Fig 11 | Figure | [paper_fig11_shapiq_network.png](05_tabpfn_interpretability/paper_figures/paper_fig11_shapiq_network.png) |
| Fig 12 | Figure | [paper_fig12_shapiq_upset.png](05_tabpfn_interpretability/paper_figures/paper_fig12_shapiq_upset.png) |
| Fig 13 | Figure | [paper_fig13_consensus_ranking.png](05_tabpfn_interpretability/paper_figures/paper_fig13_consensus_ranking.png) |
| Table 5 | Table | [paper_table5_consensus.png](05_tabpfn_interpretability/paper_figures/paper_table5_consensus.png) |

---

*Figures are the executed PNG outputs stored in `tabpfn_interpretability.ipynb`. Tables are reconstructed from those plots and the notebook’s printed CSVs. SHAP / SHAP-IQ used local TabPFN after the client thinking backend failed. MI and stability use the full cohort; SHAP uses 15 cases only. Rankings are for interpretation only and should not be reused as a leakage-free feature mask.*
