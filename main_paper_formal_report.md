# Very late stent thrombosis after ACS-PCI: association, nested cross-validation, and TabPFN attribution on the Wang 2020 derivation cohort

## Authors

[Authors, affiliations, and corresponding author to be supplied.]

## Abstract

**Background.** Very late stent thrombosis (VLST) is Academic Research Consortium 2007 definite stent thrombosis more than one year after percutaneous coronary intervention (PCI). In the file analysed here, 92 of 5,185 patients have the event (prevalence 0.0177). The file is Wang and colleagues’ 2020 derivation cohort. Wang published an 8-variable Cox integer score on these rows (published derivation c-statistic 0.80 [0.75, 0.85]). Wang also reported a Shantou c-statistic of 0.82 for that score; the Shantou file is not in this repository, and the present models were not tested on it. Binary classifiers on this file are readily inflated if follow-up time remains in the matrix. Tabular prior-fitted networks (TabPFN) perform in-context learning on mixed-type tables without a per-dataset hyperparameter grid. This study compares them with class-weighted library-default tree and linear models under one shared nested cross-validation protocol after a pre-specified anti-leakage cleaning.

**Methods.** Retrospective analysis of the derivation file (The First Hospital of Jilin University; acute coronary syndrome; age at least 18 years; PCI from 1 January 2014 to 1 June 2015; n = 5,185). The outcome is the stored binary column `Stent thrombosis`. The prediction matrix drops identifiers, `Time since stent implantation`, and `WBC`; quantizes `Cre`, `CaI`, `Fiberinogen`, and `Fast-Glu` before any split; fits the stent-brand codebook on each training fold only; and does not use SMOTE. Prediction uses one stratified nested cross-validation (5 outer × 4 inner folds, seed 42). The inner loop selects an F1 threshold only. Logistic regression, random forest, XGBoost, LightGBM, and CatBoost use library defaults plus class weighting. Grid-search winners from a separate 70/30 leakage contrast are not imported. Four TabPFN arms use local v3 and v3.5 (`tabpfn==9.0.0`) and hosted thinking-high v3 and v3.5 (`tabpfn-client==0.6.0`; effort high; metric average precision). The primary ranking metric is precision–recall area under the curve (PR-AUC) at prevalence 0.0177. Intervals are percentile intervals from a stratified bootstrap of stored pooled out-of-fold scores (`n_boot=2000`, seed 42), without refitting. Attribution uses eight-seed forward selection, mutual information, and partial dependence on the training split, and held-out Shapley values with a one-row interaction display, under the same cleaning.

**Results.** Among nine nested arms, TabPFN thinking v3.5 had PR-AUC 0.9212 [0.8785, 0.9613], ROC-AUC 0.9963, and Brier score 0.0047, with nested sensitivity 0.8370, precision 0.8851, and F1 0.8603 (counts 5083/10/15/77). Local TabPFN v3.5 had PR-AUC 0.8957 [0.8446, 0.9426] and Brier score 0.0048, with nested sensitivity 0.7826 and F1 0.8229. The highest classic PR-AUC was XGBoost 0.6322 [0.5331, 0.7247], then LightGBM 0.6271 [0.5313, 0.7202]. The paired bootstrap difference in PR-AUC, thinking v3.5 minus LightGBM, was 0.2941 (0.2071–0.3805), with P(Δ ≤ 0) = 0/2000. The frozen Wang integer score on the same 5,185 rows had ROC-AUC 0.8013 and PR-AUC 0.1032. In the 13-covariate logistic screen, adjusted odds ratios (Wald 95% intervals) included `1.1:1Post dilation` 0.152 [0.081, 0.286], Clopidogrel 0.480 [0.293, 0.787], and WBC 1.972 [1.667, 2.331]. These odds ratios describe the recorded label. They are not treatment effects.

**Conclusions.** On this derivation cohort, thinking v3.5 had the highest nested PR-AUC of the nine arms (0.9212). Local TabPFN v3.5 (0.8957) also exceeded XGBoost (0.6322) and LightGBM (0.6271) under an unmatched tuning budget: the classic arms used defaults and class weights, and TabPFN had no per-dataset grid. Follow-up time, WBC, raw laboratory decimals, a full-cohort stent codebook, and training-set SMOTE are not baseline covariates for this binary task. Nested discrimination on the derivation file is not an external test.

**Keywords:** very late stent thrombosis; TabPFN; nested cross-validation; data leakage; class imbalance; acute coronary syndrome; percutaneous coronary intervention; SHAP

## Introduction

Very late stent thrombosis is Academic Research Consortium 2007 definite stent thrombosis more than one year after implantation, confirmed angiographically. Probable and possible stent thrombosis are not counted. The clinical problem is late and uncommon. In Wang’s review, stent thrombosis accounts for a substantial share of new myocardial infarction after the index PCI and carries several-fold higher adjusted mortality than infarction unrelated to a previously stented site. The decision window of interest is risk stratification more than one year after PCI, after the mandated year of dual antiplatelet therapy.

The analysed file is Wang 2020’s derivation cohort: consecutive patients with acute coronary syndrome, aged at least 18 years, undergoing PCI at The First Hospital of Jilin University from 1 January 2014 to 1 June 2015. Wang reported 6,038 eligible patients and 5,185 analysed (236 in-hospital deaths, 413 refusals of follow-up, and 204 lost to follow-up). Those flow counts are cited from Wang and are not reconstructed here. The file contains 92 definite VLST events (prevalence 0.0177, or 1.77%). Median follow-up is 1,502 days. Median time from PCI to VLST among events is 697 days. A score already exists on these rows. In Wang’s comparison, the Dangas late stent thrombosis score had c-statistic 0.66; that figure is not recomputed here. Wang derived an 8-variable Cox score (diabetes, previous PCI, acute myocardial infarction as the admitting diagnosis, estimated glomerular filtration rate below 90, three-vessel disease, stents per lesion, a sirolimus-eluting stent class, and no post-dilation) with derivation c-statistic 0.80 [0.75, 0.85] and a Shantou c-statistic of 0.82 (n = 2,058). The Shantou file is not in this repository. The machine-learning models below were not applied to it.

Three properties of that score, and of a naive binary analysis of the same file, motivate the present work. First, Wang modelled time to event, with follow-up as the time axis. Recoded as a covariate, `Time since stent implantation` is time to thrombosis for the 92 cases (minimum 380 days) and completed event-free follow-up for the 5,093 non-events (minimum 1,241 days). A rule that calls an event whenever time is below 1,241 days has no false positives among controls. That column is leakage, not a baseline covariate. Second, the raw file is sorted by outcome, and cases were transcribed at a different numeric precision, so recording precision marks case versus control. A signature-only probe that keeps 243 decimal-grid indicators and discards magnitudes reaches average precision 0.4270. Wang excluded white-cell count from the Cox score because infection could not be ruled out; a false-discovery screen of the file still ranks `WBC`. Third, 92 events among 81 candidate columns is severe imbalance. Events per variable are about 1.14 on that candidate list and about 7.1 on the 13-covariate logistic specification below, which remains under a conventional threshold of 10. ROC-AUC is dominated by true negatives. PR-AUC at prevalence 0.0177 is the ranking scale used here.

Gradient-boosted trees and penalised logistic regression are the usual tabular baselines. On this file a 70/30 grid search that keeps follow-up time, WBC, unquantized laboratories, a full-cohort stent encoder, and training-set SMOTE raises hold-out PR-AUC by 0.30 to 0.61 relative to the inverted flags. The nested comparison therefore uses library defaults plus class weighting, an inner loop that tunes only the F1 threshold, and no imported grid-search winners. The classic arms do not receive a hyperparameter search that TabPFN does not receive. That difference in search effort is a limitation of the comparison.

TabPFN is a tabular foundation model: in-context learning on mixed-type tables, native categorical columns, and no per-dataset grid. The pins used here are `tabpfn==9.0.0` for local v3 and v3.5 checkpoints and `tabpfn-client==0.6.0` for hosted thinking-high v3 and v3.5. Version 3 and version 3.5 are different weights. Thinking-high inference and local inference are different objects. This analysis does not claim that TabPFN is ready for clinical use, that nested cross-validation is an external test, or that attributions are treatment effects.

On the same derivation cohort, beyond Wang’s integer score, the study adds four items. It specifies a five-part anti-leakage protocol and shows, on a twin 70/30 grid search of classic models, that inverting those parts together inflates hold-out ranking. It reports one nested 5×4 stratified cross-validation of five classic models and four TabPFN arms under that protocol. It scores Wang’s published integer points on these 5,185 rows with the weights frozen, as a historical comparator rather than a nested arm. It reports association screens and TabPFN attributions that are not supplied as a feature mask to the nested predictor. It does not add an external or temporal test of the machine-learning models, a refit of Wang’s Cox linear predictor, or a decision-curve analysis.

## Methods

Association denotes full-cohort univariate and multivariable screens. Prediction denotes only the nested out-of-fold ranking. Interpretation denotes classic-model selector catalogues and TabPFN explanations. Nested discrimination on the derivation file is not an external test. Adjusted odds ratios and attribution maps are not treatment effects. Wang 2020 is a historical comparator on the same rows.

### Cohort and preprocessing

The analysis is retrospective and uses `data/raw/VLST.csv`. Analysed n is 5,185, with 92 definite VLST events, 5,093 non-events, and prevalence 0.0177. Median follow-up is 1,502 days. Median time from PCI to VLST is 697 days. Ethics approval NO. 2013-256, written informed consent, and registration NCT03491891 are cited from Wang. The repository analysis was not separately pre-registered.

The outcome is binary VLST (`Stent thrombosis`): ARC 2007 definite stent thrombosis more than one year after implantation, angiographically confirmed. Wang analysed time to event. This study uses the stored 0/1 label. Exploratory analysis found no missing values. Identifiers `NO.` and `Name` are dropped. Later analyses use the same 5,185 rows with different splits: the full cohort for association, an inner 80/20 split for classic selectors, one nested 5×4 scheme for prediction, and one 70/30 split for the leakage contrast and for attribution.

Quantization is a file-level cleaning rule, applied before any train, test, or cross-validation split. It is not a parameter fit on the outcome. The incentive is that recording precision in the CSV is a case–control batch marker. The signature-only probe described above reaches average precision 0.4270 and ROC-AUC 0.9522. The grid is `Cre` to 0 decimal places, `CaI` to 2, `Fiberinogen` to 1, and `Fast-Glu` to 1. `Fiberinogen` is the spelling in the file. `Cre` is integer µmol/L. `CaI` is left at two decimal places. This rounding is not post-split leakage.

### Anti-leakage protocol

The operational protocol is five flags plus companion controls. Nested prediction, classic selectors, and TabPFN attribution use the anti-leakage state. The 70/30 contrast inverts all five flags together. SMOTE is used only on the training portion of the leaks-present arm.

| Flag | Anti-leakage state | Leaks-present demonstration only | Reason |
| --- | --- | --- | --- |
| Keep follow-up time | Drop `Time since stent implantation` | Keep it | The column mixes time to event with completed follow-up. The rule “time < 1,241 → event” has no false positives among controls. |
| Drop WBC | Drop `WBC` | Keep it | Recording precision marks batch, and Wang excluded WBC from the Cox score because infection could not be ruled out. The false-discovery screen still ranks it. |
| Quantize named laboratories | On, before any split | Off | Decimal grids fingerprint source. After the anti-leakage state, equalising leftover precision still lowers TabPFN v3.5 nested average precision by 0.0518. |
| Stent codebook | Fit on the training fold only | Fit on the full cohort before the split | Rare brands must not take levels from held-out rows. Unseen strings map to `Other`. |
| SMOTE | Off | On, training rows of that arm only | Nested cross-validation does not synthesise minority rows. |

For events, `Time since stent implantation` is time to thrombosis (minimum 380 days). For non-events it is event-free follow-up (minimum 1,241 days, maximum 1,605 days). WBC is omitted from the machine-learning matrix. Aspirin, Clopidogrel, Ticagrelor, and DAPT remain, with the caveat that they record persistence after the mandated year rather than the index prescription.

The stent column is `Stent type-SES`. Brands with count below 30 collapse to `Other` (106 raw strings to 9 levels). Nested cross-validation fits that codebook on each outer training fold. Classic models then clone imputation, scaling, and one-hot encoding inside each split. TabPFN sees the 9-level column directly. Because the file has no missing values, imputers do not change the data.

Classic selectors use one split of all 5,185 rows: fit on 80% (4,148 rows, 74 events) and score on 20% (1,037 rows, 18 events), seed 42. They do not reserve a further 70/30 test, and their catalogues are not a mask for nested prediction. Grid-search parameters from the leakage twins are not imported.

Where a hold-out is required, for the leakage twins and for attribution, one stratified split uses test size 0.30 and seed 42: training 3,629 rows and 64 events, held-out 1,556 rows and 28 events. That split is not the prediction evaluation.

### Models

Two classic regimes are not interchangeable.

The leakage contrast uses grid search on seven families: logistic regression, decision tree, random forest, Gaussian naïve Bayes, CatBoost, XGBoost, and LightGBM. The split is the shared 3,629/1,556 division. Scoring is average precision. The inner search uses stratified 5-fold cross-validation with shuffle and seed 42. SMOTE is applied only on the leaks-present training set. Those selected hyperparameters are not the nested-model settings.

The nested benchmark has nine arms: logistic regression, random forest, XGBoost, LightGBM, and CatBoost, plus four TabPFN arms. Classic constructors are library defaults with class weighting (`class_weight="balanced"`; CatBoost `auto_class_weights="Balanced"`; XGBoost `scale_pos_weight` from the outer training fold). Decision tree and Gaussian naïve Bayes are not nested arms. No SMOTE is used.

Package pins, from the nested run on a Tesla T4, are `tabpfn==9.0.0` and `tabpfn-client==0.6.0`. Automatic model resolution in that `tabpfn` release still selects v3, so checkpoints are named explicitly.

| Arm | Runtime | Checkpoint | Thinking |
| --- | --- | --- | --- |
| TabPFN v3 | Local | `tabpfn-v3-classifier-v3_default.ckpt` | Off (`n_estimators="auto"`) |
| TabPFN v3.5 | Local | `tabpfn-v3.5-20260909.safetensors` | Off |
| TabPFN thinking v3 | Hosted client | `v3_default` | On, effort high, metric average precision |
| TabPFN thinking v3.5 | Hosted client | `v3.5_default` | Same settings |

Mutual information, stability selection, and partial dependence use local TabPFN v3.5 with thinking mode off and no client fits. Held-out Shapley values were intended for the client thinking constructor. The stored run recorded an HTTP 429 response and then finished the remaining held-out rows locally. That is an execution note. It does not change the intended constructor. The hosted free-tier comments recorded with the run are 20 thinking fits and 50 million prediction cells per day.

Wang’s 8-variable integer score is applied to all 5,185 rows with the published point weights frozen. Weights are not refit. The score is not a nested arm and not a Cox refit. Sirolimus-eluting points use the column `PES`. The four post-dilation points use `No postdilation` equal to 1. The flipped encoding, with ROC-AUC 0.5084, is rejected.

### Evaluation

The primary ranking metric is average precision (PR-AUC) at prevalence 0.0177. ROC-AUC and the Brier score are reported beside it. TabPFN arms are not collapsed into one number.

Prediction uses one stratified nested cross-validation: 5 outer folds and 4 inner folds, outer shuffle, seed 42. Events in the outer folds are 18, 18, 18, 19, and 19. The inner loop chooses the F1 decision threshold only. It does not choose hyperparameters or a feature set. The scheme was run once. It is not a repeated nested cross-validation. Repeated splits elsewhere in the study mean the four inner folds, the eight stability seeds, or the 2,000 bootstrap resamples of stored scores. They do not mean that the 5×4 scheme itself was repeated.

The operating point quoted below applies each outer fold’s inner-loop F1 threshold once to that fold’s unseen rows. Precision, sensitivity, F1, and the 2×2 counts use that cut. F2 uses β = 2.0 where that metric is computed. A single F1 cut fit on the concatenated out-of-fold labels is optimistically biased and is not reported as the result.

Intervals are percentile intervals from a patient-level stratified bootstrap of the pooled out-of-fold scores, with 2,000 resamples and seed 42, keeping 92 events and 5,093 non-events. Paired differences in PR-AUC use the same resamples. Classifiers are not refit inside the bootstrap.

Events per variable are 92/81 ≈ 1.14 on the candidate list, 92/17 ≈ 5.4 for an unidentified 17-covariate logit that is not the association model, and 92/13 ≈ 7.1 for the 13-covariate logit. Association logits do not use class weights. A Firth fit on the same 13 covariates is an association sensitivity only. It is not a nested arm.

TabPFN has no per-dataset grid. Classic nested models are defaults plus class weights. The comparison is unmatched on search effort.

### Association, selection, and attribution

Continuous univariate tests use a Welch *t*-test when absolute skewness is at most 1 and excess kurtosis is at most 3, and a Mann–Whitney *U* test otherwise. Binary tests use χ² unless an expected cell is below 5, in which case Fisher’s exact test is used. Multiplicity for the discovery list is Benjamini–Hochberg false-discovery rate within each family. The exploratory multivariable model is an unweighted binomial logit. The specification that is quoted adjusts 13 covariates, one name from each collinear block, with continuous covariates scaled per 1 standard deviation. Intervals are Wald intervals. A 2,000-replicate percentile bootstrap of that fit is stored as a robustness check. The 17-covariate specification that places exact complements side by side is unidentified (infinite variance-inflation factor on the post-dilation pair) and is not the quoted model.

Classic selectors, on the 4,148/1,037 split and the 87-column scaled matrix after dropping follow-up time and WBC, are leave-one-covariate-out refits, coalition Shapley values on validation PR-AUC, and greedy forward selection. Each selector takes its own prefix of one cheap fit-slice ranking. Reported consensus is the intersection of each model’s top 20 names across the three selectors. Those catalogues do not enter the nested predictor.

TabPFN attribution uses the 3,629/1,556 split and an 80-column matrix under the same anti-leakage cleaning. Mutual information and partial dependence are computed on the training rows. The mutual-information call in `tabpfn_interpretability_fs_pdp.ipynb` cell 9 is `mutual_info_classif` with `discrete_features=False` and `random_state=42`. `n_neighbors` is not passed, so the scikit-learn default of 3 is used. The seed is fixed in that cell. Forward sequential feature selection is repeated over eight shuffled seeds on the training rows; the methods table for that run records a top-10 search scored by average precision. Frequencies below are counts out of eight. A conflicting keep-count and cross-validation fold count are confined to Appendix B and are not used as results. Shapley values explain all 1,556 held-out rows, with the training rows as the fit and background. Pairwise *k*-SII and the native SHAP-IQ plots use one held-out event, cohort row 5176, budget 256. They are not a cohort interaction screen. The consensus score is a Borda-style mean of normalized ranks over the continuous signals that were on disk when the report cell ran. The feature-selection notebook warns that, if `interpretability_shap_mean_abs.csv` is missing, the report still runs on mutual information and selection frequency. The feature-selection dump that ran without that file stores `shap_mean_abs` as 0 and `n_methods` at most 2. A code comment in that cell says “out of 4 methods”; the implemented count adds three indicators (mutual-information top 15, selection frequency at least 0.5, and Shapley top 15), so the comment overstates the count by one. Part 5 Table 5, which is the consensus quoted below, merges the Shapley file afterward. Its `n_methods` values of 3 are that merged table, not the two-signal run. A 10-seed selection plan was not finished. Provisional tables from three of ten seeds are archived and are not results.

## Results

### Cohort

The file has 5,185 patients, 92 VLST events, 5,093 non-events, and prevalence 0.0177. No column has missing values. Table 1 is association, not prediction. DAPT and Clopidogrel record follow-up persistence, not the index prescription. `LV` and `CaI` are unnamed in the file. Means of `CaI` match the peak troponin I row of Wang’s Table 1 and are not treated as a new marker. The column `PES` recovers Wang’s sirolimus-eluting stent row.

**Table 1.** Cohort characteristics. Continuous cells are mean (SD). Binary cells are n (%). TSSI is omitted because it is time at risk.

| Characteristic | No VLST (n = 5,093) | VLST (n = 92) | Test | *p* |
| --- | --- | --- | --- | --- |
| Age, years | 59.83 (9.93) | 60.71 (11.33) | Welch | 0.463 |
| Men | 3489 (68.51%) | 68 (73.91%) | χ² | 0.268 |
| Diabetes | 1293 (25.39%) | 36 (39.13%) | χ² | 0.003 |
| Hypertension | 2670 (52.42%) | 51 (55.43%) | χ² | 0.567 |
| Previous PCI | 94 (1.85%) | 10 (10.87%) | Fisher | 1.25e-05 |
| Previous MI | 347 (6.81%) | 10 (10.87%) | χ² | 0.128 |
| Admitting diagnosis AMI | 3095 (60.77%) | 65 (70.65%) | χ² | 0.054 |
| 3-vessel disease | 1422 (27.92%) | 42 (45.65%) | χ² | 0.000 |
| LVEF, % | 55.15 (4.52) | 54.55 (3.68) | Mann–Whitney | 0.033 |
| LV (unnamed) | 44.55 (4.04) | 49.11 (4.23) | Welch | 5.44e-17 |
| WBC, 10⁹/L | 8.75 (3.24) | 12.49 (3.92) | Mann–Whitney | 7.90e-21 |
| Creatinine | 72.53 (24.81) | 72.44 (19.05) | Mann–Whitney | 0.879 |
| eGFR | 120.03 (34.10) | 95.88 (19.63) | Welch | 4.64e-20 |
| CaI (unnamed) | 37.37 (61.64) | 40.55 (72.25) | Mann–Whitney | 0.051 |
| Fibrinogen (`Fiberinogen`) | 3.17 (0.88) | 3.37 (1.01) | Mann–Whitney | 0.012 |
| Stents per lesion | 1.21 (0.46) | 1.42 (0.65) | Mann–Whitney | 0.000 |
| Total stent length, mm | 31.70 (15.62) | 38.46 (20.71) | Mann–Whitney | 0.001 |
| SES (`PES` column) | 3502 (68.76%) | 76 (82.61%) | χ² | 0.004 |
| 1.1:1 post-dilation (as stored) | 2496 (49.01%) | 14 (15.22%) | χ² | 1.30e-10 |
| No postdilation (complement) | 2597 (50.99%) | 78 (84.78%) | χ² | 1.30e-10 |
| eGFR < 90 (`CKD90`) | 860 (16.89%) | 32 (34.78%) | χ² | 6.55e-06 |
| DAPT during follow-up | 2260 (44.37%) | 35 (38.04%) | χ² | 0.226 |

The plots below are the EDA screens that sit behind Table 1. They are not a second copy of that table. The test-selection map records the rule used for each continuous column (Welch when absolute skewness is at most 1 and excess kurtosis is at most 3; Mann–Whitney otherwise). The significance and effect-size plots are the same univariate screen: the smallest continuous *p* values in Table 1 are WBC, eGFR, and LV. The binary odds-ratio plot is the FDR screen whose largest stored associations include previous PCI and the post-dilation pair. The stent-type panel is the categorical rate for `Stent type-SES`, the same χ² that puts `PES` at 0.004 in Table 1.

![Figure. Continuous test-selection map](paper_results/01_eda/paper_figures/paper_fig1_test_selection_map.png)

![Figure. Univariate continuous significance](paper_results/01_eda/paper_figures/paper_fig2_univariate_significance.png)

![Figure. Continuous effect sizes](paper_results/01_eda/paper_figures/paper_fig3_continuous_effect_sizes.png)

![Figure. Binary odds ratios, FDR q < 0.05](paper_results/01_eda/paper_figures/paper_fig4_binary_odds_ratios.png)

![Figure. VLST rate by stent type](paper_results/01_eda/paper_figures/paper_fig5_categorical_rates_Stent_type-SES.png)

The quoted multivariable screen has 13 covariates (events per variable about 7.1). An odds ratio below 1 is a lower modelled odds of the recorded label in this specification. It is not a treatment effect. Adjusted odds ratios and Wald intervals include `1.1:1Post dilation` 0.152 [0.081, 0.286], Clopidogrel 0.480 [0.293, 0.787], WBC 1.972 [1.667, 2.331], previous PCI 6.710 [2.884, 15.610], eGFR 0.568 [0.449, 0.717], and LV 1.832 [1.539, 2.181]. The full 13-covariate fit, with variance-inflation factors, is Table 2.

**Table 2.** Thirteen-covariate unweighted logit. Continuous covariates are per 1 SD. Wald intervals are the primary intervals.

| Feature | VIF | Univariate OR | Adjusted OR | Wald 95% CI |
| --- | ---: | ---: | ---: | --- |
| WBC | 1.05 | 2.090 | 1.972 | [1.667, 2.331] |
| eGFR | 1.04 | 0.469 | 0.568 | [0.449, 0.717] |
| LV | 1.03 | 2.098 | 1.832 | [1.539, 2.181] |
| No.of stents per lesion | 3.96 | 1.383 | 1.421 | [0.970, 2.080] |
| HbA1c | 1.79 | 1.282 | 0.960 | [0.724, 1.272] |
| NO.of vessels | 1.06 | 1.469 | 1.212 | [0.954, 1.539] |
| Total stent length | 4.02 | 1.378 | 1.160 | [0.777, 1.731] |
| Fiberinogen | 1.03 | 1.206 | 1.024 | [0.845, 1.240] |
| 1.1:1Post dilation | 1.07 | 0.187 | 0.152 | [0.081, 0.286] |
| Previous PCI | 1.01 | 6.485 | 6.710 | [2.884, 15.610] |
| Clopidogrel | 1.00 | 0.503 | 0.480 | [0.293, 0.787] |
| Diabetes | 1.77 | 1.889 | 1.452 | [0.795, 2.652] |
| PES | 1.03 | 2.158 | 1.734 | [0.953, 3.154] |

The largest variance-inflation factor in this specification is 4.02 (`Total stent length`). The post-dilation variance-inflation factor is 1.07, against an infinite value when the exact complement is entered beside it.

![Figure. Univariate versus adjusted odds ratios](paper_results/01_eda/paper_figures/paper_fig6_uni_vs_multivariable_or.png)

The figure compares the univariate odds ratio with the adjusted odds ratio in Table 2. Columns whose Wald interval in Table 2 includes 1 move across the null after adjustment (`HbA1c`, stent count, vessel count, total stent length, `Fiberinogen`, diabetes, `PES`). A Firth fit on the same 13 covariates is stored with the EDA tables as a sensitivity; the intervals quoted here remain the Wald intervals. The 17-covariate specification that places the post-dilation complements side by side is unidentified and is not shown again as a table.

Exploratory correlation views were computed on the cohort and were not used to drop columns from the prediction matrix. `eda.ipynb` reports Pearson correlations, a Spearman check, and hierarchical clustering on Spearman |r| with average linkage (cut distance 0.3, about |r| ≥ 0.7). In the displayed top pairs, Fast-Glu with HbA1c has Pearson r = 0.761349. The same display also lists Min-stent diameter with Max-stent diameter (0.904075), TCL with LDL (0.878888), and total stent length with stents per lesion (0.863269). The notebook text recommends one representative per tight cluster before a multivariable fit. The anti-leakage prediction matrix still contains both Fast-Glu and HbA1c: Fast-Glu is quantized to 1 decimal place and remains a column (mutual-information rank 23). Table 2 keeps HbA1c and omits Fast-Glu. That omission is the 13-covariate specification, not a documented deletion of Fast-Glu from the 80-column model matrix.

### Leakage contrast

Seven classic models share one stratified 70/30 split (training 3,629; hold-out 1,556 rows and 28 events; seed 42) and a grid search scored by average precision. This contrast is not nested cross-validation and does not include TabPFN. All five flags flip together. SMOTE is on only in the leaks-present arm. Calling that arm “TSSI on” would understate the contrast: follow-up time, WBC, unquantized laboratories, a full-cohort stent encoder, and training-set SMOTE change together. Table 3 puts hold-out PR-AUC and the stored F1 on one row. Each difference is leaks-present minus anti-leakage, using the two printed values. It is not a refit.

**Table 3.** Hold-out leakage contrast. PR-AUC is the ranking metric. F1 is the stored operating point.

| Model | PR-AUC, leaks present | PR-AUC, anti-leakage | ΔPR-AUC | F1, leaks present | F1, anti-leakage | ΔF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.9134 | 0.3431 | +0.5703 | 0.6753 | 0.2093 | +0.4660 |
| Decision tree | 0.7524 | 0.1378 | +0.6146 | 0.7059 | 0.2791 | +0.4268 |
| Random forest | 0.9400 | 0.4874 | +0.4526 | 0.8333 | 0.0000 | +0.8333 |
| Gaussian naïve Bayes | 0.2728 | 0.0564 | +0.2163 | 0.0437 | 0.0370 | +0.0067 |
| CatBoost | 0.9599 | 0.4942 | +0.4656 | 0.9231 | 0.4096 | +0.5135 |
| XGBoost | 0.9547 | 0.5685 | +0.3862 | 0.9231 | 0.5500 | +0.3731 |
| LightGBM | 0.9687 | 0.6675 | +0.3011 | 0.9231 | 0.4444 | +0.4787 |

Every model’s hold-out PR-AUC is higher when the leak flags are on. The PR-AUC differences span +0.2163 to +0.6146. LightGBM falls by 0.3011 and remains the smallest PR-AUC drop among the boosting models. The F1 differences are larger for random forest (+0.8333), because anti-leakage F1 is 0, and smallest for Gaussian naïve Bayes (+0.0067). ROC-AUC moves less. Gaussian naïve Bayes ROC-AUC is slightly higher in the anti-leakage arm (0.7493 versus 0.7425) while PR-AUC still falls from 0.2728 to 0.0564. At the operating point stored with that run, random-forest F1 in the anti-leakage arm is 0, with no predicted events, while ROC-AUC remains 0.9287. Boosting F1 values of 0.9231 on the leaks-present arm fall to 0.4096–0.5500 when the flags are off. Accuracy stays high for most models because 1,528 of 1,556 hold-out rows are non-events. Accuracy is not the ranking metric.

The horizontal bars are the same hold-out PR-AUC values as Table 3. Red is leaks present. Blue is anti-leakage. The dotted line is prevalence 0.0177.

![Figure. Hold-out PR-AUC, leaks present versus anti-leakage](paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_tssi_pr_auc.png)

The ROC and PR curves show the same hold-out scores as the PR-AUC columns of Table 3. The leaks-present curves sit higher on precision across recall. The anti-leakage curves fall toward prevalence. The confusion panels are the stored 2×2 counts behind the F1 columns, including the random-forest anti-leakage panel with no predicted events.

![Figure. ROC and PR curves, leaks present](paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_leakage_roc_pr_on.png)

![Figure. ROC and PR curves, anti-leakage](paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_leakage_roc_pr_off.png)

![Figure. Confusion matrices, leaks present](paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_leakage_confusion_on.png)

![Figure. Confusion matrices, anti-leakage](paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_leakage_confusion_off.png)

A 70/30 grid search that retains follow-up time together with WBC, unquantized laboratories, a full-cohort stent encoder, and training-set SMOTE overstates hold-out ranking on this derivation file. Nested cross-validation, the classic selectors, and the TabPFN attributions therefore use the anti-leakage state. The contrast does not transfer to TabPFN: there is no matched nine-arm nested comparison with the leak flags on. An older logistic-regression pair, 0.9575 to 0.5077, is archived and is not this result (Appendix B).

### Nested cross-validation

Nine arms were evaluated in one nested run under the anti-leakage state (`tabpfn==9.0.0`, `tabpfn-client==0.6.0`, Tesla T4). Classic arms are defaults plus class weighting. Grid-search winners are not imported. No SMOTE is used. Table 4 ranks pooled out-of-fold scores. The intervals are percentile bootstrap intervals on those stored scores.

**Table 4.** Pooled nested out-of-fold ranking. One 5×4 run.

| Rank | Model | Checkpoint | Thinking | PR-AUC (percentile 95% interval) | ROC-AUC | Brier |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1 | TabPFN thinking v3.5 | Hosted `v3.5_default` | High | 0.9212 [0.8785, 0.9613] | 0.9963 | 0.0047 |
| 2 | TabPFN v3.5 | Local `tabpfn-v3.5-20260909.safetensors` | Local | 0.8957 [0.8446, 0.9426] | 0.9916 | 0.0048 |
| 3 | TabPFN thinking v3 | Hosted `v3_default` | High | 0.8319 [0.7626, 0.8945] | 0.9834 | 0.0066 |
| 4 | TabPFN v3 | Local v3 checkpoint | Local | 0.7150 [0.6260, 0.8074] | 0.9731 | 0.0099 |
| 5 | XGBoost | — | — | 0.6322 [0.5331, 0.7247] | 0.9374 | 0.0100 |
| 6 | LightGBM | — | — | 0.6271 [0.5313, 0.7202] | 0.9444 | 0.0106 |
| 7 | CatBoost | — | — | 0.5707 [0.4750, 0.6729] | 0.9404 | 0.0108 |
| 8 | Random forest | — | — | 0.3506 [0.2660, 0.4633] | 0.8883 | 0.0150 |
| 9 | Logistic regression | — | — | 0.2596 [0.1834, 0.3588] | 0.8651 | 0.0788 |

The PR and ROC curves are the pooled out-of-fold scores in Table 4. Separation on the precision–recall panel is the ranking result. The ROC panel is high for every arm, which is why ROC-AUC is the less informative scale at prevalence 0.0177.

![Figure. Nested out-of-fold PR and ROC curves](paper_results/04_tabpfn_rating/paper_figures/paper_fig1_pr_roc_curves.png)

The highest classic PR-AUC is XGBoost 0.6322, then LightGBM 0.6271. Paired bootstrap differences in PR-AUC are 0.2941 (0.2071–0.3805) for thinking v3.5 minus LightGBM, 0.2686 (0.1822–0.3559) for local v3.5 minus LightGBM, and 0.2891 (0.2038–0.3791) for thinking v3.5 minus XGBoost. In each contrast, P(Δ ≤ 0) = 0/2000.

Within a checkpoint family, v3.5 exceeds v3 on the pooled scores: local 0.8957 versus 0.7150, and thinking 0.9212 versus 0.8319. Within a version, thinking exceeds local on the pooled scores: v3.5 0.9212 versus 0.8957, and v3 0.8319 versus 0.7150. Thinking v3.5 is not higher than local v3.5 in every outer fold. In fold 1, local PR-AUC is 0.9365 and thinking PR-AUC is 0.8802. Fold-wise PR-AUC for thinking v3.5 is 0.8802, 0.9078, 0.9434, 0.9858, and 0.8961.

Table 5 uses the inner-fold F1 thresholds, not a cut fit on pooled labels. Thinking v3.5 uses threshold 0.318 ± 0.059. ROC-AUC percentile intervals are 0.9963 [0.9932, 0.9986] for thinking v3.5 and 0.9916 [0.9828, 0.9978] for local v3.5. The lowest nested expected calibration error is local v3.5 at 0.0003; thinking v3.5 is 0.0028. These calibration summaries are not a decision analysis.

**Table 5.** Nested operating point from inner-fold F1 thresholds.

| Model | Sensitivity | Precision | F1 |
| --- | ---: | ---: | ---: |
| TabPFN thinking v3.5 | 0.8370 | 0.8851 | 0.8603 |
| TabPFN v3.5 | 0.7826 | 0.8675 | 0.8229 |
| TabPFN thinking v3 | 0.6957 | 0.8767 | 0.7758 |
| TabPFN v3 | 0.6087 | 0.7089 | 0.6550 |
| XGBoost | 0.5217 | 0.6575 | 0.5818 |
| LightGBM | 0.5870 | 0.5934 | 0.5902 |
| CatBoost | 0.5978 | 0.5093 | 0.5500 |
| Random forest | 0.4565 | 0.4421 | 0.4492 |
| Logistic regression | 0.3587 | 0.3028 | 0.3284 |

The calibration curves are the pooled out-of-fold probabilities behind the Brier scores in Table 4 and the expected calibration errors quoted above. Local v3.5 (ECE 0.0003) lies closest to the diagonal. Thinking v3.5 (ECE 0.0028) is the next. These curves are not a decision analysis.

![Figure. Nested calibration](paper_results/04_tabpfn_rating/paper_figures/paper_fig2_calibration_curves.png)

Confusion counts at that operating point are 5083/10/15/77 for thinking v3.5, 5082/11/20/72 for local TabPFN v3.5, and 5056/37/38/54 for LightGBM. Order is true negatives, false positives, false negatives, and true positives, matching the counts that produce the sensitivity and precision above.

The frozen Wang integer score, scored on all 5,185 rows and not entered as a nested arm, has ROC-AUC 0.8013 and PR-AUC 0.1032. The published derivation c-statistic is 0.80. The comparison of nested thinking-v3.5 PR-AUC 0.9212 with that integer score is a same-file comparison of out-of-fold foundation-model scores against an 8-variable point scale. It is not an external test, and it does not inherit Wang’s Shantou result.

The score-bin figure shows the event rate rising across the frozen integer score on the same 5,185 rows. That rate pattern is the discrimination behind ROC-AUC 0.8013. It is not the nested PR-AUC of 0.9212.

![Figure. Event rate by frozen Wang score bin](paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_wang_score_rate.png)

## Feature relevance: statistical association, selection stability, and model-based attribution

The catalogues below are copied from printed tables. No model was refit, and no new association or importance score was calculated. Overlap counts are counts of those printed names. The catalogues are descriptive. They are not treatment effects, and they are not a feature mask for the nested predictor.

### Which split each method used

| Catalogue | What is ranked | Rows the report states | Split |
| --- | --- | --- | --- |
| Univariate FDR (χ², Fisher, Mann–Whitney, Welch) and the sparse multivariable logit | Full-cohort association | n = 5,185, 92 events | Full cohort. Not a train/held-out split. `feature_extraction_comparison.md` §1. |
| Classic LOCO, coalition SHAP, and FFS | Validation PR-AUC | Fit 4,148 rows / 74 events; val 1,037 rows / 18 events | One 80/20 split (`INNER_VAL_SIZE=0.2`, `random_state=42`). No parked 70/30 test. Part 2 opening protocol. |
| `mutual_info_classif` | Marginal association on the 80-column matrix | Train n = 3,629 | Train only. Part 5 Table 0 and Table 1. |
| Stability forward SFS | How often a feature is kept across 8/8 seeds | Train n = 3,629 | Train only. Part 5 Table 0 and Table 2. |
| PDP (binary table and continuous figure) | Average predicted probability under an empirical prior | Train n = 3,629 | Train only. Not nested-CV risk. Part 5 §3. |
| SHAP (shapiq SV), mean absolute value | Attribution | Explain all 1,556 held-out rows (28 events / 1,528 non-events); fit and background = train | Held-out explanations. Part 5 Table 0 and §4. |
| *k*-SII and native SHAP-IQ plots | Pairwise interaction for one patient | Cohort row 5176, the first held-out VLST=1 row (held-out position 20); budget 256 | One row. Not a cohort screen. Part 5 §5–§6. |
| Borda consensus | Mean of normalized ranks | Runtime feature-selection dump: mutual information and selection frequency only (`shap_mean_abs` = 0). Part 5 Table 5: those two signals plus merged held-out mean(\|SHAP\|) | The 3/3 labels below are the merged table. Part 5 §7. |

`WBC` and `Time since stent implantation` are in the FDR catalogue and are not columns in the classic-ML or TabPFN matrices on these dumps (`feature_extraction_comparison.md` §1; Part 5 Table 0).

### Name mapping

Names below are the column strings printed in the tables. They are not renamed to a clinical synonym.

| Printed variant | Row name used here | Why it is not merged |
| --- | --- | --- |
| `HbA1C` | `HbA1c` | The priority-rank excerpt looks up `HbA1C` and records no rank (`paper_table5_priority_ranks_excerpt.csv`). Selector and TabPFN tables use `HbA1c`. |
| `Fiberinogen` | `Fiberinogen` | CSV spelling, kept as printed (`feature_extraction_comparison.md` §1; `main_paper.md` §3.1). |
| `1.1:1 post-dilation`, `1.1:1Post dilation` | `1.1:1Post dilation` | Same stored column. |
| `No post-dilation`, `No postdilation` | `No postdilation` | Same stored column. It is the complement of `1.1:1Post dilation` and stays on its own row (`feature_extraction_comparison.md` §1). |
| Wang “SES” | `PES` | The FDR name is the column `PES`, not a second stent variable. |
| `Stent type-SES_xiencev` | separate from `Stent type-SES` | One-hot dummy versus the 9-level parent. Part 3 §4–§5. |
| `Age, years`, `Male sex`, `aspirin` | not mapped | Part 2 Table 5: those priority labels miss `Age`, `Men`, and `Aspirin`. |

### Core agreement

Table 6 keeps features that meet at least two of these printed screens: a stored mutual-information rank, a forward-selection count of at least 1/8, a stored Shapley rank, membership in the FDR-20 list, or membership in the classic-ML consensus. The classic-ML consensus is the 10-name union defined in `feature_extraction_comparison.md` §1 (LOCO ∩ SHAP ∩ FFS top-20 for at least one classic model): `Clopidogrel`, `Cre`, `HGB`, `HbA1c`, `LDL`, `LV`, `Men`, `No postdilation`, `Stent type-SES_xiencev`, `eGFR`. It is not a top-12 rule and it is not a 20-name catalogue. A blank rank means the name is outside the cited ranking, not that a zero score was filled in. The full row-level notes, Borda labels, and binary PDP deltas are Appendix C.

**Table 6.** Features in at least two screens. Ranks are the stored ranks. SFS is the count out of 8 seeds.

| Feature | MI rank | SFS (/8) | SHAP rank | FDR | Classic-ML consensus |
| --- | ---: | --- | ---: | --- | --- |
| eGFR | 3 | 8/8 | 1 | yes | yes |
| LV | 2 | 8/8 | 4 | yes | yes |
| HbA1c | 5 | 7/8 | 7 | yes | yes |
| No postdilation | 52 | 2/8 | 6 | yes | yes |
| LDL | 4 | 2/8 | 5 | no | yes |
| Cre | 50 | 7/8 | 3 | no | yes |
| Clopidogrel | 54 | — | 33 | yes | yes |
| 1.1:1Post dilation | 9 | 4/8 | 9 | yes | no |
| Stent type-SES | 13 | 1/8 | 10 | yes | no |
| Fiberinogen | 11 | 1/8 | 19 | yes | no |
| Previous PCI | 27 | 1/8 | 8 | yes | no |
| HGB | 8 | — | 20 | no | yes |
| Men | 80 | — | 14 | no | yes |
| CaI | 1 | 8/8 | 2 | no | no |
| TCL | 33 | 2/8 | 11 | no | no |
| Age | 26 | 6/8 | 37 | no | no |
| LVEF | 62 | 5/8 | 30 | no | no |
| CKD90 | 22 | — | 13 | yes | no |
| CKD5 | 19 | — | 16 | yes | no |
| PES | 41 | — | 39 | yes | no |
| Diabetes | 25 | — | 23 | yes | no |
| HDL | 10 | 0 | 15 | no | no |
| stent overlap | 14 | — | 21 | no | no |
| Slow flow | 16 | 1/8 | — | no | no |
| No.of stents per lesion | — | — | 17 | yes | no |
| Total stent length | — | — | 18 | yes | no |

Five names are both FDR and classic-ML consensus: `eGFR`, `LV`, `HbA1c`, `No postdilation`, and `Clopidogrel`. That is the intersection printed in `feature_extraction_comparison.md` §2. `Cre`, `HGB`, `LDL`, and `Men` are consensus names that fail FDR, which is why their FDR cell is no. `Stent type-SES_xiencev` is the fifth consensus-only name, but it is a one-hot dummy rather than a second screen, so it is not a row of Table 6. `CaI` leads mutual information and is kept in 8/8 seeds with Shapley rank 2, and it is not in the FDR-20 list or the 10-name consensus. `WBC` is FDR-only and is absent from the model matrices, so it does not meet the two-screen rule.

Mutual-information values for the printed top 15, and creatinine at rank 50 (0.000338), are Part 5 Table 1. Shapley values for the leading names are eGFR 1.2288, CaI 1.0867, Cre 0.8093, and LV 0.4828 (Part 5 Table 4). The summary and beeswarm plots show the sign of those held-out attributions, which the rank column does not. The scatter is the leading feature in that summary, eGFR, despite the file name. Partial-dependence curves are the training empirical prior, not nested risk. The one-row *k*-SII plots are cohort row 5176 only.

![Figure. SHAP summary, 1,556 held-out rows](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig3_shap_summary.png)

![Figure. SHAP beeswarm](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig6_shap_beeswarm.png)

![Figure. SHAP scatter for the leading feature, eGFR](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig4_shap_scatter_age.png)

![Figure. Continuous partial dependence, training rows, empirical prior](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig1_pdp_continuous.png)

![Figure. Binary partial dependence](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig2_pdp_binary.png)

![Figure. One-row SHAP waterfall, cohort row 5176](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig7_shap_waterfall.png)

![Figure. One-row k-SII network, cohort row 5176](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig8_ksii_network.png)

![Figure. One-row k-SII UpSet, cohort row 5176](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig9_ksii_upset.png)

![Figure. SHAP-IQ force plot, same row. The 2026-09-20 write-up says this panel was not in the dump; the file is in `paper_figures`](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig10_shapiq_force.png)

![Figure. SHAP-IQ k-SII network, same row](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig11_shapiq_network.png)

![Figure. SHAP-IQ k-SII UpSet, same row](paper_results/05_tabpfn_interpretability/paper_figures/paper_fig12_shapiq_upset.png)

### Overlap

Sets, all taken from printed lists:

- **FDR-20.** Twenty names after dropping time-at-risk (`feature_extraction_comparison.md` §1; `table_feature_by_method.csv`).
- **ML consensus.** The 10-name catalogue in `feature_extraction_comparison.md` §1, not a separate top-12 list.
- **MI-15 / SHAP-15.** Part 5 Tables 1 and 4.
- **SFS ≥ 0.5.** Eight names in Part 5 Table 2: `CaI`, `LV`, `eGFR`, `Cre`, `HbA1c`, `Age`, `LVEF`, `1.1:1Post dilation`.
- **Merged Borda 3/3.** In the merged Part 5 Table 5, not in the two-signal runtime dump: `CaI`, `eGFR`, `LV`, `HbA1c`, `1.1:1Post dilation`.

`feature_extraction_comparison.md` §2 states one overlap fraction: FDR-20 versus the 10-name ML consensus, intersection 5, union 25, Jaccard 5/25 = 0.20. The same section lists the intersection as `Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR`. A Jaccard of 5/35 ≈ 0.14 for two 20-name sets is not in that file. The other rows of Table 7 are counts of the same printed names. Their Jaccard cells are |A ∩ B| / (|A| + |B| − |A ∩ B|) applied to those printed sizes. They are not a new test and they are not in the comparison file.

**Table 7.** Overlap counts. Jaccard uses the printed set sizes.

| Pair | \|A\| | \|B\| | \|A ∩ B\| | Jaccard |
| --- | ---: | ---: | ---: | --- |
| FDR-20 ∩ ML consensus | 20 | 10 | 5 | 5/25 = 0.20 |
| FDR-20 ∩ MI-15 | 20 | 15 | 6 | 6/29 |
| FDR-20 ∩ SHAP-15 | 20 | 15 | 8 | 8/27 |
| FDR-20 ∩ SFS ≥ 0.5 | 20 | 8 | 4 | 4/24 |
| ML consensus ∩ MI-15 | 10 | 15 | 5 | 5/20 = 0.25 |
| ML consensus ∩ SHAP-15 | 10 | 15 | 7 | 7/18 |
| ML consensus ∩ SFS ≥ 0.5 | 10 | 8 | 4 | 4/14 |
| MI-15 ∩ SHAP-15 | 15 | 15 | 8 | 8/22 |
| MI-15 ∩ SFS ≥ 0.5 | 15 | 8 | 5 | 5/18 |
| SHAP-15 ∩ SFS ≥ 0.5 | 15 | 8 | 6 | 6/17 |

Member lists:

- FDR-20 ∩ ML consensus: `Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR`.
- FDR-20 ∩ MI-15: `1.1:1Post dilation`, `Fiberinogen`, `HbA1c`, `LV`, `Stent type-SES`, `eGFR`.
- FDR-20 ∩ SHAP-15: `1.1:1Post dilation`, `CKD90`, `HbA1c`, `LV`, `No postdilation`, `Previous PCI`, `Stent type-SES`, `eGFR`.
- FDR-20 ∩ SFS ≥ 0.5: `1.1:1Post dilation`, `HbA1c`, `LV`, `eGFR`.
- ML consensus ∩ MI-15: `HGB`, `HbA1c`, `LDL`, `LV`, `eGFR`.
- ML consensus ∩ SHAP-15: `Cre`, `HbA1c`, `LDL`, `LV`, `Men`, `No postdilation`, `eGFR`.
- ML consensus ∩ SFS ≥ 0.5: `Cre`, `HbA1c`, `LV`, `eGFR`.
- MI-15 ∩ SHAP-15: `1.1:1Post dilation`, `CaI`, `HDL`, `HbA1c`, `LDL`, `LV`, `Stent type-SES`, `eGFR`.
- MI-15 ∩ SFS ≥ 0.5: `1.1:1Post dilation`, `CaI`, `HbA1c`, `LV`, `eGFR`.
- SHAP-15 ∩ SFS ≥ 0.5: `1.1:1Post dilation`, `CaI`, `Cre`, `HbA1c`, `LV`, `eGFR`.

**Strict core** (in FDR-20, in the ML consensus, in MI-15, in SHAP-15, and in SFS ≥ 0.5): `HbA1c`, `LV`, `eGFR`.

**In all three families, counting a TabPFN hit as any of MI-15, SHAP-15, or SFS ≥ 0.5:** those three, plus `No postdilation` (FDR, CatBoost 3-way, and SHAP rank 6; SFS frequency 2/8; MI rank 52).

**Merged Borda 3/3 names that are not in that strict core:** `CaI` (not FDR-20 and not in the 10-name consensus; ML frequent only) and `1.1:1Post dilation` (FDR and ML frequent, not in the 10-name consensus).

**ML-consensus names that miss MI-15, SHAP-15, and SFS ≥ 0.5:** `Clopidogrel` and `Stent type-SES_xiencev`. The parent `Stent type-SES` is in MI-15 and SHAP-15; the dummy is a different column.

**FDR-20 names that miss the ML consensus and also miss MI-15, SHAP-15, and SFS ≥ 0.5:** `3-vessel disease`, `CKD5`, `Diabetes`, `Multi-vessel CAD`, `NO.of vessels`, `No.of stents per lesion`, `PES`, `Single-vessel disease`, `Total stent length`, `WBC`. Several of these still have a stored rank just outside the printed top 15 (`CKD5` MI 19 and SHAP 16; `No.of stents per lesion` SHAP 17; `Total stent length` SHAP 18; `Diabetes` SHAP 23). `WBC` is absent as a column, not as a low score. Part 3 groups the vessel-disease names, the post-dilation complement, and `CKD5` / `CKD90` / `eGFR` as repeated encodings of fewer constructs (`feature_extraction_comparison.md` §1).

Cross-model classic agreement, separate from the union above (Part 2 Table 1): LOCO top-20 shared by all 7 models is `Cre` and `eGFR`; coalition SHAP top-20 shared by all 7 is `Cre`, `HGB`, `LDL`, and `eGFR`; FFS top-20 shared by all 7 is empty. No name is in every model × selector top-20 (Part 2 Table 4). The Venn and the membership heatmap are that FDR-versus-consensus comparison. The reason-bucket and domain figures are the groupings already stated in `feature_extraction_comparison.md` §6–§7: redundant anatomy and renal encodings on the FDR side, and a thinner anatomy block on the consensus side. The Jaccard heatmap is the selector-union overlap already printed as 0.60, 0.49, and 0.37, not Table 7. The other selector panels show within-model consensus size, which model agrees on which name, counts by family, unique-name counts, union size by model, names repeated in the selector log, and counts by model and algorithm. They display those set sizes. They are not a second score table.

![Figure. Overlap of the FDR list and the classic three-way union](paper_results/03_stats_vs_ml/paper_figures/fig1_venn_overlap.png)

![Figure. Membership by extractor](paper_results/03_stats_vs_ml/paper_figures/fig2_presence_heatmap.png)

![Figure. Reason buckets for catalogue disagreement](paper_results/03_stats_vs_ml/paper_figures/fig3_reason_buckets.png)

![Figure. Counts by clinical domain](paper_results/03_stats_vs_ml/paper_figures/fig4_domain_counts.png)

![Figure. Size of the cross-model intersection](paper_results/02_ml_selectors/paper_figures/paper_fig6_cross_model_common.png)

![Figure. Jaccard overlap of selector unions](paper_results/02_ml_selectors/paper_figures/paper_fig2_jaccard.png)

![Figure. Within-model consensus size](paper_results/02_ml_selectors/paper_figures/paper_fig3_consensus_size.png)

![Figure. Which features each classic model agrees on](paper_results/02_ml_selectors/paper_figures/paper_fig4_feature_by_model.png)

![Figure. Consensus counts by model family](paper_results/02_ml_selectors/paper_figures/paper_fig5_family_stacked.png)

![Figure. Unique selected-feature counts](paper_results/02_ml_selectors/paper_figures/paper_fig1_unique_counts.png)

![Figure. Union size by model](paper_results/02_ml_selectors/paper_figures/paper_fig7_union_by_model.png)

![Figure. Features most often written into the selector log](paper_results/02_ml_selectors/paper_figures/selector_top_repeated_features.png)

![Figure. Selector counts by model and algorithm](paper_results/02_ml_selectors/paper_figures/selector_model_algorithm_counts.png)

![Figure. Selector overlap](paper_results/02_ml_selectors/paper_figures/selector_overlap_heatmap.png)

### Disagreement

Each contrast below is a difference between a marginal screen (FDR or MI), a conditional refit or greedy inclusion (LOCO or SFS), and an attribution average (SHAP). The redundancy notes are the ones already written in `feature_extraction_comparison.md` §4–§7. They are not clinical interpretations.

- **`Cre`.** Marginal screens are low: not in FDR-20 (*p* = 0.88 in Part 3 Table 4) and train MI rank 50 of 80 with value 0.000338 (Part 5 Table 1 note and Part 5 Table 5). Conditional and attribution screens are high: SFS 7/8, SHAP rank 3 with mean(\|SHAP\|) 0.8093, and classic 3-way for `lr`, `rf_b`, and `xgb_b`. Part 3’s stated algorithm reason is redundancy with `eGFR`: the marginal test of `Cre` is null because `eGFR` already carries the renal contrast, while a model that splits on creatinine, or that is scored after a refit, can still list `Cre`. LOCO shared by all 7 models is exactly `{Cre, eGFR}` (Part 2 Table 1). No Cre–eGFR correlation coefficient is printed in the tables used here, so none is stated.
- **`CaI`.** Marginal FDR does not include it (Part 3 membership: not FDR-20). Train MI rank is 1 (0.020536), SFS is 8/8, SHAP rank is 2 (1.0867), and Borda is 3/3. Classic 3-way does not include it; the frequent-selection list does. That is a marginal multiplicity screen versus train MI, greedy inclusion, and held-out attribution. The tables used here do not print a CaI correlation with another column, so no correlation is stated.
- **`No postdilation` versus `1.1:1Post dilation`.** Part 3 calls them two encodings of one bit. χ² can list both. On this dump CatBoost’s 3-way keeps `No postdilation` (SHAP rank 6, mean 0.2812; MI rank 52). The complement `1.1:1Post dilation` is the MI rank-9 / SFS 4/8 / Borda 3/3 name and is not in any model 3-way. A fitted model needs one representative; the two selectors need not pick the same representative.
- **`Men`.** Not FDR-20 (*p* = 0.27). Stored train MI is 0 at rank 80. Classic 3-way includes it for logistic regression only. Held-out mean(\|SHAP\|) rank is 14 (0.1314). Part 3’s stated reason is that a univariate screen misses an additive offset, and that the EDA interaction list already flags `Men × eGFR`, which is a different contrast from univariate `Men`.
- **`HGB`.** Not FDR-20 (raw *p* = 0.039). MI rank 8 (0.004365) and classic 3-way for `rf`, `xgb`, and `xgb_b`. Held-out SHAP rank is 20 (0.0644), and the Borda row marks `in_shap_top` as no. Part 3 Table 4 calls this a ranking signal rather than a location test. Tree SHAP agreement across all 7 models does include `HGB` inside the classic top-20 universe (Part 2 Table 1); that classic coalition score is not the TabPFN mean(\|SHAP\|).
- **`Clopidogrel`.** FDR and multivariable yes, and classic 3-way for `cat` and `lr`. It is outside MI-15 (stored MI 0, rank 54), outside the stability table, and outside SHAP-15 (stored rank 33, mean 0.0397). A full-cohort 2×2 and a linear/CatBoost validation refit are not the same object as train MI or held-out mean absolute SV.
- **`Age` and `LVEF`.** SFS frequencies are 6/8 and 5/8. Their held-out mean(\|SHAP\|) ranks are 37 and 30 (0.0373 and 0.0411). `LVEF` stored MI is 0 at rank 62. Selection frequency counts how often greedy forward search keeps a column; mean absolute SV is an average attribution on other rows. A high frequency does not imply a high mean(\|SHAP\|).
- **`Stent type-SES` versus `Stent type-SES_xiencev`.** One χ² is computed on the 9-level parent. The scaled classic matrix one-hots that column; only the `xiencev` dummy is in a 3-way set (`xgb_b`). TabPFN tables rank the parent (MI 13, SHAP 10, SFS 1/8), not the dummy. Part 3 §4–§5 states this as an encoding difference.
- **`CKD90` / `CKD5` versus `eGFR`.** Part 3 §4: the binaries are cutpoints on the same axis as continuous `eGFR`, and the classic consensus keeps the continuous lab. `CKD90` is still SHAP rank 13. `CKD5` is MI rank 19 and SHAP rank 16, and Part 3 says its adjusted odds ratio flips sign beside `eGFR`. That is collinear re-encoding, which a univariate FDR list does not remove and a single-column model list can.
- **`WBC`.** Part 3: strongest continuous FDR name on the statistical side, and not a column in the 2026-09-19 selector matrix or the TabPFN 80-column matrix. There is no ML or TabPFN rank to compare.

### Reported elsewhere but not verifiable in these tables

- **SFS step gains.** Part 5 Table 2 and `paper_table2_stability.csv` report `times_selected` and `selection_freq` only. No per-step marginal gain is printed. Part 2 describes FFS early stopping when PR-AUC gain stops rising (`FFS_MIN_GAIN=0`) but does not print those gains.
- ***k*-SII numeric interactions.** Part 5 §5–§6 documents the method and the single-row limit (cohort row 5176, budget 256). Figures 8–9 and 11–12 are plots. No table of interaction values is in the report or in `paper_figures/` as a CSV. Figure 10 (force plot) is stated as not exported. Magnitudes are not filled in here.
- **Continuous PDP grid values.** Part 5 Figure 1 is a train empirical-prior curve. The only numeric PDP table is the six binary rows (Part 5 Table 3).
- **Per-feature classic LOCO deltas, coalition SHAP magnitudes, and the cheap-importance order.** Part 2 prints set sizes, the all-7 intersections, within-model 3-way names, Jaccard of top-20 unions (LOCO–SHAP 0.60, SHAP–FFS 0.49, LOCO–FFS 0.37), and a priority-rank excerpt. It does not print a 20-row score table for LOCO, SHAP, or FFS.
- **Priority aliases with no rank.** `HbA1C`, `Age, years`, `Male sex`, and `aspirin` are blank in `paper_table5_priority_ranks_excerpt.csv`. Part 2 Table 5 says the miss is the alias, not an unscored column. CatBoost LOCO hits that do match are Hypertension rank 9, Clopidogrel rank 10, Current smoker rank 35, Current drinking rank 50; CatBoost SHAP hits Clopidogrel rank 12 and Current smoker rank 14.
- **Keep-count and CV fold count for TabPFN SFS.** Part 5 Table 0 says “top-10 forward SFS”. `main_paper.md` §4.3 says “keep 10 of 80, 5-fold CV”. The notebook source sets `STABILITY_K = min(8, n_features)` and `FS_CV = 3` (`tabpfn_interpretability_fs_pdp.ipynb` cells 8–9). The stability frequencies above are the printed 8-seed counts. The conflicting keep-k and fold settings are not treated as a result.
- **A 10-seed SFS.** `main_paper.md` §4.3 says a 10-seed plan was not finished and that 3/10 provisional tables are archived. They are not used.
- **Optional TabPFN arm inside `baseline_feature_selections.ipynb`.** The notebook can add `tabpfn` if the import exists. Part 2 Table 0 reports seven classic models. No eighth consensus row is in the tables.
- **Barnard, Boschloo, Brunner–Munzel, Bonferroni, and Holm.** Those procedures are implemented in `eda.ipynb`. The comparison report used here states the FDR discovery list and a handful of χ² / Fisher / Mann–Whitney / Welch figures. It does not reprint those other tests’ feature rankings.


## Discussion

The ranking that remains after the anti-leakage protocol is a single nested cross-validation on the derivation file. It is not a bedside rule, and it is not an external test. After follow-up time and WBC are dropped, named laboratories are quantized before splitting, the stent codebook is fit on each training fold, and SMOTE is omitted, TabPFN thinking v3.5 has the highest nested PR-AUC of the nine arms, 0.9212 [0.8785, 0.9613]. Local TabPFN v3.5 is second, at 0.8957 [0.8446, 0.9426]. Both exceed XGBoost at 0.6322 and LightGBM at 0.6271. The paired difference for thinking v3.5 minus LightGBM is 0.2941 (0.2071–0.3805), with P(Δ ≤ 0) = 0/2000. Nested F1 at the inner-fold thresholds is 0.8603 for thinking v3.5 (sensitivity 0.8370), 0.8229 for local v3.5, and 0.5902 for LightGBM.

TabPFN is a tabular foundation model used here on a small, mixed-type table with 92 events, without a per-dataset grid. The classic nested arms are library defaults plus class weighting, and the inner loop tunes only the F1 threshold. That unmatched search budget is a limitation. Version 3 and version 3.5 are different checkpoints. Thinking-high client inference and local inference are different objects. Thinking outranks local within each family on pooled PR-AUC, but not in every fold: in fold 1, local v3.5 PR-AUC is 0.9365 and thinking v3.5 is 0.8802. With 18 or 19 events in each outer fold, that movement is expected. PR-AUC is the ranking metric because prevalence is 0.0177. ROC-AUC is high for every arm, from 0.8651 to 0.9963, and is the less informative scale.

A leaks-present 70/30 scoreboard is not the TabPFN result. There is also no matched nine-arm nested run with the leak flags turned on, so a numeric nested difference for TabPFN with versus without those flags is not available. Unlabeled nested PR-AUC values of 0.9771 and 0.9635, and an earlier seven-model thinking result of 0.8553, belong to other runs and are not used (Appendix B).

Names that recur are associations or attributions. In the 13-covariate logit, post-dilation and Clopidogrel have adjusted odds ratios below 1, and WBC, previous PCI, and LV have adjusted odds ratios above 1. DAPT columns are follow-up persistence. Stability selection keeps `CaI`, `LV`, and `eGFR` in 8 of 8 seeds. Held-out mean absolute Shapley values for the leading names are eGFR 1.2288, CaI 1.0867, Cre 0.8093, and LV 0.4828. The false-discovery list and the 10-name ML consensus share `Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, and `eGFR` (Jaccard 5/25 = 0.20, as printed in `feature_extraction_comparison.md` §2). `eGFR` and `LV` appear on both the classic-selector side and the TabPFN side. `CaI` is prominent in the TabPFN training and Shapley lists and is only on the frequent classic-selector list, not in the three-way union. `WBC` is on the false-discovery list and is absent from the machine-learning matrices. The strict overlap of the false-discovery list, the classic three-way union, mutual-information top 15, Shapley top 15, and selection frequency at least 0.5 is `HbA1c`, `LV`, and `eGFR`. Disagreements, including low mutual information and high Shapley value for `Cre`, follow from marginal versus conditional versus attribution questions and from redundancy already described for `eGFR`, not from a treatment interpretation. The one-row *k*-SII display and the training partial-dependence curves are not cohort interaction screens and are not nested risks.

The frozen Wang score recovers ROC-AUC 0.8013, next to the published derivation c-statistic of 0.80, with PR-AUC 0.1032. Comparing nested thinking-v3.5 PR-AUC 0.9212 with that point score does not carry over the Shantou c-statistic of 0.82.

The leakage contrast is the result that should govern how this file is modelled. Hold-out PR-AUC falls by 0.30 to 0.61 when the five flags are turned off together (LightGBM 0.9687 to 0.6675; logistic regression 0.9134 to 0.3431). Random-forest F1 falls to 0. The two arms are not identical except for follow-up time: SMOTE and the other flags change as well. Nested cross-validation does not import those grid-search winners and does not use SMOTE. Laboratory quantization is done before splitting. The signature probe (243 indicators, average precision 0.4270) shows how far decimal-grid membership alone can rank. Equalising leftover precision still lowers TabPFN v3.5 nested average precision by 0.0518. Stent brands are encoded on the training fold only. These choices do not establish that every possible leak has been removed. A separate brand-frequency leak on the leaks-present arm was not quantified.

## Limitations

1. The machine-learning models have no external or temporal test. Nested cross-validation on 5,185 derivation rows is not Wang’s Shantou cohort. The file is from one centre, The First Hospital of Jilin University.
2. The label is binary. Wang’s analysis was Cox time-to-event. Dropping follow-up time is required for this binary classification and is not a Cox refit.
3. Events are rare. Events per variable are about 1.14 on 81 candidates, about 5.4 on the unidentified 17-covariate logit, and about 7.1 on the 13-covariate logit, still below 10. Each outer fold has 18 or 19 events.
4. The design is a retrospective derivation cohort. This analysis was not separately pre-registered.
5. Four TabPFN calibrations and two interfaces must stay separate. The lowest nested expected calibration error is local v3.5 at 0.0003; thinking v3.5 is 0.0028. No decision curve is reported. Positive predictive value at an F1 threshold is not an assessment of clinical usefulness.
6. DAPT columns are post-baseline persistence. WBC is on the association list and off the prediction matrix. Tuning effort is unequal. The bootstrap does not refit models. `LV` and `CaI` are unnamed. Attribution is not the nested predictor. The Shapley dump records HTTP 429 and then a local finish of the remaining rows.
7. A full interpretability run exceeded the session limit, so selection and Shapley values were produced separately. Selection used eight seeds. The abandoned 10-seed plan is not a result. Client quota constrained the thinking-high design. The runtime consensus file from that selection run has `shap_mean_abs` = 0 and `n_methods` at most 2. Part 5 Table 5 merges Shapley afterward. A comment in the report cell says the count is out of 4 methods; the code adds three indicators. Mutual information uses `discrete_features=False`, the default `n_neighbors=3`, and `random_state=42`.
8. SMOTE appears only in the leaks-present twin. Nested cross-validation was a single run. Wang’s Cox linear predictor and the Shantou file are absent.

Association is estimated under low events per variable, attribution uses a 70/30 split, and prediction is one nested cross-validation on one derivation file.

## Conclusion

On Wang 2020’s derivation cohort, after the anti-leakage protocol, TabPFN thinking v3.5 had the highest PR-AUC in one nested cross-validation of nine arms (0.9212). Local TabPFN v3.5 (0.8957) also exceeded class-weighted default XGBoost (0.6322) and LightGBM (0.6271). Follow-up time, WBC, raw laboratory decimals, a full-cohort stent codebook, and training-set SMOTE are not baseline covariates for that binary classifier. Recurring names in the association and attribution catalogues — `eGFR`, `LV`, the post-dilation columns, and `CaI` on the TabPFN side — are not treatment effects. Nested discrimination on this file is not an external test. The machine-learning models were not tested on an external or later cohort.

## References

1. Wang X, et al. A novel risk model for predicting very late stent thrombosis after percutaneous coronary intervention: a derivation and validation study. *Scientific Reports*. 2020;10:6378. doi:10.1038/s41598-020-63455-0.

The Dangas c-statistic 0.66 and the Shantou c-statistic 0.82 are cited from Wang 2020. They are not recomputed in this repository, and no separate Dangas bibliographic record is stored in the source reports.

## Appendix A. Reproducibility and provenance

Numbers in the main text are copied from the sources below. They were not recomputed. When sources disagree, the evidence map is followed: Revision 19 for nested prediction (`nested_cv_v35_antileakage_on`), Revision 17 for classic selectors and the FDR overlap, and Revision 16 for TabPFN attribution (2026-09-20 split dumps). Excluded historical values are in Appendix B.

| Subsection | Sources |
| --- | --- |
| Introduction and cohort flow | `manuscript/main_paper.md` Introduction and §3.1, cited from Wang 2020. Ethics NO. 2013-256; NCT03491891 in §3.1. |
| Anti-leakage protocol | `manuscript/main_paper.md` §3.1–§3.2; `paper_results/anti_leakage_protocol.md`. Signature probe AP 0.4270, ROC-AUC 0.9522; quantization Cre 0 / CaI 2 / Fiberinogen 1 / Fast-Glu 1; leftover-precision drop 0.0518; stent encoder 106→9, `min_count=30`. |
| Cohort, Table 1, EDA plots | `paper_results/01_eda/EDA_paper_figures_and_tables.md` §0 Table C and Figures 1–5. Pearson Fast-Glu–HbA1c 0.761349 and the neighbouring pairs: `eda.ipynb` displayed correlation table (Min–Max stent diameter 0.904075, TCL–LDL 0.878888, stent length–stent count 0.863269). Clustering: same notebook, Spearman \|r\|, average linkage, cut distance 0.3. |
| Table 2, 13-covariate logit | `paper_results/01_eda/EDA_paper_figures_and_tables.md` §5 Table 4b. Call: `eda.ipynb` cell 80 (`run_b4`). Firth is the sensitivity table in that section; Wald intervals are the ones quoted. |
| Table 3, leakage contrast | `manuscript/main_paper.md` §4.1, Tables 2a and 2b. ΔF1 is the difference of those two printed F1 columns. ROC/PR and confusion figures: `paper_results/04_tabpfn_rating/paper_figures/`. |
| Tables 4–5, nested ranking and operating point | `manuscript/main_paper.md` §4.2 and Table 3. Bootstrap: 2,000 stratified resamples of stored out-of-fold scores, seed 42, no refit. Curves: `paper_fig1_pr_roc_curves.png`, `paper_fig2_calibration_curves.png`. |
| Wang frozen score | `manuscript/main_paper.md` §4.2. ROC-AUC 0.8013, PR-AUC 0.1032. Rate figure: `paper_fig_s_wang_score_rate.png`. |
| Table 6 and Appendix C | Ranks and frequencies from Part 5 Tables 1, 2, and 4 and the 80-row CSVs `interpretability_mutual_info_ranking.csv`, `interpretability_shap_mean_abs.csv`, `paper_table2_stability.csv`. FDR and consensus flags: `feature_extraction_comparison.md` §1–§5 and `table_feature_by_method.csv`. |
| Table 7 | Intersection counts of those printed lists. The only Jaccard printed in the comparison file is 5/25 = 0.20 (`feature_extraction_comparison.md` §2). Other Jaccard cells are that same ratio on the printed sizes. |
| Mutual information and Borda | `tabpfn_interpretability_fs_pdp.ipynb` cells 8–9 (MI) and 12–13 (consensus). Runtime two-signal file: `code/modeling/interpretability/Kaggle_tabpfn_intrepretebility_results/fs_pdp_MI/modeling_tabpfn/interpretability_feature_importance_report.csv`. Merged three-signal table: Part 5 Table 5 and `rebuild_part5_from_split_dumps.py`. |
| Selectors, disagreement, *k*-SII, PDP | Part 2 Tables 1, 2, and 4 and §7. Part 5 §3–§7. Shapley and *k*-SII calls: `tabpfn_interpretability_shap.ipynb` cells 9 and 11. |

## Appendix B. Conflicts and the artifact that was used

| Conflict | Used in this report | Not used | Why |
| --- | --- | --- | --- |
| Nested TabPFN PR-AUC | Thinking v3.5 0.9212 [0.8785, 0.9613]; local v3.5 0.8957 [0.8446, 0.9426]. Evidence map Revision 19, freeze `nested_cv_v35_antileakage_on`, nine-arm anti-leakage dump | Unlabeled nested 0.9771 / 0.9635 (Revision 14; status excluded in Revisions 18–19). Version 4 thinking 0.8553 and local 0.6742 or 0.6754 (Revisions 7 and 10) | Revision 19 is the latest nested artifact the evidence map marks as the live anti-leakage run |
| Leakage-contrast logistic PR-AUC | 0.9134 → 0.3431, with the seven-model table in `manuscript/main_paper.md` §4.1. Evidence map Revision 16 | Archived pair 0.9575 → 0.5077, labelled archived in `main_paper.md` §4.1 | Revision 16 rebuilt the contrast from the 2026-09-19 twin dumps |
| Classic consensus versus FDR | 10-name ML consensus and Jaccard 5/25 = 0.20, intersection `Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR`. Evidence map Revision 17; `feature_extraction_comparison.md` §1–§2 | ML-13 and Jaccard 5/28 from the 2026-08-31 reconstruction (Revision 5). A 20-name consensus and Jaccard 5/35 ≈ 0.14 are not in `feature_extraction_comparison.md` | Revision 17 ingested the 2026-09-19 selector dump. The comparison file on disk defines the consensus as the union of within-model LOCO ∩ SHAP ∩ FFS top-20 (n = 10) and states Jaccard 5/25 = 0.20. This paper follows that file |
| TabPFN SFS keep-count and fold count | Frequencies out of 8 seeds, and the methods-table phrase “top-10 forward SFS” (`paper_table0_methods.csv`, aligned with Revision 16) | This paper does not state a cross-validation fold count for that search | `main_paper.md` §4.3 and Table 4 say “keep 10 of 80, 5-fold CV”. `tabpfn_interpretability_fs_pdp.ipynb` cells 8–9 set `STABILITY_K = min(8, n_features)` and `FS_CV = 3`. The stability CSV stores frequencies, not the fold count. The frequency table is the result. The keep-count and fold count remain unresolved and are not quoted as findings |
| TabPFN attribution rankings | 2026-09-20 dumps: MI CaI 0.020536; SFS 8/8 for CaI, LV, eGFR; SHAP eGFR 1.2288; PDP Previous PCI +0.001294; consensus 3/3 includes CaI, eGFR, LV. Evidence map Revision 16 | Version 5 notebook `e356bb1`: consensus {WBC, LV, eGFR}, Cre MI 0.000000, PDP +0.0137 and −0.0093, and a 10-seed WBC table. Also SHAP on 15+15 rows and *k*-SII rows 5099 or 5093 | Revision 16 says the paper pack follows the 2026-09-20 split dumps. WBC is not a column in that matrix. Provisional 3/10-seed tables are archived in `main_paper.md` §4.3 |
| Part 5 SHAP sample | All 1,556 held-out rows; *k*-SII on cohort row 5176 | Full-cohort SHAP on 5,185 rows; 15+15 explanations | Evidence map Revisions 10–16. The live report is `tabpfn_interpretability_paper_figures_and_tables.md` |

Items mentioned in code or in earlier map revisions but absent as numeric tables are not given values here: per-step SFS gains, *k*-SII magnitudes, continuous PDP grid coordinates, and per-feature classic LOCO or coalition-SHAP scores.

## Appendix C. Full harmonized table

This is the row-level catalogue that Table 6 compresses. A blank or “not in” cell means the name is outside the cited top, not that a score of zero was inferred. “SFS” is the printed stability count out of 8 seeds. “Borda” is the merged Part 5 Table 5 (top 15) or, for ranks beyond 15, `interpretability_feature_importance_report.csv`. Classic ML “3-way” means LOCO ∩ SHAP ∩ FFS top-20 for that model. The band is a reading aid only.

<details>
<summary>Full harmonized rows</summary>

| Band | Feature | Stats (Part 3) | Classic ML | MI rank | SFS | SHAP rank | Borda | PDP Δ (train, binary) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Top 10 | CaI | Not FDR-20. ML frequent only | Not in any model 3-way. Frequent | 1 (0.020536) | 8/8 | 2 (1.0867) | 1 (3/3, 0.9916) | — |
| Top 10 | eGFR | FDR. Welch *d* = −0.71, *q* = 3.7e-19. Multivariable yes | 3-way: lr, rf_b, xgb, xgb_b. LOCO shared by all 7 | 3 (0.009424) | 8/8 | 1 (1.2288) | 2 (3/3, 0.9873) | — |
| Top 10 | LV | FDR. Welch *d* = 1.13, *q* = 3.3e-16. Multivariable yes | 3-way: cat, rf. Appearance count 18 | 2 (0.012818) | 8/8 | 4 (0.4828) | 3 (3/3, 0.9789) | — |
| Top 10 | HbA1c | FDR. MW *r* = 0.052, *q* = 7e-4 | 3-way: cat, lgb (LightGBM’s only 3-way name) | 5 (0.007328) | 7/8 | 7 (0.2673) | 4 (3/3, 0.9430) | — |
| Top 10 | LDL | Not FDR (*p* = 0.33). ML consensus | 3-way: cat, rf. SHAP shared by all 7 | 4 (0.009367) | 2/8 | 5 (0.3659) | 5 (2/3, 0.9325) | — |
| Top 10 | 1.1:1Post dilation | FDR. Multivariable yes. ML frequent, not 3-way | No model 3-way on this dump | 9 (0.004057) | 4/8 | 9 (0.2483) | 6 (3/3, 0.9030) | −0.000094 |
| Top 10 | Cre | Not FDR (*p* = 0.88) | 3-way: lr, rf_b, xgb_b. LOCO shared by all 7 | 50 (0.000338) | 7/8 | 3 (0.8093) | 11 (2/3, 0.7700) | — |
| Top 10 | HGB | Not FDR (raw *p* = 0.039) | 3-way: rf, xgb, xgb_b. Appearance count 18. SHAP shared by all 7 | 8 (0.004365) | not in the stability table | 20 (0.0644) | 15 (1/3, 0.6899) | — |
| Top 10 | No postdilation | FDR. χ² OR ≈ 5.4 | 3-way: cat only | 52 (7.94e-5) | 2/8 | 6 (0.2812) | 12 (1/3, 0.7257) | +0.000153 |
| Top 10 | Stent type-SES | FDR (χ² on 9 levels) | Parent name is not a 3-way entry. Dummy `Stent type-SES_xiencev` is xgb_b only | 13 (0.003176) | 1/8 | 10 (0.2402) | 7 (2/3, 0.8565) | — |
| Top 10 | Fiberinogen | FDR. Prose: MW *r* = 0.035. Frequent, not 3-way | Not in a model 3-way | 11 (0.003712) | 1/8 | 19 (0.0694) | 8 (1/3, 0.8270) | — |
| Top 10 | Previous PCI | FDR. Prose: Fisher OR 6.49. Frequent, not 3-way. Multivariable yes | Not in a model 3-way | 27 (0.001986) | 1/8 | 8 (0.2604) | 9 (1/3, 0.8059) | +0.001294 |
| Top 10 | HDL | Not in the FDR-20 list | Not in the 10-name ML consensus list | 10 (0.003823) | Part 5 Table 5 stability 0.000 (not in the stability table) | 15 (0.1174) | 14 (2/3, 0.7025) | — |
| Top 10 | Lesion location-Ostial | Not in the FDR-20 or 10-name ML consensus lists | — | 6 (0.005293) | not in the stability table | not in SHAP top 20 | 23 in the consensus CSV | −0.000004 |
| Top 10 | P-RCA | Not in the FDR-20 or 10-name ML consensus lists | — | 7 (0.004830) | not in the stability table | not in SHAP top 20 | not in consensus top 20 | — |
| 11–20 | TCL | Not FDR-20. ML frequent | Not in a model 3-way | 33 (0.001167) | 2/8 | 11 (0.2369) | 10 (1/3, 0.7848) | — |
| 11–20 | Age | Not in the FDR-20 or 10-name ML consensus lists | — | 26 (0.002065) | 6/8 | 37 (0.0373) | 13 (1/3, 0.7215) | — |
| 11–20 | Men | Not FDR (*p* = 0.27) | 3-way: lr only | 80 (stored MI 0) | not in the stability table | 14 (0.1314) | not in Borda top 15 | — |
| 11–20 | CKD90 | FDR. Multivariable yes. Not ML frequent | Not in a model 3-way | 22 (0.002380) | not in the stability table | 13 (0.1377) | 18 in the consensus CSV | — |
| 11–20 | TG | Not in the FDR-20 or 10-name ML consensus lists | — | not in MI top 20 | not in the stability table | 12 (0.2213) | 21 in the consensus CSV | — |
| 11–20 | stent overlap | Not in the FDR-20 or 10-name ML consensus lists | — | 14 (0.003081) | not in the stability table | 21 (0.0533) | 17 in the consensus CSV | — |
| 11–20 | P-LCX | Not in the FDR-20 or 10-name ML consensus lists | — | 15 (0.003014) | not in the stability table | not in SHAP top 20 | not in consensus top 20 | — |
| 11–20 | Current drinking | Not in the FDR-20 or 10-name ML consensus lists | — | 12 (0.003202) | not in the stability table | not in SHAP top 20 | not in consensus top 20 | — |
| 11–20 | Slow flow | Not in the FDR-20 or 10-name ML consensus lists | — | 16 (0.002941) | 1/8 | not in SHAP top 20 | 20 in the consensus CSV | −0.000007 |
| 11–20 | LVEF | Not FDR-20. ML frequent | Not in a model 3-way | 62 (stored MI 0) | 5/8 | 30 (0.0411) | not in Borda top 15 | — |
| 11–20 | CKD5 | FDR. Multivariable yes. Not ML frequent | Not in a model 3-way | 19 (0.002578) | not in the stability table | 16 (0.1025) | 19 in the consensus CSV | — |
| 11–20 | No.of stents per lesion | FDR. Not ML frequent | Not in a model 3-way | not in MI top 20 | not in the stability table | 17 (0.0798) | not in consensus top 20 | — |
| 11–20 | Total stent length | FDR. ML frequent | Not in a model 3-way | not in MI top 20 | not in the stability table | 18 (0.0768) | not in consensus top 20 | — |
| 11–20 | Initial diagnosis-AMI | Not FDR-20. ML frequent | Not in a model 3-way | not in MI top 20 | 1/8 | not in SHAP top 20 | 16 in the consensus CSV | −0.000120 |
| 11–20 | Clopidogrel | FDR. Multivariable yes | 3-way: cat, lr | 54 (stored MI 0) | not in the stability table | 33 (0.0397) | not in Borda top 15 | — |
| Outside | Stent type-SES_xiencev | Not a univariate FDR column. Parent is FDR | 3-way: xgb_b only | — (dummy is not the parent column) | — | — | — | — |
| Outside | Stent type-SES_tivoli | Not FDR-20. ML frequent | Not in a model 3-way | — | — | — | — | — |
| Outside | Fast-Glu | Not FDR-20. ML frequent | Not in a model 3-way | 23 (0.002341) | not in the stability table | not in SHAP top 20 | 24 in the consensus CSV | — |
| Outside | PES | FDR. ML frequent | Not in a model 3-way | 41 (0.000836) | not in the stability table | 39 (0.0363) | not in consensus top 20 | — |
| Outside | Diabetes | FDR. Prose OR 1.89. Not ML frequent | Not in a model 3-way | 25 (0.002086) | not in the stability table | 23 (0.0521) | 22 in the consensus CSV | — |
| Outside | WBC | FDR. Dropped from the ML and TabPFN matrices | Not scored | column absent | column absent | column absent | column absent | — |
| Outside | 3-vessel disease | FDR | Not ML frequent and not 3-way | not in MI top 20 | not in the stability table | not in SHAP top 20 | not in consensus top 20 | — |
| Outside | Multi-vessel CAD | FDR | same | not in MI top 20 | not in the stability table | not in SHAP top 20 | not in consensus top 20 | — |
| Outside | Single-vessel disease | FDR | same | not in MI top 20 | not in the stability table | not in SHAP top 20 | not in consensus top 20 | — |
| Outside | NO.of vessels | FDR | same | not in MI top 20 | not in the stability table | not in SHAP top 20 | not in consensus top 20 | — |
| Outside | Time since stent implantation | Strongest univariate hit; excluded from the overlap count as time-at-risk | Dropped before ML | column absent | column absent | column absent | column absent | — |

</details>

MI and SHAP values in the printed tops are the rounded figures in Part 5 Tables 1 and 4. Ranks and values outside those tops are the stored CSV rows named in Appendix A. Previous PCI’s MI rank 27 value 0.001986 matches the Borda cell, whose `in_mi_top` flag is no. HDL’s merged Part 5 Table 5 stability cell is 0.000. PDP deltas are Part 5 Table 3. Effect sizes and *p* values in the Stats column are the ones printed in `feature_extraction_comparison.md` Tables 2–4. Appearance count 18 for `LV`, `HGB`, and `eGFR` is Part 2 §7.

## Checklist of missing human inputs

- Author names, affiliations, and a corresponding author with an email address.
- Funding sources and grant numbers, or an explicit statement that there was none.
- A conflicts-of-interest statement.
- Whether the authors will add their own ethics or data-use statement beyond the Wang 2020 citation (NO. 2013-256, NCT03491891).
- Clinical names and units for the columns `LV` and `CaI`, which remain unnamed in the file.
- The target journal’s word limit, figure policy, and reference style, if they differ from this draft.

