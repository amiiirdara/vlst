# Discrimination of very late stent thrombosis in a public acute coronary syndrome PCI cohort: a machine-learning benchmark, data-provenance audit, and comparison with a published clinical score

*Revised integrated draft, 31 October 2026. Not submission-ready: see the author queries marked in the text.*

**Authors.** [AUTHOR QUERY: names, affiliations, ORCID, corresponding author]

**Running title.** VLST discrimination and data provenance in a public PCI cohort

---

## Abstract

**Background.** Very late stent thrombosis (VLST), defined as definite stent thrombosis more than one year after implantation, is rare and hard to predict. Wang et al. derived an eight-item integer score for VLST from a Cox model in 5,185 patients with acute coronary syndrome (ACS) and deposited the derivation data publicly. We used this file to compare machine-learning classifiers, to examine how the file's recording properties affect apparent discrimination, and to evaluate the published score on the same labels.

**Methods.** This was a secondary analysis of the public derivation file (single centre, China, PCI between January 2014 and June 2015; 92 VLST events, prevalence 1.77%). We compared five conventional learners (library defaults with class weighting) and four TabPFN arms: local v3 and v3.5, and hosted v3 and v3.5 in "thinking" mode, configured to optimise average precision. All arms shared one stratified nested cross-validation (5 outer × 4 inner folds); the inner loop selected only an F1 decision threshold. The primary metric was pooled out-of-fold average precision (AP). Before modelling we excluded follow-up time and white-cell count, rounded four laboratory variables, and fitted the stent-brand codebook within training folds. We audited the file for outcome-dependent recording patterns and evaluated Wang's integer score without refitting. Association and model-attribution analyses were exploratory.

**Results.** The event label was not a fixed-horizon outcome. Events occurred on days 380–1,575 (12 after day 1,461), and event-free patients were followed for 1,241–1,605 days (60.7% for fewer than 1,461 days). The audit identified outcome-dependent recording differences:
- Recorded eGFR could not be reproduced from recorded creatinine for 56.0% of controls, against 45.7% of cases.
- When eGFR was recalculated from creatinine, age and sex, mean eGFR was 91.9 in controls and 91.5 in cases, compared with recorded means of 120.0 and 95.9.
- Laboratory values were formatted differently in cases and controls.
- Medication fields were coded inconsistently in cases.
- Precision-pattern indicators alone reached AP 0.43.

Hosted TabPFN v3.5 had the highest AP (0.921; 95% bootstrap interval 0.878–0.961), followed by local TabPFN v3.5 (0.896; 0.845–0.943) and XGBoost (0.632; 0.533–0.725). Rounding residual decimals to the coarsest precision in each column lowered local TabPFN v3.5 AP to 0.844; hosted arms were not tested. The frozen Wang score had ROC-AUC 0.801 and AP 0.103 on the same labels.

**Conclusions.** On this file, flexible models, particularly TabPFN, discriminated recorded VLST far better than the published clinical score. However, the matrix included post-baseline medication data and laboratory variables whose recording differed by outcome, so this discrimination cannot be interpreted as clinical predictive performance. A post-PCI model for VLST in years 1–4 remains unvalidated. It would require a fixed-horizon label, predictors confirmed to be available at the prediction time, resolution of the recording differences, and external data. [AUTHOR CONFIRMATION PENDING: update the Results and Conclusions with reanalysis R1 if performed.]

**Keywords.** very late stent thrombosis; acute coronary syndrome; percutaneous coronary intervention; machine learning; TabPFN; data leakage; data provenance; prediction-model evaluation

---

## 1. Introduction

Stent thrombosis occurring more than one year after implantation (VLST) is uncommon, but it often presents as acute myocardial infarction and carries substantial mortality [1–3]. Several clinical scores have been proposed for late or very late stent thrombosis:
- the late stent thrombosis score of Dangas et al. (c-statistic 0.66 for VLST, as reported by Wang et al.) [4];
- a score from the ARRIVE registries [5];
- a case-control-derived score [6];
- the eight-item score of Wang et al. [1].

Wang et al. developed their score with stepwise multivariable Cox regression in 5,185 consecutive patients with ACS treated with PCI. They reported c-statistics of 0.80 in derivation and 0.82 in an external cohort of 2,058 patients [1]. The derivation data were made publicly available [7].

Machine-learning methods have been applied to outcomes after PCI. Systematic reviews report variable and often modest gains over conventional models, frequent high risk of bias, and little external validation [8,9]. Most studies have addressed mortality, bleeding or composite events rather than definite VLST. Tabular foundation models such as TabPFN make predictions by in-context learning from the training rows and need no per-dataset gradient training [10]. Later releases (v3, v3.5) and a hosted "thinking" mode that spends extra fit-time compute against a user-chosen metric are documented in technical reports and vendor documentation rather than in peer-reviewed evaluations [11,12].

Rare outcomes in small public datasets present known risks for classifier evaluation. These include leakage of post-outcome or follow-up information into predictors [13], differences in how cases and controls were recorded, and instability of precision-based metrics [14]. A binary classifier can address a fixed-horizon outcome only if that outcome is fully observed for every patient.

Our intended clinical use is prediction shortly after index PCI, using information available at a clearly defined post-procedural time, of VLST occurring between approximately one and four years after PCI. This study did **not** establish such a model. Its aims were:
1. to compare existing machine-learning classifiers for discrimination of the recorded VLST label in the public Wang derivation file, under a shared nested cross-validation;
2. to audit the file for recording properties that could produce apparent discrimination unrelated to clinical risk;
3. to evaluate Wang's published integer score, without refitting, as a clinical reference on the same labels;
4. to describe exploratory clinical associations and model attributions as hypothesis-generating findings.

---

## 2. Methods

### 2.1 Data source and population

This was a retrospective secondary analysis of the publicly deposited derivation cohort of Wang et al. [1,7]. No new participants were enrolled. The original study recruited consecutive patients aged 18 years or older with ACS who underwent PCI at the First Hospital of Jilin University, China, between 1 January 2014 and 1 June 2015 and gave written informed consent (Wang et al., Methods, p. 6) [1].

The original exclusions were:
- expected survival under one year;
- severe organ failure;
- malignancy;
- planned major surgery within one year requiring antiplatelet interruption;
- death before discharge;
- refusal of consent or follow-up interviews;
- loss to follow-up after discharge.

Of 6,038 eligible patients, 5,185 were enrolled (p. 2) [1]. All patients received newer- or second-generation drug-eluting stents and dual antiplatelet therapy (DAPT) for at least one year (p. 6) [1].

The file contains 5,185 rows, one per patient, with 92 VLST events (prevalence 1.77%). Wang et al.'s external validation cohort (n = 2,058) is not publicly available to us and was not analysed. Ethics approval (No. 2013-256) and registration (NCT03491891) belong to the original study [1]. [AUTHOR QUERY: does the authors' institution require its own ethics determination for this secondary analysis of public, de-identified data?] The present analysis was not preregistered.

### 2.2 Outcome, time origin and prediction time

**Outcome as defined by the source.** Definite VLST under the Academic Research Consortium 2007 criteria [2]: stent thrombosis more than one year after implantation, confirmed angiographically and adjudicated by two interventional cardiologists, with a third if needed (p. 6) [1]. We used the file's binary `Stent thrombosis` column.

**Time origin.** Index PCI.

**Follow-up information in the file.** The column `Time since stent implantation` (TSSI) ranged from 380 to 1,575 days among events (median 736) and from 1,241 to 1,605 days among non-events (median 1,419). Wang et al. reported a median follow-up of 1,502 days and a median time to VLST of 697 days (p. 2) [1]; these medians are not reproduced by the TSSI column. [AUTHOR CONFIRMATION PENDING: Please verify how `Time since stent implantation` was constructed for patients with and without VLST, including the end-of-follow-up date and its relation to Wang et al.'s reported medians.]

**What the label represents.** The label records VLST at any time during follow-up:
- 12 of the 92 events occurred after day 1,461 (four years);
- 3,090 of 5,093 event-free patients (60.7%) were followed for fewer than 1,461 days, so the absence of an event is not established for a four-year window;
- no event-free patient was followed for fewer than 1,241 days.

The file contains no information on deaths after discharge. [AUTHOR CONFIRMATION PENDING: How were patients who died after discharge, without VLST, handled: excluded, retained with follow-up truncated at death, or retained with administrative follow-up?]

**Intended versus achieved prediction time.** The intended prediction time is at hospital discharge after index PCI. Discharge, rather than procedure completion, is the earliest defensible time because peak cardiac troponin, if recorded (Section 2.3), requires post-procedural observation. The current prediction matrix does not meet this cutoff because it contains antiplatelet-therapy fields recorded during follow-up (Section 2.3).

**Accordingly, the primary analyses estimate discrimination of the recorded VLST label using recorded variables. They do not evaluate a fixed-horizon 1–4-year post-PCI risk model, and that intended application remains unvalidated.**

### 2.3 Variables and their availability

The file contains 81 candidate predictors after removing identifiers, the outcome and TSSI. Supplementary Table S1 gives each variable's documented meaning, presumed recording time and availability at the intended prediction time. Key points:

- **Antiplatelet therapy (`Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT`).** These record therapy during follow-up rather than at the index PCI. Aspirin is recorded in only 61.8% of event-free patients, although all patients received at least one year of DAPT, and Wang et al. describe DAPT as received "during follow-up" (p. 2) [1]. These fields are **not available** at a post-PCI prediction time. [AUTHOR CONFIRMATION PENDING: Please verify the definition and recording time of `Aspirin`, `Clopidogrel`, `Ticagrelor` and `DAPT` (e.g., status at 12 months, at last contact, or at the time of VLST).]
- **`CaI`.** Its group means and standard deviations (37.37 ± 61.64 and 40.55 ± 72.25) equal those reported for "peak troponin I" in Wang et al.'s Table 1 (p. 3). [AUTHOR CONFIRMATION PENDING: Please verify the full definition, units, coding and recording time of `CaI`, including whether it is peak cardiac troponin I.] We use the dataset label `CaI` throughout.
- **`LV`.** Integer values 37–73. There is no corresponding row in Wang et al.'s Table 1. [AUTHOR CONFIRMATION PENDING: Please verify the full definition, units, coding and recording time of `LV`.] No physiological interpretation is made.
- **Renal variables.** `Cre` appears to be creatinine in µmol/L. `eGFR` is recorded eGFR. `CKD5` takes integer values 1–5 and equals the CKD stage of the recorded eGFR in all patients; `CKD60` and `CKD90` equal recorded eGFR < 60 and < 90. [AUTHOR CONFIRMATION PENDING: Please verify the equation, creatinine time point and units used for `eGFR`.]
- **Stent type.** `PES`, `ZES` and `EVS` are mutually exclusive flags (exactly one per patient). `PES` frequencies equal Wang et al.'s "sirolimus-eluting stent" row (68.76% and 82.61%). `Stent type-SES` contains brand names. [AUTHOR CONFIRMATION PENDING: Please verify whether `PES` denotes sirolimus-eluting stents.]
- **Post-dilation.** `1.1:1Post dilation` and `No postdilation` are exact complements. Wang et al.'s Table 1 labels as "No post-dilation" the frequencies that the file stores under `1.1:1Post dilation` (49.01% and 15.22%). [AUTHOR CONFIRMATION PENDING: Please verify the coding of `1.1:1Post dilation` and `No postdilation`.]
- **Other laboratory variables** (`WBC`, `HGB`, `Platelet`, lipids, `Fast-Glu`, `HbA1c`, `Fiberinogen` [sic; fibrinogen]). Their group summaries match Wang et al.'s Table 1, except HbA1c, whose two group columns are interchanged relative to the file. Measurement time (admission or in-hospital) is not documented. [AUTHOR CONFIRMATION PENDING: measurement time of laboratory variables.]

### 2.4 Data-provenance audit

We assessed whether cases and controls had been recorded or exported differently. The checks were:
- row order;
- the number of decimal places in the raw text of each numeric field;
- the proportion of integer-valued entries;
- the terminal-digit distribution of integer fields;
- internal consistency of derived fields, comparing recorded eGFR with CKD-EPI 2009 eGFR [15] recalculated from recorded creatinine, age and sex, and comparing `DAPT` with the component drug fields;
- agreement of group counts and means with Wang et al.'s Table 1.

A "precision probe" was also fitted: a histogram gradient-boosting classifier trained only on 243 indicators of whether each raw numeric value lay on a 0-, 1- or 2-decimal grid, with no clinical magnitudes, in 5-fold cross-validation using the benchmark's outer-fold seed. The audit used read-only scripts; a standard-library script reproducing all audit numbers is provided.

### 2.5 Cleaning rules applied before modelling

These rules were adopted after the diagnostics in Section 2.4 and were not preregistered.
- Identifiers and TSSI were excluded. TSSI is time to VLST for cases and time to the end of follow-up for controls, so it encodes the outcome. Follow-up time is appropriate as the time axis of a survival analysis, not as a predictor in a binary classifier.
- `WBC` was excluded from the prediction matrix because its recording precision almost perfectly separated cases from controls (Section 3.2). Wang et al. also excluded leukocyte count from their score, though for a different reason: an elevated count could reflect infection (p. 6) [1]. That reason does not by itself imply temporal leakage. WBC was retained for exploratory association analyses.
- `Cre` was rounded to 0 decimal places, `CaI` to 2, `Fiberinogen` to 1 and `Fast-Glu` to 1, before data splitting. These fixed rules do not use outcome data.
- Brand strings in `Stent type-SES` were canonicalised (spelling variants merged). Brands with fewer than 30 training observations were grouped as "other". This learned codebook was fitted within each training partition, and unseen brands in held-out data were mapped to "other".
- Medication fields and recorded eGFR and CKD variables were **retained** in the primary benchmark. Their limitations are addressed in Sections 2.3, 3.2 and 4. [AUTHOR CONFIRMATION PENDING: if reanalysis R1 (medications removed; eGFR recalculated from creatinine) is performed, report it as a prespecified sensitivity analysis with the date it was specified.]

There were no missing values.

### 2.6 Models

**Conventional models.** Logistic regression, random forest, XGBoost, LightGBM and CatBoost, all with library-default hyperparameters and class weighting:
- `class_weight="balanced"` for logistic regression, random forest and LightGBM;
- `auto_class_weights="Balanced"` for CatBoost;
- for XGBoost, `scale_pos_weight` computed from each training partition.

Within each training partition, numeric features were median-imputed and standardised, and the categorical brand feature was one-hot encoded.

**TabPFN** (Prior Labs) [10–12].
- Local v3 checkpoint `tabpfn-v3-classifier-v3_default.ckpt`.
- Local v3.5 checkpoint `tabpfn-v3.5-20260909.safetensors`.
- Both local arms used `tabpfn` 9.0.0, were selected explicitly by version, used `n_estimators="auto"`, and had thinking disabled.
- Hosted v3 (`v3_default`) and hosted v3.5 (`v3.5_default`) via `tabpfn-client` 0.6.0, with `thinking_mode=True`, `thinking_effort="high"` and `thinking_metric="average_precision"`.

According to the vendor documentation, thinking mode is available only through the hosted service; it "uses more compute during fitting", and the metric "defines what to optimize" [12]. Its internal mechanism is not documented. Therefore:
- the hosted arms received metric-targeted fit-time optimisation, performed within each training partition;
- the local TabPFN arms received none;
- the conventional arms received no hyperparameter tuning.

Optimisation budgets are therefore not matched across arms.

Other software: Python 3.12.13, scikit-learn 1.6.1, XGBoost 3.2.0, LightGBM 4.6.0, CatBoost 1.2.10, PyTorch 2.10.0; one NVIDIA Tesla T4 GPU. The SHA-256 hash of the analysed file is recorded in the run manifest.

### 2.7 Primary evaluation: nested cross-validation

All nine arms used the same stratified 5-fold outer split (shuffle, seed 42; 18–19 events per held-out fold). In each outer training partition, a stratified 4-fold inner split produced inner out-of-fold probabilities. These were used **only** to choose the threshold maximising F1. That threshold was applied once to the corresponding outer held-out fold. No hyperparameters or feature subsets were selected in the inner loop. All preprocessing and the brand codebook were refitted within every inner and outer training partition. The procedure was run once; it is not repeated nested cross-validation.

**Metrics.**
- **Primary: pooled out-of-fold average precision** (AP; scikit-learn `average_precision_score`, a step-wise summary of the precision–recall curve, not trapezoidal PR-AUC). The AP of a non-informative classifier equals the prevalence (0.0177).
- **Secondary:**
  - ROC-AUC;
  - Brier score, compared with the null Brier score of a model predicting the prevalence for everyone (0.0177 × 0.9823 = 0.0174);
  - expected calibration error over 8 quantile bins;
  - calibration curves;
  - sensitivity, precision and F1 at the inner-loop thresholds.

ROC-AUC and AP summarise different aspects of ranking. ROC-AUC does not depend on the class ratio. AP reflects precision across recall levels and depends on prevalence, which makes it more informative about positive predictive value when events are rare. Neither metric reflects precision at a single clinical operating point.

**Uncertainty.** Percentile 95% intervals came from 2,000 stratified bootstrap resamples of the stored pooled out-of-fold predictions (seed 42), resampling cases and controls separately, without refitting. Paired differences in AP used the same resamples. These intervals reflect sampling variation of the evaluation rows given the fitted predictions. They do not capture variation from model fitting or fold assignment.

### 2.8 Sensitivity and diagnostic analyses

- **Precision equalisation.** Every numeric column was rounded to the coarsest precision present anywhere in its raw text (for example, `CaI` and `HbA1c` to whole numbers). The local and conventional arms were refitted on the same outer folds, without the inner threshold loop, and AP and ROC-AUC were compared. Hosted arms were not refitted because of service quota. This procedure also discards genuine clinical resolution, so a fall in AP does not isolate artefact.
- **Bundled data-handling contrast.** Separately, on one stratified 70/30 split (3,629/1,556 patients; 64/28 events; seed 42), seven conventional models were grid-searched (5-fold, AP) under two conditions that differed in five respects at once: TSSI retained, WBC retained, no rounding, brand codebook fitted on the full cohort, and SMOTE oversampling [16]. In the first condition SMOTE was applied to the whole training partition **before** the grid-search cross-validation, so synthetic minority samples could fall in inner validation folds. This contrast shows the combined effect of these choices. It cannot attribute the effect to any single component, and its tuned settings were not used elsewhere.

### 2.9 Comparison with the Wang score

The published integer points were applied to every patient without refitting (Table 2 of Wang et al., p. 4) [1]:
- `Diabetes` 1;
- `Previous PCI` 3;
- `Initial diagnosis-AMI` 1;
- `CKD90` (recorded eGFR < 90) 1;
- `3-vessel disease` 1;
- 2 × `No.of stents per lesion` (integer 1–4);
- `PES` (sirolimus-eluting, Section 2.3) 1;
- `No postdilation` 4.

The post-dilation coding was chosen on documented grounds: the file's column names, and Wang et al.'s abstract, Discussion, positive coefficient and +4 weight, all of which treat absence of post-dilation as a risk factor. The contradictory Table 1 label and Table 2 hazard ratio are reported in Section 4.4 and queried.

We report ROC-AUC, AP and observed event rates in Wang et al.'s categories (≤ 7, 8–9, ≥ 10). This analysis evaluates the frozen score as a ranking reference against the same observed binary labels used for the machine-learning models. It is not external validation, because these are the score's derivation patients, and it does not reproduce the original Cox analysis or Wang et al.'s c-statistic. No probability transformation of the score was applied.

### 2.10 Exploratory association analyses

These analyses used the full cohort and did not inform the prediction matrix.
- **Continuous variables:** compared with Welch's t-test when |skewness| ≤ 1 and excess kurtosis ≤ 3, and with the Mann–Whitney U test otherwise.
- **Binary variables:** χ² test, or Fisher's exact test when any expected count was below 5. Tests were two-sided.
- **Multiplicity:** controlled with the Benjamini–Hochberg procedure within variable families [17].
- **Multivariable model:** an unweighted logistic regression with 13 covariates, one representative per collinear block chosen after the univariate screen, with continuous covariates standardised per standard deviation. Wald 95% intervals are reported. Because covariates were chosen after screening on the same data, intervals and P values are descriptive. Events per covariate were about 7.1.
- **Sensitivity analyses:** Firth-penalised estimates and a 2,000-replicate bootstrap.

These are odds ratios for the recorded label during variable follow-up, not hazard ratios, and they are not causal effects. [AUTHOR CONFIRMATION PENDING: results of reanalysis R2 (recalculated eGFR; clopidogrel removed or reported separately) if performed.]

### 2.11 Exploratory model attribution

Attribution analyses used one stratified 70/30 split (seed 42): models were fitted on 3,629 patients (64 events), and 1,556 held-out patients (28 events) were explained. This split, the model constructors and the purpose differ from the primary benchmark. The attribution results reported in the current draft came from a run that did not apply the laboratory rounding and that fitted the stent-brand codebook on all rows. The notebooks have been corrected to match the benchmark's cleaning and are to be re-run. [AUTHOR CONFIRMATION PENDING: replace Section 3.7 and Table S5 with the re-run attribution results.] The analyses were:
- **Mutual information** (scikit-learn, k = 3 neighbours) on the training rows.
- **Repeated forward feature selection:** 8 of 80 features selected in each of 8 resamples (seeds 42–49), 3-fold inner CV, AP scoring, local TabPFN with one estimator. A planned 10-resample run was not completed.
- **Partial dependence** on the training rows.
- **Shapley values** for all 1,556 held-out patients (shapiq, baseline imputer, budget 256, training data as background) [18].

Shapley values were planned for the hosted thinking model. The hosted service returned a daily-quota error (HTTP 429) after about 450 patients. The computation was then **restarted for all 1,556 patients** with a local TabPFN model, and only these local values are reported. The local models used for selection, partial dependence and Shapley values were built with the default constructor rather than the explicit v3.5 selector. [AUTHOR CONFIRMATION PENDING: implementation check C1, resolved checkpoint (v3 or v3.5) for these local models; the run log shows the v3.5 weights being loaded at the fallback step.] [AUTHOR CONFIRMATION PENDING: implementation check C2, output scale of the Shapley values; magnitudes above 1 indicate a non-probability scale.]

A rank-based consensus combined these three signals: mutual-information rank, selection frequency and mean |SHAP| rank. **These attributions describe the local model fitted on the 70% split. They are not explanations of the hosted model or of the benchmark arms, and they are not causal or biological evidence.** Agreement between correlated encodings of the same construct (for example `eGFR`, `CKD5`, `CKD90`) is not independent confirmation.

### 2.12 Reporting

Reporting follows TRIPOD+AI where applicable [19]. Items specific to model development for clinical use are not applicable because no model is proposed for clinical use.

---

## 3. Results

### 3.1 Cohort

Table 1 summarises characteristics by outcome. Of 5,185 patients, 92 had definite VLST (1.77%). Events occurred a median of 736 days after PCI (range 380–1,575). Event-free patients were followed for 1,241–1,605 days (median 1,419).

Several counts in the file differ from Wang et al.'s Table 1:
- previous PCI among cases: 10 (10.87%) vs 13 (14.13%);
- AMI as admitting diagnosis: 3,095 controls and 65 cases vs 3,046 and 72;
- HbA1c: the file gives 5.80 ± 1.61 in controls and 6.27 ± 1.53 in cases, the reverse of the published columns.

The file's HbA1c values are consistent with diabetes status in both groups (mean 7.61 in patients with diabetes in each group). [AUTHOR CONFIRMATION PENDING: Please confirm that the public file is the analysis file of Wang et al. and explain the differences in previous PCI, admitting diagnosis and HbA1c.]

### 3.2 Data-provenance audit

Table 2 lists the findings.

**Row order.** The file is ordered by outcome: all 92 cases are the last 92 rows.

**Number formatting.**
- Control laboratory values are stored with three decimal places in the raw text; case values have zero to two.
- After parsing, integer-valued WBC occurs in 3,780 of 5,093 controls (74.2%) and in no cases.
- `LV` is integer in both groups, but its terminal digits differ: even digits in 65.1% of controls vs 47.8% of cases, and the digit 0 in 1.3% of controls.
- The precision probe, which used no clinical magnitudes, reached AP 0.427 and ROC-AUC 0.952. The strongest single indicator was integer-valued WBC (ROC-AUC 0.871).

**eGFR.**
- Mean creatinine was 72.5 µmol/L in controls and 72.4 µmol/L in cases, yet recorded eGFR averaged 120.0 and 95.9 mL/min/1.73 m².
- CKD-EPI 2009 eGFR recalculated from creatinine, age and sex reproduced the recorded value within 1 unit in 2,243 of 5,093 controls (44.0%) and in 50 of 92 cases (54.3%).
- In most of the remaining controls, the recorded eGFR was 1.3–1.9 times the recalculated value.
- With recalculated eGFR, means were 91.9 (controls) and 91.5 (cases), and eGFR < 90 occurred in 35.6% and 40.2%. With recorded eGFR these proportions were 16.9% and 34.8%.
- Because `CKD5`, `CKD60` and `CKD90` are derived from recorded eGFR, they carry the same difference.

**Medications.** Among controls, `DAPT = 1` always coincided with aspirin plus a P2Y12 inhibitor (2,260 of 2,260). Among cases it did in 27 of 35.

### 3.3 Discrimination in the primary benchmark

Table 3 and Figure 1 show pooled out-of-fold performance.
- Hosted TabPFN thinking v3.5 had the highest AP, 0.921 (95% interval 0.878–0.961), with ROC-AUC 0.996.
- It was followed by local TabPFN v3.5 (AP 0.896; 0.845–0.943), hosted thinking v3 (0.832) and local v3 (0.715).
- The best conventional models were XGBoost (0.632; 0.533–0.725) and LightGBM (0.627; 0.531–0.720).
- Logistic regression had the lowest AP (0.260).

Paired AP differences were 0.294 (0.207–0.381) for hosted v3.5 minus LightGBM and 0.269 (0.182–0.356) for local v3.5 minus LightGBM. None of the 2,000 bootstrap resamples gave a difference ≤ 0.

Across outer folds, hosted v3.5 AP ranged from 0.880 to 0.986 and local v3.5 from 0.824 to 0.959. Local v3.5 exceeded hosted v3.5 in fold 1 (0.936 vs 0.880).

**Operating point** (Table 4). At inner-fold F1 thresholds (mean 0.318 ± 0.059), hosted v3.5 identified 77 of 92 events (sensitivity 0.837) with 10 false positives (precision 0.885). The corresponding values were 72 of 92 events (precision 0.868) for local v3.5 and 54 of 92 (precision 0.593) for LightGBM.

**Calibration** (Figure 2). Brier scores were 0.0047 (hosted v3.5) and 0.0048 (local v3.5), against a null Brier score of 0.0174. Logistic regression, whose class-weighted probabilities are not calibrated to the prevalence, had a Brier score of 0.0788. Expected calibration error was below 0.01 for every arm except logistic regression. With 8 quantile bins at this prevalence, most bins contain near-zero risk, so these values say little about calibration in the higher-risk range.

**These results describe discrimination of the recorded label using a matrix that contains follow-up-period medication fields and recorded eGFR.** [AUTHOR CONFIRMATION PENDING: insert R1 results: the same benchmark with the four medication fields removed and eGFR, CKD5, CKD60 and CKD90 recalculated from creatinine.]

### 3.4 Sensitivity to recording precision and data handling

Precision equalisation (Table S4) lowered AP as follows:
- local TabPFN v3.5: 0.896 to 0.844;
- XGBoost: 0.632 to 0.581;
- LightGBM: 0.627 to 0.571;
- CatBoost: 0.571 to 0.465;
- random forest: 0.351 to 0.255;
- logistic regression: 0.260 to 0.254.

Local v3 changed from 0.715 to 0.731. Hosted arms were not evaluated, and all the equalised values remain far above the prevalence.

In the bundled 70/30 contrast (Table S3), hold-out AP was 0.30–0.61 higher in every model when TSSI, WBC, unrounded laboratory values, a full-cohort brand codebook and SMOTE were used together (for example, LightGBM 0.969 vs 0.668; logistic regression 0.913 vs 0.343). The individual contributions of these five changes cannot be separated.

### 3.5 Wang score

The frozen Wang score had ROC-AUC 0.801 and AP 0.103 (fold-wise, evaluation only: ROC-AUC 0.80 ± 0.06; AP 0.11 ± 0.05). Observed VLST rates were:
- 0.51% for scores ≤ 7 (16 of 3,135 patients);
- 2.22% for scores of 8–9 (35 of 1,577);
- 8.67% for scores ≥ 10 (41 of 473).

The low- and high-risk group sizes equal those Wang et al. published. Their intermediate group (1,837) is inconsistent with their own total, since 3,135 + 1,837 + 473 = 5,445 (Figure 3).

### 3.6 Exploratory associations

Table 1 gives univariate comparisons and Table 5 the 13-covariate model. In the multivariable model:
- higher odds of the recorded label were associated with previous PCI (OR 6.71; 95% CI 2.88–15.61), `LV` (1.83 per SD; 1.54–2.18) and WBC (1.97 per SD; 1.67–2.33);
- lower odds were associated with `1.1:1Post dilation` (0.15; 0.08–0.29), recorded eGFR (0.57 per SD; 0.45–0.72) and follow-up-period clopidogrel (0.48; 0.29–0.79).

Diabetes, HbA1c, number of diseased vessels, stents per lesion, total stent length, fibrinogen and `PES` had intervals including 1. Firth estimates were similar (for example, WBC 1.95 and eGFR 0.57).

The eGFR association is not interpretable as a renal association given Section 3.2. [AUTHOR CONFIRMATION PENDING: R2 estimate with recalculated eGFR.] The clopidogrel association concerns therapy recorded during follow-up and is subject to prescribing, adherence and reverse-timing effects.

### 3.7 Exploratory attributions (local TabPFN, 70/30 split)

- **Mutual information** ranked `CaI`, `LV`, `eGFR`, `LDL` and `HbA1c` highest.
- **Repeated forward selection** retained `CaI`, `LV` and `eGFR` in 8 of 8 resamples, `Cre` and `HbA1c` in 7, `Age` in 6, `LVEF` in 5 and `1.1:1Post dilation` in 4.
- **Mean |SHAP|** was largest for `eGFR`, `CaI`, `Cre` and `LV` (Table S5).

The variables ranked highest by the local model are those with unresolved definitions (`LV`, `CaI`) or documented recording differences (`eGFR`, `Cre`, `CaI`, Section 3.2). `CaI`, for example, ranked first in mutual information despite similar group means (37.4 vs 40.5) and a univariate P value of 0.051. Partial dependence of the local model's training-set predictions on binary features was small (for example, −0.0001 for `1.1:1Post dilation` and +0.0013 for previous PCI). Single-patient interaction analyses were exploratory and are not reported as findings.

---

## 4. Discussion

### 4.1 Principal findings

In the public Wang derivation file, TabPFN, especially the hosted v3.5 thinking arm, discriminated recorded VLST far better than conventional learners with default settings and far better than the published eight-item score. However, these results cannot be read as evidence that VLST is highly predictable after PCI. The label is not a fixed-horizon outcome. The prediction matrix contains antiplatelet-therapy fields recorded during follow-up. Most importantly, the file contains outcome-dependent recording differences, which a flexible learner can exploit without any clinical signal. A recorded eGFR that is not reproducible from creatinine in most controls, differing number formats, inconsistent derived fields in cases, and a precision probe reaching AP 0.43 on number formatting alone together make an artefactual contribution to the reported discrimination plausible. The size of that contribution is unknown.

Discrimination this high (ROC-AUC 0.996) for an event occurring one to four years after PCI, from data available around the index admission, would be unusual. It is a reason to examine data provenance before drawing clinical conclusions.

### 4.2 What the comparison does and does not show

The benchmark compared nine existing classifiers on identical outer folds. The ranking held for pooled AP and, with one exception, across folds. The comparison has several limitations:
- **Tuning budgets were not matched.** The hosted thinking arms optimised AP during fitting, the local TabPFN arms used defaults, and the conventional arms used untuned defaults. The classical models' performance is therefore a lower bound.
- **Only one fold assignment was used**, and the bootstrap intervals do not capture fitting variability.
- **TabPFN's advantage may partly reflect** greater ability to use fine-grained value structure, including formatting or derivation differences, rather than clinical patterns. Rounding residual decimals reduced local v3.5 AP by about 0.05, but this procedure removes clinical resolution as well and cannot show that all artefact was removed. The eGFR discrepancy, the medication coding and the LV digit pattern are not addressed by rounding at all.

Whether TabPFN retains an advantage once the predictor set is restricted to information available after PCI, and once eGFR is recalculated, is the central open question. [AUTHOR CONFIRMATION PENDING: R1.]

We found no published evidence about whether this public dataset was among the data used to develop TabPFN versions. The vendor reports do not list such datasets in the sources we examined, so the question is unresolved. We make no allegation of overlap.

### 4.3 The Wang score as a clinical reference

Evaluated without refitting on the same labels, the Wang score gave ROC-AUC 0.80 and AP 0.10, about 5.8 times the prevalence. It stratified observed risk from 0.5% to 8.7%. This is not evidence that the score is poor or that machine learning is superior, for three reasons:
- the score uses eight items, whereas the machine-learning models drew on about 80 recorded variables, including follow-up-period fields;
- the score was developed on these same patients, while the machine-learning predictions are out-of-fold within them;
- one score item, eGFR < 90, depends on the recorded eGFR whose case–control difference largely disappears on recalculation.

The numerical similarity between our ROC-AUC (0.80) and Wang et al.'s c-statistic (0.80) does not imply reproduction. Their c-statistic came from a Cox analysis whose time axis, censoring and handling of competing events are not reported. A fair comparison would give machine-learning models the same eight inputs, use a fixed-horizon outcome, and test on independent data.

### 4.4 Inconsistencies in the source publication

Wang et al.'s Table 2 lists no post-dilation with coefficient 1.93 and 4 risk points, but with a hazard ratio of 0.145. That hazard ratio equals exp(−1.93) and is incompatible with the sign of the coefficient. Their Table 1 labels as "no post-dilation" a group that is 15.2% of cases and 49.0% of controls. In the public file, that group is stored as `1.1:1Post dilation`. These points, the arithmetic of the risk-group sizes, and the HbA1c and count differences (Section 3.1) do not alter the score's integer weights. They do show that the published tables and the public file must be reconciled before either is used as a reference standard.

### 4.5 Clinical associations

The associations below are exploratory, were estimated under low events per covariate, refer to the recorded label during variable follow-up, and are not causal.

- **Previous PCI** was associated with higher odds (OR 6.7). This direction agrees with Wang et al.'s model and with the inclusion of previous PCI in the Dangas score [1,4]. Because the data are Wang et al.'s derivation data, the agreement is not independent confirmation. Prior PCI may mark more extensive coronary disease or prior stenting, but this analysis cannot distinguish these explanations.
- **Post-dilation** was associated with lower odds (OR 0.15), consistent with the mechanistic role of stent under-expansion and malapposition in very late thrombosis found in intracoronary-imaging studies [20]. Large registry data have not shown a consistent association between post-dilation and stent thrombosis [21]. Post-dilation was chosen by operators, not randomised, and Wang et al. note that procedural variables reflected the operator's subjective assessment (p. 6) [1]. The association should not be read as evidence that post-dilation prevents VLST. Its direction also depends on resolving the coding question in Section 4.4.
- **Follow-up-period clopidogrel** was associated with lower odds (OR 0.48). The direction agrees with randomised evidence that extending P2Y12 inhibition beyond one year reduces stent thrombosis while increasing bleeding [22]. Here, however, the medication fields describe therapy during follow-up at an undocumented time, and their coding differs between cases and controls. The estimate therefore cannot be interpreted as a treatment effect or used to guide DAPT duration.
- **WBC** was strongly associated with the label (OR 1.97 per SD). Wang et al. excluded leukocyte count from their score because elevated counts could reflect infection or other conditions [1]. In this file, WBC also carries the strongest recording-precision difference between cases and controls (Section 3.2). Inflammatory, incidental and artefactual explanations therefore cannot be separated. In ATLAS ACS 2–TIMI 51, higher WBC was associated with early, not very late, stent thrombosis [23].
- **Recorded eGFR** was associated with lower odds, but the case–control difference in recorded eGFR largely disappears when eGFR is recalculated from creatinine (Section 3.2). We therefore make no renal interpretation. Renal disease has been associated with VLST in other cohorts [5], so a true association remains plausible but is not demonstrated here.
- **`LV`** was associated with higher odds (OR 1.83 per SD) and ranked highly in model attributions. Because its definition is unknown and its digit pattern differs between groups, no physiological interpretation is offered.
- **No association was found for** diabetes, stents per lesion, three-vessel disease and `PES` (sirolimus-eluting stent, pending confirmation), all of which are components of Wang et al.'s score. These variables did not have adjusted intervals excluding 1 in our 13-covariate logistic model. Differences in model form, outcome handling, covariates and the small number of events may explain this, and a non-significant estimate does not imply clinical irrelevance.

### 4.6 Implications

For investigators using this public file:
- Follow-up time, white-cell count and the medication fields should not be used as baseline predictors.
- Recorded eGFR and its derived CKD fields should be recalculated or treated with caution.
- Very high classifier performance should be checked against probes of recording artefacts.

For VLST prediction, a clinically useful post-PCI model would require:
- a defined prediction time, for example discharge;
- predictors restricted to information available then;
- an outcome window of year 1 to year 4 with complete follow-up, or a survival analysis that handles censoring and death;
- external validation;
- an assessment of calibration and of clinical utility.

Any role in decisions about the duration of antiplatelet therapy would further require evidence on the balance of ischaemic and bleeding outcomes under alternative strategies [22,24].

---

## 5. Limitations

1. **Single public file and outcome-dependent recording.** All analyses use one file from one centre. Recording and formatting differ systematically between cases and controls, and their origin is unknown. Our cleaning rules and sensitivity analyses reduce but cannot eliminate the possibility that discrimination reflects how the data were assembled rather than patient risk.
2. **Outcome definition.** The label is VLST recorded during variable follow-up. It is not a fixed 1–4-year outcome, and deaths after discharge are undocumented. The intended post-PCI application is unvalidated.
3. **Predictor timing.** Antiplatelet-therapy fields are recorded during follow-up and were included in the primary benchmark. Measurement times for laboratory variables, and the definitions of `LV` and `CaI`, are unconfirmed.
4. **Evaluation design.** One nested cross-validation with threshold-only inner selection; unmatched optimisation budgets; bootstrap intervals conditional on fitted predictions; no external or temporal validation; hosted models evaluated through a service whose internal procedures are not documented.
5. **Association analyses.** About 7 events per covariate; covariates selected after screening on the same data; odds ratios for a variable-follow-up label; possible confounding by indication, especially for post-dilation and antiplatelet therapy.
6. **Attribution analyses.** Performed on a different split and with a different local model from the benchmark; checkpoint and output scale pending verification. Attributions describe that model, not biology.
7. **Wang comparison.** Performed on the score's own derivation patients against a binary label. It does not reproduce or validate the original Cox model.
8. **Angiographic confirmation.** Definite VLST required angiographic confirmation, so events not investigated by angiography may have been missed [1].

---

## 6. Conclusion

In the public derivation file of Wang et al., TabPFN classifiers discriminated recorded VLST much better than conventional learners with default settings and the published eight-item score. However, the file contains outcome-dependent recording differences: eGFR inconsistent with creatinine in most controls, differing number formats, and inconsistent medication coding in cases. In addition, the prediction matrix contained follow-up-period treatment data and the label was not a fixed-horizon outcome. The reported discrimination therefore cannot be taken as clinical predictive performance. The work contributes an internal benchmark of existing models, a provenance audit that should inform any further use of this dataset, and a qualified reference evaluation of a published score. A post-PCI model for VLST in years 1–4 remains to be developed and validated with clearly timed predictors, a fully observed outcome window and independent data.

---

## Declarations

- **Ethics.** Secondary analysis of publicly available, de-identified data released under [AUTHOR CONFIRMATION PENDING: confirm licence on the Figshare record; drafts state CC0 1.0]. The original study received ethics approval (No. 2013-256) and written informed consent [1]. [AUTHOR QUERY: local ethics determination.]
- **Data availability.** Source data: Figshare, doi:10.6084/m9.figshare.7409606 [7]. Analysis code: [AUTHOR QUERY: repository URL and commit].
- **Competing interests.** [AUTHOR QUERY]
- **Funding.** [AUTHOR QUERY]
- **Author contributions.** [AUTHOR QUERY: CRediT statement]
- **Use of AI tools.** [AUTHOR QUERY: disclose any AI assistance with analysis or writing, as required by the target journal]
- **Acknowledgements.** We thank Wang et al. for making their data public.

---

## Tables

**Table 1. Characteristics of patients with and without recorded VLST in the public derivation file.** Columns: No VLST (n = 5,093) | VLST (n = 92) | test | P. Continuous variables are mean (SD); binary variables are n (%). Rows as in the current Table 1, plus a row for CKD-EPI 2009 eGFR recalculated from creatinine, age and sex (91.9 vs 91.5). Footnotes:
- (a) `LV` and `CaI` are dataset labels pending confirmation;
- (b) medication rows are follow-up-period fields;
- (c) previous PCI, admitting AMI and HbA1c differ from Wang et al.'s Table 1;
- (d) TSSI is not shown as a characteristic; its distribution is given in Section 3.1.

**Table 2. Data-provenance audit of the public file.**

| Check | Controls (n = 5,093) | Cases (n = 92) |
|---|---|---|
| Row position | Rows 1–5,093 | Rows 5,094–5,185 |
| Raw-text decimals, laboratory fields | 3 in all values | 0–2 |
| Integer-valued WBC | 3,780 (74.2%) | 0 |
| `LV` even terminal digit | 65.1% | 47.8% |
| Recorded eGFR, mean | 120.0 | 95.9 |
| CKD-EPI 2009 eGFR from creatinine, mean | 91.9 | 91.5 |
| Recorded eGFR within ±1 of recalculated | 2,243 (44.0%) | 50 (54.3%) |
| eGFR < 90, recorded / recalculated | 16.9% / 35.6% | 34.8% / 40.2% |
| `DAPT = 1` consistent with aspirin + P2Y12 field | 2,260 / 2,260 | 27 / 35 |
| Previous PCI vs Wang Table 1 | 94 vs 94 | 10 vs 13 |
| AMI as admitting diagnosis vs Wang Table 1 | 3,095 vs 3,046 | 65 vs 72 |
| Precision probe (243 grid indicators, no magnitudes) | AP 0.427; ROC-AUC 0.952 | |

**Table 3. Pooled out-of-fold discrimination and overall accuracy in the primary nested cross-validation (5 outer × 4 inner folds; 92 events).** AP and ROC-AUC with percentile 95% bootstrap intervals (2,000 stratified resamples of stored predictions; no refitting); Brier score (null 0.0174).

| Model | Interface / checkpoint | AP (95% interval) | ROC-AUC (95% interval) | Brier |
|---|---|---|---|---|
| TabPFN thinking v3.5 | Hosted, `v3.5_default`, effort high, AP-targeted | 0.921 (0.878–0.961) | 0.996 (0.993–0.999) | 0.0047 |
| TabPFN v3.5 | Local, `tabpfn-v3.5-20260909` | 0.896 (0.845–0.943) | 0.992 (0.983–0.998) | 0.0048 |
| TabPFN thinking v3 | Hosted, `v3_default` | 0.832 (0.763–0.895) | 0.983 (0.971–0.994) | 0.0066 |
| TabPFN v3 | Local, `v3_default` | 0.715 (0.626–0.807) | 0.973 (0.955–0.988) | 0.0099 |
| XGBoost | Defaults, `scale_pos_weight` | 0.632 (0.533–0.725) | 0.937 (0.897–0.970) | 0.0100 |
| LightGBM | Defaults, balanced | 0.627 (0.531–0.720) | 0.944 (0.914–0.970) | 0.0106 |
| CatBoost | Defaults, balanced | 0.571 (0.475–0.673) | 0.940 (0.914–0.963) | 0.0108 |
| Random forest | Defaults, balanced | 0.351 (0.266–0.463) | 0.888 (0.841–0.930) | 0.0150 |
| Logistic regression | Defaults, balanced | 0.260 (0.183–0.359) | 0.865 (0.822–0.904) | 0.0788 |
| Wang integer score (frozen; reference, not cross-validated) | — | 0.103 | 0.801 | — |

**Table 4. Operating point at inner-fold F1 thresholds.** Threshold (mean ± SD), sensitivity, precision, F1, and TN/FP/FN/TP for each arm. For example: hosted v3.5, 0.318 ± 0.059, 0.837, 0.885, 0.860, 5,083/10/15/77; local v3.5, 0.386 ± 0.061, 0.783, 0.868, 0.823, 5,082/11/20/72; LightGBM, 0.076 ± 0.037, 0.587, 0.593, 0.590, 5,056/37/38/54.

**Table 5. Exploratory multivariable logistic regression for the recorded VLST label (13 covariates; continuous per SD; Wald 95% CI).** Values as in the current Table 2, with footnotes:
- eGFR is the recorded value (see Table 2); [R2 pending];
- clopidogrel is a follow-up-period field;
- `LV` is undefined;
- the `1.1:1Post dilation` coding is pending confirmation;
- covariates were selected after univariate screening.

**Supplementary tables.**
- S1: variable dictionary with availability.
- S2: univariate screen with false-discovery control.
- S3: bundled 70/30 data-handling contrast.
- S4: precision probe and precision equalisation.
- S5: local TabPFN attribution ranks (mutual information, selection frequency of 8 resamples, mean |SHAP|).
- S6: harmonised feature-screen table.
- S7: software and reproducibility record.

## Figure captions

**Figure 1. Precision–recall (left) and ROC (right) curves of pooled out-of-fold predictions for the nine benchmark arms.** The dotted line marks the prevalence (0.0177), the AP of a non-informative classifier. Curves describe discrimination of the recorded label using a matrix that includes follow-up-period medication fields and recorded eGFR. [Visual inspection pending.]

**Figure 2. Calibration of pooled out-of-fold probabilities (8 quantile bins).** Most bins contain predicted risks near zero. Logistic-regression probabilities are inflated by class weighting. [Visual inspection pending.]

**Figure 3. Observed VLST rate by frozen Wang score.** Bars give the observed rate at each score value; numbers give patients per value; the dashed line marks the prevalence. Wang et al.'s risk categories are ≤ 7, 8–9 and ≥ 10. [Visual inspection pending.]

**Figure 4. Recorded versus recalculated eGFR, by outcome.** Recalculated values use CKD-EPI 2009 with recorded creatinine, age and sex; the line of identity is shown. [To be generated.]

**Supplementary figures.**
- S1–S2: univariate screens.
- S3: univariate versus adjusted odds ratios.
- S4: bundled-contrast PR/ROC curves and confusion matrices.
- S5: Shapley beeswarm, local TabPFN fitted on the 70% split, 1,556 held-out patients (output scale pending C2).
- S6: partial dependence, training rows.

---

## References

1. Wang X, Chen X, Tian T, You H, Li Y, Wu M, Du X, Cai H, Zheng Y, Du J. A scoring system to predict the occurrence of very late stent thrombosis following percutaneous coronary intervention for acute coronary syndrome. *Sci Rep*. 2020;10:6378. doi:10.1038/s41598-020-63455-0
2. Cutlip DE, Windecker S, Mehran R, et al. Clinical end points in coronary stent trials: a case for standardized definitions. *Circulation*. 2007;115:2344–2351. doi:10.1161/CIRCULATIONAHA.106.685313
3. Garcia-Garcia HM, McFadden EP, Farb A, et al. Standardized end point definitions for coronary intervention trials: the Academic Research Consortium-2 consensus document. *Eur Heart J*. 2018;39:2192–2207. doi:10.1093/eurheartj/ehy223
4. Dangas GD, Claessen BE, Mehran R, et al. Development and validation of a stent thrombosis risk score in patients with acute coronary syndromes. *JACC Cardiovasc Interv*. 2012;5:1097–1105. doi:10.1016/j.jcin.2012.07.012
5. Baran KW, Lasala JM, Cox DA, et al. A clinical risk score for the prediction of very late stent thrombosis in drug eluting stent patients. *EuroIntervention*. 2011;6:949–954. PMID:21330242 [REFERENCE TO VERIFY: full author list]
6. Godschalk TC, Byrne RA, Adriaenssens T, et al. A clinical risk score to identify patients at high risk of very late stent thrombosis. *J Interv Cardiol*. 2018;31:159–169. doi:10.1111/joic.12494 [REFERENCE TO VERIFY: author list]
7. Wang X, et al. Dataset for: A scoring system to predict the occurrence of very late stent thrombosis following PCI for ACS. Figshare. doi:10.6084/m9.figshare.7409606 [REFERENCE TO VERIFY: record title, authors, licence]
8. Zaka A, Mutahar D, Gorcilov J, et al. Machine learning approaches for risk prediction after percutaneous coronary intervention: a systematic review and meta-analysis. *Eur Heart J Digit Health*. 2025;6:23–44. doi:10.1093/ehjdh/ztae074
9. Nayebirad S, Hassanzadeh A, Vahdani AM, et al. Comparison of machine learning models with conventional statistical methods for prediction of percutaneous coronary intervention outcomes: a systematic review and meta-analysis. *BMC Cardiovasc Disord*. 2025;25:310. doi:10.1186/s12872-025-04746-0
10. Hollmann N, Müller S, Purucker L, et al. Accurate predictions on small data with a tabular foundation model. *Nature*. 2025;637:319–326. doi:10.1038/s41586-024-08328-6
11. Jäger B, Erickson N, Grinsztajn L, et al. TabPFN-3.5: Technical Report. arXiv:2609.17895; 2026.
12. Prior Labs. TabPFN documentation: TabPFN-3.5 release notes; Thinking mode; Selecting a model version. https://docs.priorlabs.ai (accessed 31 October 2026).
13. Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation, detection, and avoidance. *ACM Trans Knowl Discov Data*. 2012;6(4):15.
14. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS One*. 2015;10:e0118432. doi:10.1371/journal.pone.0118432
15. Levey AS, Stevens LA, Schmid CH, et al. A new equation to estimate glomerular filtration rate. *Ann Intern Med*. 2009;150:604–612.
16. Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority over-sampling technique. *J Artif Intell Res*. 2002;16:321–357.
17. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc Series B*. 1995;57:289–300.
18. Lundberg SM, Lee S-I. A unified approach to interpreting model predictions. *Adv Neural Inf Process Syst*. 2017;30. [REFERENCE TO VERIFY: shapiq software citation]
19. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*. 2024;385:e078378.
20. Adriaenssens T, Joner M, Godschalk TC, et al. Optical coherence tomography findings in patients with coronary stent thrombosis: a report of the PRESTIGE Consortium. *Circulation*. 2017;136:1007–1021. doi:10.1161/CIRCULATIONAHA.117.026788
21. Fröbert O, Sarno G, James SK, Saleh N, Lagerqvist B. Effect of stent inflation pressure and post-dilatation on the outcome of coronary artery intervention: a report of more than 90,000 stent implantations. *PLoS One*. 2013;8:e56348. doi:10.1371/journal.pone.0056348
22. Mauri L, Kereiakes DJ, Yeh RW, et al. Twelve or 30 months of dual antiplatelet therapy after drug-eluting stents. *N Engl J Med*. 2014;371:2155–2166. doi:10.1056/NEJMoa1409312
23. Chi G, AlKhalfan F, Lee JJ, et al. Factors associated with early, late, and very late stent thrombosis among patients with acute coronary syndrome undergoing coronary stent placement: analysis from the ATLAS ACS 2–TIMI 51 trial. *Front Cardiovasc Med*. 2024;10:1269011. doi:10.3389/fcvm.2023.1269011
24. Yeh RW, Secemsky EA, Kereiakes DJ, et al. Development and validation of a prediction rule for benefit and harm of dual antiplatelet therapy beyond 1 year after percutaneous coronary intervention. *JAMA*. 2016;315:1735–1749.

*Removed from the earlier Discussion draft, because the statements they supported were removed:* SCAI consensus (Klein 2023), Saeed 2024, Kakimoto 2026 (REAL-ST), Richardson-Lobbedez 2008, Seko 2019, Brodie 2012, van Werkum 2009. Reinstate van Werkum 2009 and Brodie 2012 if a broader literature paragraph on stent-thrombosis predictors is wanted.
