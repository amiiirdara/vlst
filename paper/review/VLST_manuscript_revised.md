# Discrimination of very late stent thrombosis in a public acute coronary syndrome PCI cohort: a machine-learning benchmark under leakage and recording-artefact controls, with a data-provenance audit and a comparison with a published clinical score

*Revised integrated draft, 20 November 2026. All analysis results are filled in, and the figures and tables were regenerated from the stored run outputs and inspected on 20 November 2026 (files in `paper_results/`). Remaining markers are `[AUTHOR QUERY]` (administrative items) and `[REFERENCE TO VERIFY]` / `[REFERENCE TO ADD]`. The original data authors could not be contacted, so variable definitions that only they could confirm are reported as unconfirmed. The revision history and the reasoning for each analysis choice are in `VLST_revision_process_and_defence.md`.*

**Authors.** [AUTHOR QUERY: names, affiliations, ORCID, corresponding author]

**Running title.** VLST discrimination and data provenance in a public PCI cohort

---

## Abstract

**Background.** Very late stent thrombosis (VLST), definite stent thrombosis more than one year after implantation, is rare and hard to predict. Wang et al. derived an eight-item integer score for VLST in 5,185 patients with acute coronary syndrome (ACS) and deposited the derivation data publicly. We used this file for three purposes:
- to compare machine-learning classifiers;
- to measure how leakage and the file's recording properties affect apparent discrimination;
- to evaluate the published score on the same labels.

**Methods.** This was a secondary analysis of the public derivation file: a single centre in China, PCI between January 2014 and June 2015, 92 VLST events (prevalence 1.77%).
- **Recording audit.** Before modelling, we audited the file for outcome-dependent recording differences.
- **Leakage and recording-artefact controls.** We then applied them in stages:
  - exclusion of follow-up time and white-cell count, rounding of four laboratory fields, and a stent-brand codebook fitted within training folds (unrestricted analysis);
  - removal of four follow-up-period antiplatelet fields and recalculation of eGFR from creatinine (R1);
  - removal of a troponin field with a repeated case-specific value, and binning of `LV` to 5 mm (R3, the main analysis).
- **Models.** Five conventional learners, tuned by inner random search, were compared with local TabPFN v3 and v3.5 and with hosted TabPFN v3.5 in "thinking" mode targeting average precision.
- **Evaluation.** All arms shared one stratified nested cross-validation (5 outer × 4 inner folds). The primary metric was pooled out-of-fold average precision (AP).
- **Reference score.** Wang's integer score was evaluated without refitting.
- **Exploratory analyses.** Association and attribution analyses were exploratory.

**Results.**
- **Label.** The label was not a fixed-horizon outcome. Events occurred on days 380–1,575, and event-free patients were followed for 1,241–1,605 days.
- **Recording audit.** The audit found outcome-dependent recording differences:
  - Recorded eGFR was not reproducible from creatinine in 56.0% of controls, against 45.7% of cases. Recalculated means were 91.9 and 91.5 mL/min/1.73 m², against recorded means of 120.0 and 95.9.
  - Number formatting differed between groups, and medication coding was inconsistent in cases.
  - The troponin field held the value 0.034 in 21.7% of cases but in 0.2% of controls.
  - Formatting indicators alone reached AP 0.43.
- **Effect of the controls.** Each stage lowered discrimination. For local TabPFN v3.5, AP fell from 0.896 (unrestricted) to 0.668 (R1) to 0.465 (R3; 95% bootstrap interval 0.372–0.567).
- **Main analysis (R3).** Hosted TabPFN thinking v3.5 reached AP 0.475 (0.381–0.583; ROC-AUC 0.957), and the best tuned conventional model, XGBoost, 0.266 (0.191–0.358). The paired difference between local TabPFN v3.5 and XGBoost was 0.199 (0.105–0.292).
- **Wang score.** Evaluated on its own derivation patients, the frozen score had ROC-AUC 0.801 and AP 0.103.
- **Association with eGFR.** With recalculated eGFR, the eGFR odds ratio in the exploratory logistic model changed from 0.57 to 1.18 per SD (95% CI 0.97–1.44).

**Conclusions.** Most of the near-perfect discrimination first observed in this file came from information recorded after the index admission and from outcome-dependent recording differences. Under the strictest controls, TabPFN still ranked recorded VLST better than tuned conventional learners, with AP about 27 times the prevalence. Residual recording differences, an undefined variable and a label that is not a fixed-horizon outcome make this an upper bound for this file, not clinical predictive performance. A post-PCI model for VLST in years 1–4 remains to be developed and validated in independent data.

**Keywords.** very late stent thrombosis; acute coronary syndrome; percutaneous coronary intervention; machine learning; TabPFN; data leakage; data provenance; prediction-model evaluation

---

## 1. Introduction

Stent thrombosis occurring more than one year after implantation (VLST) is uncommon, but it often presents as acute myocardial infarction and carries substantial mortality [1–3]. Several clinical scores have been proposed for late or very late stent thrombosis:
- the late stent thrombosis score of Dangas et al. (c-statistic 0.66 for VLST, as reported by Wang et al.) [4];
- a score from the ARRIVE registries [5];
- a case-control-derived score [6];
- the eight-item score of Wang et al. [1].

Wang et al. developed their score with stepwise multivariable Cox regression in 5,185 consecutive patients with ACS treated with PCI. They reported c-statistics of 0.80 in derivation and 0.82 in an external cohort of 2,058 patients [1]. The derivation data were made publicly available [7].

Machine-learning methods have been applied to outcomes after PCI. Systematic reviews report variable and often modest gains over conventional models, frequent high risk of bias and little external validation [8,9]. Most studies have addressed mortality, bleeding or composite events rather than definite VLST. Tabular foundation models such as TabPFN make predictions by in-context learning from the training rows and need no per-dataset gradient training [10]. Later releases (v3, v3.5) and a hosted "thinking" mode, which spends extra fit-time compute against a user-chosen metric, are documented in technical reports and vendor documentation rather than in peer-reviewed evaluations [11,12].

Rare outcomes in small public datasets carry known risks for classifier evaluation. These include leakage of post-outcome or follow-up information into predictors [13], differences in how cases and controls were recorded, and instability of precision-based metrics [14]. Flexible models are particularly able to exploit such differences, so a high score from them calls for a check of the data before any clinical reading. A binary classifier can address a fixed-horizon outcome only if that outcome is fully observed for every patient.

Our intended clinical use is prediction at hospital discharge after index PCI, using information available then, of VLST occurring between about one and four years after PCI. This study did **not** establish such a model. Its aims were:
1. to audit the public Wang derivation file for recording properties that could produce apparent discrimination unrelated to clinical risk;
2. to compare existing machine-learning classifiers for discrimination of the recorded VLST label under a shared nested cross-validation, as leakage and recording-artefact controls were added step by step;
3. to evaluate Wang's published integer score, without refitting, as a clinical reference on the same labels;
4. to describe exploratory clinical associations and model attributions as hypothesis-generating findings.

---

## 2. Methods

### 2.1 Data source and population

This was a retrospective secondary analysis of the publicly deposited derivation cohort of Wang et al. [1,7]. No new participants were enrolled. The original study recruited consecutive patients aged 18 years or older with ACS who underwent PCI at the First Hospital of Jilin University, China, between 1 January 2014 and 1 June 2015, and who gave written informed consent (Wang et al., Methods, p. 6) [1].

The original exclusions were:
- expected survival under one year;
- severe organ failure;
- malignancy;
- planned major surgery within one year requiring antiplatelet interruption;
- death before discharge;
- refusal of consent or follow-up interviews;
- loss to follow-up after discharge.

Of 6,038 eligible patients, 5,185 were enrolled (p. 2) [1]. All patients received newer- or second-generation drug-eluting stents and dual antiplatelet therapy (DAPT) for at least one year (p. 6) [1].

The file contains 5,185 rows, one per patient, with 92 VLST events (prevalence 1.77%). It was analysed exactly as deposited; its SHA-256 hash (`0b8cf3da…d93e`) is identical in every run manifest. Wang et al.'s external validation cohort (n = 2,058) is not publicly available and was not analysed. Ethics approval (No. 2013-256) and registration (NCT03491891) belong to the original study [1]. [AUTHOR QUERY: does the authors' institution require its own ethics determination for this secondary analysis of public, de-identified data?] The present analysis was not preregistered.

### 2.2 Outcome, time origin and prediction time

**Outcome as defined by the source.** Definite VLST under the Academic Research Consortium 2007 criteria [2]: stent thrombosis more than one year after implantation, confirmed angiographically and adjudicated by two interventional cardiologists, with a third if needed (p. 6) [1]. We used the file's binary `Stent thrombosis` column.

**Time origin.** Index PCI.

**Follow-up information in the file.** The column `Time since stent implantation` (TSSI) ranged from 380 to 1,575 days among events (median 736) and from 1,241 to 1,605 days among non-events (median 1,419). Wang et al. reported a median follow-up of 1,502 days and a median time to VLST of 697 days (p. 2) [1]; the TSSI column does not reproduce these medians. Because the file is unmodified, the difference lies between the published paper and its own deposited data. How TSSI was constructed could not be confirmed with the original authors. TSSI was excluded from every prediction matrix (Section 2.5), so this uncertainty does not affect the models. It affects only how the label is described below.

**What the label represents.** The label records VLST at any time during a follow-up of variable length:
- 12 of the 92 events occurred after day 1,461 (four years);
- 3,090 of 5,093 event-free patients (60.7%) were followed for fewer than 1,461 days, so the absence of an event is not established for a four-year window;
- no event-free patient was followed for fewer than 1,241 days.

The file contains no information on deaths after discharge. Either no patient died during follow-up, which is unlikely in an ACS cohort over four years, or deaths were excluded or recorded with administrative follow-up. This could not be confirmed. It does not change the data our models were scored on, but it limits what the label means: a patient who died without VLST is either absent or counted as event-free.

**Intended versus achieved prediction time.** The intended prediction time is hospital discharge after index PCI. Discharge, rather than procedure completion, is the earliest defensible time because peak cardiac troponin, if recorded (Section 2.3), requires post-procedural observation. The unrestricted analysis does not meet this cutoff, because its matrix contains antiplatelet-therapy fields recorded during follow-up. The R1 and R3 matrices exclude them. Even so, laboratory measurement times are undocumented, so the cutoff is met only as far as the documentation allows.

**Accordingly, every analysis estimates discrimination of the recorded VLST label using recorded variables. None evaluates a fixed-horizon 1–4-year post-PCI risk model, and that intended application remains unvalidated.**

### 2.3 Variables and their availability

The file contains 81 candidate predictors after removing identifiers, the outcome and TSSI. Supplementary Table S1 gives each variable's documented meaning, presumed recording time and availability at the intended prediction time. Key points:

- **Antiplatelet therapy (`Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT`).** These record therapy during follow-up rather than at the index PCI. Aspirin is recorded in only 61.8% of event-free patients, although all patients received at least one year of DAPT, and Wang et al. describe DAPT as received "during follow-up" (p. 2) [1]. These fields are **not available** at a post-PCI prediction time. Their exact recording time could not be confirmed.
- **`CaI`.** Its group means and standard deviations (37.37 ± 61.64 and 40.55 ± 72.25) equal those reported for "peak troponin I" in Wang et al.'s Table 1 (p. 3), which is the basis for treating it as peak troponin I. Units and timing are not documented and could not be confirmed. We use the dataset label `CaI` throughout.
- **`LV`.** Integer values 37–73 (mean 44.6 in controls, 49.1 in cases). There is no corresponding row in Wang et al.'s Table 1 or text, and no data dictionary was deposited. The range is compatible with a left-ventricular dimension in millimetres, for example end-diastolic diameter. This is an inference from the label and the values, not a documented definition, and no physiological interpretation is made.
- **Renal variables.** `Cre` appears to be creatinine in µmol/L, which is consistent with its values (median 72). `eGFR` is recorded eGFR. `CKD5` takes integer values 1–5 and equals the CKD stage of the recorded eGFR in every patient; `CKD60` and `CKD90` equal recorded eGFR < 60 and < 90. The equation behind the recorded eGFR is not documented (Section 2.4).
- **Stent type.** `PES`, `ZES` and `EVS` are mutually exclusive flags, exactly one per patient. `PES` frequencies equal Wang et al.'s "sirolimus-eluting stent" row (68.76% and 82.61%). `Stent type-SES` contains brand names, and every brand with `PES = 1` (Partner, Excel, Firebird, Tivoli, Cypher) is a sirolimus-eluting product. We therefore read `PES` as a sirolimus-eluting flag despite its name.
- **Post-dilation.** `1.1:1Post dilation` and `No postdilation` are exact complements. Wang et al.'s Table 1 labels as "No post-dilation" the frequencies that the file stores under `1.1:1Post dilation` (49.01% and 15.22%). We read `No postdilation = 1` as post-dilation not performed, for the reasons given in Section 2.10.
- **Other laboratory variables** (`WBC`, `HGB`, `Platelet`, lipids, `Fast-Glu`, `HbA1c`, `Fiberinogen` [sic; fibrinogen]). Their group summaries match Wang et al.'s Table 1, except HbA1c, whose two group columns are interchanged relative to the file. Measurement time (admission or in-hospital) is not documented.

### 2.4 Data-provenance audit

We assessed whether cases and controls had been recorded or exported differently. The checks were:
- row order;
- the number of decimal places in the raw text of each numeric field;
- the proportion of integer-valued entries;
- the terminal-digit distribution of integer fields;
- repeated exact values within each group;
- internal consistency of derived fields: recorded eGFR against eGFR recalculated from recorded creatinine, age and sex, and `DAPT` against the component drug fields;
- agreement of group counts and means with Wang et al.'s Table 1.

**eGFR equation.** Before recalculating eGFR, we tested whether the recorded values followed a documented equation other than CKD-EPI 2009. The candidates were CKD-EPI 2009 and 2021 [15], MDRD with constants 175 and 186, and the Chinese modified MDRD. For Cockcroft–Gault, which requires body weight (not in the file), we derived the weight that each recorded value would imply. Agreement within ±2% is reported in Supplementary Table S9.

**Precision probe.** A histogram gradient-boosting classifier was trained only on 243 indicators of whether each raw numeric value lay on a 0-, 1- or 2-decimal grid, with no clinical magnitudes, in 5-fold cross-validation using the benchmark's outer-fold seed.

The audit used read-only scripts; a standard-library script that reproduces every audit number is provided.

### 2.5 Leakage and recording-artefact controls

We distinguish two kinds of problem.
- **Data leakage**: predictors that encode the outcome or are recorded after the prediction time.
- **Outcome-dependent recording differences** (recording artefacts): differences between cases and controls in how values were recorded or exported, rather than in what they measure. Their origin is unknown, and we make no assumption about it.

The controls were adopted in stages, each after a diagnostic on this file. None was preregistered. Each stage was fully specified before any of its results were seen, and every earlier stage is reported alongside it, so the effect of each step remains visible. Supplementary Section S0 gives the rationale and the evidence for each control.

**Stage 0: controls applied in every analysis (the unrestricted analysis).**
1. **Identifiers and TSSI excluded.** TSSI is time to VLST for cases and time to the end of follow-up for controls, so it encodes the outcome. Follow-up time belongs on the time axis of a survival analysis, not among the predictors of a binary classifier.
2. **`WBC` excluded.** Its recording precision almost perfectly separates cases from controls (Section 3.2). Wang et al. also excluded leukocyte count from their score, for a different reason: an elevated count could reflect infection (p. 6) [1]. WBC was retained for the exploratory association analyses. A sensitivity analysis (Section 2.6) keeps WBC, rounded to whole numbers for every patient.
3. **Four laboratory fields rounded before splitting:** `Cre` to 0 decimal places, `CaI` to 2, `Fiberinogen` to 1 and `Fast-Glu` to 1. These are fixed row-wise rules that use no outcome data and no statistic from other rows, so applying them before or after splitting gives identical values.
4. **Stent-brand codebook fitted within training partitions.** Brand strings in `Stent type-SES` were canonicalised (spelling variants merged), and brands with fewer than 30 training observations were grouped as "other". This learned grouping was fitted within each training partition (for example, 87 brand strings to 9 levels on the 70% attribution split), and unseen brands in held-out data were mapped to "other".

The unrestricted matrix has 80 predictors. It retains the follow-up medication fields and the recorded eGFR and CKD variables.

**Stage R1 (specified 1 November 2026; 76 predictors).**

5. **Follow-up antiplatelet fields removed:** `Aspirin`, `Clopidogrel`, `Ticagrelor` and `DAPT`.
6. **eGFR recalculated.** `eGFR`, `CKD5`, `CKD60` and `CKD90` were replaced with values recalculated from creatinine (µmol/L ÷ 88.4), age and sex using CKD-EPI 2009 without the race term [15]. The recalculation uses a published equation with no fitted parameter and only each patient's own baseline values, so it was applied before splitting, like the rounding rules.

**Stage R3, the main analysis (specified 7 November, adopted as main analysis 9 November 2026, first run 14 November; 75 predictors).** R3 adds two rules to R1, aimed at the two recording differences that remained after R1:

7. **`CaI` removed.** The exact value 0.034 occurs in 20 of 92 cases and 10 of 5,093 controls (Section 3.2), and in the R1 attribution analyses `CaI` was ranked first by every method.
8. **`LV` rounded to 5-mm bins.** This removes the terminal-digit difference between groups while keeping the magnitude.

R1 and the unrestricted analysis are reported as sensitivity analyses. Everything else, including folds and seeds, was identical across stages.

There were no missing values.

### 2.6 Sensitivity analyses and run registry

Table 6 lists every benchmark run with its date, matrix, removed and transformed fields, tuning status, hosted arm and software versions. In addition to the unrestricted analysis and R1:

- **R3 with `CaI` censored** (specified 9 November; 76 predictors). `CaI` is kept instead of removed, with every value below 0.1 set to 0.1 (298 patients: 21 of 92 cases and 277 of 5,093 controls). This hides the 0.034 value among the controls' low values (0.01–0.09) and keeps every larger troponin. It cannot remove the difference in the share of low values between groups, which may be real or may reflect the placeholder, so it is reported beside R3, never instead of it.
- **R3 with rounded `WBC`** (specified 16 November; 76 predictors). WBC is kept, rounded to whole numbers for every patient so that the integer-value pattern carries no information. Only local and conventional arms were run.
- **R3 replicate.** The R3 benchmark was run twice with identical code and settings, on 14 and 15 November, on two compute accounts. The 15 November run is the main record; the other is reported as a reproducibility check.
- **Precision equalisation** (within each run). Every numeric column was rounded to the coarsest precision present anywhere in its raw text; for example, `CaI` and `HbA1c` were rounded to whole numbers. The local and conventional arms were then refitted on the same outer folds, the conventional arms with their per-fold tuned settings. Hosted arms were not refitted, because of service quota. This procedure also discards genuine clinical resolution, so a fall in AP is an upper bound on, not a measure of, the share due to formatting.
- **Leakage contrast.** Separately, on one stratified 70/30 split (3,629/1,556 patients; 64/28 events; seed 42), seven conventional models were tuned by grid search (5-fold, AP) under two conditions:
  - with leakage: TSSI and WBC retained, no rounding, brand codebook fitted on the full cohort;
  - without leakage: the stage-0 controls.

  Neither condition used oversampling. An earlier run had applied SMOTE [16] to the training partition in the with-leakage condition; it was replaced. The difference between conditions is attributed to these leakage sources together, not to any single one.

### 2.7 Models

**Conventional models.** Logistic regression, random forest, XGBoost, LightGBM and CatBoost, with class weighting:
- `class_weight="balanced"` for logistic regression, random forest and LightGBM;
- `auto_class_weights="Balanced"` for CatBoost;
- for XGBoost, `scale_pos_weight` computed from each training partition.

Within each training partition, numeric features were median-imputed and standardised, and the categorical brand feature was one-hot encoded. In the R1, R3 and sensitivity runs, each conventional model was tuned inside every outer training partition. The tuning used a random search of 20 configurations scored by AP over the same 4 inner folds that then choose the threshold. The search spaces are listed in Supplementary Table S7. The unrestricted analysis used library defaults.

**TabPFN** (Prior Labs) [10–12].
- **Local arms:** checkpoints `tabpfn-v3-classifier-v3_default.ckpt` (v3) and `tabpfn-v3.5-20260909.safetensors` (v3.5), selected explicitly by version, with `n_estimators="auto"` and thinking disabled. Local TabPFN was not tuned; it was used as a documented in-context learner, and a search over its settings would require a full GPU refit per configuration.
- **Hosted arm in the R1, R3 and sensitivity runs:** v3.5 (`v3.5_default`) through `tabpfn-client`, with `thinking_mode=True`, `thinking_effort="micro"` and `thinking_metric="average_precision"` ("TabPFN thinking v3.5, effort micro"). Effort "micro" was the only setting whose quote for the nested protocol fitted the service allowance: about 6.8 million tokens, against about 149 million for effort "high". A token budget of 7.5 million was enforced per run.
- **Hosted arms in the unrestricted analysis:** v3 (`v3_default`) and v3.5 (`v3.5_default`) at `thinking_effort="high"`. Because the effort differs, hosted results are not compared across the unrestricted analysis and the later runs.

According to the vendor documentation, thinking mode is available only through the hosted service. It "uses more compute during fitting", and the metric "defines what to optimise" [12]. Its internal mechanism is not documented. Hosted fitting was done within each training partition only.

**Optimisation tiers.** In R1, R3 and the sensitivity runs, results fall into two declared tiers:
- **Tuned tier:** the conventional arms (inner random search) and hosted thinking (metric-targeted fitting).
- **Untuned tier:** local TabPFN.

This asymmetry favours the comparators, not TabPFN.

**Software.**
- **Benchmark runs (November):** Python 3.12.13, `tabpfn` 9.2.0, `tabpfn-client` 0.7.1, scikit-learn 1.6.1, XGBoost 3.2.0, LightGBM 4.6.0, CatBoost 1.2.10 and PyTorch 2.10.0, on one NVIDIA Tesla T4 GPU.
- **Unrestricted analysis (September):** the same, except `tabpfn` 9.0.0 and `tabpfn-client` 0.6.0. The local v3.5 checkpoint file and its defaults are identical in both releases, and local TabPFN gave identical results in the September-software and November-software R1 runs (Section 3.4).
- Package versions are recorded in each run manifest.

### 2.8 Primary evaluation: nested cross-validation

Every arm in every run used the same stratified 5-fold outer split (shuffled, seed 42; 18–19 events per held-out fold). In each outer training partition, a stratified 4-fold inner split (about 14 events per inner held-out fold) was used for two things:
- the hyperparameter search for the conventional arms, in the runs where tuning applied;
- inner out-of-fold probabilities, from which the threshold maximising F1 was chosen.

That threshold was applied once to the corresponding outer held-out fold. Nothing was selected on an outer held-out fold. All preprocessing and the brand codebook were refitted within every inner and outer training partition. The procedure was run once per configuration; it is not repeated nested cross-validation.

**Metrics.**
- **Primary: pooled out-of-fold average precision** (AP; scikit-learn `average_precision_score`, a step-wise summary of the precision–recall curve, not trapezoidal PR-AUC). The AP of a non-informative ranking equals the prevalence, 0.0177.
- **Secondary:**
  - ROC-AUC.
  - Brier score, compared with the null Brier score of predicting the prevalence for everyone (0.0177 × 0.9823 = 0.0174), and the scaled Brier score, 1 − Brier / null Brier (0 = no better than predicting the prevalence; negative = worse).
  - Calibration-in-the-large: the observed/expected ratio (O/E) and the calibration intercept (logistic model with logit(p) as an offset).
  - Calibration slope: a logistic model of the label on logit(p). A slope of 1 is ideal; below 1 means predictions are too extreme, above 1 too modest.
  - Calibration curves on axes restricted to the range the predictions occupy.
  - Expected calibration error over 8 quantile bins, as a secondary measure only, because with 92 events most bins hold near-zero predictions and outcomes.
  - Sensitivity, precision and F1 at the inner-loop thresholds.

ROC-AUC and AP summarise different aspects of ranking. ROC-AUC does not depend on the class ratio. AP reflects precision across recall levels and depends on prevalence, which makes it more informative about positive predictive value when events are rare. Neither reflects precision at a single clinical operating point.

**Uncertainty.** Percentile 95% intervals came from 2,000 stratified bootstrap resamples of the stored pooled out-of-fold predictions (seed 42), resampling cases and controls separately, without refitting.
- **Paired differences in AP**, within a run or between runs on the same patients, used the same resamples. For each contrast we report the share of resamples in which the difference was ≤ 0. When none was, this is written as "0 of 2,000 (bootstrap P < 0.0005)", never as P = 0.
- **What the intervals cover.** They reflect sampling variation of the evaluation rows given the fitted predictions. They do not capture variation from model fitting or fold assignment. Fold-level ranges and the replicate run are reported for that purpose.

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

We report ROC-AUC, AP and observed event rates in Wang et al.'s categories (≤ 7, 8–9, ≥ 10). This analysis evaluates the frozen score as a ranking reference against the same observed binary labels used for the machine-learning models.
- **Not external validation.** These are the score's derivation patients, so its performance here is apparent (in-sample), whereas the machine-learning arms are scored out of fold.
- **Not a reproduction of the original analysis.** Wang et al.'s c-statistic is a concordance index over time to VLST; our ROC-AUC is a binary AUC for the any-time label. The two numbers are different quantities.
- **Different inputs.** The score uses 8 items, and the R3 matrix has 75 columns.
- **No transformation.** No probability transformation of the score was applied.
- **Sensitivity version.** The `CKD90` item was replaced with eGFR < 90 recalculated from creatinine.

### 2.10 Post-dilation coding

Wang et al.'s paper is internally inconsistent on this item, and the original authors could not be contacted. The coding was therefore fixed by the weight of the published evidence before any score metric was computed.

Five statements support `No postdilation = 1` as the risk item:
1. the file's column names: `No postdilation` is the exact complement of `1.1:1Post dilation`, which names the balloon-to-artery ratio of a dilation that was performed;
2. the abstract and the Discussion, which name the absence of post-dilation as a risk factor;
3. the positive Table 2 coefficient (β = +1.93);
4. the +4 points, which an additive risk score gives only to risk factors;
5. the established mechanism, in which omitting post-dilation leaves stents under-expanded, a recognised substrate of late thrombosis [20].

Against it are the Table 1 row labelled "No post-dilation" and the Table 2 hazard ratio of 0.145. That hazard ratio equals exp(−1.93), the reciprocal of exp(+1.93) = 6.9, and the Table 1 frequencies are those of `1.1:1Post dilation`. Both are what a reversed reference category in those two cells would produce, and neither can be reconciled with a positive coefficient and +4 points. The opposite coding is reported for transparency only (Section 3.7).

### 2.11 Exploratory association analyses

These analyses used the full cohort and did not inform the prediction matrices.
- **Continuous variables:** compared with Welch's t-test when |skewness| ≤ 1 and excess kurtosis ≤ 3, and with the Mann–Whitney U test otherwise.
- **Binary variables:** χ² test, or Fisher's exact test when any expected count was below 5. Tests were two-sided.
- **Multiplicity:** controlled with the Benjamini–Hochberg procedure within variable families [17].
- **Multivariable model:** an unweighted logistic regression with 13 covariates, one representative per collinear block chosen after the univariate screen, with continuous covariates standardised per standard deviation. Wald 95% intervals are reported. Because covariates were chosen after screening on the same data, intervals and P values are descriptive. There were about 7.1 events per covariate.
- **Sensitivity analyses:** Firth-penalised estimates and a 2,000-replicate bootstrap.
- **Reanalysis R2** refitted the same model three ways: (A) recorded eGFR, as originally specified; (B) eGFR recalculated from creatinine; (C) as B, without the follow-up-period clopidogrel field.

R2 addresses the eGFR and medication findings only. Its covariates still include `WBC` and unbinned `LV`, so it is not aligned with the R3 prediction matrix.

These are odds ratios for the recorded label during variable follow-up, not hazard ratios, and they are not causal effects. Continuous covariates are reported per standard deviation of that covariate in the cohort, for example 34.0 mL/min/1.73 m² for recorded eGFR and 17.7 for recalculated eGFR. Binary covariates compare presence with absence.

### 2.12 Exploratory model attribution

Attribution analyses used one stratified 70/30 split (seed 42). Models were fitted on 3,629 patients (64 events), and 1,556 held-out patients (28 events) were explained. The split and its purpose differ from the benchmark.

The main attribution analyses used the R3 matrix (75 features) with the same controls as the benchmark, and the stent-brand codebook fitted on the training split only. Every TabPFN attribution used the local v3.5 checkpoint `tabpfn-v3.5-20260909.safetensors`, selected explicitly. Packages were pinned: `tabpfn` 9.2.0, `tabpfn-extensions` 0.6.3, `shapiq` 1.8.0 and `shap` 0.54.1. A provenance file records every version, the checkpoint and the output scale. The analyses were:
- **Mutual information** (scikit-learn, k = 3 neighbours) on the training rows.
- **Repeated forward feature selection:** 8 of 75 features selected in each of 8 resamples (seeds 42–49), with 3-fold stratified CV within the training rows, AP scoring and local TabPFN v3.5 with `n_estimators = 1`.
- **Partial dependence** of the predicted probability on binary features, averaged over the training rows.
- **Shapley values** for all 1,556 held-out patients, using shapiq via tabpfn-extensions with the baseline imputer, a budget of 256 and the training data as background [18].
  - **Scale.** The values are on the scale of the class-1 logit: the baseline plus the sum of a patient's values reproduced the model's logit to within 0.0014, and missed its probability by up to 9.3.
  - **Reading.** A value of +1.2 therefore multiplies the odds by about e^1.2 ≈ 3.3. It is not a 120-percentage-point change in risk.
- **Consensus.** A rank-based consensus combined mutual-information rank, selection frequency and mean |SHAP| rank, all from the same analysis variant.
- **Classic models.** To check that conclusions did not depend on TabPFN, the seven conventional models (logistic regression; random forest, with and without row subsampling; XGBoost, with and without stronger row and column subsampling; LightGBM; CatBoost; all class-weighted) were each ranked by three methods on the R3 matrix:
  - leave-one-covariate-out (LOCO) loss in AP;
  - mean |SHAP|;
  - forward selection by AP.

  These used a fit/validation split of 4,148/1,037 patients (74/18 events), with scaling and encoding fitted on the fit rows only.

Shapley values were not computed for the hosted thinking model. One high-effort v3.5 thinking fit on the training rows alone was quoted at about 5.9 million tokens, and Shapley estimation needs thousands of model evaluations per patient. All TabPFN attribution therefore describes the local v3.5 model.

**These attributions describe the models fitted on the 70% split (or the fit rows). They are not explanations of the hosted model or of the benchmark arms, and they are not causal or biological evidence.** Agreement between correlated encodings of the same construct (for example `1.1:1Post dilation` and `No postdilation`) is not independent confirmation. The earlier attribution run on the R1 matrix is reported as context. Its feature-selection and partial-dependence part predated the version pins.

### 2.13 Reporting

Reporting follows TRIPOD+AI where applicable [19]. Items specific to model development for clinical use are not applicable, because no model is proposed for clinical use.

---

## 3. Results

### 3.1 Cohort

Table 1 summarises characteristics by outcome. Of 5,185 patients, 92 had definite VLST (1.77%). Events occurred a median of 736 days after PCI (range 380–1,575). Event-free patients were followed for 1,241–1,605 days (median 1,419).

Several counts in the file differ from Wang et al.'s Table 1:
- previous PCI among cases: 10 (10.87%) vs 13 (14.13%);
- AMI as admitting diagnosis: 3,095 controls and 65 cases vs 3,046 and 72;
- HbA1c: the file gives 5.80 ± 1.61 in controls and 6.27 ± 1.53 in cases, the reverse of the published columns.

The file's HbA1c values are consistent with diabetes status in both groups (mean 7.61 in patients with diabetes in each group). This suggests that the published HbA1c columns, rather than the file, are interchanged. We analysed the file exactly as deposited. Whether it is the final analysis file of Wang et al. could not be confirmed. The differences are small relative to the cohort and do not involve the outcome counts.

### 3.2 Data-provenance audit

Table 2 lists the findings.

**Row order.** The file is ordered by outcome: all 92 cases are the last 92 rows.

**Number formatting.**
- Control laboratory values are stored with three decimal places in the raw text; case values have zero to two.
- After parsing, integer-valued WBC occurs in 3,780 of 5,093 controls (74.2%) and in no cases.
- `LV` is an integer in both groups, but its terminal digits differ: an even digit in 65.1% of controls vs 47.8% of cases, and the digit 0 in 1.3% of controls.
- The precision probe, which used no clinical magnitudes, reached AP 0.427 and ROC-AUC 0.952. The strongest single indicator was integer-valued WBC (ROC-AUC 0.871).

**eGFR.**
- Mean creatinine was 72.5 µmol/L in controls and 72.4 µmol/L in cases, yet recorded eGFR averaged 120.0 and 95.9 mL/min/1.73 m².
- CKD-EPI 2009 eGFR recalculated from creatinine, age and sex reproduced the recorded value within 1 unit in 2,243 of 5,093 controls (44.0%) and in 50 of 92 cases (54.3%).
- In most of the remaining controls, the recorded eGFR was 1.3–1.9 times the recalculated value.
- **Other equations did not explain the remainder** (Table S9). Within ±2%, CKD-EPI 2009 matched 44.0% of controls and 57.6% of cases. CKD-EPI 2021 matched 12.2% and 20.7%, MDRD-175 4.0% and 12.0%, MDRD-186 7.3% and 23.9%, and the Chinese modified MDRD 0% and 9.8%. At least one equation matched 44% of controls but 85% of cases.
- **Nor did Cockcroft–Gault.** For the unmatched controls it would imply a median body weight of 110 kg (IQR 101–121). Their recorded-to-recalculated ratio (about 1.57) increased with age (r = 0.48), the opposite of what a creatinine-clearance estimate would do.
- **Recalculated values were similar in the two groups.** Means were 91.9 (controls) and 91.5 (cases), and eGFR < 90 occurred in 35.6% and 40.2%. With recorded eGFR, these proportions were 16.9% and 34.8%.
- Because `CKD5`, `CKD60` and `CKD90` are derived from recorded eGFR, they carry the same difference.

**Medications.** Among controls, `DAPT = 1` always coincided with aspirin plus a P2Y12 inhibitor (2,260 of 2,260). Among cases it did in 27 of 35; in 4 cases `DAPT = 1` was recorded with all three drug fields at 0.

**`CaI`.** The value 0.034 occurs in 20 of 92 cases (21.7%) and in 10 of 5,093 controls (0.2%); no other value repeats more than twice among cases. After rounding to two decimals it becomes 0.03, held by 20 cases and 16 controls, so the rounding rule does not remove it.

**`LVEF`.** The value 56 is also more frequent in cases (23 of 92, 25%) than in controls (733 of 5,093, 14%). This pattern is weaker and was not addressed by any control.

### 3.3 Main analysis: R3

Table 3 and Figure 1 show pooled out-of-fold performance on the R3 matrix (75 predictors).
- **Hosted TabPFN thinking v3.5 (effort micro)** had the highest AP, 0.475 (95% interval 0.381–0.583), with ROC-AUC 0.957 (0.936–0.975).
- **Local TabPFN v3.5** followed closely: AP 0.465 (0.372–0.567), ROC-AUC 0.947 (0.921–0.969).
- **The tuned conventional models** reached AP 0.266 for XGBoost (0.191–0.358), 0.229 for LightGBM (0.165–0.322) and 0.181 for CatBoost (0.123–0.267). Local TabPFN v3 reached 0.250 (0.183–0.356).
- **Logistic regression and random forest** had AP 0.094 and 0.068, about 5 and 4 times the prevalence.

**Paired comparisons.**
- **TabPFN against XGBoost:**
  - local TabPFN v3.5 minus XGBoost was 0.199 (0.105–0.292);
  - hosted thinking minus XGBoost was 0.209 (0.107–0.310);
  - in 0 of 2,000 resamples was either difference ≤ 0 (bootstrap P < 0.0005).
- **Local TabPFN v3.5 against LightGBM:** 0.236 (0.135–0.328).
- **Hosted thinking against local v3.5:** 0.010 (−0.032 to 0.055). The difference was ≤ 0 in 562 of 2,000 resamples, so no difference was detected.

**Fold-level AP** ranged from 0.411 to 0.564 for hosted thinking, from 0.313 to 0.610 for local v3.5 and from 0.252 to 0.339 for XGBoost.

**Operating point** (Table 4). At inner-fold F1 thresholds applied to the outer folds:
- hosted thinking (threshold 0.164 ± 0.056) identified 43 of 92 events with 55 false positives (sensitivity 0.47, precision 0.44);
- local v3.5 (0.203 ± 0.065) identified 45 of 92 with 52 false positives (0.49, 0.46);
- XGBoost identified 34 of 92 with 105 false positives (0.37, 0.25).

For comparison, in R1 local v3.5 identified 51 of 92 events with 22 false positives (0.55, 0.70).

**Calibration** (Figure 2; Table S8). Against a null Brier score of 0.0174:
- **Scaled Brier scores:** 0.28 for both hosted thinking and local v3.5, 0.13 for local v3, 0.07 for LightGBM and 0.03 for CatBoost.
- **Worse than predicting the prevalence:** XGBoost (−0.03), random forest (−1.08) and logistic regression (−6.33). In each case class weighting inflated the probabilities; for logistic regression the mean predicted risk was 26.6% against a prevalence of 1.8%.
- **Local v3.5** was right on average (O/E 1.01; calibration slope 1.10). Hosted thinking under-predicted slightly (O/E 1.23; slope 1.20).
- **The boosted trees** had slopes of 0.40–0.57, meaning their predictions were too extreme.
- **Expected calibration error** was below 0.013 for every arm except logistic regression and random forest. With 8 quantile bins at this prevalence, most bins contain near-zero risk, so ECE says little about the higher-risk range and is not used for any conclusion.

**Reproducibility.** The replicate R3 run, with identical code and settings, gave bit-identical out-of-fold predictions for logistic regression, random forest, XGBoost, CatBoost and both local TabPFN arms. Hosted thinking gave AP 0.478 against 0.475 (paired difference 0.002, −0.008 to 0.012; correlation of predictions 0.998). LightGBM was not reproducible (AP 0.214 against 0.229); the cause was not investigated. The main record is the run with the smaller TabPFN lead over LightGBM.

**Residual formatting.** Precision equalisation within R3 lowered AP as follows:
- local v3.5 from 0.465 to 0.327;
- local v3 from 0.250 to 0.178;
- XGBoost from 0.266 to 0.206;
- LightGBM from 0.229 to 0.188;
- CatBoost from 0.181 to 0.136.

Random forest and logistic regression changed by less than 0.01.

**These results describe discrimination of the recorded label in this file after the controls in Section 2.5. They are an upper bound for this file, not clinical predictive performance.**

### 3.4 How each control changed discrimination

Table 3a gives AP in every run.

**Unrestricted analysis (80 predictors; conventional arms at defaults; hosted arms at effort high).**
- Hosted TabPFN thinking v3.5 reached AP 0.921 (0.878–0.961; ROC-AUC 0.996), and local v3.5 0.896 (0.845–0.943).
- The best conventional models were XGBoost (0.632) and LightGBM (0.627).
- Precision equalisation lowered local v3.5 to 0.844 and XGBoost to 0.581.

**R1 (76 predictors).** AP fell in every arm.
- Local v3.5 fell from 0.896 to 0.668 (0.583–0.757). Hosted thinking v3.5 at effort micro reached 0.724 (0.639–0.807).
- The tuned conventional arms reached 0.409 for XGBoost, 0.389 for LightGBM, 0.344 for CatBoost, 0.202 for random forest and 0.096 for logistic regression.
- Local v3 reached 0.322.
- R1 changes two things at once (medication fields and eGFR), so the fall cannot be split between them.
- An earlier R1 run with conventional arms at library defaults gave higher AP than the tuned arms for XGBoost (0.447), LightGBM (0.422), CatBoost (0.384) and logistic regression (0.123), and lower for random forest (0.185). Local TabPFN gave identical AP in both runs (v3.5 0.668; v3 0.322).

**R1 to R3.** Removing `CaI` and binning `LV` lowered local v3.5 AP by a further 0.203 (0.126–0.277), hosted thinking by 0.248 (0.167–0.320) and XGBoost by 0.142 (0.059–0.226).

**R3 with `CaI` censored at 0.1 (76 predictors).**
- Local v3.5 reached AP 0.482 (0.391–0.585), 0.017 (−0.019 to 0.055) above R3 and 0.186 (0.120–0.250) below R1.
- Hosted thinking reached 0.522 (0.426–0.624), 0.046 (0.011–0.079) above R3.
- The conventional arms were within 0.05 of their R3 values: LightGBM 0.277, XGBoost 0.234 and CatBoost 0.150.

**R3 with rounded `WBC` (76 predictors).**
- Local v3.5 reached AP 0.516 (0.422–0.625), 0.051 (0.002–0.105) above R3; in 40 of 2,000 resamples the difference was ≤ 0.
- XGBoost reached 0.288, 0.022 (−0.043 to 0.093) above R3. Logistic regression reached 0.137, against 0.094.

Across the sequence, local TabPFN v3.5, which used the same checkpoint and settings throughout, went from AP 0.896 to 0.668 to 0.465. The lead of the best TabPFN arm over the best conventional arm was 0.289 in the unrestricted analysis (hosted thinking at effort high against XGBoost at defaults), 0.315 in R1 and 0.209 in R3.

### 3.5 Leakage contrast

In the bundled contrast on one 70/30 split without oversampling (Table S3), hold-out AP was 0.12–0.71 higher with leakage than without in every model:
- LightGBM 0.959 vs 0.668;
- XGBoost 0.955 vs 0.592;
- CatBoost 0.951 vs 0.494;
- random forest 0.957 vs 0.487;
- logistic regression 0.888 vs 0.343;
- decision tree 0.846 vs 0.138;
- Gaussian naive Bayes 0.180 vs 0.056.

The gap reflects data leakage as a whole and cannot be assigned to TSSI or any single source. The without-leakage condition corresponds to the stage-0 controls only. It still contains the follow-up medication fields and recorded eGFR, so it is not comparable with R1 or R3.

### 3.6 Exploratory attributions

**Local TabPFN v3.5 on the R3 matrix (70/30 split).**
- **Mean |SHAP|** (class-1 logit scale; 1,556 held-out patients) was largest for:
  - `HbA1c` (0.41);
  - `LDL` (0.36);
  - stent brand (0.34);
  - `No postdilation` (0.33);
  - previous PCI (0.31);
  - `1.1:1Post dilation` (0.31);
  - `LV` (0.28);
  - triglycerides (0.24);
  - total cholesterol (0.23).

  The other 66 features together contributed 2.49 of a total of 5.31 (Table S5). Recalculated `eGFR` ranked 32nd (0.035) and `Cre` 53rd.
- **Mutual information** ranked `LDL`, `LV`, `No postdilation` and `HbA1c` highest.
- **Repeated forward selection:**
  - `HbA1c` and `LV` were selected in 8 of 8 resamples, and `Fast-Glu` in 7;
  - `LDL`, `LVEF`, `No postdilation` and stent brand were selected in 5;
  - previous PCI was selected in 3;
  - recalculated `eGFR` was never selected, and `Cre` was selected once.
- **Consensus:** the top six were `HbA1c`, `LDL`, `LV`, `No postdilation`, stent brand and previous PCI.
- **Partial dependence** on binary features was small: +0.010 for previous PCI, −0.005 for `1.1:1Post dilation` and +0.0002 for `No postdilation`, on a training prevalence of 0.018.

**Comparison with the R1 attribution run.** On the R1 matrix, `CaI` had ranked first by every method (mean |SHAP| 1.29, twice the next feature). In the earlier run with recorded eGFR, `eGFR` had been selected in 8 of 8 resamples and `Cre` in 7 of 8. Under R3, no single feature dominates: the largest mean |SHAP| is under a tenth of the total.

**Conventional models on the R3 matrix (21 model × method lists).**
- **Feature ranked first:** `HbA1c` in 5 lists; `LDL`, fibrinogen and total stent length in 3 each; recalculated `eGFR` in 1.
- **Earlier comparison:** in the earlier run that kept the follow-up medication fields, `CaI` and recorded eGFR, a feature later removed or recalculated had ranked first in 16 of 21 lists, recorded `eGFR` alone in 10.
- **Wang score items in the top 20 lists:** among the top 20 features of each list, `No postdilation` appeared in 15 of 21 lists and previous PCI in 9. Each list contained between 0 and 5 of the 8 Wang score items (median 2).

`LV` remains highly ranked after binning. Its magnitude differs between groups (mean 44.6 vs 49.1), and binning does not change that, but its definition is unknown. These attributions describe how the fitted models separate the recorded label in this file; they are not evidence about biology.

### 3.7 Wang score

The frozen Wang score had ROC-AUC 0.801 and AP 0.103 (fold-wise, evaluation only: ROC-AUC 0.80 ± 0.06; AP 0.11 ± 0.05). Observed VLST rates were:
- 0.51% for scores ≤ 7 (16 of 3,135 patients);
- 2.22% for scores of 8–9 (35 of 1,577);
- 8.67% for scores ≥ 10 (41 of 473).

The low- and high-risk group sizes equal those Wang et al. published. Their intermediate group (1,837) is inconsistent with their own total, since 3,135 + 1,837 + 473 = 5,445. The file gives 1,577, which completes the total of 5,185 and reproduces their 2.2% event rate, so 1,837 is most likely a typographical error (Figure 3). It changes none of our computations.

With the opposite post-dilation coding (Section 2.10), ROC-AUC was 0.508, reported for transparency only. With eGFR < 90 recalculated from creatinine, 1,848 patients met the item instead of 892, and the score gave ROC-AUC 0.789 and AP 0.090.

### 3.8 Exploratory associations

Table 1 gives univariate comparisons and Table 5 the 13-covariate model.

**Multivariable model (specification A).**
- Higher odds of the recorded label were associated with previous PCI (OR 6.71; 95% CI 2.88–15.61), `LV` (1.83 per SD; 1.54–2.18) and WBC (1.97 per SD; 1.67–2.33).
- Lower odds were associated with `1.1:1Post dilation` (0.15; 0.08–0.29), recorded eGFR (0.57 per SD; 0.45–0.72) and follow-up-period clopidogrel (0.48; 0.29–0.79).
- Diabetes, HbA1c, number of diseased vessels, stents per lesion, total stent length, fibrinogen and `PES` had intervals including 1.
- Firth estimates were similar (for example, WBC 1.95 and eGFR 0.57).

**R2 (Table 5).**
- **eGFR.** Replacing recorded eGFR with recalculated eGFR changed its odds ratio from 0.57 per SD (0.45–0.72) to 1.18 per SD (0.97–1.44). The inverse association disappeared, and the interval now includes 1. Removing clopidogrel (specification C) left it at 1.19 (0.97–1.45).
- **Other estimates** moved little. In specification B: previous PCI 6.74, post-dilation 0.15, WBC 2.04 and `LV` 1.93.
- **Two Wang score items** moved across the conventional threshold:
  - number of diseased vessels from 1.21 (0.95–1.54) to 1.32 (1.04–1.67);
  - `PES` from 1.73 (0.95–3.15) to 1.81 (1.01–3.23) in C.

  These are small shifts within overlapping intervals; the change from "not significant" to "significant" is not itself a significant change. A plausible reason is that recorded eGFR, being lower in cases for reasons unrelated to creatinine, had absorbed part of the risk of older patients with more diseased vessels (recalculated eGFR correlates with age at −0.53). The vessel count recovered that risk once eGFR was corrected.
- **Clopidogrel.** This association concerns therapy recorded during follow-up and is subject to prescribing, adherence and reverse-timing effects.

---

## 4. Discussion

### 4.1 Principal findings

The first analysis of this public file suggested that TabPFN could identify almost every patient who would later have VLST (AP 0.921, ROC-AUC 0.996). The audit showed why that result should not be taken at face value. The prediction matrix contained antiplatelet-therapy fields recorded during follow-up. The file also contains outcome-dependent recording differences that a flexible learner can use without any clinical signal:
- a recorded eGFR that cannot be reproduced from creatinine in most controls;
- different number formats in cases and controls;
- inconsistent derived fields in cases;
- a troponin value repeated in a fifth of cases;
- precision indicators alone reaching AP 0.43.

When these were addressed step by step, discrimination fell at every step and in every model. For local TabPFN v3.5, whose settings never changed, AP fell from 0.896 to 0.668 to 0.465, roughly halving. Precision equalisation lowered it further. **Most of the near-perfect discrimination first observed was therefore a property of how this file was assembled, not of patient risk.**

Under the strictest controls (R3), however, TabPFN still ranked the recorded label better than every conventional learner. This held even with the conventional learners tuned, which favours the comparators. The lead over XGBoost was 0.20 AP (0.11–0.29), and AP of 0.47 is about 27 times the prevalence. That residual discrimination is not explained by any recording difference we identified. It is also not shown to be clinical. Residual formatting still contributes, since precision equalisation lowers R3 AP by about a third. `LV` and `LVEF` carry unresolved patterns, and the label is not a fixed-horizon outcome. **The R3 numbers are therefore an upper bound for this file.**

### 4.2 What each control showed

**Follow-up medication and eGFR (R1).** Removing post-baseline therapy fields and recalculating eGFR removed about a quarter of local TabPFN v3.5's AP and more than a third of each conventional arm's AP (tuned arms compared with the unrestricted defaults). The eGFR finding matters beyond modelling. Recorded eGFR is about 24 units lower in cases than in controls despite identical creatinine. No documented equation explains the controls' values, and the Cockcroft–Gault weights they would imply are implausible. The difference therefore cannot be read as worse renal function in cases.

**The `CaI` placeholder and `LV` digits (R3).** Removing `CaI` and binning `LV` removed a further 0.20 AP from local TabPFN v3.5. The censoring sensitivity shows that this was mainly the placeholder, not loss of troponin information. When the 0.034 value was hidden among other low values and every troponin of 0.1 or more was kept, local TabPFN v3.5 recovered only 0.017 of that loss. Hosted thinking recovered somewhat more (0.046). Real troponin magnitude may therefore carry a little signal, but the dominance of `CaI` in R1 (mean |SHAP| twice that of any other feature) is explained by one repeated value concentrated in cases.

**`WBC`.** Keeping WBC, rounded so that its integer pattern carries no information, raised local TabPFN v3.5 by 0.05 (0.002–0.105) and did not detectably change XGBoost. WBC magnitude may carry some signal, consistent with its strong association in the logistic model. Removing it was the conservative choice. Its cost is small, and leaving it out avoids building the main analysis on a variable whose recording is known to differ by outcome.

**Leakage as a whole.** In the separate leakage contrast, retaining follow-up time and WBC, skipping the rounding and fitting the codebook on all rows raised AP by 0.12–0.71. Several models reached AP above 0.95. This is the order of inflation a naive analysis of this file would report.

### 4.3 What the comparison does and does not show

The benchmark compared existing classifiers on identical outer folds, under identical controls in each run. Within those limits:
- **The TabPFN lead survives tuning of the comparators.** Tuning the conventional arms by inner random search did not improve them. In R1, where both versions exist, tuned AP was lower than default AP for four of five conventional arms. With about 14 events per inner fold, hyperparameter search fits noise. The conventional results in R1 and R3 are therefore not an artefact of neglected tuning, and the local TabPFN arms, which were not tuned at all, still led.
- **Hosted thinking added little.** At effort micro, hosted thinking did not differ detectably from local v3.5 in R3 (0.010; −0.032 to 0.055). It was modestly ahead in R1 and under `CaI` censoring. The unrestricted analysis used effort high, so the hosted results cannot be compared across the sequence.
- **TabPFN lost more than the conventional models.** Between the unrestricted analysis and R3, local TabPFN v3 lost 65% of its AP and v3.5 48%. Tuned XGBoost in R3 was 58% below XGBoost at defaults in the unrestricted run, but that comparison mixes tuning with the controls. Tabular foundation models are designed to pick up relations between columns, such as the eGFR–creatinine inconsistency, and value patterns such as a repeated placeholder. That design helps on genuine structure and on artefacts alike, so a model's ability to exploit a dataset's quirks should be checked before its ranking is read as clinical.
- **Only one fold assignment was used.** The bootstrap intervals do not capture fitting variability. The replicate run showed that fitting variability is negligible for the local and most conventional arms, small for hosted thinking and material for LightGBM.

We found no published evidence about whether this public dataset was among the data used to develop TabPFN versions. The vendor reports do not list such datasets in the sources we examined, so the question is unresolved. We make no allegation of overlap.

### 4.4 The Wang score as a clinical reference

Evaluated without refitting on the same labels, the Wang score gave ROC-AUC 0.80 and AP 0.10, about 5.8 times the prevalence. It stratified observed risk from 0.5% to 8.7%. TabPFN's R3 AP (0.47) is higher, but this is not evidence that machine learning is superior to the score, for four reasons:
- **Inputs.** The score uses eight items; the R3 matrix has 75 columns.
- **Evaluation.** The score was developed on these same patients and is evaluated in-sample, while the machine-learning predictions are out-of-fold within one file whose residual recording differences remain.
- **eGFR item.** One score item, eGFR < 90, depends on the recorded eGFR. With recalculated eGFR the score gave ROC-AUC 0.789 and AP 0.090.
- **Generalisability.** The score has external evidence (c-statistic 0.82 in 2,058 patients [1]); the machine-learning arms have none.

The numerical similarity between our ROC-AUC (0.80) and Wang et al.'s c-statistic (0.80) does not imply reproduction. Their c-statistic came from a Cox analysis whose time axis, censoring and handling of competing events are not reported. A fair comparison would:
- give the machine-learning models the same eight inputs;
- use a fixed-horizon outcome;
- test on independent data.

### 4.5 Inconsistencies in the source publication

Wang et al.'s Table 2 lists no post-dilation with coefficient 1.93 and 4 risk points, but with a hazard ratio of 0.145. That hazard ratio equals exp(−1.93) and is incompatible with the sign of the coefficient. Their Table 1 labels as "no post-dilation" a group that is 15.2% of cases and 49.0% of controls; in the public file, that group is stored as `1.1:1Post dilation`. These two entries are the only evidence for the opposite coding, and both are explained by a reversed reference category. The coefficient, the points, the text and the column names all point the other way (Section 2.10).

These points, the arithmetic of the risk-group sizes, and the HbA1c and count differences (Section 3.1) do not alter the score's integer weights. They show that the published tables contain errors, and that the deposited file, not the tables, should be treated as the primary record.

### 4.6 Clinical associations

The associations below are exploratory. They were estimated with about 7 events per covariate, refer to the recorded label during variable follow-up, and are not causal. The association model still contains WBC and unbinned `LV`, so it is not aligned with the R3 prediction matrix.

- **Previous PCI** was associated with higher odds (OR 6.7) and was among the top-ranked features in every R3 attribution. This direction agrees with Wang et al.'s model and with the inclusion of previous PCI in the Dangas score [1,4]. Because the data are Wang et al.'s derivation data, the agreement is not independent confirmation. Prior PCI may mark more extensive coronary disease or prior stenting; this analysis cannot distinguish these explanations.
- **Post-dilation** was associated with lower odds (OR 0.15), and the post-dilation fields were among the most consistently ranked features in both TabPFN and conventional attributions under R3. This is consistent with the mechanistic role of stent under-expansion and malapposition in very late thrombosis found in intracoronary-imaging studies [20]. Large registry data have not shown a consistent association between post-dilation and stent thrombosis [21]. Post-dilation was chosen by operators, not randomised, and Wang et al. note that procedural variables reflected the operator's subjective assessment (p. 6) [1]. The association should not be read as evidence that post-dilation prevents VLST, and its direction rests on the coding chosen in Section 2.10.
- **Follow-up-period clopidogrel** was associated with lower odds (OR 0.48). The direction agrees with randomised evidence that extending P2Y12 inhibition beyond one year reduces stent thrombosis while increasing bleeding [22]. Here, however, the medication fields describe therapy during follow-up at an undocumented time, and their coding differs between cases and controls. The estimate cannot be interpreted as a treatment effect or used to guide DAPT duration.
- **WBC** was strongly associated with the label (OR 1.97 per SD). Wang et al. excluded leukocyte count from their score because elevated counts could reflect infection or other conditions [1]. In this file, WBC also carries the strongest recording-precision difference between cases and controls (Section 3.2). Once rounded to remove that difference, WBC added little to the benchmark models (Section 3.4). Inflammatory, incidental and artefactual explanations cannot be separated. In ATLAS ACS 2–TIMI 51, higher WBC was associated with early, not very late, stent thrombosis [23].
- **Recorded eGFR** was associated with lower odds (OR 0.57 per SD), but with recalculated eGFR the estimate was 1.18 (0.97–1.44) (R2). The apparent inverse association was therefore an artefact of how eGFR was recorded, and we make no renal interpretation. Renal disease has been associated with VLST in other cohorts [5], so a true association remains plausible but is not demonstrated here.
- **Number of diseased vessels and `PES`** reached conventional significance only after eGFR was corrected (OR 1.32 and 1.81). Both are Wang score items, so their direction agrees with the original model. The shifts were small, and these data are the score's own derivation data, so this is not independent confirmation.
- **Glycaemic and lipid variables** (`HbA1c`, `Fast-Glu`, `LDL`) were the most consistently ranked features in the R3 attributions, although HbA1c's interval included 1 in the association model. Attribution ranks describe how a model separates the recorded label, not effect sizes. Given the file's recording differences in laboratory fields, these rankings are hypotheses only.
- **`LV`** was associated with higher odds (OR 1.83 per SD) and remained highly ranked after 5-mm binning. Because its definition is unknown, no physiological interpretation is offered.
- **No association was found for diabetes or stents per lesion**, both components of Wang et al.'s score. These variables did not have adjusted intervals excluding 1 in our 13-covariate model. Differences in model form, outcome handling, covariates and the small number of events may explain this, and a non-significant estimate does not imply clinical irrelevance.

### 4.7 Implications

For investigators using this public file:
- Follow-up time, white-cell count and the medication fields should not be used as baseline predictors.
- Recorded eGFR and its derived CKD fields should be recalculated from creatinine.
- `CaI` should be removed, or its 0.034 value neutralised, and `LV` terminal digits should not be available to models.
- Very high classifier performance on this file should be checked against probes of recording artefacts, and results reported as a sequence of controls rather than as a single number.

For VLST prediction, a clinically useful post-PCI model would require:
- a defined prediction time, for example discharge;
- predictors restricted to information available then, with documented definitions;
- an outcome window of year 1 to year 4 with complete follow-up, or a survival analysis that handles censoring and death;
- external validation;
- an assessment of calibration and of clinical utility.

Any role in decisions about the duration of antiplatelet therapy would further require evidence on the balance of ischaemic and bleeding outcomes under alternative strategies [22,24].

---

## 5. Limitations

1. **Single public file and outcome-dependent recording.** All analyses use one file from one centre. Recording and formatting differ systematically between cases and controls, and their origin is unknown. The controls remove the differences we detected. They cannot exclude ones we did not detect: `LVEF` = 56 is more frequent in cases, the precision probe still reaches AP 0.43 on the raw file, and precision equalisation still lowers R3 AP by about a third.
2. **Post hoc, data-informed controls.** Each control was chosen after a diagnostic on the same file, and none was preregistered. We specified each stage before seeing its results, report every stage, and report sensitivity variants (`CaI` censoring, rounded WBC) that bound the cost of the two removals most open to debate. A reader may still prefer a different set of controls; the run registry and code allow any variant to be reproduced.
3. **Outcome definition.** The label is VLST recorded during variable follow-up. It is not a fixed 1–4-year outcome, and deaths after discharge are undocumented. The intended post-PCI application is unvalidated.
4. **Predictor timing and definitions.** Measurement times for laboratory variables, and the definitions of `LV` and `CaI`, are not documented, and the original authors could not be contacted to confirm them. `LV` remains highly ranked after binning, and whether its magnitude difference is clinical is unknown.
5. **Evaluation design.**
   - One nested cross-validation per configuration.
   - Hyperparameter search with about 14 events per inner fold, which did not improve the conventional arms.
   - Local TabPFN untuned, while the comparators were tuned.
   - Bootstrap intervals conditional on the fitted predictions.
   - No external or temporal validation.
   - Hosted models evaluated through a service whose internal procedures are not documented. Hosted thinking ran at effort high in the unrestricted analysis and at effort micro afterwards, for quota reasons, so hosted results are not comparable across the sequence.
   - LightGBM was not bit-reproducible across two identical runs.
6. **No independent data.** No other cohort with the same variables is available to us, and Wang et al.'s external cohort (n = 2,058) is not public. Every estimate here is internal to one file whose recording differs by outcome. Only independent data could show whether any of this discrimination transfers to other patients.
7. **Association analyses.**
   - About 7 events per covariate.
   - Covariates selected after screening on the same data.
   - Odds ratios for a variable-follow-up label.
   - Possible confounding by indication, especially for post-dilation and antiplatelet therapy.
   - The association model was not re-specified to match R3: it retains WBC and unbinned `LV`.
8. **Attribution analyses.** These were performed on a 70/30 split rather than the benchmark folds, and for TabPFN with the local v3.5 model only, because the hosted model was not explainable within the token allowance. Attributions describe the fitted models, not biology.
9. **Wang comparison.** It was performed on the score's own derivation patients against a binary label, so it does not reproduce or validate the original Cox model.
10. **Angiographic confirmation.** Definite VLST required angiographic confirmation, so events not investigated by angiography may have been missed [1].

---

## 6. Conclusion

In the public derivation file of Wang et al., the near-perfect discrimination of recorded VLST first obtained with TabPFN was produced largely by follow-up-period treatment data and by outcome-dependent recording differences:
- eGFR inconsistent with creatinine in most controls;
- differing number formats;
- inconsistent medication coding in cases;
- a troponin value repeated in a fifth of cases.

When these were addressed in stages, AP fell at every stage and in every model, roughly halving for TabPFN. Under the strictest controls, TabPFN still ranked the recorded label better than tuned conventional learners and the published eight-item score. That residual discrimination is an upper bound for this file, not clinical predictive performance, because:
- recording differences remain;
- one highly ranked variable is undefined;
- the label is not a fixed-horizon outcome.

The work contributes three things:
- a provenance audit that should inform any further use of this dataset;
- a staged benchmark showing how much each control changes the apparent performance of existing models;
- a qualified reference evaluation of a published score.

A post-PCI model for VLST in years 1–4 remains to be developed and validated, with clearly timed predictors, a fully observed outcome window and independent data.

---

## Declarations

- **Ethics.** Secondary analysis of publicly available, de-identified data released under [AUTHOR QUERY: confirm the licence shown on the Figshare record; drafts state CC0 1.0]. The original study received ethics approval (No. 2013-256) and written informed consent [1]. [AUTHOR QUERY: local ethics determination.]
- **Data availability.** Source data: Figshare, doi:10.6084/m9.figshare.7409606 [7]. Analysis code, run manifests and every out-of-fold prediction file: [AUTHOR QUERY: repository URL and commit].
- **Competing interests.** [AUTHOR QUERY]
- **Funding.** [AUTHOR QUERY]
- **Author contributions.** [AUTHOR QUERY: CRediT statement]
- **Use of AI tools.** [AUTHOR QUERY: disclose any AI assistance with analysis or writing, as required by the target journal]
- **Acknowledgements.** We thank Wang et al. for making their data public.

---

## Tables

**Table 1. Characteristics of patients with and without recorded VLST in the public derivation file.** Columns: No VLST (n = 5,093) | VLST (n = 92) | test | P. Continuous variables are mean (SD); binary variables are n (%). Rows as in the current Table 1, plus a row for CKD-EPI 2009 eGFR recalculated from creatinine, age and sex (91.9 vs 91.5). Footnotes:
- (a) `LV` and `CaI` are dataset labels whose definitions are undocumented;
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
| Recorded eGFR within ±2% of any of five equations | 44% | 85% |
| eGFR < 90, recorded / recalculated | 16.9% / 35.6% | 34.8% / 40.2% |
| `DAPT = 1` consistent with aspirin + P2Y12 field | 2,260 / 2,260 | 27 / 35 |
| Previous PCI vs Wang Table 1 | 94 vs 94 | 10 vs 13 |
| AMI as admitting diagnosis vs Wang Table 1 | 3,095 vs 3,046 | 65 vs 72 |
| `CaI` exactly 0.034 | 10 (0.2%) | 20 (21.7%) |
| `LVEF` exactly 56 | 733 (14.4%) | 23 (25.0%) |
| Precision probe (243 grid indicators, no magnitudes) | AP 0.427; ROC-AUC 0.952 | |

**Table 3. Main analysis (R3): pooled out-of-fold discrimination and overall accuracy (75 predictors; 5 outer × 4 inner folds; 92 events).** AP and ROC-AUC with percentile 95% bootstrap intervals (2,000 stratified resamples of stored predictions; no refitting). Brier score (null 0.0174) and scaled Brier score. Tier: T = tuned (inner random search or metric-targeted hosted fitting); U = untuned.

| Model | Interface / checkpoint | Tier | AP (95% interval) | ROC-AUC (95% interval) | Brier | Scaled Brier |
|---|---|---|---|---|---|---|
| TabPFN thinking v3.5 | Hosted, `v3.5_default`, effort micro, AP-targeted | T | 0.475 (0.381–0.583) | 0.957 (0.936–0.975) | 0.0126 | 0.28 |
| TabPFN v3.5 | Local, `tabpfn-v3.5-20260909` | U | 0.465 (0.372–0.567) | 0.947 (0.921–0.969) | 0.0126 | 0.28 |
| XGBoost | Random search, `scale_pos_weight` | T | 0.266 (0.191–0.358) | 0.871 (0.826–0.912) | 0.0180 | −0.03 |
| TabPFN v3 | Local, `v3_default` | U | 0.250 (0.183–0.356) | 0.900 (0.863–0.932) | 0.0152 | 0.13 |
| LightGBM | Random search, balanced | T | 0.229 (0.165–0.322) | 0.870 (0.827–0.911) | 0.0163 | 0.07 |
| CatBoost | Random search, balanced | T | 0.181 (0.123–0.267) | 0.826 (0.781–0.869) | 0.0170 | 0.03 |
| Logistic regression | Random search, balanced | T | 0.094 (0.067–0.148) | 0.783 (0.720–0.837) | 0.1278 | −6.33 |
| Random forest | Random search, balanced | T | 0.068 (0.046–0.117) | 0.772 (0.728–0.812) | 0.0362 | −1.08 |
| Wang integer score (frozen; apparent; not cross-validated) | — | — | 0.103 | 0.801 | — | — |

**Table 3a. Pooled out-of-fold AP across the sequence of controls (same outer folds and seed).** Intervals are 95% bootstrap intervals; the unrestricted intervals are from the September run.

| Model | Unrestricted (80) | R1 (76) | R3 `CaI` censored (76) | **R3 (75)** | R3 + rounded WBC (76) |
|---|---|---|---|---|---|
| TabPFN thinking v3.5 (hosted) | 0.921 (0.878–0.961), effort high | 0.724 (0.639–0.807), effort micro | 0.522 (0.426–0.624) | **0.475 (0.381–0.583)** | not run |
| TabPFN v3.5 (local) | 0.896 (0.845–0.943) | 0.668 (0.583–0.757) | 0.482 (0.391–0.585) | **0.465 (0.372–0.567)** | 0.516 (0.422–0.625) |
| XGBoost | 0.632 (0.533–0.725), defaults | 0.409 (0.313–0.514) | 0.234 (0.151–0.332) | **0.266 (0.191–0.358)** | 0.288 (0.211–0.388) |
| LightGBM | 0.627 (0.531–0.720), defaults | 0.389 (0.294–0.493) | 0.277 (0.195–0.380) | **0.229 (0.165–0.322)** | 0.250 (0.175–0.347) |
| TabPFN v3 (local) | 0.715 (0.626–0.807) | 0.322 (0.237–0.426) | 0.257 (0.189–0.362) | **0.250 (0.183–0.356)** | 0.323 (0.251–0.433) |
| CatBoost | 0.571 (0.475–0.673), defaults | 0.344 (0.251–0.452) | 0.150 (0.098–0.229) | **0.181 (0.123–0.267)** | 0.178 (0.119–0.260) |
| Logistic regression | 0.260 (0.183–0.359), defaults | 0.096 (0.068–0.149) | 0.093 (0.066–0.148) | **0.094 (0.067–0.148)** | 0.137 (0.100–0.205) |
| Random forest | 0.351 (0.266–0.463), defaults | 0.202 (0.147–0.297) | 0.094 (0.064–0.154) | **0.068 (0.046–0.117)** | 0.100 (0.067–0.169) |
| TabPFN thinking v3 (hosted) | 0.832 (0.763–0.895), effort high | — | — | — | — |
| *Local v3.5 after precision equalisation* | *0.844* | *0.443* | *0.322* | ***0.327*** | *0.382* |
| *XGBoost after precision equalisation* | *0.581* | *0.237* | *0.187* | ***0.206*** | *0.210* |

Footnotes:
- Conventional arms were at library defaults in the unrestricted analysis and tuned in all later runs. An R1 run with defaults gave XGBoost 0.447, LightGBM 0.422, CatBoost 0.384, random forest 0.185 and logistic regression 0.123, with identical local TabPFN results.
- Hosted thinking effort differs between the unrestricted analysis (high) and later runs (micro).
- A replicate R3 run gave hosted thinking 0.478 and LightGBM 0.214, with all other arms identical.

**Table 4. Operating point in R3 at inner-fold F1 thresholds, each applied once to its outer held-out fold.** Threshold (mean ± SD over outer folds); TN/FP/FN/TP; sensitivity, precision and F1:
- hosted thinking v3.5: 0.164 ± 0.056; 5,038/55/49/43; 0.47, 0.44, 0.45;
- local v3.5: 0.203 ± 0.065; 5,041/52/47/45; 0.49, 0.46, 0.48;
- XGBoost: 0.287 ± 0.144; 4,988/105/58/34; 0.37, 0.25, 0.29;
- local v3: 0.092 ± 0.010; 5,002/91/60/32; 0.35, 0.26, 0.30;
- LightGBM: 0.071 ± 0.039; 5,027/66/63/29; 0.32, 0.31, 0.31;
- CatBoost: 0.089 ± 0.065; 4,975/118/62/30; 0.33, 0.20, 0.25;
- logistic regression: 0.806 ± 0.167; 4,877/216/67/25; 0.27, 0.10, 0.15;
- random forest: 0.260 ± 0.148; 4,929/164/62/30; 0.33, 0.16, 0.21.

A single F1 cut chosen on the pooled out-of-fold predictions would look better, because it uses the evaluation rows to set the threshold; it is not reported.

**Table 5. Exploratory multivariable logistic regression for the recorded VLST label (13 covariates; continuous per SD; Wald 95% CI), specifications A (recorded eGFR), B (recalculated eGFR) and C (B without clopidogrel).**

| Covariate | A | B | C |
|---|---|---|---|
| WBC (per SD) | 1.97 (1.67–2.33) | 2.04 (1.72–2.42) | 2.04 (1.72–2.42) |
| eGFR (per SD) | 0.57 (0.45–0.72) | 1.18 (0.97–1.44) | 1.19 (0.97–1.45) |
| `LV` (per SD) | 1.83 (1.54–2.18) | 1.93 (1.62–2.30) | 1.91 (1.60–2.27) |
| Stents per lesion (per SD) | 1.42 (0.97–2.08) | 1.39 (0.96–2.00) | 1.39 (0.96–2.01) |
| HbA1c (per SD) | 0.96 (0.72–1.27) | 0.98 (0.75–1.29) | 0.97 (0.74–1.27) |
| Number of diseased vessels (per SD) | 1.21 (0.95–1.54) | 1.32 (1.04–1.67) | 1.32 (1.04–1.67) |
| Total stent length (per SD) | 1.16 (0.78–1.73) | 1.17 (0.80–1.71) | 1.16 (0.79–1.71) |
| Fibrinogen (per SD) | 1.02 (0.85–1.24) | 1.08 (0.90–1.31) | 1.09 (0.90–1.32) |
| `1.1:1Post dilation` | 0.15 (0.08–0.29) | 0.15 (0.08–0.28) | 0.15 (0.08–0.27) |
| Previous PCI | 6.71 (2.88–15.61) | 6.74 (2.93–15.51) | 6.22 (2.73–14.18) |
| Clopidogrel (follow-up field) | 0.48 (0.29–0.79) | 0.51 (0.31–0.82) | not in model |
| Diabetes | 1.45 (0.80–2.65) | 1.58 (0.87–2.87) | 1.67 (0.92–3.02) |
| `PES` | 1.73 (0.95–3.15) | 1.75 (0.98–3.13) | 1.81 (1.01–3.23) |

Footnotes:
- one SD of recorded eGFR is 34.0 and of recalculated eGFR 17.7 mL/min/1.73 m²;
- clopidogrel is a follow-up-period field;
- `LV` is undefined and unbinned here;
- WBC is retained here although excluded from the prediction matrices;
- `1.1:1Post dilation = 1` is read as post-dilation performed (Section 2.10);
- covariates were selected after univariate screening.

**Table 6. Registry of benchmark runs.** All runs used the unmodified file (SHA-256 `0b8cf3da…d93e`), stratified 5 × 4 nested cross-validation with seed 42 and one Tesla T4 GPU. The predictors count excludes identifiers, the outcome and TSSI.

| Run | Date (2026) | Predictors | Removed beyond stage 0 | Transformed beyond stage 0 | Conventional arms | Hosted arm | Software | Role |
|---|---|---|---|---|---|---|---|---|
| Unrestricted | 17 Sep | 80 | — | — | defaults | v3.5 and v3, effort high | tabpfn 9.0.0, client 0.6.0 | Sensitivity |
| R1 (defaults) | 7 Nov | 76 | 4 medication fields | eGFR, CKD5/60/90 recalculated | defaults | v3, effort high | tabpfn 9.0.0, client 0.6.0 | Superseded; footnote only |
| R1 | 16 Nov | 76 | 4 medication fields | eGFR, CKD5/60/90 recalculated | random search (20, AP) | v3.5, effort micro | tabpfn 9.2.0, client 0.7.1 | Sensitivity |
| **R3** | **15 Nov** | **75** | **+ `CaI`** | **+ `LV` 5-mm bins** | **random search** | **v3.5, effort micro** | **9.2.0 / 0.7.1** | **Main** |
| R3 replicate | 14 Nov | 75 | as R3 | as R3 | random search | v3.5, effort micro | 9.2.0 / 0.7.1 | Reproducibility |
| R3 `CaI` censored | 16 Nov | 76 | 4 medication fields | as R1 + `CaI` floored at 0.1 (298 patients) + `LV` bins | random search | v3.5, effort micro | 9.2.0 / 0.7.1 | Sensitivity |
| R3 + rounded WBC | 19 Nov | 76 | as R3, but WBC kept | as R3 + WBC to whole numbers | random search | none | 9.2.0 / 0.7.1 | Sensitivity |

Stage 0, applied in every run, consists of the exclusion of identifiers, TSSI and WBC, the rounding of `Cre`, `CaI`, `Fiberinogen` and `Fast-Glu`, and the stent-brand codebook fitted within training partitions.

**Supplementary material.**
- S0: rationale and evidence for each leakage and recording-artefact control (from `VLST_revision_process_and_defence.md`, Section 3).
- Tables:
  - S1: variable dictionary with availability.
  - S2: univariate screen with false-discovery control.
  - S3: bundled 70/30 leakage contrast, without oversampling.
  - S4: precision probe and precision equalisation in every run.
  - S5: R3 attribution ranks for local TabPFN v3.5 (mutual information, selection frequency in 8 resamples, mean |SHAP|, consensus) and for the seven conventional models (LOCO, SHAP, forward selection), with the R1 ranks for comparison.
  - S6: harmonised feature-screen table.
  - S7: software, hyperparameter search spaces and per-fold tuned settings.
  - S8: calibration summary per arm and run (Brier, null Brier, scaled Brier, O/E, calibration intercept and slope, ECE).
  - S9: recorded eGFR against five equations (agreement within ±2%), and the body weight that Cockcroft–Gault would imply for unmatched controls.
  - S10: replicate R3 run, arm by arm.

## Figure captions

**Figure 1. Precision–recall (left) and ROC (right) curves of pooled out-of-fold predictions in the main analysis (R3).** The dotted line marks the prevalence (0.0177), the AP of a non-informative ranking. Curves describe discrimination of the recorded label after the controls in Section 2.5 (`paper_fig1_pr_roc_curves.png`).

**Figure 2. Calibration of pooled out-of-fold probabilities in R3 (8 quantile bins), axes restricted to the range of the predictions.** Most bins contain predicted risks near zero. Probabilities from class-weighted conventional models, most of all logistic regression, are inflated (`paper_fig2_calibration_curves.png`).

**Figure 3. Observed VLST rate by frozen Wang score.** Bars give the observed rate at each score value; labels give events/patients at that value; the dashed line marks the prevalence. Wang et al.'s risk categories are ≤ 7, 8–9 and ≥ 10. Scores of 14–16 occur in 6 patients, so the bars there (including 1 of 1 at score 16) are not estimates of risk (`paper_fig_s_wang_score_rate.png`).

**Figure 4. Recorded versus recalculated eGFR, by outcome.** Recalculated values use CKD-EPI 2009 with recorded creatinine, age and sex; the line of identity is shown (`paper_fig7_egfr_recorded_vs_recalculated.png`).

**Figure 5. AP across the sequence of controls.** For each arm: unrestricted, R1, R3 with `CaI` censored, R3 and R3 with rounded WBC, with 95% bootstrap intervals; the prevalence is marked. Hosted thinking ran at effort high in the unrestricted analysis and at effort micro afterwards; conventional arms were at defaults in the unrestricted analysis and tuned afterwards (`paper_fig5_ap_sequence_of_controls.png`).

**Supplementary figures.**
- S1–S2: univariate screens.
- S3: univariate versus adjusted odds ratios.
- S4: leakage-contrast PR/ROC curves and confusion matrices.
- S5: Shapley beeswarm, local TabPFN v3.5 fitted on the 70% split (R3 matrix), 1,556 held-out patients; values on the class-1 logit scale.
- S6: partial dependence, training rows (R3 matrix).
- S7: classic-model selector overlap (R3 matrix).

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
15. Levey AS, Stevens LA, Schmid CH, et al. A new equation to estimate glomerular filtration rate. *Ann Intern Med*. 2009;150:604–612. [REFERENCE TO ADD: CKD-EPI 2021 (Inker LA et al., N Engl J Med 2021;385:1737–1749), MDRD (Levey AS et al., Ann Intern Med 2006;145:247–254), Chinese modified MDRD (Ma YC et al., J Am Soc Nephrol 2006;17:2937–2944), Cockcroft–Gault (Nephron 1976;16:31–41) for Table S9]
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
