# VLST manuscript: revision process, data limitations and defence of the analysis choices

*Companion to `VLST_manuscript_revised.md`, 20 November 2026. It records how the analysis was revised, why each restriction was imposed, what each run did, and which words the manuscript uses for what. Every number below comes from a stored run output in the repository (folders named in Section 4) or from `vlst_data_checks.py`.*

---

## 1. How the revision unfolded

The first draft (September 2026) reported that hosted TabPFN v3.5 in thinking mode discriminated very late stent thrombosis (VLST) almost perfectly in the public Wang et al. derivation file: average precision (AP) 0.921 and ROC-AUC 0.996, against a prevalence of 0.0177. A result that strong, for an event one to four years after PCI, was the trigger for the revision. The work then moved through six stages.

| Stage | Date (2026) | What was done | What it changed |
|---|---|---|---|
| 1. Leakage cleaning | Aug–Sep | Follow-up time (TSSI) and identifiers removed; WBC removed; four laboratory fields rounded; stent-brand codebook fitted inside training folds; bundled with/without-leakage contrast | Produced the "unrestricted" benchmark (Sept 17; 80 features) |
| 2. Editorial and data audit | 21 Oct – 31 Oct | Read-only checks of the raw file (`vlst_data_checks.py`): row order, decimal places, terminal digits, derived-field consistency, agreement with Wang Table 1 | Found outcome-dependent recording differences in eGFR, number formatting, medication coding and `LV` |
| 3. R1 and R2 | 1–7 Nov | R1: follow-up antiplatelet fields removed, eGFR/CKD fields recalculated from creatinine. R2: logistic model refitted with recorded vs recalculated eGFR | AP fell 25–55% in every arm; eGFR odds ratio reversed (0.57 → 1.18 per SD) |
| 4. R1 attribution and R3 | 7–9 Nov | R1 Shapley/selection ranked `CaI` first; the 0.034 placeholder was found; `LV` digit pattern confirmed. R3 specified (drop `CaI`, `LV` in 5-mm bins) and made the main analysis on 9 Nov, before any R3 result existed. CaI-censoring sensitivity specified the same day. Classic arms given an inner random search; SMOTE removed from the contrast notebooks; package pins bumped | Main analysis redefined |
| 5. Runs under the final protocol | 10–19 Nov | R3 (with a replicate on a second account), R3 CaI-censoring, tuned R1, R3 with rounded WBC, R3 Shapley, R3 selection/PDP, R3 classic selectors, leakage contrast without SMOTE | All `[RESULTS PENDING]` markers could be filled |
| 6. Manuscript rewrite | 20 Nov | R3 made the headline throughout; unrestricted and R1 moved to the sensitivity ladder; terminology unified | This document and the revised manuscript |

Two features of this history matter for how the paper must be written:

- **The restrictions were data-informed and post hoc.** Each one was adopted after a diagnostic on the same file. The manuscript says so in Methods (Section 2.5) and Limitations, and it never presents R3 as preregistered.
- **The restrictions were fixed before their own results were seen.** R1 was specified on 1 November and run on 7 November; R3 and its CaI-censoring variant were specified on 7–9 November and first run on 14–15 November. No rule was changed after its results were seen, and every earlier variant is still reported. This is the main protection against the charge of tuning the cleaning until the numbers looked a certain way.

---

## 2. Challenges and limitations of the data

### 2.1 Provenance

- **One public file, one centre, no data dictionary.** The file (`VLST.csv`, Figshare doi:10.6084/m9.figshare.7409606; SHA-256 `0b8cf3da…d93e`, identical in every run manifest) is the derivation cohort of Wang et al. (Sci Rep 2020). No codebook was deposited.
- **The original authors could not be contacted.** Every question that only they could answer (the definitions of `LV` and `CaI`, the construction of TSSI, the timing of medication and laboratory fields, the handling of deaths) is reported as unconfirmed, and the reasoning used in its place is stated.
- **The paper and its deposit disagree in places.** Previous PCI among cases is 10 in the file and 13 in the paper. AMI on admission is 3,095/65 in the file and 3,046/72 in the paper. The HbA1c columns are interchanged in the paper. The intermediate risk group is 1,837 in the paper, which cannot be reconciled with the total. The paper's follow-up medians (1,502 and 697 days) are not reproduced by TSSI. Because the file is unmodified, these are differences between the publication and its own data. We analyse the file as deposited.

### 2.2 The outcome label

- **Not a fixed-horizon outcome.** Events occurred on days 380–1,575 after PCI (12 of 92 after day 1,461). Event-free patients were followed for 1,241–1,605 days, and 3,090 of 5,093 (60.7%) for fewer than 1,461 days. A binary label over this window is "VLST recorded during variable follow-up", not "VLST within four years".
- **Deaths are undocumented.** No event-free patient has follow-up shorter than 1,241 days. Either no one died, which is implausible over four years after ACS, or deaths were excluded or administratively censored. The label therefore cannot be interpreted as a risk in the presence of competing death.
- **Angiographic ascertainment.** Definite VLST required angiography, so events in patients who were not investigated were missed.

### 2.3 Size and precision

- **92 events.** With 18–19 events per outer held-out fold, fold-level AP varies widely (R3 local TabPFN v3.5: 0.313–0.610). With about 14 events per inner fold, hyperparameter search for the conventional arms is noisy. This is a likely reason the tuned conventional arms did not beat the untuned ones in R1 (Section 5).
- **AP is unstable at this prevalence.** A few cases moving rank changes AP visibly. Every AP is therefore reported with a bootstrap interval and fold range, and comparisons use paired bootstrap differences.
- **About 7 events per covariate** in the 13-covariate association model.

### 2.4 Outcome-dependent recording differences

These are the core finding of the audit. They are differences in **how** values were recorded or exported between cases and controls, not in what the values measure.

| Finding | Controls (n = 5,093) | Cases (n = 92) | Handled by |
|---|---|---|---|
| Row position | rows 1–5,093 | rows 5,094–5,185 | Shuffled stratified folds; row number is never a feature |
| Decimal places in raw laboratory text | 3 | 0–2 | Rounding rules; precision probe and equalisation as diagnostics |
| Integer-valued WBC | 3,780 (74.2%) | 0 | WBC removed (WBC-rounding sensitivity run) |
| Recorded eGFR reproducible from creatinine (±1 unit, CKD-EPI 2009) | 44.0% | 54.3% | eGFR and CKD5/60/90 recalculated (R1, R3) |
| Recorded eGFR mean / recalculated mean | 120.0 / 91.9 | 95.9 / 91.5 | as above |
| `DAPT = 1` consistent with the drug fields | 2,260/2,260 | 27/35 | Medication fields removed (R1, R3) |
| `CaI` exactly 0.034 | 10 (0.2%) | 20 (21.7%) | `CaI` removed (R3) or floored at 0.1 (sensitivity) |
| `LV` even terminal digit | 65.1% | 47.8% | `LV` in 5-mm bins (R3) |
| `LVEF` exactly 56 | 14.4% | 25.0% | Not changed (weaker; reported as a residual concern) |
| Precision probe (243 decimal-grid indicators, no magnitudes) | AP 0.427, ROC-AUC 0.952 | | Diagnostic only |

**Is the recorded eGFR simply a different equation?** We tested this before recalculating. Within ±2%, recorded eGFR matched CKD-EPI 2009 for 44.0% of controls and 57.6% of cases. It matched CKD-EPI 2021 for 12.2% / 20.7%, MDRD-175 for 4.0% / 12.0%, MDRD-186 for 7.3% / 23.9%, and the Chinese modified MDRD for 0% / 9.8%. Any of these equations matched only 44% of controls but 85% of cases. For the unmatched controls, a Cockcroft–Gault interpretation would require a median body weight of 110 kg (IQR 101–121), which is implausible for this cohort. Their recorded/recalculated ratio (about 1.57) also rises with age (r = 0.48), whereas creatinine clearance should fall with age. No documented equation explains the controls. CKD-EPI 2009 without the race term reproduces the largest share of both groups, and it is the equation contemporary with the study (2014–2015).

### 2.5 Predictor timing

- The four antiplatelet fields describe therapy **during follow-up**. Aspirin is recorded in only 61.8% of event-free patients, although all patients received at least a year of DAPT, and Wang et al. describe DAPT as received "during follow-up".
- Measurement times for laboratory values (admission or in-hospital) are not documented. `CaI` matches the paper's "peak troponin I", which requires post-procedural sampling, so the earliest defensible prediction time is discharge.
- `LV` has no row in Wang's tables. Its range (37–73, integer) is compatible with a left-ventricular dimension in millimetres, but that is an inference.

### 2.6 Compute and service constraints

- Hosted TabPFN thinking runs through a token-metered service (5 million tokens per day at first, then a 20-million monthly allowance on one account). Effort "high" was quoted at about 149 million tokens for the nested protocol at 75–76 columns. Effort "micro" (about 6.8 million) was the only effort that fitted. From 14 November the only hosted arm is therefore v3.5 at effort micro. The unrestricted run's hosted arms (v3.5 and v3 at effort high) are not comparable with it.
- Shapley values for the hosted model were not feasible. One high-effort fit on the training split is about 5.9 million tokens, and Shapley estimation needs thousands of evaluations. All attribution describes the local v3.5 model.
- Runs were split across two Kaggle accounts. A replicate of R3 on the second account (Section 4) was kept and is reported as a reproducibility check.

---

## 3. Defence of each leakage and artefact control

Each entry gives: what was done, the reason, the evidence, what it costs, and the strongest objection with our answer. The manuscript uses the term **"leakage"** only for information that is unavailable at the prediction time or encodes the outcome (TSSI, follow-up medication). It uses **"outcome-dependent recording difference"** (or "recording artefact" for short, once defined) for the formatting, eGFR, `CaI` and `LV` findings. The collective term is **"leakage and recording-artefact controls"**.

### 3.1 Removing follow-up time (TSSI)

- **What.** `Time since stent implantation` excluded from every prediction matrix.
- **Why.** For cases it is the time to VLST; for controls it is the time to the end of follow-up. It is defined by the outcome, and it is not known at the index PCI.
- **Evidence.** Its ranges by outcome (cases 380–1,575; controls 1,241–1,605 days) let a model flag most cases from TSSI alone. In the bundled leakage contrast (TSSI kept, WBC kept, no rounding, codebook on the full cohort; no SMOTE in either condition), hold-out AP was 0.12–0.71 higher with leakage in all seven models. For example, LightGBM reached 0.959 with leakage and 0.668 without, and logistic regression 0.888 and 0.343.
- **Cost.** None for a baseline predictor. Follow-up time belongs on the time axis of a survival model.
- **Objection.** "The contrast mixes four changes." Answer: yes. The manuscript attributes the gap to the leakage sources together, never to TSSI alone.

### 3.2 Removing WBC

- **What.** `WBC` removed from the benchmark matrices. It is retained in the association analyses.
- **Why.** Its recording precision separates the groups: 74.2% of controls have integer values and no case does. The integer-WBC indicator alone has ROC-AUC 0.871.
- **Evidence.** Wang et al. also left leukocyte count out of their score, though for a clinical reason (infection). The WBC-rounding sensitivity run keeps WBC rounded to whole numbers for every patient, so the integer pattern carries no information. In it, local TabPFN v3.5 AP rose from 0.465 to 0.516 (paired difference 0.051, 95% interval 0.002–0.105) and XGBoost from 0.266 to 0.288 (0.022, −0.043 to 0.093).
- **Cost.** WBC magnitude may carry real inflammatory signal. The sensitivity run shows how much: a small gain for TabPFN, none detectable for XGBoost.
- **Objection.** "Rounding WBC should have been the main rule rather than dropping it." Answer: rounding removes the integer pattern but cannot show that no other WBC artefact remains. With WBC kept, the main analysis would rest on a variable whose recording is known to differ by outcome. Dropping is the conservative choice, and the sensitivity run bounds what it costs.

### 3.3 Rounding four laboratory variables

- **What.** `Cre` to 0 decimal places, `CaI` to 2, `Fiberinogen` to 1 and `Fast-Glu` to 1, before splitting.
- **Why.** These fields carry extra decimals in one group's raw text.
- **Why before splitting is not leakage.** The rule is fixed and row-wise. It uses no outcome and no statistic from other rows, so applying it before or after the split gives identical values.
- **Cost.** Minimal, because the retained precision is clinically meaningful (for example, whole µmol/L for creatinine).
- **Objection.** "Rounding did not remove the `CaI` placeholder." Answer: correct (0.034 becomes 0.03 in 20 cases and 16 controls), which is why R3 added a separate rule for `CaI`.

### 3.4 Stent-brand codebook inside training folds

- **What.** Brand strings were canonicalised; brands with fewer than 30 training observations were grouped as "other", with the grouping learned in each training partition. Unseen brands in held-out data map to "other".
- **Why.** A codebook learned on all rows lets rare, case-only brand strings define their own levels. That is preprocessing leakage from held-out rows.
- **Cost.** None.

### 3.5 Removing the four antiplatelet fields (R1, R3)

- **What.** `Aspirin`, `Clopidogrel`, `Ticagrelor` and `DAPT` removed.
- **Why.** They are recorded during follow-up (aspirin in only 61.8% of controls despite universal DAPT for a year; "during follow-up" in Wang et al.), and their internal coding differs by outcome (DAPT consistent in 2,260/2,260 controls and 27/35 cases).
- **Evidence.** R1, which removed these fields and recalculated eGFR together, lowered local TabPFN v3.5 AP from 0.896 to 0.668.
- **Cost.** If therapy were known at discharge it would be a legitimate predictor. Its recording time cannot be confirmed, so it cannot be used at a post-PCI prediction time.
- **Objection.** "R1 changes two things at once." Answer: acknowledged. The manuscript does not split the R1 fall between medication and eGFR.

### 3.6 Recalculating eGFR, CKD5, CKD60 and CKD90 (R1, R3)

- **What.** eGFR recalculated with CKD-EPI 2009 (no race term) from `Cre` (µmol/L ÷ 88.4), `Age` and `Men`. The CKD stage and thresholds were rebuilt from it.
- **Why.** Recorded eGFR is not a function of recorded creatinine for most controls, while creatinine itself is identical between groups (72.5 vs 72.4 µmol/L). The recorded values differ by outcome (means 120.0 vs 95.9), and the recalculated values do not (91.9 vs 91.5).
- **Why the recalculation is not leakage.** It uses only the patient's own baseline creatinine, age and sex, with a published equation and no fitted parameter.
- **Evidence.** In R2 the eGFR odds ratio moved from 0.57 (0.45–0.72) to 1.18 (0.97–1.44) per SD. In the earlier attribution run, forward selection had picked recorded `eGFR` in 8 of 8 resamples and `Cre` in 7 of 8; after recalculation they were picked in 1 and 0. In the R3 classic selectors, `eGFR` is first in only 1 of 21 lists (it was first in 10 of 21 in the September run). In R3 Shapley values, recalculated `eGFR` ranks 32nd of 75.
- **Cost.** If the recorded values came from a later or different creatinine, recalculation discards it. That information is not usable at the prediction time anyway, because its timing is unknown.
- **Objection.** "Another equation (e.g. Cockcroft–Gault, which needs weight) explains the recorded values." Answer: Section 2.4. No published equation reproduces the unmatched controls, and the Cockcroft–Gault weights implied are implausible.

### 3.7 Removing `CaI` (R3), with a censoring sensitivity

- **What.** `CaI` removed in R3. In the sensitivity variant it is kept, with every value below 0.1 set to 0.1 (298 patients: 21 cases, 277 controls).
- **Why.** The exact value 0.034 appears in 20 of 92 cases and 10 of 5,093 controls, and no other value repeats more than twice among cases. This is the signature of a placeholder or assay limit used mainly for one group. In the R1 attributions, `CaI` ranked first by mutual information, selection frequency and mean |SHAP| (1.29, twice the next feature).
- **Evidence that the drop removes artefact rather than clinical information.** Censoring hides the 0.034 value among the low values while keeping every troponin of 0.1 or more. Under censoring, local TabPFN v3.5 AP was 0.482, against 0.465 with `CaI` dropped (paired difference 0.017, −0.019 to 0.055). Against R1 it was 0.186 lower (0.12–0.25). Most of what `CaI` contributed in R1 therefore disappears once the placeholder is hidden, even though the troponin magnitudes are kept. The hosted thinking arm gained a little more from the censored `CaI` (0.522 vs 0.475; 0.046, 0.011–0.079). Censoring cannot remove the between-group difference in the share of low values, which may be real or may itself reflect the placeholder, so this variant is reported beside R3, never instead of it.
- **Cost.** Loss of peak troponin as a predictor. The censoring run bounds that loss.

### 3.8 `LV` in 5-mm bins (R3)

- **What.** `LV` rounded to the nearest 5 mm.
- **Why.** The terminal-digit distribution differs by outcome (even digit in 65.1% vs 47.8%; digit 0 in 1.3% of controls). Binning to 5 mm makes the last digit carry no information while keeping the magnitude.
- **Evidence.** `LV` still ranks high in R3: selected in 8 of 8 resamples, mutual-information rank 2 and Shapley rank 7. Its magnitude differs between groups (mean 44.6 vs 49.1), and binning does not touch that. Whether the magnitude difference is clinical is unknown, because the variable is undefined.
- **Cost.** Resolution of a variable whose definition is unknown. No physiological claim is made either way.
- **Objection.** "Binning cannot remove every recording difference in `LV`." Answer: agreed. The manuscript lists `LV` among the residual concerns, and the R3 result is still described as an upper bound.

### 3.9 Evaluation design

- **Nested cross-validation.** Stratified 5 outer × 4 inner folds, seed 42, with the same outer folds for every arm and every variant. The inner loop chooses the F1 threshold and, for the conventional arms, hyperparameters (random search of 20 configurations scored by AP). Nothing is selected on the outer held-out fold. All preprocessing and the codebook are refitted in each training partition.
- **Hosted thinking** optimises AP during fitting, inside each training partition only.
- **Tuning tiers.** Conventional arms (tuned) and hosted thinking (metric-targeted) form the optimised tier. The local TabPFN arms are untuned defaults, used as documented in-context learners. A search over TabPFN settings would cost a full GPU refit per configuration. This asymmetry favours the comparators, not TabPFN.
- **Uncertainty.** 2,000 stratified bootstrap resamples of the stored pooled out-of-fold predictions (seed 42), paired for differences, without refitting. The intervals therefore describe sampling of evaluation rows, not fitting variability. Fold ranges and the replicate run cover part of the rest.
- **Diagnostics that are not cleaning rules.** The **precision probe** shows that the formatting alone carries outcome signal. **Precision equalisation** rounds every numeric field to its coarsest raw precision and refits. It also discards clinical resolution, so a fall in AP is an upper bound on the artefact share, not a measure of it.

### 3.10 What the controls do not achieve

- They do not make the label a fixed-horizon outcome.
- They do not confirm any variable definition.
- They cannot exclude recording differences we did not detect (for example `LVEF` = 56; the precision probe still reaches AP 0.427 on the raw file, and equalisation still lowers R3 AP).
- They do not create independent data.

The manuscript therefore calls the R3 numbers **upper bounds for this file** and never calls them clinical performance.

---

## 4. Registry of runs and settings

All benchmark runs use the same file hash, stratified 5 × 4 nested CV with seed 42, Python 3.12.13, scikit-learn 1.6.1, XGBoost 3.2.0, LightGBM 4.6.0, CatBoost 1.2.10 and PyTorch 2.10.0 on one Tesla T4. The local checkpoints are `tabpfn-v3-classifier-v3_default.ckpt` and `tabpfn-v3.5-20260909.safetensors`. The November runs use tabpfn 9.2.0 and tabpfn-client 0.7.1; the September run used 9.0.0 and 0.6.0.

| Run (folder under `code/modeling/rating/Kaggle_baseline_plus_tabpfn_results/`) | Date | Features | Removed | Changed | Conventional arms | Hosted arm(s) | Role in paper |
|---|---|---|---|---|---|---|---|
| Unrestricted (September; superseded folder, figures in `paper_results/`) | 17 Sep | 80 | IDs, TSSI, WBC | 4 lab fields rounded | defaults | v3.5 and v3, effort high | Top of the sensitivity ladder |
| R1 untuned (7 Nov; notebook output only) | 7 Nov | 76 | + 4 medication fields | + eGFR/CKD recalculated | defaults | v3, effort high | Superseded by tuned R1; quoted where it adds information |
| `r1_postpci/` (Version 15) | 16 Nov | 76 | as R1 | as R1 | tuned (20 draws, AP) | v3.5 micro (quote 6.77 M) | Sensitivity |
| **`r3_residual/` (Version 13)** | **15 Nov** | **75** | **+ `CaI`** | **+ `LV` 5-mm bins** | **tuned** | **v3.5 micro (quote 6.77 M)** | **Main analysis** |
| `r3_residual_rep2/` (second account) | 14 Nov | 75 | as R3 | as R3 | tuned | v3.5 micro | Reproducibility check |
| `r3_caicensor/` (Version 14) | 16 Nov | 76 | as R1 | `CaI` floored at 0.1 (298 rows) + `LV` bins | tuned | v3.5 micro (6.77 M) | Sensitivity |
| `r3_residual_wbcround/` (Version 16) | 19 Nov | 76 | as R3 but WBC kept | WBC rounded to integers | tuned | none (local arms only) | Sensitivity |

The two R3 runs used identical code and settings. Logistic regression, random forest, XGBoost, CatBoost and both local TabPFN arms produced bit-identical out-of-fold predictions. Hosted thinking differed slightly (AP 0.475 vs 0.478; paired difference 0.002, −0.008 to 0.012; correlation of predictions 0.998), as expected from a remote service. LightGBM differed (0.229 vs 0.214); its cause was not investigated, and LightGBM is not bit-reproducible across these runs. The 15 November run is the main record. It is also the more conservative of the two for TabPFN's lead, because its thinking AP is the lower and its LightGBM AP the higher.

Other analyses:

| Analysis | Folder | Date | Settings |
|---|---|---|---|
| Leakage contrast, with leakage | `Kaggle_baseline_tssi_leakage_results/tssi_leakage/` | 18 Nov | 70/30 split (seed 42), 90 columns, TSSI and WBC kept, no rounding, codebook on full cohort, no SMOTE, class weights, GridSearchCV 5-fold, AP |
| Leakage contrast, without leakage | `Kaggle_baseline_without_tssi_results/without_leakage/` | 19 Nov | Same split, 88 columns, unrestricted cleaning (Section 3.1–3.4), no SMOTE |
| Classic selectors, R3 | `interpretability/Kaggle_baseline_intrepretability_results/r3_residual/` | 18 Nov | 7 models × LOCO, SHAP and forward selection, AP; fit/validation 4,148/1,037 (74/18 events); 75 raw features, 82 after encoding |
| Shapley values, R3 | `interpretability/Kaggle_tabpfn_intrepretebility_results/shap_r3_residual/` | 10 Nov | Local v3.5, 70/30 split (3,629/1,556; 64/28 events), shapiq 1.8.0 via tabpfn-extensions 0.6.3, baseline imputer, budget 256, logit scale (reconstruction error 0.0014 on logit) |
| Selection, MI, PDP, R3 | `.../fs_pdp_r3_residual/` | 16 Nov | Same split; MI k = 3; forward selection 8 features × 8 seeds (42–49), 3-fold, AP, `n_estimators = 1`; PDP on training rows |
| Wang score | `wang_vlst_score.ipynb` | 20 Nov | Frozen published points; recorded and recalculated `CKD90` |
| Associations (R2) | `eda.ipynb` | 6 Nov | 13-covariate logistic, A/B/C |

---

## 5. Results across the sensitivity ladder

Pooled out-of-fold AP, with 95% bootstrap intervals for the November runs.

| Arm | Unrestricted (Sep) | R1 tuned | R3 CaI-censored | **R3 (main)** | R3 + rounded WBC |
|---|---|---|---|---|---|
| TabPFN thinking v3.5 (hosted) | 0.921 (effort high) | 0.724 (0.639–0.807) | 0.522 (0.426–0.624) | **0.475 (0.381–0.583)** | not run |
| TabPFN v3.5 (local) | 0.896 | 0.668 (0.583–0.757) | 0.482 (0.391–0.585) | **0.465 (0.372–0.567)** | 0.516 (0.422–0.625) |
| XGBoost | 0.632 (defaults) | 0.409 (0.313–0.514) | 0.234 (0.151–0.332) | **0.266 (0.191–0.358)** | 0.288 |
| LightGBM | 0.627 (defaults) | 0.389 (0.294–0.493) | 0.277 (0.195–0.380) | **0.229 (0.165–0.322)** | 0.250 |
| TabPFN v3 (local) | 0.715 | 0.322 | 0.257 | **0.250 (0.183–0.356)** | 0.323 |
| CatBoost | 0.571 (defaults) | 0.344 | 0.150 | **0.181 (0.123–0.267)** | 0.178 |
| Logistic regression | 0.260 (defaults) | 0.096 | 0.093 | **0.094 (0.067–0.148)** | 0.137 |
| Random forest | 0.351 (defaults) | 0.202 | 0.094 | **0.068 (0.046–0.117)** | 0.100 |
| Wang score (frozen, apparent) | 0.103 | | | 0.103 | |

Prevalence (AP of a non-informative ranking) is 0.0177.

Main points:

1. **Every control lowered discrimination, and the fall was largest for the most flexible models.** For local v3.5, which used the same checkpoint and settings throughout, AP went 0.896 → 0.668 → 0.465.
2. **A lead for TabPFN survives R3.** Local v3.5 minus XGBoost is 0.199 (0.105–0.292), and in 0 of 2,000 resamples was the difference ≤ 0. Hosted thinking minus XGBoost is 0.209 (0.107–0.310). This lead is measured against tuned comparators.
3. **Hosted thinking at effort micro did not differ from local v3.5 in R3** (0.010, −0.032 to 0.055). It was ahead in R1 (0.055, 0.017–0.093) and under censoring (0.039, 0.004–0.074).
4. **Tuning the conventional arms did not help.** In R1, where both exist, tuned AP was lower than default AP for XGBoost (0.409 vs 0.447), LightGBM (0.389 vs 0.422), CatBoost (0.344 vs 0.384) and logistic regression (0.096 vs 0.123), and higher only for random forest (0.202 vs 0.185). A plausible reason is that inner folds carry about 14 events, so the search fits noise. The TabPFN lead is therefore not an artefact of untuned comparators. The local TabPFN arms gave identical AP in the untuned and tuned R1 runs (0.668, 0.322), which also confirms that the package update (9.0.0 to 9.2.0) did not change the local model.
5. **Residual formatting still matters in R3.** Precision equalisation lowered local v3.5 from 0.465 to 0.327 and XGBoost from 0.266 to 0.206. Part of this is lost clinical resolution.
6. **Calibration in R3.** Null Brier is 0.0174.
   - Local v3.5: scaled Brier 0.28, O/E 1.01, slope 1.10.
   - Thinking: scaled Brier 0.28, O/E 1.23, slope 1.20.
   - XGBoost: −0.03 (no better than the prevalence), O/E 0.68, slope 0.56.
   - Logistic regression: −6.33, from class weighting.
7. **Operating point in R3 (inner-fold F1 thresholds applied to the outer folds; `nested_cv_operating_point.csv`, not the optimistic pooled F1 cut in `model_comparison.csv`).**
   - Thinking: 43 of 92 events with 55 false positives (sensitivity 0.47, precision 0.44).
   - Local v3.5: 45 of 92 with 52 false positives (0.49, 0.46).
   - XGBoost: 34 of 92 with 105 false positives (0.37, 0.25).
8. **Attribution in R3.** No variable dominates any more.
   - **Mean |SHAP|, top six:** HbA1c 0.41, LDL 0.36, stent brand 0.34, no post-dilation 0.33, previous PCI 0.31 and 1.1:1 post-dilation 0.31. `LV` is 7th (0.28).
   - **Forward selection:** HbA1c and `LV` were selected in 8 of 8 resamples, `Fast-Glu` in 7, and LDL, LVEF, no post-dilation and stent brand in 5.
   - **Mutual information:** LDL, `LV`, no post-dilation and HbA1c lead.
   - **Partial dependence:** previous PCI +0.010 on a training prevalence of 0.018; post-dilation −0.005.
   - **Classic selectors:** first-ranked features are spread out (HbA1c 5 of 21 lists; LDL, fibrinogen and total stent length 3 each; recalculated eGFR 1). No post-dilation appears in the top 20 of 15 of 21 lists and previous PCI in 9.
9. **Wang score.** Recorded `CKD90` gives ROC-AUC 0.801 and AP 0.103. Recalculated `CKD90` gives 0.789 and 0.090.

---

## 6. Terminology used in the manuscript

| Use | Meaning | Avoid |
|---|---|---|
| "discrimination of the recorded VLST label" | What every benchmark number measures | "prediction of VLST", "risk prediction", "predictive performance" (except when describing the intended, unachieved use) |
| "out-of-fold", "internal (cross-validated)" | Evaluation inside this file | "validation", "validated", "test set performance" for the nested CV |
| "apparent" | Wang score on its own derivation rows | "validation of the Wang score" |
| "average precision (AP)" | scikit-learn step-wise AP | "PR-AUC" (except where quoting stored column names) |
| "data leakage" | TSSI, follow-up medication, codebook fitted on all rows | Using it for formatting or eGFR differences |
| "outcome-dependent recording difference" / "recording artefact" | Formatting, eGFR inconsistency, `CaI` placeholder, `LV` digits, medication coding | "fraud", "manipulation", "error by the authors" (origin is unknown) |
| "leakage and recording-artefact controls" | All cleaning rules together | "anti-leakage" in the paper (fine in code) |
| "main analysis (R3)", "sensitivity analysis" | R3 vs the rest | "primary benchmark" for the unrestricted run (it is now the "unrestricted analysis") |
| "upper bound for this file" | Status of R3 numbers | "true performance", "clinical performance" |
| "hosted TabPFN thinking v3.5 (effort micro)" / "(effort high)" | Always state the effort | "thinking TabPFN" without effort |
| "tuned tier" / "untuned tier" | Optimisation status | "optimised models" without saying how |
| "associated with", "odds ratio for the recorded label" | R2 results | "risk factor", "protective", "effect" |
| "interval includes 1" | Uncertainty | "trend toward significance" |
| "attribution", "the local model used" | SHAP, selection, PDP | "important predictors of VLST", "drivers" |
| "bootstrap P < 0.0005 (0 of 2,000)" | Paired differences | "P = 0" |
| `LV`, `CaI` (code font) | Undefined dataset labels | "LV dimension", "troponin" as established facts |

By section:

- **Methods** describes what was done, in the past tense, with dates of specification.
- **Results** reports numbers without interpretation of mechanism.
- **Discussion** interprets, always with the label caveat.
- **Limitations** names what the controls cannot do.

---

## 7. What changed in the manuscript

- **Title, abstract and conclusion:** R3 is the headline. The unrestricted 0.921 now appears only as the starting point of the ladder.
- **Methods 2.5** is rewritten as a numbered list of controls, each with its reason and its specification date. A new Section 2.6 is a run registry (Table 6), and 2.7 gives the tuning tiers and hosted-effort statement. The software list now covers both package sets.
- **Results:**
  - 3.3 is now the R3 main result.
  - 3.4 covers the sensitivity ladder (unrestricted → R1 → R3, censoring, WBC rounding, replicate).
  - 3.5 is the leakage contrast without SMOTE.
  - Attribution is reported for R3, with R1 as context.
- **Discussion** is reorganised around three claims:
  - most of the original discrimination was artefact or leakage;
  - a lead remains under R3, against tuned comparators;
  - R3 is still an upper bound.

  Clinical associations keep the R2 caveats, plus a note that R2 still contains WBC and raw `LV`.
- **Limitations** adds post hoc specification, the hosted effort change, the LightGBM non-reproducibility, inner-fold event counts for tuning, and the residual `LVEF` pattern.
- **Tables:**
  - Table 3 is now R3.
  - Table 3a is the sensitivity ladder.
  - Table 4 is the R3 operating point.
  - Table 6 is the run registry.
  - Tables S3 (contrast) and S5 (attribution) are updated.
  - New Table S9 is the eGFR equation comparison.

## 8. Figures and tables (regenerated 20 November)

Every paper figure and table was rebuilt from the stored outputs; nothing was refitted.

| Script | Output folder | What changed |
|---|---|---|
| `code/modeling/tools/rebuild_part4_from_kaggle_dump.py` | `paper_results/04_tabpfn_rating/` | Rewritten for the one-folder-per-setting layout. Main tables are from R3. New outputs: Table 3a and Figure 5 (sequence of controls), Table S4 (precision equalisation, every run), Table S8 (calibration, every run), Table S10 (replicate), and paired contrasts within and between runs. Figure 1 is redrawn from the OOF file. The Wang table is left to the Wang notebook. |
| `code/modeling/rating/rebuild_tssi_leakage_table.py` | same | Reads `test_metrics.csv` and the notebooks' "Best params" prints instead of hard-coded September numbers; copies the contrast curve and confusion panels (Figure S4). |
| `code/modeling/rating/wang_vlst_score.ipynb` | same | Re-executed. Figure 3 now labels every bar with events/patients. |
| `code/modeling/tools/rebuild_part2_paper_figures.py` | `paper_results/02_ml_selectors/` | Reads the R3 selector run; titles say AP. |
| `code/modeling/tools/rebuild_part3_paper_figures.py` | `paper_results/03_stats_vs_ml/` | Rewritten to be data-driven. Each feature is classified from the screen, the logistic model and the R3 selectors, and a statistical-screen feature absent from R3 is labelled with the control that removed or recalculated it. The hand-written explanations tied to the September run are gone. `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb` is retired to a wrapper that runs this script and shows its table. |
| `code/modeling/tools/rebuild_part5_from_split_dumps.py` | `paper_results/05_tabpfn_interpretability/` | Labels come from the run provenance (local v3.5, 75 features), not from the September thinking-high SHAP run. Two superseded images were removed. |

**Correction made during the rebuild.** The first R3 draft of the manuscript quoted operating-point counts from `model_comparison.csv`. That file's F1 cut is chosen on the pooled out-of-fold predictions, so it is optimistic. The manuscript now uses `nested_cv_operating_point.csv`, where each inner-fold threshold is applied once to its outer fold. For hosted thinking that is 43 of 92 events with 55 false positives.

## 9. Still open

- The `paper_results/*/…_paper_figures_and_tables.md` reports describe the September run; the manuscript and this document supersede them.
- Author queries: authorship, ethics determination, licence, funding, AI-use statement, repository URL.
- Optional: an R2 model D aligned with R3 (no WBC, `LV` in bins), so that the association analysis uses the same matrix as the main benchmark.
