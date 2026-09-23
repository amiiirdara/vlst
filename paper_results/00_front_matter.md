# Scope, motivation, terminology, and limitations

This note is the manuscript front matter that Parts 1–5 previously lacked. It is written so it does **not** clash with the three scope decisions: `code/failed_hypothesis/` is unused (D1); TabPFN numbers come from `baseline_plus_tabpfn.ipynb` (nested performance; 9-arm anti-leakage dump) and the pair `tabpfn_interpretability_fs_pdp.ipynb` + `tabpfn_interpretability_shap.ipynb` (D2); where a report and a notebook disagree, the notebook is authoritative (D4). Freeze SoT: `paper/frozen_results.yaml`. Live nested: **`nested_cv_v35_antileakage_on`**.

**W1 (leakage)** is the five-flag protocol plus companion controls — not “drop TSSI.” Canonical write-up: [`anti_leakage_protocol.md`](anti_leakage_protocol.md). Numeric twins: Part 4b (`04_tabpfn_rating/leakage_contrast_paper_figures_and_tables.md`). Nested ranking uses the **ALL LEAKS OFF** state. This file also covers **W2–W5**.

---

## Anti-leakage protocol and incentives (W1)

Implemented in nested CV, Part 2, and Part 5 as **ALL LEAKS OFF**. The 70/30 twins invert **all five flags together** (SMOTE on the ON twin only). Freeze flow: drop `NO.` / `Name` / TSSI / WBC; quantize `Cre` / `CaI` / `Fiberinogen` / `Fast-Glu` **before split**; stent encoder **train-fold only**; **no SMOTE**.

**Incentives (why each control exists).**

1. **TSSI dropped.** Wang’s Cox *time axis* recoded as a covariate mixes time-to-event (VLST=1, min 380 d) with completed follow-up (VLST=0, min 1,241 d). A rule “time < 1,241 → event” has zero control false positives.
2. **WBC dropped.** Recording-precision / case-control batch marker (no case has a whole-number WBC; many controls do). Wang also excluded WBC from the Cox score (infection). FDR still ranks it — dual-label.
3. **Labs quantized** (`Cre` 0 / `CaI` 2 / `Fiberinogen` 1 / `Fast-Glu` 1 dp). The raw file is sorted by outcome; cases were transcribed at a different numeric convention. A signature-only probe (243 grid indicators, no magnitudes) reaches AP **0.4270**, ROC-AUC **0.9522**. Rounding is pre-specified and **pre-split**, not a *y*-fit. Equalising leftover precision still drops TabPFN v3.5 nested AP by **0.0518**.
4. **Stent codebook train-fold only** (`Stent type-SES`, `min_count=30`, `encode_stent_on_fold`). Rare brands must not define levels from held-out rows. Nested CV does **not** fit the codebook on the full frame before the split.
5. **No SMOTE** except the ON twin’s train set. Nested CV / Part 2 / Part 5 never synthesise minorities. Quote twins as leakage, not a SMOTE-matched experiment.
6. **Companion:** drop identifiers; clone scaler / OHE inside splits; Part 2/5 catalogues do **not** mask nested CV; twin GridSearch winners are **not** imported. Follow-up drugs (`Aspirin` / `Clopidogrel` / `Ticagrelor` / `DAPT`) **stay in** (post-baseline caveat, not a leak-flag drop).

Nested TabPFN ON vs OFF is **[RE-SOURCE]**. Unlabeled nested 0.9771 / 0.9635 is excluded (pre-anti-leakage dump).

---

## Terminology (W5)

| Term | Use for |
| --- | --- |
| **Association** | Part 1 univariate and multivariable results (full cohort). Part 5 mutual information is on the **train** split of Version 5, not a held-out prediction metric. |
| **Prediction** | Part 4 nested-CV out-of-fold results **only**. That is the only out-of-sample evaluation in this pack. |
| **Interpretation / attribution** | Part 2 selectors; Part 5 SHAP, k-SII, PDP, stability. Model-explanatory, not evidence about patients. |

Do **not** use: “risk factor”, “causal”, “protective”, “independent predictor”, “clinically useful”, or “validated” for any result in this pack. Wang’s 8-variable Cox score *was* externally tested on Shantou data; that word is reserved for **their** score. Nested-CV discrimination on the derivation cohort is not external validation, and it does not transfer to TabPFN by contagion.

An adjusted OR < 1 from the identified **Table 4b** screen (`1.1:1Post dilation` 0.152; `Clopidogrel` 0.480) or a negative PDP shift is a lower modelled odds / probability of recorded VLST, not a treatment benefit (confounding by indication). Do not quote Table 4’s unidentified 17-covariate fit for these examples.

---

## Clinical motivation and what this analysis adds (W2)

**Outcome.** Very late stent thrombosis (VLST) is Academic Research Consortium 2007 *definite* stent thrombosis more than one year after implantation, angiographically confirmed (Wang et al., *Sci Rep* 2020;10:6378; hereafter **Wang 2020**). Probable and possible stent thrombosis are not counted. The file `data/raw/VLST.csv` **is** Wang 2020’s derivation cohort: consecutive ACS patients ≥ 18 years undergoing PCI at The First Hospital of Jilin University, 1 January 2014 – 1 June 2015; 6,038 eligible → 5,185 analysed (236 in-hospital deaths, 413 refused follow-up, 204 lost); **92** definite VLST events (**1.77%**). Median follow-up 1,502 days; median PCI → VLST 697 days. Ethics NO. 2013-256; written informed consent; registered NCT03491891.

**Why it matters (clinical, via Wang).** VLST is rare and late. Wang reviews that stent thrombosis accounts for a substantial share of new myocardial infarction after index PCI and carries several-fold higher adjusted mortality than infarction unrelated to a previously stented site. The intended decision is risk-stratification **more than one year** after PCI (monitoring and therapy after the mandated DAPT year).

**A score already exists.** This is not an empty clinical-prediction field. The Dangas late stent thrombosis score (also used for VLST) had c-statistic 0.66 in Wang’s comparison. Wang derived an **8-variable Cox** VLST score on these same 5,185 rows (diabetes, previous PCI, AMI as admitting diagnosis, eGFR < 90, 3-vessel disease, stents per lesion, SES, no post-dilation) with derivation c-statistic **0.80** and **Shantou** external c-statistic **0.82** (n = 2,058; that file is **not** in this repository). Any claim that “no VLST score exists” is false. This pack scores the published **integer points** as a frozen comparator on the same 5,185 rows (Part 4 Table S-Wang): ROC-AUC **0.8013**, PR-AUC **0.1032**. The Cox linear predictor, decision-curve analysis vs Dangas, and the Shantou file are still absent (B11).

**What “personalised” does *not* mean.** The repository README says “Personalized Risk prediction.” Nothing here is an individual-level model, a patient-specific fine-tune, or a decision-curve analysis. The artefact is a **single global classifier** (or a single global logit for association). Nested-CV probabilities are not portable personalised risks: prevalence, calibration, and PPV are properties of this derivation cohort. We do not use “personalised” as a result claim.

**Why TabPFN is in the comparison.** VLST here is a small-n, mixed-type tabular problem (92 events, 81 raw columns). TabPFN is a tabular foundation model that does in-context learning, handles categoricals natively, and does not run a per-dataset hyperparameter grid. Pins: **`tabpfn==9.0.0`** (local v3 / v3.5) / **`tabpfn-client==0.6.0`** (hosted v3 / v3.5). Thinking-high constructor unchanged. Live nested (anti-leakage ON, 9 arms): thinking v3.5 PR-AUC **0.9212** / TabPFN v3.5 **0.8957** (`nested_cv_v35_antileakage_on`). Unlabeled dump thinking-high **0.9771** / local **0.9635** is excluded. Do not quote Version 4 (0.8553 / 0.6742) either.

**What this pack adds on the *same* derivation cohort, beyond Wang’s score:**

1. **Association catalogue** (Part 1) — FDR-controlled univariate tests, clinical Table C from `VLST.csv`, and an identified 13-covariate logit (Table 4b; the stored 17-covariate Table 4 is not identified). Not a Cox model; not Wang’s eight variables.
2. **Interpretation catalogues** (Parts 2–3, 5) — classic-model LOCO / SHAP / FFS versus FDR names; TabPFN attributions. These do not feed the predictor.
3. **Nested ranking** (Part 4) — nested 5×4 stratified CV of **four** TabPFN arms (v3 vs v3.5; thinking vs local) and five **library-default** classic arms under ALL LEAKS OFF, plus the frozen Wang integer score (Table S-Wang). Not a nested inner GridSearch of boosting.
4. **Leakage control** (Part 4 supplement) — ALL LEAKS ON vs ALL LEAKS OFF on a 70/30 GridSearch split (**five flags flip together**, not TSSI alone), plus GridSearch `best_params_` from the executed notebooks (Table S-TSSI-HP). Those winners are **not** imported into nested CV.

It does **not** add: external or temporal testing of the ML models; a re-fit of Wang’s Cox linear predictor or a Dangas decision curve; a statement that TabPFN is ready for clinical use.

**Data, ethics, consent.** Cite Wang 2020 for NCT03491891, ethics 2013-256, written consent, and their data-availability statement. This repository’s analysis of the derivation file was not separately pre-registered.

---

## Events per variable (W4)

| Comparison | Value | Consequence |
| --- | --- | --- |
| Events / candidate features (81) | 92 / 81 ≈ **1.14** | Far below any conventional EPV rule |
| Events / multivariable logit covariates (17, Table 4) | 92 / 17 ≈ **5.4** | Unidentified spec; do not publish |
| Events / reduced logit covariates (13, Table 4b) | 92 / 13 ≈ **7.1** | One name per collinear block; still below EPV ≥ 10 |
| Events per Part 4 outer fold | 18, 18, 18, 19, 19 | Nested-CV scoreboard is thin |
| Events on the Part 2 val slice | 18 of 1,037 | Selector catalogues are not prediction |

Every **adjusted odds ratio** quoted as the identified screen is from Part 1 **Table 4b** (13 covariates, unweighted logit, **EPV ≈ 7.1**). Table 4 is the stored 17-covariate unidentified fit (**EPV ≈ 5.4**; VIF = ∞ on the post-dilation pair). Both are screening / confounding context, not prediction.

---

## Limitations (W3)

1. **No external or temporal test of the ML models.** Every Part 4 number is nested CV on the 5,185 derivation rows. Wang’s Cox score **was** tested on Shantou (n = 2,058, 1.70% VLST); those rows are not here and **cannot be obtained this cycle** (B11 blocked). Nested CV is not a substitute. The Cox linear predictor and Dangas decision-curve comparison are likewise absent.

2. **Binary classification vs published Cox analysis.** Wang used time-to-event on the follow-up axis. This pack uses a 0/1 label and applies the full anti-leakage OFF protocol because TSSI as a covariate leaks (Part 4 S-TSSI), WBC is a batch/precision marker, lab decimals fingerprint source, and a full-cohort stent codebook would leak rare brands. The frozen integer score on that binary label recovers Wang’s derivation c-statistic (ROC-AUC 0.8013 vs published 0.80; Part 4 S-Wang). That is not a re-fit of the Cox linear predictor, and it is not Shantou. Nested-CV thinking v3.5 PR-AUC **0.9212** vs this frozen score is derivation-cohort nested CV, not external validation. Do not quote excluded dump 0.9771 / 0.9635.

3. **EPV ≈ 7.1** on the identified 13-covariate logit (Table 4b); **EPV ≈ 5.4** on the stored 17-covariate Table 4, which is unidentified (`1.1:1Post dilation` beside `No postdilation`; `eGFR` beside `CKD5` / `CKD90`; Wald SEs/CIs undefined). Quote Table 4b. Still below EPV ≥ 10.

4. **Four TabPFN calibrations.** Live nested Brier: thinking v3.5 **0.0047**, TabPFN v3.5 **0.0048**, thinking v3 **0.0066**, TabPFN v3 **0.0099**. Excluded dump Briers 0.0023 / 0.0025 and Version 4 0.0064 / 0.0102 / 0.0673 are other runs. Do not collapse the arms.

5. **Unequal TabPFN objects.** Thinking arms use the client API (`tabpfn-client==0.6.0`; v3_default / v3.5_default); local arms use `tabpfn==9.0.0` on Kaggle T4 (v3 ckpt / v3.5 safetensors). Classics are untuned defaults.

6. **DAPT columns are post-baseline.** All patients had DAPT for ≥ 1 year; continuation after year 1 was at the treating physician’s discretion. Wang Table 1 “DAPT” is persistence during follow-up, not a discharge prescription. `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT` must not be described as index-PCI covariates without that caveat.

7. **WBC was excluded by the original investigators.** Wang dropped WBC from the Cox score because infection could not be ruled out. The FDR screen still ranks `WBC`. The 2026-09-19 Part 2 dump and 2026-09-20 Part 5 dump **drop WBC** from the ML / TabPFN matrix. Dual-label: FDR-only vs dropped-from-ML.

8. **Unequal tuning (Part 4).** A 9-level stent encoder is fit on the train fold only. Classics then scale + one-hot that column inside each CV split. TabPFN arms see the same 9-level frame natively. Classics are untuned defaults; local TabPFN is not thinking-high; client arms are thinking-high. Part 2/5 catalogues are discovery / attribution, not a mask for Part 4.

9. **PR-AUC CIs and paired test (B3).** Stratified bootstrap CIs on the 9-arm anti-leakage OOF (`n_boot=2000`, seed 42): thinking v3.5 **0.9212 [0.8785, 0.9613]** vs LightGBM **0.6271 [0.5313, 0.7202]**; Δ **0.2941 (0.2071–0.3805)**, P(Δ ≤ 0) = 0/2000. Unlabeled 0.9771 dump CIs are excluded with that dump.

10. **`LV` (and `CaI`) are not named in the CSV.** Until the columns are named, timed, and unit-defined, do not treat `LV` as a novel echo marker. `CaI` means match Wang Table 1 peak troponin I but the file still does not expand the name. Clinical Table C is rebuilt from `VLST.csv` (B7), including both, and does not photocopy Wang’s post-dilation label.

11. **Part 5 is not the Part 4 predictor.** Live dump (2026-09-20): `tabpfn_interpretability_fs_pdp.ipynb` + `tabpfn_interpretability_shap.ipynb`; `tabpfn==9.0.0` / `tabpfn-client==0.6.0` v3.5; ALL LEAKS OFF (IDs+TSSI+WBC dropped; labs quantized; stent train-only; 80 columns; no SMOTE); SFS 8/8 {CaI, LV, eGFR}; SHAP eGFR 1.2288; k-SII row **5176**. Parent `tabpfn_interpretability.ipynb` is archived.

---

## Sources for this note

Wang X, et al. A novel risk model for predicting very late stent thrombosis after percutaneous coronary intervention: a derivation and validation study. *Sci Rep*. 2020;10:6378. doi:10.1038/s41598-020-63455-0.

TabPFN configuration actually used: `code/modeling/rating/baseline_plus_tabpfn.ipynb` (nested 9-arm anti-leakage dump; freeze `nested_cv_v35_antileakage_on`) and `code/modeling/interpretability/tabpfn_interpretability_fs_pdp.ipynb` + `tabpfn_interpretability_shap.ipynb` (attribution). No other TabPFN notebook is in scope.
