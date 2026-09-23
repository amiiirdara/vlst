# Discussion

`photo_2026-08-30_21-48-00.jpg` was not in the repository or home tree. Critique themes here follow `paper_results/00_front_matter.md` (W3–W5), `paper/audit/unsupported_claims.md`, and `docs/paper_evidence_map.md` (W1/B11/B15). Numbers are only those frozen in `paper/frozen_results.yaml`. Association ≠ prediction ≠ attribution (**W5**). No causal or clinical-utility claims.

<!-- TRACE: W5 terminology; W3 banned phrases; unsupported_claims §1 -->

---

## 1. Leakage as the methodological contribution

The first result that should change how this derivation file is modelled is not a TabPFN score. It is that a published-style 70/30 GridSearch pipeline **overstates hold-out ranking** when follow-up time and related leak flags remain in the matrix. <!-- TRACE: W1; TODO-LEAK; Table S-TSSI; YAML leakage_contrast -->

On the paired twins, ALL LEAKS ON versus ALL LEAKS OFF (same seven classics, same 3,629/1,556 split) drops PR-AUC by **0.30–0.61** (LightGBM 0.9687 → 0.6675; logistic regression 0.9134 → 0.3431; decision tree 0.7524 → 0.1378). F1 at the notebook cut collapses as well (random forest **0.8333 → 0**; boosters 0.9231 → 0.41–0.55). ROC-AUC moves less because 1,528/1,556 hold-out rows are non-events; Gaussian NB ROC-AUC even rises slightly while PR-AUC still falls. <!-- TRACE: kaggle_tssi_leakage / kaggle_without_tssi test_metrics; Results §4 -->

Those arms are **not** “identical except TSSI.” `KEEP_TSSI`, `DROP_WBC`, `QUANTIZE_CLINICAL`, `STENT_ENCODER_TRAIN_ONLY`, and `USE_SMOTE` flip together. Nested CV, Part 2, and Part 5 implement the OFF state for **all five**, plus companion controls: identifiers dropped; scaler/OHE cloned inside splits; selector catalogues not used as a nested mask; twin GridSearch winners not imported. Nested CV does not use SMOTE.

**Incentives.** TSSI is mixed time-to-event vs completed follow-up. WBC is a recording-precision / batch marker (Wang also excluded it from Cox). Named labs are quantized **before** split because decimal-grid membership alone ranks (signature probe: 243 indicators, AP **0.4270**, ROC-AUC **0.9522**); leftover precision still moves TabPFN v3.5 nested AP by **0.0518**. The stent codebook is fit on each **train fold** so rare brands cannot leak from held-out rows. SMOTE on the ON twin is unmatched inflation. Follow-up drugs stay in with a post-baseline caveat. Canonical write-up: `paper_results/anti_leakage_protocol.md`. <!-- TRACE: W3.7–W3.8; B15 anti_leakage_flow; Methods flags; leakage_precision_probe -->

**How much of the apparent v3.5 advantage survives that handling?** Nested ranking is already on anti-leakage ON. After those drops, TabPFN thinking v3.5 still has the highest nested PR-AUC among nine fitted arms (**0.9212** [0.8785, 0.9613]) and TabPFN v3.5 is second (**0.8957** [0.8446, 0.9426]), both above **library-default** XGBoost **0.6322** and LightGBM **0.6271** (paired Δ thinking v3.5 − LightGBM **0.2941** (0.2071–0.3805), P(Δ ≤ 0) = 0/2000). That ranking is not a nested inner GridSearch of boosting. <!-- TRACE: nested_cv_v35_antileakage_on; W3.9 B3 -->

What does **not** survive is treating a leaks-on 70/30 scoreboard, or the excluded unlabeled nested dump (thinking **0.9771** / local **0.9635**), as the TabPFN result. **[RE-SOURCE]:** there is no matched 9-arm nested dump with ALL LEAKS ON, so a numeric “nested v3.5 ON minus nested v3.5 OFF” cannot be written. The honest statement is: classic GridSearch ranking on this file is highly sensitive to TSSI/WBC/quantize/encoder/SMOTE; the nested TabPFN comparison that remains is the anti-leakage-ON 5×4 OOF. <!-- TRACE: historical_unlabeled_two_arm excluded; W3.2 -->

Wang dropped WBC from the Cox score because infection could not be ruled out. The FDR catalogue still ranks WBC; the ML matrix does not. That is a dual label, not a validated inflammatory marker. <!-- TRACE: W3.7; unsupported_claims WBC -->

---

## 2. TabPFN v3 versus v3.5, thinking versus local

v3 and v3.5 are different checkpoints; thinking-high (`tabpfn-client`, effort high, metric `average_precision`) and local `tabpfn` are different objects. They are not interchangeable (**W3.4–W3.5**). <!-- TRACE: W3.5; run_manifest pins 9.0.0 / 0.6.0 -->

On anti-leakage nested OOF, v3.5 outranks v3 in both thinking (0.9212 vs 0.8319) and local (0.8957 vs 0.7150). Thinking outranks local within each family, but **not in every fold** (fold 1: local v3.5 PR-AUC 0.9365 vs thinking 0.8802). With **92** events and **18–19** events per outer fold, fold-to-fold movement is expected. Ranking uses PR-AUC because prevalence is **0.0177**; ROC-AUC is high for every arm (0.865–0.996) and is the less informative scale. Bootstrap CIs are stratified resamples of stored OOF (`n_boot=2000`, seed 42), not re-fits. Classics are a library-default reference panel; TabPFN has no per-dataset grid — unmatched effort is in the ranking claim, not only in Limitations. <!-- TRACE: W4 fold events; W3.8–W3.9; study.event_rate -->

A single nested 5×4 CV (`random_state=42`) is not repeated nested CV. Differences between arms should be read as derivation-cohort ranking with wide rare-event uncertainty, not as a settled architecture ranking. <!-- TRACE: validation.repetitions; W3.1 -->

---

## 3. Calibration and thresholds — not clinical utility

Live nested Brier scores: thinking v3.5 **0.0047**, TabPFN v3.5 **0.0048**, thinking v3 **0.0066**, TabPFN v3 **0.0099** (ECE, 8 quantile bins: 0.0028, 0.0003, 0.0036, 0.0059). Reliability curves are Part 4 Figure 2. Logistic regression Brier **0.0788** / ECE **0.116** is a different calibration object. Do not collapse arms into “TabPFN is well/poorly calibrated.” <!-- TRACE: W3.4; kaggle_baseline_plus_tabpfn.calibration_ece; Part 4 Fig 2 / Table S-ECE -->

Nested operating points (inner-fold F1 thresholds, Table 2 — not pooled Table 3): thinking v3.5 mean *t* 0.318 ± 0.059, 5083/10/15/77, recall 0.8370, F1 0.8603. Pooled F1 cuts are optimistic and are not the headline. Accuracy is high because negatives dominate. <!-- TRACE: nested_cv_operating_point; W5 prediction = nested OOF -->

**Decision-curve analysis is absent** (B11). There is no net-benefit curve versus Dangas or treat-all/treat-none. Nested PPV/NPV at an F1 cut on this derivation file is not a bedside rule and is not “clinical utility” (**W3**, **W5**). Calibration and threshold numbers describe this OOF, not a transportable decision. <!-- TRACE: B11; unsupported_claims clinical utility; W2 “does not add … ready for clinical use” -->

---

## 4. EPV, Firth, and the association/descriptive frame

Events per variable are far below conventional rules (**W4**): 92/81 ≈ **1.14** on the candidate list; 92/17 ≈ **5.4** on unidentified Table 4; 92/13 ≈ **7.1** on identified Table 4b — still below EPV ≥ 10. Outer-fold event counts are 18–19. Part 2 scores 18 events on the val slice. <!-- TRACE: W4; W3.3; B4 -->

Table 4b (unweighted 13-covariate Bernoulli logit, Wald 95% CI) is the association screen that may be quoted. Table 4 is unidentified and is not “the” multivariable model. Firth (Heinze/Kosmidis half-correction on the same 13 covariates) is a sparse-event **association sensitivity**. It is not a nested-CV arm and was never fused into GridSearch. Adjusted OR < 1 (post-dilation 0.152; Clopidogrel 0.480) is lower modelled odds of recorded VLST, not a treatment benefit; DAPT flags are follow-up persistence. <!-- TRACE: W3.3, W3.6; firth_logistic_regression.role; W5 -->

This study’s inferential half is therefore **association / descriptive** under an EPV constraint. Its predictive half is **nested-CV ranking** on the same derivation rows, not a clinically specified risk model.

---

## 5. Interpretability synthesis

Part 2 LOCO / coalition SHAP / FFS and Part 5 MI / stability SFS / PDP / held-out SHAP / *k*-SII are **attribution**, not a feature mask for nested CV (**W3.11**, **W5**). <!-- TRACE: W3.11; interpretability.caveats -->

Names that recur after anti-leakage (WBC not in the ML matrix):

- Part 3 FDR ∩ ML three-way (n = 5): `Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR` (Jaccard 5/25 = 0.20).
- Part 5 stability SFS **8/8**: `{CaI, LV, eGFR}`.
- Part 5 held-out mean |SHAP|: eGFR 1.2288, CaI 1.0867, Cre 0.8093, LV 0.4828.
- Part 5 consensus after merging SHAP: `{CaI, eGFR, LV}`.

`eGFR` and `LV` appear on both classic-selector and TabPFN sides. `CaI` is a TabPFN-train/SHAP name, not in the classic three-way union. `WBC` is FDR-only. `LV` and `CaI` remain unnamed in the CSV (**W3.10**). *k*-SII is one held-out VLST=1 row (cohort 5176), not a cohort interaction screen. PDP is a train empirical prior near prevalence, not nested-CV risk. <!-- TRACE: feature_extraction_models.results; kaggle_interpretability; W3.10–W3.11 -->

**SFS stability, stated honestly.** Live dump: `STABILITY_N_SEEDS=8`, all eight seeds completed, 8/8 `{CaI, LV, eGFR}`. A 10-seed plan was not finished (Kaggle session limit / 9-hour cap). A **3/10-seed provisional** table from an earlier incomplete run is **archived** and is not the live result. Eight seeds is a compute-constrained count, not a pre-specified larger reliability study. <!-- TRACE: kaggle_interpretability.sfs_stability; obsolete PROVISIONAL 3/10; outline 0.2 -->

---

## 6. Wang 2020 integer score and external validity

The frozen published integer points, scored on these 5,185 rows without re-fit, recover Wang’s derivation c-statistic on ROC-AUC (**0.8013** vs published **0.80** [0.75, 0.85]) with PR-AUC **0.1032**. Label mapping is material: SES points use `PES`; four post-dilation points use `No postdilation` = 1. The flipped encoding yields ROC-AUC **0.5084** and is rejected. <!-- TRACE: wang_2020; B10; wang_vlst_score.ipynb caveats -->

That recovery means the CSV can reproduce Wang’s **derivation-cohort ranking on a binary label**. It does **not** mean this pack’s nested-CV models inherit Wang’s Shantou test (n = 2,058, c = 0.82). Shantou is not in the repository (**B11**). Nested thinking v3.5 PR-AUC 0.9212 versus frozen-score PR-AUC 0.1032 is a same-file comparison of a foundation-model OOF against an integer Cox score — derivation-cohort nested CV, not external validation, and not a claim that TabPFN was “validated” because Wang’s score was. <!-- TRACE: W3.1–W3.2; unsupported_claims “TabPFN was validated”; W5 reserved word validated -->

Dangas c = 0.66 is Wang’s published comparison, not computed here. “No VLST score exists” is false.

---

## 7. Limitations (W3), stated without softening

1. **No external or temporal test of the ML models.** Nested CV on 5,185 derivation rows is not a substitute for Shantou or a later cohort (**W3.1**, **B11**). The centre is The First Hospital of Jilin University (single-centre structure implied by the file).
2. **Binary label versus Wang’s Cox time-to-event analysis** (**W3.2**). TSSI as a covariate leaks; dropping it is required for honest binary classification and is not a Cox re-fit.
3. **Rare-event statistics.** EPV ≈ 7.1 / 5.4 / 1.14; 18–19 events per outer fold (**W4**).
4. **Four TabPFN calibrations and two APIs** must not be collapsed (**W3.4–W3.5**).
5. **DAPT columns are post-baseline persistence** (**W3.6**).
6. **WBC** ranks in FDR and is dropped from ML views (**W3.7**).
7. **Unequal tuning** (**W3.8**): nested classics are library defaults, not inner-loop GridSearch. Train-fold-only stent encoding. Brand-frequency leak from a full-cohort encoder is not quantified on the 70/30 ON arm.
8. **Bootstrap does not re-fit models** (**W3.9**).
9. **`LV` / `CaI` unnamed** (**W3.10**).
10. **Part 5 ≠ Part 4 predictor**; SHAP dump records HTTP 429 then local finish (**W3.11**).
11. **Kaggle compute.** Interpretability was split because a full run exceeds the session limit (protocol: 9-hour cap). SFS used **8** seeds rather than the abandoned 10-seed plan. Client quota (20 thinking fits, 50M prediction cells/day) constrained which arms could be thinking-high.
12. **SMOTE** only in ALL LEAKS ON. Single nested CV, not repeated. Decision-curve / Cox LP / Shantou remain absent (**B11**). This analysis was not separately pre-registered.

---

These constraints do not erase the leakage contrast or the anti-leakage nested ranking. They bound what those numbers are: association under low EPV, attribution on a 70/30 split, and nested-CV discrimination on one derivation file.

<!--
DISCUSSION → EVIDENCE-MAP / FREEZE IDs
1 Leakage: W1, TODO-LEAK, B15 anti_leakage, leakage_contrast, kaggle_tssi_leakage, kaggle_without_tssi, W3.7–8
2 TabPFN arms: W3.4–5, W3.8–9, B3, nested_cv_v35_antileakage_on, W4 fold events
3 Calibration: W3.4, calibration_ece, Part 4 Fig 2, Table 2 vs Table 3; B11 DCA absent; W3 no clinical utility
4 EPV/Firth: W4, W3.3, B4, firth_logistic_regression.role, W3.6
5 Interpretability: W3.10–11, W5, feature_extraction_models, kaggle_interpretability (8/8 live; 3/10 archived)
6 Wang: wang_2020, B10, B11, W3.1–2, unsupported_claims contagion of “validated”
7 Limitations: W3 items 1–11; B11; quota comments; STABILITY_N_SEEDS=8 vs 10-seed plan
Photo photo_2026-08-30_21-48-00.jpg: not found on disk; themes taken from W3 + unsupported_claims.md
-->
