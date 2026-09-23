# Discussion

Numbers are only those frozen in `paper/frozen_results.yaml`. Association ≠ prediction ≠ attribution (**W5**). No causal, “risk factor,” or clinical-utility claims. Decision-curve analysis is absent (**B11**).

<!-- TRACE: W3–W5; B11; unsupported_claims; evidence_map W1/B15 -->

---

## 5.1 Principal findings — TabPFN under extreme imbalance without a per-dataset grid

The ranking result that survives anti-leakage handling is derivation-cohort nested CV, not a bedside rule. After the full OFF protocol, TabPFN thinking v3.5 has the highest nested PR-AUC among nine fitted arms (**0.9212** [0.8785, 0.9613]); local TabPFN v3.5 is second (**0.8957** [0.8446, 0.9426]). Both exceed **library-default** XGBoost **0.6322** and LightGBM **0.6271** (paired Δ thinking v3.5 − LightGBM **0.2941** (0.2071–0.3805), P(Δ ≤ 0) = 0/2000). That is a ranking of these objects, not “TabPFN versus tuned boosting.” Nested F1 at inner-fold thresholds: thinking v3.5 **0.8603** (sensitivity 0.8370); local v3.5 **0.8229**; LightGBM **0.5902**. <!-- TRACE: nested_cv_v35_antileakage_on; B3 -->

Why TabPFN can rank above this **untuned reference panel** on this file is a design fact, not a mechanistic proof: it is a tabular foundation model (in-context learning, native categoricals, no per-dataset grid) on a small-*n* mixed-type table (92 events). Classics in the nested zoo are library defaults plus class weighting; the inner loop tunes only the F1 threshold. Unmatched search effort is part of the **claim** (W3.8), not only a limitation. v3 and v3.5 are different checkpoints; thinking-high (`tabpfn-client`, effort high) and local `tabpfn` are different objects. Thinking outranks local within each family on pooled PR-AUC, **but not in every fold** (fold 1: local v3.5 0.9365 vs thinking 0.8802). With 18–19 events per outer fold, fold-to-fold movement is expected. Ranking uses PR-AUC because prevalence is **0.0177**; ROC-AUC is high for every arm (0.865–0.996) and is the less informative scale. <!-- TRACE: W2 why TabPFN; W3.4–5; W4 fold events -->

What does **not** survive is treating a leaks-on 70/30 scoreboard, or the excluded unlabeled nested dump (thinking **0.9771** / local **0.9635**), as the TabPFN result. **[RE-SOURCE]:** there is no matched 9-arm nested OFF dump, so a numeric “nested v3.5 ON minus nested v3.5 OFF” cannot be written.

---

## 5.2 Clinical context — stent characteristics and biomarkers (association / attribution)

Names that recur after anti-leakage, as **associations or attributions**, not treatment effects:

- Table 4b (identified logit): post-dilation and Clopidogrel have adjusted OR < 1; WBC, previous PCI, and `LV` have OR > 1. DAPT flags are follow-up persistence after the mandated year, not index-PCI prescriptions (W3.6).
- Part 5 stability SFS **8/8**: `{CaI, LV, eGFR}`. Held-out mean |SHAP|: eGFR 1.2288, CaI 1.0867, Cre 0.8093, LV 0.4828.
- Part 3 FDR ∩ ML three-way (n = 5): `Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR` (Jaccard 0.20).

`eGFR` and `LV` appear on both classic-selector and TabPFN sides. `CaI` is a TabPFN-train/SHAP name (means match Wang peak troponin I; still unnamed in the CSV). `WBC` is FDR-only because it was dropped from the ML matrix (Wang also excluded WBC from the Cox score). *k*-SII is one held-out VLST=1 row (cohort 5176). PDP is a train empirical prior near prevalence. None of this is a locked-in feature mask for nested CV, a causal stent-technique effect, or a monitoring rule. <!-- TRACE: W3.6, W3.10–11; W5; table4b -->

The frozen Wang integer score recovers derivation-cohort ROC-AUC **0.8013** (published c = 0.80) with PR-AUC **0.1032**. Nested thinking v3.5 PR-AUC 0.9212 versus that integer score is a same-file comparison of foundation-model OOF against an 8-variable Cox point scale — **not** external validation and not inheritance of Wang’s Shantou test (n = 2,058, c = 0.82, file absent). <!-- TRACE: wang_2020; B11; W3.1 -->

---

## 5.3 Methodological insight — subtle leakage in tabular cardiovascular ML

The first result that should change how this derivation file is modelled is not a TabPFN score. A published-style 70/30 GridSearch **overstates hold-out ranking** when follow-up time and related leak flags remain in the matrix (W1). ALL LEAKS ON versus OFF drops PR-AUC by **0.30–0.61** (LightGBM 0.9687 → 0.6675; logistic regression 0.9134 → 0.3431). Random forest F1 falls to **0**. Those arms are **not** “identical except TSSI”: five flags flip together, including SMOTE. Nested CV does not import those GridSearch winners and does not use SMOTE.

**Incentives, as implemented.** (1) TSSI as a covariate is binary-ified survival time (time-to-event vs completed follow-up). (2) WBC is a recording-precision / batch marker (and Wang excluded it from Cox). (3) Lab quantization is done **before** splitting so it is not a *y*-dependent encoder; a signature-only probe (243 indicators, AP 0.4270) shows how far decimal-grid membership alone can rank. Equalising leftover precision still drops TabPFN v3.5 nested AP by 0.0518. (4) Stent brands are encoded on the **train fold only** so rare strings cannot leak from the test distribution. (5) SMOTE is OFF except the leaks-on twin train set. Companion: identifiers dropped; scaler/OHE cloned inside splits; Part 2/5 catalogues do not mask nested CV. Follow-up drugs stay in with a post-baseline caveat. These are operational choices; they are not a claim that all possible leakage has been eliminated (brand-frequency leak on the ON arm is not separately quantified). <!-- TRACE: W1; §4.6; B15; W3.8; leakage_precision_probe -->

---

## 5.4 Limitations

1. **No external or temporal test of the ML models** (W3.1, B11). Nested CV on 5,185 derivation rows is not Shantou. Single-centre file (The First Hospital of Jilin University).
2. **Binary label versus Wang’s Cox time-to-event analysis** (W3.2). Dropping TSSI is required for honest binary classification and is not a Cox re-fit.
3. **Rare-event statistics / EPV** (W4): 92/81 ≈ **1.14**; Table 4 ≈ **5.4**; Table 4b ≈ **7.1** (still below EPV ≥ 10); 18–19 events per outer fold.
4. **Retrospective derivation-cohort design**; analysis not separately pre-registered.
5. Four TabPFN calibrations and two APIs must not be collapsed (W3.4–W3.5). Lowest nested ECE is local v3.5 **0.0003**; thinking v3.5 ECE **0.0028**. No decision-curve (B11) — nested PPV at an F1 cut is not clinical utility.
6. DAPT columns are post-baseline persistence (W3.6). WBC dual-label (W3.7). Unequal tuning (W3.8). Bootstrap does not re-fit (W3.9). `LV` / `CaI` unnamed (W3.10). Part 5 ≠ Part 4 predictor; SHAP dump records HTTP 429 then local finish (W3.11).
7. Interpretability was split because a full run exceeds the Kaggle session limit. SFS used **8** seeds, not the abandoned 10-seed plan. Client quota (20 thinking fits, 50M cells/day) constrained thinking-high design.
8. SMOTE only in ALL LEAKS ON. Single nested CV, not repeated. Cox LP / Shantou remain absent (B11).

These constraints bound the numbers: association under low EPV, attribution on a 70/30 split, and nested-CV discrimination on one derivation file.

---

## 5.5 Conclusion

On Wang 2020’s derivation cohort, after the full ALL LEAKS OFF protocol, TabPFN thinking v3.5 had the highest nested-CV PR-AUC among nine fitted arms (0.9212); local TabPFN v3.5 (0.8957) also exceeded library-default XGBoost (0.6322) and LightGBM (0.6271). That is not a hyperparameter-matched contest against tuned boosting. TSSI, WBC, raw lab decimals, a full-cohort stent codebook, and train SMOTE must not enter nested-CV binary classifiers as if they were baseline covariates. Recurring names in association and attribution (`eGFR`, `LV`, post-dilation flags, `CaI` on the TabPFN side) are not treatment effects. Nested-CV discrimination on this file is not external validation; the ML models were not externally or temporally tested.

<!--
DISCUSSION TRACE
5.1 nested_cv_v35_antileakage_on; W3.8 unmatched tuning
5.2 table4b; kaggle_interpretability; wang_2020; B11
5.3 W1; §4.6; B15
5.4 W3 items 1–11; W4 EPV; B11
5.5 W5; no “validated” / “clinically useful”
-->
