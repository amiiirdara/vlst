# Anti-leakage protocol (implemented) and why each control exists

This is the pack’s operational anti-leakage definition. It is **not** “drop TSSI and WBC and stop.” Nested CV, Part 2 selectors, and Part 5 attribution implement the **ALL LEAKS OFF** state. The 70/30 twins invert **all five flags together** (plus SMOTE on the ON twin only). Do not write “identical protocol except TSSI.”

Freeze string (`nested_cv_v35_antileakage_on.anti_leakage_flow`): drop `NO.` / `Name` / TSSI / WBC; quantize `Cre` / `CaI` / `Fiberinogen` / `Fast-Glu`; stent encoder **train-fold only**; **no SMOTE**.

Live code: `baseline_plus_tabpfn.ipynb` (nested); `baseline_without_tssi.ipynb` (OFF twin); `baseline_tssi_leakage.ipynb` (ON twin); `baseline_feature_selections.ipynb` (`split_manifest.json`); `tabpfn_interpretability_fs_pdp.ipynb` / `_shap.ipynb`.

---

## Why a protocol is needed (artefacts in `VLST.csv`)

The raw file is **sorted by outcome**: all 92 VLST cases are the last 92 rows. Those cases were transcribed under a **different numeric-precision convention** from the controls. Recording precision therefore partly identifies the label (a case/control **batch marker**), independent of clinical magnitude. The nested notebook’s precision-signature probe keeps only “which decimal grid does this float lie on?” and discards magnitudes: **243** indicators, AP **0.4270**, ROC-AUC **0.9522** at prevalence 0.0177 (`leakage_precision_probe.csv`). That is an artefact floor, not a clinical result, and **not** a reason to restore TSSI.

A second, structural leak is **time**: Wang analysed this cohort with Cox regression, in which follow-up duration is the *time axis*. Recoded as a binary covariate, `Time since stent implantation` mixes time-to-event (VLST = 1) with completed event-free follow-up (VLST = 0).

---

## Five flags (twins) plus companion controls

| Control | OFF = prediction / attribution / selectors | ON = leakage demonstration only | Incentive (why) |
| --- | --- | --- | --- |
| `KEEP_TSSI` | `False` — drop `Time since stent implantation` | `True` | **Temporal / definition leak.** VLST=1: time from PCI to thrombosis (min 380 d). VLST=0: completed follow-up (min 1,241, max 1,605 d; cohort median 1,502). A rule “time < 1,241 → event” has **zero** false positives among controls. Not a baseline covariate. |
| `DROP_WBC` | `True` — drop `WBC` | `False` | **Recording-precision / batch marker**, and Wang excluded WBC from the Cox score because infection could not be ruled out. No VLST case has a whole-number WBC; many controls do. The FDR catalogue still ranks WBC (dual-label). |
| `QUANTIZE_CLINICAL` | `True`, **file-level before any split** | `False` | **Spurious decimal precision** fingerprints source/batch. Rounding is a pre-specified clinical grid, **not** a parameter fit on *y*. Grid: `Cre` 0 dp (µmol/L integer); `CaI` 2 dp (conventional TnI; do not round to 0); `Fiberinogen` / `Fast-Glu` 1 dp (CSV spelling `Fiberinogen`). After anti-leakage, equalising remaining precision still drops TabPFN v3.5 nested AP by **0.0518** (0.8957 → 0.8439). |
| `STENT_ENCODER_TRAIN_ONLY` | `True` — `encode_stent_on_fold`, `min_count=30` | `False` (codebook on the **full** cohort before split) | **Test-brand leak.** Rare `Stent type-SES` strings must not define levels using held-out rows. Unseen strings map to `Other`. Nested CV fits the codebook on **each outer-training fold**. |
| `USE_SMOTE` | `False` | `True` on the ON-twin **train** set only | **Unmatched inflation.** SMOTE synthesises minorities on the leaks-on train split; nested CV / Part 2 / Part 5 never use it. Quote twins as a leakage demonstration, not a SMOTE-matched experiment. |

**Identifiers.** `NO.` and `Name` are always dropped (not predictors).

**Preprocessors inside splits.** Classic pipelines clone imputer / `StandardScaler` / `OneHotEncoder` on **train only** (Part 2 `scaler_ohe_train_only`; nested `ColumnTransformer` inside every CV split). EDA printed no missing values, so imputers are inert.

**Follow-up drugs stay in.** `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT` are **not** leak-flag drops. They are post-baseline persistence after the mandated DAPT year (limitation W3.6), not index-PCI prescriptions.

**Selectors do not feed prediction.** Part 2 uses an inner 80/20 (`INNER_VAL_SIZE=0.2`) on all 5,185 rows — **no parked 70/30 test**. `USE_CACHE=False`. Part 2 / Part 5 catalogues are **not** a feature mask for nested CV.

**GridSearch winners are not imported.** Twin `best_params_` stay on the leakage axis. Nested classics = library defaults + class weighting; inner loop = F1 **threshold** only.

---

## Where each state is used

| Analysis | Leak state | Split |
| --- | --- | --- |
| Nested 5×4 CV (`baseline_plus_tabpfn.ipynb`) | OFF | All 5,185 rows as outer OOF — **only prediction evaluation** |
| Part 2 selectors | OFF | Fit 4,148 / 74 events; val 1,037 / 18 events |
| Part 5 fs_pdp + SHAP | OFF | Stratified 70/30: 3,629 / 64 vs 1,556 / 28 |
| ALL LEAKS OFF twin | OFF | Same 70/30; GridSearch; **no SMOTE** |
| ALL LEAKS ON twin | ON (all five flags + SMOTE) | Same 70/30; **not** nested ranking; **not** TabPFN |

Nested TabPFN ON vs OFF is **[RE-SOURCE]** (no matched 9-arm nested OFF dump). Classic ON vs OFF is Part 4b. Unlabeled nested PR-AUC 0.9771 / 0.9635 is a **pre-anti-leakage** scoreboard and is **excluded**.
