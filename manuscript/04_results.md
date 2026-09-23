# Results

All prediction numbers below carry checkpoint / thinking / leak-status tags. Nested ranking is anti-leakage **ON** (ALL LEAKS OFF: IDs+TSSI+WBC dropped; labs quantized pre-split; stent codebook train-fold only; no SMOTE). Nested with vs without anti-leakage for TabPFN is **[RE-SOURCE]** — no matched 9-arm nested OFF dump. Do not quote excluded unlabeled nested PR-AUC 0.9771 / 0.9635. Language: association ≠ prediction ≠ attribution (**W5**).

<!-- TRACE: nested_cv_v35_antileakage_on; W5; evidence_map §7.1 historical 0.8553 NOT used -->

---

The analysed file has **5,185** patients, **92** VLST events, **5,093** non-events, prevalence **0.0177**. Wang’s flow (cited): 6,038 eligible → 5,185 analysed. No missing values. Cohort contrasts (association, not prediction) are **Table 1**. Example: previous PCI 1.85% vs 10.87% (Fisher p = 1.25e-05); eGFR 120.03 (34.10) vs 95.88 (19.63) (Welch p = 4.64e-20); `1.1:1Post dilation` 49.01% vs 15.22% (χ² p = 1.30e-10). DAPT / Clopidogrel columns are follow-up persistence, not index-PCI prescriptions. `LV` and `CaI` remain unnamed. Identified multivariable screen (Table 4b, 13 covariates, EPV ≈ 7.1): `1.1:1Post dilation` adj. OR **0.152** [0.081, 0.286]; Clopidogrel **0.480** [0.293, 0.787]; WBC **1.972** [1.667, 2.331]; Previous PCI **6.710** [2.884, 15.610]; eGFR **0.568** [0.449, 0.717]; LV **1.832** [1.539, 2.181]. OR < 1 is lower modelled odds of recorded VLST, not a treatment benefit. Quote Table 4b, not unidentified Table 4 (EPV ≈ 5.4). <!-- TRACE: YAML study.*; table4b_adjusted_or; Part 1 Table C -->

![Table 1. Cohort characteristics](../paper_results/01_eda/paper_figures/paper_table_c_cohort_characteristics.png)

---

## 4.1 Baseline and anti-leakage contrast

Same seven classic models, same stratified 70/30 (train 3,629 / test 1,556, seed 42), `GridSearchCV` (`GRID_SCORING=average_precision`). **Not nested CV. Not TabPFN.** Five flags flip together; SMOTE is ON only in the leaks-on arm. **Incentives:** TSSI = mixed time-to-event vs completed follow-up; WBC = recording-precision / batch marker; unquantized labs = decimal-grid fingerprint (signature probe AP 0.4270); full-cohort stent codebook = test-brand leak; train SMOTE = unmatched inflation. Full protocol: `paper_results/anti_leakage_protocol.md`. Full tables: **Table 2** and `leakage_contrast_paper_figures_and_tables.md`. <!-- TRACE: W1; §4.6; YAML kaggle_tssi_leakage / kaggle_without_tssi -->

| Tag | Notebook | Flags |
| --- | --- | --- |
| **ALL LEAKS ON** | `baseline_tssi_leakage.ipynb` | `KEEP_TSSI=True`, `DROP_WBC=False`, `QUANTIZE_CLINICAL=False`, `STENT_ENCODER_TRAIN_ONLY=False`, `USE_SMOTE=True` |
| **ALL LEAKS OFF** | `baseline_without_tssi.ipynb` | inverse flags, `USE_SMOTE=False` |

Hold-out PR-AUC (primary ranking metric at prevalence 0.0177); Δ = ON − OFF:

| Model | ON | OFF | Δ PR-AUC |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.9134 | 0.3431 | **+0.5703** |
| Decision tree | 0.7524 | 0.1378 | **+0.6146** |
| Random forest | 0.9400 | 0.4874 | **+0.4526** |
| Gaussian NB | 0.2728 | 0.0564 | **+0.2163** |
| CatBoost | 0.9599 | 0.4942 | **+0.4656** |
| XGBoost | 0.9547 | 0.5685 | **+0.3862** |
| LightGBM | 0.9687 | 0.6675 | **+0.3011** |

Every model’s hold-out PR-AUC is higher on ALL LEAKS ON. Inflation ranges **+0.30 to +0.61**. LightGBM is the least inflated booster and still drops by 0.30.

![Figure S-TSSI. PR-AUC ALL LEAKS ON vs OFF](../paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_tssi_pr_auc.png)

ROC-AUC compression is smaller. **Dual-label:** Gaussian NB ROC-AUC is slightly *higher* OFF (0.7493 vs 0.7425) while PR-AUC still falls (0.2728 → 0.0564). Do not write “every metric falls.”

At the notebook operating point, random forest F1 on ALL LEAKS OFF is **0** (no predicted events) while ROC-AUC remains 0.9287. Boosters that look near-perfect on ON (F1 0.9231, precision 1.0) drop to F1 0.41–0.55 once leak flags are off. Accuracy stays high except GNB because 1,528/1,556 hold-out rows are non-events; accuracy is not the ranking metric.

**Reading.** A 70/30 GridSearch pipeline that retains TSSI (and WBC, unquantized labs, a full-cohort stent encoder, and train SMOTE) **overstates hold-out ranking on this derivation file**. Nested CV, Part 2, and Part 5 therefore implement the OFF flags. This subsection does not transfer to TabPFN. **ARCHIVED:** LR PR-AUC 0.9575 → 0.5077.

---

## 4.2 Nested ranking: TabPFN checkpoints vs library-default reference panel

Dump: `Kaggle_baseline_plus_tabpfn_results/baseline_plus_tabpfn_results/` (`tabpfn==9.0.0`, `tabpfn-client==0.6.0`, Tesla T4, 9 arms). Nested 5×4, seed 42, inner loop = F1 threshold only. Five classic arms = **untuned reference panel** (defaults + class weighting; GridSearch winners **not** imported). No SMOTE. Leak status for the entire table: anti-leakage **ON** (five-flag OFF: IDs+TSSI+WBC dropped; labs quantized pre-split; `encode_stent_on_fold`; no SMOTE). Full numbers: **Table 3**. Bootstrap 95% CIs: stratified resample of stored OOF, `n_boot=2000`, seed 42; models not re-fit. <!-- TRACE: nested_cv_v35_antileakage_on; B3; §5.8 live YAML not map §7.1 -->

**Figure 1.** Nested-CV out-of-fold PR and ROC curves.

![Figure 1. Nested OOF PR and ROC](../paper_results/04_tabpfn_rating/paper_figures/paper_fig1_pr_roc_curves.png)

Pooled nested OOF ranking (threshold-independent):

| Rank | Model | Checkpoint | Thinking | PR-AUC (95% CI) | ROC-AUC | Brier |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 1 | TabPFN thinking v3.5 | hosted `v3.5_default` | thinking-high | **0.9212** [0.8785, 0.9613] | 0.9963 | 0.0047 |
| 2 | TabPFN v3.5 | local `tabpfn-v3.5-20260909.safetensors` | local | **0.8957** [0.8446, 0.9426] | 0.9916 | 0.0048 |
| 3 | TabPFN thinking v3 | hosted `v3_default` | thinking-high | 0.8319 [0.7626, 0.8945] | 0.9834 | 0.0066 |
| 4 | TabPFN v3 | local v3 ckpt | local | 0.7150 [0.6260, 0.8074] | 0.9731 | 0.0099 |
| 5 | XGBoost | n/a | n/a | 0.6322 [0.5331, 0.7247] | 0.9374 | 0.0100 |
| 6 | LightGBM | n/a | n/a | 0.6271 [0.5313, 0.7202] | 0.9444 | 0.0106 |
| 7 | CatBoost | n/a | n/a | 0.5707 [0.4750, 0.6729] | 0.9404 | 0.0108 |
| 8 | Random forest | n/a | n/a | 0.3506 [0.2660, 0.4633] | 0.8883 | 0.0150 |
| 9 | Logistic regression | n/a | n/a | 0.2596 [0.1834, 0.3588] | 0.8651 | 0.0788 |

**TabPFN vs library-default reference panel.** Highest default-classic PR-AUC is XGBoost 0.6322, then LightGBM 0.6271. These are not GridSearch winners. Paired bootstrap Δ PR-AUC: thinking v3.5 − LightGBM **0.2941** (0.2071–0.3805), P(Δ ≤ 0) = 0/2000; TabPFN v3.5 − LightGBM **0.2686** (0.1822–0.3559), P = 0/2000; thinking v3.5 − XGBoost **0.2891** (0.2038–0.3791), P = 0/2000. The Δ is a contrast of stored OOF vectors, not evidence that tuned boosting would lose by the same margin.

**v3 vs v3.5 (same thinking status).** Local: 0.8957 vs 0.7150. Thinking: 0.9212 vs 0.8319.

**Thinking vs local (same checkpoint family).** v3.5: 0.9212 vs 0.8957. v3: 0.8319 vs 0.7150. Thinking v3.5 is **not** higher than local v3.5 in every fold (fold 1: local 0.9365 vs thinking 0.8802). Fold PR-AUC thinking v3.5: 0.8802, 0.9078, 0.9434, 0.9858, 0.8961.

**Honest nested operating points (inner-fold F1 thresholds — quote these, not pooled cuts).** Thinking v3.5: *t* 0.318 ± 0.059; 5083/10/15/77; sensitivity **0.8370**; precision **0.8851**; F1 **0.8603**. TabPFN v3.5: 5082/11/20/72; sensitivity **0.7826**; precision **0.8675**; F1 **0.8229**. XGBoost: sensitivity 0.5217; F1 0.5818. LightGBM: sensitivity **0.5870**; F1 **0.5902**. ROC-AUC CIs (thinking v3.5 **0.9963** [0.9932, 0.9986]; TabPFN v3.5 **0.9916** [0.9828, 0.9978]). Nested-CV discrimination on the derivation file is not external validation.

![Figure 2. Nested calibration](../paper_results/04_tabpfn_rating/paper_figures/paper_fig2_calibration_curves.png)

**Wang integer score (frozen comparator, not a nested arm).** Full-cohort ROC-AUC **0.8013**, PR-AUC **0.1032**. SES points use `PES`; four post-dilation points use `No postdilation` = 1. Flipped encoding ROC-AUC **0.5084** is rejected. <!-- TRACE: wang_2020; B10 -->

---

## 4.3 Feature-selection stability across 8 seeds

Forward sequential feature selection (keep 10 of 80, 5-fold CV, average precision) repeated over **8/8 shuffled seeds** on the **train** split (n = 3,629), local TabPFN v3.5, `FS_THINKING_MODE=False`, ALL LEAKS OFF (same quantization + train-only stent codebook as nested CV). **Table 4.** Selected in **8/8**: `CaI`, `LV`, `eGFR`. Selected in 7/8: `Cre`, `HbA1c`. Selected in 6/8: `Age`. Selected in 5/8: `LVEF`. Selected in 4/8 (0.5 cutoff): `1.1:1Post dilation`. `WBC` is not a column. A 10-seed plan was not finished; **3/10 provisional** tables are archived and are not this result. <!-- TRACE: YAML kaggle_interpretability.sfs_stability; §5.9 live 8/8 not 10/10 WBC -->

![Table 4. SFS stability (8/8 seeds)](../paper_results/05_tabpfn_interpretability/paper_figures/paper_table2_stability.png)

Train mutual information (top 5 of 80): CaI 0.020536, LV 0.012818, eGFR 0.009424, LDL 0.009367, HbA1c 0.007328. `Cre` train MI is 0.000338 (rank 50 of 80). These catalogues are **attribution**, not a nested-CV feature mask.

Classic-selector overlap (Part 2/3, anti-leak 19-Sep dump, ML consensus n = 10 vs FDR-20): intersection `{Clopidogrel, HbA1c, LV, No postdilation, eGFR}`; Jaccard **5/25 = 0.20**. Dual-label: `WBC` is FDR-only. <!-- TRACE: feature_extraction_models.results -->

---

## 4.4 Feature attribution and interactions (SHAP, *k*-SII, PDP)

Fit on train; SHAP explains **all 1,556 held-out rows** (client thinking-high intended constructor). Mean(|SHAP|) top 4 of 80: **eGFR 1.2288**, **CaI 1.0867**, **Cre 0.8093**, **LV 0.4828**. `WBC` is absent. Do not call this 15+15 or global SHAP on 5,185. <!-- TRACE: YAML interpretability.shap.part5 -->

![Figure. SHAP summary (held-out)](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig3_shap_summary.png)

![Figure. Mean |SHAP| bar](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig5_shap_bar.png)

Borda consensus after merging held-out SHAP into the fs dump: **CaI, eGFR, LV** are 3/3 (also HbA1c and `1.1:1Post dilation` at the 0.5 SFS cutoff).

**PDP** (local TabPFN, empirical prior, **train** n = 3,629). Y-axis near prevalence (~0.018). **Not** nested-CV risk. Largest binary |Δ|: `Previous PCI` 0.017711 → 0.019006 (Δ **+0.001294**). `1.1:1Post dilation` Δ **−0.000094**. A negative Δ is a lower modelled probability of recorded VLST, not a treatment benefit.

![Figure. Continuous PDP (train, empirical prior)](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig1_pdp_continuous.png)

![Figure. Binary PDP](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig2_pdp_binary.png)

***k*-SII / waterfall** use one held-out VLST=1 row (cohort **5176**, budget 256). They illustrate how TabPFN combines features **for that patient**; they are not a cohort interaction screen.

![Figure. k-SII network (row 5176)](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig8_ksii_network.png)

![Figure. Consensus ranking](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig13_consensus_ranking.png)

<!--
TRACE: 4.1 kaggle_tssi_leakage / kaggle_without_tssi; W1; §4.6
4.2 nested_cv_v35_antileakage_on; nested_table2; B3; wang_2020
4.3 STABILITY_N_SEEDS=8; feature_extraction_models.results
4.4 interpretability.shap; shapiq k_sii_row 5176; PDP empirical_prior
-->
