# Tables and figures

Markdown tables and figures only. Raw LaTeX is not used here (it does not render in the Markdown paper). Numbers from `paper/frozen_results.yaml`. Optimistic pooled F1 is **excluded**. Evidence-map §7.1 historical 0.8553 is **not** used.

Paths are relative to `manuscript/` so they display in `main_paper.md`.

<!-- TRACE: YAML study.*; nested_cv_v35_antileakage_on; kaggle_tssi_leakage; kaggle_interpretability -->

---

## Table 1. Cohort characteristics (association, not prediction)

Wang 2020 derivation file; n = 5,185 (no VLST 5,093; VLST 92). Continuous cells: mean (SD). Binary cells: n (%). Tests as in `eda.ipynb`. TSSI omitted (time-at-risk). DAPT is follow-up persistence. `LV` and `CaI` unnamed.

![Table 1. Cohort characteristics](../paper_results/01_eda/paper_figures/paper_table_c_cohort_characteristics.png)

**Table 1.** Restyle of `paper_table_c_cohort_characteristics.png`. Numeric copy:

| Characteristic | No VLST (n = 5,093) | VLST (n = 92) | Test | *p* |
| --- | --- | --- | --- | ---: |
| Age, years | 59.83 (9.93) | 60.71 (11.33) | Welch | 0.463 |
| Men | 3489 (68.51%) | 68 (73.91%) | χ² | 0.268 |
| Diabetes | 1293 (25.39%) | 36 (39.13%) | χ² | 0.003 |
| Hypertension | 2670 (52.42%) | 51 (55.43%) | χ² | 0.567 |
| Previous PCI | 94 (1.85%) | 10 (10.87%) | Fisher | 1.25e-05 |
| Previous MI | 347 (6.81%) | 10 (10.87%) | χ² | 0.128 |
| Admitting diagnosis AMI | 3095 (60.77%) | 65 (70.65%) | χ² | 0.054 |
| 3-vessel disease | 1422 (27.92%) | 42 (45.65%) | χ² | 0.000 |
| LVEF, % | 55.15 (4.52) | 54.55 (3.68) | MW | 0.033 |
| LV (unnamed) | 44.55 (4.04) | 49.11 (4.23) | Welch | 5.44e-17 |
| WBC, 10⁹/L | 8.75 (3.24) | 12.49 (3.92) | MW | 7.90e-21 |
| Creatinine | 72.53 (24.81) | 72.44 (19.05) | MW | 0.879 |
| eGFR | 120.03 (34.10) | 95.88 (19.63) | Welch | 4.64e-20 |
| CaI (unnamed) | 37.37 (61.64) | 40.55 (72.25) | MW | 0.051 |
| Fibrinogen (`Fiberinogen`) | 3.17 (0.88) | 3.37 (1.01) | MW | 0.012 |
| Stents per lesion | 1.21 (0.46) | 1.42 (0.65) | MW | 0.000 |
| Total stent length, mm | 31.70 (15.62) | 38.46 (20.71) | MW | 0.001 |
| SES (`PES` column) | 3502 (68.76%) | 76 (82.61%) | χ² | 0.004 |
| 1.1:1 post-dilation (as stored) | 2496 (49.01%) | 14 (15.22%) | χ² | 1.30e-10 |
| No postdilation (complement) | 2597 (50.99%) | 78 (84.78%) | χ² | 1.30e-10 |
| eGFR < 90 (`CKD90`) | 860 (16.89%) | 32 (34.78%) | χ² | 6.55e-06 |
| DAPT during follow-up | 2260 (44.37%) | 35 (38.04%) | χ² | 0.226 |

---

## Table 2. Leakage contrast (ALL LEAKS ON vs ALL LEAKS OFF)

Stratified 70/30 hold-out (1,556 rows / 28 events). Seven classics, GridSearchCV. **Not nested CV. Not TabPFN.** Δ = ON − OFF. Gaussian NB ROC dual-labelled (higher OFF). RF F1 OFF = 0.

![Table 2. Leakage metrics](../paper_results/04_tabpfn_rating/paper_figures/paper_table_s_tssi_leakage.png)

**Figure S-TSSI.** PR-AUC collapse on the 1,556-row hold-out. Dotted line = prevalence 0.0177.

![Figure S-TSSI. PR-AUC ON vs OFF](../paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_tssi_pr_auc.png)

**2a. PR-AUC**

| Model | ON | OFF | Δ |
| --- | ---: | ---: | ---: |
| Logistic regression | 0.9134 | 0.3431 | +0.5703 |
| Decision tree | 0.7524 | 0.1378 | +0.6146 |
| Random forest | 0.9400 | 0.4874 | +0.4526 |
| Gaussian NB | 0.2728 | 0.0564 | +0.2163 |
| CatBoost | 0.9599 | 0.4942 | +0.4656 |
| XGBoost | 0.9547 | 0.5685 | +0.3862 |
| LightGBM | 0.9687 | 0.6675 | +0.3011 |

**2b. F1 at the notebook cut**

| Model | F1 ON | F1 OFF |
| --- | ---: | ---: |
| Logistic regression | 0.6753 | 0.2093 |
| Decision tree | 0.7059 | 0.2791 |
| Random forest | 0.8333 | **0.0000** |
| Gaussian NB | 0.0437 | 0.0370 |
| CatBoost | 0.9231 | 0.4096 |
| XGBoost | 0.9231 | 0.5500 |
| LightGBM | 0.9231 | 0.4444 |

Dump ROC/PR curves:

![ALL LEAKS ON ROC/PR](../paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_leakage_roc_pr_on.png)

![ALL LEAKS OFF ROC/PR](../paper_results/04_tabpfn_rating/paper_figures/paper_fig_s_leakage_roc_pr_off.png)

---

## Table 3. Nested-CV benchmark (anti-leakage ON, 9 arms)

Pooled OOF ranking + honest nested operating point (inner-fold F1 thresholds). Anti-leakage ON = ALL LEAKS OFF. Five classic rows are **library defaults + class weights**, not nested GridSearch. Bootstrap 95% CIs on PR-AUC (`n_boot=2000`, seed 42). Pins `tabpfn==9.0.0` / `tabpfn-client==0.6.0`. Do **not** quote pooled F1 cuts instead of nested Table 2.

**Figure 1.** Nested-CV out-of-fold PR (left) and ROC (right).

![Figure 1. Nested OOF PR and ROC](../paper_results/04_tabpfn_rating/paper_figures/paper_fig1_pr_roc_curves.png)

**Figure 2.** Nested-CV calibration (quantile bins).

![Figure 2. Nested calibration](../paper_results/04_tabpfn_rating/paper_figures/paper_fig2_calibration_curves.png)

![Ranking metrics table](../paper_results/04_tabpfn_rating/paper_figures/paper_table1_ranking.png)

![Honest nested operating point](../paper_results/04_tabpfn_rating/paper_figures/paper_table2_nested_operating_point.png)

![Bootstrap 95% CIs](../paper_results/04_tabpfn_rating/paper_figures/paper_table_s_bootstrap_ci.png)

| Model | PR-AUC [95% CI] | ROC-AUC | Brier | Sensitivity | Precision | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TabPFN thinking v3.5 | **0.9212** [0.8785, 0.9613] | 0.9963 | 0.0047 | 0.8370 | 0.8851 | 0.8603 |
| TabPFN v3.5 | 0.8957 [0.8446, 0.9426] | 0.9916 | 0.0048 | 0.7826 | 0.8675 | 0.8229 |
| TabPFN thinking v3 | 0.8319 [0.7626, 0.8945] | 0.9834 | 0.0066 | 0.6957 | 0.8767 | 0.7758 |
| TabPFN v3 | 0.7150 [0.6260, 0.8074] | 0.9731 | 0.0099 | 0.6087 | 0.7089 | 0.6550 |
| XGBoost | 0.6322 [0.5331, 0.7247] | 0.9374 | 0.0100 | 0.5217 | 0.6575 | 0.5818 |
| LightGBM | 0.6271 [0.5313, 0.7202] | 0.9444 | 0.0106 | 0.5870 | 0.5934 | 0.5902 |
| CatBoost | 0.5707 [0.4750, 0.6729] | 0.9404 | 0.0108 | 0.5978 | 0.5093 | 0.5500 |
| Random forest | 0.3506 [0.2660, 0.4633] | 0.8883 | 0.0150 | 0.4565 | 0.4421 | 0.4492 |
| Logistic regression | 0.2596 [0.1834, 0.3588] | 0.8651 | 0.0788 | 0.3587 | 0.3028 | 0.3284 |

Thinking v3.5 nested 2×2: 5083/10/15/77. TabPFN v3.5: 5082/11/20/72. LightGBM: 5056/37/38/54. Frozen Wang integer score (not a nested arm): ROC-AUC 0.8013, PR-AUC 0.1032.

---

## Table 4. Eight-seed forward SFS rankings (train, local TabPFN v3.5)

Keep 10 of 80 columns; 5-fold CV; average precision; `STABILITY_N_SEEDS=8`. Train n = 3,629. `WBC` dropped. Not a nested-CV feature mask.

![Table 4. SFS stability](../paper_results/05_tabpfn_interpretability/paper_figures/paper_table2_stability.png)

| Rank | Feature | Selected | Frequency |
| ---: | --- | --- | ---: |
| 1 | CaI | 8/8 | 1.000 |
| 2 | LV | 8/8 | 1.000 |
| 3 | eGFR | 8/8 | 1.000 |
| 4 | Cre | 7/8 | 0.875 |
| 5 | HbA1c | 7/8 | 0.875 |
| 6 | Age | 6/8 | 0.750 |
| 7 | LVEF | 5/8 | 0.625 |
| 8 | 1.1:1Post dilation | 4/8 | 0.500 |
| 9 | LDL | 2/8 | 0.250 |
| 10 | No postdilation | 2/8 | 0.250 |
| 11 | TCL | 2/8 | 0.250 |
| 12 | Fiberinogen | 1/8 | 0.125 |
| 13 | Initial diagnosis-AMI | 1/8 | 0.125 |
| 14 | Previous PCI | 1/8 | 0.125 |
| 15 | Slow flow | 1/8 | 0.125 |
| 16 | Stent type-SES | 1/8 | 0.125 |

Held-out mean(|SHAP|) for the 8/8 names (1,556 rows): eGFR 1.2288, CaI 1.0867, LV 0.4828 (Cre 0.8093 is 7/8 SFS).

---

## Interpretability figures

**Mutual information (train).**

![Table. Top 15 mutual information](../paper_results/05_tabpfn_interpretability/paper_figures/paper_table1_mutual_info.png)

**Figure. Continuous PDP** (train; empirical prior; not nested-CV risk).

![Figure. Continuous PDP](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig1_pdp_continuous.png)

**Figure. Binary PDP.**

![Figure. Binary PDP](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig2_pdp_binary.png)

**Figure. SHAP summary** (1,556 held-out rows).

![Figure. SHAP summary](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig3_shap_summary.png)

**Figure. Mean |SHAP| bar.** Leading: eGFR 1.2288, CaI 1.0867, Cre 0.8093, LV 0.4828.

![Figure. SHAP bar](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig5_shap_bar.png)

**Figure. SHAP beeswarm.**

![Figure. SHAP beeswarm](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig6_shap_beeswarm.png)

**Figure. One-row waterfall** (cohort row 5176).

![Figure. SHAP waterfall](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig7_shap_waterfall.png)

**Figure. *k*-SII network** (same one VLST=1 row; not a cohort interaction screen).

![Figure. k-SII network](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig8_ksii_network.png)

**Figure. Consensus ranking** (train MI + train SFS + held-out SHAP).

![Figure. Consensus ranking](../paper_results/05_tabpfn_interpretability/paper_figures/paper_fig13_consensus_ranking.png)

![Table. Consensus report](../paper_results/05_tabpfn_interpretability/paper_figures/paper_table5_consensus.png)

Do not quote pooled confusion matrices (`paper_fig3_confusion_matrices.png`) as nested operating points.
