# Nested-CV baselines plus TabPFN — paper figures and tables

This document gathers publication-oriented figures and tables from the nested cross-validation comparison in [`baseline_plus_tabpfn.ipynb`](baseline_plus_tabpfn.ipynb).

**Cohort / protocol.** Full VLST cohort, n = 5,185 (92 events; prevalence = 0.0177). Target = `Stent thrombosis`. Identifiers (`NO.`, `Name`) and `Time since stent implantation` are dropped; the latter is treated as a time-at-risk / follow-up column, not a baseline covariate. **No Part 2 / Part 5 feature mask is applied.** Evaluation is nested stratified CV: **5 outer folds / 4 inner folds** (outer `random_state=42`). Ranking metrics (PR-AUC, ROC-AUC, Brier) use pooled outer out-of-fold probabilities and are threshold-independent. For precision / recall / F1 / F2, **quote the nested inner-fold thresholds** (Table 2): each outer fold’s cut is chosen on inner OOF scores and applied once to that fold’s unseen cases. Figure 3 / Table 3 additionally show a single pooled F1 cut; that cut is **optimistically biased** (methods note below). These nested-CV metrics are this pack’s only **prediction** results.

**This run (D4).** Kaggle nested CV, Tesla T4. `RUN_MODELS["TabPFN"]=False` (client thinking-high constructor is still in the notebook, unused). The TabPFN arm is **`TabPFN (local)`** (`from tabpfn import TabPFNClassifier`, `n_estimators="auto"`, `balance_probabilities=True`, no thinking). Shared **9-level** stent encoder (106 raw strings → 9 levels, min_count=30) is applied before the split. Classics then scale + one-hot that 9-level column inside each CV split (~89 columns, not the old ~186). TabPFN (local) sees the same 9-level frame natively.

**Methods note — feature views.** Classics sit in an sklearn `Pipeline` with a `ColumnTransformer` **cloned and fitted inside every CV split**: numeric columns get `SimpleImputer(median)` + `StandardScaler`; the encoded `Stent type-SES` gets most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")`. EDA found **no missing values**, so both imputers are inert. TabPFN (local) is **not** in that pipeline.

**Methods note — GridSearch is a different notebook.** `baseline_without_tssi.ipynb` / `baseline_tssi_leakage.ipynb` tune hyperparameters on a single 70/30 split. Those `best_params_` are **not** imported here. Classics use library defaults plus class weighting. The inner loop tunes only the F1 **threshold**.

**Methods note — why the follow-up-time column is dropped.** Wang 2020 analysed this cohort with Cox regression, in which follow-up duration is the *time axis*, not a covariate. Recoded as a binary classifier, the same column (`Time since stent implantation`) mixes two definitions: time-to-event for the 92 VLST cases (min 380 days) and event-free follow-up length for the 5,093 non-events (min 1,241 days). A rule “time < 1,241 → event” has zero false positives among controls. `baseline_tssi_leakage.ipynb` (same 70/30 split, GridSearchCV) shows the resulting inflation; `baseline_without_tssi.ipynb` is the identical protocol with the column removed. Nested-CV results in this document use the without-TSSI feature view. See Supplementary Table S-TSSI.

**Models.** Logistic regression, random forest, XGBoost, LightGBM, CatBoost, and TabPFN (local). Average precision (PR-AUC) is the common ranking metric. On this run **LightGBM is first** (PR-AUC **0.6937**). TabPFN (local) is third on PR-AUC (**0.6754**) and first on ROC-AUC (**0.9845**). Quote PR-AUC at 1.77% prevalence. The prior thinking-high client numbers (PR-AUC 0.8534 / Brier 0.0060) are **not** this notebook.

**Methods note — published clinical baseline.** Wang 2020’s 8-variable integer score is scored as a **frozen** comparator in [`wang_vlst_score.ipynb`](wang_vlst_score.ipynb) (published Table 2 points; weights not re-fit). It is not a seventh nested-CV arm. See Supplementary Table S-Wang.

**Methods note — two F1 operating points.** Ranking metrics do not use a threshold. Precision, recall, F1, and F2 do. The executed notebook prints both. **Honest nested** (Table 2): inner-CV OOF F1 threshold applied once to the unseen outer fold. **Optimistic pooled** (Figure 3, Table 3): one F1-maximising cut on the concatenated OOF labels that are then scored. Reusing the evaluation labels to pick the cut **optimistically biases** precision, recall, F1, and F2. Quote Table 2. LightGBM nested recall **0.6630** vs pooled **0.6739**. TabPFN (local) nested recall **0.6848** vs pooled **0.8261**.

**Methods note — imbalance, SMOTE, and tuning.** Prevalence is 1.77%. Class weighting (`class_weight="balanced"`, `scale_pos_weight`, `auto_class_weights="Balanced"`) is used for *prediction* so the 92 events are not ignored. SMOTE is **not** used. The five classic models use library defaults plus class weighting. TabPFN (local) is not thinking-high. Inner nested CV selects only the F1 **threshold**, not hyperparameters.

**Asset root:** [paper_figures](paper_figures/) (also at `data/result/modeling_results/paper_figures/`)

> Figures 1–3 and the ranking / operating-point tables are exported from this executed notebook (Kaggle). Quote the notebook print if a PNG title ever disagrees.

---

## Contents

1. [Models (Table 0)](#1-models)
2. [Ranking curves (Figure 1, Table 1)](#2-ranking-curves)
3. [Calibration (Figure 2)](#3-calibration)
4. [F1 operating point (Table 2 nested; Figure 3 / Table 3 pooled)](#4-f1-operating-point)
5. [Supplementary: follow-up-time leakage](#5-supplementary-follow-up-time-leakage)
6. [Supplementary: Wang 2020 integer score](#6-supplementary-wang-2020-integer-score)
7. [File index](#7-file-index)

---

## 1. Models

### Table 0. Nested-CV models

![Table 0](paper_figures/paper_table0_models.png)

**Table 0.** Six classifiers compared under the same nested-CV *split and threshold* protocol. Classics get scaled one-hot input after the 9-level stent encoder; TabPFN (local) gets that frame natively. Tree boosters use average-precision / PR-AUC as their internal metric. Classics are not grid-searched. The unused client thinking-high constructor remains in the notebook (`RUN_MODELS["TabPFN"]=False`).

| Model | Family | GPU | Specification (notebook) |
| --- | --- | --- | --- |
| Logistic Regression | Linear | No | L2, class_weight=balanced, max_iter=1000 |
| Random Forest | Bagged trees | No | class_weight=balanced, random_state=42 |
| XGBoost | Boosting | Yes | eval_metric=aucpr; scale_pos_weight from train fold |
| LightGBM | Boosting | Yes | metric=average_precision; class_weight=balanced |
| CatBoost | Boosting | Yes | auto_class_weights=Balanced; eval_metric=PRAUC |
| TabPFN (local) | Foundation (tabular) | Kaggle T4 | n_estimators=auto; balance_probabilities=True; no thinking |

**Source files:** [paper_figures/paper_table0_models.png](paper_figures/paper_table0_models.png), [paper_figures/paper_table0_models.csv](paper_figures/paper_table0_models.csv)

---

## 2. Ranking curves

### Figure 1. Nested-CV out-of-fold PR and ROC curves

![Figure 1](paper_figures/paper_fig1_pr_roc_curves.png)

**Figure 1.** Precision–recall (left) and ROC (right) from pooled nested-CV OOF probabilities. The dotted line on the PR panel is prevalence (0.0177). **LightGBM ranks first on PR-AUC (0.6937)**; XGBoost 0.6815; TabPFN (local) 0.6754. TabPFN (local) ranks first on ROC-AUC (0.9845). On a 1.8% prevalence outcome, PR-AUC is the informative ranking metric. CatBoost is fourth on PR-AUC (0.6172) in this 9-level-encoder run (it was second under the old 106-string one-hot).

**Source file:** [paper_figures/paper_fig1_pr_roc_curves.png](paper_figures/paper_fig1_pr_roc_curves.png)

### Table 1. Pooled OOF ranking metrics

![Table 1](paper_figures/paper_table1_ranking.png)

**Table 1.** Threshold-independent metrics from the executed notebook (D4). Fold mean ± SD uses `ddof=1` across the five outer folds.

| Rank | Model | PR-AUC | PR fold mean ± SD | ROC-AUC | ROC fold mean ± SD | Brier |
| ---: | --- | ---: | --- | ---: | --- | ---: |
| 1 | LightGBM | **0.6937** | 0.6941 ± 0.0917 | 0.9681 | 0.9695 ± 0.0164 | 0.0093 |
| 2 | XGBoost | 0.6815 | 0.6928 ± 0.1288 | 0.9439 | 0.9431 ± 0.0418 | **0.0088** |
| 3 | TabPFN (local) | 0.6754 | 0.6739 ± 0.0812 | **0.9845** | 0.9846 ± 0.0030 | 0.0673 |
| 4 | CatBoost | 0.6172 | 0.6353 ± 0.0540 | 0.9594 | 0.9612 ± 0.0137 | 0.0101 |
| 5 | Random Forest | 0.4865 | 0.5034 ± 0.0793 | 0.9209 | 0.9206 ± 0.0423 | 0.0143 |
| 6 | Logistic Regression | 0.3326 | 0.3451 ± 0.1213 | 0.9224 | 0.9235 ± 0.0251 | 0.0563 |

**Source files:** [paper_figures/paper_table1_ranking.png](paper_figures/paper_table1_ranking.png), [paper_figures/paper_table1_ranking.csv](paper_figures/paper_table1_ranking.csv)

TabPFN (local) PR-AUC by outer fold: 0.6384, 0.6353, 0.5829, 0.7274, 0.7855. LightGBM: 0.7527, 0.7142, 0.5399, 0.7718, 0.6916. LightGBM is higher in **3 of 5** folds; TabPFN (local) in 2 of 5 (folds 3 and 5).

---

## 3. Calibration

### Figure 2. Reliability curves (quantile bins)

![Figure 2](paper_figures/paper_fig2_calibration_curves.png)

**Figure 2.** Calibration plots from nested-CV OOF probabilities (quantile bins). Dashed diagonal = perfect calibration. Brier scores match Table 1: XGBoost **0.0088** (best), LightGBM 0.0093, CatBoost 0.0101, RF 0.0143, LR 0.0563, TabPFN (local) **0.0673** (worst of the six). Local TabPFN is **not** well calibrated on this run.

**Source file:** [paper_figures/paper_fig2_calibration_curves.png](paper_figures/paper_fig2_calibration_curves.png)

---

## 4. F1 operating point

Two cuts exist in the executed notebook. **Quote Table 2 (honest nested).** Figure 3 and Table 3 are the pooled F1 cut: the same concatenated OOF labels are used to *pick* and *score* the threshold, so precision, recall, F1, and F2 are **optimistically biased**. Counts sum to n = 5,185 with 92 events.

### Table 2. Honest nested-CV operating point (quote this)

![Table 2](paper_figures/paper_table2_nested_operating_point.png)

**Table 2.** Per-fold inner-CV F1 thresholds applied once to the unseen outer fold. LightGBM: mean threshold 0.121 ± 0.085, precision 0.6703, recall **0.6630**, F1 0.6667, TN/FP/FN/TP = 5063/30/31/61. TabPFN (local): 0.915 ± 0.012, precision 0.5478, recall **0.6848**, F1 0.6087, 5041/52/29/63 — more events caught, more false positives.

| Model | Threshold (mean ± SD) | Accuracy | Precision | Recall | Specificity | F1 | F2 | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.121 ± 0.085 | 0.9882 | 0.6703 | 0.6630 | 0.9941 | 0.6667 | 0.6645 | 5063 | 30 | 31 | 61 |
| XGBoost | 0.225 ± 0.060 | 0.9875 | 0.6452 | 0.6522 | 0.9935 | 0.6486 | 0.6508 | 5060 | 33 | 32 | 60 |
| TabPFN (local) | 0.915 ± 0.012 | 0.9844 | 0.5478 | 0.6848 | 0.9898 | 0.6087 | 0.6522 | 5041 | 52 | 29 | 63 |
| CatBoost | 0.167 ± 0.040 | 0.9815 | 0.4836 | 0.6413 | 0.9876 | 0.5514 | 0.6020 | 5030 | 63 | 33 | 59 |
| Random Forest | 0.118 ± 0.013 | 0.9840 | 0.5517 | 0.5217 | 0.9923 | 0.5363 | 0.5275 | 5054 | 39 | 44 | 48 |
| Logistic Regression | 0.947 ± 0.035 | 0.9769 | 0.3654 | 0.4130 | 0.9870 | 0.3878 | 0.4025 | 5027 | 66 | 54 | 38 |

**Source files:** [paper_figures/paper_table2_nested_operating_point.png](paper_figures/paper_table2_nested_operating_point.png), [paper_figures/paper_table2_nested_operating_point.csv](paper_figures/paper_table2_nested_operating_point.csv). Notebook print `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt`.

### Figure 3. Confusion matrices at the pooled F1 threshold (optimistic)

![Figure 3](paper_figures/paper_fig3_confusion_matrices.png)

**Figure 3.** 2×2 counts at the F1-maximising **pooled** OOF threshold (`t_F1` in each panel title). This is **not** Table 2. TabPFN (local) pooled recall **0.8261** (TP = 76, FN = 16, t = 0.886) vs nested **0.6848** (TP = 63, FN = 29). Do not quote the pooled TabPFN recall as the nested result. Accuracy is uniformly high because negatives dominate.

**Source file:** [paper_figures/paper_fig3_confusion_matrices.png](paper_figures/paper_fig3_confusion_matrices.png)

### Table 3. Optimistic pooled F1 metrics (do not quote instead of Table 2)

![Table 3](paper_figures/paper_table3_pooled_f1.png)

**Table 3.** Same pooled F1 cut as Figure 3. Precision / recall / F1 / F2 here are **optimistically biased** versus Table 2.

| Model | t_F1 | Accuracy | Precision | Recall | Specificity | F1 | F2 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.064 | 0.9871 | 0.6263 | 0.6739 | 0.9927 | 0.6492 | 0.6638 | 5056 | 37 | 30 | 62 |
| XGBoost | 0.203 | 0.9884 | 0.6739 | 0.6739 | 0.9941 | 0.6739 | 0.6739 | 5063 | 30 | 30 | 62 |
| TabPFN (local) | 0.886 | 0.9826 | 0.5067 | 0.8261 | 0.9855 | 0.6281 | 0.7336 | 5019 | 74 | 16 | 76 |
| CatBoost | 0.416 | 0.9873 | 0.6806 | 0.5326 | 0.9955 | 0.5976 | 0.5568 | 5070 | 23 | 43 | 49 |
| Random Forest | 0.104 | 0.9826 | 0.5098 | 0.5652 | 0.9902 | 0.5361 | 0.5532 | 5043 | 50 | 40 | 52 |
| Logistic Regression | 0.985 | 0.9819 | 0.4857 | 0.3696 | 0.9929 | 0.4198 | 0.3881 | 5057 | 36 | 58 | 34 |

**Source files:** [paper_figures/paper_table3_pooled_f1.png](paper_figures/paper_table3_pooled_f1.png), [paper_figures/paper_table3_pooled_f1.csv](paper_figures/paper_table3_pooled_f1.csv)

---

## 5. Supplementary: follow-up-time leakage

These numbers are **not** the nested-CV headline. They come from the two single-split (70/30, GridSearchCV) notebooks that diagnosed why `Time since stent implantation` cannot enter a classifier. Nothing was re-run; values are the stored test-set metrics.

**What the column is.** For VLST = 1 it is time from index PCI to angiographic thrombosis (min 380 days, Wang median 697). For VLST = 0 it is completed event-free follow-up (min 1,241, max 1,605 days; cohort median follow-up 1,502). That is binary-ified survival time, not a baseline covariate.

### Supplementary Table S-TSSI. Single-split metrics with vs without the column

![Table S-TSSI](paper_figures/paper_table_s_tssi_leakage.png)

**Table S-TSSI.** Same stratified 70/30 split and tuning protocol. Logistic regression PR-AUC falls from 0.958 to 0.508 when the column is dropped; CatBoost from 0.977 to 0.658. Gaussian NB is unaffected (it never used the column). Nested-CV models in the main tables use the *without-TSSI* protocol.

**Source files:** [paper_figures/paper_table_s_tssi_leakage.png](paper_figures/paper_table_s_tssi_leakage.png), [paper_figures/paper_table_s_tssi_leakage.csv](paper_figures/paper_table_s_tssi_leakage.csv)

### Supplementary Figure S-TSSI. PR-AUC collapse

![Figure S-TSSI](paper_figures/paper_fig_s_tssi_pr_auc.png)

**Figure S-TSSI.** PR-AUC on the 1,556-row hold-out. The dotted line is class prevalence (0.0177). The leaky column produces near-perfect ranking; removing it returns models to a rare-event scale.

**Source file:** [paper_figures/paper_fig_s_tssi_pr_auc.png](paper_figures/paper_fig_s_tssi_pr_auc.png)

Notebooks: `baseline_tssi_leakage.ipynb`, `baseline_without_tssi.ipynb`. Table rebuilt by [`rebuild_tssi_leakage_table.py`](rebuild_tssi_leakage_table.py).

---

## 6. Supplementary: Wang 2020 integer score

These numbers are **not** a nested-CV fit. They come from [`wang_vlst_score.ipynb`](wang_vlst_score.ipynb), which scores Wang 2020 Table 2 **integer points** on all 5,185 rows with the published weights frozen. The same five outer folds as Part 4 (`StratifiedKFold(5, shuffle=True, random_state=42)`) are used only to evaluate that frozen score.

**Headline.** Full-cohort ROC-AUC **0.8013** (Wang published derivation c-statistic 0.80) and PR-AUC **0.1032**. Fold-mean ROC-AUC **0.8005 ± 0.0607**, PR-AUC **0.1134 ± 0.0518**. Nested-CV LightGBM is PR-AUC **0.6937** / ROC-AUC **0.9681**; TabPFN (local) is PR-AUC **0.6754** / ROC-AUC **0.9845**. The ML models still beat the published integer score on PR-AUC. It is **not** external validation (Wang’s c = 0.82 was Shantou).

**Encoding (do not photocopy Wang Table 1).** The SES point is on **`PES`**. The 4 post-dilation points go to **`No postdilation` = 1**. Using Wang Table 1’s 14 VLST “No post-dilation” cases as the 4-point group yields ROC-AUC **0.5084**.

**Risk bins.** Low ≤7: n = 3,135 (60.5%), rate 0.51%. Intermediate 8–9: n = 1,577 (30.4%), rate 2.22%. High ≥10: n = 473 (9.1%), rate 8.67%. Wang’s published n’s 3,135 / 1,837 / 473 sum to 5,445 ≠ 5,185; low and high n match this file.

The Cox linear predictor, Dangas decision-curve analysis, and Shantou scoring are **not** in this notebook.

### Supplementary Table S-Wang-bins. Observed VLST rate by published risk category

![Table S-Wang-bins](paper_figures/paper_table_s_wang_score_bins.png)

**Table S-Wang-bins.** Frozen integer score, cut at Wang’s published thresholds (≤7 / 8–9 / ≥10). Observed rates match Wang’s 0.5% / 2.2% / 8.7%. The intermediate *count* does not: Wang printed n = 1,837 for that bin.

**Source files:** [paper_figures/paper_table_s_wang_score_bins.png](paper_figures/paper_table_s_wang_score_bins.png), [paper_figures/paper_table_s_wang_score_bins.csv](paper_figures/paper_table_s_wang_score_bins.csv)

| Risk category | n | % of cohort | VLST events | Observed rate | Wang published n | Wang published rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low (≤7) | 3135 | 60.5 | 16 | 0.0051 | 3135 | 0.005 |
| intermediate (8–9) | 1577 | 30.4 | 35 | 0.0222 | 1837 | 0.022 |
| high (≥10) | 473 | 9.1 | 41 | 0.0867 | 473 | 0.087 |

### Supplementary Table S-Wang. Frozen score vs nested-CV models

![Table S-Wang](paper_figures/paper_table_s_wang_vs_ml.png)

**Table S-Wang.** Wang integer score: full-cohort ranking plus the same five outer folds, score not refit. LightGBM / TabPFN (local) / logistic regression: this Part 4 nested 5×4 CV (D4). PR-AUC is the informative metric at 1.77% prevalence.

**Source files:** [paper_figures/paper_table_s_wang_vs_ml.png](paper_figures/paper_table_s_wang_vs_ml.png), [paper_figures/paper_table_s_wang_vs_ml.csv](paper_figures/paper_table_s_wang_vs_ml.csv)

| Model | ROC-AUC | PR-AUC | ROC fold mean ± SD | PR fold mean ± SD | Protocol |
| --- | ---: | ---: | --- | --- | --- |
| Wang 2020 integer score (frozen) | 0.8013 | 0.1032 | 0.8005 ± 0.0607 | 0.1134 ± 0.0518 | Published points; folds evaluate only |
| LightGBM (untuned nested CV) | 0.9681 | **0.6937** | 0.9695 ± 0.0164 | 0.6941 ± 0.0917 | Part 4 nested 5×4 CV OOF |
| TabPFN (local) | **0.9845** | 0.6754 | 0.9846 ± 0.0030 | 0.6739 ± 0.0812 | Part 4 nested 5×4 CV OOF |
| Logistic regression (untuned nested CV) | 0.9224 | 0.3326 | 0.9235 ± 0.0251 | 0.3451 ± 0.1213 | Part 4 nested 5×4 CV OOF |

Fold-level frozen-score metrics: [paper_table_s_wang_score_folds.csv](paper_figures/paper_table_s_wang_score_folds.csv).

### Supplementary Figure S-Wang. Observed VLST rate by integer score

![Figure S-Wang](paper_figures/paper_fig_s_wang_score_rate.png)

**Figure S-Wang.** Observed VLST rate at each integer total. Bar labels are cell n (shown when n ≥ 20). The dashed line is cohort prevalence (0.0177). The score is a ranker, not a calibrated probability.

**Source file:** [paper_figures/paper_fig_s_wang_score_rate.png](paper_figures/paper_fig_s_wang_score_rate.png)

Notebook: [`wang_vlst_score.ipynb`](wang_vlst_score.ipynb).

---

## 7. File index

| ID | Type | File |
| --- | --- | --- |
| Table 0 | Table | [paper_table0_models.png](paper_figures/paper_table0_models.png) |
| Fig 1 | Figure | [paper_fig1_pr_roc_curves.png](paper_figures/paper_fig1_pr_roc_curves.png) |
| Table 1 | Table | [paper_table1_ranking.png](paper_figures/paper_table1_ranking.png) |
| Fig 2 | Figure | [paper_fig2_calibration_curves.png](paper_figures/paper_fig2_calibration_curves.png) |
| Fig 3 | Figure | [paper_fig3_confusion_matrices.png](paper_figures/paper_fig3_confusion_matrices.png) |
| Table 2 | Table | [paper_table2_nested_operating_point.png](paper_figures/paper_table2_nested_operating_point.png) |
| Table 3 | Table | [paper_table3_pooled_f1.png](paper_figures/paper_table3_pooled_f1.png) |
| Sweep | Figure | [best_model_threshold_fpfn_panel.png](paper_figures/best_model_threshold_fpfn_panel.png) |
| Table S-TSSI | Table | [paper_table_s_tssi_leakage.png](paper_figures/paper_table_s_tssi_leakage.png) |
| Fig S-TSSI | Figure | [paper_fig_s_tssi_pr_auc.png](paper_figures/paper_fig_s_tssi_pr_auc.png) |
| Table S-Wang-bins | Table | [paper_table_s_wang_score_bins.png](paper_figures/paper_table_s_wang_score_bins.png) |
| Table S-Wang | Table | [paper_table_s_wang_vs_ml.png](paper_figures/paper_table_s_wang_vs_ml.png) |
| Fig S-Wang | Figure | [paper_fig_s_wang_score_rate.png](paper_figures/paper_fig_s_wang_score_rate.png) |

---

*Figures 1–3 and Tables 0–3 are exported from the executed Kaggle run of `baseline_plus_tabpfn.ipynb` (local TabPFN; 9-level stent encoder). Thinking-high client metrics are not in this notebook.*
