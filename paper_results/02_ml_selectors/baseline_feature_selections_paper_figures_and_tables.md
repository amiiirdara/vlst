# Classic-model feature selection — paper figures and tables

This document gathers publication-oriented figures and tables from the multi-model feature selectors in `baseline_feature_selections.ipynb`.

**Cohort / protocol (2026-08-31 Kaggle run).** Full VLST cohort, n = 5,185. Target = `Stent thrombosis`. `Time since stent implantation` is dropped. This is **not** the TabPFN playground notebook (out of scope). **Paper protocol:** no parked 70/30 test — every row is split once into fit / val (`INNER_VAL_SIZE=0.2`, `random_state=42`): **fit = 4,148 rows (74 events)** / **val = 1,037 rows (18 events)**. **PR-AUC only.** LOCO / SHAP / FFS are **independent** (each takes its own cheap fit-slice importance pool). Budget: top-20; SHAP universe 40; LOCO cap 60; FFS pool 24 × 12 steps with early stop (`FFS_MIN_GAIN=0`); boosting 400 rounds. `USE_CACHE=False`. GPU: Tesla T4. Models use the **scaled** view: shared 9-level stent-brand encoder, then `ColumnTransformer` one-hot (drop-first) + `StandardScaler` → **88 columns** (81 raw − 1 brand + 8 dummies). Median / most-frequent imputers sit in that transformer; the CSV has **no missing values**, so they are inert. Selector hyperparameters are the notebook’s own factories (`C=2`, RF 500 trees, `lr=0.05`) — **not** `GridSearchCV` winners from `baseline_without_tssi.ipynb`.

**Kaggle note.** Per-selector CSVs (`selector_summary_long.csv`, `loco_*.csv`, …) were written to `/kaggle/working/model_feature_selectors` and are **not** in this repo. Tables below are reconstructed from the notebook’s displayed frames and the three compact PNGs embedded in the report cell. XGBoost’s 7-name three-way list was truncated in HTML as `… LV; WB…`; the alphabetically sorted completion is `WBC; eGFR`.

**Selectors.** LOCO = drop-one and refit on the val slice (cheap-importance prefix of 60). Coalition SHAP = permutation coalitions on a cheap-importance universe of 40 (not LOCO’s names). FFS = greedy forward search on its own 24-name pool, stop at 12 steps or when PR-AUC stops rising. Objective: **`pr_auc` only**. These catalogues are **interpretation / attribution**, not prediction, and do **not** feed Part 4. SMOTE is not used.

**Asset root:** [paper_figures/](paper_figures/)

---

## Contents

1. [Classic models (Table 0)](#1-classic-models)
2. [How much each selector keeps (Figure 1, Table 3, Figure 7)](#2-how-much-each-selector-keeps)
3. [Cross-model consensus (Table 1, Figure 6, Figure 2)](#3-cross-model-consensus)
4. [Within-model consensus, by classic model (Table 2, Figures 3–5)](#4-within-model-consensus-by-classic-model)
5. [Global intersection (Table 4)](#5-global-intersection)
6. [Priority-feature ranks (Table 5)](#6-priority-feature-ranks)
7. [Notebook compact plots (supplementary)](#7-notebook-compact-plots-supplementary)
8. [File index](#8-file-index)

---

## 1. Classic models

### Table 0. Models used in the selector notebook

![Table 0](paper_figures/paper_table0_classic_models.png)

**Table 0.** Seven sklearn-style classifiers from the notebook `MODEL_SPECS` (TabPFN omitted). Row colour encodes family: linear (navy), bagged trees (teal), boosting (violet). All seven share the same 88-column scaled matrix; only the inductive bias changes.

| Code | Classic model | Family | GPU | Specification (notebook) |
| --- | --- | --- | --- | --- |
| lr | Logistic regression | Linear | No | L2-penalized log-odds (C=2, balanced); additive on scaled features |
| rf | Random forest | Bagged trees | No | 500 deep trees, class_weight=balanced_subsample |
| rf_b | Random forest (subsample) | Bagged trees | No | Same as RF with max_samples=0.88 |
| cat | CatBoost | Boosting | Yes | GPU **Plain** boosting (`task_type=GPU`; Ordered is CPU-only); `eval_metric=PRAUC`; balanced class weights |
| xgb | XGBoost | Boosting | Yes | scale_pos_weight for VLST imbalance |
| xgb_b | XGBoost (subsample) | Boosting | Yes | Lower subsample / colsample_bytree (0.78) |
| lgb | LightGBM | Boosting | Yes | Leaf-wise growth, balanced class weights |

**How to read later tables through this lens.**

- **Logistic regression** can only use additive log-odds. Consensus sits on renal labs (`Cre`, `eGFR`), inflammation (`WBC`), sex (`Men`), ACS presentation (`UA`), and `LV`.
- **Random forests** split on interactions and keep both a lab and its clinical twin (`LVEF` beside `LV`).
- **Boosting** recovers post-dilation and `WBC` most often; LightGBM’s three-way set is `HbA1c; LV` only.

**Source files:** [paper_figures/paper_table0_classic_models.png](paper_figures/paper_table0_classic_models.png), [paper_figures/paper_table0_classic_models.csv](paper_figures/paper_table0_classic_models.csv)

---

## 2. How much each selector keeps

LOCO scores a 60-name cheap-importance prefix, so every model reports **60** unique LOCO names in `selector_summary_long`. SHAP’s universe is **40** by construction. FFS is the path length after early stop (4–12). These are **not** top-20 counts; the consensus tables below use top-20.

### Figure 1. Unique selected features by classic model and selector

![Figure 1](paper_figures/paper_fig1_unique_counts.png)

**Figure 1.** Unique feature counts in the selector log (PR-AUC). LOCO saturates the 60-feature cap for every model because 60 columns were scored, not because 60 were independently important. FFS is sparse because it stops when PR-AUC stops rising.

| Model | Family | LOCO | SHAP | FFS |
| --- | --- | ---: | ---: | ---: |
| lr | Linear | 60 | 40 | 12 |
| rf | Bagged trees | 60 | 40 | 12 |
| rf_b | Bagged trees | 60 | 40 | 8 |
| cat | Boosting | 60 | 40 | 4 |
| xgb | Boosting | 60 | 40 | 12 |
| xgb_b | Boosting | 60 | 40 | 11 |
| lgb | Boosting | 60 | 40 | 5 |

**Source file:** [paper_figures/paper_fig1_unique_counts.png](paper_figures/paper_fig1_unique_counts.png)

### Table 3. Union size per classic model

![Table 3](paper_figures/paper_table3_union_by_model.png)

**Table 3.** Size of the union of **top-20** sets across LOCO, SHAP, and FFS (PR-AUC only). Feature-name lists were truncated in the notebook HTML and are not reconstructed here.

| Code | Classic model | Family | Union size |
| --- | --- | --- | ---: |
| lr | Logistic regression | Linear | 32 |
| rf | Random forest | Bagged trees | 35 |
| rf_b | Random forest (subsample) | Bagged trees | 34 |
| cat | CatBoost | Boosting | 32 |
| xgb | XGBoost | Boosting | 31 |
| xgb_b | XGBoost (subsample) | Boosting | 30 |
| lgb | LightGBM | Boosting | 30 |

**Source files:** [paper_figures/paper_table3_union_by_model.png](paper_figures/paper_table3_union_by_model.png), [paper_figures/paper_table3_union_by_model.csv](paper_figures/paper_table3_union_by_model.csv)

### Figure 7. Union size (same numbers as Table 3)

![Figure 7](paper_figures/paper_fig7_union_by_model.png)

**Figure 7.** Per-model top-20 unions relative to the global unique count of **86** scored names (dashed line). No classic model recovers the full 86-name union on its own.

**Source file:** [paper_figures/paper_fig7_union_by_model.png](paper_figures/paper_fig7_union_by_model.png)

---

## 3. Cross-model consensus

A feature is “shared by all 7 models” only if it appears in every classic model’s **top-20** for that selector (PR-AUC).

### Table 1. Features shared by all classic models

![Table 1](paper_figures/paper_table1_common_by_algorithm.png)

**Table 1.** Cross-model intersection (row colour = selector). LOCO agrees on five labs/cardiac names. SHAP agrees only on `HGB` and `WBC`. FFS agrees on **nothing** — greedy paths diverge once each model’s own 24-name pool is searched independently.

| Algorithm | Metric | n common | Features shared by all 7 models |
| --- | --- | ---: | --- |
| LOCO | pr_auc | 5 | Cre; LV; LVEF; WBC; eGFR |
| SHAP | pr_auc | 2 | HGB; WBC |
| FFS | pr_auc | 0 | — |

**Source files:** [paper_figures/paper_table1_common_by_algorithm.png](paper_figures/paper_table1_common_by_algorithm.png), [paper_figures/paper_table1_common_by_algorithm.csv](paper_figures/paper_table1_common_by_algorithm.csv)

### Figure 6. Same intersection as bars

![Figure 6](paper_figures/paper_fig6_cross_model_common.png)

**Figure 6.** Bar height is `n common` from Table 1; labels are the shared names.

**Source file:** [paper_figures/paper_fig6_cross_model_common.png](paper_figures/paper_fig6_cross_model_common.png)

### Figure 2. Jaccard overlap of selector unions

![Figure 2](paper_figures/paper_fig2_jaccard.png)

**Figure 2.** Jaccard index between the unions of **top-20** sets (all seven models pooled). LOCO vs SHAP = **0.62**; SHAP vs FFS = **0.48**; LOCO vs FFS = **0.43**. These are moderate because the three selectors are **independent**. The previous 0.95–0.97 figure was an artefact of nesting SHAP/FFS inside one LOCO pool.

**Source file:** [paper_figures/paper_fig2_jaccard.png](paper_figures/paper_fig2_jaccard.png)

---

## 4. Within-model consensus, by classic model

Here the intersection is inside one model: names that LOCO, SHAP, and FFS all put in that model’s top-20 for PR-AUC.

### Table 2. LOCO ∩ SHAP ∩ FFS per model (PR-AUC)

![Table 2](paper_figures/paper_table2_consensus_by_model.png)

**Table 2.** Within-model three-selector consensus. Row colour = family. XGBoost has the largest set (7 names); LightGBM the smallest (2).

| Code | Classic model | Family | Metric | n (LOCO ∩ SHAP ∩ FFS) | Consensus features |
| --- | --- | --- | --- | ---: | --- |
| lr | Logistic regression | Linear | pr_auc | 6 | Cre; LV; Men; UA; WBC; eGFR |
| rf | Random forest | Bagged trees | pr_auc | 6 | HGB; LDL; LVEF; Men; WBC; eGFR |
| rf_b | Random forest (subsample) | Bagged trees | pr_auc | 4 | CaI; HGB; LVEF; WBC |
| cat | CatBoost | Boosting | pr_auc | 3 | 1.1:1Post dilation; HGB; WBC |
| xgb | XGBoost | Boosting | pr_auc | 7 | 1.1:1Post dilation; Aneurysm; Cre; HGB; LV; WBC; eGFR |
| xgb_b | XGBoost (subsample) | Boosting | pr_auc | 5 | 1.1:1Post dilation; LV; LVEF; WBC; eGFR |
| lgb | LightGBM | Boosting | pr_auc | 2 | HbA1c; LV |

**Source files:** [paper_figures/paper_table2_consensus_by_model.png](paper_figures/paper_table2_consensus_by_model.png), [paper_figures/paper_table2_consensus_by_model.csv](paper_figures/paper_table2_consensus_by_model.csv)

**ML consensus catalogue (union of Table 2, n = 13):** `1.1:1Post dilation`, `Aneurysm`, `CaI`, `Cre`, `HGB`, `HbA1c`, `LDL`, `LV`, `LVEF`, `Men`, `UA`, `WBC`, `eGFR`. This is the set compared with statistical FDR in Part 3.

### Figure 3. Consensus-set size

![Figure 3](paper_figures/paper_fig3_consensus_size.png)

**Figure 3.** Bar height is `n (LOCO ∩ SHAP ∩ FFS)` from Table 2. Colour = model family.

**Source file:** [paper_figures/paper_fig3_consensus_size.png](paper_figures/paper_fig3_consensus_size.png)

### Figure 4. Which features each classic model agrees on

![Figure 4](paper_figures/paper_fig4_feature_by_model.png)

**Figure 4.** Cell = 1 if the feature is in that model’s LOCO ∩ SHAP ∩ FFS set (PR-AUC). `WBC` appears in six of seven models; `eGFR` and `LV` in four; `1.1:1Post dilation` in the three boosting variants except LightGBM.

**Source file:** [paper_figures/paper_fig4_feature_by_model.png](paper_figures/paper_fig4_feature_by_model.png)

### Figure 5. Family stacked counts

![Figure 5](paper_figures/paper_fig5_family_stacked.png)

**Figure 5.** For each consensus feature, how many models in each family include it. `WBC` has support in all three families. `Men` is linear + bagged. `1.1:1Post dilation` is boosting-only. `Aneurysm` is XGBoost-only.

**Source file:** [paper_figures/paper_fig5_family_stacked.png](paper_figures/paper_fig5_family_stacked.png)

### Reading Table 2 / Figures 3–5 by classic model

**Logistic regression (`lr`).** Linear three-way: `Cre`, `LV`, `Men`, `UA`, `WBC`, `eGFR`. Sex and unstable-angina are almost unique to LR among the consensus names.

**Random forest (`rf`).** `HGB`, `LDL`, `LVEF`, `Men`, `WBC`, `eGFR`. Keeps haemoglobin and LDL that the linear model does not.

**Random forest, subsampled (`rf_b`).** `CaI`, `HGB`, `LVEF`, `WBC`. Treat `rf_b` as a sensitivity check on `rf`; `CaI` is unique to this variant.

**CatBoost (`cat`).** `1.1:1Post dilation`, `HGB`, `WBC`. Does not put `eGFR` or `LV` in the three-way set on this run.

**XGBoost (`xgb` / `xgb_b`).** Both recover `1.1:1Post dilation`, `LV`, `WBC`; the full XGB run also keeps `Aneurysm`, `Cre`, `HGB`, `eGFR`.

**LightGBM (`lgb`).** Smallest three-way: `HbA1c; LV`. This is the only model that puts `HbA1c` in the intersection — and that is enough to put `HbA1c` in the Part 3 consensus union.

---

## 5. Global intersection

### Table 4. Strictest intersection vs global union

![Table 4](paper_figures/paper_table4_global_common.png)

**Table 4.** Features that appear in **every** model × selector top-20: **none**. The complementary union of all scored names is **86**.

| Scope | n features | Features |
| --- | ---: | --- |
| All 7 models × LOCO, SHAP, FFS (PR-AUC top-20) | 0 | — |
| Any model / selector (union of scored names) | 86 | 86 unique names (full string not downloaded from Kaggle) |

**Source files:** [paper_figures/paper_table4_global_common.png](paper_figures/paper_table4_global_common.png), [paper_figures/paper_table4_global_common.csv](paper_figures/paper_table4_global_common.csv)

---

## 6. Priority-feature ranks

The notebook scores a hand-specified `PRIORITY_FEATURES` list (Wang Table 1 English labels) against each model × selector ranking. Most labels **do not match** the dataset column names (`Age, years` vs `Age`, `Male sex` vs `Men`, `aspirin` vs `Aspirin`). The display is the first 20 rows: CatBoost × LOCO then SHAP, PR-AUC only.

### Table 5. Priority ranks (display excerpt)

![Table 5](paper_figures/paper_table5_priority_ranks_excerpt.png)

**Table 5.** Hits under CatBoost LOCO: Current smoker (rank 24), Clopidogrel (50), Current drinking (51), Hypertension (55). SHAP hit: Hypertension (35). The rest miss because of the alias mismatch, not because the clinical variables were unscored.

**Source files:** [paper_figures/paper_table5_priority_ranks_excerpt.png](paper_figures/paper_table5_priority_ranks_excerpt.png), [paper_figures/paper_table5_priority_ranks_excerpt.csv](paper_figures/paper_table5_priority_ranks_excerpt.csv)

---

## 7. Notebook compact plots (supplementary)

![Figure S1](paper_figures/selector_model_algorithm_counts.png)

**Supplementary Figure S1.** Notebook heatmap of unique selected-feature counts. Paper restyle: Figure 1.

**Source file:** [paper_figures/selector_model_algorithm_counts.png](paper_figures/selector_model_algorithm_counts.png)

![Figure S2](paper_figures/selector_top_repeated_features.png)

**Supplementary Figure S2.** Features most often written into `selector_summary_long` (max 21 = 7 models × 3 selectors). `WBC` leads; `HGB` / `LV` / post-dilation complements / `eGFR` follow. `Stent type-SES_resolute` (a 9-level dummy) appears in the top 25 — the brand signal is now a named level, not 106 fragments.

**Source file:** [paper_figures/selector_top_repeated_features.png](paper_figures/selector_top_repeated_features.png)

![Figure S3](paper_figures/selector_overlap_heatmap.png)

**Supplementary Figure S3.** Notebook Jaccard heatmap of selector unions. Paper restyle: Figure 2.

**Source file:** [paper_figures/selector_overlap_heatmap.png](paper_figures/selector_overlap_heatmap.png)

---

## 8. File index

| ID | Type | File |
| --- | --- | --- |
| Table 0 | Table | [paper_table0_classic_models.png](paper_figures/paper_table0_classic_models.png) |
| Fig 1 | Figure | [paper_fig1_unique_counts.png](paper_figures/paper_fig1_unique_counts.png) |
| Table 1 | Table | [paper_table1_common_by_algorithm.png](paper_figures/paper_table1_common_by_algorithm.png) |
| Fig 2 | Figure | [paper_fig2_jaccard.png](paper_figures/paper_fig2_jaccard.png) |
| Table 2 | Table | [paper_table2_consensus_by_model.png](paper_figures/paper_table2_consensus_by_model.png) |
| Fig 3 | Figure | [paper_fig3_consensus_size.png](paper_figures/paper_fig3_consensus_size.png) |
| Fig 4 | Figure | [paper_fig4_feature_by_model.png](paper_figures/paper_fig4_feature_by_model.png) |
| Fig 5 | Figure | [paper_fig5_family_stacked.png](paper_figures/paper_fig5_family_stacked.png) |
| Fig 6 | Figure | [paper_fig6_cross_model_common.png](paper_figures/paper_fig6_cross_model_common.png) |
| Table 3 | Table | [paper_table3_union_by_model.png](paper_figures/paper_table3_union_by_model.png) |
| Fig 7 | Figure | [paper_fig7_union_by_model.png](paper_figures/paper_fig7_union_by_model.png) |
| Table 4 | Table | [paper_table4_global_common.png](paper_figures/paper_table4_global_common.png) |
| Table 5 | Table | [paper_table5_priority_ranks_excerpt.png](paper_figures/paper_table5_priority_ranks_excerpt.png) |
| Fig S1 | Supp. figure | [selector_model_algorithm_counts.png](paper_figures/selector_model_algorithm_counts.png) |
| Fig S2 | Supp. figure | [selector_top_repeated_features.png](paper_figures/selector_top_repeated_features.png) |
| Fig S3 | Supp. figure | [selector_overlap_heatmap.png](paper_figures/selector_overlap_heatmap.png) |

---

*Numbers from the 2026-08-31 paper-protocol Kaggle run of* `baseline_feature_selections.ipynb` *(seven classic models, PR-AUC, independent selectors, 9-level stent encoder → 88 columns, fit/val 4148/1037). Regenerated by* `code/modeling/tools/rebuild_part2_paper_figures.py`.
