# Statistical vs machine-learning feature extraction in VLST

This note compares **what was extracted** from the same VLST cohort by (i) classical statistical association tests and (ii) classic-model feature selectors, then explains **why the two catalogues only partly overlap**.

Sources: [EDA_paper_figures_and_tables.md](../EDA_paper_figures_and_tables.md) (`eda.ipynb`) and [baseline_feature_selections_paper_figures_and_tables.md](../../modeling/interpretability/baseline_feature_selections_paper_figures_and_tables.md) (`baseline_feature_selections.ipynb`).

**Asset root:** [paper_figures/](paper_figures/)

---



## Contents

1. [What each approach is asking](#1-what-each-approach-is-asking)
2. [How common are the extracted features?](#2-how-common-are-the-extracted-features)
3. [Features found by both approaches](#3-features-found-by-both-approaches)
4. [Statistical-only features](#4-statistical-only-features)
5. [Machine-learning-only features](#5-machine-learning-only-features)
6. [Domain pattern](#6-domain-pattern)
7. [Methodological reasons for disagreement](#7-methodological-reasons-for-disagreement)
8. [File index](#8-file-index)

---



## 1. What each approach is asking

The two pipelines are not two estimates of the same quantity. They optimize different questions on slightly different feature views.


|                              | Statistical EDA                                                                       | Classic-model selectors                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Question**                 | Does this column’s *marginal* distribution differ by VLST after multiplicity control? | If I train `lr` / `rf` / boosting, which columns does the *fitted model* need for hold-out PR-AUC / F1 / F2? |
| **Unit of evidence**         | One test per feature (Welch, Mann–Whitney, χ² / Fisher) plus FDR                      | LOCO (refit without the column), coalition SHAP, greedy FFS                                                  |
| **Sample**                   | Full cohort, n ≈ 5,185, ~92 events                                                    | Same split protocol: train 3,629 / test 1,556 (64 / 28 events)                                               |
| **Feature view**             | Raw clinical columns                                                                  | Scaled / encoded matrix (185 columns); `Time since stent implantation` dropped                               |
| **Discovery rule used here** | Univariate FDR q < 0.05 (plus a sparse multivariable logistic check)                  | Names in **LOCO ∩ SHAP ∩ FFS** top-12 for at least one model × metric                                        |
| **Multiplicity**             | Benjamini–Hochberg across the tested family                                           | Implicit: top-12 of a 40-column LOCO pool (smoke run)                                                        |


**Statistical catalogue (n = 20, excluding time-at-risk).** Continuous FDR: `WBC`, `eGFR`, `LV`, `CKD5`, `No.of stents per lesion`, `HbA1c`, `NO.of vessels`, `Total stent length`, `Fiberinogen`. Binary FDR: `1.1:1Post dilation`, `No postdilation`, `CKD90`, `Previous PCI`, `3-vessel disease`, `Clopidogrel`, `Diabetes`, `PES`, `Multi-vessel CAD`, `Single-vessel disease`. Categorical: `Stent type-SES`. `Time since stent implantation` is the strongest univariate hit but is a follow-up / time-at-risk variable, not a baseline predictor, and is excluded from ML.

**ML consensus catalogue (n = 20).** Union of LOCO ∩ SHAP ∩ FFS names across logistic regression, random forests, CatBoost, XGBoost, and LightGBM: `WBC`, `eGFR`, `LV`, `Cre`, `Men`, `LVEF`, `Previous PCI`, `Fiberinogen`, `HGB`, `Platelet`, `HL`, `STEMI`, `Hypertension`, `Fast-Glu`, `TG`, `TCL`, `CaI`, `Min-stent diameter`, `Current drinking`, `History of HF`.

A looser ML set (**frequent selection**, top-repeated names in the selector log) additionally includes ACS presentation and history variables (`NSTEMI`, `UA`, `Previous MI`, `Previous CABG`, …) that are selected often but rarely survive the three-selector intersection.

---



## 2. How common are the extracted features?

Only **5 of 20** statistical FDR features also sit in the ML three-selector consensus. Conversely, **15 of 20** ML-consensus names fail univariate FDR. Jaccard overlap of the two 20-name sets is 5 / 35 ≈ **0.14**.

### Figure 1. Overlap of the two extraction catalogues

Figure 1

**Figure 1.** Left circle: univariate FDR q < 0.05 (time-since-stent excluded). Right circle: features in LOCO ∩ SHAP ∩ FFS top-12 for at least one classic model and metric. The intersection is `WBC`, `eGFR`, `LV`, `Fiberinogen`, `Previous PCI`.

**Source file:** [paper_figures/fig1_venn_overlap.png](paper_figures/fig1_venn_overlap.png)

### Figure 2. Presence by extractor

Figure 2

**Figure 2.** Navy cells mark membership. Columns: statistical FDR; statistical multivariable (bootstrap CI excluding 1 in the sparse logistic); ML consensus; ML frequent (top-repeated selector log). `Time since stent implantation` is statistical-only by construction (dropped before ML). `LVEF` and `Men` are ML-side even though they fail FDR.

**Source file:** [paper_figures/fig2_presence_heatmap.png](paper_figures/fig2_presence_heatmap.png)

### Table 1. Membership of every compared name

Table 1

**Table 1.** Green = both catalogues; navy tint = stats FDR only; violet = ML consensus only; pale violet = ML frequent only; orange = structural time-at-risk.

**Source files:** [paper_figures/table_feature_by_method.png](paper_figures/table_feature_by_method.png), [paper_figures/table_feature_by_method.csv](paper_figures/table_feature_by_method.csv)

---



## 3. Features found by both approaches

These five names are the only ones that are both a **full-cohort association discovery** and a **predictive-model necessity**.

### Table 2. Shared features

Table 2


| Feature      | Domain     | Statistical evidence         | ML evidence                                                    | Why both keep it                                                  |
| ------------ | ---------- | ---------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------------- |
| WBC          | Laboratory | MW r = 0.13, q = 9.5e-20     | In **every** model × selector union; CatBoost/XGB/RF consensus | Inflammation is a mean shift *and* a column models cannot replace |
| eGFR         | Laboratory | Welch d = −0.71, q = 3.7e-19 | Global ML intersection; LOCO/SHAP core                         | Filtration: largest continuous effect; LOCO drop is costly        |
| LV           | Cardiac    | Welch d = 1.13, q = 3.3e-16  | Cross-model LOCO/SHAP; CatBoost/XGB consensus                  | Large location shift and a high-gain tree split                   |
| Fiberinogen  | Laboratory | MW r = 0.035, q = 0.029      | RF F2 consensus                                                | Weak haemostasis signal; still used at a recall-heavy threshold   |
| Previous PCI | History    | Fisher OR = 6.5, q = 2e-4    | RF F1 consensus                                                | Rare, high-OR binary: a 2×2 hit and a clean tree split            |


**Source files:** [paper_figures/table_shared_features.png](paper_figures/table_shared_features.png), [paper_figures/table_shared_features.csv](paper_figures/table_shared_features.csv)

`WBC` and `eGFR` are the strictest ML global intersection (all seven models × all three selectors). That matches their position at the top of the FDR ranking. `Fiberinogen` and `Previous PCI` are weaker shared hits: they survive FDR but only some model families (mainly forests) put them in the three-way intersection.

---



## 4. Statistical-only features

Fifteen FDR discoveries never enter ML LOCO ∩ SHAP ∩ FFS. They are not “false”; they fail a *different* filter: a 12-column predictive shortlist on an encoded matrix, after a 40-column LOCO cap, on a 28-event test set.

### Table 3. FDR hits missing from ML consensus

Table 3


| Feature                 | Domain         | Why statistics keeps it and ML top-12 does not                                                             |
| ----------------------- | -------------- | ---------------------------------------------------------------------------------------------------------- |
| 1.1:1Post dilation      | Procedural     | Strong 2×2 (OR 0.19); complement of `No postdilation`. Collinear pair; may never enter the LOCO pool of 40 |
| No postdilation         | Procedural     | Univariate OR 5.4; multivariable OR collapses toward 1 once the complement is modelled                     |
| CKD90                   | Renal cutpoint | Binary threshold on the same axis as `eGFR`. ML keeps the continuous lab, not the cut                      |
| CKD5                    | Renal cutpoint | FDR hit; adjusted OR *flips sign* (collinear with eGFR). In the ML union prefix, not in 3-way consensus    |
| 3-vessel disease        | Anatomy        | χ² discovery; collinear with `NO.of vessels` / multi-vessel CAD                                            |
| Multi-vessel CAD        | Anatomy        | Same information as single-vessel (complements)                                                            |
| Single-vessel disease   | Anatomy        | Protective complement of multi-vessel disease                                                              |
| NO.of vessels           | Anatomy        | Continuous count of the same anatomy cluster                                                               |
| No.of stents per lesion | Procedural     | Tiny effect (MW r = 0.037); not competitive in a 12-feature predictive list                                |
| Total stent length      | Procedural     | Small effect; collinear with stent count / vessel burden                                                   |
| HbA1c                   | Laboratory     | FDR hit that attenuates after adjustment (OR 0.87); Diabetes / Fast-Glu compete                            |
| Clopidogrel             | Medication     | Full-cohort drug association; weak for ranking 28 test events                                              |
| Diabetes                | Comorbidity    | Univariate FDR; multivariable CI includes 1                                                                |
| PES                     | Stent type     | Polymer binary; collinear with `Stent type-SES`                                                            |
| Stent type-SES          | Stent type     | Multi-level factor; one-hot encoding *splits* the χ² signal across rare dummy columns                      |


**Source files:** [paper_figures/table_stats_only.png](paper_figures/table_stats_only.png), [paper_figures/table_stats_only.csv](paper_figures/table_stats_only.csv)

Three recurring mechanisms:

1. **Collinear families.** Univariate tests score *every* member of a redundant block (vessel-disease binaries, postdilation complements, CKD cutpoints vs eGFR, PES vs stent type). FDR can declare several of them significant. A fitted model only needs one representative, and greedy FFS / LOCO will keep the mate that helps hold-out metric, not every correlated twin.
2. **Encoding.** `Stent type-SES` is one χ² test on 9 levels. In the scaled ML view it becomes many sparse dummies; none of them ranks in a top-12 of 185 columns.
3. **Candidate-pool truncation.** Smoke-mode LOCO is capped at 40 columns. Procedural binaries that are not in that pool cannot appear in SHAP or FFS either, because those selectors are nested inside the LOCO-ranked universe.

---



## 5. Machine-learning-only features

Fifteen consensus names fail univariate FDR. ML is not “finding associations the tests missed” in the NHST sense; it is finding **columns that change a model’s hold-out score**, including surrogates, interactions, and weak splits.

### Table 4. ML consensus names that fail FDR

Table 4


| Feature            | Univariate vs VLST    | Why ML consensus keeps it and FDR does not                                                       |
| ------------------ | --------------------- | ------------------------------------------------------------------------------------------------ |
| Cre                | ns (p = 0.88)         | Redundant with eGFR marginally; still a renal surrogate when eGFR is noisy or left out           |
| Men                | ns (p = 0.27)         | `Men × eGFR` interaction is FDR-significant in the EDA screen; LR uses sex as an additive offset |
| LVEF               | raw p = 0.033, FDR ns | Borderline mean test; domain multivariable OR persists; trees split on systolic function         |
| HGB                | raw p = 0.039, FDR ns | CatBoost/XGB F-score consensus: thresholded metrics, not a location test                         |
| Fast-Glu           | raw p = 0.025, FDR ns | LR F2; correlated with HbA1c / diabetes (those *do* pass FDR)                                    |
| Platelet           | ns                    | RF/CatBoost haemostasis panel with Fiberinogen                                                   |
| HL                 | ns                    | Lipid split that helps rare-event ranking in boosting / RF                                       |
| STEMI              | ns                    | Presentation split on hold-out PR-AUC/F2, not a 2×2 discovery                                    |
| Current drinking   | ns                    | LightGBM F1; lifestyle split, unstable with 28 test events                                       |
| History of HF      | ns                    | LightGBM F2; sparse history indicator                                                            |
| Hypertension       | ns                    | CatBoost F2; common comorbidity, weak marginal φ                                                 |
| TG                 | ns                    | LR PR-AUC additive lipid term                                                                    |
| TCL                | ns                    | XGB PR-AUC lipid surrogate                                                                       |
| Min-stent diameter | ns                    | LR PR-AUC geometric term after scaling                                                           |
| CaI                | raw p = 0.051, FDR ns | RF_b PR-AUC; sits on the FDR boundary                                                            |


**Source files:** [paper_figures/table_ml_only.png](paper_figures/table_ml_only.png), [paper_figures/table_ml_only.csv](paper_figures/table_ml_only.csv)

Three recurring mechanisms:

1. **Surrogates of a stronger FDR hit.** `Cre` carries almost no univariate VLST signal because `eGFR` already does. A linear or tree model that cannot use eGFR (or that splits on creatinine first) will still list Cre. That is predictive redundancy, not a new biological discovery.
2. **Interactions and offsets that univariate tests do not see.** `Men` is not associated with VLST on its own (p = 0.27), but `Men × eGFR` is an FDR-significant interaction in the EDA screen, and the domain joint logistic gives Men an adjusted OR of 3.3. Logistic regression’s consensus (`Cre`, `Men`, …) is that interaction/offset showing up as a main-effect column.
3. **Different error and sample.** FDR is a full-cohort mean/2×2 statement with ~92 events. LOCO/SHAP/FFS optimize PR-AUC or F2 on 28 test events. Weak ACS/history/lipid splits can move that metric without moving a χ² p-value across the FDR line. LightGBM’s consensus (`Current drinking`, `History of HF`, `HL`) is the clearest example of metric-and-hold-out artefacts.

---



## 6. Domain pattern



### Figure 4. Extracted counts by clinical domain

Figure 4

**Figure 4.** Statistical FDR is concentrated in laboratory, procedural/stent, and anatomy blocks (the EDA domain screen). ML consensus is heavier on laboratory *plus* cardiac function, demographics, and ACS presentation, and almost empty on anatomy binaries and medications. That is the collinear-family vs surrogate/interaction split from sections 4–5, drawn by domain.

**Source file:** [paper_figures/fig4_domain_counts.png](paper_figures/fig4_domain_counts.png)

Statistics therefore “owns” **stent technique and anatomy coding** (postdilation, vessel-disease labels, stent type). Machine learning “owns” **cardiac function twins** (`LVEF` next to `LV`), **sex**, and **labs that are collinear with FDR hits** (`Cre`, `HGB`, lipids). Both own **WBC, eGFR, LV**.

---



## 7. Methodological reasons for disagreement



### Figure 3. Buckets

Figure 3

**Figure 3.** Counts of names in this comparison assigned to a primary methodological bucket (one bucket per feature; the anatomy/stent collinear family is grouped).

**Source file:** [paper_figures/fig3_reason_buckets.png](paper_figures/fig3_reason_buckets.png)

**Why a feature can appear in statistics and not in ML**

- Univariate tests do not penalize redundancy. FDR will list `3-vessel disease`, `Multi-vessel CAD`, `Single-vessel disease`, and `NO.of vessels` if each 2×2/t-test is small. A model only needs one of them.
- Complements are two encodings of one bit (`1.1:1Post dilation` vs `No postdilation`). χ² sees both; multivariable logistic already showed the pair is not independently identified.
- Categorical χ² on `Stent type-SES` does not survive one-hot fragmentation in 185 columns.
- The ML smoke run never scores the full 185-column matrix with LOCO (cap 40). Absence from consensus is sometimes “not in the pool,” not “the model disproved the association.”
- Hold-out PR-AUC with 28 events is under-powered for moderate ORs (Clopidogrel 0.50, Diabetes 1.89) that FDR can still detect on 92 events.

**Why a feature can appear in ML and not in statistics**

- Predictive importance is not a marginal p-value. LOCO asks whether the *rest of the model* can compensate after a refit. SHAP asks for coalition credit. FFS asks for greedy hold-out gain. None of these is a two-sample test.
- Correlated twins: the univariate test of `Cre` is null because `eGFR` already captures renal function; the model may still split on Cre.
- Interactions: `Men × eGFR` is an EDA FDR hit; univariate `Men` is not. LR consensus includes `Men`.
- Operating-point metrics (F1/F2) promote labs that shift a decision threshold (`HGB`, `Platelet`) even when the mean difference is FDR-small.
- LightGBM’s leaf-wise default disagrees with CatBoost/XGBoost (no `LV`/`eGFR` in its three-way intersection). Some ML-only names are **algorithm artefacts**, not cohort discoveries.

**Practical reading.** Treat the 5-name intersection (`WBC`, `eGFR`, `LV`, `Fiberinogen`, `Previous PCI`) as the robust extraction set: associated in the cohort *and* used by fitted classic models. Treat statistical-only anatomy/stent/drug names as **association findings that need a non-redundant representative** before they enter a predictor. Treat ML-only names as **hypothesis-generating predictive correlates** until they pass a pre-specified association or external-validation bar.

---



## 8. File index


| ID      | Type   | File                                                                     |
| ------- | ------ | ------------------------------------------------------------------------ |
| Fig 1   | Figure | [fig1_venn_overlap.png](paper_figures/fig1_venn_overlap.png)             |
| Fig 2   | Figure | [fig2_presence_heatmap.png](paper_figures/fig2_presence_heatmap.png)     |
| Table 1 | Table  | [table_feature_by_method.png](paper_figures/table_feature_by_method.png) |
| Table 2 | Table  | [table_shared_features.png](paper_figures/table_shared_features.png)     |
| Table 3 | Table  | [table_stats_only.png](paper_figures/table_stats_only.png)               |
| Table 4 | Table  | [table_ml_only.png](paper_figures/table_ml_only.png)                     |
| Fig 3   | Figure | [fig3_reason_buckets.png](paper_figures/fig3_reason_buckets.png)         |
| Fig 4   | Figure | [fig4_domain_counts.png](paper_figures/fig4_domain_counts.png)           |


---

*Statistical names: univariate FDR q < 0.05 from* `eda.ipynb` *(time-since-stent excluded from the overlap count). ML names: LOCO ∩ SHAP ∩ FFS top-12 from the smoke run of* `baseline_feature_selections.ipynb` *(seven classic models). Re-run either notebook in full mode if the catalogues change, then refresh this comparison.*