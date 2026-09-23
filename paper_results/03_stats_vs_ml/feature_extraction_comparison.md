# Statistical vs machine-learning feature extraction in VLST

This note compares **what was extracted** from the same VLST cohort by (i) classical statistical association tests and (ii) classic-model feature selectors, then explains **why the two catalogues only partly overlap**. This is a methods comparison of two association / attribution catalogues, not a prediction result. The ML side is **ALL LEAKS OFF** (TSSI+WBC dropped, labs quantized, stent/scaler train-only, no SMOTE — [`../anti_leakage_protocol.md`](../anti_leakage_protocol.md)); the FDR side still ranks TSSI and WBC as association hits. Dual-label that discrepancy.

Sources: [EDA_paper_figures_and_tables.md](../01_eda/EDA_paper_figures_and_tables.md) (`eda.ipynb`) and [baseline_feature_selections_paper_figures_and_tables.md](../02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md) (`baseline_feature_selections.ipynb`). Overlap arithmetic and figures are produced by [`rebuild_part3_paper_figures.py`](../../code/modeling/tools/rebuild_part3_paper_figures.py) from the 2026-09-19 selector dump. `stats_vs_ml_comparison.ipynb` loads that dump (ML-10; Jaccard 5/25 = 0.20) and calls the same rebuild. Pre-antileak ML-13 / Jaccard 5/28 catalogues are superseded.

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
| **Question**                 | Does this column’s *marginal* distribution differ by VLST after multiplicity control? | If I train `lr` / `rf` / boosting, which columns does the *fitted model* need for hold-out PR-AUC? |
| **Sample**                   | Full cohort, n = 5,185, 92 events                                                     | Full-cohort fit/val: 4,148 / 1,037 rows (74 / 18 events). No unused outer test. |
| **Feature view**             | Raw clinical columns; `Stent type-SES` collapsed to 9 levels for χ²                  | Scaled matrix (**87 columns**), **ALL LEAKS OFF**: TSSI **and WBC dropped** (time-definition leak; WBC batch/precision marker), named labs quantized pre-split, shared 9-level brand encoder **train-only**, then OHE drop-first + `StandardScaler` (train-only). Imputers are inert (no NaNs). No SMOTE. |
| **Discovery rule used here** | Univariate FDR q < 0.05 (plus a sparse multivariable logistic check)                  | Names in **LOCO ∩ SHAP ∩ FFS** top-20 for at least one model (PR-AUC)                                        |
| **Multiplicity**             | Benjamini–Hochberg across the tested family                                           | Top-20 of independent cheap-importance pools (LOCO 60 / SHAP 40 / FFS 24)                     |


**Statistical catalogue (n = 20 names, excluding time-at-risk).** Continuous FDR: `WBC`, `eGFR`, `LV`, `CKD5`, `No.of stents per lesion`, `HbA1c`, `NO.of vessels`, `Total stent length`, `Fiberinogen`. Binary FDR: `1.1:1Post dilation`, `No postdilation`, `CKD90`, `Previous PCI`, `3-vessel disease`, `Clopidogrel`, `Diabetes`, `PES`, `Multi-vessel CAD`, `Single-vessel disease`. Categorical: `Stent type-SES`. `Time since stent implantation` is the strongest univariate hit but is a follow-up / time-at-risk variable, not a baseline covariate, and is excluded from ML. **`WBC` remains in this FDR list** even though the 2026-09-19 selector dump drops it from the ML matrix.

Of these 20 names, at least 8 are redundant re-encodings: the post-dilation complements (`1.1:1Post dilation`, `No postdilation` — 2 slots for 1 bit), the vessel-disease family (`3-vessel disease`, `Multi-vessel CAD`, `Single-vessel disease`, `NO.of vessels` — 4 slots for 1 construct), and the renal family (`CKD5`, `CKD90` alongside continuous `eGFR` — 3 slots for 1 construct). The “20 statistical discoveries” headline is a **name count**. Report it as roughly **12 distinct clinical constructs**. Jaccard 5/25 uses the name lists (that is what the selectors and FDR tests emit); do not rewrite the Venn as 12 vs 10.

**ML consensus catalogue (n = 10).** Union of LOCO ∩ SHAP ∩ FFS names across logistic regression, random forests, CatBoost, XGBoost, and LightGBM (PR-AUC, top-20, 2026-09-19 dump): `Clopidogrel`, `Cre`, `HGB`, `HbA1c`, `LDL`, `LV`, `Men`, `No postdilation`, `Stent type-SES_xiencev`, `eGFR`.

A looser ML set (**frequent selection**, names with ≥14 appearances in `selector_summary_long` that are not in the three-way union) additionally includes `LVEF`, `Total stent length`, `Previous PCI`, `1.1:1Post dilation`, `CaI`, `Stent type-SES_tivoli`, `Initial diagnosis-AMI`, `TCL`, `Fast-Glu`, `PES`.

---

## 2. How common are the extracted features?

Only **5 of 20** statistical FDR *names* also sit in the ML three-selector consensus. Conversely, **5 of 10** ML-consensus names fail univariate FDR. Jaccard overlap of the 20-name and 10-name sets is 5 / 25 = **0.20**. Dual label: `WBC` cannot enter the ML catalogue on this dump because it was dropped before encoding (same anti-leakage list as TSSI). If WBC is stripped from the FDR side as well, the name-count Jaccard is 5 / 24.

### Figure 1. Overlap of the two extraction catalogues

![Figure 1](paper_figures/fig1_venn_overlap.png)

**Figure 1.** Left circle: univariate FDR q < 0.05 (time-since-stent excluded). Right circle: features in LOCO ∩ SHAP ∩ FFS top-20 for at least one classic model (PR-AUC). The intersection is `Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR`.

**Source file:** [paper_figures/fig1_venn_overlap.png](paper_figures/fig1_venn_overlap.png)

### Figure 2. Presence by extractor

![Figure 2](paper_figures/fig2_presence_heatmap.png)

**Figure 2.** Navy cells mark membership. Columns: statistical FDR; statistical multivariable (Wald CI excluding 1 in the sparse logistic); ML consensus; ML frequent (top-repeated selector log). `Time since stent implantation` is statistical-only by construction (dropped before ML). `WBC` is FDR-only because it is not in the ML matrix. `Men` and `Cre` are ML-side even though they fail FDR.

**Source file:** [paper_figures/fig2_presence_heatmap.png](paper_figures/fig2_presence_heatmap.png)

### Table 1. Membership of every compared name

![Table 1](paper_figures/table_feature_by_method.png)

**Table 1.** Green = both catalogues; navy tint = stats FDR only; violet = ML consensus only; pale violet = ML frequent only; orange = structural time-at-risk.

**Source files:** [paper_figures/table_feature_by_method.png](paper_figures/table_feature_by_method.png), [paper_figures/table_feature_by_method.csv](paper_figures/table_feature_by_method.csv)

---

## 3. Features found by both approaches

These five names are the only ones that are both a **full-cohort association discovery** and a **predictive-model necessity** under the 2026-09-19 selectors.

### Table 2. Shared features

![Table 2](paper_figures/table_shared_features.png)


| Feature | Domain | Statistical evidence | ML evidence | Why both keep it |
| --- | --- | --- | --- | --- |
| Clopidogrel | Medication | χ² FDR; multivariable Wald CI excludes 1 | cat / lr three-way | Full-cohort drug association that LR and CatBoost also need for val PR-AUC |
| HbA1c | Laboratory | MW r = 0.052, q = 7e-4 | cat / lgb three-way | Glycaemic FDR hit that boosting still needs after WBC is dropped |
| LV | Cardiac | Welch d = 1.13, q = 3.3e-16 | cat / rf three-way | Large location shift and a high-gain tree split |
| No postdilation | Procedural | χ² OR ≈ 5.4 | CatBoost three-way | Complement of 1.1:1 post-dilation; CatBoost keeps this flag on this dump |
| eGFR | Laboratory | Welch d = −0.71, q = 3.7e-19 | lr / rf_b / xgb / xgb_b three-way; LOCO all-7 | Filtration: largest continuous effect; LOCO drop is costly |


**Source files:** [paper_figures/table_shared_features.png](paper_figures/table_shared_features.png), [paper_figures/table_shared_features.csv](paper_figures/table_shared_features.csv)

There is **no** name in all 7 × 3 selector top-20s. `WBC` and `1.1:1Post dilation` were shared hits on the 2026-08-31 run; they are now stats-only (`WBC` structurally unavailable; the 1.1:1 flag remains frequently selected).

---

## 4. Statistical-only features

Fifteen FDR discoveries never enter ML LOCO ∩ SHAP ∩ FFS. They are not “false”; they fail a *different* filter: a 20-column predictive shortlist on an 87-column encoded matrix (WBC already dropped), scored on an 18-event val slice.

### Table 3. FDR hits missing from ML consensus

![Table 3](paper_figures/table_stats_only.png)


| Feature | Domain | Why statistics keeps it and ML top-20 does not |
| --- | --- | --- |
| WBC | Laboratory | Strongest continuous FDR hit, but dropped from the ML matrix with TSSI on this run. Selectors never see it |
| 1.1:1Post dilation | Procedural | χ² complement of `No postdilation` (now in consensus). Still in several unions; no model three-way on this dump |
| CKD90 | Renal cutpoint | Binary threshold on the same axis as `eGFR`. ML keeps the continuous lab, not the cut |
| CKD5 | Renal cutpoint | FDR hit; adjusted OR *flips sign* (collinear with eGFR). Often selected, not in 3-way consensus |
| 3-vessel disease | Anatomy | χ² discovery; collinear with `NO.of vessels` / multi-vessel CAD |
| Multi-vessel CAD | Anatomy | Same information as single-vessel (complements) |
| Single-vessel disease | Anatomy | Complement of multi-vessel disease (same 2×2 inverted) |
| NO.of vessels | Anatomy | Continuous count of the same anatomy cluster |
| No.of stents per lesion | Procedural | Tiny effect (MW r = 0.037); not in any model three-way set |
| Total stent length | Procedural | Small effect; collinear with stent count / vessel burden; frequently selected |
| Diabetes | Comorbidity | Univariate FDR; multivariable CI includes 1; trees may split on HbA1c |
| PES | Stent type | Polymer binary; collinear with the 9-level brand column |
| Stent type-SES | Stent type | χ² on 9 collapsed brands. ML one-hots those 9 levels; `Stent type-SES_xiencev` is the three-way dummy |
| Previous PCI | History | Fisher OR 6.49. Frequently selected, but no model puts it in LOCO ∩ SHAP ∩ FFS on PR-AUC |
| Fiberinogen | Laboratory | Weak MW r = 0.035. Frequently selected; PR-AUC three-way no longer keeps it |


**Source files:** [paper_figures/table_stats_only.png](paper_figures/table_stats_only.png), [paper_figures/table_stats_only.csv](paper_figures/table_stats_only.csv)

Three recurring mechanisms:

1. **Collinear families.** Univariate tests score *every* member of a redundant block (vessel-disease binaries, postdilation complements, CKD cutpoints vs eGFR, PES vs stent type). FDR can declare several of them significant. A fitted model only needs one representative.
2. **Encoding.** `Stent type-SES` is one χ² test on 9 levels. In the scaled ML view it becomes 8 sparse dummies; `Stent type-SES_xiencev` is the three-way name, and `Stent type-SES_tivoli` is frequently selected.
3. **Different filter, not a missing pool — plus one dropped column.** Selectors rank their own cheap-importance prefixes (60 / 40 / 24). Absence from consensus means “not in LOCO ∩ SHAP ∩ FFS top-20,” except `WBC`, which was never scored.

---

## 5. Machine-learning-only features

Five consensus names fail univariate FDR. ML is not “finding associations the tests missed” in the NHST sense; it is finding **columns that change a model’s hold-out PR-AUC**, including surrogates, interactions, and weak splits.

### Table 4. ML consensus names that fail FDR

![Table 4](paper_figures/table_ml_only.png)


| Feature | Univariate vs VLST | Why ML consensus keeps it and FDR does not |
| --- | --- | --- |
| Cre | ns (p = 0.88) | Redundant with eGFR marginally; still a renal surrogate when eGFR is noisy or left out |
| Men | ns (p = 0.27) | `Men × eGFR` interaction is FDR-significant in the EDA screen; LR uses sex as an additive offset |
| HGB | raw p = 0.039, FDR ns | rf / xgb / xgb_b three-way: ranking, not a location test |
| LDL | ns (p = 0.33) | cat / rf three-way lipid split on the val-slice PR-AUC |
| Stent type-SES_xiencev | parent `Stent type-SES` is FDR; dummy is not a univariate column | xgb_b three-way on one 9-level brand dummy after OHE |


**Source files:** [paper_figures/table_ml_only.png](paper_figures/table_ml_only.png), [paper_figures/table_ml_only.csv](paper_figures/table_ml_only.csv)

Three recurring mechanisms:

1. **Surrogates of a stronger FDR hit.** `Cre` carries almost no univariate VLST signal because `eGFR` already does. A linear or tree model that cannot use eGFR (or that splits on creatinine first) will still list Cre.
2. **Interactions and offsets that univariate tests do not see.** `Men` is not associated with VLST on its own (p = 0.27), but `Men × eGFR` is an FDR-significant interaction in the EDA screen, and the domain joint logistic gives Men an adjusted OR of 3.3.
3. **Different error and sample.** FDR is a full-cohort mean/2×2 statement with 92 events. LOCO/SHAP/FFS optimize PR-AUC on 18 val events. Weak lipid / brand splits can move that metric without moving a χ² p-value across the FDR line. `Stent type-SES_xiencev` (xgb_b only) is the clearest encoding artefact.

The 2026-08-31 consensus names `WBC`, `1.1:1Post dilation`, `Aneurysm`, `CaI`, `LVEF`, and `UA` are **no longer in the three-way union**. `CaI` and `LVEF` remain frequently selected.

---

## 6. Domain pattern

### Figure 4. Extracted counts by clinical domain

![Figure 4](paper_figures/fig4_domain_counts.png)

**Figure 4.** Statistical FDR is concentrated in laboratory, procedural/stent, and anatomy blocks (the EDA domain screen). ML consensus is heavier on laboratory *plus* a medication, a demographic, a brand dummy, and cardiac `LV`, and thinner on anatomy binaries. Post-dilation now appears on both sides via the **`No postdilation`** flag (not the 1.1:1 complement).

**Source file:** [paper_figures/fig4_domain_counts.png](paper_figures/fig4_domain_counts.png)

Statistics therefore still “owns” **anatomy coding, WBC, and most stent-technique flags**. Machine learning “owns” **sex**, **labs that are collinear with FDR hits** (`Cre`, `HGB`, `LDL`), and **one brand dummy**. Both own **Clopidogrel, eGFR, LV, HbA1c, and No postdilation**.

---

## 7. Methodological reasons for disagreement

### Figure 3. Buckets

![Figure 3](paper_figures/fig3_reason_buckets.png)

**Figure 3.** Counts of names in this comparison assigned to a primary methodological bucket (one bucket per feature; the anatomy/stent collinear family is grouped). `WBC` has its own bucket: dropped from the ML view.

**Source file:** [paper_figures/fig3_reason_buckets.png](paper_figures/fig3_reason_buckets.png)

**Why a feature can appear in statistics and not in ML**

- Univariate tests do not penalize redundancy. FDR will list `3-vessel disease`, `Multi-vessel CAD`, `Single-vessel disease`, and `NO.of vessels` if each 2×2/t-test is small. A model only needs one of them.
- Complements are two encodings of one bit (`1.1:1Post dilation` vs `No postdilation`). χ² sees both; CatBoost kept `No postdilation` on this dump.
- Categorical χ² on `Stent type-SES` does not survive as the parent name after 9-level one-hot (`Stent type-SES_xiencev` is the three-way dummy).
- `WBC` is FDR-significant and **absent from the scaled matrix** (`anti_leakage.drop_features`).
- Hold-out PR-AUC with 18 val events is under-powered for moderate ORs (Diabetes 1.89, Previous PCI 6.49) that FDR can still detect on 92 events.

**Why a feature can appear in ML and not in statistics**

- Predictive importance is not a marginal p-value. LOCO asks whether the *rest of the model* can compensate after a refit. SHAP asks for coalition credit. FFS asks for greedy hold-out gain. None of these is a two-sample test.
- Correlated twins: the univariate test of `Cre` is null because `eGFR` already captures renal function; the model may still split on Cre.
- Interactions: `Men × eGFR` is an EDA FDR hit; univariate `Men` is not. LR consensus includes `Men`.
- Independent selectors disagree. LightGBM’s three-way set is `HbA1c` only. Some ML-only names (`Stent type-SES_xiencev`) are **algorithm artefacts**, not cohort discoveries.

**Practical reading.** Treat the 5-name intersection (`Clopidogrel`, `HbA1c`, `LV`, `No postdilation`, `eGFR`) as the robust extraction set: associated in the cohort *and* used by fitted classic models under PR-AUC. Treat statistical-only anatomy/stent names as **association findings that need a non-redundant representative** before they enter a predictor. Treat `WBC` as a **Part 1 association that this selector dump cannot recover**. Treat ML-only names as **hypothesis-generating predictive correlates** until they pass a pre-specified association screen or a **held-out / external cohort** that this pack does not contain. This comparison is a **methods result**, not a biological ranking of “true” risk markers.

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

*Statistical names: univariate FDR q < 0.05 from* `eda.ipynb` *(time-since-stent excluded from the overlap count). ML names: LOCO ∩ SHAP ∩ FFS top-20, PR-AUC only, 2026-09-19 dump of* `baseline_feature_selections.ipynb` *(seven classic models; 9-level stent encoder; TSSI+WBC dropped; fit/val 4148/1037). Figures and CSVs regenerated by* `code/modeling/tools/rebuild_part3_paper_figures.py` *(also the last cell of* `stats_vs_ml_comparison.ipynb`*).*
