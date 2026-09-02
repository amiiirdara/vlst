# TabPFN interpretability — paper figures and tables

This document gathers publication-oriented figures and tables from the TabPFN interpretability notebook `tabpfn_interpretability.ipynb`.

**Cohort / protocol.** Raw VLST.csv, n = 5,185, 81 features after dropping identifiers (`NO.`, `Name`) and `Time since stent implantation` (time-at-risk / follow-up, not a baseline covariate). Target = `Stent thrombosis`. EDA found **no missing values** — there is no missingness to “keep.” `Stent type-SES` is collapsed with the **shared 9-level encoder** (106 raw brand strings → 9 levels, min_count=30), then coded as integer categoricals with the other text columns (no scaling / one-hot). That is the TabPFN-native representation: 9 brand codes, not 106 strings and not the Part 2/4 one-hot. A PDP sweep across those integers is still not a meaningful nominal contrast, so continuous PDP drops the brand column. Feature ranking, PDP, and SHAP are **interpretation / attribution** — not prediction, not external validation, and not a locked-in feature mask for Part 4.

**This run (D4).** Kaggle Tesla T4, notebook commit `645fb0e` (Interpretability plus Version 2). Protocol met: 9-level encoder; MI CSV **all 81 scores**; SFS 10 seeds on the full cohort; PDP `balance_probabilities=False` on n = 5,185; SHAP **15 VLST=1 + 15 VLST=0** with **client thinking succeeding** (`Explaining all 30 rows`); k-SII / SHAP-IQ remain **one illustrative VLST=1 row** (cohort index **5099**).

**Methods note — selection vs explanation.** Mutual information, stability (repeated forward SFS), and PDP use the **full cohort**. SHAP explains **15 VLST=1 + 15 VLST=0** (`SHAP_N_PER_CLASS`, seed 42); the model is still fit on all 5,185 rows. Indices are stored in `interpretability_shap_explain_indices.csv`. k-SII / SHAP-IQ force and network plots remain that one VLST=1 row from the slice. Do not SHAP all 5,185 rows on the client.

**Backends.** Mutual information, stability selection, and PDP use **local** `tabpfn` (0 client thinking fits). SHAP and SHAP-IQ on this run used **tabpfn-client + thinking** (`effort=high`, `metric=average_precision`) — the client did **not** fall back to local. Ranking / SHAP / stability use `balance_probabilities=True` so a 1.8% outcome is visible on the attribution scale. **PDP only** uses `balance_probabilities=False` (empirical prior; y-axis near prevalence; **not Part 4 nested-CV risk**). PDP fit and average are on the **full cohort**, not a 70/30 test slice. The shapiq `imputer="baseline"` is **not** a missing-value fill: it replaces *hidden* features with a baseline value while attributing.

**Asset root:** [paper_figures/](paper_figures/)

---

## Contents

1. [Methods (Table 0)](#1-methods)
2. [Univariate and stability screens (Tables 1–2)](#2-univariate-and-stability-screens)
3. [Partial dependence (Figures 1–2, Table 3)](#3-partial-dependence)
4. [SHAP attributions (Figures 3–7, Table 4)](#4-shap-attributions)
5. [Pairwise interactions — k-SII (Figures 8–9)](#5-pairwise-interactions--k-sii)
6. [SHAP-IQ native plots (Figures 10–12)](#6-shap-iq-native-plots)
7. [Consensus ranking (Figure 13, Table 5)](#7-consensus-ranking)
8. [File index](#8-file-index)

---

## 1. Methods

### Table 0. Interpretability methods

![Table 0](paper_figures/paper_table0_methods.png)

**Table 0.** Five signals plus a Borda-style consensus. No single method is trusted alone. Stability frequency is the reliability signal (how often forward SFS keeps a feature across 10 resamples). MI and SFS use the **full cohort**. SHAP uses **15 VLST=1 + 15 VLST=0**. Pairwise k-SII is a one-row interaction view (row 5099, VLST=1), not a global interaction ranking.

| Method | Question | Backend | Notebook setting |
| --- | --- | --- | --- |
| mutual_info_classif | Univariate association | sklearn | 0 TabPFN calls; median fill is inert (no NaNs); all 81 scores stored |
| Stability (repeated SFS) | Selection frequency | local TabPFN | 10 resamples × top-10 forward SFS, AP scoring, full cohort |
| PDP | Average predicted probability (empirical prior) | local TabPFN | Full cohort; `balance_probabilities=False`; y-axis labeled “empirical prior / not Part 4 risk”. Ranking / SHAP stay True |
| SHAP (shapiq SV) | Local attributions | tabpfn-client + thinking | 15 VLST=1 + 15 VLST=0; fit/background = full cohort; budget=256 |
| k-SII / SHAP-IQ | Pairwise interactions | tabpfn-client + thinking | One VLST=1 row (5099) from that 15+15 slice; budget=256 |
| Consensus (Borda) | Mean of normalized ranks | aggregate | MI + stability frequency + mean(\|SHAP\|); MI not fill-zeroed |

**Source files:** [paper_figures/paper_table0_methods.png](paper_figures/paper_table0_methods.png), [paper_figures/paper_table0_methods.csv](paper_figures/paper_table0_methods.csv)

---

## 2. Univariate and stability screens

### Table 1. Top 15 by mutual information

![Table 1](paper_figures/paper_table1_mutual_info.png)

**Table 1.** `mutual_info_classif` ranking of the **81-column** raw matrix on the **full cohort**. The code applies a column-median fill before MI; the CSV has no missing values, so that fill does nothing. All 81 scores are stored (`paper_table1_mutual_info.csv` / `interpretability_mutual_info_ranking.csv`). Calcium index (`CaI`), `WBC`, and `LV` lead, then `eGFR` and `1.1:1Post dilation`. `Fast-Glu` is 10th (0.0044) and `ZES` is 13th (0.0038) — those cells are measured, not blanks. `Cre` is 25th (**0.0023**), not a fill-zero. `Stent type-SES` is 32nd (0.0016) after the 9-level encoder, not a top-5 MI name. This is a marginal association screen, not a model attribution.

| Rank | Feature | Mutual information |
| ---: | --- | ---: |
| 1 | CaI | 0.019224 |
| 2 | WBC | 0.018360 |
| 3 | LV | 0.015178 |
| 4 | eGFR | 0.009931 |
| 5 | 1.1:1Post dilation | 0.007932 |
| 6 | LDL | 0.007693 |
| 7 | No postdilation | 0.007535 |
| 8 | HbA1c | 0.007439 |
| 9 | HGB | 0.004606 |
| 10 | Fast-Glu | 0.004429 |
| 11 | TCL | 0.004257 |
| 12 | Fiberinogen | 0.004040 |
| 13 | ZES | 0.003783 |
| 14 | Visual thrombus | 0.003707 |
| 15 | CKD5 | 0.003145 |

**Source files:** [paper_figures/paper_table1_mutual_info.png](paper_figures/paper_table1_mutual_info.png), [paper_figures/paper_table1_mutual_info.csv](paper_figures/paper_table1_mutual_info.csv)

### Table 2. Stability selection frequency

![Table 2](paper_figures/paper_table2_stability.png)

**Table 2.** Forward sequential feature selection (keep 10 of 81, 5-fold CV, average precision) repeated over 10 shuffled seeds on the **full cohort**. The only 10/10 feature is **`WBC`**. Selected in 7/10: `Staged PCI`. Selected in 6/10: `Fiberinogen`, `LV`, `ZES`. Selected in 5/10: `Cre`, `PES`, `eGFR`. **`Stent type-SES` is not a 10/10 feature on this run** (it does not appear in the selected-feature list). A long tail of names appears only once and should not be treated as robust TabPFN features.

| Feature | Selected | Frequency |
| --- | --- | ---: |
| WBC | 10/10 | 1.0 |
| Staged PCI | 7/10 | 0.7 |
| Fiberinogen | 6/10 | 0.6 |
| LV | 6/10 | 0.6 |
| ZES | 6/10 | 0.6 |
| Cre | 5/10 | 0.5 |
| PES | 5/10 | 0.5 |
| eGFR | 5/10 | 0.5 |
| Age | 4/10 | 0.4 |
| STEMI | 4/10 | 0.4 |
| 1.1:1Post dilation | 3/10 | 0.3 |
| Dissection | 3/10 | 0.3 |
| HbA1c | 3/10 | 0.3 |
| Ticagrelor | 3/10 | 0.3 |
| Aneurysm | 2/10 | 0.2 |
| CaI | 2/10 | 0.2 |
| LVEF | 2/10 | 0.2 |
| No postdilation | 2/10 | 0.2 |
| No.of stents per lesion | 2/10 | 0.2 |

**Source files:** [paper_figures/paper_table2_stability.png](paper_figures/paper_table2_stability.png), [paper_figures/paper_table2_stability.csv](paper_figures/paper_table2_stability.csv), [paper_figures/interpretability_feature_stability_summary.csv](paper_figures/interpretability_feature_stability_summary.csv)

---

## 3. Partial dependence

PDP candidates were taken from the stability / MI screens. Continuous PDP uses grid resolution 30. Binary PDP forces each flag to 0 vs 1 and reports the change in average predicted P[Stent thrombosis]. Fit and average are on the **full cohort**.

**Methods note — PDP is empirical prior, not Part 4 risk.** PDP uses `balance_probabilities=False`. Average predicted probabilities sit near prevalence 0.0177 (binary P(y=1) ≈ 0.017–0.023). Do **not** quote 0.13–0.26 or “toward ~0.6” as clinical risk — those were the old balanced-prior / test-slice export. Ranking / SHAP / stability still use `balance_probabilities=True` on a separate fit. Neither scale is the Part 4 nested-CV client.

### Figure 1. Continuous partial dependence

![Figure 1](paper_figures/paper_fig1_pdp_continuous.png)

**Figure 1.** Continuous PDP on the **empirical-prior** scale, full cohort (n = 5,185), y-axis labeled **empirical prior / not Part 4 risk**, dashed prevalence line. Nominal `Stent type-SES` is dropped from continuous curves (integer brand codes are not a meaningful grid). Shapes are average predicted probability under local TabPFN, not Part 4 nested-CV risk and not a treatment effect.

**Source file:** [paper_figures/paper_fig1_pdp_continuous.png](paper_figures/paper_fig1_pdp_continuous.png)

### Figure 2. Binary partial dependence

![Figure 2](paper_figures/paper_fig2_pdp_binary.png)

**Figure 2.** Binary flags forced to 0 vs 1 and averaged over the **full cohort** on the same empirical-prior axis as Figure 1. Largest Δ is `1.1:1Post dilation` (0.0234 → 0.0191). Do not mix this axis with ranking/SHAP (`balance_probabilities=True`). A negative Δ is a lower modelled probability of recorded VLST, not a treatment benefit (confounding by indication).

**Source file:** [paper_figures/paper_fig2_pdp_binary.png](paper_figures/paper_fig2_pdp_binary.png)

### Table 3. Binary PDP numeric values

![Table 3](paper_figures/paper_table3_pdp_binary.png)

**Table 3.** Empirical-prior binary PDP on n = 5,185 (prevalence 0.0177). These values are **not** clinical risk and **not** Part 4 nested-CV probabilities.

| Feature | P(y=1 \| 0) | P(y=1 \| 1) | ΔP |
| --- | ---: | ---: | ---: |
| Staged PCI | 0.0184 | 0.0169 | −0.0015 |
| ZES | 0.0184 | 0.0174 | −0.0010 |
| PES | 0.0170 | 0.0183 | +0.0013 |
| STEMI | 0.0189 | 0.0172 | −0.0017 |
| 1.1:1Post dilation | 0.0234 | 0.0191 | −0.0043 |
| Dissection | 0.0183 | 0.0165 | −0.0018 |

**Source files:** [paper_figures/paper_table3_pdp_binary.png](paper_figures/paper_table3_pdp_binary.png), [paper_figures/paper_table3_pdp_binary.csv](paper_figures/paper_table3_pdp_binary.csv)

---

## 4. SHAP attributions

Fit on the full cohort; explain **15 VLST=1 + 15 VLST=0** (client thinking-high succeeded). Mean(|SHAP|) is that 30-row slice, **not** global SHAP on 5,185 rows. The attribution scale uses `balance_probabilities=True` (stretched 1.8% prior). Waterfall E[f(x)] ≈ 0.90 is that scale, not the PDP empirical prior (~0.018).

### Figure 3. SHAP summary

![Figure 3](paper_figures/paper_fig3_shap_summary.png)

**Figure 3.** SHAP summary / beeswarm for the 15+15 slice (colour = feature value). Client thinking-high; 30 rows explained.

**Source file:** [paper_figures/paper_fig3_shap_summary.png](paper_figures/paper_fig3_shap_summary.png)

### Figure 4. SHAP scatter for Age

![Figure 4](paper_figures/paper_fig4_shap_scatter_age.png)

**Figure 4.** Age versus SHAP on the same 15+15 slice. A local scatter, not a cohort dose–response.

**Source file:** [paper_figures/paper_fig4_shap_scatter_age.png](paper_figures/paper_fig4_shap_scatter_age.png)

### Figure 5. Mean absolute SHAP (global bar)

![Figure 5](paper_figures/paper_fig5_shap_bar.png)

**Figure 5.** Mean(|SHAP|) on the 30-row slice. **`Cre` leads** (0.158), then `eGFR` (0.077), `WBC` (0.061), `LV` (0.052). This ranking is not the old 15-case-only list (`LV` 1.24 / `WBC` 1.16).

**Source file:** [paper_figures/paper_fig5_shap_bar.png](paper_figures/paper_fig5_shap_bar.png)

### Figure 6. Compact SHAP beeswarm

![Figure 6](paper_figures/paper_fig6_shap_beeswarm.png)

**Figure 6.** Compact beeswarm of the same 15+15 attributions.

**Source file:** [paper_figures/paper_fig6_shap_beeswarm.png](paper_figures/paper_fig6_shap_beeswarm.png)

### Figure 7. One-row SHAP waterfall

![Figure 7](paper_figures/paper_fig7_shap_waterfall.png)

**Figure 7.** Waterfall for the first VLST=1 patient in the slice (**row 5099**). Baseline E[f(x)] ≈ **0.903** → f(x) ≈ **1.00** on the **balanced-prior SHAP scale**. `LV` and `WBC` raise the output; `Cre` lowers it on this row. Local explanation for one patient, not a global ranking, and not the PDP 0.018 axis.

**Source file:** [paper_figures/paper_fig7_shap_waterfall.png](paper_figures/paper_fig7_shap_waterfall.png)

### Table 4. Mean(|SHAP|) ranking

![Table 4](paper_figures/paper_table4_shap_mean_abs.png)

**Table 4.** Mean absolute SHAP on **15 VLST=1 + 15 VLST=0**. Used as the SHAP column of Table 5.

| Rank | Feature | mean(\|SHAP\|) |
| ---: | --- | ---: |
| 1 | Cre | 0.1585 |
| 2 | eGFR | 0.0771 |
| 3 | WBC | 0.0609 |
| 4 | LV | 0.0525 |
| 5 | No.of stents per lesion | 0.0477 |
| 6 | Men | 0.0455 |
| 7 | Aspirin | 0.0388 |
| 8 | CKD5 | 0.0313 |
| 9 | DAPT | 0.0273 |
| 10 | LDL | 0.0253 |
| 11 | 1.1:1Post dilation | 0.0172 |
| 12 | No postdilation | 0.0139 |
| 13 | CaI | 0.0135 |
| 14 | TCL | 0.0111 |
| 15 | HbA1c | 0.0091 |

**Source files:** [paper_figures/paper_table4_shap_mean_abs.png](paper_figures/paper_table4_shap_mean_abs.png), [paper_figures/paper_table4_shap_mean_abs.csv](paper_figures/paper_table4_shap_mean_abs.csv), [paper_figures/interpretability_shap_explain_indices.csv](paper_figures/interpretability_shap_explain_indices.csv)

---

## 5. Pairwise interactions — k-SII

k-SII plots use **one illustrative VLST=1 row** from the 15+15 SHAP slice (row **5099**, budget = 256). Node size is the main effect; edge width is the pairwise interaction. They illustrate how TabPFN combines features for that row; they are **not** a cohort interaction screen. The notebook print lists the top-20 |SV| names for this row as `LV`, `WBC`, `CKD5`, `eGFR`, `Men`, `No postdilation`, `Cre`, … — not a statement about the 5,185-row cohort.

### Figure 8. k-SII network (SHAP section)

![Figure 8](paper_figures/paper_fig8_ksii_network.png)

**Figure 8.** Circular k-SII network for the top features by |Shapley value| on **one VLST=1 row**. Thick edges are pairwise terms **for that patient**. Do not treat them as cohort interactions.

**Source file:** [paper_figures/paper_fig8_ksii_network.png](paper_figures/paper_fig8_ksii_network.png)

### Figure 9. k-SII UpSet plot (SHAP section)

![Figure 9](paper_figures/paper_fig9_ksii_upset.png)

**Figure 9.** UpSet-style listing of the largest main effects and pairwise k-SII values for the same row. Some panels show a large intercept / base term near 0.90 on the balanced-prior scale; that is the SHAP baseline for this explainer, not cohort prevalence.

**Source file:** [paper_figures/paper_fig9_ksii_upset.png](paper_figures/paper_fig9_ksii_upset.png)

---

## 6. SHAP-IQ native plots

Section [4/5] of the notebook recomputes imputation-based Shapley values and k-SII with shapiq’s native plotting API, again on **tabpfn-client + thinking**. Figures 10–12 are a second view of the **same one-row explanation** (row 5099), not an independent replication on new rows.

### Figure 10. SHAP-IQ force plot (one row)

![Figure 10](paper_figures/paper_fig10_shapiq_force.png)

**Figure 10.** Force / additive layout for row 5099. Read it as the compact counterpart of the waterfall in Figure 7, on the same balanced-prior attribution scale.

**Source file:** [paper_figures/paper_fig10_shapiq_force.png](paper_figures/paper_fig10_shapiq_force.png)

### Figure 11. SHAP-IQ k-SII network

![Figure 11](paper_figures/paper_fig11_shapiq_network.png)

**Figure 11.** Native shapiq network for the same one-row k-SII. Layout is a restyle of Figure 8, not a new sample of patients. Printed top-20 |SV| names on this pass include `LV`, `WBC`, `eGFR`, `Men`, `Cre`, `CKD5`, …

**Source file:** [paper_figures/paper_fig11_shapiq_network.png](paper_figures/paper_fig11_shapiq_network.png)

### Figure 12. SHAP-IQ k-SII UpSet plot

![Figure 12](paper_figures/paper_fig12_shapiq_upset.png)

**Figure 12.** Native shapiq UpSet plot of top main effects and pairwise interactions for the same row. Read it as a restyle of Figure 9, not as a new sample of patients.

**Source file:** [paper_figures/paper_fig12_shapiq_upset.png](paper_figures/paper_fig12_shapiq_upset.png)

---

## 7. Consensus ranking

Ranking uses a **Borda-style mean of normalized ranks** across mutual information, stability frequency, and mean(|SHAP|), with `n_methods` (out of 3) as a consensus count. The notebook reports the top 15 as *associations* with stent thrombosis under TabPFN — exploratory, not causal, on a ~2% prevalence cohort. MI values come from the full 81-row ranking (no fill-zero for names outside a truncated top-15).

### Figure 13. Top 15 by consensus

![Figure 13](paper_figures/paper_fig13_consensus_ranking.png)

**Figure 13.** Aggregated importance (1 = strongest mean normalized rank). Annotations give how many of the three signals placed the feature in their top set. **`WBC`, `LV`, and `eGFR` are 3/3.** `Stent type-SES` is **not** a 3/3 name on this run. `LVEF` and `STEMI` rank 11–12 on Borda with **0/3** top-set membership — middling ranks on all three lists can still enter the top 15.

**Source file:** [paper_figures/paper_fig13_consensus_ranking.png](paper_figures/paper_fig13_consensus_ranking.png)

### Table 5. Consensus feature report

![Table 5](paper_figures/paper_table5_consensus.png)

**Table 5.** The notebook’s `interpretability_feature_importance_report` top 15. `importance_score` is the Borda aggregate. `n_methods` counts how many of {MI top, stability, SHAP top} contributed. The three names with n_methods = 3 (`WBC`, `LV`, `eGFR`) are the most consistent TabPFN associations in this run. `Cre` has measured MI **0.0023** (25th of 81) and leads mean(|SHAP|); it is 2/3, not a fill-zero. `CaI` is first on MI but only 2/10 in stability. `ZES` is 6/10 stable and in the MI top 15 but not the SHAP top (2/3).

| Rank | Feature | Score | n methods | Stability | mean(\|SHAP\|) | MI | In MI top | In SHAP top |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | WBC | 0.9875 | 3/3 | 1.0 | 0.0609 | 0.018360 | yes | yes |
| 2 | LV | 0.9667 | 3/3 | 0.6 | 0.0525 | 0.015178 | yes | yes |
| 3 | eGFR | 0.9583 | 3/3 | 0.5 | 0.0771 | 0.009931 | yes | yes |
| 4 | 1.1:1Post dilation | 0.8937 | 2/3 | 0.3 | 0.0172 | 0.007932 | yes | yes |
| 5 | CaI | 0.8833 | 2/3 | 0.2 | 0.0135 | 0.019224 | yes | yes |
| 6 | Cre | 0.8750 | 2/3 | 0.5 | 0.1585 | 0.002281 | no | yes |
| 7 | HbA1c | 0.8646 | 2/3 | 0.3 | 0.0091 | 0.007439 | yes | yes |
| 8 | No postdilation | 0.8625 | 2/3 | 0.2 | 0.0139 | 0.007535 | yes | yes |
| 9 | LDL | 0.8229 | 2/3 | 0.1 | 0.0253 | 0.007693 | yes | yes |
| 10 | Fiberinogen | 0.8000 | 2/3 | 0.6 | 0.0040 | 0.004040 | yes | no |
| 11 | LVEF | 0.7417 | 0/3 | 0.2 | 0.0047 | 0.002604 | no | no |
| 12 | STEMI | 0.7313 | 0/3 | 0.4 | 0.0086 | 0.001086 | no | no |
| 13 | HGB | 0.7229 | 1/3 | 0.1 | 0.0045 | 0.004606 | yes | no |
| 14 | Fast-Glu | 0.7104 | 1/3 | 0.1 | 0.0043 | 0.004429 | yes | no |
| 15 | ZES | 0.6875 | 2/3 | 0.6 | 0.0019 | 0.003783 | yes | no |

**Source files:** [paper_figures/paper_table5_consensus.png](paper_figures/paper_table5_consensus.png), [paper_figures/paper_table5_consensus.csv](paper_figures/paper_table5_consensus.csv)

---

## 8. File index

| ID | Type | File |
| --- | --- | --- |
| Table 0 | Table | [paper_table0_methods.png](paper_figures/paper_table0_methods.png) |
| Table 1 | Table | [paper_table1_mutual_info.png](paper_figures/paper_table1_mutual_info.png) |
| Table 2 | Table | [paper_table2_stability.png](paper_figures/paper_table2_stability.png) |
| Fig 1 | Figure | [paper_fig1_pdp_continuous.png](paper_figures/paper_fig1_pdp_continuous.png) |
| Fig 2 | Figure | [paper_fig2_pdp_binary.png](paper_figures/paper_fig2_pdp_binary.png) |
| Table 3 | Table | [paper_table3_pdp_binary.png](paper_figures/paper_table3_pdp_binary.png) |
| Fig 3 | Figure | [paper_fig3_shap_summary.png](paper_figures/paper_fig3_shap_summary.png) |
| Fig 4 | Figure | [paper_fig4_shap_scatter_age.png](paper_figures/paper_fig4_shap_scatter_age.png) |
| Fig 5 | Figure | [paper_fig5_shap_bar.png](paper_figures/paper_fig5_shap_bar.png) |
| Fig 6 | Figure | [paper_fig6_shap_beeswarm.png](paper_figures/paper_fig6_shap_beeswarm.png) |
| Fig 7 | Figure | [paper_fig7_shap_waterfall.png](paper_figures/paper_fig7_shap_waterfall.png) |
| Table 4 | Table | [paper_table4_shap_mean_abs.png](paper_figures/paper_table4_shap_mean_abs.png) |
| Fig 8 | Figure | [paper_fig8_ksii_network.png](paper_figures/paper_fig8_ksii_network.png) |
| Fig 9 | Figure | [paper_fig9_ksii_upset.png](paper_figures/paper_fig9_ksii_upset.png) |
| Fig 10 | Figure | [paper_fig10_shapiq_force.png](paper_figures/paper_fig10_shapiq_force.png) |
| Fig 11 | Figure | [paper_fig11_shapiq_network.png](paper_figures/paper_fig11_shapiq_network.png) |
| Fig 12 | Figure | [paper_fig12_shapiq_upset.png](paper_figures/paper_fig12_shapiq_upset.png) |
| Fig 13 | Figure | [paper_fig13_consensus_ranking.png](paper_figures/paper_fig13_consensus_ranking.png) |
| Table 5 | Table | [paper_table5_consensus.png](paper_figures/paper_table5_consensus.png) |

---

*Figures are the executed PNG outputs from `tabpfn_interpretability.ipynb` (`645fb0e`). Tables are rebuilt from the Kaggle CSVs. SHAP / SHAP-IQ used tabpfn-client thinking (no local fallback). MI, stability, and PDP use the full cohort; SHAP explains 15 VLST=1 + 15 VLST=0; k-SII is one VLST=1 row (5099). Rankings are for interpretation only and should not be reused as a leakage-free feature mask.*
