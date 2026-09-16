# TabPFN interpretability — paper figures and tables

This document gathers publication-oriented figures and tables from the TabPFN interpretability notebook `tabpfn_interpretability.ipynb`.

**Cohort / protocol.** Raw VLST.csv, n = 5,185, 81 features after dropping identifiers (`NO.`, `Name`) and `Time since stent implantation` (time-at-risk / follow-up, not a baseline covariate). Target = `Stent thrombosis`. EDA found **no missing values** — there is no missingness to “keep.” `Stent type-SES` is collapsed with the **shared 9-level encoder** (106 raw brand strings → 9 levels, min_count=30), then coded as integer categoricals with the other text columns (no scaling / one-hot). That is the TabPFN-native representation: 9 brand codes, not 106 strings and not the Part 2/4 one-hot. A PDP sweep across those integers is still not a meaningful nominal contrast, so continuous PDP drops the brand column. Feature ranking, PDP, and SHAP are **interpretation / attribution** — not prediction, not external validation, and not a locked-in feature mask for Part 4.

**This run (D4).** Kaggle Tesla T4, notebook commit `e356bb1` (Interpretability Version 5, papermill 2026-09-08–09). Protocol: stratified `train_test_split(test_size=0.3, random_state=42)` → train **3,629 / 64 events**, held-out **1,556 / 28 events**. MI, stability SFS, and PDP **fit/average on train**. `SHAP_EXPLAIN_HELDOUT=True`: SHAP explains **all 1,556 held-out rows**; fit/background = train. k-SII / waterfall / SHAP-IQ = first held-out VLST=1 (held-out position 20, **cohort row 5176**). `FS_THINKING_MODE=False`; `PDP_USE_CLIENT=False`; `INTERP_THINKING_MODE=True` (effort high, metric average_precision). Client SHAP started then hit HTTP **429** (~row 550/1556); **[3/5] and [4/5] finished on local `tabpfn` + KV cache**. Local constructors omit `balance_probabilities`.

**Methods note — selection vs explanation.** Mutual information, stability (repeated forward SFS), and PDP use the **train split**. SHAP explains **every held-out row** (28 events / 1,528 non-events). k-SII / SHAP-IQ remain **one** held-out VLST=1 patient (row **5176**). Do not describe this run as full-cohort MI/SFS/PDP, as 15+15 SHAP, or as SHAP-all 5,185.

**Backends.** Mutual information, stability, and PDP use **local** `tabpfn` (0 client thinking fits). SHAP and SHAP-IQ **tried** tabpfn-client + thinking, then **fell back to local** after HTTP 429. PDP uses the empirical prior (`PDP_USE_CLIENT=False`; y-axis near prevalence; **not Part 4 nested-CV risk**). The shapiq `imputer="baseline"` is **not** a missing-value fill: it replaces *hidden* features with a baseline value while attributing.

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

**Table 0.** Five signals plus a Borda-style consensus. No single method is trusted alone. Stability frequency is the reliability signal (how often forward SFS keeps a feature across 10 **train** resamples). MI and SFS use **train**. SHAP uses **all 1,556 held-out rows** (local after 429). Pairwise k-SII is a one-row interaction view (row **5176**, VLST=1), not a global interaction ranking.

| Method | Question | Backend | Notebook setting |
| --- | --- | --- | --- |
| mutual_info_classif | Univariate association | sklearn | 0 TabPFN calls; **train** n=3629; all 81 scores written on Kaggle |
| Stability (repeated SFS) | Selection frequency | local TabPFN | 10 resamples × top-10 forward SFS, AP scoring, **train** |
| PDP | Average predicted probability (empirical prior) | local TabPFN | **Train** n=3629; `PDP_USE_CLIENT=False`; y-axis “empirical prior / not Part 4 risk” |
| SHAP (shapiq SV) | Local attributions | client 429 → local KV cache | All 1,556 held-out rows; fit/background = train; budget=256 |
| k-SII / SHAP-IQ | Pairwise interactions | client 429 → local KV cache | One held-out VLST=1 (pos 20, cohort **5176**); budget=256 |
| Consensus (Borda) | Mean of normalized ranks | aggregate | Train MI + train stability + held-out mean(\|SHAP\|) |

**Source files:** [paper_figures/paper_table0_methods.png](paper_figures/paper_table0_methods.png), [paper_figures/paper_table0_methods.csv](paper_figures/paper_table0_methods.csv)

---

## 2. Univariate and stability screens

### Table 1. Top 15 by mutual information

![Table 1](paper_figures/paper_table1_mutual_info.png)

**Table 1.** `mutual_info_classif` ranking of the **81-column** matrix on the **train split** (n=3629). Top 15 from `interpretability_mutual_info_ranking.csv`. `Fast-Glu` / `ZES` are **not** in this train top 15. `Cre` is 51st of 81 with train MI **0.000000**. This is a marginal association screen, not a model attribution.

| Rank | Feature | Mutual information |
| ---: | --- | --- |
| 1 | CaI | 0.022005 |
| 2 | WBC | 0.020165 |
| 3 | LV | 0.012768 |
| 4 | LDL | 0.009893 |
| 5 | eGFR | 0.009830 |
| 6 | HDL | 0.005353 |
| 7 | HGB | 0.005327 |
| 8 | Lesion location-Ostial | 0.005128 |
| 9 | HbA1c | 0.004805 |
| 10 | Clopidogrel | 0.004669 |
| 11 | Stent type-SES | 0.004587 |
| 12 | No.of stents per lesion | 0.004371 |
| 13 | Multi-vessel CAD | 0.003733 |
| 14 | NSTEMI | 0.003606 |
| 15 | Fiberinogen | 0.003007 |

**Source files:** [paper_figures/paper_table1_mutual_info.png](paper_figures/paper_table1_mutual_info.png), [paper_figures/paper_table1_mutual_info.csv](paper_figures/paper_table1_mutual_info.csv), [paper_figures/interpretability_mutual_info_ranking.csv](paper_figures/interpretability_mutual_info_ranking.csv)

### Table 2. Stability selection frequency

![Table 2](paper_figures/paper_table2_stability.png)

**Table 2.** Forward sequential feature selection (keep 10 of 81, 5-fold CV, average precision) repeated over 10 shuffled seeds on the **train split**. The only 10/10 feature is **`WBC`**. Selected in 8/10: `Cre`, `LV`. Selected in 7/10: `eGFR`. Selected in 5/10: `Previous PCI`, `Staged PCI`. **`Stent type-SES` is not a high-frequency feature on this run.**

| Feature | Selected | Frequency |
| --- | --- | ---: |
| WBC | 10/10 | 1.0 |
| Cre | 8/10 | 0.8 |
| LV | 8/10 | 0.8 |
| eGFR | 7/10 | 0.7 |
| Previous PCI | 5/10 | 0.5 |
| Staged PCI | 5/10 | 0.5 |
| Fiberinogen | 4/10 | 0.4 |
| LVEF | 4/10 | 0.4 |
| UA | 4/10 | 0.4 |
| 1.1:1Post dilation | 3/10 | 0.3 |
| Age | 3/10 | 0.3 |
| HbA1c | 3/10 | 0.3 |
| ZES | 3/10 | 0.3 |

**Source files:** [paper_figures/paper_table2_stability.png](paper_figures/paper_table2_stability.png), [paper_figures/paper_table2_stability.csv](paper_figures/paper_table2_stability.csv), [paper_figures/interpretability_feature_stability_summary.csv](paper_figures/interpretability_feature_stability_summary.csv)

---

## 3. Partial dependence

PDP candidates were taken from the stability / MI screens. Continuous PDP uses grid resolution 30. Binary PDP forces each flag to 0 vs 1 and reports the change in average predicted P[Stent thrombosis]. Fit and average are on the **train split** (n=3629, events=64).

**Methods note — PDP is empirical prior, not Part 4 risk.** PDP uses local TabPFN with empirical class prior (`PDP_USE_CLIENT=False`). Average predicted probabilities sit near prevalence (~0.018). Do **not** quote 0.13–0.26 or “toward ~0.6” as clinical risk. Neither scale is the Part 4 nested-CV client.

### Figure 1. Continuous partial dependence

![Figure 1](paper_figures/paper_fig1_pdp_continuous.png)

**Figure 1.** Continuous PDP on the **empirical-prior** scale, **train** (n = 3,629), features `WBC`, `Cre`, `LV`, `eGFR`. Nominal `Stent type-SES` is dropped from continuous curves. Dashed line = cohort prevalence. Not Part 4 nested-CV risk and not a treatment effect.

**Source file:** [paper_figures/paper_fig1_pdp_continuous.png](paper_figures/paper_fig1_pdp_continuous.png)

### Figure 2. Binary partial dependence

![Figure 2](paper_figures/paper_fig2_pdp_binary.png)

**Figure 2.** Binary flags forced to 0 vs 1 and averaged over **train**. Largest |Δ| is `1.1:1Post dilation` (0.0258 → 0.0165, Δ **−0.0093**). Largest positive Δ is `Previous PCI` (+0.0137). A negative Δ is a lower modelled probability of recorded VLST, not a treatment benefit (confounding by indication).

**Source file:** [paper_figures/paper_fig2_pdp_binary.png](paper_figures/paper_fig2_pdp_binary.png)

### Table 3. Binary PDP numeric values

![Table 3](paper_figures/paper_table3_pdp_binary.png)

**Table 3.** Empirical-prior binary PDP on **train** n=3629. These values are **not** clinical risk and **not** Part 4 nested-CV probabilities.

| Feature | P(y=1 \| 0) | P(y=1 \| 1) | ΔP |
| --- | ---: | ---: | ---: |
| Previous PCI | 0.0175 | 0.0312 | +0.0137 |
| UA | 0.0177 | 0.0205 | +0.0028 |
| Staged PCI | 0.0184 | 0.0172 | −0.0011 |
| ZES | 0.0184 | 0.0170 | −0.0014 |
| Cardiogenic shock | 0.0184 | 0.0158 | −0.0026 |
| 1.1:1Post dilation | 0.0258 | 0.0165 | −0.0093 |

**Source files:** [paper_figures/paper_table3_pdp_binary.png](paper_figures/paper_table3_pdp_binary.png), [paper_figures/paper_table3_pdp_binary.csv](paper_figures/paper_table3_pdp_binary.csv)

---

## 4. SHAP attributions

Fit on **train**; explain **all 1,556 held-out rows** (client thinking started, HTTP 429 at ~row 550, finished **local**). Mean(|SHAP|) below is the top 15 of the **81-column** ranking (`interpretability_shap_mean_abs.csv`). Indices: `interpretability_shap_explain_indices.csv` (1,556 rows; 28 events). Do not call this 15+15 or global SHAP on 5,185.

### Figure 3. SHAP summary

![Figure 3](paper_figures/paper_fig3_shap_summary.png)

**Figure 3.** SHAP summary / beeswarm for the **1,556-row held-out** slice (local after 429).

**Source file:** [paper_figures/paper_fig3_shap_summary.png](paper_figures/paper_fig3_shap_summary.png)

### Figure 4. SHAP scatter for Age

![Figure 4](paper_figures/paper_fig4_shap_scatter_age.png)

**Figure 4.** Age versus SHAP on the held-out slice. A local scatter, not a cohort dose–response.

**Source file:** [paper_figures/paper_fig4_shap_scatter_age.png](paper_figures/paper_fig4_shap_scatter_age.png)

### Figure 5. Mean absolute SHAP (bar)

![Figure 5](paper_figures/paper_fig5_shap_bar.png)

**Figure 5.** Mean(|SHAP|) on the 1,556-row held-out slice. Scale is ~1.0 for leading names (`eGFR` 1.04, `WBC` 1.02, `LV` 0.87 in the consensus print) — **not** the old 30-row Cre-leading 0.158 ranking.

**Source file:** [paper_figures/paper_fig5_shap_bar.png](paper_figures/paper_fig5_shap_bar.png)

### Figure 6. Compact SHAP beeswarm

![Figure 6](paper_figures/paper_fig6_shap_beeswarm.png)

**Figure 6.** Compact beeswarm of the same held-out attributions.

**Source file:** [paper_figures/paper_fig6_shap_beeswarm.png](paper_figures/paper_fig6_shap_beeswarm.png)

### Figure 7. One-row SHAP waterfall

![Figure 7](paper_figures/paper_fig7_shap_waterfall.png)

**Figure 7.** Waterfall for the first held-out VLST=1 patient (held-out pos 20, **cohort row 5176**). Local explanation for one patient, not a global ranking, and not the PDP ~0.018 axis.

**Source file:** [paper_figures/paper_fig7_shap_waterfall.png](paper_figures/paper_fig7_shap_waterfall.png)

### Table 4. Mean(|SHAP|) ranking (held-out, top 15 of 81)

![Table 4](paper_figures/paper_table4_shap_mean_abs.png)

**Table 4.** mean(|SHAP|) top 15 of all 81 columns on **1,556 held-out rows**. `CKD5`, `Stent type-SES`, and `Men` enter this SHAP top 15 but are not 3/3 consensus names.

| Rank | Feature | mean(\|SHAP\|) |
| ---: | --- | ---: |
| 1 | eGFR | 1.0439 |
| 2 | WBC | 1.0202 |
| 3 | LV | 0.8695 |
| 4 | 1.1:1Post dilation | 0.6906 |
| 5 | LDL | 0.4973 |
| 6 | No postdilation | 0.2958 |
| 7 | Cre | 0.2449 |
| 8 | CKD5 | 0.1872 |
| 9 | HbA1c | 0.1783 |
| 10 | Previous PCI | 0.1356 |
| 11 | Stent type-SES | 0.1125 |
| 12 | CaI | 0.0793 |
| 13 | HGB | 0.0664 |
| 14 | Men | 0.0553 |
| 15 | Fiberinogen | 0.0502 |

**Source files:** [paper_figures/paper_table4_shap_mean_abs.png](paper_figures/paper_table4_shap_mean_abs.png), [paper_figures/paper_table4_shap_mean_abs.csv](paper_figures/paper_table4_shap_mean_abs.csv), [paper_figures/interpretability_shap_mean_abs.csv](paper_figures/interpretability_shap_mean_abs.csv), [paper_figures/interpretability_shap_explain_indices.csv](paper_figures/interpretability_shap_explain_indices.csv)

---

## 5. Pairwise interactions — k-SII

k-SII plots use **one illustrative held-out VLST=1 row** (held-out pos 20, **cohort row 5176**, budget = 256). Node size is the main effect; edge width is the pairwise interaction. They illustrate how TabPFN combines features for that row; they are **not** a cohort interaction screen. The [3/5] print lists the top-20 |SV| names for this row as `LV`, `WBC`, `1.1:1Post dilation`, `LDL`, `eGFR`, `Stent type-SES`, `No postdilation`, … — not a statement about the 5,185-row cohort.

### Figure 8. k-SII network (SHAP section)

![Figure 8](paper_figures/paper_fig8_ksii_network.png)

**Figure 8.** Circular k-SII network for the top features by |Shapley value| on **one held-out VLST=1 row** (cohort **5176**). Thick edges are pairwise terms **for that patient**. Do not treat them as cohort interactions.

**Source file:** [paper_figures/paper_fig8_ksii_network.png](paper_figures/paper_fig8_ksii_network.png)

### Figure 9. k-SII UpSet plot (SHAP section)

![Figure 9](paper_figures/paper_fig9_ksii_upset.png)

**Figure 9.** UpSet-style listing of the largest main effects and pairwise k-SII values for the same row. The intercept / base term is the explainer baseline for this one-row plot, not cohort prevalence.

**Source file:** [paper_figures/paper_fig9_ksii_upset.png](paper_figures/paper_fig9_ksii_upset.png)

---

## 6. SHAP-IQ native plots

Section [4/5] of the notebook recomputes imputation-based Shapley values and k-SII with shapiq’s native plotting API. Client thinking **failed immediately** (HTTP 429); the plots finished on **local `tabpfn` + KV cache**. Figures 10–12 are a second view of the **same one-row explanation** (cohort row **5176**), not an independent replication on new rows.

### Figure 10. SHAP-IQ force plot (one row)

![Figure 10](paper_figures/paper_fig10_shapiq_force.png)

**Figure 10.** Force / additive layout for held-out pos 20 / cohort row **5176**. Read it as the compact counterpart of the waterfall in Figure 7.

**Source file:** [paper_figures/paper_fig10_shapiq_force.png](paper_figures/paper_fig10_shapiq_force.png)

### Figure 11. SHAP-IQ k-SII network

![Figure 11](paper_figures/paper_fig11_shapiq_network.png)

**Figure 11.** Native shapiq network for the same one-row k-SII. Layout is a restyle of Figure 8, not a new sample of patients. Printed top-20 |SV| names on this pass include `LV`, `WBC`, `1.1:1Post dilation`, `eGFR`, `LDL`, `No postdilation`, `Previous PCI`, …

**Source file:** [paper_figures/paper_fig11_shapiq_network.png](paper_figures/paper_fig11_shapiq_network.png)

### Figure 12. SHAP-IQ k-SII UpSet plot

![Figure 12](paper_figures/paper_fig12_shapiq_upset.png)

**Figure 12.** Native shapiq UpSet plot of top main effects and pairwise interactions for the same row. Read it as a restyle of Figure 9, not as a new sample of patients.

**Source file:** [paper_figures/paper_fig12_shapiq_upset.png](paper_figures/paper_fig12_shapiq_upset.png)

---

## 7. Consensus ranking

Ranking uses a **Borda-style mean of normalized ranks** across **train** mutual information, **train** stability frequency, and **held-out** mean(|SHAP|), with `n_methods` (out of 3) as a consensus count. The notebook reports the top 15 as *associations* with stent thrombosis under TabPFN — exploratory, not causal, on a ~2% prevalence cohort. MI values in Table 5 come from the consensus print (the full 81-row MI CSV stayed on Kaggle). `Cre` prints **0.000000** on train — a measured train-split zero, not a truncated-top-15 fill-zero.

### Figure 13. Top 15 by consensus

![Figure 13](paper_figures/paper_fig13_consensus_ranking.png)

**Figure 13.** Aggregated importance (1 = strongest mean normalized rank). Annotations give how many of the three signals placed the feature in their top set. **`WBC`, `LV`, and `eGFR` are 3/3.** `Stent type-SES` is **not** a 3/3 name on this run. `History of HF` and `Cardiogenic shock` enter the top 15 with **0/3** top-set membership — middling ranks on all three lists can still enter the top 15.

**Source file:** [paper_figures/paper_fig13_consensus_ranking.png](paper_figures/paper_fig13_consensus_ranking.png)

### Table 5. Consensus feature report

![Table 5](paper_figures/paper_table5_consensus.png)

**Table 5.** The notebook’s `[5/5]` `interpretability_feature_importance_report` top 15. `importance_score` is the Borda aggregate. `n_methods` counts how many of {MI top, stability, SHAP top} contributed. The three names with n_methods = 3 (`WBC`, `LV`, `eGFR`) are the most consistent TabPFN associations in this run. `Cre` is 2/3 (stability 8/10, SHAP yes) with train MI **0.000000**. `CaI` is first on train MI but **0/10** in stability. `1.1:1Post dilation` is 1/3 (SHAP yes; not in the MI top). Do not quote the old 30-row Cre-leading SHAP (0.158) as this table.

| Rank | Feature | Score | n methods | Stability | mean(\|SHAP\|) | MI | In MI top | In SHAP top |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | WBC | 0.9917 | 3/3 | 1.0 | 1.0202 | 0.020165 | yes | yes |
| 2 | LV | 0.9771 | 3/3 | 0.8 | 0.8695 | 0.012768 | yes | yes |
| 3 | eGFR | 0.9708 | 3/3 | 0.7 | 1.0439 | 0.009830 | yes | yes |
| 4 | LDL | 0.9062 | 2/3 | 0.2 | 0.4973 | 0.009893 | yes | yes |
| 5 | HbA1c | 0.8896 | 2/3 | 0.3 | 0.1783 | 0.004805 | yes | yes |
| 6 | 1.1:1Post dilation | 0.8729 | 1/3 | 0.3 | 0.6906 | 0.002637 | no | yes |
| 7 | Previous PCI | 0.8604 | 2/3 | 0.5 | 0.1356 | 0.002358 | no | yes |
| 8 | Fiberinogen | 0.8542 | 2/3 | 0.4 | 0.0502 | 0.003007 | yes | yes |
| 9 | HGB | 0.8042 | 2/3 | 0.1 | 0.0664 | 0.005327 | yes | yes |
| 10 | No postdilation | 0.7833 | 1/3 | 0.1 | 0.2958 | 0.002406 | no | yes |
| 11 | Lesion location-Ostial | 0.7292 | 1/3 | 0.1 | 0.0222 | 0.005128 | yes | no |
| 12 | History of HF | 0.7292 | 0/3 | 0.1 | 0.0483 | 0.002289 | no | no |
| 13 | Cardiogenic shock | 0.7271 | 0/3 | 0.2 | 0.0316 | 0.001431 | no | no |
| 14 | Cre | 0.7083 | 2/3 | 0.8 | 0.2449 | 0.000000 | no | yes |
| 15 | CaI | 0.7042 | 2/3 | 0.0 | 0.0793 | 0.022005 | yes | yes |

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

*Figures are the executed PNG outputs from `tabpfn_interpretability.ipynb` (`e356bb1` Version 5), copied from the Kaggle working tree. Tables 0–5 are rebuilt from those CSVs (81-row MI and SHAP rankings are in this folder). SHAP / SHAP-IQ started on tabpfn-client thinking then fell back to local after HTTP 429. MI, stability, and PDP use the **train** split (n=3629); SHAP explains all 1,556 held-out rows; k-SII / waterfall / SHAP-IQ are one held-out VLST=1 row (**5176**). Rankings are for interpretation only and should not be reused as a leakage-free feature mask.*
