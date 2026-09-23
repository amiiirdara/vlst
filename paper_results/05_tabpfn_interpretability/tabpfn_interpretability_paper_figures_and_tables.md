# TabPFN interpretability — paper figures and tables

This document gathers publication-oriented figures and tables from the **two split notebooks** (parent `tabpfn_interpretability.ipynb` archived; Kaggle session-limit split):

| Notebook | Role | Live dump |
| --- | --- | --- |
| `code/modeling/interpretability/tabpfn_interpretability_fs_pdp.ipynb` | Mutual information, stability SFS, PDP, consensus inputs | `Kaggle_tabpfn_intrepretebility_results/fs_pdp_MI/` (papermill 2026-09-20T10:58Z–21:28Z) |
| `code/modeling/interpretability/tabpfn_interpretability_shap.ipynb` | SHAP / SHAP-IQ / k-SII | `Kaggle_tabpfn_intrepretebility_results/shap/` (papermill 2026-09-20T10:05Z–11:50Z) |

Rebuild: `code/modeling/tools/rebuild_part5_from_split_dumps.py` copies dump PNGs/CSVs into `paper_figures/` and merges held-out SHAP into Table 5 / Figure 13 (the fs dump CSV has `shap_mean_abs=0`). Pins: **`tabpfn==9.0.0`** / **`tabpfn-client==0.6.0`**, v3.5.

**Cohort / protocol.** Raw VLST.csv, n = 5,185. **ALL LEAKS OFF** (same as nested CV; [`../anti_leakage_protocol.md`](../anti_leakage_protocol.md)): identifiers (`NO.`, `Name`) and `Time since stent implantation` dropped (mixed time-to-event vs completed follow-up). **`WBC` dropped** (recording-precision / case-control batch marker; Wang also excluded it from Cox). Labs `Cre`/`CaI`/`Fiberinogen`/`Fast-Glu` quantized **pre-split**. `Stent type-SES` uses the **shared 9-level encoder** fit on **train only** (`min_count=30`). No SMOTE. Follow-up drugs stay in. Target = `Stent thrombosis`. EDA found **no missing values**. Feature ranking, PDP, and SHAP are **interpretation / attribution** — not prediction, not external validation, and not a locked-in feature mask for Part 4.

**This run (D4).** Kaggle Tesla T4, `tabpfn==9.0.0` / `tabpfn-client==0.6.0`, v3.5. fs_pdp papermill **2026-09-20T10:58Z–21:28Z**; shap **2026-09-20T10:05Z–11:50Z**. Dump: `code/modeling/interpretability/Kaggle_tabpfn_intrepretebility_results/{fs_pdp_MI,shap}/`. Protocol: stratified `train_test_split(test_size=0.3, random_state=42)` → train **3,629 / 64 events**, held-out **1,556 / 28 events**. MI, stability SFS (`STABILITY_N_SEEDS=8`), and PDP **fit/average on train**. SHAP explains **all 1,556 held-out rows**. k-SII / waterfall / SHAP-IQ = first held-out VLST=1 (held-out position 20, **cohort row 5176**). `FS_THINKING_MODE=False`; `PDP_USE_CLIENT=False`; `INTERP_THINKING_MODE=True` (effort high, metric average_precision). **80 columns** after IDs + TSSI + WBC drop and quantization.

**Methods note — selection vs explanation.** Mutual information, stability (repeated forward SFS), and PDP use the **train split**. SHAP explains **every held-out row** (28 events / 1,528 non-events). k-SII / SHAP-IQ remain **one** held-out VLST=1 patient (row **5176**). Do not describe this run as full-cohort MI/SFS/PDP, as 15+15 SHAP, as 10 SFS seeds, or as a WBC-leading ranking.

**Backends.** Mutual information, stability, and PDP use **local** `tabpfn` (`FS_THINKING_MODE=False`). SHAP and SHAP-IQ use **client thinking-high** (`INTERP_THINKING_MODE=True`). PDP uses the empirical prior (`PDP_USE_CLIENT=False`; y-axis near prevalence; **not Part 4 nested-CV risk**). The shapiq `imputer="baseline"` is **not** a missing-value fill.

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

**Table 0.** Five signals plus a Borda-style consensus. No single method is trusted alone. Stability frequency is the reliability signal (how often forward SFS keeps a feature across **8/8 train** resamples). MI and SFS use **train**. SHAP uses **all 1,556 held-out rows**. Pairwise k-SII is a one-row interaction view (row **5176**, VLST=1), not a global interaction ranking. `WBC` is not in this dump.

| Method | Question | Backend | Notebook setting |
| --- | --- | --- | --- |
| mutual_info_classif | Univariate association | sklearn | 0 TabPFN calls; **train** n=3629; 80 scores in `interpretability_mutual_info_ranking.csv` |
| Stability (repeated SFS) | Selection frequency | local TabPFN | **8/8** resamples × top-10 forward SFS, AP scoring, **train** |
| PDP | Average predicted probability (empirical prior) | local TabPFN | **Train** n=3629; `PDP_USE_CLIENT=False`; y-axis “empirical prior / not Part 4 risk” |
| SHAP (shapiq SV) | Local attributions | client thinking-high | All 1,556 held-out rows; fit/background = train; budget=256 |
| k-SII / SHAP-IQ | Pairwise interactions | client thinking-high | One held-out VLST=1 (pos 20, cohort **5176**); budget=256 |
| Consensus (Borda) | Mean of normalized ranks | aggregate | Train MI + train stability + held-out mean(\|SHAP\|); SHAP merged from the shap dump |

**Source files:** [paper_figures/paper_table0_methods.png](paper_figures/paper_table0_methods.png), [paper_figures/paper_table0_methods.csv](paper_figures/paper_table0_methods.csv)

---

## 2. Univariate and stability screens

### Table 1. Top 15 by mutual information

![Table 1](paper_figures/paper_table1_mutual_info.png)

**Table 1.** `mutual_info_classif` ranking of the **80-column** matrix on the **train split** (n=3629). Top 15 from `interpretability_mutual_info_ranking.csv`. `WBC` is not a column. `Cre` is 50th of 80 with train MI **0.000338**. This is a marginal association screen, not a model attribution.

| Rank | Feature | Mutual information |
| ---: | --- | --- |
| 1 | CaI | 0.020536 |
| 2 | LV | 0.012818 |
| 3 | eGFR | 0.009424 |
| 4 | LDL | 0.009367 |
| 5 | HbA1c | 0.007328 |
| 6 | Lesion location-Ostial | 0.005293 |
| 7 | P-RCA | 0.004830 |
| 8 | HGB | 0.004365 |
| 9 | 1.1:1Post dilation | 0.004057 |
| 10 | HDL | 0.003823 |
| 11 | Fiberinogen | 0.003712 |
| 12 | Current drinking | 0.003202 |
| 13 | Stent type-SES | 0.003176 |
| 14 | stent overlap | 0.003081 |
| 15 | P-LCX | 0.003014 |

**Source files:** [paper_figures/paper_table1_mutual_info.png](paper_figures/paper_table1_mutual_info.png), [paper_figures/paper_table1_mutual_info.csv](paper_figures/paper_table1_mutual_info.csv), [paper_figures/interpretability_mutual_info_ranking.csv](paper_figures/interpretability_mutual_info_ranking.csv)

### Table 2. Stability selection frequency

![Table 2](paper_figures/paper_table2_stability.png)

**Table 2.** Forward sequential feature selection (keep 10 of 80, 5-fold CV, average precision) repeated over **8/8 shuffled seeds** on the **train split**. Selected in **8/8**: `CaI`, `LV`, `eGFR`. Selected in 7/8: `Cre`, `HbA1c`. Selected in 6/8: `Age`. Selected in 5/8: `LVEF`. Selected in 4/8 (`0.5` cutoff): `1.1:1Post dilation`. **`WBC` is not in this dump.** Earlier 10-seed / “PROVISIONAL — 3/10 seeds” tables are obsolete.

| Feature | Selected | Frequency |
| --- | --- | ---: |
| CaI | 8/8 | 1.000 |
| LV | 8/8 | 1.000 |
| eGFR | 8/8 | 1.000 |
| Cre | 7/8 | 0.875 |
| HbA1c | 7/8 | 0.875 |
| Age | 6/8 | 0.750 |
| LVEF | 5/8 | 0.625 |
| 1.1:1Post dilation | 4/8 | 0.500 |
| LDL | 2/8 | 0.250 |
| No postdilation | 2/8 | 0.250 |
| TCL | 2/8 | 0.250 |
| Fiberinogen | 1/8 | 0.125 |
| Initial diagnosis-AMI | 1/8 | 0.125 |
| Previous PCI | 1/8 | 0.125 |
| Slow flow | 1/8 | 0.125 |
| Stent type-SES | 1/8 | 0.125 |

**Source files:** [paper_figures/paper_table2_stability.png](paper_figures/paper_table2_stability.png), [paper_figures/paper_table2_stability.csv](paper_figures/paper_table2_stability.csv), [paper_figures/interpretability_feature_stability_summary.csv](paper_figures/interpretability_feature_stability_summary.csv)

---

## 3. Partial dependence

PDP candidates were taken from the stability / MI screens. Continuous PDP uses grid resolution 30. Binary PDP forces each flag to 0 vs 1 and reports the change in average predicted P[Stent thrombosis]. Fit and average are on the **train split** (n=3629, events=64). `WBC` is not a PDP column on this run.

**Methods note — PDP is empirical prior, not Part 4 risk.** PDP uses local TabPFN with empirical class prior (`PDP_USE_CLIENT=False`). Average predicted probabilities sit near prevalence (~0.018). Do **not** quote 0.13–0.26 or “toward ~0.6” as clinical risk. Neither scale is the Part 4 nested-CV client.

### Figure 1. Continuous partial dependence

![Figure 1](paper_figures/paper_fig1_pdp_continuous.png)

**Figure 1.** Continuous PDP on the **empirical-prior** scale, **train** (n = 3,629), from `Kaggle_tabpfn_intrepretebility_results/fs_pdp_MI`. Dashed line = cohort prevalence. Not Part 4 nested-CV risk and not a treatment effect.

**Source file:** [paper_figures/paper_fig1_pdp_continuous.png](paper_figures/paper_fig1_pdp_continuous.png)

### Figure 2. Binary partial dependence

![Figure 2](paper_figures/paper_fig2_pdp_binary.png)

**Figure 2.** Binary flags forced to 0 vs 1 and averaged over **train**. Largest |Δ| is `Previous PCI` (0.017711 → 0.019006, Δ **+0.001294**). `1.1:1Post dilation` Δ is **−0.000094**. A negative Δ is a lower modelled probability of recorded VLST, not a treatment benefit (confounding by indication).

**Source file:** [paper_figures/paper_fig2_pdp_binary.png](paper_figures/paper_fig2_pdp_binary.png)

### Table 3. Binary PDP numeric values

![Table 3](paper_figures/paper_table3_pdp_binary.png)

**Table 3.** Empirical-prior binary PDP on **train** n=3629. These values are **not** clinical risk and **not** Part 4 nested-CV probabilities.

| Feature | P(y=1 \| 0) | P(y=1 \| 1) | ΔP |
| --- | ---: | ---: | ---: |
| Previous PCI | 0.017711 | 0.019006 | +0.001294 |
| No postdilation | 0.019160 | 0.019312 | +0.000153 |
| Slow flow | 0.017715 | 0.017708 | −0.000007 |
| Lesion location-Ostial | 0.017715 | 0.017711 | −0.000004 |
| 1.1:1Post dilation | 0.019260 | 0.019166 | −0.000094 |
| Initial diagnosis-AMI | 0.017935 | 0.017815 | −0.000120 |

**Source files:** [paper_figures/paper_table3_pdp_binary.png](paper_figures/paper_table3_pdp_binary.png), [paper_figures/paper_table3_pdp_binary.csv](paper_figures/paper_table3_pdp_binary.csv)

---

## 4. SHAP attributions

Fit on **train**; explain **all 1,556 held-out rows**. Mean(|SHAP|) below is the top 15 of the **80-column** ranking (`interpretability_shap_mean_abs.csv`). Indices: `interpretability_shap_explain_indices.csv` (1,556 rows; 28 events). Do not call this 15+15 or global SHAP on 5,185. `WBC` is not a column.

### Figure 3. SHAP summary

![Figure 3](paper_figures/paper_fig3_shap_summary.png)

**Figure 3.** SHAP summary / beeswarm for the **1,556-row held-out** slice (`sv_interpretability_shap_summary.png`).

**Source file:** [paper_figures/paper_fig3_shap_summary.png](paper_figures/paper_fig3_shap_summary.png)

### Figure 4. SHAP scatter (leading feature)

![Figure 4](paper_figures/paper_fig4_shap_scatter_age.png)

**Figure 4.** Scatter of the leading SHAP feature versus SHAP on the held-out slice (`sv_interpretability_shap_scatter_f0.png`; this run’s f0 is **eGFR**, not Age). A local scatter, not a cohort dose–response.

**Source file:** [paper_figures/paper_fig4_shap_scatter_age.png](paper_figures/paper_fig4_shap_scatter_age.png)

### Figure 5. Mean absolute SHAP (bar)

![Figure 5](paper_figures/paper_fig5_shap_bar.png)

**Figure 5.** Mean(|SHAP|) on the 1,556-row held-out slice. Leading names: `eGFR` 1.2288, `CaI` 1.0867, `Cre` 0.8093, `LV` 0.4828.

**Source file:** [paper_figures/paper_fig5_shap_bar.png](paper_figures/paper_fig5_shap_bar.png)

### Figure 6. Compact SHAP beeswarm

![Figure 6](paper_figures/paper_fig6_shap_beeswarm.png)

**Figure 6.** Compact beeswarm of the same held-out attributions.

**Source file:** [paper_figures/paper_fig6_shap_beeswarm.png](paper_figures/paper_fig6_shap_beeswarm.png)

### Figure 7. One-row SHAP waterfall

![Figure 7](paper_figures/paper_fig7_shap_waterfall.png)

**Figure 7.** Waterfall for the first held-out VLST=1 patient (held-out pos 20, **cohort row 5176**). Local explanation for one patient, not a global ranking, and not the PDP ~0.018 axis.

**Source file:** [paper_figures/paper_fig7_shap_waterfall.png](paper_figures/paper_fig7_shap_waterfall.png)

### Table 4. Mean(|SHAP|) ranking (held-out, top 15 of 80)

![Table 4](paper_figures/paper_table4_shap_mean_abs.png)

**Table 4.** mean(|SHAP|) top 15 of 80 columns on **1,556 held-out rows**. `WBC` is absent.

| Rank | Feature | mean(\|SHAP\|) |
| ---: | --- | ---: |
| 1 | eGFR | 1.2288 |
| 2 | CaI | 1.0867 |
| 3 | Cre | 0.8093 |
| 4 | LV | 0.4828 |
| 5 | LDL | 0.3659 |
| 6 | No postdilation | 0.2812 |
| 7 | HbA1c | 0.2673 |
| 8 | Previous PCI | 0.2604 |
| 9 | 1.1:1Post dilation | 0.2483 |
| 10 | Stent type-SES | 0.2402 |
| 11 | TCL | 0.2369 |
| 12 | TG | 0.2213 |
| 13 | CKD90 | 0.1377 |
| 14 | Men | 0.1314 |
| 15 | HDL | 0.1174 |

**Source files:** [paper_figures/paper_table4_shap_mean_abs.png](paper_figures/paper_table4_shap_mean_abs.png), [paper_figures/paper_table4_shap_mean_abs.csv](paper_figures/paper_table4_shap_mean_abs.csv), [paper_figures/interpretability_shap_mean_abs.csv](paper_figures/interpretability_shap_mean_abs.csv), [paper_figures/interpretability_shap_explain_indices.csv](paper_figures/interpretability_shap_explain_indices.csv)

---

## 5. Pairwise interactions — k-SII

k-SII plots use **one illustrative held-out VLST=1 row** (held-out pos 20, **cohort row 5176**, budget = 256). Node size is the main effect; edge width is the pairwise interaction. They illustrate how TabPFN combines features for that row; they are **not** a cohort interaction screen.

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

Section of the shap notebook recomputes imputation-based Shapley values and k-SII with shapiq’s native plotting API. Figures 10–12 are a second view of the **same one-row explanation** (cohort row **5176**), not an independent replication on new rows. **Figure 10** (force plot) is not exported in the 2026-09-20 shap dump; the waterfall is Figure 7.

### Figure 10. SHAP-IQ force plot (one row)

![Figure 10](paper_figures/paper_fig10_shapiq_force.png)

**Figure 10.** Force / additive layout for held-out pos 20 / cohort row **5176**. Read it as the compact counterpart of the waterfall in Figure 7.

**Source file:** [paper_figures/paper_fig10_shapiq_force.png](paper_figures/paper_fig10_shapiq_force.png)

### Figure 11. SHAP-IQ k-SII network

![Figure 11](paper_figures/paper_fig11_shapiq_network.png)

**Figure 11.** Native shapiq network for the same one-row k-SII. Layout is a restyle of Figure 8, not a new sample of patients.

**Source file:** [paper_figures/paper_fig11_shapiq_network.png](paper_figures/paper_fig11_shapiq_network.png)

### Figure 12. SHAP-IQ k-SII UpSet plot

![Figure 12](paper_figures/paper_fig12_shapiq_upset.png)

**Figure 12.** Native shapiq UpSet plot of top main effects and pairwise interactions for the same row. Read it as a restyle of Figure 9, not as a new sample of patients.

**Source file:** [paper_figures/paper_fig12_shapiq_upset.png](paper_figures/paper_fig12_shapiq_upset.png)

---

## 7. Consensus ranking

Ranking uses a **Borda-style mean of normalized ranks** across **train** mutual information, **train** stability frequency, and **held-out** mean(|SHAP|), with `n_methods` (out of 3) as a consensus count. The FS dump’s consensus CSV had `shap_mean_abs=0` (SHAP not in that RESULT_DIR). Table 5 below **merges** `interpretability_shap_mean_abs.csv` from the shap dump using the notebook formula. Report the top 15 as *associations* with stent thrombosis under TabPFN — exploratory, not causal, on a ~2% prevalence cohort. `Cre` train MI is **0.000338** (rank 50 of 80). `WBC` is not a column.

### Figure 13. Top 15 by consensus

![Figure 13](paper_figures/paper_fig13_consensus_ranking.png)

**Figure 13.** Aggregated importance (1 = strongest mean normalized rank). Annotations give how many of the three signals placed the feature in their top set. **`CaI`, `eGFR`, and `LV` are 3/3** (also `HbA1c` and `1.1:1Post dilation` at the 0.5 SFS cutoff). **`WBC` is not in this dump.**

**Source file:** [paper_figures/paper_fig13_consensus_ranking.png](paper_figures/paper_fig13_consensus_ranking.png)

### Table 5. Consensus feature report

![Table 5](paper_figures/paper_table5_consensus.png)

**Table 5.** Borda aggregate after merging the shap dump. `n_methods` counts {MI top, SFS freq ≥ 0.5, SHAP top}. Three-way names at rank 1–3: `CaI`, `eGFR`, `LV`. `Cre` is 2/3 (SFS 7/8, SHAP yes) with train MI **0.000338**.

| Rank | Feature | Score | n methods | Stability | mean(\|SHAP\|) | MI | In MI top | In SHAP top |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | CaI | 0.9916 | 3/3 | 1.000 | 1.0867 | 0.020536 | yes | yes |
| 2 | eGFR | 0.9873 | 3/3 | 1.000 | 1.2288 | 0.009424 | yes | yes |
| 3 | LV | 0.9789 | 3/3 | 1.000 | 0.4828 | 0.012818 | yes | yes |
| 4 | HbA1c | 0.9430 | 3/3 | 0.875 | 0.2673 | 0.007328 | yes | yes |
| 5 | LDL | 0.9325 | 2/3 | 0.250 | 0.3659 | 0.009367 | yes | yes |
| 6 | 1.1:1Post dilation | 0.9030 | 3/3 | 0.500 | 0.2483 | 0.004057 | yes | yes |
| 7 | Stent type-SES | 0.8565 | 2/3 | 0.125 | 0.2402 | 0.003176 | yes | yes |
| 8 | Fiberinogen | 0.8270 | 1/3 | 0.125 | 0.0694 | 0.003712 | yes | no |
| 9 | Previous PCI | 0.8059 | 1/3 | 0.125 | 0.2604 | 0.001986 | no | yes |
| 10 | TCL | 0.7848 | 1/3 | 0.250 | 0.2369 | 0.001167 | no | yes |
| 11 | Cre | 0.7700 | 2/3 | 0.875 | 0.8093 | 0.000338 | no | yes |
| 12 | No postdilation | 0.7257 | 1/3 | 0.250 | 0.2812 | 0.000079 | no | yes |
| 13 | Age | 0.7215 | 1/3 | 0.750 | 0.0373 | 0.002065 | no | no |
| 14 | HDL | 0.7025 | 2/3 | 0.000 | 0.1174 | 0.003823 | yes | yes |
| 15 | HGB | 0.6899 | 1/3 | 0.000 | 0.0644 | 0.004365 | yes | no |

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

*Figures are the executed PNG outputs from `tabpfn_interpretability_fs_pdp.ipynb` and `tabpfn_interpretability_shap.ipynb` (papermill 2026-09-20; `tabpfn==9.0.0` / v3.5), copied from `Kaggle_tabpfn_intrepretebility_results/`. Tables 0–5 are rebuilt from those CSVs (80-row MI and SHAP rankings; SFS **8/8 seeds**). MI, stability, and PDP use the **train** split (n=3629); SHAP explains all 1,556 held-out rows; k-SII / waterfall / SHAP-IQ are one held-out VLST=1 row (**5176**). Rankings are for interpretation only and should not be reused as a leakage-free feature mask. `WBC` is dropped on this run.*
