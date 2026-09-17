# Figure and table plan

Assets are existing files named in the Markdown reports. **Do not invent new numerical panels.** Status follows `paper/frozen_results.yaml` `figures_and_tables` plus freeze_manifest include/exclude rules.

**Display classes**

| Class | Meaning |
| --- | --- |
| **Main** | Intended for the printed paper |
| **Supplement** | Online only |
| **Methods box** | Small table in Methods, not a Results exhibit |
| **Exclude** | Do not use as a headline exhibit (may be cited as a negative control) |

**Completeness:** `ready` = freeze + report aligned; `incomplete` = caption/TODO; `exclude` = must not be the quoted result.

**Numbering aliases.** “MS Table 3” is Part 4 ranking + CIs, **not** Part 4 pooled-F1 Table 3. “MS Table 4” is Part 4 nested operating point, **not** Part 1 Table 4. “MS Figure 3” is Part 1 Figure 6, **not** Part 4 Figure 3. Always write the MS prefix in captions until the journal PDF is typeset.

---

## Proposed manuscript numbering

| MS ID | Class | Title (working) | Source report ID | Source file | Source location | Values allowed | Maps to outline | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Table 1 | Main | Derivation-cohort characteristics | Part 1 Table C | paper_results/01_eda/EDA_paper_figures_and_tables.md | §0 Table C | n=5093 vs 92; cells as printed; Wang flow in footnote | 5.1 | ready (association) |
| Table 2 | Main | Identified 13-covariate unweighted logit (Table 4b) | Part 1 Table 4b | same | Table 4b | adj OR + Wald CI from YAML `table4b_adjusted_or`; EPV≈7.1 | 5.4 | ready |
| Table 3 | Main | Nested-CV ranking (PR-AUC, ROC-AUC, Brier) with bootstrap 95% CIs | Part 4 Table 1 + Table S-CI | paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md | Tables 1, S-CI | seven models; thinking-high ≠ local; CIs n_boot=2000 | 5.5, 5.7, 5.8 | ready |
| Table 4 | Main | Honest nested-CV operating point | Part 4 Table 2 | same | Table 2 | nested t, PPV, recall, spec, F1, F2, 2×2 | 5.5, 5.7, 5.8 | ready |
| Table 5 | Main | TabPFN attribution consensus (Borda top 15) | Part 5 Table 5 | paper_results/05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md | Table 5 | 3/3 WBC, LV, eGFR; Cre MI 0.000000; not a Part 4 mask | 5.9 | ready |
| Figure 1 | Main | Nested-CV PR and ROC curves | Part 4 Figure 1 | Part 4 | Figure 1 | prevalence line 0.0177; legends thinking mode vs TabPFN | 5.5–5.8 | ready |
| Figure 2 | Main | Nested-CV calibration (quantile bins) | Part 4 Figure 2 | Part 4 | Figure 2 | Brier as in **MS Table 3** (ranking); do not infer ECE from the plot | 5.5–5.8 | ready |
| Figure 3 | Main | Univariate vs Table 4b adjusted associations | Part 1 Figure 6 | Part 1 | Figure 6 | association; not causal | 5.4 | ready |
| Figure 4 | Main | Overlap of FDR-20 and ML-13 catalogues | Part 3 Figure 1 | paper_results/03_stats_vs_ml/feature_extraction_comparison.md | Figure 1 | intersection 5 names; Jaccard 5/28 | 5.6 | ready |
| Figure 5 | Main | Held-out mean \|SHAP\| (1556 rows) | Part 5 Figure 5 (bar) ± beeswarm/summary | Part 5 | Figures 3–6 | eGFR 1.0439, WBC 1.0202, LV 0.8695; not 0.158 Cre ranking | 5.9 | ready |

---

## Supplementary exhibits (recommended)

| MS ID | Class | Title | Source | Values / role | Outline | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Table S0 | Supplement | 81 baseline predictor names (CSV order) | paper/table_s0_baseline_predictors.md | listing, not a metric | 4.3 | ready |
| Table M1 | Methods box | Nested-CV model specifications | Part 4 Table 0 | constructors; no unpublished HPs | 4.7 | ready |
| Table M2 | Methods box | TSSI 70/30 with vs without (not nested CV) | Part 4 Table S-TSSI + CSV | LR 0.9575→0.5077; CatBoost 0.9773→0.6582; SMOTE mismatch in caption | 4.4 | ready |
| Table S1 | Supplement | Continuous test-selection rationale | Part 1 Table R + Figure 1 | Welch vs MW | 4.6, 5.4 | ready |
| Table S2 | Supplement | Univariate FDR continuous / binary / categorical | Part 1 Tables 1–3 | associated with; exclude TSSI as baseline | 5.4 | ready |
| Table S3 | Supplement | Exploratory interaction LR tests (16 pairs) | Part 1 Table S2 | hypothesis-generating; LV×eGFR, Men×eGFR q<0.05 | 4.6, 5.4 | ready |
| Table S4 | Exclude (moved to Methods M2) | TSSI 70/30 | Part 4 Table S-TSSI | Use **Table M2**, not a Results supplement | 4.4 | exclude from Results |
| Table S5 | Supplement | Paired Δ PR-AUC vs LightGBM | Part 4 Table S-Δ | +0.1611 vs −0.0201 | 5.7, 5.8 | ready |
| Table S6 | Supplement | Outer-fold PR-AUC wins | Part 4 Table S-folds | thinking-high 5/5; local 2/5 | 5.7, 5.8 | ready |
| Table S7 | Exclude (text only) | Wang risk bins | Part 4 Table S-Wang-bins | Cite 1577 vs 1837 in 5.10 prose if needed; no exhibit | 5.10 | exclude |
| Table S8 | Supplement | Selector unique counts / FFS path lengths | Part 2 Figure 1 table | LOCO 60 / SHAP 40 / FFS 4–12 | 5.6 | ready |
| Table S9 | Supplement | LOCO ∩ SHAP ∩ FFS per model; global 0 / union 86 | Part 2 Tables 2, 4 | attribution; not Part 4 mask | 5.6 | ready |
| Table S10 | Supplement | Part 5 MI top 15 (train) | Part 5 Table 1 | CaI 0.022005; Cre 0.000000 | 5.9 | ready |
| Table S11 | Supplement | Part 5 stability frequencies (train) | Part 5 Table 2 | WBC 10/10 | 5.9 | ready |
| Table S12 | Supplement | Binary PDP (train empirical prior) | Part 5 Table 3 | ΔP +0.0137 / −0.0093; not Part 4 risk | 5.9 | ready |
| Table S13 | Supplement | Held-out mean \|SHAP\| ranking | Part 5 Table 4 | top 15 including Cre **0.2449** (rank 7); YAML `shap.top15_mean_abs_heldout` | 5.9 | ready |
| Figure S1 | Exclude (moved to Methods) | TSSI PR-AUC collapse | Part 4 Figure S-TSSI | Optional Methods figure only | 4.4 | exclude from Results |
| Figure S2 | Supplement | VLST rate by 9-level stent encoder | Part 1 Figure 5 | 106→9; not 99 bars | 5.2 | ready |
| Figure S3 | Exclude by default | Part 4 pooled-F1 confusion matrices | Part 4 Figure 3 | Optimistic pooled threshold. Easy to misread as **MS Table 4**. Omit unless a methods contrast is required, then caption “pooled F1, not nested.” | — | exclude |
| Figure S4 | Exclude (text only) | Wang integer score vs observed rate | Part 4 Figure S-Wang | No dedicated Wang exhibit | 5.10 | exclude |
| Figure S5 | Supplement | Part 2 unique-count / Jaccard heatmaps | Part 2 Figs 1–2, 7 | attribution | 5.6 | ready |
| Figure S6 | Supplement | Continuous + binary PDP | Part 5 Figures 1–2 | empirical prior | 5.9 | ready |
| Figure S7 | Supplement | SHAP summary / beeswarm / scatter / waterfall | Part 5 Figures 3, 4, 6, 7 | waterfall = row 5176 only | 5.9 | ready |
| Figure S8 | Supplement | k-SII / SHAP-IQ one-row plots | Part 5 Figures 8–12 | **one patient, not cohort interactions** | 5.9 | ready |
| Figure S9 | Supplement | Consensus ranking plot | Part 5 Figure 13 | pairs with **MS Table 5** | 5.9 | ready |
| Figure S10 | Supplement | Univariate continuous/binary overviews | Part 1 Figures 2–4 | association | 5.4 | ready |
| Figure S11 | Supplement | Domain supplementary panels | Part 1 S1–S5 | association; S4 is a **different** logit than Table 4b | 5.4 | ready if labelled |

---

## Explicitly excluded as headline exhibits

| Asset | Why exclude | If mentioned at all |
| --- | --- | --- |
| Part 1 Table 4 (17-covariate logit) | Unidentified; EPV≈5.4; adj OR 0.144/0.464 | Methods: “stored unidentified specification, not quoted” |
| Part 4 Table 3 pooled F1 | Optimistic; nested recall must be Table 2 | Methods contrast only |
| Part 4 Figure 3 without “pooled” caption | Easy to misquote as nested 2×2 | Supplement only with optimistic label |
| EDA sidecar “Raw levels=99” | Not encoder n_raw | Never |
| Evidence-map Rev 7 local Brier 0.0673 panels | Historical dump | Never |
| Old SHAP 15+15 / Cre 0.158 figures | Superseded by Version 5 1556-row SHAP | Never |
| k-SII rows 5099/5093 | Wrong row | Never |
| De-novo CONSORT from this repo | Flow is cited from Wang; CSV is analysed n | Footnote to Table 1 |
| Decision-curve / ECE tables | ECE not computed | leave absent |
| NPV | Identity from Table 2 2×2 only | optional; not a new metric table |
| Complete 81-name table | Unresolved in freeze | TODO |
| Part 2 Table 5 priority-feature alias mismatches | Display artefact, not a clinical ranking | Omit or caption as alias mismatch only |
| Part 4 Table S-Wang / S-Wang-bins / Figure S-Wang | Author decision: text only | Cite 0.8013 / 0.1032 in Results 5.10 prose |
| Part 4 Table S-TSSI / Figure S-TSSI as Results exhibits | Author decision: Methods only | Table M2 |

---

## Combined ranking table (Table 3) — required row order

Keep arms and metrics distinct. Suggested columns: Model, PR-AUC (95% CI), ROC-AUC (95% CI), Brier (95% CI). Optional extra columns: PR fold mean±SD.

| Row | Model label in manuscript | YAML path |
| --- | --- | --- |
| 1 | TabPFN thinking-high | `model_results.tabpfn_thinking_high.metrics` |
| 2 | LightGBM | `baseline_models_without_tssi.lightgbm.nested_cv_oof` |
| 3 | XGBoost | `...xgboost` |
| 4 | TabPFN local | `model_results.tabpfn_local.metrics` |
| 5 | CatBoost | `...catboost` |
| 6 | Random forest | `...random_forest` |
| 7 | Logistic regression | `...logistic_regression` |

Do **not** add Wang as a row in this table. Cite the frozen integer score in Results 5.10 **prose only**.

---

## Combined operating-point table (Table 4)

Quote **only** Part 4 Table 2. NPV is optional as TN/(TN+FN) from those counts, not a new computation. F2 is β=2.0.

| Model | Threshold mean±SD | Accuracy | PPV | Recall | Spec | F1 | F2 | TN/FP/FN/TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Thinking-high | 0.271±0.067 | 0.9915 | 0.7927 | 0.7065 | 0.9967 | 0.7471 | 0.7222 | 5076/17/27/65 |
| LightGBM | 0.112±0.090 | 0.9878 | 0.6526 | 0.6739 | 0.9935 | 0.6631 | 0.6695 | 5060/33/30/62 |
| XGBoost | 0.225±0.060 | 0.9875 | 0.6452 | 0.6522 | 0.9935 | 0.6486 | 0.6508 | 5060/33/32/60 |
| Local | 0.166±0.020 | 0.9844 | 0.5478 | 0.6848 | 0.9898 | 0.6087 | 0.6522 | 5041/52/29/63 |
| CatBoost | 0.167±0.040 | 0.9815 | 0.4836 | 0.6413 | 0.9876 | 0.5514 | 0.6020 | 5030/63/33/59 |
| RF | 0.118±0.013 | 0.9840 | 0.5517 | 0.5217 | 0.9923 | 0.5363 | 0.5275 | 5054/39/44/48 |
| LR | 0.947±0.035 | 0.9769 | 0.3654 | 0.4130 | 0.9870 | 0.3878 | 0.4025 | 5027/66/54/38 |

YAML: each model’s `nested_operating_point_table2`. Source: Part 4 Table 2.

---

## Caption constraints (copy into production captions)

| Exhibit | Must say | Must not say |
| --- | --- | --- |
| Table 1 | Association, not prediction; TSSI omitted; DAPT = follow-up persistence; flow cited from Wang | Validation cohort; independent risk factors |
| Table 2 | Unweighted 13-covariate logit; Wald 95% CI; EPV≈7.1; OR<1 is not treatment benefit | Independent risk factors; **Part 1 Table 4** unidentified ORs |
| Table 3 | Nested-CV pooled OOF ranking; derivation cohort; thinking-high vs local named; this is **not** Part 4 Table 3 | External validation; a single TabPFN Brier; pooled F1 |
| Table 4 | Inner-CV F1 threshold applied once to the outer fold; this is **not** Part 1 Table 4 | Pooled F1; unpublished NPV (use Table 2 identity only) |
| Table 5 | Attribution on a 70/30 split of the same file; not a Part 4 mask | Validated biomarkers; nested-CV features |
| Figure 1 | Nested-CV OOF; prevalence 0.0177 on PR panel | Test-set performance |
| Figure 2 | Quantile-bin reliability; Brier in **MS Table 3** | ECE (unreported) |
| Figure 3 | Association after adjustment in this sparse spec | Causal / standalone clinical claim |
| Figure 4 | Methods comparison of catalogues | True risk markers |
| Figure 5 | Held-out SHAP, n=1556, local after HTTP 429 | Full-cohort SHAP; 15+15 |
| Table M2 | SMOTE True vs False; Methods demonstration; not nested CV | Identical protocol except TSSI; Results exhibit |
| Figures S8 | One held-out VLST=1 row (5176) | Cohort interactions |
| Figure S3 | *(exclude by default)* Optimistic pooled threshold | Nested 2×2; MS Table 4 |

---

## File paths for production copy

| MS ID | PNG / CSV (from reports) |
| --- | --- |
| Table 1 | `paper_results/01_eda/paper_figures/paper_table_c_cohort_characteristics.{png,csv}` |
| Table 2 | `.../paper_table4b_reduced_or.{png,csv}` |
| Table 3 | `paper_results/04_tabpfn_rating/paper_figures/paper_table1_ranking.{png,csv}` + `paper_table_s_bootstrap_ci.{png,csv}` |
| Table 4 | `.../paper_table2_nested_operating_point.{png,csv}` |
| Table 5 | `paper_results/05_tabpfn_interpretability/paper_figures/paper_table5_consensus.{png,csv}` |
| Figure 1 | `.../paper_fig1_pr_roc_curves.png` |
| Figure 2 | `.../paper_fig2_calibration_curves.png` |
| Figure 3 | `paper_results/01_eda/paper_figures/paper_fig6_uni_vs_multivariable_or.png` |
| Figure 4 | `paper_results/03_stats_vs_ml/paper_figures/fig1_venn_overlap.png` |
| Figure 5 | `paper_results/05_tabpfn_interpretability/paper_figures/paper_fig5_shap_bar.png` |

---

## Count and load

- **Main tables:** 5  
- **Main figures:** 5  
- **Methods boxes:** M1 (model specs), M2 (TSSI leakage).  
- **Supplement:** as listed; Wang integer-score panels are **text only**. Do not drop MS Tables 3–4.

**Safe for drafting:** yes, if excluded assets stay excluded and captions use the constraints above.
