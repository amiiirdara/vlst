# Numerical claims inventory

Every number below appears in the inspect set. Cell numbers are 0-based JSON indices. Item class: numerical result unless noted. Do not treat concatenated `paper_results.md` Part 5 captions as current when they disagree with the standalone Part 5 file.

TabPFN thinking-high and TabPFN local are listed separately everywhere.

---

## A. Cohort and outcome

| Claim | Value | File | Location | Source type | Item class |
| --- | --- | --- | --- | --- | --- |
| Analysed n | 5,185 | `00_front_matter.md`; Part 1 Table C; Part 4 protocol; `wang_vlst_score.ipynb` cell 2 `assert len(df) == 5185` | report / notebook | numerical result |
| VLST events | 92 | same; wang cell 2 `assert int(y.sum()) == 92` | report / notebook | numerical result |
| Non-events | 5,093 | Part 1 Table C header; Part 4 bootstrap note | report | numerical result |
| Prevalence | 0.0177 (1.77%) | Front matter; Part 4 protocol | report | numerical result |
| Eligible → analysed | 6,038 → 5,185 | Front matter Outcome; Part 1 Table C caption | report | numerical result (cited from Wang) |
| Exclusions | 236 in-hospital deaths; 413 refused follow-up; 204 lost | same | report | numerical result (cited from Wang) |
| Median follow-up | 1,502 days | Front matter Outcome; Part 4 §6 | report | numerical result (cited from Wang / column description) |
| Median PCI → VLST | 697 days | Front matter Outcome; Part 4 §6 | report | numerical result |
| TSSI min (events) | 380 days | Part 4 §6 | report | numerical result |
| TSSI min (non-events) | 1,241 days | Part 4 §6 | report | numerical result |
| TSSI max (non-events) | 1,605 days | Part 4 §6 | report | numerical result |
| Part 2 fit / val | 4,148 rows (74 events) / 1,037 rows (18 events) | Part 2 header | report | numerical result |
| Part 5 train / test | 3,629 / 64 events; 1,556 / 28 events | Part 5 protocol | report | numerical result |
| Part 4 events per outer fold | 18, 18, 18, 19, 19 | `00_front_matter.md` EPV table | report | numerical result |
| Ethics / NCT | 2013-256; NCT03491891 | Front matter | report | provenance detail |

### Table C selected cells (Part 1, association)

Source: `EDA_paper_figures_and_tables.md` Table C. n = 5,093 vs 92.

| Variable | No VLST | VLST | p (as printed) |
| --- | --- | --- | --- |
| Age, years | 59.83 (9.93) | 60.71 (11.33) | 0.463 |
| Men | 3489 (68.51%) | 68 (73.91%) | 0.268 |
| Diabetes | 1293 (25.39%) | 36 (39.13%) | 0.003 |
| Previous PCI | 94 (1.85%) | 10 (10.87%) | 1.25e-05 |
| 3-vessel disease | 1422 (27.92%) | 42 (45.65%) | 0.000 |
| LVEF, % | 55.15 (4.52) | 54.55 (3.68) | 0.033 |
| LV | 44.55 (4.04) | 49.11 (4.23) | 5.44e-17 |
| WBC | 8.75 (3.24) | 12.49 (3.92) | 7.90e-21 |
| eGFR | 120.03 (34.10) | 95.88 (19.63) | 4.64e-20 |
| CKD90 | 860 (16.89%) | 32 (34.78%) | 6.55e-06 |
| CaI | 37.37 (61.64) | 40.55 (72.25) | 0.051 |
| SES (`PES`) | 3502 (68.76%) | 76 (82.61%) | 0.004 |
| 1.1:1 post-dilation | 2496 (49.01%) | 14 (15.22%) | 1.30e-10 |
| No postdilation | 2597 (50.99%) | 78 (84.78%) | 1.30e-10 |
| DAPT during follow-up | 2260 (44.37%) | 35 (38.04%) | 0.226 |

Wang Table 1 “No post-dilation” 14/92 is **not** the CSV `No postdilation` column (78/92). Documented in Table C caption.

---

## B. Predictor counts and encoder

| Claim | Value | File | Location | Source type |
| --- | --- | --- | --- | --- |
| Features after drop IDs + TSSI | 81 | Part 4/5 protocols; front matter EPV | report | numerical result |
| Part 2 scaled width | 88 (81 − 1 brand + 8 dummies, drop-first) | Part 2 header | report | numerical result |
| Part 4 classic width | ~89 (OHE without drop-first) | Part 4 methods; front matter limitation 8 | report | methodological detail (approximate) |
| Stent encoder levels | 9; `min_count=30` | Part 1 Figure 5; Part 4 cell 5 `STENT_BRAND_MIN_COUNT = 30` | report / notebook | numerical result |
| Raw distinct stent strings (encoder `n_raw`, last Kaggle/code run) | **106** | Part 4/5 protocol; `stent_encoding.py` on `VLST.csv`; Kaggle print `Stent brand: 106 raw strings -> 9 levels` | report / notebook / script | numerical result |
| Unique names after current `canonicalize_stent_brand` (before n<30 collapse) | **30** | live `stent_encoding.py` on current CSV (2026-09-17 recount) | Python (encoder) | numerical result |
| Encoder levels after min_count=30 | **9** | same | report | numerical result |
| EDA χ² helper “Raw levels” (not `n_raw`) | **99** | `categorical_target_association_tests.csv`; `.nbdump` EDA print. Do **not** quote as encoder n_raw. | report sidecar | provenance detail |
| Figure 5 VLST rates by 9-level brand | `other` 5.5%; `xiencev` 3.3%; `partner` 2.3%; `excel` 2.1%; `firebird` 1.6%; `tivoli` 0.83%; `resolute` 0.82%; `xv` 0%; `xx` 0% | Part 1 Figure 5 | report | numerical result |
| PES means (Wang encoding check) | VLST 0.8261; control 0.6876 | `wang_vlst_score.ipynb` cell 2 asserts | notebook | numerical result |

---

## C. TSSI leakage metrics

Source of the table: `rebuild_tssi_leakage_table.py` `ROWS` (lines 27–33). Part 4 Table S-TSSI / Figure S-TSSI. Hold-out n = 1,556 (Figure S-TSSI caption).

| Model | with_pr_auc | without_pr_auc | with_roc_auc | without_roc_auc | with_f1 | without_f1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9575 | 0.5077 | 0.9990 | 0.9171 | 0.6923 | 0.2637 |
| Decision Tree | 0.7308 | 0.0405 | 0.9696 | 0.5774 | 0.8500 | 0.1875 |
| Random Forest | 0.9680 | 0.4700 | 0.9993 | 0.9338 | 0.8000 | 0.4348 |
| Gaussian NB | 0.0209 | 0.0209 | 0.5854 | 0.5854 | 0.0409 | 0.0409 |
| CatBoost | 0.9773 | 0.6582 | 0.9995 | 0.9669 | 0.9412 | 0.5806 |
| XGBoost | 0.9609 | 0.6118 | 0.9987 | 0.9380 | 0.9412 | 0.5714 |
| LightGBM | 0.9708 | 0.6018 | 0.9989 | 0.9483 | 0.9412 | 0.5625 |

Notebook print check (with-TSSI, `baseline_tssi_leakage.ipynb` stored outputs): Logistic PR-AUC `0.957530864197531`; CatBoost `0.9773406947319991` — matches the script to 4 dp.

Part 4 headline rounding: LR 0.958 → 0.508; CatBoost 0.977 → 0.658.

---

## D–E. Splits (numeric)

Already in §A. Bootstrap: `n_boot = 2000`, seed 42 (Part 4 §3).

---

## F–H. Part 4 nested-CV prediction performance

Authoritative: `baseline_plus_tabpfn_paper_figures_and_tables.md` Tables 1–3 and S-CI / S-Δ / S-folds, matching Version 4 notebook `139d143`. These are **prediction performance**, not associations.

### Table 1. Pooled OOF ranking

| Rank | Model | PR-AUC | PR fold mean ± SD | ROC-AUC | ROC fold mean ± SD | Brier |
| ---: | --- | ---: | --- | ---: | --- | ---: |
| 1 | TabPFN (thinking-high) | 0.8553 | 0.8488 ± 0.0861 | 0.9905 | 0.9906 ± 0.0070 | 0.0064 |
| 2 | LightGBM | 0.6942 | 0.6957 ± 0.0889 | 0.9681 | 0.9695 ± 0.0165 | 0.0093 |
| 3 | XGBoost | 0.6815 | 0.6928 ± 0.1288 | 0.9439 | 0.9431 ± 0.0418 | 0.0088 |
| 4 | TabPFN (local) | 0.6742 | 0.6739 ± 0.0812 | 0.9845 | 0.9846 ± 0.0030 | 0.0102 |
| 5 | CatBoost | 0.6172 | 0.6353 ± 0.0540 | 0.9594 | 0.9612 ± 0.0137 | 0.0101 |
| 6 | Random Forest | 0.4865 | 0.5034 ± 0.0793 | 0.9209 | 0.9206 ± 0.0423 | 0.0143 |
| 7 | Logistic Regression | 0.3326 | 0.3451 ± 0.1213 | 0.9224 | 0.9235 ± 0.0251 | 0.0563 |

Outer-fold PR-AUC (Part 4 after Table 1):

- Thinking-high: 0.8640, 0.7837, 0.7407, 0.9497, 0.9061 (higher than LightGBM in **5/5**)
- LightGBM: 0.7528, 0.7138, 0.5473, 0.7731, 0.6916
- Local: 0.6384, 0.6353, 0.5829, 0.7274, 0.7855 (higher than LightGBM in **2/5**, folds 3 and 5)

### Table S-CI. Stratified bootstrap 95% CIs (n_boot=2000)

| Model | PR-AUC | ROC-AUC | Brier |
| --- | --- | --- | --- |
| TabPFN (thinking-high) | 0.8553 [0.7957, 0.9131] | 0.9905 [0.9834, 0.9964] | 0.0064 [0.0052, 0.0077] |
| LightGBM | 0.6942 [0.6065, 0.7782] | 0.9681 [0.9490, 0.9831] | 0.0093 [0.0076, 0.0110] |
| XGBoost | 0.6815 [0.5881, 0.7703] | 0.9439 [0.9100, 0.9742] | 0.0088 [0.0071, 0.0106] |
| TabPFN (local) | 0.6742 [0.5864, 0.7657] | 0.9845 [0.9760, 0.9917] | 0.0102 [0.0092, 0.0113] |
| CatBoost | 0.6172 [0.5250, 0.7148] | 0.9594 [0.9398, 0.9765] | 0.0101 [0.0084, 0.0119] |
| Random Forest | 0.4865 [0.3860, 0.6034] | 0.9209 [0.8824, 0.9555] | 0.0143 [0.0137, 0.0148] |
| Logistic Regression | 0.3326 [0.2486, 0.4345] | 0.9224 [0.8966, 0.9449] | 0.0563 [0.0511, 0.0611] |

### Table S-Δ vs LightGBM

| Contrast | Δ PR-AUC (95% CI) | P(Δ ≤ 0) |
| --- | --- | --- |
| Thinking-high − LightGBM | 0.1611 (0.0984–0.2289) | 0/2000 |
| Local − LightGBM | −0.0201 (−0.0974–0.0566) | compatible with no difference (as written) |

### Table 2. Honest nested operating point (quote this)

| Model | t mean ± SD | Acc | Prec | Rec | Spec | F1 | F2 | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabPFN (thinking-high) | 0.271 ± 0.067 | 0.9915 | 0.7927 | 0.7065 | 0.9967 | 0.7471 | 0.7222 | 5076 | 17 | 27 | 65 |
| LightGBM | 0.112 ± 0.090 | 0.9878 | 0.6526 | 0.6739 | 0.9935 | 0.6631 | 0.6695 | 5060 | 33 | 30 | 62 |
| XGBoost | 0.225 ± 0.060 | 0.9875 | 0.6452 | 0.6522 | 0.9935 | 0.6486 | 0.6508 | 5060 | 33 | 32 | 60 |
| TabPFN (local) | 0.166 ± 0.020 | 0.9844 | 0.5478 | 0.6848 | 0.9898 | 0.6087 | 0.6522 | 5041 | 52 | 29 | 63 |
| CatBoost | 0.167 ± 0.040 | 0.9815 | 0.4836 | 0.6413 | 0.9876 | 0.5514 | 0.6020 | 5030 | 63 | 33 | 59 |
| Random Forest | 0.118 ± 0.013 | 0.9840 | 0.5517 | 0.5217 | 0.9923 | 0.5363 | 0.5275 | 5054 | 39 | 44 | 48 |
| Logistic Regression | 0.947 ± 0.035 | 0.9769 | 0.3654 | 0.4130 | 0.9870 | 0.3878 | 0.4025 | 5027 | 66 | 54 | 38 |

TN+FP+FN+TP = 5,185; TP+FN = 92 for each row (checked: 65+27=92, 62+30=92, etc.).

### Table 3. Optimistic pooled F1 (do not quote as nested)

| Model | t_F1 | Rec | F1 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TabPFN (thinking-high) | 0.193 | 0.8152 | 0.7979 | 5072 | 21 | 17 | 75 |
| LightGBM | 0.109 | 0.6196 | 0.6514 | 5067 | 26 | 35 | 57 |
| XGBoost | 0.203 | 0.6739 | 0.6739 | 5063 | 30 | 30 | 62 |
| TabPFN (local) | 0.119 | 0.8478 | 0.6290 | 5015 | 78 | 14 | 78 |
| CatBoost | 0.416 | 0.5326 | 0.5976 | 5070 | 23 | 43 | 49 |
| Random Forest | 0.104 | 0.5652 | 0.5361 | 5043 | 50 | 40 | 52 |
| Logistic Regression | 0.985 | 0.3696 | 0.4198 | 5057 | 36 | 58 | 34 |

Nested vs pooled recall (methods note): thinking-high 0.7065 vs 0.8152; LightGBM 0.6739 vs 0.6196; local 0.6848 vs 0.8478.

Historical **not** Version 4: local Brier 0.0673; client Brier 0.0060 / 0.0360.

---

## Part 1 inferential numbers (association)

### EPV

| Comparison | Value | File |
| --- | --- | --- |
| Events / 81 candidates | 92 / 81 ≈ 1.14 | `00_front_matter.md` W4 |
| Table 4 | 92 / 17 ≈ 5.4 | same; Part 1 Table 4 |
| Table 4b | 92 / 13 ≈ 7.1 | same; Part 1 Table 4b |
| `CKD90` Wald interval (Table 4) | 2.708–639.506 | Part 1 Table 4 caption |

### Table 4b adjusted ORs (unweighted logit; quote this)

File: `EDA_paper_figures_and_tables.md` Table 4b.

| Feature | Univ OR | Adj OR | Wald 95% CI | VIF |
| --- | ---: | ---: | --- | ---: |
| WBC | 2.090 | 1.972 | [1.667, 2.331] | 1.05 |
| eGFR | 0.469 | 0.568 | [0.449, 0.717] | 1.04 |
| LV | 2.098 | 1.832 | [1.539, 2.181] | 1.03 |
| No.of stents per lesion | 1.383 | 1.421 | [0.970, 2.080] | 3.96 |
| HbA1c | 1.282 | 0.960 | [0.724, 1.272] | 1.79 |
| NO.of vessels | 1.469 | 1.212 | [0.954, 1.539] | 1.06 |
| Total stent length | 1.378 | 1.160 | [0.777, 1.731] | 4.02 |
| Fiberinogen | 1.206 | 1.024 | [0.845, 1.240] | 1.03 |
| 1.1:1Post dilation | 0.187 | 0.152 | [0.081, 0.286] | 1.07 |
| Previous PCI | 6.485 | 6.710 | [2.884, 15.610] | 1.01 |
| Clopidogrel | 0.503 | 0.480 | [0.293, 0.787] | 1.00 |
| Diabetes | 1.889 | 1.452 | [0.795, 2.652] | 1.77 |
| PES | 2.158 | 1.734 | [0.953, 3.154] | 1.03 |

Front matter also quotes adjusted OR `1.1:1Post dilation` 0.144 and `Clopidogrel` 0.464 — **those two values are not the Table 4b cells above** (0.152 and 0.480). Treat as a numerical conflict (see ledger).

`Previous PCI` OR estimators (do not mix): Table 2 2×2 **6.49**; Table 4 univariate logit **6.46**; Table S4 joint-domain univariate **6.73**; Table 4b univariate **6.485**.

Max VIF Table 4b: 4.02 (`Total stent length`). Table 4 post-dilation VIF = ∞.

Bootstrap on Table 4b: 2,000-replicate percentile, stored in CSV (caption); primary interval is Wald.

---

## Part 2 / Part 3 extraction counts

| Claim | Value | File | Location |
| --- | --- | --- | --- |
| LOCO unique names per model | 60 (cap) | Part 2 Figure 1 table | report |
| SHAP unique names | 40 (universe) | same | report |
| FFS path lengths | lr 12; rf 12; rf_b 8; cat 4; xgb 12; xgb_b 11; lgb 5 | Part 2 Figure 1 | report |
| Strict 7×3 intersection | 0 names | Part 2 Table 4 | report |
| Union of scored names | 86 | Part 2 Table 4 | report |
| Statistical FDR names | 20 | Part 3 §1; stats notebook cell 2 | report / notebook |
| ML three-way union | 13 | Part 3 §1; stats notebook cell 2 | report / notebook |
| Intersection | 5 | same | numerical result |
| Union | 28 | same | numerical result |
| Jaccard | 5/28 ≈ 0.18; printed **0.1786** | Part 3 §2; `stats_vs_ml_comparison.ipynb` cell 4 `OUT Jaccard=0.1786` | report / notebook |
| Distinct statistical constructs (interpretation) | ~12, not 20 independent findings | Part 3 §1 | interpretation |

---

## Part 5 attribution numbers (Version 5)

Standalone file: `tabpfn_interpretability_paper_figures_and_tables.md`. Notebook `tabpfn_interpretability.ipynb` (`e356bb1`).

| Claim | Value | Location |
| --- | --- | --- |
| Train / test | 3629/64 ; 1556/28 | protocol |
| MI top (train) | `CaI` 0.022005 | Table 5 / protocol |
| `Cre` train MI | 0.000000 (51st; measured zero) | Table 5 caption |
| SFS | `WBC` 10/10 seeds | protocol / Table 2 (as reported) |
| PDP Previous PCI | +0.0137 | Part 5 PDP section (report) |
| PDP 1.1:1Post | −0.0093 | same |
| SHAP rows | 1,556 held-out; HTTP 429 ~row 550; finished local | protocol |
| k-SII row | held-out pos 20; cohort **5176**; budget 256 | §5 |
| Consensus 3/3 | `WBC`, `LV`, `eGFR` | Figure 13 / Table 5 |

### Table 5 consensus top 15

| Rank | Feature | Score | n methods | Stability | mean(\|SHAP\|) | MI |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1 | WBC | 0.9917 | 3/3 | 1.0 | 1.0202 | 0.020165 |
| 2 | LV | 0.9771 | 3/3 | 0.8 | 0.8695 | 0.012768 |
| 3 | eGFR | 0.9708 | 3/3 | 0.7 | 1.0439 | 0.009830 |
| 4 | LDL | 0.9062 | 2/3 | 0.2 | 0.4973 | 0.009893 |
| 5 | HbA1c | 0.8896 | 2/3 | 0.3 | 0.1783 | 0.004805 |
| 6 | 1.1:1Post dilation | 0.8729 | 1/3 | 0.3 | 0.6906 | 0.002637 |
| 7 | Previous PCI | 0.8604 | 2/3 | 0.5 | 0.1356 | 0.002358 |
| 8 | Fiberinogen | 0.8542 | 2/3 | 0.4 | 0.0502 | 0.003007 |
| 9 | HGB | 0.8042 | 2/3 | 0.1 | 0.0664 | 0.005327 |
| 10 | No postdilation | 0.7833 | 1/3 | 0.1 | 0.2958 | 0.002406 |
| 11 | Lesion location-Ostial | 0.7292 | 1/3 | 0.1 | 0.0222 | 0.005128 |
| 12 | History of HF | 0.7292 | 0/3 | 0.1 | 0.0483 | 0.002289 |
| 13 | Cardiogenic shock | 0.7271 | 0/3 | 0.2 | 0.0316 | 0.001431 |
| 14 | Cre | 0.7083 | 2/3 | 0.8 | 0.2449 | 0.000000 |
| 15 | CaI | 0.7042 | 2/3 | 0.0 | 0.0793 | 0.022005 |

Do not quote old 30-row Cre-leading SHAP 0.158 as this table (Part 5 caption).

---

## I. Wang 2020 comparator (derivation cohort only)

File: `wang_vlst_score.ipynb` cells 2–3; Part 4 §7.

### Points formula (cell 2)

| Variable | Points | CSV column |
| ---: | ---: | --- |
| 1 | Diabetes | `Diabetes` |
| 3 | Previous PCI | `Previous PCI` |
| 1 | Initial diagnosis-AMI | `Initial diagnosis-AMI` |
| 1 | CKD90 | `CKD90` |
| 1 | 3-vessel disease | `3-vessel disease` |
| 2 × count | stents per lesion | `No.of stents per lesion` |
| 1 | SES | `PES` |
| 4 | no post-dilation | `No postdilation` |

### Ranking

| Claim | Value | Location |
| --- | --- | --- |
| Frozen score ROC-AUC (full cohort) | 0.8013 | wang cell 2 print; Part 4 headline |
| Frozen score PR-AUC | 0.1032 | same |
| Wang published derivation c | 0.80 (95% CI 0.75–0.85) | wang cell 2 print (cited) |
| Fold-mean ROC | 0.8005 ± 0.0607 | Part 4 §7 |
| Fold-mean PR | 0.1134 ± 0.0518 | Part 4 §7 |
| Flipped post-dilation polarity ROC | 0.5084 | wang cell 2; Part 4 encoding note |
| Wang Shantou c | 0.82 (n = 2,058; **not in repo**) | Front matter; Part 4 “not external validation” | **Wang’s** external test, not this pack |
| Dangas c in Wang comparison | 0.66 | Front matter | cited, not computed here |

### Risk bins (wang cell 3 / Table S-Wang-bins)

| Category | n | % | events | observed rate | Wang published n | Wang published rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| low (≤7) | 3135 | 60.5 | 16 | 0.0051 | 3135 | 0.005 |
| intermediate (8–9) | 1577 | 30.4 | 35 | 0.0222 | 1837 | 0.022 |
| high (≥10) | 473 | 9.1 | 41 | 0.0867 | 473 | 0.087 |

Wang published n’s 3135+1837+473 = 5445 ≠ 5185. Documented in notebook output and Part 4.

---

## J. Numbers that must not be mixed

| Do not quote as | Actual meaning |
| --- | --- |
| Nested recall 0.8152 / 0.8478 | Pooled F1 (Table 3), optimistic |
| Local Brier 0.0673 | Previous dump with `balance_probabilities=True` |
| Cre SHAP 0.158 | Old 15+15 / row 5099 dump |
| k-SII row 5099 or 5093 | Superseded; live row is **5176** |
| Jaccard 5/35 | Live is 5/28 |
| Wang 0.82 as this pack’s validation | Shantou test of Wang’s score |
| Table 4 as the clinical logit | Unidentified; use Table 4b |
| Part 5 SHAP as nested-CV prediction | Attribution on a 70/30 split |
