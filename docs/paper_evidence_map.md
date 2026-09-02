# VLST paper — evidence map

**Purpose.** Pre-writing audit of every claim, number, feature set, figure and table that currently exists in this
repository, with its exact provenance. Nothing here is new analysis. Where the repository does not contain something,
this document says so explicitly rather than filling the gap.

**Revision 2.** This version applies three author scope decisions (§0) and re-resolves every code-vs-Markdown
conflict in the code's favour. Numbers that changed as a result are flagged in place.

**Revision 3.** Cohort provenance is no longer a repository-only problem. Wang et al., *Sci Rep* 2020;10:6378
(doi:10.1038/s41598-020-63455-0) — cited below as **Wang 2020** — published a Cox VLST risk score on a derivation
cohort of **5,185 ACS-PCI patients with 92 events (1.77%)**, identical to `data/raw/VLST.csv`. Several §2 / §3 / §4
gaps that this map treated as unanswerable from the repo are answered there. What remains unanswered is listed
explicitly.

**Revision 5.** Part 1–3 figures and catalogues were refreshed from the 2026-08-31 re-runs of `eda.ipynb` and
`baseline_feature_selections.ipynb` (Kaggle, paper protocol). Part 2 stored export is no longer the leaky
70/30 / F1/F2 / 185-column smoke run. ML consensus is now 13 PR-AUC three-way names; Jaccard vs FDR is
5/28 ≈ 0.18 with intersection `{WBC, eGFR, LV, HbA1c, 1.1:1Post dilation}`. Part 2 scaled width is **88**
(9-level stent encoder, OHE drop-first). Kaggle `selector_summary_long.csv` was not downloaded; Part 2
tables were reconstructed from notebook displays. Part 5 stored SHAP/PDP remain mixed; Part 4 nested CV is the 2026-09 Kaggle **local TabPFN** run (9-level encoder).

**Revision 7.** Part 4 and Part 5 were re-run on Kaggle Tesla T4 (commits `de46f92` Version 5 nested CV,
`645fb0e` Interpretability plus Version 2). **Both TabPFN arms finished** in one nested-CV notebook
(`RUN_MODELS["TabPFN"]=True` and `TabPFN (local)=True`; thinking-high constructor unchanged). Client
thinking-high **leads** the seven-model scoreboard: pooled PR-AUC **0.8553**, ROC-AUC **0.9905**, Brier
**0.0064** (best of seven); higher PR-AUC than LightGBM in **5/5** outer folds. TabPFN (local) is unchanged
in rank among classics (PR **0.6754**, ROC **0.9845**, Brier **0.0673** worst). Honest nested: thinking-high
recall **0.7065** (5076/17/27/65); LightGBM **0.6630** (5062/31/31/61); local **0.6848** (5041/52/29/63).
Part 5 met the protocol we were waiting for: 9-level encoder; MI CSV all 81 scores; SFS 10 seeds on the full
cohort; PDP `balance_probabilities=False` on n = 5,185 (binary P(y=1) ≈ 0.017–0.023, not 0.24); SHAP
**15 VLST=1 + 15 VLST=0** with client thinking-high succeeding (`Explaining all 30 rows`); k-SII still one
VLST=1 row (5099). **Reports and `paper_figures/` in both trees now match those notebooks** (B1 / B12 closed
for the PNG/CSV copy). **Still open:** OOF CSVs were written on Kaggle but **not committed** (B2); no CIs /
paired test (B3); EDA Table 4 collinearity (B4); clinical Table 1 rebuild (B7); Shantou / Cox LP / Dangas DCA
(B11). D4: quote the notebooks if a later re-run disagrees with these exports.

**Revision 6.** Superseded by Revision 7 as the *current* Part 4/5 snapshot. That revision documented the
Kaggle **local-only** nested CV (`RUN_MODELS["TabPFN"]=False`): LightGBM PR-AUC 0.6937, TabPFN (local)
0.6754 / 0.9845 / Brier 0.0673. Keep those numbers only when explicitly labelled as the six-model local-only
run. Client constructor was already present and unused in that snapshot.

**Revision 4.** Paper-style Markdown reports (both trees + `paper_results/paper_results.md`) were aligned to the
*code* on methodology. Imputers are described as inert (no missing values). Part 4 now states that GridSearch
winners are not imported. (The ~186-column / thinking-high description in that revision is superseded by
Revision 6.) Part 5 now states MI/SFS/SHAP = full cohort (the 15-row all-case SHAP slice is removed), shapiq `imputer` ≠
NaN fill, and `balance_probabilities=True` (not absolute risk). Part 2/3 now distinguish the **prior reduced export**
(column-order LOCO pool; 87.5% SHAP sample) from the **paper-protocol selector code** (full-cohort fit/val;
independent cheap-importance pools; PR-AUC only; no unused outer test; no smoke switch). Part 3 no longer says the LVEF adjusted OR “persists.” Revision 4’s “Part 4 PNGs are STALE for Brier / confusion” note is
**superseded by Revision 6**: Figures 1–3 and Tables 0–3 match the local-TabPFN dump. Section C wording (C1–C16) is rewritten in
the reports: no “protective” claims; CatBoost is GPU Plain + PRAUC; mixed d/r, three Previous-PCI ORs, and
three SES encodings are labelled rather than silently mixed; Table S2 lists all 16 pairs. PNG re-exports that
would split Figure 3 or unify SES encoding remain on the B list.

**Audit scope.** The paper-style Markdown reports, both copies of each (`paper_results/`** and `code/`**), the
root `README.md`, the raw data file `data/raw/VLST.csv`, and the executed notebook outputs. Notebook code and outputs
were extracted to plain text under `.nbdump/` for line-addressable citation; the mapping is
`code/<path>/<name>.ipynb` → `.nbdump/code__<path>__<name>.txt`.

**Conventions used below**


| Tag               | Meaning                                                                                                           |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| **[VERIFIED]**    | The Markdown claim matches the executed notebook output or the raw data.                                          |
| **[DISCREPANCY]** | The Markdown and the code disagree. **Resolved in the code's favour** (D4); the figure to use is stated in place. |
| **[STALE]**       | An exported PNG/CSV disagrees with the notebook that wrote it. Regenerate the export.                             |
| **[GAP]**         | The repository does not contain the information at all.                                                           |
| **[RISK]**        | Methodologically valid to compute, but not safe to state the way it is currently stated.                          |
| **[TODO-*]**      | A concrete task. Every one is collected in §13.                                                                   |


---

## Contents

1. [Scope in force](#0-scope-in-force)
2. [Clinical and scientific motivation](#1-clinical-and-scientific-motivation)
3. [Dataset, target, event count, prevalence](#2-dataset-target-event-count-prevalence)
4. [Variable dictionary](#3-variable-dictionary)
5. [Leakage and quasi-leakage variables](#4-leakage-and-quasi-leakage-variables)
6. [The analysis pipeline as actually implemented](#5-the-analysis-pipeline-as-actually-implemented)
7. [Data splits and validation procedures](#6-data-splits-and-validation-procedures)
8. [Every reported metric](#7-every-reported-metric)
9. [Feature sets produced by each method](#8-feature-sets-produced-by-each-method)
10. [Overlap and disagreement: statistics vs machine learning](#9-overlap-and-disagreement-statistics-vs-machine-learning)
11. [Complete figure and table inventory](#10-complete-figure-and-table-inventory)
12. [Recommended main text, supplement, and redundant items](#11-recommended-main-text-supplement-and-redundant-items)
13. [Unsupported claims, ambiguities, contradictions, missing information](#12-unsupported-claims-ambiguities-contradictions-missing-information)
14. [What is left to do](#13-what-is-left-to-do)

---

## 0. Scope in force

### 0.1 The three decisions


| #      | Decision                                                                                                                                                                                                                                                                                         | Effect on this audit                                                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D1** | `code/failed_hypothesis/`** **is out of scope** — all ten notebooks (`anomaly_detection`, `baseline_blending`, `baseline_plus_tabpfn_blending`, `ffs`, `fp_precision_mining`, `llm_tabular_small_n`, `tabpfn_5fold_fp_mining`, `tabpfn_fp_followup`, `tabpfn_oversampling`, `tabpfn_synthesis`). | The selective-reporting finding built on this directory is **withdrawn** (§12.11).                                                                                                 |
| **D2** | **TabPFN comes from exactly two notebooks:** `rating/baseline_plus_tabpfn.ipynb` (performance) and `interpretability/tabpfn_interpretability.ipynb` (interpretability). `rating/tabpfn_playground.ipynb` is out of scope.                                                                        | No other file may be cited for a TabPFN number.                                                                                                                                    |
| **D4** | **The Markdown reports remain part of the analysis and are read on their merits — but where a report and the code disagree, the code is authoritative.**                                                                                                                                         | Every `[DISCREPANCY]` below now carries an explicit resolution. Reports still count as the source for *reasoning, framing and caveats*; the notebooks are the source for *values*. |


### 0.2 In-scope notebooks


| #   | Notebook                                                      | Written up as | Export status                              |
| --- | ------------------------------------------------------------- | ------------- | ------------------------------------------ |
| 1   | `analyzes/eda.ipynb`                                          | Part 1        | verified against notebook                  |
| 2   | `modeling/interpretability/baseline_feature_selections.ipynb` | Part 2        | verified against notebook                  |
| 3   | `modeling/rating/baseline_plus_tabpfn.ipynb`                  | Part 4        | **Current** — 7-arm nested CV, both TabPFN arms (§5.8, `de46f92`) |
| 4   | `modeling/interpretability/tabpfn_interpretability.ipynb`     | Part 5        | **Current** — 15+15 SHAP, empirical PDP, 81-row MI (`645fb0e`) |
| 5   | `modeling/rating/baseline_tssi_leakage.ipynb`                 | Part 4 supp.  | stored metrics → Table S-TSSI              |
| 6   | `modeling/rating/baseline_without_tssi.ipynb`                 | Part 4 supp.  | stored metrics → Table S-TSSI              |
| 7   | `modeling/preprocessing/preprocessing.ipynb`                  | **nothing**   | artefacts unused by any analysis (§12.5)   |
| 8   | `modeling/rating/wang_vlst_score.ipynb`                       | Part 4 supp.  | frozen Wang integer score → Table S-Wang   |


Eleven notebooks on disk are excluded by D1–D2 and are not audited (ten under `failed_hypothesis/`, plus `tabpfn_playground.ipynb`). D4 is a reading rule, not an exclusion.

### 0.3 Part 3 now has generating code

Part 3, the statistics-vs-ML comparison, previously had **no notebook and no script**. That is closed.

Generating code: `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb`.
It recomputes the overlap from the verified
§8.1 statistical FDR set and the §8.6 ML consensus set, asserts Jaccard = 5/28 ≈ 0.1786 with intersection
`{WBC, eGFR, LV, HbA1c, 1.1:1Post dilation}`, and writes Figures 1–4 plus Tables 1–4 to both
`paper_results/03_stats_vs_ml/paper_figures/` and `code/analyzes/stats_vs_ml/paper_figures/`.
**[TODO-P3 — closed]**

---

## 1. Clinical and scientific motivation

### What the repository actually states

The **only** motivation text anywhere in the repository is the root `README.md`, in full:

> `# vlst`
> `Personalized Risk prediction  for very late Stent Thrombosis (VLST) after PCI in ACS using Machin Learning`

Source: `README.md` lines 1–2 (two typographical errors in the original: "Machin", double space).

From this single line the repository commits to three scope statements that the paper must honour:


| Element          | Repository position                                   | Source         |
| ---------------- | ----------------------------------------------------- | -------------- |
| Clinical outcome | Very late stent thrombosis (VLST)                     | `README.md` L2 |
| Clinical setting | After PCI in acute coronary syndrome (ACS)            | `README.md` L2 |
| Stated goal      | *Personalised risk prediction* using machine learning | `README.md` L2 |


The five report Markdown files each open with a "Cohort / protocol" paragraph, but these are methodological
preambles, not motivation. No file in *this* repository states a research question. **Wang 2020 does**, and it is
the paper that assembled this cohort: derive and validate a clinical score that identifies ACS-PCI patients at
high risk of VLST so that intensive follow-up and treatment choices after year 1 can be tailored.

### Clinical motivation that is now sourced (Wang 2020)


| Element                  | What Wang 2020 states                                                                                                                                   | Use in our manuscript                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| VLST definition          | Thrombosis **> 1 year** after stent implantation; **ARC 2007 definite ST**, angiographically confirmed                                                  | Cite; this is our target (§2.3)                                                                   |
| Why it matters           | ~20% of new MI after index PCI; adjusted mortality **4-fold** higher than MI unrelated to a previously stented site                                     | Cite Lemesle et al. via Wang                                                                      |
| Existing score           | Dangas LST score (2012) also used for VLST, **c-statistic 0.66**; Wang's own 8-variable Cox score **c = 0.80 / 0.82** (derivation / Shantou validation) | Frozen integer points now scored on these 5,185 rows (Part 4 S-Wang; §5.10). Do not write as if no VLST score exists |
| Intended user / decision | Risk-stratify after PCI; inform monitoring and therapy **more than 1 year** after the index procedure                                                   | Keep; our models are the same clinical task on the same patients                                  |
| Pre-registration         | Cohort study **NCT03491891**; ethics NO. 2013-256, both hospitals                                                                                       | The *cohort* was pre-registered. The TabPFN analysis was not. Do not imply otherwise              |


### [GAP] Motivation this repository still does not contain

- Why a tabular foundation model (TabPFN) is a reasonable choice for a rare-event tabular problem.
- What "personalised" means operationally here, given that the deployed artefact would be a single global model.
- Any citation file in the repo itself — Wang 2020 and its reference list now supply the clinical ones; TabPFN still needs its own.

---

## 2. Dataset, target, event count, prevalence

### 2.1 Verified from the raw file


| Quantity                       | Value                    | Source                                                                                           |
| ------------------------------ | ------------------------ | ------------------------------------------------------------------------------------------------ |
| Data file                      | `data/raw/VLST.csv`      | repository                                                                                       |
| Rows (patients)                | **5,185**                | raw file; `.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L223, L267 |
| Target column                  | `Stent thrombosis` (0/1) | `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L166                                  |
| Positive events                | **92**                   | `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L608                                  |
| Prevalence                     | **0.0177** (1.77%)       | `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L674                                  |
| Columns dropped as identifiers | `NO.`, `Name`            | `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L167                                  |
| Analysis features after drops  | **81**                   | `.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L223                 |
| Missing values                 | **0** in every column    | `paper_results/01_eda/paper_figures/domain_feature_map.csv` ("Missing %" = 0.0 for all 82 rows)  |


**[VERIFIED]** All five Markdown reports state n = 5,185, 92 events, prevalence 0.0177. This is internally
consistent and matches the raw file. It is the one set of numbers with no ambiguity anywhere in the repository.

**How far D4 actually bites.** Parts 1, 2 and 5 were spot-checked report-vs-notebook and **agree**: `LV`
Cohen d = 1.127413 with q = 3.262999e-16 appears identically in `.nbdump/code__analyzes__eda.txt` L1961/L3772 and
in `paper_table1_continuous_univariate.csv`; Part 2's consensus tables at
`.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L1468–1535 reproduce Part 2's
Tables 1–4 exactly. So D4 does **not** invalidate Parts 1–3 wholesale. **Part 4 and Part 5 Markdown /
`paper_figures/` now match the Revision 7 notebooks** (§5.8–5.9, §7.1–7.4): seven-arm thinking-high-first
scoreboard; 15+15 SHAP with client thinking; empirical-prior PDP. OOF prediction CSVs remain uncommitted (B2).

### 2.2 Events per variable


| Comparison                                   | Value              | Consequence                                                                                                        |
| -------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------ |
| Events / candidate features (81)             | 92 / 81 ≈ **1.14** | Far below any conventional threshold                                                                               |
| Events / multivariable-model covariates (17) | 92 / 17 ≈ **5.4**  | Below the conventional EPV ≥ 10 rule                                                                               |
| Events in the stored Part 2 test fold        | **28**             | Stored figures scored on 28 events. Current code uses a full-cohort val slice (~18 events at `INNER_VAL_SIZE=0.2`) |
| Events per outer CV fold (Part 4)            | 18, 18, 18, 19, 19 | `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L1030–1060                                              |


**[RISK]** The EDA Markdown says the multivariable model "is sparse and intended for screening/confounding context"
(`paper_results/01_eda/EDA_paper_figures_and_tables.md`) — a fair caveat. **EPV = 92 / 17 ≈ 5.4** is now stated in
Part 0 and next to Table 4 / Figure S4 adjusted ORs. **[TODO-EPV — closed in reports]**

### 2.3 Target definition and recruitment — filled by Wang 2020

`data/raw/VLST.csv` is Wang 2020's **derivation cohort**. Counts match exactly (5,185 patients, 92 events, 1.77%).
The Methods of that paper are therefore the data dictionary for this file.


| Question                           | Answer                                                                                                                                                                                                                                          | Source                                         |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| What is `Stent thrombosis = 1`?    | **ARC 2007 definite** stent thrombosis, **> 1 year** after implantation, **confirmed by coronary angiography**. Probable/possible ST were not counted. Two interventional cardiologists reviewed every angiogram; a third resolved disagreement | Wang 2020 Methods, "Definitions and endpoints" |
| Time window                        | VLST = ST **more than one year** after coronary stent implantation. Median time PCI → VLST = **697 days** (derivation), 803 days (validation)                                                                                                   | Wang 2020 Results                              |
| Consecutive or selected?           | **Consecutive** ACS patients ≥ 18 years undergoing PCI                                                                                                                                                                                          | Wang 2020 Methods                              |
| Centre / country / period          | The First Hospital of **Jilin University**, China; **1 January 2014 – 1 June 2015**                                                                                                                                                             | Wang 2020 Methods                              |
| Flow into the 5,185                | 6,038 eligible → exclude 236 in-hospital deaths, 413 refused follow-up, 204 lost to follow-up → **5,185**                                                                                                                                       | Wang 2020 Results                              |
| Follow-up                          | Median **1,502 days**. All patients had DAPT for ≥ 1 year; continuation after year 1 at the treating physician's discretion. 24-hour online consultation; suspected cases transferred to a Chest Pain Center                                    | Wang 2020 Methods                              |
| Ethics / consent                   | Written informed consent. Ethics NO. **2013-256** (Medical Ethics Committee of The First Hospital of Jilin University). Helsinki 1964. Registered **NCT03491891**                                                                               | Wang 2020 Methods                              |
| External cohort (not in this repo) | 2,438 eligible at First Affiliated Hospital of **Shantou University**, same window → 2,058 enrolled, 1.70% VLST. Used to validate Wang's Cox score, **not present in** `VLST.csv`                                                               | Wang 2020 Results                              |
| Multiple events per patient        | Not stated. Analysis is one row per patient; treat as a single binary endpoint                                                                                                                                                                  | Wang 2020                                      |


**Still missing from both the repository and Wang 2020:** an explicit statement of how patients with more than one
definite VLST were handled (likely one row, first event). That is a one-sentence Methods clarification, not a
blocker.

**Consequence for §4.2.** This is a **prospective consecutive cohort with complete follow-up**, not a case-control
sample. The 1.77% figure is the published incidence among patients who survived to discharge and completed
follow-up. It is **not** a sampling fraction. See the revised §4.2.

---

## 3. Variable dictionary

### 3.1 Domain assignment as used by the analysis

The EDA notebook assigns all 82 columns to eight clinical domains plus one "time-at-risk" domain. This is the
repository's own grouping and is the closest thing to a data dictionary that exists.
Source: `paper_results/01_eda/paper_figures/domain_feature_map.csv`.


| Domain                                   | n   | Members                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------------- | --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Demographics / lifestyle                 | 4   | `Age`, `Men`, `Current smoker`, `Current drinking`                                                                                                                                                                                                                                                                                                                                                       |
| Comorbidities / history                  | 10  | `Diabetes`, `Hypertension`, `HL`, `Stroke/TIA`, `History of peripheral vascualr disease`, `Chronic renal insufficiency`, `History of HF`, `Previous CABG`, `Previous PCI`, `Previous MI`                                                                                                                                                                                                                 |
| Presentation / ACS                       | 5   | `Initial diagnosis-AMI`, `STEMI`, `NSTEMI`, `UA`, `Cardiogenic shock`                                                                                                                                                                                                                                                                                                                                    |
| Anatomy / lesion                         | 23  | `NO.of vessels`, `Multi-vessel CAD`, `Single-vessel disease`, `2-vessel disease`, `3-vessel disease`, `Chronic total occlusion`, `Moderate/severe calcification`, `Moderate/severe tortuosity`, `Lesion location-Ostial`, `Proximal`, `Bifurcation`, `Visual thrombus`, `Aneurysm`, `Vessel dialation`, `Ulceration`, `P-LM`, `P-LAD`, `P-LCX`, `P-RCA`, `Pre-TIMI flow-3`, `TIMI-2`, `TIMI-1`, `TIMI-0` |
| Procedural / stent                       | 16  | `Staged PCI`, `Thrombus aspiration`, `Slow flow`, `No reflow`, `Dissection`, `Stent type-SES`, `PES`, `ZES`, `EVS`, `Min-stent diameter`, `Max-stent diameter`, `Total stent length`, `Stent release pressure`, `No.of stents per lesion`, `stent overlap`, `1.1:1Post dilation`, `No postdilation`                                                                                                      |
| Cardiac function                         | 2   | `LV`, `LVEF`                                                                                                                                                                                                                                                                                                                                                                                             |
| Laboratory                               | 16  | `CaI`, `WBC`, `HGB`, `Platelet`, `Cre`, `eGFR`, `CKD60`, `CKD90`, `CKD5`, `TCL`, `LDL`, `HDL`, `TG`, `Fast-Glu`, `HbA1c`, `Fiberinogen`                                                                                                                                                                                                                                                                  |
| Medications (1 y)                        | 4   | `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT`                                                                                                                                                                                                                                                                                                                                                           |
| Time-at-risk (excluded from risk models) | 1   | `Time since stent implantation`                                                                                                                                                                                                                                                                                                                                                                          |


Note: the CSV lists `1.1:1Post dilation` and `No postdilation` within the 16 procedural entries; the domain file
counts 82 rows total including the target-adjacent time column.

### 3.2 Variables that carry most of the reported signal

These four appear at the top of nearly every ranking in the repository. Their empirical behaviour, computed
directly from `data/raw/VLST.csv`:


| Variable | Non-VLST mean (SD)   | VLST mean (SD)       | Univariate ROC-AUC | Notes                                                                                               |
| -------- | -------------------- | -------------------- | ------------------ | --------------------------------------------------------------------------------------------------- |
| `WBC`    | 8.75 (3.24)          | 12.49 (3.92)         | 0.784              | White cell count                                                                                    |
| `LV`     | 44.55 (4.04)         | 49.11 (4.23)         | 0.802              | Integer, range 37–73. **Absent from Wang 2020 Table 1.** Units and exact meaning still undocumented |
| `eGFR`   | 120.03 (34.10)       | 95.88 (19.63)        | 0.711              | Estimated GFR; Wang dichotomised at 90 mL/min/1.73 m²                                               |
| `LVEF`   | 55.15 (4.52) in Wang | 54.55 (3.68) in Wang | 0.564              | Ejection fraction (%). Wang 2020 Table 1 matches these means                                        |


A plain 5-fold cross-validated logistic regression on `WBC + LV + eGFR` **alone** reaches ROC-AUC 0.894 /
PR-AUC 0.116. The full models reach ROC-AUC 0.925–0.988. Most of the discriminative signal in this dataset lives
in these three columns.

### 3.3 Variables whose meaning is not documented in the repository — now checked against Wang 2020


| Variable                                       | Status after Wang 2020                                                                                                                                                                                                                                                                                                                                                                                                            |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LV`                                           | **Still undocumented.** Not in Wang Table 1. Almost certainly a left-ventricular linear dimension in mm (integers 37–73, correlates −0.41 with LVEF). This is a **main ML finding** and its definition must still be obtained from the data owners.                                                                                                                                                                               |
| `CaI`                                          | **Still undocumented.** Expanded nowhere. Called "Calcium index" once in the Part 5 Markdown, with no definition. Wang Table 1 reports **peak troponin I** (37.37 ± 61.64 vs 40.55 ± 72.25) which is not obviously `CaI`. Do not equate them without checking.                                                                                                                                                                    |
| `TCL`                                          | **Solved.** Wang 2020 abbreviation list: **total cholesterol**, mmol/L. Table 1: 4.57 ± 1.05 vs 4.69 ± 1.31.                                                                                                                                                                                                                                                                                                                      |
| `HL`                                           | **Solved.** Wang Table 1 "Dyslipidaemia" 1,612 (31.65%) vs 28 (30.43%) — this is `HL` (hyperlipidaemia).                                                                                                                                                                                                                                                                                                                          |
| `Fiberinogen`                                  | **Solved as a name.** Misspelling of fibrinogen; Wang: **g/L**, 3.17 ± 0.88 vs 3.37 ± 1.01.                                                                                                                                                                                                                                                                                                                                       |
| `History of peripheral vascualr disease`       | Misspelling of "vascular". Wang: History of PVD.                                                                                                                                                                                                                                                                                                                                                                                  |
| `Vessel dialation`                             | Misspelling of "dilatation" / ectasia. Wang: "Vessel ectasia".                                                                                                                                                                                                                                                                                                                                                                    |
| `Stent release pressure`                       | **Solved.** Wang: **atm**, 13.95 ± 2.99 vs 13.93 ± 2.80.                                                                                                                                                                                                                                                                                                                                                                          |
| `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT` | **Mostly solved.** All patients received DAPT for **at least one year**. Continuation after year 1 was at the treating physician's discretion. Wang Table 1 "DAPT" is **DAPT during follow-up** (44.37% vs 38.04%, p = 0.226) — i.e. persistence beyond the mandated year, **not** a baseline discharge prescription. These four columns are **post-baseline** and must not be treated as index-PCI covariates without saying so. |
| `Stent type-SES`                               | **Partially solved, encoding still a mess.** Wang treated SES as a **binary class flag** (sirolimus-eluting stent; 68.76% vs 82.61%). In this repository the same column holds **106 free-text brand strings**. `wang_vlst_score.ipynb` uses the **`PES` flag**, whose counts match Wang Table 1 SES. Part 1/2 use a 9-level encoder; stored Part 4/5 still one-hot the raw strings (§12.5). |
| **Laboratory and echo timing**                 | **Narrowed, not closed.** See §4.3.                                                                                                                                                                                                                                                                                                                                                                                               |


**Do not copy Wang Table 1's post-dilation coding blindly.** Wang reports "No post-dilation" in 14/92 VLST cases (15.22%) versus 2,496/5,093 controls (49.01%), with multivariable HR 0.145 — i.e. coded as *protective* — while the Discussion treats no post-dilation as a *risk* factor and assigns it 4 score points. In `VLST.csv` the 14 VLST cases sit on `1.1:1Post dilation` = 1, and `No postdilation` has OR 5.355. The published table and our file disagree on the label even though the 14/78 split of the 92 events is the same split. Rebuild Table 1 from the CSV.

### 3.4 Structurally dependent variable groups (must be handled as blocks, not as independent covariates)

Verified directly from the raw data:


| Block                                                                        | Relationship                                                                                                                                                                                                                                                                       | Evidence                          |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `1.1:1Post dilation` / `No postdilation`                                     | **Exact complements.** Cross-tabulation is perfectly anti-diagonal: 2,510 / 0 / 0 / 2,675. One is `1 −` the other.                                                                                                                                                                 | computed from `data/raw/VLST.csv` |
| `Multi-vessel CAD` / `Single-vessel disease`                                 | **Exact complements.** 3,054 / 0 / 0 / 2,131.                                                                                                                                                                                                                                      | computed from `data/raw/VLST.csv` |
| `eGFR` / `CKD5` / `CKD90` / `CKD60`                                          | `CKD5` is the CKD **stage** (1–5) and is a deterministic step function of `eGFR` (stage 1: eGFR ≥ 90, n = 4,293; 2: 60–90, n = 716; 3: 30–60, n = 158; 4: 15–30, n = 15; 5: < 15, n = 3). `CKD90 = 1[eGFR < 90]`, `CKD60 = 1[eGFR < 60]`. All four encode one underlying quantity. | computed from `data/raw/VLST.csv` |
| `NO.of vessels` / `Single-` / `2-` / `3-vessel disease` / `Multi-vessel CAD` | Five encodings of one anatomical count.                                                                                                                                                                                                                                            | domain map + raw data             |
| `Stent type-SES` / `PES` / `ZES` / `EVS`                                     | Brand string and drug-class flags describe the same device.                                                                                                                                                                                                                        | domain map                        |
| `Pre-TIMI flow-3` / `TIMI-2` / `TIMI-1` / `TIMI-0`                           | Four dummies of one ordinal flow grade.                                                                                                                                                                                                                                            | domain map                        |
| `LV` / `LVEF`                                                                | Spearman/Pearson correlation −0.41.                                                                                                                                                                                                                                                | computed from `data/raw/VLST.csv` |


**Why this matters:** the exploratory multivariable model in EDA Table 4 contains **both** members of the
post-dilation complement pair **and** three encodings of renal function simultaneously (see §12.1).

---

## 4. Leakage and quasi-leakage variables

### 4.1 `Time since stent implantation` — confirmed structural leakage, correctly excluded

**How the Markdown describes it.** All five reports flag it as a "time-at-risk / follow-up column, not a baseline
predictor" and confirm it is dropped before modelling. Example:
`paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md` L5.
It **is** retained in the EDA univariate screen and reported as the strongest hit
(Mann–Whitney r = −0.170, p = 1.70e-34, q = 4.07e-33), labelled "Time-at-risk / structural"
(`paper_results/01_eda/paper_figures/paper_table1_continuous_univariate.csv` row 1).

**[VERIFIED]** The drop is real and appears in every modelling notebook:
`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L167;
`.nbdump/code__modeling__preprocessing__preprocessing.txt`;
`.nbdump/code__modeling__interpretability__tabpfn_interpretability.txt`.

**The mechanism, documented inside the repository.** `code/modeling/rating/baseline_tssi_leakage.ipynb` contains an
explicit written diagnosis (`.nbdump/code__modeling__rating__baseline_tssi_leakage.txt` L846–902):

- Every control has `Time since stent implantation` ≥ **1,241** days (min 1,241, max 1,605).
- Cases include many values **< 1,241** (min 380).
- The rule "time < 1,241 → predict thrombosis" produces **zero** false positives among controls.
- The notebook concludes: *"For non-events, 'time since stent implantation' behaves like a narrow follow-up window
… For events, it behaves like time from implant to thrombosis … the column is not a clean 'baseline predictor of
VLST'; it mixes different time definitions for cases vs controls. That is structural / temporal leakage"*
(L888).

**Quantified impact — the with/without contrast is strong supplementary material:**


| Model               | With TSSI (ROC-AUC / PR-AUC) | Without TSSI, same single split (ROC-AUC / PR-AUC) |
| ------------------- | ---------------------------- | -------------------------------------------------- |
| Logistic regression | 0.9990 / 0.9575              | 0.9171 / 0.5077                                    |
| Random forest       | 0.9993 / 0.9680              | 0.9338 / 0.4700                                    |
| CatBoost            | 0.9995 / 0.9773              | 0.9669 / 0.6582                                    |
| XGBoost             | 0.9987 / 0.9609              | 0.9380 / 0.6118                                    |
| LightGBM            | 0.9989 / 0.9708              | 0.9483 / 0.6018                                    |


Sources: `.nbdump/code__modeling__rating__baseline_tssi_leakage.txt` L797–817;
`.nbdump/code__modeling__rating__baseline_without_tssi.txt` L906–924.

**[TODO-LEAK — closed]** `baseline_tssi_leakage.ipynb` and `baseline_without_tssi.ipynb` are now cited in
Part 4 (Methods note + Supplementary Table S-TSSI / Figure S-TSSI). Nothing was re-run — values are the
stored test-set metrics, rebuilt into a table by `code/modeling/rating/rebuild_tssi_leakage_table.py`.

### 4.2 The TSSI pattern is time-to-event encoded as a binary label, not case-control sampling

**Revision 1 interpretation (withdrawn).** The min-control = 1,241 / min-case = 380 pattern was read as evidence that
controls were *selected* for surviving past a fixed horizon, i.e. a case-control frame, making 1.77% a sampling
artefact.

**What Wang 2020 actually describes.** Consecutive ACS-PCI patients, followed for a median 1,502 days, with VLST
defined as angiographic ST **after day 365**. In that design the column must behave exactly as the leakage notebook
observed:


| Row type             | What `Time since stent implantation` is | Why the range looks like that                                                                           |
| -------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| VLST = 1 (n = 92)    | Time from index PCI **to the event**    | Must be > 365 days; observed min 380, median 697 (Wang)                                                 |
| VLST = 0 (n = 5,093) | Duration of **event-free follow-up**    | Completers only (204 lost-to-follow-up already dropped); observed min 1,241, median of the cohort 1,502 |


That is the standard layout of a binary-ified survival dataset, not control sampling. **The 1.77% is the published
incidence in this complete-follow-up derivation cohort** (Wang: 1.77% derivation, 1.70% Shantou, P = 0.829).
PR-AUC, precision, PPV, Brier and calibration are prevalence-dependent, but the relevant prevalence is this
cohort's incidence among completers — not an unknown sampling fraction.

**What remains true, and is still leakage for a classifier.** Using follow-up length / time-to-event as a
*predictor* of the event is illegitimate, which is why every modelling notebook drops the column (§4.1). Wang
analysed the same data with **Cox regression**, which is the design-correct model. Recoding the endpoint as a
binary label and then fitting classifiers discards censoring structure (already reduced by dropping 204
lost-to-follow-up) and is a methodological choice this paper must disclose, not a hidden case-control sample.

**[TODO-LEAK — closed]** The with/without-TSSI contrast is in the Part 4 supplement, framed as "why a follow-up-time
column cannot enter a classifier," not as proof of case-control sampling.

### 4.3 [RISK — narrowed by Wang 2020] Laboratory and echocardiographic measurement timing

Dropping `Time since stent implantation` removes the *explicit* time variable. It does not by itself prove that
the remaining variables are index-PCI measurements. The empirical pattern is still striking:


| Variable | Non-VLST | VLST  | Standardised difference          | Univariate AUC | In Wang 2020 Table 1?                                   |
| -------- | -------- | ----- | -------------------------------- | -------------- | ------------------------------------------------------- |
| `WBC`    | 8.75     | 12.49 | MW r = 0.130, Δmean = +3.75      | 0.784          | **Yes — identical means** (8.75 ± 3.24 vs 12.49 ± 3.92) |
| `LV`     | 44.55    | 49.11 | Cohen d = 1.127, Δmean = +4.56   | 0.802          | **No**                                                  |
| `eGFR`   | 120.03   | 95.88 | Cohen d = −0.712, Δmean = −24.15 | 0.711          | Yes, as the dichotomous eGFR < 90                       |


Wang 2020 presents WBC, LVEF, lipids, fibrinogen, HbA1c and eGFR as **"baseline, procedural characteristics, and
laboratory test results"** of a Cox model whose covariates are taken at the index PCI. The whole published analysis
is unintelligible if those labs were drawn at VLST presentation. That is strong evidence they are **index-admission
values**, and it is how the manuscript should describe them, with one citation.

Two residual problems, both now specific rather than open-ended:

1. **Wang 2020 deliberately left WBC out of the score.** Quote: *"Some variables were excluded, such as leukocyte
  count (for which there was insufficient evidence to definitively conclude that the increased count was not owing
   to infection or other factors)."* Our models use WBC as a top feature. That is a scientific disagreement with the
   original investigators on the same table, and it must be discussed — not treated as a newly discovered predictor.
2. `LV` **is not in Wang Table 1 at all.** Its +4.6 mm case-control gap is therefore not vouched for by the published
  baseline table. Until the data owners name the measurement and its timing, `LV` cannot be a headline "risk
   marker" in the same sentence as WBC and eGFR.

The event-time leakage hypothesis for the *published* labs is no longer the leading explanation. It is not fully
killed for `LV`. **[TODO]** Confirm `LV` definition and timing with the data owners (A4). Disclose the WBC
exclusion from Wang's Cox model in Discussion.

### 4.4 Analysis-induced leakage in Part 2 (feature selection)

**Code fix applied; re-run stored 2026-08-31.** The stored Part 2 figures/tables and the Part 3 consensus
come from the paper-protocol Kaggle run: **full cohort** (no parked 70/30 test), **PR-AUC only**, LOCO / SHAP / FFS
**independent** (each takes its own cheap fit-slice importance pool). Budget: top-20, SHAP 40/40/3, LOCO cap 60,
FFS `24 × 12` with early stop, 400 boosting rounds. `USE_CACHE=False`. Split: fit 4,148 (74 events) / val 1,037
(18 events). Scaled width **88** after the shared 9-level stent encoder.

Historical leak, for provenance of the *previous* figures (replaced):


| Selector       | Evaluation target (stored run)                                                                                                               | Source                                                                                 |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| LOCO           | `metric_score(y_test, …)` — the 1,556-row / 28-event hold-out                                                                                | `.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L521, L532 |
| Coalition SHAP | `metric_score(y_exp, …)` where `y_exp` are selected **test** rows; the candidate universe is the LOCO top-24, itself test-derived            | same file, L556, L561–567, L584                                                        |
| FFS            | Honest inner hold-out (20% of train, ≈ 726 rows, ≈ 13 events) — **but** restricted to the LOCO top-30 pool, so it inherits the contamination | same file, L621–634, L647                                                              |


SHAP and FFS no longer read LOCO’s selected names. Each selector builds its own cheap fit-slice importance
pool (LOCO 60 / SHAP 40 / FFS 24). The historical table above is the *replaced* leaky export.

Part 4 nested CV now uses the shared 9-level encoder (Kaggle run). Classics one-hot that column without
`drop="first"` (~89). Part 2/3 catalogues remain the 88-column drop-first view. They are no longer the leaky
28-event shortlist.

### 4.5 Non-leakage but comparability-breaking design choices

- **Feature-selection ranking in Part 5 uses the full cohort.** Mutual information and stability selection run on
`X_all, y_all` (all 5,185 rows). The report acknowledges this ("should not be reused as a leakage-free feature mask").
Keep that caveat. (A train-only ranking in this notebook was applied briefly and **reverted** so TabPFN interpretability
code stays as stored.)
- **No feature selection feeds the Part 4 models.** Nested-CV uses all 81 features (IDs and TSSI dropped). Part 2
/ Part 5 selection leakage does **not** contaminate the headline Part 4 metrics. Stated explicitly in the Part 4
protocol paragraph.

---

## 5. The analysis pipeline as actually implemented

### 5.1 Notebook inventory (in scope only)


| Stage                                | Notebook                                                                             | Reported in                            |
| ------------------------------------ | ------------------------------------------------------------------------------------ | -------------------------------------- |
| Statistical EDA                      | `code/analyzes/eda.ipynb`                                                            | Part 1                                 |
| Classic-ML feature selection         | `code/modeling/interpretability/baseline_feature_selections.ipynb`                   | Part 2                                 |
| Stats-vs-ML comparison               | `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb`                              | Part 3                                 |
| Nested-CV baselines + TabPFN         | `code/modeling/rating/baseline_plus_tabpfn.ipynb`                                    | Part 4                                 |
| Wang 2020 integer score (frozen)     | `code/modeling/rating/wang_vlst_score.ipynb`                                         | Part 4 Table S-Wang                    |
| TabPFN interpretability              | `code/modeling/interpretability/tabpfn_interpretability.ipynb`                       | Part 5                                 |
| Leaky baselines (with TSSI)          | `code/modeling/rating/baseline_tssi_leakage.ipynb`                                   | Part 4 Table S-TSSI                    |
| Baselines without TSSI, single split | `code/modeling/rating/baseline_without_tssi.ipynb`                                   | Part 4 Table S-TSSI                    |
| Preprocessing artefacts              | `code/modeling/preprocessing/preprocessing.ipynb`                                    | not reported; artefacts unused (§12.5) |


Excluded by D1–D2 and not audited: the ten `code/failed_hypothesis/*.ipynb` and `rating/tabpfn_playground.ipynb`.

**Selective-reporting finding withdrawn.** Revision 1 raised a `[RISK — selective reporting]` because exploratory
notebooks were unreported. D1–D2 declare those out of scope, so the previous framing — that the
reported TabPFN advantage "cannot be distinguished from selection over many attempts" — is withdrawn. See §12.11
for what remains of it (one sentence, not a table).

### 5.2 EDA and univariate screening (Part 1)

- **Test selection rule.** Welch t-test when |skew| ≤ 1 **and** excess kurtosis ≤ 3; Mann–Whitney U otherwise.
Applied to 24 continuous variables. **[VERIFIED]** against
`paper_results/01_eda/paper_figures/paper_table_test_rationale.csv` — every row obeys the stated rule.
- **Binary screening.** χ² or Fisher exact across 58 binary variables, reporting OR, RR, φ.
- **Categorical screening.** One χ² for `Stent type-SES` after collapsing levels with n < 30 to `other`, leaving
9 levels (χ² = 44.90, df = 8, Cramér's V = 0.093, p = 3.85e-07).
- **Multiplicity.** Benjamini–Hochberg, applied **within each of the three families separately** (24 continuous,
58 binary, 1 categorical), not globally. `Time since stent implantation` is included in the continuous family,
which shifts every other continuous q-value.

### 5.3 "Bivariate" feature extraction

**Resolved.** No extra extraction step. Three layers already exist in `eda.ipynb` and are now named in Part 1 §6:

- **Predictor × VLST** — the FDR screens of §5.2 (called “univariate” because each test has one predictor). This is the discovery catalogue.
- **Feature × feature** — Pearson/Spearman heatmaps (`03_`*, now Supplementary Figure S5) and domain Spearman clustermaps (Figure S2). Multicollinearity / clustering, not a second FDR set.
- **Pair × VLST** — the limited interaction screen of §5.5 (Table S2).

The correlation heatmap is admissible as bivariate analysis of *feature–feature* structure. It is not a substitute for the Welch / Mann–Whitney / Fisher screens.

### 5.4 Multivariable logistic regression (Part 1, Table 4)

Implementation, from `.nbdump/code__analyzes__eda.txt`:


| Element            | Value                                                                                                             | Line                    |
| ------------------ | ----------------------------------------------------------------------------------------------------------------- | ----------------------- |
| Estimator          | `statsmodels.Logit` unweighted MLE (sklearn `class_weight` removed from this table)                               | `eda.ipynb` 10f / 10g-3 |
| Regularisation     | None (unpenalised MLE); previously sklearn `C=1e6` near-unpenalised ridge                                         | superseded              |
| Class weighting    | **Not used** for ORs. Reserved for Part 2/4 predictive models                                                     | —                       |
| Covariate entry    | All FDR-significant continuous features, plus FDR-significant binaries **truncated to the first 8**               | unchanged               |
| Continuous scaling | per 1 SD                                                                                                          | Table 4 header          |
| Inference          | Wald SE, z, p, Wald 95% CI; LR vs intercept-only; 2,000-replicate percentile bootstrap of the same unweighted fit | `N_BOOT = 2000`         |
| Excluded           | `Time since stent implantation`                                                                                   | Table 4 caption         |


**[CLOSED]** Inference is now unweighted `statsmodels.Logit`: Wald SE, z, p, Wald 95% CI, LR vs intercept-only,
plus a 2,000-replicate percentile bootstrap of the same unweighted fit. Re-run `eda.ipynb` cells 10f / 10g-3 to
refresh Table 4 numbers. Profile-likelihood intervals are still not computed (Wald is the conventional default).

**[CLOSED for inference / keep for prediction]** `class_weight="balanced"` is **not** used in Table 4. It remains
appropriate in Part 2/4 *classifiers*. Imbalance is handled there by class weights, not SMOTE.

**[RISK]** The 8-binary cap is an undocumented model-specification rule. `Multi-vessel CAD` and
`Single-vessel disease` were FDR-significant but excluded purely by that cap. The Markdown never mentions it.

### 5.5 Domain analysis and interaction screen (Part 1, supplementary)

- Per-domain sparse logistic models: core demographics plus up to five non-redundant domain representatives,
selected by correlation clustering; bootstrap 95% CIs (`.nbdump/code__analyzes__eda.txt` L4655, L4755, L4781).
- Joint cross-domain model with 12 covariates (`domain_joint_multivariable_or.csv`).
- Interaction screen: **16 hand-picked, clinically motivated pairs**, likelihood-ratio test against the
main-effects model, BH-FDR **within those 16 only**. LR tests now use the same unweighted `statsmodels.Logit`
as the OR tables.

### 5.6 Classic ML baselines

Two distinct baseline exercises exist and must not be conflated:

1. **Single-split baselines** (`baseline_tssi_leakage.ipynb`, `baseline_without_tssi.ipynb`): 70/30 split,
  GridSearchCV, optional SMOTE, 7 models incl. decision tree and Gaussian NB. Not reported in any Markdown.
2. **Nested-CV baselines** (`baseline_plus_tabpfn.ipynb`, Part 4): 5 outer × 4 inner folds, 5 classic models
  + **two** TabPFN arms, **no hyperparameter tuning** of the classics (see §6.3).

### 5.7 Classic ML feature selection (Part 2)

- Feature view: shared 9-level stent encoder, then OHE drop-first + scale → **88 columns**
(Kaggle log: `scaled features: 88`; `Stent brand: 106 raw strings -> 9 levels`).
- **Stored export (2026-08-31 paper protocol):** top-20, SHAP universe 40, LOCO cap 60, FFS pool 24 × 12
with early stop, PR-AUC only, independent selectors, fit/val 4148/1037.
- Seven classic models: `lr`, `rf`, `rf_b`, `cat`, `xgb`, `xgb_b`, `lgb`. TabPFN was optional and not in this run.
- One objective: `pr_auc`.

**[CLOSED]** The prior reduced dump set `order = list(range(n))` then truncated to the first 40 ColumnTransformer
columns and nested SHAP/FFS in that pool, scored on the 28-event test fold. That export is **replaced**.
Current selectors rank each pool by cheap fit-slice importance, score PR-AUC on the val slice of the full
cohort, and do not nest. Kaggle `selector_summary_long.csv` was not committed; tables were reconstructed from
notebook displays (XGBoost’s 7-name list completed as `WBC; eGFR` from the truncated `WB…`).

**[CLOSED]** Stored SHAP is a stratified val sample at cohort prevalence, not the old 87.5% positive 32-row
slice.

### 5.8 TabPFN (Part 4)

**This snapshot (D4, Revision 7).** Nested 5×4 CV on Kaggle Tesla T4, papermill 2026-09-01 21:27–23:20 UTC
(~1.9 h), commit `de46f92`. **`RUN_MODELS["TabPFN"]=True` and `TabPFN (local)=True`.** Both arms finished.
Dump: `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt`.

| Arm | Implementation | This run |
| --- | --- | --- |
| **TabPFN** | `tabpfn_client.TabPFNClassifier` (`thinking_mode=True`, `thinking_effort="high"`, `thinking_metric="average_precision"`). Constructor unchanged. | Fit. Pooled PR-AUC **0.8553**, ROC **0.9905**, Brier **0.0064** (best of seven). |
| **TabPFN (local)** | `from tabpfn import TabPFNClassifier`, `n_estimators="auto"`, `balance_probabilities=True`, `device=cuda` | Fit. PR **0.6754**, ROC **0.9845**, Brier **0.0673** (worst of seven). |

Shared 9-level stent encoder (106 raw strings → 9 levels, min_count=30) is applied **before** the split. Classics
then `ColumnTransformer` scale + `OneHotEncoder(handle_unknown="ignore")` on that 9-level column inside each CV
split (~89 columns). Neither TabPFN arm is in that pipeline; both see the same 9-level frame natively. Imputers
are inert (no missing values).

OOF arrays were written to `/kaggle/working/modeling_results/oof/` (dump cell 7) but **not copied into the repo**
(§12.10, B2).

The six-model local-only snapshot (Rev 6: LightGBM PR-AUC 0.6937, thinking-high unused) is **not** this run.
LightGBM on this GPU pass is PR-AUC **0.6926** (small non-determinism vs 0.6937). Historical thinking-high
prints of PR-AUC 0.8534 / Brier 0.0060 or 0.0360 / nested recall 0.7174 are nearby client numbers, not this
dump (this dump: 0.8553 / 0.0064 / nested recall 0.7065).

### 5.9 TabPFN interpretability (Part 5)


| Block               | Configuration                                                                        | Backend                           | Source                                          |
| ------------------- | ------------------------------------------------------------------------------------ | --------------------------------- | ----------------------------------------------- |
| Mutual information  | `mutual_info_classif` on the 81-column matrix, **full cohort**; CSV = all 81 scores  | sklearn, 0 TabPFN calls           | dump [1a]; `Fast-Glu` / `ZES` in printed top 15 |
| Stability selection | Forward SFS keeping 10 of 81, 5-fold CV, AP scoring, **10 seeds**, **full cohort** (~8.6 h) | local TabPFN, 0 client calls | dump [1b]: WBC 10/10; Staged PCI 7/10; Fiberinogen / LV / ZES 6/10 |
| PDP                 | 4 continuous (grid 30) + 6 binary; **full cohort** n=5185; `balance_probabilities=False` (empirical prior; not Part 4 risk). Nominal `Stent type-SES` excluded from continuous curves | local TabPFN | dump [2/5]: binary P(y=1) ≈ 0.017–0.023 |
| SHAP (shapiq SV)    | **15 VLST=1 + 15 VLST=0** (n=30); fit/background = full cohort; budget 256            | **tabpfn-client + thinking** (succeeded) | dump [3/5] L1558–1567 `Explaining all 30 rows` |
| k-SII / SHAP-IQ     | **one** VLST=1 row (cohort index 5099) from that 15+15 slice, budget 256             | **tabpfn-client + thinking** (succeeded) | dump [3/5] and [4/5] |
| Consensus           | Borda mean of normalised ranks over MI + stability + mean \|SHAP\|                     | mix of local MI/SFS + client SHAP | dump [5/5]: WBC, LV, eGFR lead |


**[MET in this run]** SHAP explains **15 VLST=1 + 15 VLST=0**, not 15 cases only and not 5,185 rows. Client
thinking-high **did not fall back** to local. k-SII remains one illustrative VLST=1 row.

**[MET in this run — PDP]** Empirical prior on the full cohort. Do **not** quote stored Table 3 0.24 / Figure 1
~0.6 as this run. Binary Δ on this dump is small (e.g. `1.1:1Post dilation` P(y=1\|0)=0.0234 vs P(y=1\|1)=0.0191).

**[MET in this run — MI]** All 81 scores written; printed top 15 includes `Fast-Glu` and `ZES`. `Cre` consensus
MI is 0.002281, not a fill-zero. Dual-tree `paper_table1_mutual_info.csv` is the 81-row file (B6 closed).

**[CLOSED — stored assets]** `paper_figures/` in both Part 5 trees (and `data/result/modeling_tabpfn/`) were
copied from `645fb0e` (B12).

### 5.10 The published clinical baseline — integer score now scored **[CLOSED]**

Wang 2020's VLST score is an 8-variable Cox model (DM, previous PCI, AMI as admitting diagnosis, eGFR < 90,
3-vessel disease, number of stents per lesion, SES, no post-dilation), derivation c-statistic 0.80 (cross-validated
0.75), Shantou c-statistic 0.82, with decision-curve analysis against the Dangas LST score.

**Now in-scope.** `code/modeling/rating/wang_vlst_score.ipynb` scores the **published Table 2 integer points**
on all 5,185 rows (frozen; not a nested-CV fit). Encoding: SES → `PES` (matches Wang Table 1 SES 82.61% / 68.76%);
4 post-dilation points → `No postdilation` = 1. Using Wang Table 1's 14 VLST "No post-dilation" cases
(`1.1:1Post dilation` = 1) as the 4-point group yields ROC-AUC 0.5084.

| Quantity | Value | Source |
| --- | ---: | --- |
| Full-cohort ROC-AUC | **0.8013** | notebook (Wang published c = 0.80) |
| Full-cohort PR-AUC | **0.1032** | notebook |
| Fold-mean ROC-AUC | 0.8005 ± 0.0607 | same 5 outer folds as Part 4; score not refit |
| Fold-mean PR-AUC | 0.1134 ± 0.0518 | same |
| Nested-CV LightGBM ROC / PR | 0.9680 / **0.6926** | Part 4 notebook this run |
| Nested-CV TabPFN (thinking-high) ROC / PR | **0.9905 / 0.8553** | Part 4 notebook this run |
| Nested-CV TabPFN (local) ROC / PR | 0.9845 / 0.6754 | Part 4 notebook this run |

Risk bins (≤7 / 8–9 / ≥10): n = 3135 / 1577 / 473; rates 0.51% / 2.22% / 8.67%. Wang's printed intermediate n = 1837
does not add (3135+1837+473 = 5445 ≠ 5185); low and high n match this file.

**Still absent (not B10):** the Cox linear predictor itself, Dangas decision-curve analysis, and the Shantou file
(B11). Reports: Part 4 Table S-Wang / Figure S-Wang (both trees + concat).

---

## 6. Data splits and validation procedures

### 6.1 Three different splitting schemes are in use


| Scheme                             | Where                                                 | Details                                                                                                                                              |
| ---------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. 70/30 stratified hold-out**   | `preprocessing.ipynb`, **stored** Part 2 / old Part 5 PDP (code no longer) | `train_test_split(test_size=0.3, stratify=y, random_state=42)` → train 3,629 (64 events) / test 1,556 (28 events). Current Part 5 PDP is full-cohort. |
| **B. Nested 5×4 stratified CV**    | Part 4                                                | Outer `StratifiedKFold(5, shuffle=True, random_state=42)`; inner `StratifiedKFold(4, shuffle=True, random_state=10_000 + outer_fold)`                |
| **C. Single 70/30 + GridSearchCV** | `baseline_tssi_leakage`, `baseline_without_tssi`      | Not reported in any Markdown                                                                                                                         |
| **D. Full-cohort fit / val**       | Part 2 **current code**                               | One stratified split of all 5,185 rows (`INNER_VAL_SIZE=0.2`, `random_state=42`) → ~4,148 fit / ~1,037 val (~74 / ~18 events). No unused outer test. |


**[VERIFIED]** Scheme A produces identical counts in `preprocessing.ipynb` and in the **stored** Part 2 dump
(`.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L259, L267). Current Part 2
**code** is scheme D (full-cohort fit/val). Not re-run.

### 6.2 What the inner loop of the nested CV actually does

The inner 4-fold loop tunes **only the decision threshold**, not hyperparameters:

```
fold_threshold = optimal_f1_threshold(y_train, inner_oof_probabilities)   # inner OOF, train portion only
outer_model.fit(X_train, y_train)
oof_binary_predictions[val_idx] = (val_probabilities >= fold_threshold)
```

Source: `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L293–309.

`GridSearchCV` winners from `baseline_without_tssi.ipynb` / `baseline_tssi_leakage.ipynb` are **not**
imported. Those notebooks are a separate 70/30 scout (scheme C). Passing that scout's `best_params_`
into nested CV would leak: the 70/30 test slice overlaps later outer folds.

This is a correct, honest threshold-selection design. Calling the procedure "nested CV" is defensible only if the
paper states plainly that the nesting is for **threshold** selection.

### 6.3 [RISK] No hyperparameter tuning; thinking-high vs untuned classics

- The five classic models use **library defaults** apart from class weighting and internal eval metrics.
  No grid inside the nested loop.
- **TabPFN (local)** is a foundation-model default (`n_estimators="auto"`, no thinking).
- **TabPFN (client thinking-high)** is a different object (unchanged constructor). It **did** run in this
  snapshot and leads PR-AUC. That is still **not** a fair hyperparameter search against untuned trees.
- Part 4 Table 0 / Methods must name **seven** arms and say classics are not grid-searched.

Do not describe this scoreboard as “local TabPFN only.” Do not restore “LightGBM is first on PR-AUC” without
the thinking-high row.

### 6.4 Feature representation (9-level encoder; classics still one-hot)

**[REV6]** Part 4 applies the shared 9-level stent encoder **before** the split. Both arms see that collapsed
brand column. What still differs is the sklearn pipeline:

| Model                       | Input |
| --------------------------- | ----- |
| LR, RF, XGB, LGBM, CatBoost | `ColumnTransformer`: median + `StandardScaler` on numerics; most-frequent + `OneHotEncoder(handle_unknown="ignore")` on the **9-level** `Stent type-SES` (~89 columns; no `drop="first"`) |
| TabPFN (thinking-high) | Same 9-level frame, native, client API |
| TabPFN (local)              | Same 9-level frame, native categorical handling, no scaling |

Imputation is inert (no missing values). Do not describe imputers as a data-cleaning step. The old ~186-column
106-string one-hot is **not** this run. Part 2 still uses drop-first → 88 columns. The shapiq `imputer="baseline"`
in Part 5 is a different object (hidden-feature replacement for attribution).

### 6.5 Threshold reporting: honest and optimistic versions both exist

The notebook computes both. Part 4 **Table 2 quotes the honest nested print.** Figure 3 / Table 3 remain the optimistic pooled cut, labelled as biased.


| Version | How the threshold is chosen | TabPFN (thinking-high) | LightGBM | TabPFN (local) |
| --- | --- | --- | --- | --- |
| **Honest (nested)** | Inner-CV OOF on the training portion, applied once to unseen outer fold | precision 0.7927, recall **0.7065**, F1 0.7471, 5076/17/27/65, mean t 0.271 ± 0.067 | precision 0.6630, recall 0.6630, F1 0.6630, 5062/31/31/61, mean t 0.117 ± 0.087 | precision 0.5478, recall 0.6848, F1 0.6087, 5041/52/29/63, mean t 0.915 ± 0.012 |
| **Optimistic (pooled)** | F1 cut on the concatenated OOF labels that are then scored | precision 0.7812, recall 0.8152, F1 0.7979, 5072/21/17/75, t = 0.193 | precision 0.6263, recall 0.6739, F1 0.6492, 5056/37/30/62, t = 0.064 | precision 0.5067, recall **0.8261**, F1 0.6281, 5019/74/16/76, t = 0.886 |

Sources: dump nested block L1333–1351, pooled block L1312–1330.

Part 4 **Table 2 is the honest nested print.** Figure 3 / Table 3 are the pooled cut, labelled as optimistically
biased. Do not quote TabPFN (local) pooled recall 0.8261 or thinking-high pooled recall 0.8152 as the nested
result. Historical nested 5080/13/26/66 and PNG 5066/27/15/77 are **not** this dump.

### 6.6 [GAP] No external validation of the *machine-learning* models

Every ML number in the repository comes from the same 5,185 rows. There is no temporal split and no held-out
recalibration set.

Wang 2020 **did** externally validate their Cox score on 2,058 patients from Shantou (c-statistic 0.82). Those
rows are **not in this repository**. If they can be obtained, they are the natural external test set for TabPFN
and the five baselines — and the comparison that reviewers will expect, because it is how the published score was
already tested. **[TODO-EXT]**

Until then, the honest statement is: nested-CV discrimination on the derivation cohort only; the published score
has an external cohort this analysis did not use. The frozen integer points *are* now scored on the derivation
file (ROC-AUC 0.8013, PR-AUC 0.1032; §5.10). That is not a substitute for Shantou.

---

## 7. Every reported metric

### 7.1 Part 4 — nested-CV ranking metrics (the paper's headline results)

Evaluation protocol for all rows: **pooled out-of-fold probabilities from 5 outer folds, n = 5,185, 92 events,
threshold-independent**. Source: executed Kaggle notebook `de46f92`
`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L1291–1309.
Committed `paper_table1_ranking.csv` is **Rev 6 (six models)** and must not be quoted as this run.


| Rank | Model | PR-AUC | PR fold mean ± SD | ROC-AUC | ROC fold mean ± SD | Brier | Status |
| ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| 1 | TabPFN (thinking-high) | **0.8553** | 0.8488 ± 0.0861 | **0.9905** | 0.9906 ± 0.0070 | **0.0064** | ✓ this run |
| 2 | LightGBM | 0.6926 | 0.6936 ± 0.0915 | 0.9680 | 0.9694 ± 0.0165 | 0.0093 | ✓ |
| 3 | XGBoost | 0.6815 | 0.6928 ± 0.1288 | 0.9439 | 0.9431 ± 0.0418 | 0.0088 | ✓ |
| 4 | TabPFN (local) | 0.6754 | 0.6739 ± 0.0812 | 0.9845 | 0.9846 ± 0.0030 | 0.0673 | ✓ |
| 5 | CatBoost | 0.6172 | 0.6353 ± 0.0540 | 0.9594 | 0.9612 ± 0.0137 | 0.0101 | ✓ |
| 6 | Random Forest | 0.4865 | 0.5034 ± 0.0793 | 0.9209 | 0.9206 ± 0.0423 | 0.0143 | ✓ |
| 7 | Logistic Regression | 0.3326 | 0.3451 ± 0.1213 | 0.9224 | 0.9235 ± 0.0251 | 0.0563 | ✓ |
| — | Wang 2020 integer score (frozen) | 0.1032 | 0.1134 ± 0.0518 | 0.8013 | 0.8005 ± 0.0607 | — | Full cohort + same 5 folds; not a nested-CV fit (§5.10) |


**Prevalence baseline for PR-AUC: 0.0177.** Every PR-AUC above should be reported against this reference.

**Notebook.** `code/modeling/rating/baseline_plus_tabpfn.ipynb` on `origin/main` (`de46f92`). Dump
`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt`.

**D4 for this snapshot — two TabPFN Briers.** Client thinking-high Brier is **0.0064**, the **best** of the
seven (calibration cell L1046–1052). TabPFN (local) Brier is **0.0673**, the **worst**. Sentences of the form
“TabPFN's Brier is worse than the tree ensembles” are **true for the local arm only** and **false for
thinking-high**. Do not collapse the two arms. Historical 0.0060 vs PNG 0.0360 was client non-determinism on
a different dump; this dump is 0.0064 and internally consistent.

**Fold mean ± SD.** Ranking super-table L1291–1309. Part 4 Markdown Table 1 and `paper_table1_ranking.csv` in
both trees now match this dump (B1 closed).

### 7.2 Part 4 — optimistic pooled operating-point metrics (Table 3; do not quote instead of §7.3)

Protocol: pooled OOF probabilities thresholded at the **F1-maximising pooled threshold** (optimistically biased,
see §6.5). Source: dump L1312–1330. Repo `paper_table3_pooled_f1.csv` now includes the thinking-high row (B1 closed).


| Model | t_F1 | Accuracy | Precision | Recall | Specificity | F1 | F2 | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TabPFN (thinking-high) | 0.193 | 0.9927 | 0.7812 | 0.8152 | 0.9959 | 0.7979 | 0.8082 | 5072/21/17/75 |
| LightGBM | 0.064 | 0.9871 | 0.6263 | 0.6739 | 0.9927 | 0.6492 | 0.6638 | 5056/37/30/62 |
| XGBoost | 0.203 | 0.9884 | 0.6739 | 0.6739 | 0.9941 | 0.6739 | 0.6739 | 5063/30/30/62 |
| TabPFN (local) | 0.886 | 0.9826 | 0.5067 | **0.8261** | 0.9855 | 0.6281 | 0.7336 | 5019/74/16/76 |
| CatBoost | 0.416 | 0.9873 | 0.6806 | 0.5326 | 0.9955 | 0.5976 | 0.5568 | 5070/23/43/49 |
| Random Forest | 0.104 | 0.9826 | 0.5098 | 0.5652 | 0.9902 | 0.5361 | 0.5532 | 5043/50/40/52 |
| Logistic Regression | 0.985 | 0.9819 | 0.4857 | 0.3696 | 0.9929 | 0.4198 | 0.3881 | 5057/36/58/34 |


**D4.** Use this table only as the *optimistic* companion. Nested results are §7.3.

### 7.3 Part 4 — honest nested operating point (Part 4 Table 2; quote this)

Protocol: per-fold thresholds from inner CV, applied once to unseen outer-fold cases.
Source: dump L1333–1351. Repo `paper_table2_nested_operating_point.csv` now includes the thinking-high row
(B1 closed).


| Model | Threshold mean ± SD | Accuracy | Precision | Recall | Specificity | F1 | F2 | TN/FP/FN/TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TabPFN (thinking-high) | 0.271 ± 0.067 | 0.9915 | 0.7927 | **0.7065** | 0.9967 | **0.7471** | 0.7222 | 5076/17/27/65 |
| LightGBM | 0.117 ± 0.087 | 0.9880 | 0.6630 | 0.6630 | 0.9939 | 0.6630 | 0.6630 | 5062/31/31/61 |
| XGBoost | 0.225 ± 0.060 | 0.9875 | 0.6452 | 0.6522 | 0.9935 | 0.6486 | 0.6508 | 5060/33/32/60 |
| TabPFN (local) | 0.915 ± 0.012 | 0.9844 | 0.5478 | 0.6848 | 0.9898 | 0.6087 | 0.6522 | 5041/52/29/63 |
| CatBoost | 0.167 ± 0.040 | 0.9815 | 0.4836 | 0.6413 | 0.9876 | 0.5514 | 0.6020 | 5030/63/33/59 |
| Random Forest | 0.118 ± 0.013 | 0.9840 | 0.5517 | 0.5217 | 0.9923 | 0.5363 | 0.5275 | 5054/39/44/48 |
| Logistic Regression | 0.947 ± 0.035 | 0.9769 | 0.3654 | 0.4130 | 0.9870 | 0.3878 | 0.4025 | 5027/66/54/38 |


**This table, not §7.2, is the defensible operating-point result.** Thinking-high has the highest nested F1
(0.7471) and nested recall (0.7065) among the seven. TabPFN (local) still has more false positives than
LightGBM (52 vs 31) at similar event capture.

### 7.4 Part 4 — fold-wise ranking metrics

Source: dump L1354–1395. Each fold n = 1,037, with n_pos = 18, 18, 18, 19, 19.


| Model | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | PR-AUC | 0.2177 | 0.3040 | 0.2566 | 0.4662 | 0.4811 |
| Logistic Regression | ROC-AUC | 0.8881 | 0.9497 | 0.9092 | 0.9425 | 0.9279 |
| Random Forest | PR-AUC | 0.4739 | 0.5447 | 0.4451 | 0.4315 | 0.6218 |
| Random Forest | ROC-AUC | 0.9317 | 0.9219 | 0.8502 | 0.9638 | 0.9351 |
| XGBoost | PR-AUC | 0.8085 | 0.6320 | 0.5226 | 0.8331 | 0.6680 |
| XGBoost | ROC-AUC | 0.9778 | 0.9081 | 0.8918 | 0.9861 | 0.9515 |
| LightGBM | PR-AUC | 0.7505 | 0.7130 | 0.5399 | 0.7727 | 0.6919 |
| LightGBM | ROC-AUC | 0.9851 | 0.9464 | 0.9637 | 0.9856 | 0.9664 |
| CatBoost | PR-AUC | 0.5668 | 0.6560 | 0.6440 | 0.7082 | 0.6015 |
| CatBoost | ROC-AUC | 0.9749 | 0.9433 | 0.9631 | 0.9731 | 0.9516 |
| TabPFN (thinking-high) | PR-AUC | 0.8640 | 0.7837 | 0.7407 | 0.9497 | 0.9061 |
| TabPFN (thinking-high) | ROC-AUC | 0.9954 | 0.9881 | 0.9802 | 0.9981 | 0.9914 |
| TabPFN (local) | PR-AUC | 0.6384 | 0.6353 | 0.5829 | 0.7274 | 0.7855 |
| TabPFN (local) | ROC-AUC | 0.9875 | 0.9840 | 0.9812 | 0.9880 | 0.9826 |


Thinking-high PR-AUC is higher than LightGBM in **5 of 5** folds. TabPFN (local) is higher than LightGBM in
**2 of 5** folds (3 and 5), same pattern as Rev 6. There is still no paired significance test (§12.9). Do not
say the *local* arm wins 5/5.

### 7.5 Unreported metrics: single-split baselines with and without the leakage variable

Protocol: single stratified 70/30 split, GridSearchCV, optional SMOTE. Not in any Markdown report.
Sources: `.nbdump/code__modeling__rating__baseline_tssi_leakage.txt` L797–817;
`.nbdump/code__modeling__rating__baseline_without_tssi.txt` L906–924.


| Model               | With TSSI: Acc / F1 / Recall / Prec / ROC-AUC / PR-AUC | Without TSSI: Acc / F1 / Recall / Prec / ROC-AUC / PR-AUC |
| ------------------- | ------------------------------------------------------ | --------------------------------------------------------- |
| Logistic Regression | 0.9846 / 0.6923 / 1.0000 / 0.5294 / 0.9990 / 0.9575    | 0.9354 / 0.2637 / 0.6667 / 0.1644 / 0.9171 / 0.5077       |
| Decision Tree       | 0.9942 / 0.8500 / 0.9444 / 0.7727 / 0.9696 / 0.7308    | 0.9749 / 0.1875 / 0.1667 / 0.2143 / 0.5774 / 0.0405       |
| Random Forest       | 0.9942 / 0.8000 / 0.6667 / 1.0000 / 0.9993 / 0.9680    | 0.9749 / 0.4348 / 0.5556 / 0.3571 / 0.9338 / 0.4700       |
| Gaussian NB         | 0.1851 / 0.0409 / 1.0000 / 0.0209 / 0.5854 / 0.0209    | 0.1851 / 0.0409 / 1.0000 / 0.0209 / 0.5854 / 0.0209       |
| CatBoost            | 0.9981 / 0.9412 / 0.8889 / 1.0000 / 0.9995 / 0.9773    | 0.9875 / 0.5806 / 0.5000 / 0.6923 / 0.9669 / 0.6582       |
| XGBoost             | 0.9981 / 0.9412 / 0.8889 / 1.0000 / 0.9987 / 0.9609    | 0.9884 / 0.5714 / 0.4444 / 0.8000 / 0.9380 / 0.6118       |
| LightGBM            | 0.9981 / 0.9412 / 0.8889 / 1.0000 / 0.9989 / 0.9708    | 0.9865 / 0.5625 / 0.5000 / 0.6429 / 0.9483 / 0.6018       |


### 7.6 Part 1 — statistical effect estimates

**Continuous, FDR q < 0.05** (`paper_table1_continuous_univariate.csv`; full-cohort n = 5,185):


| Feature                       | Test           | Effect metric | ES     | Δ mean  | Δ median | p        | q        |
| ----------------------------- | -------------- | ------------- | ------ | ------- | -------- | -------- | -------- |
| Time since stent implantation | Mann–Whitney U | MW r          | −0.170 | −572.05 | −683.00  | 1.70e-34 | 4.07e-33 |
| WBC                           | Mann–Whitney U | MW r          | 0.130  | 3.75    | 3.82     | 7.90e-21 | 9.48e-20 |
| eGFR                          | Welch t        | Cohen d       | −0.712 | −24.15  | −17.58   | 4.64e-20 | 3.71e-19 |
| LV                            | Welch t        | Cohen d       | 1.127  | 4.56    | 4.00     | 5.44e-17 | 3.26e-16 |
| CKD5                          | Mann–Whitney U | MW r          | 0.040  | 0.18    | 0.00     | 1.19e-05 | 5.69e-05 |
| No.of stents per lesion       | Mann–Whitney U | MW r          | 0.037  | 0.21    | 0.00     | 1.00e-04 | 6.00e-04 |
| HbA1c                         | Mann–Whitney U | MW r          | 0.052  | 0.47    | 1.05     | 2.00e-04 | 7.00e-04 |
| NO.of vessels                 | Welch t        | Cohen d       | 0.388  | 0.32    | 0.00     | 5.00e-04 | 0.0014   |
| Total stent length            | Mann–Whitney U | MW r          | 0.045  | 6.76    | 2.00     | 0.0011   | 0.0030   |
| Fiberinogen                   | Mann–Whitney U | MW r          | 0.035  | 0.20    | 0.17     | 0.0119   | 0.0286   |


Near-misses (informative for the knife-edge nature of the cut-off): `Fast-Glu` q = 0.0536,
`LVEF` q = 0.0666, `HGB` q = 0.0716, `CaI` q = 0.0869.

**Binary, FDR q < 0.05** (`paper_table2_binary_univariate.csv`):


| Feature               | Test   | OR    | RR    | φ      | VLST% if 1 | VLST% if 0 | p        | q        |
| --------------------- | ------ | ----- | ----- | ------ | ---------- | ---------- | -------- | -------- |
| 1.1:1Post dilation    | χ²     | 0.187 | 0.191 | −0.089 | 0.56       | 2.92       | 1.30e-10 | 3.69e-09 |
| No postdilation       | χ²     | 5.355 | 5.228 | 0.089  | 2.92       | 0.56       | 1.30e-10 | 3.69e-09 |
| CKD90                 | χ²     | 2.625 | 2.567 | 0.063  | 3.59       | 1.40       | 6.55e-06 | 0.0001   |
| Previous PCI          | Fisher | 6.485 | 5.958 | 0.085  | 9.62       | 1.61       | 1.25e-05 | 0.0002   |
| 3-vessel disease      | χ²     | 2.169 | 2.135 | 0.052  | 2.87       | 1.34       | 0.0002   | 0.0021   |
| Clopidogrel           | χ²     | 0.503 | 0.509 | −0.043 | 1.17       | 2.29       | 0.0022   | 0.0209   |
| Diabetes              | χ²     | 1.889 | 1.865 | 0.042  | 2.71       | 1.45       | 0.0028   | 0.0226   |
| PES                   | χ²     | 2.158 | 2.133 | 0.040  | 2.12       | 1.00       | 0.0044   | 0.0315   |
| Multi-vessel CAD      | χ²     | 1.890 | 1.870 | 0.038  | 2.19       | 1.17       | 0.0062   | 0.0351   |
| Single-vessel disease | χ²     | 0.529 | 0.535 | −0.038 | 1.17       | 2.19       | 0.0062   | 0.0351   |


Note the first pair and the last pair are exact complements: they are two reports of one association.

**Categorical:** `Stent type-SES`, χ² = 44.90, df = 8, Cramér's V = 0.093, p = 3.85e-07, q = 3.85e-07,
9 levels after collapsing n < 30 to `other`.

**Multivariable adjusted ORs** (`paper_table4_multivariable_or_numeric.csv`; 17 covariates, continuous per 1 SD,
percentile bootstrap CI, `class_weight="balanced"`, `C=1e6`, N_BOOT = 200):


| Feature                 | Type      | Univariate OR | Adjusted OR | 95% CI              |
| ----------------------- | --------- | ------------- | ----------- | ------------------- |
| WBC                     | cont. /SD | 2.694         | 3.002       | 2.424 – 4.225       |
| eGFR                    | cont. /SD | 0.334         | 0.113       | 0.060 – 0.158       |
| LV                      | cont. /SD | 3.116         | 3.282       | 2.335 – 5.254       |
| CKD5                    | cont. /SD | 1.391         | **0.156**   | 0.023 – 0.285       |
| No.of stents per lesion | cont. /SD | 1.379         | 1.297       | 0.693 – 2.177       |
| HbA1c                   | cont. /SD | 1.367         | 0.870       | 0.569 – 1.280       |
| NO.of vessels           | cont. /SD | 1.456         | 1.498       | 0.774 – 3.392       |
| Total stent length      | cont. /SD | 1.391         | 1.127       | 0.658 – 2.269       |
| Fiberinogen             | cont. /SD | 1.220         | 1.045       | 0.804 – 1.348       |
| 1.1:1Post dilation      | binary    | 0.187         | 0.144       | 0.040 – 0.245       |
| No postdilation         | binary    | 5.353         | **0.895**   | 0.429 – 1.358       |
| CKD90                   | binary    | 2.625         | **12.540**  | **2.708 – 639.506** |
| Previous PCI            | binary    | 6.465         | 8.977       | 3.226 – 28.618      |
| 3-vessel disease        | binary    | 2.168         | 0.605       | 0.135 – 2.936       |
| Clopidogrel             | binary    | 0.504         | 0.464       | 0.195 – 0.817       |
| Diabetes                | binary    | 1.889         | 1.875       | 0.678 – 4.331       |
| PES                     | binary    | 2.156         | 1.239       | 0.585 – 4.056       |


**Joint cross-domain model** (`domain_joint_multivariable_or.csv`, 12 covariates):


| Feature                 | Univariate OR | Adjusted OR | 95% CI         |
| ----------------------- | ------------- | ----------- | -------------- |
| Age                     | 1.080         | 1.099       | 0.836 – 1.344  |
| WBC                     | 2.694         | 2.942       | 2.343 – 4.396  |
| eGFR                    | 0.334         | 0.203       | 0.127 – 0.299  |
| LV                      | 3.116         | 2.950       | 2.197 – 5.036  |
| LVEF                    | **0.851**     | **1.650**   | 1.301 – 2.260  |
| No.of stents per lesion | 1.379         | 1.552       | 1.173 – 2.270  |
| Men                     | 1.286         | **3.283**   | 1.577 – 7.900  |
| Current smoker          | 1.105         | 0.976       | 0.541 – 2.074  |
| Current drinking        | 0.981         | 1.071       | 0.423 – 2.114  |
| 1.1:1Post dilation      | 0.192         | 0.191       | 0.044 – 0.382  |
| Previous PCI            | 6.733         | 9.576       | 3.308 – 23.576 |
| Diabetes                | 1.898         | 1.460       | 0.789 – 2.597  |


**Interaction screen** (`domain_interaction_screen.csv`, 16 hand-picked pairs, LR test, BH within the 16):


| Pair                             | LR statistic | p       | Interaction OR | q      |
| -------------------------------- | ------------ | ------- | -------------- | ------ |
| LV × eGFR                        | 9.813        | 0.00173 | 1.245          | 0.0277 |
| Men × eGFR                       | 8.527        | 0.00350 | 0.342          | 0.0280 |
| WBC × eGFR                       | 2.649        | 0.104   | 1.125          | 0.553  |
| (13 further pairs, all q > 0.55) |              |         |                |        |


**Domain strength summary** (`domain_strength_summary.csv`): Laboratory 16 features / 6 global FDR hits;
Cardiac function 2 / 1; Procedural-stent 16 / 5; Comorbidities 10 / 2; Anatomy 23 / 4; Medications 4 / 1;
Presentation 5 / 0; Demographics 4 / 0.

### 7.7 Part 5 — interpretability quantities

These are **not** performance metrics. They are model-attribution and screening quantities. **Notebook
`645fb0e` is current** (15+15 SHAP, empirical PDP, 81-row MI). Repo CSVs under
`paper_results/05_tabpfn_interpretability/paper_figures/` match that run (B12 closed). Consensus on this dump
(top 5): WBC, LV, eGFR, `1.1:1Post dilation`, CaI.

### 7.8 Metric comparability audit


| Question                                                        | Answer                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Are the six Part 4 models compared on identical data and folds? | **Seven models, same folds.** Same `oof_probabilities` arrays, same outer folds. |
| Are they compared with identical tuning effort?                 | **No.** Classics are untuned defaults; TabPFN (local) has no thinking; TabPFN client uses thinking-high. Thinking-high leads PR-AUC; that is still not a matched hyperparameter search. |
| Are they compared on identical feature representations?         | **Closer than before.** Shared 9-level encoder; classics still one-hot (~89) while both TabPFN arms see the 9-level frame natively. |
| Are the Part 4 operating-point metrics unbiased?                | **Table 2 (nested) is the honest protocol.** Figure 3 / Table 3 pooled F1 is optimistically biased (§6.5, §7.3). |
| Is Part 2 comparable with Part 4?                               | **No.** Part 2 is a full-cohort fit/val discovery split, **88** encoded columns, seven classic models, PR-AUC only. It still does not feed Part 4. |
| Is Part 5 comparable with Part 4?                               | **No.** Part 5 is full-cohort attribution; Part 4 is nested-CV prediction. Ranking/SHAP in Part 5 use `balance_probabilities=True`; PDP uses `False`. Part 4 both TabPFN arms use `True`. Part 5 SHAP used the **client**; Part 4 scored **both** client and local. |
| Are effect sizes in EDA Table 1 comparable across rows?         | **No.** Cohen's d and Mann–Whitney r are mixed in one column and plotted on one axis in Figure 3. MW r = Z/√N is severely attenuated by 1.77% class imbalance: `WBC` has the second-smallest p-value in the entire study (7.9e-21) but an "effect size" of 0.130, next to `LV`'s d = 1.127. Figure 3 must not be read as a magnitude ranking. |
| Are univariate ORs consistent across tables?                    | **They are three named estimators, not one OR.** Table 2 = 2×2/Fisher; Table 4 = unweighted logit; Table S4 = joint-domain univariate (§12.4, C13).                                                                                                                                                                                            |


---

## 8. Feature sets produced by each method

### 8.1 EDA / statistical (univariate FDR q < 0.05)

**Continuous (9, excluding time-at-risk):** `WBC`, `eGFR`, `LV`, `CKD5`, `No.of stents per lesion`, `HbA1c`,
`NO.of vessels`, `Total stent length`, `Fiberinogen`.
**Binary (10):** `1.1:1Post dilation`, `No postdilation`, `CKD90`, `Previous PCI`, `3-vessel disease`,
`Clopidogrel`, `Diabetes`, `PES`, `Multi-vessel CAD`, `Single-vessel disease`.
**Categorical (1):** `Stent type-SES`.
**Total: 20** (Part 3 L43). Plus `Time since stent implantation`, excluded by construction.

**[RISK]** Of these 20, at least 8 are redundant re-encodings: the post-dilation complements (2 slots for 1 bit),
the vessel-disease family (`3-vessel`, `Multi-vessel`, `Single-vessel`, `NO.of vessels` — 4 slots for 1
construct), and the renal family (`CKD5`, `CKD90` alongside continuous `eGFR` — 3 slots for 1 construct). The
"20 statistical discoveries" headline should be reported as roughly **12 distinct clinical constructs**.
**[REV4]** Part 1 and Part 3 markdowns now say this. Jaccard / Venn stay on the **name** lists (now 20 vs 13).

### 8.2 Multivariable-model survivors (bootstrap CI excluding 1)

From EDA Table 4: `WBC`, `eGFR`, `LV`, `CKD5` (**sign-flipped**), `1.1:1Post dilation`, `CKD90` (CI 2.71–639.5),
`Previous PCI`, `Clopidogrel`.
From the joint domain model: `WBC`, `eGFR`, `LV`, `LVEF` (**sign-flipped**), `No.of stents per lesion`, `Men`,
`1.1:1Post dilation`, `Previous PCI`.

### 8.3 LOCO (Part 2)

- Pool: cheap fit-slice importance prefix, cap **60**.
- Scored on the 1,037-row val slice (18 events), **PR-AUC only**, 7 models.
- Unique names in the log: **60 for every model** — the cap, not a finding.
- Cross-model intersection of top-20: `Cre`, `LV`, `LVEF`, `WBC`, `eGFR`.

### 8.4 Coalition SHAP (Part 2)

- Universe: independent cheap-importance prefix of **40** (not nested in LOCO).
- Value function: PR-AUC on a stratified val subsample at cohort prevalence.
- Unique names per model: **40** (the universe).
- Cross-model intersection of top-20: `HGB`, `WBC`.

### 8.5 FFS (Part 2)

- Candidate pool: independent cheap-importance prefix of **24**.
- Greedy forward addition on the val slice; stop at 12 steps or when PR-AUC gain ≤ 0.
- Unique names per model: 4–12 (sparsest; CatBoost 4, LightGBM 5 after early stop).
- Cross-model intersection of top-20: **empty**.

### 8.6 Classic-model consensus (Part 2 Table 2 and Table 4)

**Strictest global intersection** (all 7 models × all 3 selectors, PR-AUC top-20): **empty**.
**Global union of scored names:** 86.

**Within-model LOCO ∩ SHAP ∩ FFS, PR-AUC** (Part 2 Table 2):


| Model | PR-AUC |
| ----- | ------ |
| lr    | Cre, LV, Men, UA, WBC, eGFR |
| rf    | HGB, LDL, LVEF, Men, WBC, eGFR |
| rf_b  | CaI, HGB, LVEF, WBC |
| cat   | 1.1:1Post dilation, HGB, WBC |
| xgb   | 1.1:1Post dilation, Aneurysm, Cre, HGB, LV, WBC, eGFR |
| xgb_b | 1.1:1Post dilation, LV, LVEF, WBC, eGFR |
| lgb   | HbA1c, LV |


**Union of the above (the "ML consensus catalogue", n = 13):** `1.1:1Post dilation`, `Aneurysm`, `CaI`, `Cre`,
`HGB`, `HbA1c`, `LDL`, `LV`, `LVEF`, `Men`, `UA`, `WBC`, `eGFR`.

**Union sizes per model** (Part 2 Table 3, top-20 unions): lr 32, rf 35, rf_b 34, cat 32, xgb 31, xgb_b 30, lgb 30.
**Jaccard between selector unions** (top-20, models pooled): LOCO–SHAP 0.62, SHAP–FFS 0.48, LOCO–FFS 0.43 —
moderate because the selectors are independent (the old 0.95–0.97 figure was nested-pool artefact).

### 8.7 TabPFN interpretability (Part 5)

**Mutual information, top 15** (`paper_table1_mutual_info.csv`; full cohort, median-imputed):
`CaI` 0.019224, `WBC` 0.018360, `LV` 0.015178, `Stent type-SES` 0.011357, `eGFR` 0.009931,
`1.1:1Post dilation` 0.007932, `LDL` 0.007693, `No postdilation` 0.007535, `HbA1c` 0.007439, `HGB` 0.004606,
`Fast-Glu` **(value blank in CSV)**, `TCL` 0.004257, `Fiberinogen` 0.004040, `ZES` **(value blank in CSV)**,
`Visual thrombus` 0.003707.

**Stability selection, 10 seeds of forward SFS keeping 10 of 81** (`paper_table2_stability.csv`):
1.0 → `LV`, `Stent type-SES`, `eGFR`; 0.9 → `Age`, `Cre`, `WBC`; 0.7 → `No postdilation`; 0.6 → `STEMI`;
0.5 → `HbA1c`, `LVEF`; 0.3 → `No.of stents per lesion`, `Staged PCI`; 0.2 → `1.1:1Post dilation`, `CKD5`,
`CKD60`, `EVS`; 0.1 → `Current drinking`, `History of peripheral vascualr disease`, `Initial diagnosis-AMI`,
`PES`, `Single-vessel disease`, `Visual thrombus`.

**Mean |SHAP| over 15 rows — all 15 are VLST cases** (`paper_table4_shap_mean_abs.csv`):
`LV` 1.2368, `WBC` 1.1648, `LDL` 0.6408, `eGFR` 0.4713, `1.1:1Post dilation` 0.2392, `Stent type-SES` 0.2296,
`No postdilation` 0.1779, `CaI` 0.1779, `HbA1c` 0.1602, `Cre` 0.1539, `Fiberinogen` 0.0942, `HGB` 0.0760,
`TCL` 0.0500, `No.of stents per lesion` 0.0424, `Visual thrombus` 0.0384. The residual bundle
("sum of 72 other features") is **1.41**, larger than any single feature — importance is not concentrated.

**Binary PDP, balanced-prior scale, not absolute risk** (`paper_table3_pdp_binary.csv`):

| Feature            | P(y=1 | 0) | P(y=1 | 1) | ΔP      |
| ------------------ | ---------- | ---------- | ------- |
| 1.1:1Post dilation | 0.2643     | 0.1781     | −0.0862 |
| No postdilation    | 0.2430     | 0.2067     | −0.0363 |
| STEMI              | 0.1401     | 0.1271     | −0.0130 |
| CKD60              | 0.1359     | 0.1234     | −0.0124 |
| Staged PCI         | 0.1334     | 0.1303     | −0.0031 |
| EVS                | 0.1328     | 0.1354     | +0.0026 |

**Borda consensus, top 15** (`paper_table5_consensus.csv`): `LV` 0.9875 (3/3), `WBC` 0.9750 (3/3),
`eGFR` 0.9667 (3/3), `Stent type-SES` 0.9625 (3/3), `No postdilation` 0.9208 (3/3),
`1.1:1Post dilation` 0.9062 (2/3), `HbA1c` 0.8979 (3/3), `Visual thrombus` 0.7854 (1/3), `CaI` 0.7583 (2/3),
`LDL` 0.7542 (2/3), `Cre` 0.7479 (2/3), `HGB` 0.6958 (2/3), `Fiberinogen` 0.6875 (2/3),
`No.of stents per lesion` 0.6833 (0/3), `TCL` 0.6792 (1/3).

**[RISK]** The Borda score mixes three quantities on incommensurable scales, one of which (mean |SHAP|) comes
from 15 case rows and one of which (stability) comes from an 8.6-hour SFS on the full cohort. `Cre` and
`No.of stents per lesion` carry `mutual_info = 0.0` in the consensus table because they fell outside the MI
top-15 and were filled with zero — those are **imputed zeros, not measured zeros**. Part 5's Table 5 caption
does not say this; Table 1's caption partly does (L53).

### 8.8 What every method agrees on

Present in the statistical FDR set, the classic-ML consensus, **and** the TabPFN consensus:
`WBC`**,** `eGFR`**,** `LV`. Nothing else clears all three. These three are also, by §3.2, the three variables whose
measurement timing is undocumented.

---

## 9. Overlap and disagreement: statistics vs machine learning

Source: `paper_results/03_stats_vs_ml/feature_extraction_comparison.md`. Generating code:
`code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb`.
Headline arithmetic is asserted in that notebook: Jaccard = 5/28 ≈ 0.1786, intersection
`{WBC, eGFR, LV, HbA1c, 1.1:1Post dilation}`. **[TODO-P3 — closed]**

### 9.1 Headline overlap


| Quantity                  | Value                                                            | Source     |
| ------------------------- | ---------------------------------------------------------------- | ---------- |
| Statistical FDR catalogue | 20 names                                                         | Part 3 §1  |
| ML consensus catalogue    | 13 names                                                         | Part 3 §1  |
| Intersection              | **5** — `WBC`, `eGFR`, `LV`, `HbA1c`, `1.1:1Post dilation`       | Part 3 §2  |
| Jaccard                   | 5 / 28 ≈ **0.18**                                                | Part 3 §2  |
| Statistics-only           | 15                                                               | Part 3 §4  |
| ML-only                   | 8                                                                | Part 3 §5  |


**[VERIFIED]** 20 + 13 − 5 = 28. The input lists match §8.1 and §8.6. `stats_vs_ml_comparison.ipynb` asserts the
same intersection and Jaccard.

### 9.2 Shared features and the evidence behind each


| Feature            | Statistical evidence           | ML evidence                                      |
| ------------------ | ------------------------------ | ------------------------------------------------ |
| WBC                | MW r = 0.130, q = 9.48e-20     | In 6/7 model three-way sets; cross-model LOCO/SHAP |
| eGFR               | Cohen d = −0.712, q = 3.71e-19 | Cross-model LOCO; lr/rf/xgb_b three-way          |
| LV                 | Cohen d = 1.127, q = 3.26e-16  | Cross-model LOCO; lr/lgb/xgb/xgb_b three-way     |
| HbA1c              | MW r = 0.052, q = 7e-4         | LightGBM three-way only                          |
| 1.1:1Post dilation | χ² OR = 0.187, q = 3.69e-9     | CatBoost / XGBoost / XGB_b three-way             |


**[REV5]** `Fiberinogen` and `Previous PCI` left the intersection when F1/F2 were dropped. `Previous PCI` is
still frequently selected. The three Previous-PCI ORs are named estimators, not a conflict (§12.4).

### 9.3 Statistics-only (15) — the repository's own explanations, audited

Part 3's Table 3 gives a reason for each. The reasons fall into three buckets:


| Bucket                       | Names                                                                                                                                             | Assessment                                                                                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Genuine collinear redundancy | `No postdilation`, `Multi-vessel CAD`, `Single-vessel disease`, `3-vessel disease`, `NO.of vessels`, `CKD90`, `CKD5`, `PES` | **Real.** Univariate testing scores every member of a redundant block; a fitted model needs one. The 1.1:1 flag *is* now in consensus; its complement is not. |
| Brand encoding               | `Stent type-SES`                                                                                                                                  | **Real but encoding.** χ² is on 9 collapsed brands; ML one-hots those 9 levels. `Stent type-SES_resolute` is frequently selected; the parent name is not in a 3-way set. |
| Weak / not in three-way      | `No.of stents per lesion`, `Total stent length`, `Clopidogrel`, `Diabetes`, `Previous PCI`, `Fiberinogen`                               | **Filter, not a missing pool.** Selectors now rank their own prefixes. Absence means not in LOCO ∩ SHAP ∩ FFS top-20. Previous PCI is frequently selected anyway. |


### 9.4 ML-only (8) — the repository's own explanations, audited


| Feature                                                                                         | Univariate status         | Repository's stated reason                                                | Assessment                                                                                                                               |
| ----------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Cre                                                                                             | p = 0.88 (MW r = 0.002)   | Renal surrogate for eGFR                                                  | Plausible; note `Cre` skew = 7.48, kurtosis = 161                                                                                        |
| Men                                                                                             | p = 0.27                  | `Men × eGFR` interaction is FDR-significant; joint model adjusted OR 3.28 | The interaction q = 0.028 comes from **16 hand-picked pairs**; the adjusted OR 3.28 arises from a null univariate. **Exploratory only.** |
| LVEF                                                                                            | raw p = 0.033, q = 0.0666 | Sign reversal 0.851 → 1.65 when `LV` is in the same model                 | Unchanged. Trees still split on systolic function.                                                                                       |
| HGB                                                                                             | raw p = 0.039, q = 0.0716 | CatBoost/RF/XGB three-way                                                 | Plausible                                                                                                                                |
| CaI                                                                                             | raw p = 0.051, q = 0.0869 | RF_b three-way; FDR boundary                                              | Plausible; note `CaI` is the **top** mutual-information feature in Part 5                                                                |
| LDL                                                                                             | ns (p = 0.33)             | RF three-way lipid split                                                  | Val-slice artefact risk                                                                                                                  |
| UA                                                                                              | ns (p = 0.17)             | LR three-way ACS offset                                                   | Val-slice artefact risk                                                                                                                  |
| Aneurysm                                                                                        | ns (p = 0.40)             | XGB three-way only                                                        | Unstable; do not headline                                                                                                                |


The old F1/F2-only names (`Platelet`, `HL`, `STEMI`, `Current drinking`, `History of HF`, `Hypertension`, `TG`,
`TCL`, `Min-stent diameter`, `Fast-Glu`) are **no longer in consensus**.

### 9.5 The comparison's structural weakness

Part 3 compares:

- a **full-cohort, 92-event, multiplicity-controlled association screen** with
- an **18-event, val-slice, top-20-of-independent-pools predictive shortlist** (88-column encoded matrix).

That is a fairer methods comparison than the old 28-event / nested-40-prefix export, but it is still a
**methods result**, not a biological ranking. Rewrite any sentence that treats Jaccard 0.18 as a finding
about VLST biology.

---

## 10. Complete figure and table inventory

**Regeneration status at a glance.** Part 1 figures were copied from the 2026-08-31 `eda.ipynb` outputs
(categorical SES rates, paper tables, heatmaps, domain Wald panels). Part 2 and Part 3 were rebuilt from the
paper-protocol selector run (`rebuild_part2_paper_figures.py`, `stats_vs_ml_comparison.ipynb`). Part 4's nested-CV export set is **current** for the Kaggle local-TabPFN run (§5.8, §7.1–7.4). Part 4's TSSI leakage supplement is current
(stored single-split metrics, no re-run). Part 5 PNGs remain mixed stale/current.

### Part 1 — EDA (`paper_results/01_eda/EDA_paper_figures_and_tables.md`)


| ID        | Type       | File                                                                                                                                                                                                                                    | Content                                                 |
| --------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Fig 1     | Figure     | `paper_fig1_test_selection_map.png`                                                                                                                                                                                                     | Skew–kurtosis test-selection map, 3 panels              |
| Table R   | Table      | `paper_table_test_rationale.png/.csv`                                                                                                                                                                                                   | 24 continuous variables, chosen test, shape stats, p, q |
| Fig 2     | Figure     | `paper_fig2_univariate_significance.png`                                                                                                                                                                                                | −log10(p) ranking of continuous features                |
| Table 1   | Table      | `paper_table1_continuous_fdr.png` / `paper_table1_continuous_univariate.csv`                                                                                                                                                            | 10 FDR-significant continuous                           |
| Fig 3     | Figure     | `paper_fig3_continuous_effect_sizes.png`                                                                                                                                                                                                | Effect sizes, FDR-significant continuous                |
| Fig 4     | Figure     | `paper_fig4_binary_odds_ratios.png`                                                                                                                                                                                                     | ORs, FDR-significant binary                             |
| Table 2   | Table      | `paper_table2_binary_fdr.png` / `paper_table2_binary_univariate.csv`                                                                                                                                                                    | 10 FDR-significant binary                               |
| Fig 5     | Figure     | `paper_fig5_categorical_rates_Stent_type-SES.png`                                                                                                                                                                                       | VLST rate by stent type (9 collapsed levels)            |
| Table 3   | Table      | `paper_table3_categorical.png/.csv`                                                                                                                                                                                                     | χ² for `Stent type-SES`                                 |
| Table 4   | Table      | `paper_table4_multivariable_or.png/.csv` + `_numeric.csv`                                                                                                                                                                               | 17-covariate adjusted ORs                               |
| Fig 6     | Figure     | `paper_fig6_uni_vs_multivariable_or.png`                                                                                                                                                                                                | Univariate vs adjusted OR                               |
| Fig S5a–b | Supp fig   | `03_correlation_heatmap_top42_vs_next41_with_target.png`, `03b_spearman_correlation_heatmap_top42_vs_next41_with_target.png`                                                                                                            | Pearson / Spearman pairwise heatmaps (with target)      |
| Fig S1    | Supp fig   | `domain_univariate_top_hits.png`                                                                                                                                                                                                        | Top hits per clinical domain                            |
| Table S1  | Supp table | `domain_strength_summary.csv`, `domain_univariate_summary.csv`                                                                                                                                                                          | Domain-level screening summary                          |
| Fig S2a–d | Supp fig   | `domain_clustermap_global/lab/anatomy/procedural.png`                                                                                                                                                                                   | Spearman clustermaps                                    |
| Fig S3    | Supp fig   | `domain_multivariable_or_panels.png`                                                                                                                                                                                                    | Per-domain sparse logistic ORs                          |
| Fig S4    | Supp fig   | `domain_joint_uni_vs_multi_or.png`                                                                                                                                                                                                      | Joint cross-domain model                                |
| Table S2  | Supp table | `domain_interaction_screen.csv`                                                                                                                                                                                                         | 16-pair interaction screen                              |
| —         | Data       | `domain_feature_map.csv`, `paper_domain_feature_map.csv`, `domain_multivariable_or.csv`, `domain_joint_multivariable_or.csv`, `feature_correlation_clusters.csv`, `feature_correlation_cluster_reps.csv`, `domain_vif_cluster_reps.csv` | Supporting CSVs, not referenced as numbered items       |


**[DISCREPANCY]** Table S2 in the Markdown shows **8 rows**; `domain_interaction_screen.csv` contains **16**.
No truncation note.

### Part 2 — classic-ML selectors


| ID      | Type     | File                                           |
| ------- | -------- | ---------------------------------------------- |
| Table 0 | Table    | `paper_table0_classic_models.png/.csv`         |
| Fig 1   | Figure   | `paper_fig1_unique_counts.png`                 |
| Table 1 | Table    | `paper_table1_common_by_algorithm.png/.csv`    |
| Fig 2   | Figure   | `paper_fig2_jaccard.png`                       |
| Table 2 | Table    | `paper_table2_consensus_by_model.png/.csv`     |
| Fig 3   | Figure   | `paper_fig3_consensus_size.png`                |
| Fig 4   | Figure   | `paper_fig4_feature_by_model.png`              |
| Fig 5   | Figure   | `paper_fig5_family_stacked.png`                |
| Fig 6   | Figure   | `paper_fig6_cross_model_common.png`            |
| Table 3 | Table    | `paper_table3_union_by_model.png/.csv`         |
| Fig 7   | Figure   | `paper_fig7_union_by_model.png`                |
| Table 4 | Table    | `paper_table4_global_common.png/.csv`          |
| Table 5 | Table    | `paper_table5_priority_ranks_excerpt.png/.csv` |
| Fig S1  | Supp fig | `selector_model_algorithm_counts.png`          |
| Fig S2  | Supp fig | `selector_top_repeated_features.png`           |
| Fig S3  | Supp fig | `selector_overlap_heatmap.png`                 |


### Part 3 — stats vs ML


| ID      | Type   | File                               |
| ------- | ------ | ---------------------------------- |
| Fig 1   | Figure | `fig1_venn_overlap.png`            |
| Fig 2   | Figure | `fig2_presence_heatmap.png`        |
| Fig 3   | Figure | `fig3_reason_buckets.png`          |
| Fig 4   | Figure | `fig4_domain_counts.png`           |
| Table 1 | Table  | `table_feature_by_method.png/.csv` |
| Table 2 | Table  | `table_shared_features.png/.csv`   |
| Table 3 | Table  | `table_stats_only.png/.csv`        |
| Table 4 | Table  | `table_ml_only.png/.csv`           |


**[CLOSED]** Generating code: `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb`. Jaccard 5/28 and the five-name intersection `{WBC, eGFR, LV, HbA1c, 1.1:1Post dilation}` are asserted in the notebook.

### Part 4 — nested-CV rating  — **current (7 arms; paper_figures match notebook)**


| ID | Type | File | Status |
| --- | --- | --- | --- |
| Table 0 | Table | `paper_table0_models.png/.csv` | Current — seven models including thinking-high |
| Fig 1 | Figure | `paper_fig1_pr_roc_curves.png` | Current — extracted from `de46f92` cell 9; thinking-high AP 0.855 |
| Table 1 | Table | `paper_table1_ranking.png/.csv` | Current — dump L1291–1309 |
| Fig 2 | Figure | `paper_fig2_calibration_curves.png` | Current — thinking-high Brier **0.0064** best; local **0.0673** worst |
| Fig 3 | Figure | `paper_fig3_confusion_matrices.png` | Current — 7-arm pooled panel from notebook cell 13 |
| Table 2 | Table | `paper_table2_nested_operating_point.png/.csv` | Current — nested thinking-high 5076/17/27/65 |
| Table 3 | Table | `paper_table3_pooled_f1.png/.csv` | Current — pooled thinking-high 5072/21/17/75 |
| Counts | Table | `paper_table3_confusion_counts.png/.csv` | Current — nested counts |
| Sweep | Figure | `best_model_threshold_fpfn_panel.png` | Current — best-by-PR-AUC = **TabPFN** (0.8553) |
| Table S-TSSI | Table | `paper_table_s_tssi_leakage.png/.csv` | Current — stored 70/30 metrics, not nested-CV |
| Fig S-TSSI | Figure | `paper_fig_s_tssi_pr_auc.png` | Current — PR-AUC with vs without TSSI |
| Table S-Wang-bins | Table | `paper_table_s_wang_score_bins.png/.csv` | Current — frozen Wang bins vs published n/rates |
| Table S-Wang | Table | `paper_table_s_wang_vs_ml.png/.csv` | Current — thinking-high 0.9905 / 0.8553 added |
| Fig S-Wang | Figure | `paper_fig_s_wang_score_rate.png` | Current — observed VLST rate by integer score |


Do **not** quote leftover `paper_table2_f1_operating_point.*` (deleted; it was a prior pooled export with TabPFN t=0.901).

**Produced by the notebook but absent as committed CSVs:** `model_comparison.csv`,
`nested_cv_operating_point.csv`, `fold_metrics.csv`,
`best_model_threshold_sweep.csv`, `oof_predictions.csv`, `fold_thresholds.csv`.

**[GAP — confirmed by directory listing]** None of the CSV artefacts written to `/kaggle/working/...` were
copied into the repository except the `paper_figures/` exports. `data/result/modeling_results/` contains **only** a `paper_figures/` folder; there is no
`oof_predictions.csv`, `fold_metrics.csv`, `fold_thresholds.csv`, `model_comparison.csv` or
`nested_cv_operating_point.csv` anywhere under `data/result/`. The out-of-fold predictions — which would let a
reviewer recompute every Part 4 number, add confidence intervals, and run DeLong or bootstrap comparisons — do not
exist in the repo. **[TODO-REPRO]**

### Part 5 — TabPFN interpretability

Notebook `645fb0e` is current. Files below in `paper_figures/` (both report trees + `data/result/modeling_tabpfn/`)
were copied from that Kaggle run (B12 closed).


| ID      | Type   | File                                  |
| ------- | ------ | ------------------------------------- |
| Table 0 | Table  | `paper_table0_methods.png/.csv`       |
| Table 1 | Table  | `paper_table1_mutual_info.png/.csv`   |
| Table 2 | Table  | `paper_table2_stability.png/.csv`     |
| Fig 1   | Figure | `paper_fig1_pdp_continuous.png`       |
| Fig 2   | Figure | `paper_fig2_pdp_binary.png`           |
| Table 3 | Table  | `paper_table3_pdp_binary.png/.csv`    |
| Fig 3   | Figure | `paper_fig3_shap_summary.png`         |
| Fig 4   | Figure | `paper_fig4_shap_scatter_age.png`     |
| Fig 5   | Figure | `paper_fig5_shap_bar.png`             |
| Fig 6   | Figure | `paper_fig6_shap_beeswarm.png`        |
| Fig 7   | Figure | `paper_fig7_shap_waterfall.png`       |
| Table 4 | Table  | `paper_table4_shap_mean_abs.png/.csv` |
| Fig 8   | Figure | `paper_fig8_ksii_network.png`         |
| Fig 9   | Figure | `paper_fig9_ksii_upset.png`           |
| Fig 10  | Figure | `paper_fig10_shapiq_force.png`        |
| Fig 11  | Figure | `paper_fig11_shapiq_network.png`      |
| Fig 12  | Figure | `paper_fig12_shapiq_upset.png`        |
| Fig 13  | Figure | `paper_fig13_consensus_ranking.png`   |
| Table 5 | Table  | `paper_table5_consensus.png/.csv`     |


**Totals: 34 figures and 26 tables across the five reports** (including Part 4 S-Wang).

---

## 11. Recommended main text, supplement, and redundant items

Recommendations assume the leakage questions in §4.3 and §12.1 are resolved favourably. If they are not, the
paper's framing changes and this layout does not apply.

### 11.1 Main text (target: 4 figures, 3 tables)


| Slot         | Item                                                                                                                                                                          | Rationale                                                                                                                                                                                                                                         |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Table 1**  | Baseline cohort characteristics, cases vs controls                                                                                                                            | Wang 2020 Table 1 **already is this table** for this cohort. Rebuild it from `VLST.csv` (do not photocopy: post-dilation labelling is inconsistent in Wang, and `LV` / `CaI` need to appear). Cite Wang for the recruitment flow (6,038 → 5,185). |
| **Table 2**  | Part 4 ranking + nested operating point from §7.1 / §7.3 | Use the **7-arm** notebook: thinking-high first on PR-AUC (0.8553), local Brier 0.0673 as a separate row. Repo tables now match. |
| **Table 3**  | Part 1 Table 4 **respecified** — see §12.1; one representative per collinear block, EPV stated                                                                                | Currently unpublishable as written.                                                                                                                                                                                                               |
| **Figure 1** | Part 4 Figure 1 (PR and ROC curves) with the prevalence line, **PR panel first and larger**                                                                                   | The single most important result. PR-AUC leads because prevalence is 1.77%.                                                                                                                                                                       |
| **Figure 2** | Part 4 Figure 2 (calibration), 7-arm notebook                                                                 | Thinking-high Brier **0.0064** is best of seven; local **0.0673** is worst. Do not claim “TabPFN is poorly calibrated” without naming the arm. |
| **Figure 3** | A **new** single figure merging Part 1 Figure 4 (binary ORs) and Part 1 Figure 3 (continuous effect sizes) with **separate panels per effect-size metric**                    | Fixes the d-vs-r comparability problem of §7.8 while keeping the association story in one place.                                                                                                                                                  |
| **Figure 4** | Part 5 Figure 13 (consensus ranking) **or** Part 3 Figure 1 (Venn) — pick one                                                                                                 | Both answer "which variables matter". Two is redundant in the main text. Prefer Figure 13 if the paper's thesis is interpretation; prefer the Venn if the thesis is methods comparison. The Venn is now generated by `stats_vs_ml_comparison.ipynb`.     |


### 11.2 Supplement


| Group                                           | Items                                                                                                                                                                                                 |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Leakage evidence (**now in Part 4 supplement**) | Table S-TSSI and Figure S-TSSI: with-TSSI vs without-TSSI metric contrast (§7.5) and the case vs non-case time distribution (min 1,241 vs min 380 days), framed as binary-ified survival time (§4.2). |
| Published clinical baseline (**now in Part 4 supplement**) | Table S-Wang / Figure S-Wang: frozen Wang 2020 integer score vs nested-CV thinking-high / LightGBM / TabPFN (local) (§5.10). Cox LP, Dangas DCA, and Shantou remain absent (B11). |
| Statistical detail                              | Part 1 Fig 1, Table R, Fig 2, Table 2, Table 3, Fig 5, Fig 6, Fig S5a–b (Pearson/Spearman pairwise)                                                                                                   |
| Domain analysis                                 | Part 1 Fig S1, Table S1, Fig S2a–d, Fig S3, Fig S4, Table S2 (**all 16 interaction rows, not 8**)                                                                                                     |
| Feature-selection methods                       | Part 2 Table 0, Fig 1, Table 1, Table 2, Table 3, Table 4 — each with an explicit note that LOCO's pool is a column-order prefix and that scoring used the test set                                   |
| Stats-vs-ML                                     | Part 3 Fig 1 or 2, Table 1, Tables 2–4                                                                                                                                                                |
| Interpretability                                | Part 5 Table 1, Table 2, Fig 1, Fig 2, Table 3, Fig 5, Table 4, Fig 7, Fig 8 — every caption restated per §12.6                                                                                       |
| Reproducibility                                 | `oof_predictions.csv`, `fold_thresholds.csv`, `fold_metrics.csv`, `nested_cv_operating_point.csv` — **must be regenerated and committed**                                                             |


### 11.3 Redundant — cut or merge


| Item                           | Reason                                                                                                                                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Part 4 Table 3                 | Optimistic pooled F1 companion to Table 2 (honest nested). Keep both only if captions stay distinct; do not merge the counts.                                                                    |
| Part 4 Figure 3                | Pooled confusion heatmaps. Keep with a caption that forbids quoting them instead of Table 2.                                                                                                    |
| Part 2 Figure 7                | Explicitly "same numbers as Table 3" (Part 2 L94).                                                                                                                                                                  |
| Part 2 Figures S1, S3          | Explicitly labelled "Paper restyle: Figure 1" and "Paper restyle: Figure 2" (Part 2 L280, L292). Unstyled duplicates.                                                                                               |
| Part 2 Figure 6                | "Bar height is `n common` from Table 1" (Part 2 L132). A 9-row table as a bar chart.                                                                                                                                |
| Part 5 Figures 10, 11, 12      | Part 5's own caption: "a second view of the **same one-row explanation**, not an independent replication" (L245). Figures 11/12 restyle Figures 8/9. Keep **one** network view of the one-row k-SII, drop the rest. |
| Part 5 Figures 3 and 6         | Figure 6 is "Same 15-row attributions as Figure 3, restricted to the top-9" (L181). Keep one.                                                                                                                       |
| Part 5 Figure 4                | A scatter whose entire content is "almost all points sit at SHAP = 0" (L165) for a feature that is 9/10 stable. It documents a null. Supplement at most.                                                            |
| Part 2 Table 5                 | The caption itself says most rows are **string-matching failures** (`Age, years` vs `Age`) (L247). This is a bug report, not a result. Fix the labels or cut the table.                                             |
| `paper_domain_feature_map.csv` | Byte-identical duplicate of `domain_feature_map.csv`.                                                                                                                                                               |
| Duplicated Markdown trees      | Every report exists twice: `paper_results/**` and `code/**`. Pick one as canonical before any of them drifts further.                                                                                               |


---

## 12. Unsupported claims, ambiguities, contradictions, missing information

### 12.1 [CRITICAL] The exploratory multivariable model is not identified

EDA Table 4 contains, simultaneously:

- `1.1:1Post dilation` **and** `No postdilation` — verified **exact complements** (§3.4). Two parameters for one
bit. The split between adjusted OR 0.144 and adjusted OR 0.895 is determined entirely by the `C=1e6` ridge
penalty, not by the data. **Neither coefficient is interpretable.**
- `eGFR` (continuous) **and** `CKD5` (its stage) **and** `CKD90` (its dichotomisation at 90) — three deterministic
functions of one measurement. This produces `CKD5`'s sign flip (univariate 1.391 → adjusted 0.156, CI
excluding 1 in the **opposite** direction) and `CKD90`'s adjusted OR of 12.54 with a bootstrap CI of
**2.708 – 639.506**, a CI spanning more than two orders of magnitude.
- `3-vessel disease` **and** `NO.of vessels` — the same anatomical construct twice.

Additionally: no variance inflation factors are reported for this model, although
`domain_vif_cluster_reps.csv` exists for the domain models; 200 bootstrap replicates is low; and the 8-binary cap
is an undisclosed specification rule.

**This table cannot be published as it stands.** Refit with one representative per collinear block and report
EPV, or drop the multivariable analysis and present univariate associations only.

### 12.2 [CLOSED] Part 4 `paper_figures/` match the 7-arm notebook

**Notebook.** `code/modeling/rating/baseline_plus_tabpfn.ipynb` on `origin/main` (`de46f92`). Dump
`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt`.

**This snapshot.** Seven arms, 9-level encoder. Thinking-high PR-AUC **0.8553**, ROC **0.9905**,
Brier **0.0064**; nested 5076/17/27/65; pooled t_F1 0.193, 5072/21/17/75. TabPFN (local) Brier **0.0673**,
nested 5041/52/29/63. LightGBM PR-AUC **0.6926**. Embedded notebook figures, dual-tree `paper_figures/`,
and Part 4 Markdown agree.

**Client non-determinism.** This dump Brier 0.0064 sits next to earlier client prints 0.0060 and 0.0360.
Limitations should say the client arm has moved across runs; this dump is internally consistent at 0.0064.
**[TODO-REPRO]** OOF CSVs remain uncommitted.

### 12.3 [CRITICAL] Claims that ROC-AUC and F1 support clinical usefulness

Qualify, do not delete ranking facts:

- Thinking-high **does** lead PR-AUC on this dump (0.8553 vs LightGBM 0.6926; 5/5 folds). That is still not
  clinical usefulness: no decision-curve, no net benefit, no costed threshold, no external test.
- "TabPFN (local) dominates ranking" remains **false** (local PR 0.6754, third among the six non-client
  models if client is set aside; fourth of seven overall).
- Pooled local TP = 76 / thinking-high TP = 75 are optimistic cuts. Honest nested: thinking-high **TP = 65,
  FN = 27**; local **TP = 63, FN = 29**; LightGBM **TP = 61, FN = 31** (§7.3).
- Accuracy is high because negatives dominate; keep that sentence.

### 12.4 [CLOSED in reports] Three Previous-PCI ORs are three estimators, now named

The three numbers are still in the file. They are **not** a single OR printed three times. Part 1 captions now
name the estimator (C13).


| Feature | Table 2 (2×2 / Fisher) | Table 4 univariate column (unweighted logit) | Figure S4 / Table S4 (joint-domain univariate logit) |
| --- | ---: | ---: | ---: |
| Previous PCI | **6.49** (6.485) | **6.46** (6.465) | **6.73** (6.733) |
| 1.1:1Post dilation | 0.187 | 0.187 | 0.192 |
| Diabetes | 1.889 | 1.889 | 1.898 |


**Quote rule.** Name the table when you quote Previous PCI. Table 2 = 2×2 cross-product (Fisher when the
notebook chose Fisher). Table 4 “Univariate OR” = unweighted `statsmodels.Logit`, one covariate. Figure S4 /
Table S4 = univariate column of the joint-domain specification. Do not collapse them to one headline OR.

The old wording that Table 4’s univariate column was class-weighted is **wrong for the current reports**:
Table 4 is unweighted MLE (`class_weight="balanced"` is not used there).

### 12.5 [CLOSED for Part 4] `Stent type-SES` encoding — Part 2 and Part 4 use the 9-level encoder; Part 5 stored SHAP/PDP still mixed

**[REV6]** `baseline_plus_tabpfn.ipynb` was re-run with `code/modeling/tools/stent_encoding.py`
(canonicalize aliases, collapse n < 30 → `other`, **9 levels**). Classics one-hot those 9 levels without
`drop="first"` (~89). TabPFN (local) sees the 9-level column natively.

**[REV5]** `eda.ipynb` and `baseline_feature_selections.ipynb` already used that encoder. Part 2 scaled width is
**88** (drop-first). EDA χ²: 99 raw levels → 9 used, χ² = 44.90, df = 8, V = 0.093.
`PES` / `ZES` / `EVS` stay as the drug-class partition.

Part 5 **notebook** SHAP/PDP now use the 9-level encoder (`645fb0e`). Stored `paper_figures/` were copied from
that run (B12).


| Notebook | Code now | Stored width |
| --- | --- | --- |
| `eda.ipynb` | Shared 9-level encoder at load | 9 |
| `preprocessing.ipynb` | Same encoder; column name kept; then OHE | 88 if last run used `stent_brand` |
| `baseline_feature_selections.ipynb` | Shared encoder, then OHE drop-first | **88** (2026-08-31 run) |
| `baseline_plus_tabpfn.ipynb` (classic arm) | Shared encoder, then ColumnTransformer OHE (no drop-first) | **~89** (this Kaggle run) |
| `baseline_plus_tabpfn.ipynb` (TabPFN local) | Same 9-level column, native | 81 columns, 9 brand levels |
| `tabpfn_interpretability.ipynb` | 9-level codes; brand dropped from continuous PDP | 81; **notebook and stored Fig 1 current** |

The `preprocessing.ipynb` artefacts (`X_train.npy`, `preprocessor.joblib`) are used only by the
TSSI scout notebooks. Re-run preprocessing before those scouts if you want matching brand dummies.

### 12.6 [CRITICAL] Interpretability claims that overreach their sample


| Claim                                                                             | Actual basis                                                                                                                                                                                                                    | Required restatement                                                                                                                                                                           |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Part 5 Figures 3, 5, 6 and Table 4 presented as global SHAP importance | **Notebook and stored PNGs:** 15 VLST=1 + 15 VLST=0, client thinking, `Explaining all 30 rows` | Quote the 30-row protocol. Not global SHAP on 5,185 rows. |
| Part 5 Figures 8, 9, 11, 12 — "dominant pairwise terms" among LV/WBC/eGFR/LDL | **One** VLST=1 patient (row 5099), budget 256 | Keep as illustration, not cohort interactions. |
| Part 5 Figure 1 / Table 3 0.24 / ~0.6 risk | **Notebook and stored PNGs:** empirical prior, P(y=1) ≈ 0.017–0.023 | Do not quote 0.24 or 0.6. Largest binary Δ is `1.1:1Post dilation` −0.0043. |
| Part 5 Table 5 `Cre` / stents MI = 0 | **Notebook and stored CSV:** `Cre` MI 0.002281; all 81 scores saved | Fill-zero issue closed (B6). |
| Part 5 Table 1 `Fast-Glu` / `ZES` as "—" | **Notebook and stored CSV:** Fast-Glu 10th 0.0044; ZES 13th 0.0038 | **[TODO-MI closed]** |
| Part 5 Table 2 caption "8.6 h" | ~8.56 h on this dump ([1/5] 30824 s) | Fine. |


### 12.7 [CLOSED] Part 2 Table 0 CatBoost spec

**[REV5]** Table 0 PNG was re-exported with GPU **Plain** boosting and `eval_metric=PRAUC`.

### 12.8 [CLOSED] Part 2 figures are the paper-protocol run

The notebook is a single paper protocol (top-20, SHAP 40 permutations, LOCO 60, FFS 24×12, 400 rounds).
**Stored figures (2026-08-31)** match that protocol: PR-AUC only, independent selectors, 88 columns,
fit/val 4148/1037. Kaggle long CSVs were not committed; Harmony panels were rebuilt from notebook
displays (`rebuild_part2_paper_figures.py`). Do not quote the old F1/F2 / 0.95 Jaccard / 185-column
story.

### 12.9 [RISK] Unquantified uncertainty everywhere

- No confidence intervals on **any** PR-AUC, ROC-AUC, or Brier score.
- No paired statistical comparison between models (no DeLong, no bootstrap difference test). "Thinking-high is first on PR-AUC" and "TabPFN (local) is fourth of seven" are point-estimate orderings. Fold SDs (§7.1): thinking-high PR 0.8488 ± 0.0861 vs LightGBM 0.6936 ± 0.0915; thinking-high higher in **5/5** folds. Local vs LightGBM is still 2/5. No test was run.
- No confidence intervals on precision/recall/F1 at any operating point.
- No calibration slope or intercept; only Brier and a visual reliability curve.
- Univariate effect sizes in Table 1 and Table 2 carry no confidence intervals — only p and q.

### 12.10 [GAP] Reproducibility


| Missing                                                                                   | Impact                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `oof_predictions.csv` / `fold_thresholds.csv` not in repo (written to `/kaggle/working/`) | No reviewer can recompute Part 4 or add CIs                                                                                                                                   |
| No `environment.yml` / lockfile; `requirements.txt` has no pins verified                  | TabPFN, shapiq, CatBoost versions unknown                                                                                                                                     |
| Stored Part 4 TabPFN is a local checkpoint (`tabpfn-v3-classifier-v3_default.ckpt`)       | Pin `tabpfn` package version; the unused thinking-high client/server versions remain unrecorded                                                                               |
| Unused client thinking-high arm is non-deterministic across runs (§12.2) | **This dump ran the client** (Brier 0.0064). Earlier client prints 0.0060 / 0.0360 still belong in Limitations. Local Brier remains 0.0673. |
| No data-availability, ethics, or consent statement **in this repo**                       | Wang 2020 already has all three (NCT03491891, ethics 2013-256, written consent, figshare data statement). Cite them; still put a one-paragraph statement in *this* manuscript |
| Notebooks explicitly excluded from the results pack (`paper_results/README.md`)           | The pack cannot be audited on its own                                                                                                                                         |


### 12.11 [WITHDRAWN under D1–D2] Selective reporting

Revision 1 argued that because many notebooks exist and five analyses are reported, the TabPFN advantage
"cannot be distinguished from selection over many attempts", and demanded a CONSORT/TRIPOD-style declaration of
everything tried. **That finding is withdrawn.** The excluded notebooks (ten `failed_hypothesis/`, the TabPFN
playground) are out of scope by author decision, contribute no number to the paper, and are
not cited anywhere in it.

One sentence of residual substance, kept for honesty and needing no table: exploratory work beyond the reported
seven notebooks did execute against the same 92 events, so if a reviewer asks what else was tried, the accurate
answer is "exploratory analyses that were abandoned and are not reported" — not "nothing".

The stronger version of this concern, which does still bind, is not about notebook count at all: it is that no
paired significance test separates thinking-high TabPFN from LightGBM (§12.9), and the ML models have no external validation
(§6.6) even though Wang's Cox score does.

### 12.12 Terminology discipline for the manuscript


| Term                             | Reserve for                                                                                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Association**                  | Part 1 univariate and multivariable results; Part 5 mutual information. Full-cohort, no held-out evaluation.                                         |
| **Prediction**                   | Part 4 nested-CV out-of-fold results **only**. This is the only genuinely out-of-sample evaluation in the repository.                                |
| **Interpretation / attribution** | Part 2 selectors, Part 5 SHAP / k-SII / PDP / stability. Model-explanatory, not evidence about patients.                                             |
| Never use                        | "risk factor", "causal", "protective", "independent predictor", "clinically useful", "validated" — none is supported by anything in this repository. |


Note in particular that `1.1:1Post dilation` has an adjusted OR of 0.144 and a negative empirical-prior PDP
shift of **−0.0043**, and that `Clopidogrel` has an OR of 0.464. In an observational cohort with
confounding by indication, neither is a treatment benefit. Reports no longer use “protective” for these flags.
Do not quote the old balanced-prior Δ of −0.086 as this run.

---

## 13. What is left to do

Grouped by who can do it and what it costs. **Group A can only be answered by you** and gates the paper's framing.
Groups B–E are execution work.

### Closed by this revision (including Wang 2020)


| Was                                                        | Now                                                                                                                            |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| "Which TabPFN run is authoritative?" (old blocking item 4) | **Revision 7:** quote `de46f92` 7-arm dump. Thinking-high PR **0.8553** / Brier **0.0064** / nested 5076/17/27/65. Local PR 0.6754 / Brier 0.0673 / nested 5041/52/29/63. Rev 6 six-model local-only is historical. |
| **C1–C16** false or mixed claims in reports                | **C1–C4, C6–C7, C9, C14 closed in Rev 7 reports** after B1/B12 copy. Remaining C items are earlier closures. |
| "Fold-wise mean ± SD are not available"                    | **Closed in the notebook** (§7.1, §7.4). |
| **A2** How were controls sampled?                          | **Closed by Wang 2020.** Consecutive complete-follow-up cohort, not case-control. 1.77% is published incidence (§2.3, §4.2).   |
| **A3** What does `Stent thrombosis = 1` mean?              | **Closed by Wang 2020.** ARC 2007 definite ST, angiographically confirmed, > 1 year (§2.3).                                    |
| **A5** Recruitment frame, ethics, consent                  | **Closed by Wang 2020.** Jilin University, Jan 2014–Jun 2015, NCT03491891, ethics 2013-256 (§2.3).                             |
| **B10** Wang integer score as comparator                   | **Closed.** Frozen points ROC 0.8013 / PR 0.1032 vs thinking-high **0.9905 / 0.8553**, LightGBM 0.9680 / 0.6926, local 0.9845 / 0.6754 (§5.10). Cox LP / DCA / Shantou still out (B11). |


### A. Remaining author questions after Wang 2020


| #      | Question                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Status                                                                                                                              |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **A1** | **Lab/echo timing, now specific.** Wang 2020 treats WBC, LVEF, lipids, fibrinogen and eGFR as **index-PCI baselines** and built a Cox score from them, so event-time measurement is no longer the leading hypothesis for those columns. Two things are still on you: (1) confirm `LV` — name, units, timing — because it is absent from Wang Table 1 and is a top ML feature; (2) decide how to discuss **WBC**, which Wang *excluded* from the score because infection could not be ruled out, and which our models rank at the top. | Narrowed. Does not gate the whole paper the way it did, but gates any sentence that treats `LV` or WBC as a novel validated marker. |
| **A4** | `LV` **and** `CaI` **only.** `TCL` = total cholesterol (mmol/L), `HL` = dyslipidaemia, fibrinogen in g/L, stent release pressure in atm, DAPT = mandated ≥1 year plus physician-directed continuation — all from Wang. `LV` and `CaI` are still unnamed.                                                                                                                                                                                                                                                                              | Partial.                                                                                                                            |


### B. Re-run or compute (roughly in dependency order)


| #                         | Task                                                                                                                                                                                                                                                                                         | Cost                                                                                                                                                            | Notes                                                                                                                                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **B1** [TODO-P4 — **closed**] | **Re-export all of Part 4** from the 7-arm notebook (`de46f92`): Figures 1–3, Tables 0–3, sweep panel. | Done — notebook cell PNGs + rebuilt tables in both trees + `data/result/modeling_results/` | Thinking-high-first 7-arm figures. |
| **B2** [TODO-REPRO]       | **Persist** `oof_predictions.csv`, `fold_thresholds.csv`, `fold_metrics.csv`, `model_comparison.csv`, `nested_cv_operating_point.csv`. Written on Kaggle; **absent from repo**. | Low | Needed for CIs (B3). |
| **B3** [TODO-CI]          | **Bootstrap CIs** and a paired test between thinking-high TabPFN and LightGBM (primary), and local vs LightGBM (secondary). | Low once B2 exists | Thinking-high higher PR in 5/5 folds is the interim fallback. |
| **B6** [TODO-MI — **closed**] | `[1a]` wrote all 81 MI scores; `Fast-Glu` / `ZES` in top 15. Table 1/5 PNGs/CSVs exported from `645fb0e`. | Done | `Cre` MI 0.002281 stored; no fill-zero. |
| **B4** [TODO-T4]          | **Refit EDA Table 4** with one representative per collinear block; report VIFs and EPV; raise `N_BOOT` from 200 to ≥ 2,000.                                                                                                                                                                  | Low                                                                                                                                                             | As it stands the model is not identified: `1.1:1Post dilation` sits beside its exact complement, and `eGFR` beside `CKD5` and `CKD90` (§12.1). `CKD90`'s CI is 2.708–639.506. **This table cannot be published as written.**  |
| **B5** [TODO-P3 — closed] | **Write the missing Part 3 notebook.** Done: `stats_vs_ml_comparison.ipynb`. Jaccard 5/28 asserted after the 2026-08-31 catalogues. | Done | Inputs are the §8.1 FDR set and §8.6 ML consensus. Re-run the notebook if either catalogue changes. |
| **B8** [CLOSED]           | **Part 2 paper protocol re-run on Kaggle** (2026-08-31): full-cohort discovery, PR-AUC only, independent selectors, top-20 / SHAP 40 / LOCO 60 / FFS 24×12. Parts 2–3 figures replaced. Download `selector_summary_long.csv` when convenient (tables were reconstructed from notebook HTML). | Done | Part 2/3 still do not support a *predictive* claim — they are discovery catalogues on a val slice (§4.4, §9.5). |
| **B7** [TODO-TABLE1]      | **Rebuild Table 1 from** `VLST.csv`, including `LV` and the variables Wang omitted, and cite Wang for the 6,038 → 5,185 flow. Do not photocopy Wang Table 1 (post-dilation label is inconsistent, §3.3).                                                                                     | Low                                                                                                                                                             | A conventional Table 1 already exists in Wang 2020; the repo still needs a verified, ML-complete version.                                                                                                                     |
| **B9** [encoding — closed] | **Part 4 and Part 5 notebooks use the 9-level encoder.** Optional: tune classics. | Encoding done | Untuned classics vs thinking-high is disclosed (§6.3). |
| **B10** [TODO-SCORE — closed] | Wang integer score on 5,185 rows: ROC-AUC 0.8013, PR-AUC 0.1032 vs nested-CV thinking-high **0.9905 / 0.8553**, LightGBM 0.9680 / 0.6926, local 0.9845 / 0.6754. | Done | Cox LP / DCA / Shantou still B11. |
| **B12** [TODO-P5 — **closed**] | **Copy Part 5 Kaggle artefacts** (`interpretability_*.csv/png`, SHAP 30-row indices) into both `paper_figures/` trees from `645fb0e`. | Done | Dual tree + `data/result/modeling_tabpfn/`. |
| **B11** [TODO-EXT]        | **Ask for the Shantou n = 2,058 file.** If it exists, it is the external test set Wang already used.                                                                                                                                                                                         | Political, not computational                                                                                                                                    | Without it, state clearly that ML validation is derivation-cohort nested CV only.                                                                                                                                             |


### C. Rewrite or delete — claims that are now known to be wrong

The sixteen items were closed against **Rev 6 reports**, then C1–C4 / C6–C7 / C9 / C14 were re-closed against
**Revision 7 reports** after the B1/B12 figure copy. Remaining C rows below that still say REV4/REV5 are earlier
closures that were not reopened.


| #       | Where                                                                                     | Action                                                                                                                                                                                                                                                         |
| ------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C1**  | Part 4 Table 1 / Figure 2 Brier                                                           | **[REV7/closed in reports]** Thinking-high Brier **0.0064** (best of seven); local **0.0673** (worst). Dual-tree PNGs match.                                                                                                                               |
| **C2**  | Part 4 event counts                                                                       | **[REV7/closed in reports]** Nested thinking-high 5076/17/27/65 (recall 0.7065). Local nested 5041/52/29/63. Pooled local TP=76 is Table 3 only.                                                                                                           |
| **C3**  | Part 4 provenance / fold SD                                                               | **[REV7/closed in reports]** Fold SD and thinking-high row are in Table 1. OOF CSVs still uncommitted (B2).                                                                                                                                               |
| **C4**  | Part 4 Tables 2–3                                                                         | **[REV7/closed in reports]** Thinking-high nested/pooled rows present; nested vs pooled labels kept.                                                                                                                                                      |
| **C6**  | Part 5 SHAP captions                                                                      | **[REV7/closed in reports]** 15+15 / client thinking; Cre leads mean(\|SHAP\|).                                                                                                                                                                            |
| **C7**  | Part 5 PDP captions                                                                       | **[REV7/closed in reports]** Empirical prior (~0.018); largest binary Δ −0.0043.                                                                                                                                                                           |
| **C5**  | Part 3: "domain multivariable OR persists" for `LVEF`                                     | **[REV4/closed in reports]** Fixed: Part 3 states the sign reversal.                                                                                                                                                                                           |
| **C8**  | Part 5 k-SII captions (Figures 8–12)                                                      | **[REV4/closed in reports]** Already one-row; Fig 8 no longer calls the blue node a cohort benefit.                                                                                                                                                            |
| **C9**  | Part 5 Table 5 caption                                                                    | **[REV7/closed in reports]** MI from full 81-row ranking; `Cre` 0.002281 not fill-zero. |
| **C10** | Anywhere "protective" appears — `1.1:1Post dilation` (OR 0.144), `Clopidogrel` (OR 0.464) | **[REV4/closed in reports]** Word removed from paper-style reports (Parts 1, 3, 5 and the concatenated bundle). OR < 1 / negative PDP is association or model output, not a treatment benefit (§12.12). Audit text below still names the banned word.          |
| **C11** | Part 2 Table 0: CatBoost "Ordered boosting"                                               | **[REV5/closed]** Markdown + CSV + PNG: GPU **Plain**, `eval_metric=PRAUC`.                                                                                                    |
| **C12** | Part 1 Figure 3 / Table 1 effect-size column                                              | **[REV4/closed in reports]** Caption: Cohen's d and Mann–Whitney r are different metrics; do not compare bar lengths (`WBC` r = 0.13 vs `LV` d = 1.13). Splitting the PNG into two panels still needs an EDA re-export.                                        |
| **C13** | The three different "univariate OR" values for `Previous PCI` (6.485 / 6.465 / 6.733)     | **[CLOSED]** Reports name the estimator. Evidence-map §12.4 no longer treats this as an unlabeled discrepancy. Table 2 = 2×2/Fisher (**6.49**); Table 4 = unweighted logit (**6.46**); Table S4 = joint-domain univariate (**6.73**).                             |
| **C14** | `Stent type-SES`                                                                          | **[REV7/closed in reports]** EDA, Part 2, Part 4, and Part 5 use the 9-level encoder (Fig 1 drops brand from continuous PDP). Wang binary SES remains a third encoding. |
| **C15** | Any statement of the form "LOCO saturates the 40-feature cap"                             | **[REV5]** Replaced: LOCO unique count is 60 because 60 columns were scored, not because 60 were independently important.                                                                                                         |
| **C16** | Part 1 Table S2                                                                           | **[REV4/closed in reports]** All **16** pairs from `domain_interaction_screen.csv` are shown; only LV×eGFR and Men×eGFR pass FDR.                                                                                                                              |


### D. Content that must be written from scratch (items W1–W5, to avoid clashing with the scope decisions D1, D2, D4)


| #                           | Item                                                                                                                                                                                                                                                                                                                                                                                                               | Notes                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **W1** [TODO-LEAK — closed] | **The leakage section** — with-TSSI vs without-TSSI contrast is now Part 4 Table S-TSSI / Figure S-TSSI (logistic PR-AUC 0.958 → 0.508; CatBoost 0.977 → 0.658) plus the event vs follow-up time distribution (controls min 1,241 days, cases min 380), framed as binary-ified survival time with Wang's Cox analysis as the design-correct alternative.                                                           | Closed. Nothing was re-run (§4.1, §4.2, §7.5). Methods paragraph is in the Part 4 header. |
| **W2** [closed in reports]  | **Clinical motivation and citations.** Drafted in `paper_results/00_front_matter.md` (and concatenated `paper_results.md` Part 0): Wang 2020 definition/cohort/score; why TabPFN; what “personalised” does *not* mean; what this pack adds *beyond* Wang’s 8-variable Cox score. Frozen integer points are now scored (Part 4 S-Wang; B10 closed for that comparator). | Cite Wang; do not write as if no VLST score exists.                                       |
| **W3** [closed in reports]  | **Limitations** Rev 7 pass: thinking-high ran (Brier 0.0064, PR 0.8553, 5/5 folds vs LightGBM); local Brier 0.0673 still worst; client non-determinism across dumps; no CIs; no external ML test. | Drafted in Part 0. |
| **W4** [TODO-EPV — closed]  | **EPV stated explicitly** (92 / 17 ≈ **5.4**) in Part 0 and next to Part 1 Table 4 / Figure S4 adjusted ORs (both report trees + concat).                                                                                                                                                                                                                                                            | Was computed nowhere (§2.2).                                                              |
| **W5** [closed in reports]  | **Terminology pass** enforcing §12.12 in Part 0 and part headers: association (Parts 1, 5 MI); prediction (Part 4 nested CV only); interpretation/attribution (Parts 2, 5). Banned words removed from captions (“risk factor”, “independent signal”). Wang’s score remains the only result called externally tested.                                                                                  |                                                                                           |


### E. Housekeeping


| #      | Item                                                                                                                                                                                     |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **E1** | **Designate one canonical report tree.** Every report exists twice (`paper_results/`** and `code/`**); they will drift.                                                                  |
| **E2** | Delete `paper_domain_feature_map.csv` — byte-identical duplicate of `domain_feature_map.csv`.                                                                                            |
| **E3** | Fix or cut Part 2 Table 5: its rows are string-matching failures (`Age, years` vs `Age` → rank `NaN`), visible in the notebook output itself at L1502–1522. This is a bug, not a result. |
| **E4** | Pin package versions (`environment.yml` or a lockfile) and record the TabPFN client **and** server-side model version.                                                                   |
| **E5** | Add data-availability, ethics and consent statements **to this manuscript**, citing Wang 2020 (NCT03491891, ethics 2013-256, written consent, figshare).                                 |
| **E6** | Cut the redundant items in §11.3 (Part 4 Table 3 and Figure 3; Part 2 Figures 6, 7, S1, S3; Part 5 Figures 10–12, and one of Figures 3/6).                                               |


### Suggested order

1. **A1/A4 (`LV`, WBC vs Wang's exclusion) in parallel with B2 (commit OOF CSVs).** Lab timing no longer gates the whole paper; `LV` still gates any claim that names it. Part 4/5 PNG export (B1, B12) is closed for this 7-arm / 15+15 snapshot.
2. **B11** (Shantou file) as a data-access ask, not a compute task. B10 (Wang integer score on the derivation file) is closed.
3. **B4, B7** — both cheap. B7 is "rebuild and verify against Wang Table 1." (B5 / Part 3 notebook and B6 MI export are done.)
4. **C1–C16 report wording is done for Parts 4–5** against Revision 7. Optional leftover: C12 split-axis Figure 3.
5. **W1–W5 are drafted** in `paper_results/00_front_matter.md` (Part 0 of `paper_results.md`) and Part 1 Table 4. Remaining B-list: B2 OOF CSVs, B3 CIs once B2 exists, B4 Table 4 refit, B7 clinical Table 1, B11 Shantou file.

