# VLST paper — evidence map

**Purpose.** Pre-writing audit of every claim, number, feature set, figure and table that currently exists in this
repository, with its exact provenance. Nothing here is new analysis. Where the repository does not contain something,
this document says so explicitly rather than filling the gap.

**Revision 2.** This version applies four author scope decisions (§0) and re-resolves every code-vs-Markdown
conflict in the code's favour. Numbers that changed as a result are flagged in place.

**Revision 3.** Cohort provenance is no longer a repository-only problem. Wang et al., *Sci Rep* 2020;10:6378
(doi:10.1038/s41598-020-63455-0) — cited below as **Wang 2020** — published a Cox VLST risk score on a derivation
cohort of **5,185 ACS-PCI patients with 92 events (1.77%)**, identical to `data/raw/VLST.csv`. Several §2 / §3 / §4
gaps that this map treated as unanswerable from the repo are answered there. What remains unanswered is listed
explicitly.

**Revision 4.** Paper-style Markdown reports (both trees + `paper_results/paper_results.md`) were aligned to the
*code* on methodology. Imputers are described as inert (no missing values). Part 4 now states the unequal
feature view (classics ~186 scaled one-hot columns vs TabPFN raw 81) and that GridSearch winners are not
imported. Part 5 now states MI/SFS = full cohort, SHAP = 15 VLST cases by construction, shapiq `imputer` ≠
NaN fill, and `balance_probabilities=True` (not absolute risk). Part 2/3 now distinguish the **stored smoke
run** (column-order LOCO pool; 87.5% SHAP sample) from **current selector code** (inner-val; train-importance
pool). Part 3 no longer says the LVEF adjusted OR “persists.” Exported Part 4 PNGs remain **[STALE]** for
TabPFN Brier / confusion counts; captions now quote the notebook.

**Audit scope.** The paper-style Markdown reports, both copies of each (`paper_results/`** and `code/**`), the
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



### 0.1 The four decisions


| #      | Decision                                                                                                                                                                                                                                                                                         | Effect on this audit                                                                                                                                                               |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D1** | `code/failed_hypothesis/`** **is out of scope** — all ten notebooks (`anomaly_detection`, `baseline_blending`, `baseline_plus_tabpfn_blending`, `ffs`, `fp_precision_mining`, `llm_tabular_small_n`, `tabpfn_5fold_fp_mining`, `tabpfn_fp_followup`, `tabpfn_oversampling`, `tabpfn_synthesis`). | The selective-reporting finding built on this directory is **withdrawn** (§12.11).                                                                                                 |
| **D2** | **TabPFN comes from exactly two notebooks:** `rating/baseline_plus_tabpfn.ipynb` (performance) and `interpretability/tabpfn_interpretability.ipynb` (interpretability). `rating/tabpfn_playground.ipynb` is out of scope.                                                                        | No other file may be cited for a TabPFN number.                                                                                                                                    |
| **D3** | `analyzes/causal_analysis.ipynb` **is out of scope** — not needed for paper results.                                                                                                                                                                                                             | No causal/ATE/ATT claim enters the paper. Rationale recorded in §5.10.                                                                                                             |
| **D4** | **The Markdown reports remain part of the analysis and are read on their merits — but where a report and the code disagree, the code is authoritative.**                                                                                                                                         | Every `[DISCREPANCY]` below now carries an explicit resolution. Reports still count as the source for *reasoning, framing and caveats*; the notebooks are the source for *values*. |




### 0.2 The seven in-scope notebooks


| #   | Notebook                                                      | Written up as | Export status                              |
| --- | ------------------------------------------------------------- | ------------- | ------------------------------------------ |
| 1   | `analyzes/eda.ipynb`                                          | Part 1        | verified against notebook                  |
| 2   | `modeling/interpretability/baseline_feature_selections.ipynb` | Part 2        | verified against notebook                  |
| 3   | `modeling/rating/baseline_plus_tabpfn.ipynb`                  | Part 4        | **[STALE]** for TabPFN only (§12.2)        |
| 4   | `modeling/interpretability/tabpfn_interpretability.ipynb`     | Part 5        | verified except two blank MI cells (§12.6) |
| 5   | `modeling/rating/baseline_tssi_leakage.ipynb`                 | Part 4 supp.  | stored metrics → Table S-TSSI              |
| 6   | `modeling/rating/baseline_without_tssi.ipynb`                 | Part 4 supp.  | stored metrics → Table S-TSSI              |
| 7   | `modeling/preprocessing/preprocessing.ipynb`                  | **nothing**   | artefacts unused by any analysis (§12.5)   |


Twelve of the nineteen notebooks on disk are excluded by D1–D3 and are not audited.

### 0.3 Part 3 now has generating code

Part 3, the statistics-vs-ML comparison, previously had **no notebook and no script**. That is closed.

Generating code: `code/analyzes/stats_vs_ml/rebuild_comparison.py`, invoked by
`code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb`. It recomputes the overlap from the verified
§8.1 statistical FDR set and the §8.6 ML consensus set, asserts Jaccard = 5/35 = 0.1429 with intersection
`{WBC, eGFR, LV, Fiberinogen, Previous PCI}`, and writes Figures 1–4 plus Tables 1–4 to both
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
| Existing score           | Dangas LST score (2012) also used for VLST, **c-statistic 0.66**; Wang's own 8-variable Cox score **c = 0.80 / 0.82** (derivation / Shantou validation) | The natural comparator this ML paper must beat (§13 B10). Do not write as if no VLST score exists |
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
Tables 1–4 exactly. So D4 does **not** invalidate the reports wholesale. It changes exactly two things: the TabPFN
rows of Part 4 (§12.2) and the handful of narrative claims listed in §13.C.

### 2.2 Events per variable


| Comparison                                   | Value              | Consequence                                                           |
| -------------------------------------------- | ------------------ | --------------------------------------------------------------------- |
| Events / candidate features (81)             | 92 / 81 ≈ **1.14** | Far below any conventional threshold                                  |
| Events / multivariable-model covariates (17) | 92 / 17 ≈ **5.4**  | Below the conventional EPV ≥ 10 rule                                  |
| Events in the 30% hold-out test set          | **28**             | Feature-selection metrics in Part 2 are estimated on 28 events        |
| Events per outer CV fold (Part 4)            | 18, 18, 18, 19, 19 | `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L1030–1060 |


**[RISK]** The EDA Markdown says the multivariable model "is sparse and intended for screening/confounding context"
(`paper_results/01_eda/EDA_paper_figures_and_tables.md` L176) — a fair caveat, and under D4 it stands. But EPV is
computed nowhere in `eda.ipynb` and appears in no report. **[TODO-EPV]** State it explicitly (92 / 17 ≈ 5.4) next
to every adjusted odds ratio.

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
| `Stent type-SES`                               | **Partially solved, encoding still a mess.** Wang treated SES as a **binary class flag** (sirolimus-eluting stent; 68.76% vs 82.61%). In this repository the same column holds **106 free-text brand strings**. Wang's published analysis used the collapsed class; our notebooks do not (§12.5).                                                                                                                                 |
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

**Code fix applied; not yet re-run.** The stored Part 2 figures/tables and the Part 3 consensus still come from
the test-scored smoke run below. `baseline_feature_selections.ipynb` now scores LOCO / SHAP / FFS on an inner
hold-out of *train* (`INNER_VAL_SIZE=0.2`), ranks the LOCO cap by cheap train importance (not column order), and
sets `USE_CACHE=False`. Re-run on Kaggle, then rebuild Part 3 from the new catalogues.

Historical (stored) leak, for provenance of current figures:


| Selector       | Evaluation target (stored run)                                                                                                               | Source                                                                                 |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| LOCO           | `metric_score(y_test, …)` — the 1,556-row / 28-event hold-out                                                                                | `.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L521, L532 |
| Coalition SHAP | `metric_score(y_exp, …)` where `y_exp` are selected **test** rows; the candidate universe is the LOCO top-24, itself test-derived            | same file, L556, L561–567, L584                                                        |
| FFS            | Honest inner hold-out (20% of train, ≈ 726 rows, ≈ 13 events) — **but** restricted to the LOCO top-30 pool, so it inherits the contamination | same file, L621–634, L647                                                              |


SHAP and FFS “read what LOCO returned” **on purpose**: a compute cap (full 185-column SHAP/FFS is too slow). That
is not a second model copying LOCO. The stored run nested them inside a *column-order prefix* that was also
test-scored. After the code change they still nest, but the pool is train-importance-ordered and inner-val scored.

Until Kaggle re-export, the "ML consensus catalogue" in Part 3 remains a **discovery** list selected against the
same 28 test events, and no generalisation claim can rest on it.

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


| Stage                                | Notebook                                                           | Reported in                            |
| ------------------------------------ | ------------------------------------------------------------------ | -------------------------------------- |
| Statistical EDA                      | `code/analyzes/eda.ipynb`                                          | Part 1                                 |
| Classic-ML feature selection         | `code/modeling/interpretability/baseline_feature_selections.ipynb` | Part 2                                 |
| Stats-vs-ML comparison               | `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb` (+ `rebuild_comparison.py`) | Part 3                                 |
| Nested-CV baselines + TabPFN         | `code/modeling/rating/baseline_plus_tabpfn.ipynb`                  | Part 4                                 |
| TabPFN interpretability              | `code/modeling/interpretability/tabpfn_interpretability.ipynb`     | Part 5                                 |
| Leaky baselines (with TSSI)          | `code/modeling/rating/baseline_tssi_leakage.ipynb`                 | Part 4 Table S-TSSI                    |
| Baselines without TSSI, single split | `code/modeling/rating/baseline_without_tssi.ipynb`                 | Part 4 Table S-TSSI                    |
| Preprocessing artefacts              | `code/modeling/preprocessing/preprocessing.ipynb`                  | not reported; artefacts unused (§12.5) |


Excluded by D1–D3 and not audited: the ten `code/failed_hypothesis/*.ipynb`, `rating/tabpfn_playground.ipynb`,
and `analyzes/causal_analysis.ipynb`.

**Selective-reporting finding withdrawn.** Revision 1 raised a `[RISK — selective reporting]` because twelve of
nineteen notebooks were unreported. D1–D3 declare those twelve out of scope, so the previous framing — that the
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
- **Feature × feature** — Pearson/Spearman heatmaps (`03_*`, now Supplementary Figure S5) and domain Spearman clustermaps (Figure S2). Multicollinearity / clustering, not a second FDR set.
- **Pair × VLST** — the limited interaction screen of §5.5 (Table S2).

The correlation heatmap is admissible as bivariate analysis of *feature–feature* structure. It is not a substitute for the Welch / Mann–Whitney / Fisher screens.

### 5.4 Multivariable logistic regression (Part 1, Table 4)

Implementation, from `.nbdump/code__analyzes__eda.txt`:


| Element            | Value                                                                                                                             | Line            |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------- | --------------- |
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
  - TabPFN, **no hyperparameter tuning** (see §6.3).



### 5.7 Classic ML feature selection (Part 2)

- Feature view: raw one-hot of all 81 columns → **185 columns**
(`.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L223).
- Run mode: `RUN_MODE = "smoke"` with `FEATURE_TOPK = 12`, `SHAP_UNIVERSE = 24`, `LOCO_MAX_FEATURES = 40`,
`FFS_CANDIDATE_POOL = 30`, `SHAP_N_PERM = 12`, `SHAP_N_INSTANCES = 32`, `SHAP_N_BACKGROUND = 32` (L90–97).
- Seven classic models: `lr`, `rf`, `rf_b`, `cat`, `xgb`, `xgb_b`, `lgb`. TabPFN was configured but unavailable
in the stored run (L446–447).
- Three objectives: `pr_auc`, `f1`, `f2`.

**[CRITICAL — stored LOCO is not a ranking; code has changed]** The **stored smoke dump** set
`order = list(range(n))` then truncated to the first 40 (old L524–526): the first 40 **columns in
ColumnTransformer output order**. Every published "ML consensus" name in Parts 2 and 3 is still from that
prefix. The **current** notebook ranks the cap by cheap train importance and scores on an inner hold-out of
train. Not re-run. Part 2/3 markdowns now say so; Figure 1's caption no longer reads the 40-cap as a finding.

**[CRITICAL — stored SHAP sample is 87.5% positive; code has changed]** The stored `run_shap` used all 28 test
positives plus 4 random negatives, then scored PR-AUC/F1/F2 on those 32 rows. Current code uses a stratified
inner-val sample at train prevalence. Not re-run. Part 2 markdown now discloses the stored sample.

### 5.8 TabPFN (Part 4)

- `tabpfn_client.TabPFNClassifier(thinking_mode=True, thinking_effort="high", thinking_metric="average_precision", random_state=42)`
(`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L235–241).
- TabPFN is **not** wrapped in the sklearn `Pipeline` used by the five classic models; it receives the raw frame
including the 106-level `Stent type-SES` string column, while the classic models receive **scaled + one-hot**
input (imputers in that transformer are inert). See §6.4. **[REV4]** disclosed in Part 4 Methods.
- 25 TabPFN fits total (5 outer × [4 inner + 1 outer]); the first inner fit took 24 min 38 s (L327–330).



### 5.9 TabPFN interpretability (Part 5)


| Block               | Configuration                                                                        | Backend                           | Source                                          |
| ------------------- | ------------------------------------------------------------------------------------ | --------------------------------- | ----------------------------------------------- |
| Mutual information  | `mutual_info_classif` on the median-imputed 81-column matrix, **full cohort**        | sklearn, 0 TabPFN calls           | Part 5 Table 0                                  |
| Stability selection | Forward SFS keeping 10 of 81, 5-fold CV, AP scoring, **10 seeds**, **full cohort**   | local TabPFN, ~8.6 h              | `.nbdump/…tabpfn_interpretability.txt` L604–626 |
| PDP                 | 4 continuous (grid 30) + 6 binary (0 vs 1); fit on 70% train, evaluated on the frame | local TabPFN                      | L755–797                                        |
| SHAP (shapiq SV)    | **15 explained rows**, budget 256, baseline imputer                                  | local TabPFN after client failure | L1042–1130                                      |
| k-SII / SHAP-IQ     | **one** row (`X_explain[0]`), budget 256                                             | local TabPFN after client failure | L1213                                           |
| Consensus           | Borda mean of normalised ranks over MI + stability + mean|SHAP|                      | aggregate                         | Part 5 Table 5                                  |


**[CRITICAL — the 15 SHAP rows are all VLST cases]** The explained set is built as
`_order = np.concatenate([_pos_idx, _neg_idx])[:15]` on a stratified 30% test split (L1056–1060). With
`TEST_SIZE = 0.3` and 92 events, the test split holds **28 positives**, so the first 15 entries of `_order` are
**15 positive-class rows and zero controls**, by construction. This is **explanation**, not feature selection:
MI and stability SFS use `X_all`, `y_all` (both classes).

**[REV4]** Part 5 captions now state “15 VLST cases; no controls.” Keep that wording in any paper figure list.

**[CRITICAL — PDP values are on a balanced-prior scale, not absolute risk]** Every TabPFN instantiation in Part 5
uses `balance_probabilities=True` (L610, L774, L1067, L1087, L1107, L1415…). That rescales outputs to a uniform
class prior. It is why the binary PDP table reports baseline "P(y=1 | 0)" values of **0.24, 0.14, 0.13** against a
true prevalence of **0.0177**, and why Figure 1's LV curve is described as rising "toward ~0.6". These are
**not** predicted absolute risks.

**[REV4]** Part 5 now has a Methods note and Table 3 caption saying so. Re-export is not required for the
wording; **[TODO-PDP]** remains only if a paper draft still quotes 0.24 as absolute risk.

Verified: `balance_probabilities=True` appears at L610, L774, L1067, L1087, L1107, L1415, L1433 and L1453 — every
TabPFN instantiation in the notebook, with no exceptions.

### 5.10 Why excluding the causal analysis (D3) is the right call

Recorded so the decision is defensible if a reviewer asks why no causal analysis appears. `causal_analysis.ipynb`
was fully executed, but as configured it could not have supported a claim:


| Element        | Value                                                                                               | Source (`.nbdump/code__analyzes__causal_analysis.txt`) |
| -------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| Exposure       | `Current drinking` — univariately null (OR 0.981)                                                   | L360                                                   |
| Exposed cases  | **13**                                                                                              | L393–401                                               |
| Adjustment set | 80 covariates auto-selected as "all numeric columns", **including** `Time since stent implantation` | L371–379, L399                                         |
| Estimates      | IPTW ATE 0.0021, IPTW ATT 0.0031, PSM ATT 0.0052, **AIPW ATE 0.0158**                               | L665–671                                               |
| Uncertainty    | **none** — no SEs, no bootstrap, no CIs                                                             | whole notebook                                         |


The adjustment set contains the structural-leakage column from §4.1; AIPW disagrees with IPTW by ~7× with no
interval to judge it; and 13 exposed events cannot support an 80-covariate propensity model. The notebook's own
header calls itself a template with a default exposure. Excluding it is correct.

### 5.11 The published clinical baseline is not in this repository

Wang 2020's VLST score is an 8-variable Cox model (DM, previous PCI, AMI as admitting diagnosis, eGFR < 90,
3-vessel disease, number of stents per lesion, SES, no post-dilation), derivation c-statistic 0.80 (cross-validated
0.75), Shantou c-statistic 0.82, with decision-curve analysis against the Dangas LST score. **None of that model,
its linear predictor, or its score points is implemented in any in-scope notebook.** The ML comparison is currently
TabPFN vs untuned sklearn defaults, not TabPFN vs the score already published on these patients. **[TODO-SCORE]**

---



## 6. Data splits and validation procedures



### 6.1 Three different splitting schemes are in use


| Scheme                             | Where                                            | Details                                                                                                                               |
| ---------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **A. 70/30 stratified hold-out**   | `preprocessing.ipynb`, Part 2, Part 5            | `train_test_split(test_size=0.3, stratify=y, random_state=42)` → train 3,629 (64 events) / test 1,556 (28 events)                     |
| **B. Nested 5×4 stratified CV**    | Part 4                                           | Outer `StratifiedKFold(5, shuffle=True, random_state=42)`; inner `StratifiedKFold(4, shuffle=True, random_state=10_000 + outer_fold)` |
| **C. Single 70/30 + GridSearchCV** | `baseline_tssi_leakage`, `baseline_without_tssi` | Not reported in any Markdown                                                                                                          |


**[VERIFIED]** Scheme A produces identical counts in `preprocessing.ipynb` and in
`baseline_feature_selections.ipynb`, which asserts them explicitly
(`.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L259, L267).

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

### 6.3 [DISCREPANCY / RISK] No hyperparameter tuning, unequal model effort

- The five classic models use **library defaults** apart from class weighting and internal eval metrics
(`baseline_plus_tabpfn.ipynb` fit cell). No grid, no random search inside the nested loop.
- **TabPFN** (client) still runs with `thinking_mode=True, thinking_effort="high"` — constructor unchanged.
- **TabPFN (local)** is now a seventh arm in the same notebook (`from tabpfn import TabPFNClassifier`,
  no thinking). Toggle `RUN_MODELS` to run only that arm. **Not yet executed**; stored OOF/figures remain
  the six-model thinking-high run.

Comparing an untuned default RandomForest / LogisticRegression against a high-effort TabPFN and concluding
"TabPFN dominates" is **not a fair comparison**. Nothing in Part 4 discloses that the baselines are untuned.
**[REV4]** Part 4 Table 0 / Methods now say: same split and threshold protocol; classics not grid-searched;
unequal search budget disclosed; `TabPFN (local)` exists in the notebook but is not in the stored six-model run.

### 6.4 [RISK] The models do not see the same feature representation


| Model                       | Input                                                                                                                                                                                                     |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LR, RF, XGB, LGBM, CatBoost | `ColumnTransformer`: median imputation + `StandardScaler` on numerics; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` on `Stent type-SES` → the 106 brand strings become 106 columns |
| TabPFN                      | Raw frame, native categorical handling, no scaling                                                                                                                                                        |


Since there are no missing values, imputation is inert. The one-hot expansion of 106 sparse brand levels is not:
it hands the tree models 106 near-empty columns while TabPFN handles the column natively.

**[REV4]** Part 4 Methods now state this (classics ~186 columns vs TabPFN raw 81). Do not describe imputers as a
data-cleaning step. The shapiq `imputer="baseline"` in Part 5 is a different object (hidden-feature replacement
for attribution), also now labelled in that report.

### 6.5 Threshold reporting: honest and optimistic versions both exist

The notebook computes both. The paper tables report the optimistic one.


| Version                 | How the threshold is chosen                                                                                       | TabPFN result                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Honest (nested)**     | Inner-CV OOF on the training portion, applied once to unseen outer fold                                           | precision 0.8354, recall 0.7174, F1 0.7719, TN/FP/FN/TP = 5080/13/26/66, mean threshold 0.297 ± 0.053 |
| **Optimistic (pooled)** | `best_fbeta_threshold(y_true_oof, probs)` — grid-searched to maximise F1 **against the same labels being scored** | precision 0.7766, recall 0.7935, F1 0.7849, TN/FP/FN/TP = 5072/21/19/73, threshold 0.173              |


Sources: `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L831 (`t_f1 = best_fbeta_threshold(y_true_oof, …)`),
L1010–1026 (nested block), L992–1008 (pooled block).

Part 4 discloses that the pooled threshold is used ("a single operating point for the comparison table, not the
nested inner-fold thresholds", L89) but does not say that this makes precision/recall/F1/F2 optimistically biased,
and does not report the honest nested numbers at all. The honest numbers are lower on recall
(0.717 vs the reported 0.837).

### 6.6 [GAP] No external validation of the *machine-learning* models

Every ML number in the repository comes from the same 5,185 rows. There is no temporal split and no held-out
recalibration set.

Wang 2020 **did** externally validate their Cox score on 2,058 patients from Shantou (c-statistic 0.82). Those
rows are **not in this repository**. If they can be obtained, they are the natural external test set for TabPFN
and the five baselines — and the comparison that reviewers will expect, because it is how the published score was
already tested. **[TODO-EXT]**

Until then, the honest statement is: nested-CV discrimination on the derivation cohort only; the published score
has an external cohort this analysis did not use.

---



## 7. Every reported metric



### 7.1 Part 4 — nested-CV ranking metrics (the paper's headline results)

Evaluation protocol for all rows: **pooled out-of-fold probabilities from 5 outer folds, n = 5,185, 92 events,
threshold-independent**. Source of the Markdown values:
`paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md` §2, Table 1 (L62–69) and
`paper_figures/paper_table1_ranking.csv`. Source of the notebook values:
`.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L974–990 (ranking) and L728–737 (Brier).


| Model               | Metric  | Markdown value | Notebook value | Fold mean ± SD (notebook only) | Status                            |
| ------------------- | ------- | -------------- | -------------- | ------------------------------ | --------------------------------- |
| TabPFN              | PR-AUC  | 0.852          | 0.8534         | 0.8503 ± 0.0746                | rounding only                     |
| TabPFN              | ROC-AUC | 0.990          | 0.9883         | 0.9884 ± 0.0061                | rounding only                     |
| TabPFN              | Brier   | 0.0360         | **0.0060**     | —                              | **[DISCREPANCY] 6× → use 0.0060** |
| CatBoost            | PR-AUC  | 0.697          | 0.6967         | 0.7007 ± 0.0684                | ✓                                 |
| CatBoost            | ROC-AUC | 0.970          | 0.9704         | 0.9712 ± 0.0117                | ✓                                 |
| CatBoost            | Brier   | 0.0090         | 0.0090         | —                              | ✓                                 |
| LightGBM            | PR-AUC  | 0.677          | 0.6770         | 0.6841 ± 0.0937                | ✓                                 |
| LightGBM            | ROC-AUC | 0.961          | 0.9613         | 0.9633 ± 0.0198                | ✓                                 |
| LightGBM            | Brier   | 0.0096         | 0.0096         | —                              | ✓                                 |
| XGBoost             | PR-AUC  | 0.665          | 0.6647         | 0.6792 ± 0.0883                | ✓                                 |
| XGBoost             | ROC-AUC | 0.949          | 0.9493         | 0.9492 ± 0.0343                | ✓                                 |
| XGBoost             | Brier   | 0.0093         | 0.0093         | —                              | ✓                                 |
| Random Forest       | PR-AUC  | 0.456          | 0.4563         | 0.4740 ± 0.0319                | ✓                                 |
| Random Forest       | ROC-AUC | 0.931          | 0.9313         | 0.9304 ± 0.0186                | ✓                                 |
| Random Forest       | Brier   | 0.0147         | 0.0147         | —                              | ✓                                 |
| Logistic Regression | PR-AUC  | 0.342          | 0.3418         | 0.3568 ± 0.1153                | ✓                                 |
| Logistic Regression | ROC-AUC | 0.925          | 0.9246         | 0.9256 ± 0.0225                | ✓                                 |
| Logistic Regression | Brier   | 0.0543         | 0.0543         | —                              | ✓                                 |


**Prevalence baseline for PR-AUC: 0.0177.** Every PR-AUC above should be reported against this reference.

**D4 resolution — the single most important number change in this revision.** TabPFN's Brier is **0.0060**, which
is the **best of the six models**, not 0.0360, which would have been the worst. Section §12.2 shows the notebook is
internally consistent on this. Consequently every sentence of the form "TabPFN's Brier is worse than the tree
ensembles" or "TabPFN is not a well-calibrated risk engine" is **false and must be deleted**, and TabPFN's
calibration becomes a supporting result rather than a caveat. **[TODO-P4]**

**Fold mean ± SD is available for every model** (right-hand column, notebook L976–982). Part 4's own provenance note
claims it is not (L11); under D4 that note is simply wrong.

### 7.2 Part 4 — operating-point metrics

Protocol: pooled OOF probabilities thresholded at the **F1-maximising pooled threshold** (optimistically biased,
see §6.5). Source: Part 4 Table 2 (L105–112) / `paper_table2_f1_operating_point.csv`; notebook L992–1008.


| Model               | Threshold (md / nb) | Accuracy        | Precision           | Recall              | Specificity     | F1              | F2              | TN/FP/FN/TP (md) | TN/FP/FN/TP (nb)  |
| ------------------- | ------------------- | --------------- | ------------------- | ------------------- | --------------- | --------------- | --------------- | ---------------- | ----------------- |
| TabPFN              | **0.901 / 0.173**   | 0.9919 / 0.9923 | **0.7404 / 0.7766** | **0.8370 / 0.7935** | 0.9947 / 0.9959 | 0.7857 / 0.7849 | 0.8157 / 0.7900 | 5066/27/15/77    | **5072/21/19/73** |
| XGBoost             | 0.381               | 0.9896          | 0.7639              | 0.5978              | 0.9967          | 0.6707          | 0.6250          | 5076/17/37/55    | identical         |
| CatBoost            | 0.347               | 0.9882          | 0.6703              | 0.6630              | 0.9941          | 0.6667          | 0.6645          | 5063/30/31/61    | identical         |
| LightGBM            | 0.228               | 0.9892          | 0.7812              | 0.5435              | 0.9973          | 0.6410          | 0.5787          | 5079/14/42/50    | identical         |
| Random Forest       | 0.084               | 0.9786          | 0.4275              | 0.6087              | 0.9853          | 0.5022          | 0.5611          | 5018/75/36/56    | identical         |
| Logistic Regression | 0.970               | 0.9799          | 0.4318              | 0.4130              | 0.9902          | 0.4222          | 0.4167          | 5043/50/54/38    | identical         |


**Every classic-model row matches exactly. Only the TabPFN row differs.** See §12.2.

**D4 resolution.** Use the notebook column: TabPFN t_F1 = **0.173**, precision **0.7766**, recall **0.7935**,
**5072/21/19/73**. The widely-quoted "TP = 77, FN = 15" belongs to the superseded run and must not appear. Under the
honest nested threshold it is TP = 66, FN = 26 (§7.3).

### 7.3 Part 4 — honest nested operating point (in the notebook, absent from the reports)

Protocol: per-fold thresholds from inner CV, applied once to unseen outer-fold cases.
Source: `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L1010–1026.


| Model               | Threshold mean ± SD | Accuracy | Precision | Recall | Specificity | F1     | F2     | TN/FP/FN/TP   |
| ------------------- | ------------------- | -------- | --------- | ------ | ----------- | ------ | ------ | ------------- |
| TabPFN              | 0.297 ± 0.053       | 0.9925   | 0.8354    | 0.7174 | 0.9974      | 0.7719 | 0.7383 | 5080/13/26/66 |
| CatBoost            | 0.317 ± 0.068       | 0.9867   | 0.6211    | 0.6413 | 0.9929      | 0.6310 | 0.6371 | 5057/36/33/59 |
| XGBoost             | 0.237 ± 0.072       | 0.9882   | 0.6782    | 0.6413 | 0.9945      | 0.6592 | 0.6484 | 5065/28/33/59 |
| LightGBM            | 0.099 ± 0.048       | 0.9863   | 0.6154    | 0.6087 | 0.9931      | 0.6120 | 0.6100 | 5058/35/36/56 |
| Random Forest       | 0.114 ± 0.009       | 0.9834   | 0.5357    | 0.4891 | 0.9923      | 0.5114 | 0.4978 | 5054/39/47/45 |
| Logistic Regression | 0.953 ± 0.035       | 0.9757   | 0.3365    | 0.3804 | 0.9865      | 0.3571 | 0.3708 | 5024/69/57/35 |


**This table, not §7.2, is the defensible operating-point result.**

### 7.4 Part 4 — fold-wise ranking metrics

Source: `.nbdump/code__modeling__rating__baseline_plus_tabpfn.txt` L1030–1060. Each fold n = 1,037, with
n_pos = 18, 18, 18, 19, 19. Both metrics for all six models — the full grid, not the excerpt reported previously:


| Model               | Metric  | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| ------------------- | ------- | ------ | ------ | ------ | ------ | ------ |
| Logistic Regression | PR-AUC  | 0.2314 | 0.2850 | 0.3093 | 0.4727 | 0.4856 |
| Logistic Regression | ROC-AUC | 0.8911 | 0.9440 | 0.9147 | 0.9420 | 0.9361 |
| Random Forest       | PR-AUC  | 0.4524 | 0.5101 | 0.4543 | 0.4458 | 0.5075 |
| Random Forest       | ROC-AUC | 0.8978 | 0.9435 | 0.9362 | 0.9336 | 0.9408 |
| XGBoost             | PR-AUC  | 0.7252 | 0.6587 | 0.5398 | 0.7731 | 0.6991 |
| XGBoost             | ROC-AUC | 0.9826 | 0.9402 | 0.8943 | 0.9602 | 0.9686 |
| LightGBM            | PR-AUC  | 0.7374 | 0.7078 | 0.5234 | 0.7606 | 0.6914 |
| LightGBM            | ROC-AUC | 0.9862 | 0.9402 | 0.9469 | 0.9788 | 0.9644 |
| CatBoost            | PR-AUC  | 0.6987 | 0.6749 | 0.6086 | 0.7952 | 0.7261 |
| CatBoost            | ROC-AUC | 0.9816 | 0.9595 | 0.9600 | 0.9846 | 0.9704 |
| **TabPFN**          | PR-AUC  | 0.8241 | 0.7776 | 0.7928 | 0.9482 | 0.9088 |
| **TabPFN**          | ROC-AUC | 0.9920 | 0.9850 | 0.9827 | 0.9972 | 0.9850 |


TabPFN's PR-AUC beats every other model in **all five folds** — a 5/5 sweep. That is a more informative and more
honest statement than the pooled point estimate, and it is free: no new computation. It is still not a paired
significance test (§12.9), so "wins in 5 of 5 folds" is the strongest claim available without one.

**[DISCREPANCY → resolved]** Part 4's Markdown states that fold-wise mean ± SD "are therefore not available from
this notebook snapshot" (L11) and that the comparison-table cells were not executed (L149). Both statements are
wrong; the outputs are at L974–1060 of the same notebook.

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

These are **not** performance metrics. They are model-attribution and screening quantities on the full cohort or
on 15 case rows. Full tables in `paper_results/05_tabpfn_interpretability/paper_figures/*.csv`; reproduced in
§8.6.

### 7.8 Metric comparability audit


| Question                                                        | Answer                                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Are the six Part 4 models compared on identical data and folds? | **Yes** — same `oof_probabilities` arrays, same outer folds.                                                                                                                                                                                                                                                                                  |
| Are they compared with identical tuning effort?                 | **No.** Classic models are untuned defaults; TabPFN uses high-effort thinking mode.                                                                                                                                                                                                                                                           |
| Are they compared on identical feature representations?         | **No.** TabPFN bypasses the ColumnTransformer.                                                                                                                                                                                                                                                                                                |
| Are the Part 4 operating-point metrics unbiased?                | **No** — threshold selected on the evaluation labels. Honest version exists (§7.3).                                                                                                                                                                                                                                                           |
| Is Part 2 comparable with Part 4?                               | **No.** Part 2 uses a single 70/30 split, 185 encoded columns, seven differently configured models, and test-set-based selection.                                                                                                                                                                                                             |
| Is Part 5 comparable with Part 4?                               | **No.** Part 5 uses local TabPFN with `balance_probabilities=True`; Part 4 uses the client with thinking mode and no probability balancing. Different models, different output scales.                                                                                                                                                        |
| Are effect sizes in EDA Table 1 comparable across rows?         | **No.** Cohen's d and Mann–Whitney r are mixed in one column and plotted on one axis in Figure 3. MW r = Z/√N is severely attenuated by 1.77% class imbalance: `WBC` has the second-smallest p-value in the entire study (7.9e-21) but an "effect size" of 0.130, next to `LV`'s d = 1.127. Figure 3 must not be read as a magnitude ranking. |
| Are univariate ORs consistent across tables?                    | **No.** See §12.4.                                                                                                                                                                                                                                                                                                                            |


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
**[REV4]** Part 1 and Part 3 markdowns now say this. Jaccard / Venn stay on the 20-name lists.

### 8.2 Multivariable-model survivors (bootstrap CI excluding 1)

From EDA Table 4: `WBC`, `eGFR`, `LV`, `CKD5` (**sign-flipped**), `1.1:1Post dilation`, `CKD90` (CI 2.71–639.5),
`Previous PCI`, `Clopidogrel`.
From the joint domain model: `WBC`, `eGFR`, `LV`, `LVEF` (**sign-flipped**), `No.of stents per lesion`, `Men`,
`1.1:1Post dilation`, `Previous PCI`.

### 8.3 LOCO (Part 2)

- Pool: the **first 40 columns of the encoded matrix in ColumnTransformer order** (23 continuous + 17 binary),
not an importance ranking.
- Scored on the 1,556-row test set (28 events), for `pr_auc`, `f1`, `f2`, for each of 7 models.
- Unique names after pooling all three metrics: **40 for every model** — i.e. the cap, not a result.
- Cross-model intersection (present in all 7 models' top-12): PR-AUC → `LV`, `eGFR`; F1 → `LV`, `eGFR`;
F2 → `LV`.



### 8.4 Coalition SHAP (Part 2)

- Universe: LOCO top-24 (nested inside the arbitrary 40).
- Value function: PR-AUC / F1 / F2 computed on **32 test rows, of which 28 are cases (87.5% prevalence)**,
averaged over 3 background draws, 12 permutations.
- Unique names per model: 30–36.
- Cross-model intersection: PR-AUC → `LV`; F1 → `eGFR`; F2 → `LV`, `WBC`.



### 8.5 FFS (Part 2)

- Candidate pool: LOCO top-30.
- Greedy forward addition, scored on an inner 20% hold-out of the training set (≈ 726 rows, ≈ 13 events).
- Unique names per model: 18–24 (sparsest of the three selectors).
- Cross-model intersection: PR-AUC → `Cre`; F1 → `WBC`; F2 → `WBC`.



### 8.6 Classic-model consensus (Part 2 Table 2 and Table 4)

**Strictest global intersection** (all 7 models × all 3 selectors × all metrics): `WBC`**,** `eGFR` — 2 names.
**Global union:** 40 names (= the LOCO cap).

**Within-model LOCO ∩ SHAP ∩ FFS, by model and metric** (Part 2 Table 2, L156–178):


| Model | PR-AUC                           | F1                               | F2                           |
| ----- | -------------------------------- | -------------------------------- | ---------------------------- |
| lr    | Cre, Men, Min-stent diameter, TG | Cre, Men                         | Fast-Glu, LV, Men, WBC, eGFR |
| rf    | LVEF, WBC, eGFR                  | LV, Previous PCI, WBC, eGFR      | Cre, Fiberinogen, Platelet   |
| rf_b  | CaI, LVEF, WBC                   | eGFR                             | HL, LV, LVEF, STEMI          |
| cat   | LV, WBC, eGFR                    | HGB, HL, LV, Platelet, WBC, eGFR | Hypertension, LV, WBC, eGFR  |
| xgb   | Cre, LV, TCL                     | LV, LVEF, WBC, eGFR              | LV, WBC, eGFR                |
| xgb_b | Cre, HGB, LV, WBC                | WBC, eGFR                        | LV, WBC, eGFR                |
| lgb   | Cre, HL                          | Current drinking, HL, WBC        | History of HF, Men, WBC      |


**Union of the above (the "ML consensus catalogue", n = 20, Part 3 L45):** `WBC`, `eGFR`, `LV`, `Cre`, `Men`,
`LVEF`, `Previous PCI`, `Fiberinogen`, `HGB`, `Platelet`, `HL`, `STEMI`, `Hypertension`, `Fast-Glu`, `TG`,
`TCL`, `CaI`, `Min-stent diameter`, `Current drinking`, `History of HF`.

**Union sizes per model** (Part 2 Table 3): lr 33, rf 32, rf_b 33, cat 30, xgb 32, xgb_b 30, lgb 30.
**Jaccard between selector unions:** 0.95–0.97 — high **because the selectors are nested in one pool**, which
Part 2 correctly notes (L140).

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
`code/analyzes/stats_vs_ml/rebuild_comparison.py` (notebook wrapper `stats_vs_ml_comparison.ipynb`).
Headline arithmetic is asserted in that script: Jaccard = 5/35 ≈ 0.1429, intersection
`{WBC, eGFR, LV, Fiberinogen, Previous PCI}`. **[TODO-P3 — closed]**

### 9.1 Headline overlap


| Quantity                  | Value                                                      | Source     |
| ------------------------- | ---------------------------------------------------------- | ---------- |
| Statistical FDR catalogue | 20 names                                                   | Part 3 L43 |
| ML consensus catalogue    | 20 names                                                   | Part 3 L45 |
| Intersection              | **5** — `WBC`, `eGFR`, `LV`, `Fiberinogen`, `Previous PCI` | Part 3 L61 |
| Jaccard                   | 5 / 35 ≈ **0.14**                                          | Part 3 L55 |
| Statistics-only           | 15                                                         | Part 3 §4  |
| ML-only                   | 15                                                         | Part 3 §5  |


**[VERIFIED]** 20 + 20 − 5 = 35. The two 20-name input lists match §8.1 and §8.6, which are themselves
verified against `eda.ipynb` and `baseline_feature_selections.ipynb`. `rebuild_comparison.py` asserts the
same intersection and Jaccard.

### 9.2 Shared features and the evidence behind each


| Feature      | Statistical evidence           | ML evidence                              |
| ------------ | ------------------------------ | ---------------------------------------- |
| WBC          | MW r = 0.130, q = 9.48e-20     | Global 7-model × 3-selector intersection |
| eGFR         | Cohen d = −0.712, q = 3.71e-19 | Global 7-model × 3-selector intersection |
| LV           | Cohen d = 1.127, q = 3.26e-16  | Cross-model LOCO/SHAP; cat/xgb consensus |
| Fiberinogen  | MW r = 0.035, q = 0.0286       | RF F2 consensus only                     |
| Previous PCI | Fisher OR = 6.485, q = 0.0002  | RF F1 consensus only                     |


**[DISCREPANCY, minor]** Part 3 L96 writes "MW r = 0.13, q = 9.5e-20" for WBC — the CSV gives q = 9.48e-20 ✓.
Part 3 L100 writes "Fisher OR = 6.5" for Previous PCI, while three different tables in this repository give
6.485, 6.465, and 6.733 for the same quantity. See §12.4.

### 9.3 Statistics-only (15) — the repository's own explanations, audited

Part 3's Table 3 gives a reason for each. The reasons fall into three buckets, and **all three are correct but
two of them are artefacts of this analysis rather than findings about the data**:


| Bucket                       | Names                                                                                                                                             | Assessment                                                                                                                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Genuine collinear redundancy | `1.1:1Post dilation`, `No postdilation`, `Multi-vessel CAD`, `Single-vessel disease`, `3-vessel disease`, `NO.of vessels`, `CKD90`, `CKD5`, `PES` | **Real.** Univariate testing scores every member of a redundant block; a fitted model needs one.                                                                                           |
| One-hot fragmentation        | `Stent type-SES`                                                                                                                                  | **Real but self-inflicted.** The selector notebook one-hot-encodes the raw 106 brand strings, whereas the EDA collapses to 9 levels. Same variable, two incompatible encodings. See §12.5. |
| Never in the candidate pool  | `No.of stents per lesion`, `Total stent length`, `HbA1c`, `Clopidogrel`, `Diabetes`                                                               | **Artefact.** These may simply sit beyond column 40 in ColumnTransformer order. Their absence is not evidence.                                                                             |


Part 3 states the third mechanism honestly at L145 and L226. The paper must carry that caveat into every
sentence about statistics-only features, not bury it in a methods note.

### 9.4 ML-only (15) — the repository's own explanations, audited


| Feature                                                                                         | Univariate status         | Repository's stated reason                                                | Assessment                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cre                                                                                             | p = 0.88 (MW r = 0.002)   | Renal surrogate for eGFR                                                  | Plausible; note `Cre` skew = 7.48, kurtosis = 161                                                                                                                                              |
| Men                                                                                             | p = 0.27                  | `Men × eGFR` interaction is FDR-significant; joint model adjusted OR 3.28 | The interaction q = 0.028 comes from **16 hand-picked pairs**; the adjusted OR 3.28 arises from a null univariate. **Exploratory only.**                                                       |
| LVEF                                                                                            | raw p = 0.033, q = 0.0666 | (was "OR persists")                                                       | **[REV4]** Part 3 now states the **sign reversal** (0.851 → 1.65) when `LV` is in the same model. |
| HGB                                                                                             | raw p = 0.039, q = 0.0716 | Threshold-metric consensus                                                | Plausible                                                                                                                                                                                      |
| Fast-Glu                                                                                        | raw p = 0.025, q = 0.0536 | Correlated with HbA1c / diabetes                                          | Plausible                                                                                                                                                                                      |
| CaI                                                                                             | raw p = 0.051, q = 0.0869 | On the FDR boundary                                                       | Plausible; note `CaI` is the **top** mutual-information feature in Part 5                                                                                                                      |
| Platelet, HL, STEMI, Current drinking, History of HF, Hypertension, TG, TCL, Min-stent diameter | ns                        | Metric/hold-out artefacts                                                 | Part 3 L235 already calls LightGBM's set "algorithm artefacts". Endorse that framing.                                                                                                          |




### 9.5 The comparison's structural weakness

Part 3 compares:

- a **full-cohort, 92-event, multiplicity-controlled association screen** with
- a **28-event, test-set-scored, top-12-of-an-arbitrary-40-column-prefix predictive shortlist**.

Part 3 says as much in its own table (L33–40) and is careful throughout. But the framing "Only 5 of 20 ... also sit
in the ML three-selector consensus" invites the reader to treat 0.14 as a scientific finding about the biology.
It is largely a finding about the two procedures' differing power, encodings and candidate pools. Rewrite the
comparison as a **methods-comparison** result, not a biological one.

---



## 10. Complete figure and table inventory

**Regeneration status at a glance.** Part 1, Part 2, Part 3 and Part 5 exports match their generating code
(Part 3 via `rebuild_comparison.py`). Part 4's nested-CV export set is **[STALE]** for TabPFN and must be
re-exported (§12.2). Part 4's TSSI leakage supplement is current (stored single-split metrics, no re-run).

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
| Fig S5a–b | Supp fig   | `03_correlation_heatmap_top42_vs_next41_with_target.png`, `03b_spearman_correlation_heatmap_top42_vs_next41_with_target.png` | Pearson / Spearman pairwise heatmaps (with target) |
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


**[CLOSED]** Generating code: `code/analyzes/stats_vs_ml/rebuild_comparison.py` (notebook `stats_vs_ml_comparison.ipynb`). Jaccard 5/35 and the five-name intersection are asserted in the script. Markdown reports now embed the PNGs rather than placeholders.

### Part 4 — nested-CV rating  — **every item below is [STALE] for TabPFN**


| ID      | Type   | File                                       | Status                                                             |
| ------- | ------ | ------------------------------------------ | ------------------------------------------------------------------ |
| Table 0 | Table  | `paper_table0_models.png/.csv`             | model descriptions only; check the untuned-defaults wording (§6.3) |
| Fig 1   | Figure | `paper_fig1_pr_roc_curves.png`             | **[STALE]** — TabPFN curve from the superseded run                 |
| Table 1 | Table  | `paper_table1_ranking.png/.csv`            | **[STALE]** — TabPFN Brier 0.036 → 0.0060; add fold mean ± SD      |
| Fig 2   | Figure | `paper_fig2_calibration_curves.png`        | **[STALE]** — title reads Brier 0.0360                             |
| Fig 3   | Figure | `paper_fig3_confusion_matrices.png`        | **[STALE]** — title reads t_F1 0.901                               |
| Table 2 | Table  | `paper_table2_f1_operating_point.png/.csv` | **[STALE]** — TabPFN row only                                      |
| Table 3 | Table  | `paper_table3_confusion_counts.png/.csv`   | **[STALE]** — TabPFN row only; also redundant (§11.3)              |
| Table S-TSSI | Table | `paper_table_s_tssi_leakage.png/.csv`     | Current — stored 70/30 metrics, not nested-CV                      |
| Fig S-TSSI | Figure | `paper_fig_s_tssi_pr_auc.png`             | Current — PR-AUC with vs without TSSI                              |


**Produced by the notebook but absent from the report:** `best_model_threshold_fpfn_panel.png`,
`model_comparison.csv`, `nested_cv_operating_point.csv`, `fold_metrics.csv`,
`best_model_threshold_sweep.csv`, `oof_predictions.csv`, `fold_thresholds.csv`.

**[GAP — confirmed by directory listing]** None of the CSV/PNG artefacts written to `/kaggle/working/...` were
copied into the repository. `data/result/modeling_results/` contains **only** a `paper_figures/` folder; there is no
`oof_predictions.csv`, `fold_metrics.csv`, `fold_thresholds.csv`, `model_comparison.csv` or
`nested_cv_operating_point.csv` anywhere under `data/result/`. The out-of-fold predictions — which would let a
reviewer recompute every Part 4 number, add confidence intervals, and run DeLong or bootstrap comparisons — do not
exist in the repo. **[TODO-REPRO]**

### Part 5 — TabPFN interpretability


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


**Totals: 33 figures and 24 tables across the five reports.**

---



## 11. Recommended main text, supplement, and redundant items

Recommendations assume the leakage questions in §4.3 and §12.1 are resolved favourably. If they are not, the
paper's framing changes and this layout does not apply.

### 11.1 Main text (target: 4 figures, 3 tables)


| Slot         | Item                                                                                                                                                                          | Rationale                                                                                                                                                                                                                                                |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Table 1**  | Baseline cohort characteristics, cases vs controls                                                                                                                            | Wang 2020 Table 1 **already is this table** for this cohort. Rebuild it from `VLST.csv` (do not photocopy: post-dilation labelling is inconsistent in Wang, and `LV` / `CaI` need to appear). Cite Wang for the recruitment flow (6,038 → 5,185).        |
| **Table 2**  | Part 4 Table 1 **rebuilt**: PR-AUC and ROC-AUC as pooled value **and** fold mean ± SD, Brier, with prevalence 0.0177 stated; plus the honest nested operating point from §7.3 | Combines the two threshold-independent and threshold-dependent results into one defensible table. Must use the notebook values, not the stale Markdown values.                                                                                           |
| **Table 3**  | Part 1 Table 4 **respecified** — see §12.1; one representative per collinear block, EPV stated                                                                                | Currently unpublishable as written.                                                                                                                                                                                                                      |
| **Figure 1** | Part 4 Figure 1 (PR and ROC curves) with the prevalence line, **PR panel first and larger**                                                                                   | The single most important result. PR-AUC leads because prevalence is 1.77%.                                                                                                                                                                              |
| **Figure 2** | Part 4 Figure 2 (calibration), **re-exported** from the current notebook state                                                                                                | The Brier question is settled (§12.2): TabPFN's 0.0060 is the best of the six. Calibration now *supports* the result instead of qualifying it — but the caption must be rewritten from scratch, because the existing one says the opposite.              |
| **Figure 3** | A **new** single figure merging Part 1 Figure 4 (binary ORs) and Part 1 Figure 3 (continuous effect sizes) with **separate panels per effect-size metric**                    | Fixes the d-vs-r comparability problem of §7.8 while keeping the association story in one place.                                                                                                                                                         |
| **Figure 4** | Part 5 Figure 13 (consensus ranking) **or** Part 3 Figure 1 (Venn) — pick one                                                                                                 | Both answer "which variables matter". Two is redundant in the main text. Prefer Figure 13 if the paper's thesis is interpretation; prefer the Venn if the thesis is methods comparison. The Venn is now generated by `rebuild_comparison.py`. |




### 11.2 Supplement


| Group                                          | Items                                                                                                                                                                         |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Leakage evidence (**now in Part 4 supplement**) | Table S-TSSI and Figure S-TSSI: with-TSSI vs without-TSSI metric contrast (§7.5) and the case vs non-case time distribution (min 1,241 vs min 380 days), framed as binary-ified survival time (§4.2). |
| Statistical detail                             | Part 1 Fig 1, Table R, Fig 2, Table 2, Table 3, Fig 5, Fig 6, Fig S5a–b (Pearson/Spearman pairwise)                                                                          |
| Domain analysis                                | Part 1 Fig S1, Table S1, Fig S2a–d, Fig S3, Fig S4, Table S2 (**all 16 interaction rows, not 8**)                                                                             |
| Feature-selection methods                      | Part 2 Table 0, Fig 1, Table 1, Table 2, Table 3, Table 4 — each with an explicit note that LOCO's pool is a column-order prefix and that scoring used the test set           |
| Stats-vs-ML                                    | Part 3 Fig 1 or 2, Table 1, Tables 2–4                                                                                                                                        |
| Interpretability                               | Part 5 Table 1, Table 2, Fig 1, Fig 2, Table 3, Fig 5, Table 4, Fig 7, Fig 8 — every caption restated per §12.6                                                               |
| Reproducibility                                | `oof_predictions.csv`, `fold_thresholds.csv`, `fold_metrics.csv`, `nested_cv_operating_point.csv` — **must be regenerated and committed**                                     |




### 11.3 Redundant — cut or merge


| Item                           | Reason                                                                                                                                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Part 4 Table 3                 | Identical content to Part 4 Table 2's last four columns. Pure duplication.                                                                                                                                          |
| Part 4 Figure 3                | The same confusion counts as Tables 2–3, as a heatmap. Keep the table, drop the figure.                                                                                                                             |
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

### 12.2 [RESOLVED under D4] Part 4's exported artefacts are stale; the notebook is not

Revision 1 recorded this as an unresolvable conflict between "two different runs" and asked the author to decide
which was authoritative. **That question is now answered, and it did not need the author.** Re-examining the
notebook's own *embedded output images* — not just its text — settles it:


| Quantity                         | Notebook text output       | Notebook **embedded figures** | Exported `paper_figures/`* + Markdown |
| -------------------------------- | -------------------------- | ----------------------------- | ------------------------------------- |
| TabPFN Brier                     | **0.0060** (L737)          | **0.0060**                    | 0.0360                                |
| TabPFN t_F1                      | **0.173** (L995)           | **0.173**                     | 0.901                                 |
| TabPFN TN/FP/FN/TP               | **5072/21/19/73** (L1003)  | **5072/21/19/73**             | 5066/27/15/77                         |
| TabPFN precision / recall        | **0.7766 / 0.7935** (L995) | —                             | 0.7404 / 0.8370                       |
| Five classic models, all metrics | —                          | —                             | **identical to notebook**             |


Evidence for the middle column: the calibration panel stored inside the notebook
(`.nbdump/imgs/rating_cell11_0.png`) is titled "TabPFN (Brier = 0.0060)" and its TabPFN reliability curve stops at
~0.13 predicted probability; the confusion-matrix panel (`.nbdump/imgs/rating_cell13_0.png`) is titled
"TabPFN (t_F1 = 0.173)" and shows 5072 / 21 / 19 / 73. Both agree with the text output of the same notebook.

By contrast `paper_results/04_tabpfn_rating/paper_figures/paper_fig2_calibration_curves.png` is titled
"TabPFN (Brier = 0.0360)" with a curve reaching ~0.57, and `paper_table1_ranking.csv` records
`TabPFN,0.852,0.99,0.036`. That file is byte-identical (md5 `88faaa39…`) to
`data/result/modeling_results/paper_figures/paper_fig2_calibration_curves.png`, so both copies of the export
descend from the same superseded execution.

**Conclusion.** The notebook run is **internally consistent throughout**. The `paper_figures/` layer — PNGs, CSVs
and the Markdown tables built from them — was exported from an **earlier TabPFN execution** and never refreshed
when TabPFN was re-run; the classic-model rows were unaffected because those models are deterministic.

**What this changes:**

1. **No author decision is required.** The authoritative numbers are the notebook's: Brier 0.0060, t_F1 0.173,
  5072/21/19/73, precision 0.7766, recall 0.7935. Previous blocking item 4 is closed.
2. **The calibration narrative inverts.** **[REV4]** Part 4 captions now say the notebook Brier is **0.0060**
  (best of six) and that the PNG title 0.0360 is stale. **[TODO-P4]** is now only the PNG/CSV re-export.
3. **What remains real is a reproducibility caveat, not a data conflict.** Two client executions of
  `thinking_mode=True` with `random_state=42` produced materially different probabilities (Brier 6× apart, 4 of 92
   events reclassified). That is a genuine non-determinism finding about the TabPFN client and belongs in
   Limitations (§12.10) — but it no longer casts doubt on which numbers to report.
4. **Part 4's entire export set must be regenerated** from the current notebook state: 3 PNGs, 4 CSVs, 4 PNG table
  images, and the Markdown that quotes them. Patching the single Brier cell is not sufficient, because
   `paper_table2_f1_operating_point.csv` and `paper_table3_confusion_counts.csv` carry the stale threshold and
   confusion counts too. **[TODO-P4]**



### 12.3 [CRITICAL] Claims that ROC-AUC and F1 support clinical usefulness

Statements to remove or heavily qualify:

- "TabPFN dominates ranking (AP = 0.852, AUC = 0.990)" (Part 4 L52) — true as a ranking statement, but adjacent
text must not slide into utility. There is no decision-curve analysis, no net-benefit analysis, no
cost-weighted threshold, and no stated clinical action.
- "TabPFN catches the most events (TP = 77, FN = 15)" (Part 4 L95) — at a threshold chosen on the evaluation
labels. The honest nested figure is **TP = 66, FN = 26** (§7.3).
- "Accuracy is uniformly high because negatives dominate and is not a useful ranking criterion here"
(Part 4 L103) — this is correct and should be kept, but accuracy should then be **removed from the table**
rather than reported alongside.

What is *not* in the repository and would be needed for any usefulness claim: decision-curve analysis,
number-needed-to-screen, an explicitly costed FP:FN ratio, a defined clinical action triggered by a positive
prediction, and any external validation.

### 12.4 [DISCREPANCY] The same odds ratio is reported with three different values


| Feature                 | Table 2 (2×2 test) | Table 4 (single-feature weighted logistic) | Joint domain model (single-feature logistic) |
| ----------------------- | ------------------ | ------------------------------------------ | -------------------------------------------- |
| Previous PCI            | 6.485              | 6.465                                      | 6.733                                        |
| 1.1:1Post dilation      | 0.187              | 0.187                                      | 0.192                                        |
| Diabetes                | 1.889              | 1.889                                      | 1.898                                        |
| Men                     | 1.303              | —                                          | 1.286                                        |
| No.of stents per lesion | —                  | 1.379                                      | 1.379                                        |


All three are labelled "Univariate OR". Three different estimators are in use (2×2 cross-product,
class-weighted logistic, unweighted logistic) without any of the tables saying so. Pick one estimator, or label
each column with its estimator.

### 12.5 [DISCREPANCY] `Stent type-SES` is encoded two incompatible ways


| Notebook                                   | Handling                                                        | Resulting width        |
| ------------------------------------------ | --------------------------------------------------------------- | ---------------------- |
| `eda.ipynb`                                | Collapse levels with n < 30 to `other` → **9 levels**; one χ²   | 9                      |
| `preprocessing.ipynb`                      | Canonicalise brands, collapse rare → **9 categories** → one-hot | 88 total features      |
| `baseline_feature_selections.ipynb`        | One-hot the **raw 106 strings** directly, no canonicalisation   | **185 total features** |
| `baseline_plus_tabpfn.ipynb` (classic arm) | One-hot the raw strings via ColumnTransformer                   | ~186                   |
| `baseline_plus_tabpfn.ipynb` (TabPFN arm)  | Native categorical, no encoding                                 | 81                     |
| `tabpfn_interpretability.ipynb`            | Integer codes, treated as **numeric**                           | 81                     |


Part 3 attributes `Stent type-SES`'s absence from the ML consensus to "one-hot fragmentation" (L136, L144) —
correct, but the fragmentation is a **choice made in one notebook and not in the others**, not a property of the
variable. Also note that in Part 5 the integer coding means the PDP for `Stent type-SES` sweeps an **arbitrary
brand ordering as if it were a continuous scale** (Part 5 Figure 1 describes it as "nearly flat across its coded
range"), which is not a meaningful operation for a nominal variable.

The `preprocessing.ipynb` artefacts (`X_train.npy`, `preprocessor.joblib`, 88 features) are **not used by any of
the five reported analyses**.

### 12.6 [CRITICAL] Interpretability claims that overreach their sample


| Claim                                                                                         | Actual basis                                                                                                                                                                                                                    | Required restatement                                                                                                                                                                           |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Part 5 Figures 3, 5, 6 and Table 4 presented as global SHAP importance                        | **15 rows, all of them VLST cases** (§5.9)                                                                                                                                                                                      | **[REV4]** Captions now say 15 VLST cases / not a population measure. Keep that wording in the paper.                                                                                           |
| Part 5 Figures 8, 9, 11, 12 — "dominant pairwise terms" among LV/WBC/eGFR/LDL                 | **One** positive-class patient, budget 256                                                                                                                                                                                      | Already flagged in the text (L223) — keep and strengthen. These are **not** cohort interactions. The only cohort-level interaction evidence in the repository is the 16-pair LR screen (§7.6). |
| Part 5 Figure 1: "predicted risk … rises steeply toward ~0.6"; Table 3: "P(y=1 | 0) = 0.2430" | `balance_probabilities=True` — a **uniform-prior** rescaling; true prevalence is 0.0177                                                                                                                                         | **[REV4]** Methods note + Table 3 caption now say balanced-prior, not absolute risk.                                                                                                           |
| Part 5 Table 5: `Cre` and `No.of stents per lesion` with `mutual_info = 0.000000`             | Imputed zeros for features outside the MI top-15                                                                                                                                                                                | **[REV4]** Table 5 caption now says these are fill zeros, not measured zeros.                                                                                                                   |
| Part 5 Table 1: `Fast-Glu` and `ZES` shown as "—"                                             | Confirmed: `paper_table1_mutual_info.csv` rows 11 and 14 have an empty `mutual_info` field. The notebook printed only the ranked **names** (L679), never the values, so the numbers are not recoverable from the stored output. | Re-run `mutual_info_classif` — it is pure sklearn, seconds, zero TabPFN calls (L566) — and re-export. Do not publish a "top 15" with two blanks. **[TODO-MI]**                                 |
| Part 5 Table 2 caption: "8.6 h" wall time                                                     | Verified in the notebook                                                                                                                                                                                                        | Fine; keep as a reproducibility note.                                                                                                                                                          |




### 12.7 [DISCREPANCY] Part 2 Table 0 misdescribes CatBoost

Part 2 Table 0 (L39) states CatBoost used **"Ordered boosting"**. The factory
(`.nbdump/code__modeling__interpretability__baseline_feature_selections.txt` L335–353) sets no `boosting_type`
and sets `task_type="GPU"`. CatBoost on GPU supports only **Plain** boosting. The same table also omits that
`eval_metric="AUC"` (not PR-AUC) was used for the selector models, in contrast to Part 4 where CatBoost uses
`eval_metric="PRAUC"`.

### 12.8 [RISK] Part 2 numbers are from a deliberately reduced "smoke" run

Part 2 discloses `RUN_MODE = "smoke"` (L5). What it does not foreground is what smoke mode costs:
`SHAP_N_PERM = 12` permutations, 3 background draws per coalition evaluation, `FEATURE_TOPK = 12`,
`LOCO_MAX_FEATURES = 40` of 185. Monte-Carlo error on the Shapley estimates is unquantified and unreported. No
seed-stability analysis. Publishing a smoke-mode result as a feature-selection finding requires either a full-mode
re-run or an explicit statement that these are provisional.

### 12.9 [RISK] Unquantified uncertainty everywhere

- No confidence intervals on **any** PR-AUC, ROC-AUC, or Brier score.
- No paired statistical comparison between models (no DeLong, no bootstrap difference test). "TabPFN dominates"
and "CatBoost is next" are point-estimate orderings with no inference. Fold SDs (§7.1) show TabPFN's PR-AUC SD
is 0.075 and CatBoost's is 0.068 across 5 folds of ~18 events each — the gap of 0.15 may or may not survive a
paired test on 92 events, and no such test was run.
- No confidence intervals on precision/recall/F1 at any operating point.
- No calibration slope or intercept; only Brier and a visual reliability curve.
- Univariate effect sizes in Table 1 and Table 2 carry no confidence intervals — only p and q.



### 12.10 [GAP] Reproducibility


| Missing                                                                                   | Impact                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `oof_predictions.csv` / `fold_thresholds.csv` not in repo (written to `/kaggle/working/`) | No reviewer can recompute Part 4 or add CIs                                                                                                                                   |
| No `environment.yml` / lockfile; `requirements.txt` has no pins verified                  | TabPFN, shapiq, CatBoost versions unknown                                                                                                                                     |
| TabPFN client version and server-side model version unrecorded                            | The headline result depends on a remote service that can change                                                                                                               |
| TabPFN is non-deterministic across runs (§12.2) despite `random_state=42`                 | The headline number cannot be reproduced even locally                                                                                                                         |
| No data-availability, ethics, or consent statement **in this repo**                       | Wang 2020 already has all three (NCT03491891, ethics 2013-256, written consent, figshare data statement). Cite them; still put a one-paragraph statement in *this* manuscript |
| Notebooks explicitly excluded from the results pack (`paper_results/README.md`)           | The pack cannot be audited on its own                                                                                                                                         |




### 12.11 [WITHDRAWN under D1–D3] Selective reporting

Revision 1 argued that because nineteen notebooks exist and five analyses are reported, the TabPFN advantage
"cannot be distinguished from selection over many attempts", and demanded a CONSORT/TRIPOD-style declaration of
everything tried. **That finding is withdrawn.** The twelve excluded notebooks (ten `failed_hypothesis/`, the TabPFN
playground, the causal analysis) are out of scope by author decision, contribute no number to the paper, and are
not cited anywhere in it.

One sentence of residual substance, kept for honesty and needing no table: exploratory work beyond the reported
seven notebooks did execute against the same 92 events, so if a reviewer asks what else was tried, the accurate
answer is "exploratory analyses that were abandoned and are not reported" — not "nothing".

The stronger version of this concern, which does still bind, is not about notebook count at all: it is that no
paired significance test separates TabPFN from CatBoost (§12.9), and the ML models have no external validation
(§6.6) even though Wang's Cox score does.

### 12.12 Terminology discipline for the manuscript


| Term                             | Reserve for                                                                                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Association**                  | Part 1 univariate and multivariable results; Part 5 mutual information. Full-cohort, no held-out evaluation.                                         |
| **Prediction**                   | Part 4 nested-CV out-of-fold results **only**. This is the only genuinely out-of-sample evaluation in the repository.                                |
| **Interpretation / attribution** | Part 2 selectors, Part 5 SHAP / k-SII / PDP / stability. Model-explanatory, not evidence about patients.                                             |
| Never use                        | "risk factor", "causal", "protective", "independent predictor", "clinically useful", "validated" — none is supported by anything in this repository. |


Note in particular that `1.1:1Post dilation` appears with a protective adjusted OR of 0.144 and a negative PDP
shift of −0.086, and that `Clopidogrel` appears with an OR of 0.464. In an observational cohort with
confounding by indication, neither can be described as protective.

---



## 13. What is left to do

Grouped by who can do it and what it costs. **Group A can only be answered by you** and gates the paper's framing.
Groups B–E are execution work.

### Closed by this revision (including Wang 2020)


| Was                                                        | Now                                                                                                                            |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| "Which TabPFN run is authoritative?" (old blocking item 4) | **Closed.** The notebook is internally consistent; the exports are stale. Use Brier 0.0060, t_F1 0.173, 5072/21/19/73 (§12.2). |
| "Declare the full analysis history" (old item 16)          | **Closed** by D1–D3 (§12.11).                                                                                                  |
| "Fold-wise mean ± SD are not available"                    | **Closed.** They are in the notebook and are now tabulated in §7.1 and §7.4.                                                   |
| **A2** How were controls sampled?                          | **Closed by Wang 2020.** Consecutive complete-follow-up cohort, not case-control. 1.77% is published incidence (§2.3, §4.2).   |
| **A3** What does `Stent thrombosis = 1` mean?              | **Closed by Wang 2020.** ARC 2007 definite ST, angiographically confirmed, > 1 year (§2.3).                                    |
| **A5** Recruitment frame, ethics, consent                  | **Closed by Wang 2020.** Jilin University, Jan 2014–Jun 2015, NCT03491891, ethics 2013-256 (§2.3).                             |




### A. Remaining author questions after Wang 2020


| #      | Question                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Status                                                                                                                              |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **A1** | **Lab/echo timing, now specific.** Wang 2020 treats WBC, LVEF, lipids, fibrinogen and eGFR as **index-PCI baselines** and built a Cox score from them, so event-time measurement is no longer the leading hypothesis for those columns. Two things are still on you: (1) confirm `LV` — name, units, timing — because it is absent from Wang Table 1 and is a top ML feature; (2) decide how to discuss **WBC**, which Wang *excluded* from the score because infection could not be ruled out, and which our models rank at the top. | Narrowed. Does not gate the whole paper the way it did, but gates any sentence that treats `LV` or WBC as a novel validated marker. |
| **A4** | `LV` **and** `CaI` **only.** `TCL` = total cholesterol (mmol/L), `HL` = dyslipidaemia, fibrinogen in g/L, stent release pressure in atm, DAPT = mandated ≥1 year plus physician-directed continuation — all from Wang. `LV` and `CaI` are still unnamed.                                                                                                                                                                                                                                                                              | Partial.                                                                                                                            |




### B. Re-run or compute (roughly in dependency order)


| #                    | Task                                                                                                                                                                                                                                                                                 | Cost                                                                                                                                                            | Notes                                                                                                                                                                                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **B1** [TODO-P4]     | **Re-export all of Part 4** from the current notebook state: 3 figures, 4 CSVs, 4 table images.                                                                                                                                                                                      | Low, if the notebook state is intact — but TabPFN is non-deterministic, so a full re-run gives *different* numbers again. Prefer exporting from the stored run. | This is the highest-value single fix. Everything downstream quotes these files.                                                                                                                                                                                                                                                            |
| **B2** [TODO-REPRO]  | **Persist** `oof_predictions.csv`**,** `fold_thresholds.csv`**,** `fold_metrics.csv`**,** `model_comparison.csv`**,** `nested_cv_operating_point.csv` into the repo.                                                                                                                 | Low                                                                                                                                                             | Confirmed absent from `data/result/` entirely. Without OOF predictions nobody — including you — can add confidence intervals later without re-running TabPFN. Do this *at the same time as B1* while the run is reproducible.                                                                                                              |
| **B3** [TODO-CI]     | **Bootstrap CIs on PR-AUC / ROC-AUC / Brier, and a paired test** (bootstrap difference or DeLong for ROC) between TabPFN and CatBoost.                                                                                                                                               | Low **once B2 exists** — pure post-processing of OOF probabilities, no model refits.                                                                            | Currently there is no interval on any metric and no test behind "TabPFN dominates" (§12.9). Interim fallback that costs nothing: TabPFN wins PR-AUC in **5 of 5 folds** (§7.4).                                                                                                                                                            |
| **B4** [TODO-T4]     | **Refit EDA Table 4** with one representative per collinear block; report VIFs and EPV; raise `N_BOOT` from 200 to ≥ 2,000.                                                                                                                                                          | Low                                                                                                                                                             | As it stands the model is not identified: `1.1:1Post dilation` sits beside its exact complement, and `eGFR` beside `CKD5` and `CKD90` (§12.1). `CKD90`'s CI is 2.708–639.506. **This table cannot be published as written.**                                                                                                               |
| **B5** [TODO-P3 — closed] | **Write the missing Part 3 notebook.** Done: `rebuild_comparison.py` + `stats_vs_ml_comparison.ipynb`. Jaccard 5/35 asserted; figures/tables written to both report trees. | Done | Inputs remain the §8.1 FDR set and §8.6 ML consensus. Re-run the script if either catalogue changes. |
| **B6** [TODO-MI]     | **Re-run** `mutual_info_classif` and re-export Part 5 Table 1 to fill the blank `Fast-Glu` and `ZES` cells.                                                                                                                                                                          | Seconds — pure sklearn, zero TabPFN calls                                                                                                                       | A published "top 15" with two empty cells is not acceptable (§12.6).                                                                                                                                                                                                                                                                       |
| **B7** [TODO-TABLE1] | **Rebuild Table 1 from** `VLST.csv`, including `LV` and the variables Wang omitted, and cite Wang for the 6,038 → 5,185 flow. Do not photocopy Wang Table 1 (post-dilation label is inconsistent, §3.3).                                                                             | Low                                                                                                                                                             | A conventional Table 1 already exists in Wang 2020; the repo still needs a verified, ML-complete version.                                                                                                                                                                                                                                  |
| **B8**               | **Part 2 code is now train-only** (option a): inner-val scoring + cheap-train importance-ordered LOCO pool; SHAP/FFS still nest in that pool as a compute cap. **Re-run on Kaggle** (`USE_CACHE=False`), then refresh Part 3. Stored figures remain the leaky smoke run until then. | Kaggle — 7 models × 3 selectors | Until re-export, Part 2/3 cannot support a predictive claim (§4.4). |
| **B9**               | **Optional but strengthens the comparison:** tune the five classic baselines, or drop the ~106 one-hot brand dummies so all six models see comparable input.                                                                                                                         | Medium                                                                                                                                                          | Right now it is untuned defaults vs high-effort TabPFN (§6.3) on non-identical representations (§6.4). Disclosure is the minimum; tuning is the fix.                                                                                                                                                                                       |
| **B10** [TODO-SCORE] | **Implement Wang's 8-variable VLST score as a comparator** on the same 5,185 rows (and on Shantou if obtained): DM, previous PCI, AMI as admitting diagnosis, eGFR < 90, 3-vessel disease, stents per lesion, SES, no post-dilation. Report nested-CV PR-AUC/ROC-AUC against TabPFN. | Low–medium                                                                                                                                                      | This is the published clinical baseline. A paper that claims better prediction on the same cohort without beating this score will be the first reviewer comment. Watch the post-dilation coding (§3.3).                                                                                                                                    |
| **B11** [TODO-EXT]   | **Ask for the Shantou n = 2,058 file.** If it exists, it is the external test set Wang already used.                                                                                                                                                                                 | Political, not computational                                                                                                                                    | Without it, state clearly that ML validation is derivation-cohort nested CV only.                                                                                                                                                                                                                                                          |




### C. Rewrite or delete — claims that are now known to be wrong


| #       | Where                                                                                                                                                                                                    | Action                                                                                                                                                                                                                                                    |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C1**  | Part 4 Table 1 / Figure 2 stale Brier story                                                                                                                                                              | **[REV4]** Captions rewritten to notebook 0.0060. PNG/CSV re-export still **[TODO-P4]**.                                                                                                                                                                  |
| **C2**  | Part 4 "TabPFN catches the most events (TP = 77, FN = 15)"                                                                                                                                               | **[REV4]** Caption now quotes notebook pooled 73/19 and nested 66/26. PNG still stale.                                                                                                                                                                    |
| **C3**  | Part 4 provenance notes ("cells were not executed", "fold-wise mean ± SD not available")                                                                                                                 | **[REV4]** Replaced with a note that the notebook *did* print those tables; CSVs are uncommitted; PNGs are mixed stale/current.                                                                                                                            |
| **C4**  | Part 4 Tables 2–3 operating point                                                                                                                                                                        | **[REV4]** Text now labels the PNG as optimistic pooled F1 and quotes the honest nested counts. Keep both, labelled.                                                                                                                                      |
| **C5**  | Part 3: "domain multivariable OR persists" for `LVEF`                                                                                                                                                    | **[REV4]** Fixed: Part 3 states the sign reversal.                                                                                                                                                                                                        |
| **C6**  | Every Part 5 SHAP caption (Figures 3, 5, 6, 7; Table 4)                                                                                                                                                  | **[REV4]** Captions now say 15 VLST cases / not a population measure.                                                                                                                                                                                      |
| **C7**  | Every Part 5 PDP caption (Figures 1, 2; Table 3)                                                                                                                                                         | **[REV4]** Methods note + Table 3: balanced-prior, not absolute risk.                                                                                                                                                                                     |
| **C8**  | Part 5 k-SII captions (Figures 8–12)                                                                                                                                                                     | Already said one-row; keep that wording.                                                                                                                                                                                                                  |
| **C9**  | Part 5 Table 5 caption                                                                                                                                                                                   | **[REV4]** Caption now says MI 0.0 for `Cre` / `No.of stents per lesion` are fill zeros.                                                                                                                                                                   |
| **C10** | Anywhere "protective" appears — `1.1:1Post dilation` (OR 0.144), `Clopidogrel` (OR 0.464)                                                                                                                | **Delete the word.** Confounding by indication (§12.12).                                                                                                                                                                                                  |
| **C11** | Part 2 Table 0: CatBoost "Ordered boosting"                                                                                                                                                              | Wrong — GPU CatBoost uses **Plain**. Also state `eval_metric="AUC"` here vs `"PRAUC"` in Part 4 (§12.7).                                                                                                                                                  |
| **C12** | Part 1 Figure 3 / Table 1 effect-size column                                                                                                                                                             | Cohen's d and Mann–Whitney r are mixed on one axis. `WBC` (q = 7.9e-21) shows r = 0.130 next to `LV` (q = 3.3e-16) at d = 1.127. **Split into separate panels** (§7.8).                                                                                   |
| **C13** | The three different "univariate OR" values for `Previous PCI` (6.485 / 6.465 / 6.733)                                                                                                                    | Pick one estimator or label each column with its estimator (§12.4).                                                                                                                                                                                       |
| **C14** | `Stent type-SES`                                                                                                                                                                                         | Adopt **one** encoding across all analyses. Wang used a **binary SES class flag**; 9 collapsed brand levels is the next-best choice if brands are kept. Drop or relabel the Part 5 PDP that sweeps brand codes as if they were a continuous axis (§12.5). |
| **C15** | Any statement of the form "LOCO saturates the 40-feature cap"                                                                                                                                            | That describes the cap, not the data (§5.7).                                                                                                                                                                                                              |
| **C16** | Part 1 Table S2                                                                                                                                                                                          | Shows 8 of 16 interaction rows with no truncation note. Show all 16 or say it is truncated.                                                                                                                                                               |




### D. Content that must be written from scratch (items W1–W5, to avoid clashing with the scope decisions D1–D4)


| #                  | Item                                                                                                                                                                                                                                                                                                                                                                                                               | Notes                                               |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| **W1** [TODO-LEAK — closed] | **The leakage section** — with-TSSI vs without-TSSI contrast is now Part 4 Table S-TSSI / Figure S-TSSI (logistic PR-AUC 0.958 → 0.508; CatBoost 0.977 → 0.658) plus the event vs follow-up time distribution (controls min 1,241 days, cases min 380), framed as binary-ified survival time with Wang's Cox analysis as the design-correct alternative. | Closed. Nothing was re-run (§4.1, §4.2, §7.5). Methods paragraph is in the Part 4 header. |
| **W2**             | **Clinical motivation and citations.** Wang 2020 now supplies VLST definition, incidence/mortality citations, the LST-score gap, and this cohort's protocol. Still to write: why TabPFN, what "personalised" means, and an honest statement of what this paper adds *beyond Wang's already-validated 8-variable score*.                                                                                            | Cite Wang; do not write as if no VLST score exists. |
| **W3**             | **Limitations section**, covering at minimum: ML models have no external/temporal validation (Wang's score does, on Shantou data we did not use); binary classification vs the published Cox analysis; EPV ≈ 5.4; TabPFN client non-determinism despite `random_state=42`; TabPFN as a remote service whose version is unrecorded; DAPT columns are post-baseline; WBC was excluded by the original investigators. | §4.2, §6.6, §12.9, §12.10.                          |
| **W4** [TODO-EPV]  | **EPV stated explicitly** (92 / 17 ≈ 5.4) next to every adjusted odds ratio.                                                                                                                                                                                                                                                                                                                                       | Computed nowhere (§2.2).                            |
| **W5**             | **Terminology pass** enforcing §12.12: "association" for Part 1 and mutual information, "prediction" for Part 4 OOF results only, "interpretation/attribution" for Parts 2 and 5. Never "risk factor", "causal", "independent predictor", "clinically useful", "validated". Wang's own score *was* externally validated — do not use "validated" for TabPFN by contagion.                                          |                                                     |




### E. Housekeeping


| #      | Item                                                                                                                                                                                     |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **E1** | **Designate one canonical report tree.** Every report exists twice (`paper_results/`** and `code/**`); they will drift.                                                                  |
| **E2** | Delete `paper_domain_feature_map.csv` — byte-identical duplicate of `domain_feature_map.csv`.                                                                                            |
| **E3** | Fix or cut Part 2 Table 5: its rows are string-matching failures (`Age, years` vs `Age` → rank `NaN`), visible in the notebook output itself at L1502–1522. This is a bug, not a result. |
| **E4** | Pin package versions (`environment.yml` or a lockfile) and record the TabPFN client **and** server-side model version.                                                                   |
| **E5** | Add data-availability, ethics and consent statements **to this manuscript**, citing Wang 2020 (NCT03491891, ethics 2013-256, written consent, figshare).                                 |
| **E6** | Cut the redundant items in §11.3 (Part 4 Table 3 and Figure 3; Part 2 Figures 6, 7, S1, S3; Part 5 Figures 10–12, and one of Figures 3/6).                                               |




### Suggested order

1. **A1/A4 (**`LV`**, WBC vs Wang's exclusion) in parallel with B1+B2.** Lab timing no longer gates the whole paper; `LV` still gates any claim that names it.
2. **B10** (Wang score as comparator) as soon as Part 4 exports are clean — same 5,185 rows, no new data required.
3. **B11** (Shantou file) as a data-access ask, not a compute task.
4. **B4, B6, B7** — all cheap, all independent. B7 is now "rebuild and verify against Wang Table 1," not "invent from nothing." (B5 / Part 3 notebook is done.)
5. **C1–C4** immediately after B1, since they are the claims that are actually false today.
6. **B8** decision, then the rest of C and D.

