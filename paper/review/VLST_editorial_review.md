# VLST manuscript: editorial and scientific review

**Sources used (confirmed with the authors).** `VLST_Introduction.docx` and `VLST_methods.docx` (both 21 Oct 2026), `VLST_Discussion_Clinical_Limitations_WBC_Revised.docx` (28 Oct 2026), `main_paper_formal_report.md` (24 Sep 2026) and `wang_vlst.pdf` (Wang et al., *Sci Rep* 2020;10:6378). The older Discussion draft, `main_paper.pdf`, every other `.md` file in the repository and the earlier review in `paper/review/` were **not** used.

**What was inspected beyond the reports.** I read the analysis code and its stored outputs: `preprocessing.ipynb`, `baseline_plus_tabpfn.ipynb` (the primary nested benchmark), `wang_vlst_score.ipynb`, `baseline_tssi_leakage.ipynb`, `baseline_without_tssi.ipynb`, `baseline_feature_selections.ipynb`, `tabpfn_interpretability_fs_pdp.ipynb`, `tabpfn_interpretability_shap.ipynb`, `eda.ipynb` and `tools/rebuild_part4_from_kaggle_dump.py`. I checked the result CSVs these produce against the report. With the authors' permission I ran read-only checks on `data/raw/VLST.csv` using `paper/review/vlst_data_checks.py`; it uses the standard library only and reproduces every data-level number below.

**What was not done.** I re-ran none of the models. **No figure image was viewed**, so captions are judged from the text alone. I did not open Wang's supplementary material, which is not in the source set.

**Evidence labels used throughout.** **[D]** directly documented in a source or in code output. **[V]** verified by my own read-only check of the data or code. **[I]** my interpretation. **[U]** unverified. **[C]** contradiction between sources. **[M]** missing information.

---

## A. Executive assessment

### A1. Most defensible framing

This is a **secondary analysis of a single public file**: the derivation cohort deposited by Wang et al. (5,185 patients with ACS treated with PCI; 92 definite VLST events). The work has four defensible contributions.

1. **An internal comparison of existing classifiers** for discriminating the *recorded* VLST label in that file. These are five conventional learners and four TabPFN arms, all evaluated on one stratified 5 × 4 nested cross-validation.
2. **A data-provenance audit.** It shows that the public file contains outcome-dependent recording differences, and that flexible models can exploit them.
3. **A frozen evaluation of Wang's published integer score** on the same rows and labels, used as a clinical reference. This is not validation.
4. **Exploratory, hypothesis-generating association and model-attribution analyses.**

None of these is a new algorithm, an independent replication of Wang, or a validated risk model.

The data checks moved contribution 2 from a side issue to the centre of the paper. The very high TabPFN discrimination (AP 0.92, ROC-AUC 0.996) cannot be presented as evidence of clinical predictability until the artefacts below are resolved. As currently evidenced, it is better read as a measure of how much outcome-related signal, clinical **or** recording-related, a flexible model can extract from this specific file.

### A2. The most important scientific issues (ranked)

1. **Recorded eGFR is not a function of recorded creatinine for most controls, and the mismatch differs by outcome.** [V]
   - Creatinine is the same in both groups: mean 72.5 vs 72.4 µmol/L.
   - Recorded eGFR is 120.0 in controls vs 95.9 in cases.
   - Recalculating CKD-EPI 2009 from each patient's creatinine, age and sex reproduces the recorded value (within 1 unit) for 2,243/5,093 controls (44.0%) and 50/92 cases (54.3%).
   - The remaining controls carry eGFR values about 1.3–1.9 times the recalculated value.
   - With recalculated eGFR, the groups are similar: mean 91.9 vs 91.5; eGFR < 90 in 35.6% vs 40.2%.
   - `CKD90`, `CKD60` and `CKD5` are exact functions of the *recorded* eGFR, so they inherit the problem. `CKD90` is a Wang score component, and its distribution is identical to Wang's Table 1.
   - **Consequences.** The renal paragraph of the Discussion (adjusted OR 0.57 per SD), eGFR's rank 1 in SHAP and its 8/8 stability selection, and the eGFR < 90 point in the Wang score are all affected.
   - Why the controls' values differ is unknown. One possibility is creatinine and eGFR taken at different times. → Author query; recalculation is the targeted fix.
2. **The 1–4-year outcome was never constructed.** [V]
   - The binary label counts VLST at any time during follow-up.
   - 12/92 events occurred **after day 1,461** (latest day 1,575).
   - 3,090/5,093 controls (60.7%) were followed for **less than 1,461 days** (range 1,241–1,605).
   - The current analyses therefore classify "VLST recorded during variable follow-up of roughly 3.4–4.4 years", not "VLST in years 1–4".
   - The intended fixed-horizon, post-PCI application is **unvalidated**.
3. **The medication variables are post-baseline and are coded differently in cases.** [V]
   - Aspirin is recorded in only 61.8% of controls, in a cohort in which every patient received DAPT for at least one year (Wang, p. 6). These cannot be index-PCI prescriptions.
   - In controls, `DAPT = 1` always coincides with aspirin plus a P2Y12 inhibitor (2,260/2,260). In cases it does so in only 27/35.
   - These four columns (`Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT`) are in the primary prediction matrix and in the association model (clopidogrel OR 0.48).
   - **They are unavailable at a post-PCI prediction time.**
4. **There are other outcome-dependent recording signatures.** [V]
   - Cases occupy the last 92 rows of the file.
   - In the text, control laboratory values are formatted to three decimals, while case values carry 0–2.
   - After parsing, integer-valued WBC is present in 3,780 controls and 0 cases.
   - LV integers show a different terminal-digit pattern: even last digit in 65.1% of controls vs 47.8% of cases, and the digit 0 in only 1.3% of controls.
   - A probe using only 243 decimal-grid indicators reached AP 0.427 (ROC-AUC 0.952) [D].
   - Equalising residual precision lowered nested-fold AP for local TabPFN v3.5 from 0.896 to 0.844 [D]. Hosted arms were not tested.
5. **Feature definitions are unresolved.** [V/U]
   - `LV` has no counterpart in Wang's Table 1, and the Discussion's "diastolic function" interpretation is unsupported.
   - `CaI` matches Wang's "peak troponin I" mean and SD exactly in both groups. A *peak* value is a post-PCI measurement.
   - `PES` matches Wang's "SES" row exactly, and its brands (Partner, Excel, Firebird, Tivoli, Cypher) are all drug-flag `PES = 1`.
   - Post-dilation coding contradicts itself inside Wang's paper (Section C).
6. **Interpretation-model provenance is mis-described.** [V]
   - The SHAP notebook did *not* "finish the remaining rows locally". After an HTTP 429 at about row 450, it discarded the hosted results and recomputed **all 1,556 rows** with a local model. The outputs are therefore not mixed, but they explain a **local** model.
   - Neither that local model nor the stability-selection (SFS) or partial-dependence (PDP) models were built with the explicit v3.5 constructor. The resolved checkpoint was not logged.
   - SHAP magnitudes (eGFR 1.23) exceed 1, so the output scale is not probability. The scale is undocumented.
   - The report's description of stability selection ("top-10, 5-fold") is wrong. The code and output show K = 8, 3-fold CV and `n_estimators = 1`.
7. **The tuning budget is described in the wrong direction for the hosted arms.** [V]
   - The hosted thinking arms were run with `thinking_effort="high"` and `thinking_metric="average_precision"`. Prior Labs documents that "the metric defines what to optimize" and that thinking mode "uses more compute during fitting" [D, docs].
   - So "TabPFN had no per-dataset optimisation, unlike the classical models" is incorrect for these arms. The classical arms were untuned defaults, while the hosted arms ran a metric-targeted fit-time procedure whose mechanism is not documented.
8. **The Wang comparison needs a coding justification that does not rest on AUC, and needs its framing corrected** (Section C).

### A3. Does the current evidence support the intended post-PCI, 1–4-year application?

**No.** Four things would have to hold, and none does yet:

- the label is a fixed-horizon outcome that is fully observed;
- every predictor is available at a defined post-PCI time;
- discrimination is not attributable to outcome-dependent recording;
- performance is shown on independent data.

The first two can be partly addressed inside this file: relabel to a horizon that is fully observed, and restrict the matrix to predictors confirmed to be available. The third needs answers from the custodian plus targeted sensitivity analyses. The fourth needs another cohort.

### A4. What remains unverified

- Definitions, units and recording times for `LV`, `CaI`, `CKD5`, the medication columns, `TSSI` and post-dilation.
- Why recorded eGFR diverges from creatinine.
- Why the counts for previous PCI (10 vs 13 cases) and AMI on admission (65 vs 72 cases) differ from Wang's Table 1.
- Why the HbA1c group columns appear swapped.
- How deaths after discharge were handled.
- The checkpoint and output scale of the interpretation models.
- How Wang's Cox time axis, censoring and competing events were constructed.
- All figures (not viewed).

---

## B. Issue–solution table

Locations: **Int** = Introduction docx, **Met** = Methods docx, **Dis** = Discussion docx, **FR** = formal report (section), **W** = Wang PDF page.

### B1. Fixable editorially

| # | Issue | Location | Evidence | Consequence | Minimum solution | Status |
|---|---|---|---|---|---|---|
| E1 | "PR-AUC" is used for `average_precision_score` | Met §ML; FR throughout | Code calls `average_precision_score` [V] | Mislabels the metric | Write "average precision (AP)"; define it once | Fixed in draft |
| E2 | "Pre-specified clinical grid" / "pre-specified anti-leakage protocol" | FR Abstract, Methods; code comments | No protocol or registration exists; the rules were adopted after the leakage probe [D: notebook text] | Overclaims planning | Write "cleaning rules adopted after the leakage diagnostics" | Fixed |
| E3 | "ROC-AUC is dominated by true negatives" | FR Introduction | Incorrect: ROC-AUC is invariant to class ratio | Statistical error | Explain that ROC-AUC and AP answer different ranking questions and that AP depends on prevalence | Fixed |
| E4 | "P(Δ ≤ 0) = 0/2000" presented like a p-value | FR Results | Bootstrap of stored OOF scores without refitting [V] | Reads as a population P = 0 | Write "0 of 2,000 resamples (P_boot < 1/2,000)" and state that the bootstrap is conditional on the fitted predictions | Fixed |
| E5 | "No per-dataset grid" used as a fairness argument | FR Abstract, Methods, Discussion | Hosted arms optimised AP during fitting [V; docs] | Mis-states which arms were tuned | Disclose `thinking_effort`/`thinking_metric`; describe budgets as unmatched, with the classical arms the least tuned | Fixed |
| E6 | "Stability selection: top-10, 5-fold" | FR Methods/Appendix | Code and output: K = 8, `FS_CV = 3`, 8 seeds, `n_estimators = 1` [V] | Wrong method description | State K = 8, 3-fold, 8 seeds; the 10-seed plan was not run | Fixed |
| E7 | "Remaining held-out rows finished locally" | FR Methods | All 1,556 rows recomputed locally after the 429 error [V] | Implies mixed models | Describe accurately (draft Methods §2.9) | Fixed |
| E8 | "Out of 4 methods" comment vs three indicators; two-signal vs three-signal consensus | FR Methods | Code [V] | Confusing | Report only the three-signal consensus, rebuilt from saved CSVs, and label it as such; move the code-comment issue to the reproducibility appendix | Fixed |
| E9 | Filenames, cell numbers and revision IDs in the main text | FR throughout | — | Trivia | Move to a supplementary reproducibility table | Fixed |
| E10 | "Tabular foundation model … no per-dataset grid" repeated many times | FR | Repetition | Formulaic prose | Say it once in Methods | Fixed |
| E11 | Wang's median follow-up (1,502 days) and median time to VLST (697 days) given as properties of the analysed file | Met §Design; FR | The file's medians are 1,416 (all) or 1,419 (controls), and 736 among events [V] | Inconsistent cohort description | Cite Wang's figures as Wang's; report the file's own distribution; add an author query on how TSSI was constructed | Fixed + query |
| E12 | Methods claims comparisons used "Welch or Mann–Whitney selected by skewness and kurtosis" without stating the rule | Met | Rule found in code: \|skew\| ≤ 1 and excess kurtosis ≤ 3 → Welch [V] | Missing detail | State the rule | Fixed |
| E13 | Clinical-utility language ("plan follow-up", "fewer avoidable procedures", "lower healthcare costs") | Dis §Aim | No utility analysis was done | Unsupported | Rewrite as conditional future work | Fixed |
| E14 | "Diastolic dysfunction" mechanism paragraph for `LV` | Dis §LV | No definition source; not in Wang | Guessed construct | Remove; report `LV` as an undefined dataset variable | Fixed |
| E15 | Renal mechanism paragraph | Dis §eGFR | Issue A2.1 [V] | Interpreting an artefact | Replace with an artefact finding; any renal interpretation waits for recalculated-eGFR results | Fixed |
| E16 | ARC-2 (2018) cited as the endpoint definition | Int ref 2 | Wang used ARC 2007 (W p. 6) | Wrong definition source | Cite Cutlip et al. 2007; ARC-2 only as context | Fixed |
| E17 | The formal report's Wang reference has an incorrect title ("A novel risk model…") | FR References | PDF title differs [V] | Citation error | Correct it | Fixed |
| E18 | Previous-PCI paragraph cites Wang Table 1 framing; the file has 10 vs Wang's 13 cases | Dis §Previous PCI | [V] | Inconsistent counts | Report the file's counts; add an author query | Fixed + query |
| E19 | "Leakage contrast" attributed to TSSI | Int/FR (partly) | Five changes were bundled; SMOTE was applied before inner grid-search CV [V] | Over-attribution | Describe it as a bundled contrast; note SMOTE's position | Fixed |
| E20 | Very low ECE (0.0003) read as near-perfect calibration | FR Results/Limitations | With 8 quantile bins, ~7 of 8 bins have near-zero risk [I] | Overinterpretation | Report Brier against the null Brier (0.0174) and a calibration plot; describe ECE's limits | Fixed |

### B2. Pending author or custodian confirmation (see the email list)

| # | Issue | Evidence | Consequence | Minimum solution |
|---|---|---|---|---|
| Q1 | eGFR not reproducible from Cre in 56.0% of controls | [V] | Renal findings and the eGFR < 90 Wang item are unreliable | Query: equation, creatinine time point, source. Meanwhile use recalculated eGFR in sensitivity analyses |
| Q2 | TSSI meaning; event times after day 1,461; deaths after discharge | [V] | Horizon and censoring | Query |
| Q3 | Medication columns: time point and definition; DAPT inconsistent in 8 cases | [V] | Post-baseline; differential coding | Query; exclude from any post-PCI model |
| Q4 | `LV` definition | [V/U] | Mechanistic claims impossible | Query |
| Q5 | `CaI` = peak troponin I? Units and timing | [V] | Prediction time must follow peak troponin | Query |
| Q6 | `PES` = sirolimus-eluting? | [V] | Wang SES mapping | Query |
| Q7 | Post-dilation coding (`1.1:1Post dilation` vs `No postdilation`) and the sign in Wang's Table 2 | [C] | Wang score direction | Query |
| Q8 | Previous PCI 10 vs 13; AMI 65 vs 72; HbA1c columns swapped | [C] | Is the file the analysis file? | Query |
| Q9 | Why cases were recorded or exported differently (row order, decimals, LV digits) | [V] | Central validity threat | Query |

### B3. Pending code or implementation verification

| # | Check | Why it matters |
|---|---|---|
| C1 | Log `model_path` for the local models used in SFS (`TabPFNClassifier(device, n_estimators=1, random_state)`), PDP and the SHAP fallback (`LocalTabPFNClassifier(device, random_state, fit_mode="fit_with_cache")`). The code comments say `"auto"` resolves to v3 in tabpfn 9.0.0, yet the SHAP fallback log shows `tabpfn-v3.5-20260909.safetensors` being downloaded [C] | It decides whether the attribution model is v3 or v3.5 |
| C2 | The output scale returned by `tabpfn_shapiq.get_tabpfn_imputation_explainer` (probability vs log-odds). Mean \|SV\| of 1.23 rules out probability scale [I] | Captions and interpretation |
| C3 | Whether `restore_tabpfn_empirical_prior` changed any probabilities | No log line was printed in the stored run, so the branch probably did not execute [I]. Confirm, because Brier and ECE depend on it |
| C4 | The Wang-vs-ML table (`paper_table_s_wang_vs_ml.csv`) compares against an **obsolete** OOF file (thinking AP 0.8553) [V] | Regenerate from the primary run or drop it |
| C5 | The local and hosted v3.5 arms differ in the fold-1 AP ranking; individual-level agreement was not computed | Do not claim agreement |
| C6 | Found during the code fixes: both TabPFN interpretability notebooks defined `quantize_clinical_precision` but never called it, and they fitted the stent codebook on all 5,185 rows before the split [V] | The report's "same cleaning as the benchmark" was false for MI, SFS, PDP and SHAP. Fixed in the notebooks; attribution results must be re-run |

**Code status (31 Oct 2026).** C1, C2, C4 and C6 are fixed in the notebooks: pinned v3.5 constructors, a provenance JSON, a run-time output-scale check, and current OOF dumps for the Wang comparison. R1 is implemented in `baseline_plus_tabpfn.ipynb` and R2 in `eda.ipynb`. None of these has been re-run yet.

### B4. Essential targeted reanalysis (only if the related claim is kept)

| # | Reanalysis | Needed for | Cost |
|---|---|---|---|
| R1 | Re-run the primary 5 × 4 benchmark on a **post-PCI-available, artefact-reduced matrix**: drop the four medication columns; replace `eGFR`/`CKD5`/`CKD60`/`CKD90` with versions recalculated from `Cre`, age and sex (CKD-EPI 2009); keep the existing cleaning. At minimum run local TabPFN v3.5, XGBoost, LightGBM and logistic regression; add the hosted v3.5 arm if quota allows | Any statement about prediction from post-PCI information, or any ranking of models for the intended use | Local arms: hours. Hosted arm: 25 fits |
| R2 | Refit the 13-covariate association model with recalculated eGFR and without `Clopidogrel` (or report clopidogrel separately as a follow-up-period covariate) | Keeping the renal and clopidogrel association paragraphs | Minutes |

If R1 is **not** done, the paper must be framed only as "classification of the recorded label in this file, using all recorded variables". In that framing, the AP values are evidence about the file and the models, not about post-PCI prediction. The draft manuscript is written so that it holds under either outcome. R1 results go into marked placeholders.

### B5. Optional improvements

- **O1.** A fixed-horizon label at day 1,241 (70 events), the longest horizon fully observed in the file, assuming TSSI is the follow-up end and no competing deaths are missing. This is the only fixed horizon the file can support. It is shorter than the intended 4 years and should be reported as a sensitivity analysis, not as the 1–4-year model.
- **O2.** Machine-learning models restricted to Wang's 8 inputs, for an equal-information comparison with the score.
- **O3.** A drop-column sensitivity for `LV` and `CaI`.
- **O4.** Repeated outer cross-validation (for example 5 × 5), to show split variability.
- **O5.** Decision-curve analysis, but only after R1 and a credible horizon.

---

## C. Wang et al. 2020: what was done, what was not reported, and what comparison is defensible

### C1. Explicitly reported (page numbers refer to the PDF)

- **Cohorts (pp. 2, 6).**
  - Derivation: consecutive patients aged ≥ 18 years with ACS treated with PCI at the First Hospital of Jilin University, 1 Jan 2014 – 1 Jun 2015, with written consent. Of 6,038 eligible, the authors excluded 236 deaths before discharge, 413 refusals of follow-up and 204 lost to follow-up, leaving **5,185**. Mean age 59.85 ± 9.95 years; 68.60% men; **median follow-up 1,502 days** (p. 2).
  - External validation: First Affiliated Hospital of Shantou University, same dates and criteria. Of 2,438 eligible, the authors excluded 88 deaths, 186 refusals and 106 lost, leaving **2,058**; median follow-up **1,422 days** (p. 2).
- **Exclusions (p. 6).** Expected survival < 1 year; severe organ failure; malignancy; planned surgery on major organs within 1 year requiring antiplatelet interruption; death before discharge; refusal of consent or follow-up interviews; loss to follow-up after discharge.
- **Treatment (p. 6).** All patients received newer or second-generation DES and DAPT for at least one year. Treatment beyond that was at the treating physicians' discretion (Supplementary Tables S3–S4, not inspected).
- **Endpoint (p. 6).** VLST under the **ARC 2007** criteria: stent thrombosis more than one year after implantation, confirmed by coronary angiography. Each case was reviewed independently by two interventional cardiologists, with a third if they disagreed. "The primary endpoint of this study was definite ST."
- **Ascertainment (p. 6).** A 24-hour online consultation service; suspected cases transferred to a chest-pain centre; angiograms uploaded to a core laboratory.
- **Event timing (p. 2).** "Median time from index PCI to the occurrence of VLST" was 697 days (derivation) and 803 days (validation). Incidence was 1.77% and 1.70%.
- **Variable selection (pp. 2, 6).** Variables with P < 0.05 in **univariable Cox regression**, plus 4 risk factors from the literature (Supplementary Table S2), entered a multivariable Cox model with **backward stepwise** selection. Leukocyte count was excluded because infection or other causes could not be ruled out. Procedural variables (stent diameter, length, type, post-dilation) "were based on the operator's subjective assessment during PCI".
- **Final model (p. 2, Table 2 p. 4).** Eight predictors. Variance-inflation factor < 2. Hosmer–Lemeshow P = 0.082. c-statistic 0.81 (0.76–0.86). Leave-one-out cross-validated c 0.75 (0.70–0.80).
- **Integer score (p. 7).** "Corrected coefficients" were divided by the smallest coefficient (0.51, three-vessel disease) and rounded:

  | Item | Points |
  |---|---|
  | Diabetes | 1 |
  | Previous PCI | 3 |
  | AMI as admitting diagnosis | 1 |
  | eGFR < 90 mL/min/1.73 m² | 1 |
  | Three-vessel disease | 1 |
  | Stents per lesion | 2 per stent |
  | SES | 1 |
  | No post-dilation | 4 |

  Arithmetic check [V]: 0.58/0.51 = 1.1 → 1; 1.68/0.51 = 3.3 → 3; 0.54/0.51 = 1.1 → 1; 0.76/0.51 = 1.5 → 1; 0.96/0.51 = 1.9 → 2; 0.66/0.51 = 1.3 → 1; 1.93/0.51 = 3.8 → 4.
- **Score performance (p. 2).** Derivation c 0.80 (0.75–0.85); HR per point 1.69 (1.54–1.85); Hosmer–Lemeshow P = 0.345; decile calibration plot (Supplementary Fig. S2). Validation c 0.82 (0.76–0.90); HR 1.74 (1.53–1.99).
- **Risk groups (p. 2).** Defined by predicted incidence: ≤ 7 points (< 1%), 8–9 (1–3%), ≥ 10 (> 3%). Observed incidence 0.5%, 2.2% and 8.7% (derivation) and 0.45%, 2.34% and 9.3% (validation). Decision-curve analysis compared the score with Dangas's LST score (Fig. 2).
- **Software (p. 7).** SPSS 23; R 3.5.0 with `rms`.

### C2. Generally required for a Cox analysis but **not reported** by Wang

A survival outcome has two parts for each patient: whether the event happened, and the time at which it happened or at which observation stopped. A patient who is still event-free when observation stops is *censored*: they contribute information up to that time without being counted as a non-event for the whole period. A Cox model needs a time origin, a rule for when each patient enters the risk set, and censoring rules. Wang does not report:

- **The time origin and risk-set entry.** The phrase "time from index PCI to VLST" implies index PCI as the origin. VLST can only occur after day 365, but the paper does not say whether follow-up started at PCI (with no events possible in year 1), whether a one-year landmark or delayed entry was used, or how patients who had early or late ST were handled.
- **Censoring rules.** Whether censoring was at last contact, at study end, or at death.
- **Deaths after discharge and other competing events.** Not reported. Lost-to-follow-up patients were excluded rather than censored (p. 2).
- **Meaning of "corrected β".** Undefined. It could be shrinkage or optimism correction, but the paper does not say.
- **How "predicted incidence" was derived from the score, and over what horizon.** This is the basis of the risk groups (Supplementary Fig. S1, not inspected).
- **Proportional-hazards checks, missing data, and how c was computed** (for example Harrell's C).

### C3. Internal inconsistencies in Wang's paper [C]

1. **Post-dilation.**
   - Table 2 gives "No post-dilation" β = 1.93, HR **0.145** (0.08–0.26) and **+4** points.
   - exp(1.93) = 6.9, and 1/0.145 = 6.9. The HR therefore corresponds to −1.93, the opposite sign.
   - Table 1 labels a row "No post-dilation" with 49.01% in patients without VLST vs 15.22% in patients with VLST, which would make *no* post-dilation protective.
   - The abstract, the Discussion and the score treat no post-dilation as a risk factor.
   - In the public file, the Table 1 percentages belong to the column `1.1:1Post dilation`; its complement `No postdilation` is 84.78% in cases.
2. **Risk-group sizes.**
   - The published sizes, 3,135 + 1,837 + 473, sum to 5,445, not 5,185. The percentages (60.5 + 35.4 + 9.1) sum to 105.
   - Applied to the file, the score gives 3,135 / **1,577** / 473 [V]. So 1,837 is very likely a typographical error.
3. **SES described as first-generation.** The Discussion calls SES "first-generation" DES (p. 5), while the Methods say all patients received newer or second-generation DES (p. 6).
4. **HbA1c.** Wang's Table 1 gives HbA1c 6.27 ± 1.53 (no VLST) vs 5.80 ± 1.61 (VLST). The file gives exactly the reverse assignment [V]. The file's values are internally consistent with diabetes status in both groups [V], so a column transposition in Wang's Table 1 is a working hypothesis.
5. **Counts.** Previous PCI is 13 (14.13%) cases in Wang's Table 1 vs 10 (10.87%) in the file. AMI on admission is 3,046/72 in Wang vs 3,095/65 in the file [V].

### C4. How our frozen-score evaluation relates to Wang's

- **What was done [V].**
  - The published integer weights were applied unchanged to all 5,185 rows, using `Diabetes`, `Previous PCI`, `Initial diagnosis-AMI`, `CKD90`, `3-vessel disease`, 2 × `No.of stents per lesion` (integers 1–4), `PES` and 4 × `No postdilation`.
  - Discrimination of the binary label: ROC-AUC 0.8013 and AP 0.1032. Per outer fold (evaluation only): ROC-AUC 0.80 ± 0.06 and AP 0.11 ± 0.05.
  - Observed rates by risk group: 16/3,135 (0.51%), 35/1,577 (2.22%) and 41/473 (8.67%).
- **Coding of post-dilation.** The notebook rejects the alternative coding partly because it gives ROC-AUC 0.508. **Choosing a mapping because it reproduces the published AUC is not acceptable as the sole basis.** The defensible basis is:
  - the column names (`No postdilation` vs `1.1:1Post dilation`);
  - Wang's abstract, Discussion and Table 2, which state that no post-dilation is a risk factor with positive β and +4 points.

  Report it this way, state the contradiction in Wang's Table 1 and Table 2 HR, and submit Q7.
- **Defensible interpretation.** A *reference ranking score* evaluated against the same observed labels as the ML models. It is **not**:
  - a reproduction of the Cox model or of Wang's c-statistic, which was presumably time-to-event (unreported);
  - external validation, because these are Wang's own derivation patients;
  - a fair "ML vs clinical score" contest. The ML models use roughly 80 recorded variables, including post-baseline medications and artefact-bearing laboratory values. The score uses 8 items, one of which (eGFR < 90) is affected by issue A2.1. The development histories also differ: the score was developed on these rows, while ML predictions are out-of-fold within these rows.
- **The numerical similarity between ROC-AUC 0.80 and Wang's c of 0.80 is not a replication.** The two statistics are not established to be the same quantity.
- **Interpretations to avoid:** "TabPFN outperforms Wang"; "the Wang score was validated"; "ML improves on clinical risk scoring for treatment decisions".

---

## D. Integrated revised manuscript

See [VLST_manuscript_revised.md](VLST_manuscript_revised.md).

---

## E. Variable queries and email-ready list

See [VLST_author_queries_email.md](VLST_author_queries_email.md). Working hypotheses are kept there, separate from confirmed facts.

---

## Evidence and conflict log (compact)

| Item | Sources | Status |
|---|---|---|
| n = 5,185; events 92; prevalence 1.77% | Wang p. 2; file [V]; all drafts | Consistent |
| Median follow-up 1,502; median time to VLST 697 | Wang p. 2 | **[C]** with file (TSSI medians 1,416 and 736) |
| Controls' follow-up 1,241–1,605 days; events days 380–1,575 | File [V]; FR | Consistent |
| 12 events after day 1,461; 3,090 controls followed < 1,461 days | File [V] | New |
| Medication = follow-up persistence | FR (stated); Wang p. 2 ("DAPT during follow-up"); file (aspirin 62%) [V] | Consistent; exact time point [M] |
| Thinking v3.5 AP 0.9212 [0.8785, 0.9613]; ROC-AUC 0.9963; Brier 0.0047 | FR; stored CSV [V] | Consistent |
| Local v3.5 AP 0.8957 [0.8446, 0.9426] | FR; CSV [V] | Consistent |
| Operating point, thinking v3.5: 5,083/10/15/77 | FR; CSV [V] | Consistent |
| Paired ΔAP 0.2941 (0.2071–0.3805) vs LightGBM | FR; CSV [V] | Consistent |
| Wang score ROC-AUC 0.8013, AP 0.1032 | FR; notebook [V] | Consistent |
| 13-covariate OR table | FR; CSV [V] | Consistent |
| SHAP from local model on all 1,556 rows | Code output [V] | **[C]** with FR wording |
| SFS: K = 8, 3-fold, 8 seeds | Code output [V] | **[C]** with FR "top-10, 5-fold" |
| Leakage contrast: SMOTE before inner grid-search CV | Code [V] | Not disclosed in FR |
| Precision equalisation: plain 5-fold, hosted arms excluded, CaI and HbA1c rounded to integers | Code [V] | Partly disclosed |
| "Pre-specified" cleaning | FR; code comments | **[M]**: no protocol |
| `LV` = "LV diastolic" | Met items list | **[U]**; no source |

---

## Prediction timestamp and outcome window (Section 4 of the brief)

| Element | Intended | What the data and analyses actually support |
|---|---|---|
| Time origin | Index PCI | Index PCI (Wang p. 2: "time from index PCI") [D] |
| Prediction time | A specified post-PCI time | Not defined in any draft. If `CaI` is peak troponin I, the earliest admissible time is after peak troponin has been determined, practically **at discharge** [I]. Medication columns imply a time **after year 1** [V] |
| Predictor availability cutoff | That prediction time | Not enforced: four medication columns (and possibly others; see the table below) are recorded later |
| Outcome window | Days 366–1,461 | Not constructed. The label counts VLST at any time during follow-up (events on days 380–1,575) [V] |
| Eligibility | Alive at discharge, followed | Wang's exclusions (deaths before discharge, refusals, lost to follow-up) [D]. Post-discharge deaths: [M] |
| Follow-up requirement | ≥ 1,461 days or an event | Not met for 3,090/5,093 controls [V] |
| Censoring and competing events | Required for a fixed horizon | No controls followed < 1,241 days, so either no post-discharge deaths are in the file or TSSI is not time to last contact [I]. Query Q2 |

**Recommended wording:** "The present analyses classify definite VLST recorded during follow-up that ended between 1,241 and 1,605 days after PCI in event-free patients. They do not establish a fixed 1–4-year risk model; that intended application remains unvalidated."

---

## Feature-timing table (Section 5 of the brief)

| Feature or group | Documented meaning | Timing | Available at post-PCI prediction? | Concern | Minimum remedy |
|---|---|---|---|---|---|
| `Aspirin`, `Clopidogrel`, `Ticagrelor`, `DAPT` | Antiplatelet use during follow-up (Wang p. 2 for DAPT; FR) | After year 1 (aspirin 62% despite mandatory DAPT) [V]; exact time [M] | **No** | Post-baseline; could reflect events (discontinuation, or recording at the event); case coding inconsistent (8/35) | Exclude from any post-PCI model (R1); report associations as follow-up-period exposures only |
| `Time since stent implantation` | Time to VLST (cases) or to end of follow-up (controls) [I] | Follow-up | **No** | Outcome-dependent time at risk; cases and controls overlap only on days 1,241–1,575 (22 events lie in the control range), so separation is **not** complete | Already excluded. Appropriate only as the time axis of a survival analysis |
| `eGFR`, `CKD5`, `CKD60`, `CKD90` | eGFR and CKD-stage cut-points (`CKD5` = stage 1–5 of recorded eGFR) [V] | Probably admission [U] | Probably yes [U] | Recorded eGFR not reproducible from Cre in 56.0% of controls; outcome-differential | Recalculate from Cre (R1, R2); Q1 |
| `Cre` | Creatinine, µmol/L [I: values] | [U] | Probably | Formatting differs by group | Quantised; Q1 |
| `CaI` | Matches Wang's "peak troponin I" mean and SD in both groups [V] | Peak, therefore post-PCI | Only if prediction is at or after discharge | Precision signature (2-dp indicator AUC 0.62) | Define prediction time as discharge; Q5 |
| `LV` | Unknown | [U] | [U] | Large difference (d ≈ 1.1); different digit pattern | Q4; drop-column sensitivity (O3); no mechanism |
| `WBC` | WBC count ×10⁹/L (matches Wang) [V] | [U] (admission or peak?) | [U] | Strongest precision signature (integer-valued 74% of controls vs 0% of cases) | Already excluded from prediction; association only, with an artefact caveat |
| `Fiberinogen`, `Fast-Glu`, `HbA1c`, lipids | Fibrinogen g/L; fasting glucose mmol/L; HbA1c %; lipids mmol/L (match Wang) [V] | [U] | Probably | Precision signatures; HbA1c group columns swapped vs Wang | Quantised (partly); Q8 |
| Procedural items (stents, post-dilation, overlap, diameters) | Procedure records | Index PCI | Yes | Post-dilation coding (Q7) | Retain |
| `PES`/`ZES`/`EVS`, `Stent type-SES` | Drug-class flags and brand names | Index PCI | Yes | Labels (Q6) | Retain; fit the codebook within folds (done) |
| History and presentation flags | Admission | Admission | Yes | Previous-PCI and AMI counts differ from Wang | Q8 |

---

## Analysis streams (Section 7 of the brief)

| Analysis | Population | Split | Purpose | Selection or tuning | Outputs | Relation to primary benchmark |
|---|---|---|---|---|---|---|
| Univariate and multivariable association | All 5,185 | None | Exploratory association | Covariates chosen from the FDR screen, one per collinear block (post-selection inference) | Table 1, Table 4 (ORs) | Independent; not used for feature selection |
| Classical selectors (LOCO, coalition SHAP, FFS) | All 5,185 | One stratified 80/20 split (4,148/1,037), seed 42 | Feature relevance for 7 classical models | Per-selector prefixes | Catalogues | Not used in the benchmark |
| Bundled leakage contrast | All 5,185 | One stratified 70/30 split (3,629/1,556), seed 42 | Show the effect of five combined data choices | Grid search (5-fold, AP); SMOTE before grid-search CV in the leaky arm | Table S-leak | Separate; winning settings not imported |
| **Primary benchmark** | All 5,185 | Stratified 5 outer × 4 inner, seed 42 (outer); inner seeds 10,001–10,005 | Compare 9 arms | Inner loop: F1 threshold only. Hosted arms: AP-targeted thinking mode | Tables 3–4 | — |
| Precision-equalisation sensitivity | All 5,185 | Same 5 outer folds, no inner loop | Residual artefact | None | Table S-eq | Same folds; hosted arms excluded |
| Precision probe | All 5,185 | 5-fold (same seed) | Artefact floor | None | AP 0.427 | Same folds |
| TabPFN interpretation (MI, SFS, PDP, SHAP, k-SII) | All 5,185 | Same 70/30 split as the leakage contrast | Exploratory attribution | SFS: K = 8, 3-fold, 8 seeds, `n_estimators = 1` | Table 5, Figure S | Different model, split and constructor from the benchmark |
| Frozen Wang score | All 5,185 | None (folds used for evaluation only) | Clinical reference | None | Table 3 row; Figure 2 | Same labels; different development history |

Patients are not duplicated: 5,185 rows, one per patient [V: no duplicate rows were removed in preprocessing]. All benchmark arms share identical outer folds [V]. One 5 × 4 run is **not** repeated nested cross-validation.

---

## Writing-quality review (representative examples)

The drafts are mostly readable. The formal report, however, uses a compressed style of short declarative fragments full of internal artefact names, which reads as a lab log. The Discussion docx swings the other way, into generic clinical-benefit language. I make no judgement about authorship.

| Original | Problem | Revision |
|---|---|---|
| "The decision window of interest is risk stratification more than one year after PCI, after the mandated year of dual antiplatelet therapy." (FR Introduction) | Contradicts the intended post-PCI timestamp | Define the prediction time once in Methods and use it throughout |
| "In the future, this approach could support more focused care, fewer avoidable procedures, and lower healthcare costs." (Dis) | Unsupported utility chain | "Whether any VLST model can improve follow-up or treatment decisions would require external validation and an assessment of clinical utility." |
| "This rounding is not post-split leakage." (FR) | Assertion without the argument | "Because the rounding rule does not use outcome data, applying it before splitting does not transfer information from held-out folds; it does not, however, remove other outcome-related recording differences." |
| "Rather than treating the published risk score as a final answer, we sought to examine the same clinical problem in greater detail…" (Dis) | Vague purpose | State the actual aims |
| "Thinking-high client inference and local inference are different objects." (FR, repeated) | Repetition, jargon | State once: "hosted and local models were evaluated as separate arms" |
| "Recent advances … TabPFN, have introduced new approaches…" (Int) | Generic | Say what TabPFN is and which version-specific claims are supported |

---

## F. Figure and table action log

No images were viewed, so recommendations rest on captions and context. **Visual inspection is pending for all figures.**

| Current item (FR) | Recommendation | New number |
|---|---|---|
| Table 1, cohort characteristics | Keep; add a footnote on file vs Wang discrepancies; add the recalculated-eGFR row | Table 1 |
| New: data-provenance audit | **Add to main text** | Table 2 |
| Table 4 + bootstrap CI (ranking) | Keep; merge with ROC-AUC CI and null Brier | Table 3 |
| Table 5 + confusion counts (operating point) | Merge into Table 3 or keep as Table 4 | Table 4 (operating point) |
| Table 2, 13-covariate logit | Keep in main text with an artefact flag; move the Firth and bootstrap versions to the supplement | Table 5 |
| Table 3, leakage contrast | Move to the supplement (bundled contrast) | Table S3 |
| Precision probe and equalisation | Supplement | Table S4 |
| Table 6 (feature screens) | Supplement; condense to ≤ 10 rows in the main text, with model and split labels | Table S5 (main-text excerpt in Results) |
| Table 7 (overlap counts and Jaccard) | Remove (redundant with Table S5) | — |
| Appendix C (harmonised table) | Supplement | Table S6 |
| Fig. nested PR/ROC curves | Keep | Figure 1 |
| Fig. calibration curves | Keep (only calibration evidence); caption must state the quantile bins and the null Brier | Figure 2 |
| Fig. Wang score event rate by bin | Keep | Figure 3 |
| New: eGFR recorded vs recalculated scatter, by outcome | **Add** (strongest single provenance figure) | Figure 4 (to be generated) |
| EDA figs 1–5 (test map, significance, effect sizes, binary ORs, stent rates) | Supplement figs S1–S2 (merge significance and effect sizes); remove the test-selection map | S1–S2 |
| Uni vs adjusted OR plot | Supplement | S3 |
| Leakage ROC/PR on/off, confusion on/off, PR-AUC bars | Merge into one supplementary panel | S4 |
| SHAP summary + beeswarm | Keep **one** (beeswarm) in the supplement, labelled "local TabPFN fitted on the 70% split" | S5 |
| SHAP scatter (file named "age", shows eGFR) | Remove or fix the name; it depicts the artefact-affected eGFR | — |
| PDP continuous + binary | Supplement, labelled "training rows, in-sample" | S6 |
| Single-patient waterfall, k-SII network, UpSet, SHAP-IQ force/network/UpSet (6 figures) | **Remove**. One patient, budget 256 for 80 features; the two k-SII runs on the same row gave different top-20 lists [V] | — |
| Selector overlap figures (Venn, heatmaps, reason buckets, domain counts, 10 selector figures) | Remove; keep one membership heatmap in the supplement if needed | S7 (optional) |

---

## Reference-verification log

| Reference | Claim supported | Status |
|---|---|---|
| Wang X et al. *Sci Rep* 2020;10:6378. doi:10.1038/s41598-020-63455-0 | Cohort, score | Verified (PDF). FR title was wrong |
| Wang X et al. Figshare doi:10.6084/m9.figshare.7409606 | Data source | DOI as given in drafts; record not opened. [REFERENCE TO VERIFY: Figshare record title, author list and licence] |
| Cutlip DE et al. Clinical end points in coronary stent trials. *Circulation* 2007;115:2344–2351 | ARC 2007 definitions used by Wang | Added; standard reference |
| Garcia-Garcia HM et al. ARC-2. *Eur Heart J* 2018;39:2192–2207 | Contemporary definitions | In drafts; consistent |
| Dangas GD et al. *JACC Cardiovasc Interv* 2012;5:1097–1105 | LST score, c 0.66 | Verified via Wang's reference list |
| Godschalk TC et al. *J Interv Cardiol* 2018;31:159–169 | Prior VLST score | Verified via Wang's reference list |
| Baran KW et al. *EuroIntervention* 2011;6:949–954 (PMID 21330242) | ARRIVE VLST score: renal disease, prior MI, multiple stenting, bifurcation, prior CABG, smoking | Verified (web) |
| Yeh RW et al. DAPT score. *JAMA* 2016;315:1735–1749 | Treatment-decision scores need trial-based evidence | Added; standard reference |
| Mauri L et al. *N Engl J Med* 2014;371:2155–2166 | DAPT trial | In Dis; consistent |
| van Werkum JW et al. *J Am Coll Cardiol* 2009;53:1399–1409 | Dutch ST registry | In Dis; verified via Wang's references |
| Brodie B et al. *JACC Cardiovasc Interv* 2012;5:1043–1051 | ST predictors after STEMI | In Dis. [REFERENCE TO VERIFY: pages and DOI] |
| Adriaenssens T et al. PRESTIGE. *Circulation* 2017;136:1007–1021 | OCT mechanisms | Verified via Wang's references |
| Fröbert O et al. *PLoS One* 2013;8:e56348 | SCAAR post-dilation | In Dis. [REFERENCE TO VERIFY: volume and e-number] |
| Klein LW et al. SCAI consensus. *JSCAI* 2023;2:100971 | ISR/ST management | In Dis. [REFERENCE TO VERIFY] |
| Chi G et al. ATLAS ACS 2–TIMI 51. *Front Cardiovasc Med* 2024;10:1269011. doi:10.3389/fcvm.2023.1269011 | WBC and early ST; VLST predictors | Verified (web); Dis had no authors or journal |
| Kakimoto Y et al. REAL-ST hemodialysis. *Circ J* 2026;90(8). doi:10.1253/circj.CJ-26-0034 | ST in dialysis | Verified (web). Pages conflict: PubMed 969–977 vs J-STAGE 1010–1021. **Dropped** from the revised text (outcomes after ST, not VLST prediction) |
| Saeed Z et al. *Int J Mol Sci* 2024;25:8705 | CKD thrombosis mechanisms | Verified (web). **Dropped** (renal mechanism paragraph removed) |
| Seko Y et al. *BMJ Open* 2019;9:e032663 | Diastolic dysfunction outcomes | Verified (web). **Dropped** (LV paragraph removed) |
| Richardson-Lobbedez M et al. *Eur J Echocardiogr* 2008;9:594–598 | Diastolic function in ACS | **Dropped** (same reason) |
| Zaka A et al. *Eur Heart J Digit Health* 2025;6:23–44 | ML after PCI | Verified (web) |
| Nayebirad S et al. *BMC Cardiovasc Disord* 2025;25:310 | ML vs LR after PCI | Verified (web; results not read) |
| Saito T, Rehmsmeier M. *PLoS One* 2015;10:e0118432 | PR plots for imbalance | In drafts; consistent |
| Hollmann N et al. *Nature* 2025;637:319–326 | TabPFN (v2) foundations | In drafts; supports TabPFN generally, **not** v3/v3.5 specifics |
| Jäger B et al. TabPFN-3.5: Technical Report. arXiv:2609.17895 (2026) | v3.5, Thinking mode | Verified (arXiv abstract) |
| Prior Labs documentation: TabPFN-3.5 changelog; Thinking mode; model-version selection | Package versions; API-only Thinking; metric-targeted fitting | Verified (web docs). Cite as documentation with an access date |
| [REFERENCE TO VERIFY: TabPFN-3 technical report] | v3 checkpoint | Not located |
| Collins GS et al. TRIPOD+AI. *BMJ* 2024;385:e078378 | Reporting guidance | Added; standard reference |
| Varma S, Simon R. *BMC Bioinformatics* 2006;7:91 | Nested CV | Added |
| Kaufman S et al. Leakage in data mining. *ACM TKDD* 2012;6(4):15 | Leakage | Added |
| Chawla NV et al. SMOTE. *J Artif Intell Res* 2002;16:321–357 | SMOTE | Added |
| Benjamini Y, Hochberg Y. *J R Stat Soc B* 1995;57:289–300 | FDR | Added |
| Van Calster B et al. Calibration. *BMC Med* 2019;17:230 | Calibration reporting | Added |
| Levey AS et al. CKD-EPI. *Ann Intern Med* 2009;150:604–612 | eGFR recalculation | Added |
| Lundberg SM, Lee S-I. *NeurIPS* 2017 | SHAP | Added |
| [REFERENCE TO VERIFY: shapiq library (Muschalik et al., NeurIPS 2024 Datasets & Benchmarks)] | Implementation | Not verified this session |
| [REFERENCE TO VERIFY: k-SII definition (Bordt & von Luxburg, AISTATS 2023)] | Interaction index | Not verified this session |

---

## G. Concise change log and cross-section consistency check

**Main changes in the revised manuscript**

1. **Title and framing.** Changed from "prediction of VLST" to "discrimination of recorded VLST, provenance audit, and reference comparison".
2. **New Methods §2.2, "Prediction time and outcome definition".** It states the intended use, what was actually modelled, and the unvalidated status of the 1–4-year use.
3. **New data-provenance audit** (Methods §2.4, Results §3.2, Table 2).
4. **Medication columns** relabelled as follow-up-period covariates. R1 is specified as a placeholder.
5. **eGFR artefact** reported. The renal mechanism paragraph is removed. R2 is specified.
6. **LV and CaI** reported as undefined or provisionally matched dataset variables. The LV diastolic-function paragraph is removed.
7. **Wang section** rewritten: the frozen reference score, the coding rationale, and the inconsistencies in Wang's paper.
8. **Metrics** corrected: AP, ROC-AUC explanation, null Brier, ECE limits, bootstrap wording.
9. **Tuning budgets** described accurately: hosted thinking arms optimised AP; classical arms untuned.
10. **Interpretation provenance** corrected: a local model, all rows recomputed locally, K = 8 / 3-fold SFS, unlogged checkpoint, non-probability SHAP scale.
11. **Clinical-utility claims** removed. The DAPT-guidance implication is explicitly disclaimed.
12. **References** consolidated and verified, with the remaining items marked.

**Cross-section consistency (checked in the revised manuscript)**

| Element | Value used everywhere |
|---|---|
| Cohort and events | 5,185; 92; 1.77% |
| Prediction time | Intended: at discharge after index PCI. Not achieved by the current matrix |
| Outcome | Definite VLST recorded during follow-up (events days 380–1,575; controls followed 1,241–1,605). The 1–4-year horizon is not constructed |
| Predictor availability | Medications post-baseline; eGFR artefact; LV and CaI unconfirmed |
| Splits | 5 × 4 nested (primary); 70/30 (bundled contrast, attribution); 80/20 (classical selectors); full cohort (association, Wang) |
| Models | 5 classical + local v3, local v3.5, hosted thinking v3, hosted thinking v3.5 |
| Interpretation model | Local TabPFN (checkpoint pending C1), fitted on the 70% split |
| Wang | Frozen integer score; reference only |
| Metrics | AP (primary), ROC-AUC, Brier with null Brier, operating point from inner F1 thresholds |
| Clinical claims | Association only; no causal, treatment or utility claims |
| Contribution | Internal benchmark + provenance audit + reference score + hypothesis generation |
| Limitations | Single file; artefacts; horizon; timing; no external validation |

**Submission status: not submission-ready.** Queries Q1–Q9 and checks C1–C2 are material. R1 is essential if any post-PCI prediction claim is kept.
