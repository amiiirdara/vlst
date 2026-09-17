# Manuscript traceability matrix

Insert values only from `paper/frozen_results.yaml` unless a cell is marked **TODO**.  
Claim types: `methods` · `association` · `prediction` · `attribution` · `historical_comparator` · `leakage` · `limitation` · `terminology` · `todo`.  
Status: `frozen` · `methods_only` · `unresolved` · `excluded` · `complete_if_worded_as_allowed`.

Global prohibited wording for **all** rows: external validation (of this pack’s ML); derivation cohort as external validation cohort; independent risk factor; causal / protective / clinically useful / validated / personalised (as a result); collapsing TabPFN arms; TSSI as a baseline predictor; **Part 4 Table 3** pooled F1 as nested performance; **Part 1 Table 4** as the clinical logit; encoder n_raw = 99; local Brier 0.0673 as Version 4.

Use **MS Table / MS Figure** IDs from `paper/figure_table_plan.md` when a number could mean two report tables.

---

## Legend for allowed wording (default)

| Claim type | Allowed | Prohibited |
| --- | --- | --- |
| association | associated with; adjusted odds ratio in the Table 4b specification; lower modelled odds of recorded VLST | independent risk factor; causal; protective; treatment benefit |
| prediction | nested-CV out-of-fold ranking performance; model discrimination; calibration (Brier / reliability curves); predictive performance on the derivation cohort | external validation; test-set performance (unless nested OOF is named); clinically useful |
| attribution | attribution; model explanation; held-out split of the same derivation file | feature mask for Part 4; cohort interactions (for k-SII); Part 4 nested-CV risk (for PDP) |
| historical_comparator | frozen Wang 2020 integer score; historical comparator on the derivation cohort | validation cohort; externally validated TabPFN; eighth nested-CV arm |
| leakage | leakage-sensitive follow-up time; time-at-risk; demonstration with SMOTE mismatch named | identical protocol except TSSI; nested-CV result |

---

## Matrix

| manuscript section | sentence/claim ID | claim text or placeholder | YAML path | source file | source location | claim type | allowed wording | prohibited wording | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 Title | T-01 | Working title: association, nested-CV prediction, and TabPFN attribution on the Wang 2020 derivation cohort | `study.title_or_working_title` | paper_results/00_front_matter.md | Clinical motivation / What this pack adds | terminology | derivation cohort; nested CV | validated; personalised; external | methods_only; paragraph incomplete |
| 2.1 Abstract Background | A-BG-01 | VLST = ARC 2007 definite ST >1 year, angiographically confirmed | `study.outcome_definition` | paper_results/00_front_matter.md | Clinical motivation — Outcome | methods | definite VLST as defined | probable/possible counted | frozen |
| 2.1 Abstract Background | A-BG-02 | Wang 2020 published an 8-variable Cox score on this derivation cohort (c=0.80, 0.75–0.85) | `wang_2020.reported_performance.wang_published_derivation_c` | paper_results/00_front_matter.md | A score already exists | historical_comparator | published derivation c-statistic, cited | this pack externally validated Wang | frozen |
| 2.1 Abstract Background | A-BG-03 | Dangas c=0.66 in Wang’s comparison | `wang_2020.reported_performance.dangas_in_wang_comparison` | paper_results/00_front_matter.md | A score already exists | historical_comparator | Wang’s published comparison, not computed here | we found Dangas c=0.66 | methods_only |
| 2.2 Abstract Objective | A-OB-01 | Aims: association catalogue; nested-CV prediction without TSSI; two TabPFN arms; frozen Wang comparator; TSSI leakage demo | `study.analysis_setting`; `wang_2020.role_in_current_study` | paper_results/00_front_matter.md | What this pack adds | methods | historical comparator; nested-CV prediction performance | validate TabPFN; clinical utility objective | methods_only |
| 2.3 Abstract Methods | A-ME-01 | n=5185 patients, 92 events, prevalence 0.0177 | `study.n_total`; `study.n_vlst`; `study.event_rate` | paper_results/00_front_matter.md | Outcome | methods | 5,185 / 92 / 1.77% | other sample sizes | frozen |
| 2.3 Abstract Methods | A-ME-02 | Nested stratified 5×4 CV, outer seed 42; inner loop F1 threshold only | `validation.outer_cv`; `validation.inner_cv` | paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md | Cohort / protocol | methods | nested CV on the derivation file | external validation; hyperparameter nested search | frozen |
| 2.3 Abstract Methods | A-ME-03 | TSSI excluded from primary nested-CV models | `data_and_predictors.tssi_status.used_as_baseline_predictor_in_nested_cv` | Part 4 protocol | Cohort / protocol | leakage | excluded; leakage-sensitive | baseline predictor | frozen |
| 2.3 Abstract Methods | A-ME-04 | Thinking-high = tabpfn_client thinking high; local = tabpfn, n_estimators auto, no balance_probabilities | `model_results.tabpfn_thinking_high.model_definition`; `tabpfn_local.model_definition` | Part 4 | This run (D4); Table 0 | methods | name both constructors | one “TabPFN” result | methods_only |
| 2.3 Abstract Methods | A-ME-05 | Bootstrap n_boot=2000 seed 42 on pooled OOF, models not re-fit | `validation.bootstrap_method` | Part 4 §3 | Uncertainty | methods | stratified bootstrap of stored OOF | re-fitted bootstrap; nested-CV CI from inner grid | frozen |
| 2.3 Abstract Methods | A-ME-06 | Complete 81-name list | `data_and_predictors.baseline_predictors.complete_81_name_list` | — | — | todo | omit list; state count 81 | invent names | unresolved |
| 2.4 Abstract Results | A-RE-01 | Thinking-high PR-AUC 0.8553 [0.7957, 0.9131] | `model_results.tabpfn_thinking_high.metrics.pr_auc` | Part 4 | Table 1; Table S-CI | prediction | nested-CV OOF ranking performance | external test PR-AUC | frozen |
| 2.4 Abstract Results | A-RE-02 | Thinking-high ROC-AUC 0.9905; Brier 0.0064 | `...metrics.roc_auc`; `...metrics.brier` | Part 4 | Table 1; S-CI | prediction | discrimination; calibration (Brier) | clinically useful; collapse with local | frozen |
| 2.4 Abstract Results | A-RE-03 | LightGBM PR-AUC 0.6942 [0.6065, 0.7782] | `model_results.baseline_models_without_tssi.lightgbm.nested_cv_oof.pr_auc` | Part 4 | Table 1; S-CI | prediction | second on PR-AUC among seven | best model without naming metric | frozen |
| 2.4 Abstract Results | A-RE-04 | Local PR-AUC 0.6742 [0.5864, 0.7657]; Brier 0.0102 | `model_results.tabpfn_local.metrics.pr_auc`; `...brier` | Part 4 | Table 1; S-CI | prediction | fourth on PR-AUC; booster-band Brier | worst Brier; merge with thinking-high | frozen |
| 2.4 Abstract Results | A-RE-05 | Δ thinking−LGB PR-AUC 0.1611 (0.0984–0.2289), P(Δ≤0)=0/2000 | `tabpfn_thinking_high.metrics.delta_pr_auc_vs_lightgbm` | Part 4 | Table S-Δ | prediction | paired bootstrap on pooled OOF | superiority in new patients | frozen |
| 2.4 Abstract Results | A-RE-06 | Nested thinking-high recall 0.7065, F1 0.7471, 5076/17/27/65 | `tabpfn_thinking_high.metrics.nested_operating_point_table2` | Part 4 | Table 2 | prediction | honest nested operating point | pooled recall 0.8152 | frozen |
| 2.4 Abstract Results | A-RE-07 | Wang frozen integer score ROC-AUC 0.8013, PR-AUC 0.1032 | `wang_2020.reported_performance.this_pack_frozen_integer_score` | Part 4 | §7 Headline | historical_comparator | frozen points on the same 5,185 rows | external validation of ML | frozen |
| 2.4 Abstract Results | A-RE-08 | TSSI LR PR-AUC 0.9575→0.5077; CatBoost 0.9773→0.6582 | `tssi_leakage_analysis.models_*` | Part 4 | Table S-TSSI | leakage | 70/30 leakage demonstration; SMOTE mismatch | nested-CV; identical protocol | frozen |
| 2.4 Abstract Results | A-RE-09 | Table 4b post-dilation adj OR 0.152; Clopidogrel 0.480; WBC 1.972 | `statistical_analysis.association_estimates.table4b_adjusted_or` | Part 1 | Table 4b | association | associated with recorded VLST after adjustment in the 13-covariate logit | independent risk factor; protective | frozen |
| 2.5 Abstract Conclusions | A-CO-01 | ML models were not externally or temporally tested | `validation.external_validation.ml_models` | paper_results/00_front_matter.md | Limitations item 1 | limitation | nested CV is not external validation | validated; ready for clinical use | frozen |
| 2.5 Abstract Conclusions | A-CO-02 | Local vs LightGBM Δ PR-AUC −0.0201 (−0.0974–0.0566) compatible with no difference | `tabpfn_local.metrics.delta_pr_auc_vs_lightgbm` | Part 4 | Table S-Δ | prediction | compatible with no difference as written | local inferior/superior as a clinical claim | frozen |
| 3 Introduction | I-01 | File is Wang 2020 derivation cohort; Jilin University ACS-PCI 1 Jan 2014–1 Jun 2015 | `study.cohort` | 00_front_matter.md | Outcome | methods | derivation cohort | external validation cohort | frozen |
| 3 Introduction | I-02 | Flow 6038→5185: 236 deaths, 413 refused, 204 lost, cited from Wang | `study.exclusion_criteria` | 00_front_matter.md; Part 1 Table C caption | Outcome | methods | cited from Wang 2020 | reconstructed from extra repo eligibility code | frozen |
| 3 Introduction | I-03 | Wang Shantou c=0.82, n=2058, file not in repository | `wang_2020.reported_performance.wang_shantou`; `validation.external_validation.wang_shantou` | 00_front_matter.md | A score already exists; limitation 1 | historical_comparator | Wang’s external test of Wang’s score | this pack’s external validation | methods_only |
| 3 Introduction | I-04 | This pack does not add ML external/temporal testing, Cox re-fit, or Dangas DCA | `wang_2020.limitations`; `study.analysis_setting` | 00_front_matter.md | What this pack adds / does not add | limitation | absent | we validated | methods_only |
| 3 Introduction | I-05 | Extra epidemiology beyond Wang citation | — | — | — | todo | omit or add later sourced sentence | invent incidence | unresolved |
| 4.1 Study design | M-410-01 | Derivation-cohort analysis only; all Parts use 5185 rows; splits differ | `study.analysis_setting` | 00_front_matter.md | Limitations 1; methods inventory A | methods | same file, different splits | one train/test throughout | methods_only |
| 4.1 Study design | M-410-02 | Median follow-up 1502 days; median PCI→VLST 697 days | `study.median_follow_up_days`; `study.median_pci_to_vlst_days` | 00_front_matter.md | Outcome | methods | descriptive follow-up | TSSI as predictor | frozen |
| 4.1 Study design | M-410-03 | Ethics 2013-256; NCT03491891; cite Wang; this analysis not pre-registered | `study.ethics_and_registration` | 00_front_matter.md | Data, ethics, consent | methods | cite Wang | new IRB claimed | methods_only |
| 4.2 Outcome | M-420-01 | Binary `Stent thrombosis`; probable/possible not counted | `study.outcome`; `study.outcome_definition` | 00_front_matter.md; Part 4 protocol | Outcome | methods | ARC 2007 definite | time-to-event outcome in this pack | frozen |
| 4.3 Predictors | M-430-01 | 81 columns after dropping NO., Name, TSSI | `data_and_predictors.baseline_predictors.n_after_drop_ids_and_tssi` | Part 4 protocol | Cohort / protocol | methods | 81 non-ID non-TSSI columns | TSSI included | frozen |
| 4.3 Predictors | M-430-02 | 106 raw Stent type-SES strings → 9 levels, min_count=30 | `...stent_encoder.n_raw_csv_strings`; `n_levels_after_collapse`; `min_count` | Part 4; Part 1 | This run (D4); Figure 5 | methods | 106→9 | n_raw=99 | frozen |
| 4.3 Predictors | M-430-03 | Part 2 88 columns (OHE drop-first); Part 4 classics ~89 (no drop-first) | `part2_scaled_width`; `part4_classic_width` | Part 2 header; Part 4 methods | feature views | methods | two OHE views of the same encoder | one width for all analyses | frozen / methods_only for ~89 |
| 4.3 Predictors | M-430-04 | Aspirin, Clopidogrel, Ticagrelor, DAPT = follow-up persistence after mandated year | `data_and_predictors.post_index_or_time_dependent_variables` | 00_front_matter.md | Limitation 6; Table C caption | methods | follow-up persistence, not index prescription | index-PCI covariate without caveat | methods_only |
| 4.3 Predictors | M-430-05 | LV and CaI unnamed in CSV | — (caveat in interpretability / front matter) | 00_front_matter.md | Limitation 10 | limitation | unnamed; CaI means match Wang peak TnI in Table C | novel echo/troponin marker | methods_only |
| 4.3 Predictors | M-430-06 | 81 names in CSV header order (Table S0) | `data_and_predictors.baseline_predictors.complete_81_name_list` | data/raw/VLST.csv; paper/table_s0_baseline_predictors.md | column headers | methods | 81 baseline names as stored | invent/rename columns | frozen |
| 4.4 TSSI | M-440-01 | VLST=1 min 380 d; VLST=0 min 1241 max 1605; median follow-up 1502 | `tssi_status.interpretation` | Part 4 §6 | follow-up-time leakage | leakage | time-at-risk / completed follow-up | baseline covariate | frozen |
| 4.4 TSSI | M-440-02 | Used in leakage notebooks only; dropped in nested-CV, Part 2, Part 5 | `tssi_status.used_as_*` | Part 4 §6; methods inventory C | — | leakage | excluded from primary models | included in baseline ML | frozen |
| 4.4 TSSI | M-440-03 | With-TSSI USE_SMOTE=True; without USE_SMOTE=False | `tssi_leakage_analysis.smote` | Part 4 Table S-TSSI caption; conflict_ledger M1 | Table S-TSSI | methods | SMOTE mismatch named | identical protocol except TSSI | frozen |
| 4.4 TSSI | M-440-04 | TSSI 70/30 preprocessing.ipynb details | `preprocessing.unknown_or_unreported_items` | unsupported_claims.md | preprocessing.ipynb | todo | omit pipeline details | invent scaler/imputer for npy | unresolved |
| 4.5 Preprocessing | M-450-01 | No missing values printed; imputers inert | `data_and_predictors.missingness_summary`; `preprocessing.imputation` | Part 1 header; Part 4 methods | Cohort context | methods | no missing values in the analysed file | multiple-imputation results | frozen |
| 4.5 Preprocessing | M-450-02 | Classics: imputer/scaler/OHE fitted inside each CV split; stent encoder before split | `preprocessing.foldwise_preprocessing` | Part 4 | Methods note — feature views | methods | fold-wise transformer; full-cohort min_count | no leakage of brand frequencies (unquantified) | methods_only |
| 4.5 Preprocessing | M-450-03 | Outlier handling: IQR 1.5 screen, no row drop | `preprocessing.outlier_handling` | eda.ipynb | IQR cell | methods | detection-only | winsorise/exclude as modeling rule | methods_only |
| 4.6 Statistics | M-460-01 | Univariate Welch/MW by skew/kurtosis; χ²/Fisher; BH-FDR | `statistical_analysis.inferential_models.univariate` | Part 1 | Cohort context; Table R | association | association tests | causal tests | methods_only |
| 4.6 Statistics | M-460-02 | Table 4: 17 cov, unidentified, EPV≈5.4, do not publish as clinical model | `inferential_models.multivariable_table4`; `epv.table4` | Part 1 | Table 4 | association | unidentified stored fit | the multivariable risk model | excluded as headline; EPV frozen |
| 4.6 Statistics | M-460-03 | Table 4b: 13 cov, identified, unweighted, EPV≈7.1, Wald 95% CI | `multivariable_table4b`; `epv.table4b`; `wald_tests` | Part 1 | Table 4b | association | identified association screen | independent predictors; Firth | frozen |
| 4.6 Statistics | M-460-04 | Part 4 LR is class_weight=balanced — different object from Table 4b | `weighted_vs_unweighted_models` | Part 1 vs Part 4 Table 0 | captions | methods | association logit vs prediction LR | one logistic model | methods_only |
| 4.6 Statistics | M-460-05 | Firth on Table 4b 13-cov design; not nested CV | `statistical_analysis.firth_logistic_regression` | Part 1 Table 4b Firth | paper_table4b_firth_or | association | sensitivity | Firth nested-CV classifier | frozen |
| 4.6 Statistics | M-460-06 | Table S2: 16 LR interaction tests; LV×eGFR q=0.0277; Men×eGFR q=0.028 | `statistical_analysis.likelihood_ratio_tests` | Part 1 | Supplementary Table S2 | association | hypothesis-generating interactions | confirmed effect modification | frozen |
| 4.6 Statistics | M-460-07 | EPV 92/81≈1.14 | `epv.events_per_81_candidates` | 00_front_matter.md | W4 | methods | events per candidate feature | adequate EPV | frozen |
| 4.7 ML models | M-470-01 | Seven nested-CV arms: LR, RF, XGB, LGB, CatBoost, thinking-high, local | `model_results.baseline_models_without_tssi`; TabPFN blocks | Part 4 | Table 0 | methods | seven classifiers, TSSI dropped | DT/NB as nested-CV models | methods_only |
| 4.7 ML models | M-470-02 | Classics: RUN_MODELS kwargs + library defaults; no Part 4 grid | `model_results.part4_constructors`; `validation.inner_cv.tunes` | Part 4 nbdump RUN_MODELS | Table 0 + constructors | methods | inner loop = F1 threshold only | GridSearch nested CV; TSSI max_depth | methods_only |
| 4.7 ML models | M-470-03 | Thinking-high constructor flags | `tabpfn_thinking_high.model_definition` | Part 4 | This run (D4); Table 0 | methods | client thinking-high | local thinking | methods_only |
| 4.7 ML models | M-470-04 | Local: n_estimators auto; balance_probabilities false; restore skipped | `tabpfn_local.model_definition` | Part 4 | This run (D4) | methods | unbalanced local Version 4 | balance_probabilities=True (old dump) | methods_only |
| 4.7 ML models | M-470-05 | TabPFN client/server versions | `provenance.unresolved_items` | 00_front_matter.md | Limitation 5 | todo | unrecorded | invent versions | unresolved |
| 4.8 Feature selection | M-480-01 | Part 2 fit 4148 (74 events) / val 1037 (18 events); PR-AUC; TSSI dropped | `feature_selection_models.protocol` | Part 2 | Cohort / protocol header | attribution | selector catalogues, not prediction | nested-CV feature mask | methods_only |
| 4.8 Feature selection | M-480-02 | LOCO cap 60; SHAP universe 40; FFS pool 24 max 12 min_gain 0 | `feature_selection_models.ffs`; `other_selection_methods` | Part 2 | Figure 1; Selectors paragraph | attribution | independent selectors | one nested importance | frozen |
| 4.8 Feature selection | M-480-03 | FFS path lengths lr=12, rf=12, rf_b=8, cat=4, xgb=12, xgb_b=11, lgb=5 | `ffs.path_lengths` | Part 2 | Figure 1 table | attribution | path length after early stop | 60 important FFS features | frozen |
| 4.8 Feature selection | M-480-04 | 7×3 intersection n=0; scored union n=86 | `other_selection_methods.strict_7x3_intersection_n`; `scored_union_n` | Part 2 | Table 4 | attribution | no name in all 21 top-20s | universal ML biomarker | frozen |
| 4.8 Feature extraction | M-480-05 | FDR n=20 vs ML consensus n=13; intersection 5; Jaccard 0.1786 = 5/28 | `feature_extraction_models.results` | Part 3 | §2; notebook print | attribution | methods comparison | biological ranking of true markers | frozen |
| 4.8 Feature extraction | M-480-06 | Intersection names: WBC, eGFR, LV, HbA1c, 1.1:1Post dilation | `results.intersection_names` | Part 3 | Figure 1 | attribution | names in both catalogues | causal intersection | frozen |
| 4.8 Feature extraction | M-480-07 | Catalogues do not feed Part 4 | `feature_selection_models.protocol`; Part 4 protocol | Part 2/4/5 headers | — | methods | not a leakage-free mask | selected features used in nested CV | methods_only |
| 4.9 Nested CV | M-490-01 | 5 outer / 4 inner, stratified, shuffle, random_state=42 | `validation.outer_cv`; `inner_cv`; `stratification` | Part 4 | Cohort / protocol | methods | nested stratified CV | external validation; repeated nested CV | frozen |
| 4.9 Nested CV | M-490-02 | Events per outer fold 18,18,18,19,19 | `validation.outer_cv.events_per_fold` | 00_front_matter.md | EPV table | methods | thin fold events | adequate per-fold events | frozen |
| 4.9 Nested CV | M-490-03 | Single nested CV, not repeated k-fold | `validation.repetitions` | methods_inventory.md | E. Data splitting | methods | one nested CV | repeated nested CV | methods_only |
| 4.9 Nested CV | M-490-04 | External validation of ML absent | `validation.external_validation.ml_models` | 00_front_matter.md | Limitation 1 | limitation | not performed | nested CV as EV | frozen |
| 4.10 Metrics | M-4100-01 | Primary ranking metric PR-AUC at prevalence 0.0177 | `metrics.pr_auc`; `study.event_rate` | Part 4 | Models paragraph | prediction | PR-AUC ranking performance | ROC-AUC as the informative rare-event metric without caveat | frozen |
| 4.10 Metrics | M-4100-02 | Quote Table 2 nested F1; Table 3 pooled F1 excluded from headline | `metrics.f1`; `baseline_models_without_tssi.*.pooled_f1_table3` | Part 4 | §5 F1 operating point | prediction | honest nested operating point | apparent/in-sample thresholded metrics as nested | frozen / excluded |
| 4.10 Metrics | M-4100-03 | F2 beta=2.0 | `metrics.other_metrics` | Part 4 | Table 2 | prediction | F2 (β=2) | unspecified F-beta | frozen |
| 4.10 Metrics | M-4100-04 | NPV = TN/(TN+FN) from Table 2 2×2 | `metrics.npv` | Part 4 Table 2 counts | Table 2 | prediction | identity from frozen 2×2 | separately computed NPV | derived_from_frozen_2x2 |
| 4.10 Metrics | M-4100-05 | Calibration: quantile-bin curves + Brier; ECE/slope not reported | `metrics.calibration` | Part 4 | Figure 2 | prediction | Brier; reliability curves | well calibrated ECE | frozen / unresolved ECE |
| 4.11 Interpretation | M-411-01 | Part 5 70/30 seed 42: train 3629/64; held-out 1556/28 | `interpretability.caveats`; Part 5 protocol in YAML shap/ffs | Part 5 | protocol | attribution | held-out split of the same file | external test; nested-CV prediction | frozen |
| 4.11 Interpretation | M-411-02 | MI/SFS/PDP on train; SHAP all 1556 held-out; HTTP 429 → local | `interpretability.shap` | Part 5 | protocol; Table 0 | attribution | local after 429 | full-cohort SHAP; 15+15 | frozen |
| 4.11 Interpretation | M-411-03 | k-SII one row cohort index 5176, budget 256 | `interpretability.shapiq` | Part 5 | §5–6 | attribution | one-patient illustration | cohort interaction screen | frozen |
| 4.11 Interpretation | M-411-04 | PDP y-axis train empirical prior, not Part 4 risk | `interpretability.caveats` | Part 5 | PDP section | attribution | empirical prior near prevalence | nested-CV predicted risk | methods_only |
| 4.12 Wang comparator | M-412-01 | Frozen published Table 2 integer points; weights not re-fit | `wang_2020.score_definition` | Part 4 | §7 | historical_comparator | historical comparator | nested-CV eighth arm; Cox re-fit | frozen |
| 4.12 Wang comparator | M-412-02 | SES point on PES; 4 points on No postdilation=1 | `wang_2020.variables` | Part 4 | §7 Encoding | historical_comparator | encoding as implemented | photocopy Wang Table 1 post-dilation | frozen |
| 4.12 Wang comparator | M-412-03 | Alternate 14-event “No post-dilation” encoding ROC-AUC 0.5084 — do not use | `wang_2020.variables.encoding_note` | Part 4 | §7 Encoding | historical_comparator | polarity check, rejected | comparator performance 0.5084 | frozen |
| 4.12 Wang comparator | M-412-04 | Same five outer folds evaluate frozen score only | `wang_2020.reported_performance.this_pack_frozen_integer_score.fold_mean_roc` | Part 4 | §7 | historical_comparator | score not refit | nested-CV Wang model | frozen |
| 5.1 Cohort | R-510-01 | Table 1 n=5093 vs 92; cells as printed in Table C | `study.n_non_vlst`; `figures_and_tables` Part1-TableC | Part 1 | Table C | association | clinical characteristics; association | prediction Table 1 | frozen |
| 5.1 Cohort | R-510-02 | Do not photocopy Wang Table 1 post-dilation; CSV 14/92 have 1.1:1Post dilation=1 | Part 1 Table C caption (copy; not a YAML scalar beyond freeze note) | Part 1 | Table C caption | association | columns as stored | Wang’s 14 as No postdilation in this CSV | complete_if_worded_as_allowed |
| 5.2 Missingness | R-520-01 | No missing values; 106→9 encoder; Fig 5 is 9-level rates | `missingness_summary`; `stent_encoder` | Part 1 | Cohort context; Figure 5 | methods | 106 raw → 9 levels | Raw levels=99 as n_raw | frozen |
| 5.3 TSSI results | R-530-01 | Results subsection omitted; leakage numbers are Methods 4.4 / Table M2 | `tssi_leakage_analysis` | Part 4 | Table S-TSSI | leakage | Methods demonstration; SMOTE mismatch | Results nested-CV; identical protocol | frozen |
| 4.4 TSSI | R-530-02 | With/without PR-AUC, ROC-AUC, F1 per model from YAML/CSV | `tssi_leakage_analysis.models_with_tssi`; `models_without_tssi` | Part 4 CSV | paper_table_s_tssi_leakage.csv | leakage | Methods Table M2 | nested-CV headline | frozen |
| 4.4 TSSI | R-530-03 | Gaussian NB unchanged (never used the column) | `models_* .gaussian_nb` | Part 4 | Table S-TSSI caption | leakage | unaffected | proof SMOTE is irrelevant | frozen |
| 5.4 Associations | R-540-01 | Table 4b 13-row adj OR + Wald CI (WBC 1.972 [1.667, 2.331] … PES 1.734 [0.953, 3.154]) | `association_estimates.table4b_adjusted_or.rows` | Part 1 | Table 4b | association | associated with; Wald 95% CI | independent risk factor; Table 4 0.144/0.464 | frozen |
| 5.4 Associations | R-540-02 | Previous PCI OR estimators differ by table (6.49 / 6.46 / 6.73 / 6.485) — do not mix | `association_estimates.previous_pci_or_estimators_do_not_mix` | Part 1 | Table 2/4/S4/4b captions | association | name the estimator | one Previous PCI OR | frozen |
| 5.4 Associations | R-540-03 | Table 4 excluded as unidentified (EPV 5.4; VIF ∞) | `table4_do_not_quote_examples`; `inferential_models.multivariable_table4` | Part 1 | Table 4 | association | do not publish as clinical model | the adjusted model | excluded |
| 5.4 Associations | R-540-04 | S2 interactions hypothesis-generating | `likelihood_ratio_tests` | Part 1 | Table S2 | association | hypothesis-generating | confirmed interaction effects | frozen |
| 5.5 Baseline ML | R-550-01 | LR PR-AUC 0.3326 [0.2486, 0.4345]; ROC 0.9224; Brier 0.0563 | `baseline_models_without_tssi.logistic_regression.nested_cv_oof` | Part 4 | Table 1; S-CI | prediction | nested-CV OOF | TSSI-with LR 0.9575 | frozen |
| 5.5 Baseline ML | R-550-02 | XGB PR-AUC 0.6815 [0.5881, 0.7703]; ROC 0.9439; Brier 0.0088 | `...xgboost.nested_cv_oof` | Part 4 | Table 1; S-CI | prediction | ranking performance | — | frozen |
| 5.5 Baseline ML | R-550-03 | LightGBM PR-AUC 0.6942 [0.6065, 0.7782]; ROC 0.9681; Brier 0.0093; folds 0.7528, 0.7138, 0.5473, 0.7731, 0.6916 | `...lightgbm` | Part 4 | Table 1; after Table 1 | prediction | highest classic PR-AUC | best overall without naming thinking-high | frozen |
| 5.5 Baseline ML | R-550-04 | CatBoost PR-AUC 0.6172 [0.5250, 0.7148]; ROC 0.9594; Brier 0.0101 | `...catboost.nested_cv_oof` | Part 4 | Table 1; S-CI | prediction | — | — | frozen |
| 5.5 Baseline ML | R-550-05 | RF PR-AUC 0.4865 [0.3860, 0.6034]; ROC 0.9209; Brier 0.0143 | `...random_forest.nested_cv_oof` | Part 4 | Table 1; S-CI | prediction | — | — | frozen |
| 5.5 Baseline ML | R-550-06 | Nested Table 2 2×2 and thresholded metrics for five classics | `*.nested_operating_point_table2` | Part 4 | Table 2 | prediction | nested inner-F1 threshold | Table 3 pooled F1 | frozen |
| 5.5 Baseline ML | R-550-07 | LightGBM nested 5060/33/30/62; recall 0.6739; t 0.112±0.090 | `lightgbm.nested_operating_point_table2` | Part 4 | Table 2 | prediction | honest nested | 5062/31/31/61 (old dump) | frozen |
| 5.6 Extraction | R-560-01 | Jaccard 0.1786; intersection 5; union 28 | `feature_extraction_models.results.jaccard` | Part 3 | §2 | attribution | 5/28 ≈ 0.18 | Jaccard 5/35 | frozen |
| 5.6 Extraction | R-560-02 | ML consensus 13 names as listed in YAML | `results.ml_consensus_names` | Part 3 | §2 | attribution | catalogue membership | Part 4 feature set | frozen |
| 5.7 Thinking-high | R-570-01 | PR-AUC 0.8553 [0.7957, 0.9131]; fold mean 0.8488±0.0861; folds 0.8640…0.9061; wins vs LGB 5/5 | `tabpfn_thinking_high.metrics.pr_auc` | Part 4 | Table 1; S-folds | prediction | thinking-high nested-CV ranking | TabPFN (unspecified) | frozen |
| 5.7 Thinking-high | R-570-02 | ROC-AUC 0.9905 [0.9834, 0.9964]; Brier 0.0064 [0.0052, 0.0077] best of seven | `...roc_auc`; `...brier` | Part 4 | Table 1; S-CI | prediction | name the arm | TabPFN poorly/best calibrated without arm | frozen |
| 5.7 Thinking-high | R-570-03 | Nested t 0.271±0.067; PPV 0.7927; recall 0.7065; spec 0.9967; F1 0.7471; F2 0.7222; 5076/17/27/65 | `...nested_operating_point_table2` | Part 4 | Table 2 | prediction | nested operating point | pooled 0.8152 / t=0.193 | frozen |
| 5.7 Thinking-high | R-570-04 | Δ vs LGB 0.1611 (0.0984–0.2289) P=0/2000 | `...delta_pr_auc_vs_lightgbm` | Part 4 | Table S-Δ | prediction | paired OOF bootstrap | generalises to new cohorts | frozen |
| 5.8 TabPFN local | R-580-01 | PR-AUC 0.6742 [0.5864, 0.7657]; folds 0.6384…0.7855; wins vs LGB 2/5 | `tabpfn_local.metrics.pr_auc` | Part 4 | Table 1; S-folds | prediction | local nested-CV ranking | thinking-high numbers | frozen |
| 5.8 TabPFN local | R-580-02 | ROC-AUC 0.9845 [0.9760, 0.9917] second; Brier 0.0102 [0.0092, 0.0113] booster band | `...roc_auc`; `...brier` | Part 4 | Table 1; S-CI | prediction | booster band; not worst Brier | local worst Brier; 0.0673 | frozen |
| 5.8 TabPFN local | R-580-03 | Nested t 0.166±0.020; PPV 0.5478; recall 0.6848; 5041/52/29/63 | `...nested_operating_point_table2` | Part 4 | Table 2 | prediction | nested | t=0.915; pooled recall 0.8478 | frozen |
| 5.8 TabPFN local | R-580-04 | Δ vs LGB −0.0201 (−0.0974–0.0566) | `...delta_pr_auc_vs_lightgbm` | Part 4 | Table S-Δ | prediction | compatible with no difference | local significantly worse | frozen |
| 5.8 TabPFN local | R-580-05 | Historical excluded: PR 0.6754; Brier 0.0673; t 0.915 | `tabpfn_local.metrics.historical_excluded` | freeze_manifest §4 | excluded dumps | prediction | do not quote | Version 4 local Brier 0.0673 | excluded |
| 5.9 Interpretation | R-590-01 | Held-out mean \|SHAP\| eGFR 1.0439, WBC 1.0202, LV 0.8695 | `interpretability.shap.top_mean_abs_heldout` | Part 5 | Table 5 / Table 4 | attribution | held-out SHAP (local after 429) | Cre-leading 0.158; 15+15 | frozen |
| 5.9 Interpretation | R-590-02 | Consensus 3/3 WBC, LV, eGFR | `interpretability.feature_rankings.part5_consensus_3_of_3` | Part 5 | Table 5 | attribution | most consistent attributions this run | Part 4 feature set; validated markers | frozen |
| 5.9 Interpretation | R-590-03 | Train MI top CaI 0.022005; Cre train MI 0.000000 | `part5_mi_top1_train`; `cre_train_mi` | Part 5 | Table 1 | attribution | train-split MI | full-cohort MI | frozen |
| 5.9 Interpretation | R-590-04 | Stability WBC 10/10; Cre and LV 8/10; eGFR 7/10 | `interpretability.ffs.part5_stability_sfs` | Part 5 | Table 2 | attribution | train resamples | nested-CV stability | frozen |
| 5.9 Interpretation | R-590-05 | Binary PDP ΔP Previous PCI +0.0137; post-dilation −0.0093 | `feature_rankings.pdp_binary` | Part 5 | Table 3 | attribution | train empirical prior; not treatment effect | Part 4 risk; protective post-dilation | frozen |
| 5.9 Interpretation | R-590-06 | k-SII row 5176 | `interpretability.shapiq.k_sii_row_cohort_index` | Part 5 | §5–6 | attribution | one held-out VLST=1 row | cohort interactions; rows 5099/5093 | frozen |
| 5.9 Interpretation | R-590-07 | Cre held-out mean \|SHAP\| 0.2449 (rank 7 of 81) | `interpretability.feature_rankings.cre_heldout_mean_abs_shap`; `shap.part5.top15_mean_abs_heldout` | Part 5 | Table 4 | attribution | held-out SHAP rank 7 | Cre \|SHAP\| 0.158 | frozen |
| 5.10 Wang | R-5100-01 | Frozen score ROC-AUC 0.8013; PR-AUC 0.1032 (prose only; no table) | `this_pack_frozen_integer_score.roc_auc_full_cohort`; `pr_auc_full_cohort` | Part 4 | §7 Headline | historical_comparator | full-cohort integer score, not nested-CV; no exhibit | ML external validation | frozen |
| 5.10 Wang | R-5100-02 | Fold-mean ROC 0.8005±0.0607; PR 0.1134±0.0518 | `fold_mean_roc`; `fold_mean_pr` | Part 4 | Table S-Wang | historical_comparator | same folds, score not refit | eighth nested-CV arm | frozen |
| 5.10 Wang | R-5100-03 | Bins: low 3135/16/0.0051; int 1577/35/0.0222 (Wang printed n 1837); high 473/41/0.0867 | `reported_performance.risk_bins` | Part 4 | Table S-Wang-bins | historical_comparator | this file’s n; note Wang 1837 | intermediate n=1837 on this CSV | frozen |
| 5.10 Wang | R-5100-04 | Nested-CV thinking-high/LGB/local PR-AUC vs frozen 0.1032 is derivation-cohort ranking only | S-Wang table in YAML via model_results + wang_2020 | Part 4 | Table S-Wang | historical_comparator | ranking comparison on this file | ML beat Wang on external data | frozen |
| 6.1 Principal findings | D-610-01 | Restate A-RE / R-570 / R-580 / R-540 / R-590 without new numbers; TSSI only via 4.4 | same paths | freeze_manifest.md | §2 Frozen values | terminology | as allowed | TabPFN validated; arms collapsed | complete_if_worded_as_allowed |
| 6.2 Predictive performance | D-620-01 | PR-AUC more informative than ROC-AUC at 1.77% prevalence | `metrics.pr_auc`; `study.event_rate` | Part 4 | Figure 1 caption | prediction | ranking at low prevalence | excellent discrimination from ROC alone | frozen |
| 6.2 Predictive performance | D-620-02 | Nested vs pooled F1 optimistic bias | `metrics.f1` | Part 4 | §5 | prediction | quote nested | pooled as honest | frozen |
| 6.2 Predictive performance | D-620-03 | Unmatched tuning: defaults vs thinking-high vs local | front matter limitation 8 (methods_only) | 00_front_matter.md | Limitation 8 | limitation | unmatched effort | fair hyperparameter-matched contest | methods_only |
| 6.3 TSSI discussion | D-630-01 | Binary TSSI ≠ Cox time axis; leakage demo with SMOTE caveat | `tssi_status`; `tssi_leakage_analysis` | Part 4 §6; limitation 2 | — | leakage | time-at-risk | legitimate covariate | frozen |
| 6.4 Prior work | D-640-01 | “No VLST score exists” is false | unsupported_claims.md §4 | 00_front_matter.md | A score already exists | historical_comparator | Wang score exists | empty field | complete_if_worded_as_allowed |
| 6.4 Prior work | D-640-02 | WBC discrepancy vs Wang exclusion for infection | 00_front_matter.md limitation 7 | 00_front_matter.md | Limitation 7 | limitation | discrepancy to report | validated inflammatory marker | methods_only |
| 6.4 Prior work | D-640-03 | Additional literature | — | — | — | todo | later sourced citations | invented papers/numbers | unresolved |
| 6.5 Strengths | D-650-01 | TSSI control; nested honest threshold; two TabPFN arms; Table 4b; frozen Wang; OOF bootstrap | protocol YAML fields | freeze_manifest.md | §3 Methods-only / §2 | methods | only file-supported strengths | external validation as a strength | complete_if_worded_as_allowed |
| 6.6 Limitations | D-660-01 | No ML external/temporal test | `validation.external_validation.ml_models` | 00_front_matter.md | Limitation 1 | limitation | every Part 4 number is nested CV on 5185 rows | nested CV substitutes for EV | frozen |
| 6.6 Limitations | D-660-02 | Binary vs Cox; EPV; two TabPFN objects; unrecorded versions; DAPT timing; unnamed LV/CaI; Part 5≠Part 4; encoder before split; SMOTE mismatch; single nested CV | corresponding YAML / unresolved_items | 00_front_matter.md | Limitations 2–11; conflict M1 M3 | limitation | list as limitations | omit EV gap | methods_only / unresolved as listed |
| 6.6 Limitations | D-660-03 | Historical thinking-high Brier 0.0060/0.0360 are other dumps | freeze_manifest excluded | 00_front_matter.md | Limitation 4 | limitation | client non-determinism across dumps | this run Brier 0.0060 | excluded as this-run numbers |
| 6.7 Future validation | D-670-01 | Held-out/external cohort this pack does not contain is required for transportability claims | `validation.external_validation.ml_models` | 00_front_matter.md; Part 3 practical reading | Limitation 1 | limitation | future work; not done | we externally validated | frozen |
| 7 Conclusion | C-01 | On this derivation cohort, thinking-high highest nested-CV PR-AUC; local not above LightGBM on PR-AUC; TSSI excluded; Table 4b associations; Wang historical comparator; no ML EV | headline YAML paths | freeze_manifest.md | §8 | terminology | derivation-cohort predictive performance | clinically useful; validated | complete_if_worded_as_allowed |

---

## Claims that must never appear (unsupported → do not assign a Results ID)

| Ban ID | Unsupported statement | Source of ban |
| --- | --- | --- |
| BAN-01 | Nested CV / Part 5 1556 rows / Wang Shantou = external validation of this pack’s ML | unsupported_claims.md §1, §6 |
| BAN-02 | Clinically useful / clinically validated / net benefit | unsupported_claims.md §1 |
| BAN-03 | Independent risk factor / independent predictor | unsupported_claims.md §1 |
| BAN-04 | Protective effect of post-dilation or clopidogrel | unsupported_claims.md §1 |
| BAN-05 | Personalised risk | unsupported_claims.md §1 |
| BAN-06 | Validated inflammatory marker (WBC) | unsupported_claims.md §1 |
| BAN-07 | Local TabPFN has the worst Brier | unsupported_claims.md §4 |
| BAN-08 | Nested recall 0.8152 or 0.8478 | unsupported_claims.md §4 |
| BAN-09 | Part 5 consensus as Part 4 feature set | unsupported_claims.md §4 |
| BAN-10 | k-SII as cohort interactions | unsupported_claims.md §4 |
| BAN-11 | Table 4 as the clinical multivariable model | unsupported_claims.md §4 |
| BAN-12 | TSSI 70/30 as nested-CV performance | unsupported_claims.md §4 |
| BAN-13 | Integer Wang score as nested-CV eighth arm | unsupported_claims.md §4 |
| BAN-14 | Intermediate-bin n=1837 on this CSV | unsupported_claims.md §4 |
| BAN-15 | No VLST score exists | unsupported_claims.md §4 |
| BAN-16 | PDP as Part 4 nested-CV risk | unsupported_claims.md §4 |
| BAN-17 | SHAP 15+15; k-SII 5099/5093; Cre \|SHAP\| 0.158 | unsupported_claims.md §4 |
| BAN-18 | Jaccard 5/35 | unsupported_claims.md §4 |
| BAN-19 | Encoder n_raw=99 | conflict_ledger C1 |
| BAN-20 | Local Brier 0.0673 as Version 4 | conflict_ledger M6; freeze excluded |

---

## Audit note on likelihood-ratio tests

Table 4/4b intervals remain **Wald**. Exploratory **interaction** LR tests are frozen from Part 1 Supplementary Table S2 (claims M-460-06 / R-540-04). Do not invent additional LR tests.
