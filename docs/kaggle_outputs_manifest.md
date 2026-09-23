# Kaggle output ingest manifest

Generated 2026-09-22. Folders were **not** at repo root; they sit under `code/modeling/{rating,interpretability}/`. Names below are the owner folder names.

Pin (interp notebooks + Part 4 protocol): `tabpfn==9.0.0` / `tabpfn-client==0.6.0`, local+thinking **v3.5**.

| Folder (owner name) | On-disk path | Source notebook | Run date (papermill) | freeze YAML key | n_files |
| --- | --- | --- | --- | --- | ---: |
| `Kaggle_baseline_plus_tabpfn_results/` | `code/modeling/rating/Kaggle_baseline_plus_tabpfn_results/` | `code/modeling/rating/baseline_plus_tabpfn.ipynb` | 9-arm dump `baseline_plus_tabpfn_results/` (run_manifest Tesla T4; tabpfn 9.0.0 / client 0.6.0) | `kaggle_baseline_plus_tabpfn` | 24 |
| `Kaggle_baseline_tssi_leakage_results/` | `code/modeling/rating/Kaggle_baseline_tssi_leakage_results/` | `code/modeling/rating/baseline_tssi_leakage.ipynb` | papermill 2026-09-19T14:00Z–2026-09-19T14:57Z | `kaggle_tssi_leakage` | 19 |
| `Kaggle_baseline_without_tssi_results/` | `code/modeling/rating/Kaggle_baseline_without_tssi_results/` | `code/modeling/rating/baseline_without_tssi.ipynb` | papermill 2026-09-19T14:02Z–2026-09-19T14:22Z | `kaggle_without_tssi` | 19 |
| `Kaggle_baseline_intrepretability_results/` | `code/modeling/interpretability/Kaggle_baseline_intrepretability_results/` | `code/modeling/interpretability/baseline_feature_selections.ipynb` | `selector_report.md` generated 2026-09-19 14:39:53 | `kaggle_feature_selectors` | 37 |
| `Kaggle_tabpfn_intrepretebility_results/` | `code/modeling/interpretability/Kaggle_tabpfn_intrepretebility_results/` | `code/modeling/interpretability/tabpfn_interpretability_fs_pdp.ipynb + tabpfn_interpretability_shap.ipynb (split of tabpfn_interpretability.ipynb)` | fs_pdp papermill 2026-09-20T10:58Z–2026-09-20T21:28Z; shap 2026-09-20T10:05Z–2026-09-20T11:50Z | `kaggle_interpretability` | 36 |

## File lists

### `Kaggle_baseline_plus_tabpfn_results/`

- **Source notebook:** `code/modeling/rating/baseline_plus_tabpfn.ipynb`
- **Run:** 9 nested arms; anti-leakage ON; `run_manifest.json` Tesla T4; `tabpfn==9.0.0` / `tabpfn-client==0.6.0`
- **YAML key:** `kaggle_baseline_plus_tabpfn`
- **Inner folder:** `baseline_plus_tabpfn_results/` (not the unlabeled two-arm `baseline_plus_tabpfn/` dump)
- **Notes:** Nested 5×4 CV; TSSI+WBC dropped; thinking v3.5 PR-AUC 0.9212; TabPFN v3.5 0.8957.

| path | type |
| --- | --- |
| `baseline_plus_tabpfn_results/modeling_results/run_manifest.json` | json |
| `baseline_plus_tabpfn_results/modeling_results/oof/oof_predictions.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/oof/fold_thresholds.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/tables/model_comparison.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/tables/nested_cv_operating_point.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/tables/calibration_ece.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/tables/confusion_counts.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/tables/fold_metrics.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/tables/leakage_precision_probe.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/tables/precision_equalised_sensitivity.csv` | table |
| `baseline_plus_tabpfn_results/modeling_results/figures/pr_roc_curves.png` | figure |
| `baseline_plus_tabpfn_results/modeling_results/figures/calibration_curves.png` | figure |
| `baseline_plus_tabpfn_results/modeling_results/figures/confusion_matrices.png` | figure |
| `baseline_plus_tabpfn_results/modeling_results/figures/best_model_threshold_fpfn_panel.png` | figure |

### `Kaggle_baseline_tssi_leakage_results/`

- **Source notebook:** `code/modeling/rating/baseline_tssi_leakage.ipynb`
- **Run date:** papermill 2026-09-19T14:00Z–2026-09-19T14:57Z
- **YAML key:** `kaggle_tssi_leakage`
- **Notes:** 70/30 GridSearch WITH TSSI; inner folder spelling baseline_leakge_results.

| path | bytes | type |
| --- | ---: | --- |
| `baseline_leakge_results/__results___files/__results___18_0.png` | 342331 | figure |
| `baseline_leakge_results/__results___files/__results___35_0.png` | 63140 | figure |
| `baseline_leakge_results/__results___files/__results___37_0.png` | 72655 | figure |
| `baseline_leakge_results/catboost_info/catboost_training.json` | 22069 | json |
| `baseline_leakge_results/catboost_info/learn/events.out.tfevents` | 20940 | tensorboard |
| `baseline_leakge_results/catboost_info/learn_error.tsv` | 5582 | log |
| `baseline_leakge_results/catboost_info/time_left.tsv` | 2406 | log |
| `baseline_leakge_results/modeling_tssi_leakage/best_catboost.joblib` | 83929 | fitted-model |
| `baseline_leakge_results/modeling_tssi_leakage/best_decision_tree.joblib` | 6201 | fitted-model |
| `baseline_leakge_results/modeling_tssi_leakage/best_gaussian_nb.joblib` | 3671 | fitted-model |
| `baseline_leakge_results/modeling_tssi_leakage/best_lightgbm.joblib` | 717460 | fitted-model |
| `baseline_leakge_results/modeling_tssi_leakage/best_logistic.joblib` | 1599 | fitted-model |
| `baseline_leakge_results/modeling_tssi_leakage/best_random_forest.joblib` | 15300809 | fitted-model |
| `baseline_leakge_results/modeling_tssi_leakage/best_xgboost.joblib` | 234887 | fitted-model |
| `baseline_leakge_results/modeling_tssi_leakage/confusion_matrices.png` | 123518 | figure |
| `baseline_leakge_results/modeling_tssi_leakage/decision_tree_plot.png` | 631632 | figure |
| `baseline_leakge_results/modeling_tssi_leakage/roc_pr_curves.png` | 100024 | figure |
| `baseline_leakge_results/modeling_tssi_leakage/test_metrics.csv` | 857 | table |
| `baseline_leakge_results/modeling_tssi_leakage/threshold_analysis.csv` | 1117 | table |

### `Kaggle_baseline_without_tssi_results/`

- **Source notebook:** `code/modeling/rating/baseline_without_tssi.ipynb`
- **Run date:** papermill 2026-09-19T14:02Z–2026-09-19T14:22Z
- **YAML key:** `kaggle_without_tssi`
- **Notes:** 70/30 GridSearch WITHOUT TSSI.

| path | bytes | type |
| --- | ---: | --- |
| `baseline_without_leakage/__results___files/__results___21_0.png` | 432912 | figure |
| `baseline_without_leakage/__results___files/__results___38_0.png` | 85833 | figure |
| `baseline_without_leakage/__results___files/__results___40_0.png` | 79369 | figure |
| `baseline_without_leakage/catboost_info/catboost_training.json` | 22487 | json |
| `baseline_without_leakage/catboost_info/learn/events.out.tfevents` | 20940 | tensorboard |
| `baseline_without_leakage/catboost_info/learn_error.tsv` | 5863 | log |
| `baseline_without_leakage/catboost_info/time_left.tsv` | 2280 | log |
| `baseline_without_leakage/modeling_without_tssi/best_catboost.joblib` | 83139 | fitted-model |
| `baseline_without_leakage/modeling_without_tssi/best_decision_tree.joblib` | 9081 | fitted-model |
| `baseline_without_leakage/modeling_without_tssi/best_gaussian_nb.joblib` | 3607 | fitted-model |
| `baseline_without_leakage/modeling_without_tssi/best_lightgbm.joblib` | 651764 | fitted-model |
| `baseline_without_leakage/modeling_without_tssi/best_logistic.joblib` | 1595 | fitted-model |
| `baseline_without_leakage/modeling_without_tssi/best_random_forest.joblib` | 3906889 | fitted-model |
| `baseline_without_leakage/modeling_without_tssi/best_xgboost.joblib` | 417670 | fitted-model |
| `baseline_without_leakage/modeling_without_tssi/confusion_matrices.png` | 138358 | figure |
| `baseline_without_leakage/modeling_without_tssi/decision_tree_plot.png` | 795168 | figure |
| `baseline_without_leakage/modeling_without_tssi/roc_pr_curves.png` | 134150 | figure |
| `baseline_without_leakage/modeling_without_tssi/test_metrics.csv` | 851 | table |
| `baseline_without_leakage/modeling_without_tssi/threshold_analysis.csv` | 1033 | table |

### `Kaggle_baseline_intrepretability_results/`

- **Source notebook:** `code/modeling/interpretability/baseline_feature_selections.ipynb`
- **Run date:** `selector_report.md` generated 2026-09-19 14:39:53
- **YAML key:** `kaggle_feature_selectors`
- **Notes:** Owner spelling intrepretability. Inner dir `baseline_interpretability_results/model_feature_selectors_antileak/`. TSSI+WBC dropped; scaled 87 columns.

| path | bytes | type |
| --- | ---: | --- |
| `baseline_interpretability_results/__results___files/__results___10_0.png` | 33876 | figure |
| `baseline_interpretability_results/__results___files/__results___10_1.png` | 58190 | figure |
| `baseline_interpretability_results/__results___files/__results___10_2.png` | 23516 | figure |
| `baseline_interpretability_results/model_feature_selectors_antileak/ffs_cat_pr_auc.csv` | 639 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/ffs_lgb_pr_auc.csv` | 337 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/ffs_lr_pr_auc.csv` | 900 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/ffs_rf_b_pr_auc.csv` | 474 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/ffs_rf_pr_auc.csv` | 596 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/ffs_xgb_b_pr_auc.csv` | 950 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/ffs_xgb_pr_auc.csv` | 823 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/loco_cat_pr_auc.csv` | 5619 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/loco_lgb_pr_auc.csv` | 5525 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/loco_lr_pr_auc.csv` | 5676 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/loco_rf_b_pr_auc.csv` | 5617 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/loco_rf_pr_auc.csv` | 5452 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/loco_xgb_b_pr_auc.csv` | 5651 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/loco_xgb_pr_auc.csv` | 5653 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_all_unique_features.csv` | 1103 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_common_by_model_algorithms.csv` | 274 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_common_features_by_algorithm.csv` | 107 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_common_global_all_models_algorithms.csv` | 53 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_model_algorithm_counts.png` | 55480 | figure |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_overlap_heatmap.png` | 37764 | figure |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_priority_feature_ranks.csv` | 8115 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_report.md` | 9681 | report |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_summary_long.csv` | 23005 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_top_repeated_features.png` | 88125 | figure |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_union_all_models_algorithms.csv` | 1247 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/selector_union_by_model.csv` | 2803 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/shap_cat_pr_auc.csv` | 3017 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/shap_lgb_pr_auc.csv` | 3030 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/shap_lr_pr_auc.csv` | 3012 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/shap_rf_b_pr_auc.csv` | 3055 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/shap_rf_pr_auc.csv` | 2964 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/shap_xgb_b_pr_auc.csv` | 3113 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/shap_xgb_pr_auc.csv` | 3085 | table |
| `baseline_interpretability_results/model_feature_selectors_antileak/split_manifest.json` | 488 | json |


### `Kaggle_tabpfn_intrepretebility_results/`

- **Source notebook:** `code/modeling/interpretability/tabpfn_interpretability_fs_pdp.ipynb + tabpfn_interpretability_shap.ipynb (split of tabpfn_interpretability.ipynb)`
- **Run date:** fs_pdp papermill 2026-09-20T10:58Z–2026-09-20T21:28Z; shap 2026-09-20T10:05Z–2026-09-20T11:50Z
- **YAML key:** `kaggle_interpretability`
- **Notes:** Owner spelling intrepretebility. Subdirs fs_pdp_MI/ and shap/.

| path | bytes | type |
| --- | ---: | --- |
| `fs_pdp_MI/__results___files/__results___11_1.png` | 166606 | figure |
| `fs_pdp_MI/__results___files/__results___11_3.png` | 65594 | figure |
| `fs_pdp_MI/__results___files/__results___13_1.png` | 56707 | figure |
| `fs_pdp_MI/modeling_tabpfn/interpretability_feature_importance_report.csv` | 5708 | table |
| `fs_pdp_MI/modeling_tabpfn/interpretability_feature_importance_report.png` | 93476 | figure |
| `fs_pdp_MI/modeling_tabpfn/interpretability_feature_stability.csv` | 344 | table |
| `fs_pdp_MI/modeling_tabpfn/interpretability_feature_stability_summary.csv` | 223 | table |
| `fs_pdp_MI/modeling_tabpfn/interpretability_heldout_indices.csv` | 18358 | table |
| `fs_pdp_MI/modeling_tabpfn/interpretability_mutual_info_ranking.csv` | 2461 | table |
| `fs_pdp_MI/modeling_tabpfn/interpretability_pdp.png` | 387069 | figure |
| `fs_pdp_MI/modeling_tabpfn/interpretability_pdp_binary.csv` | 850 | table |
| `fs_pdp_MI/modeling_tabpfn/interpretability_pdp_binary.png` | 141242 | figure |
| `fs_pdp_MI/modeling_tabpfn/interpretability_train_indices.csv` | 46423 | table |
| `shap/__results___files/__results___11_3.png` | 37242 | figure |
| `shap/__results___files/__results___11_5.png` | 197434 | figure |
| `shap/__results___files/__results___11_7.png` | 109532 | figure |
| `shap/__results___files/__results___9_11.png` | 43639 | figure |
| `shap/__results___files/__results___9_13.png` | 105028 | figure |
| `shap/__results___files/__results___9_15.png` | 61451 | figure |
| `shap/__results___files/__results___9_17.png` | 156134 | figure |
| `shap/__results___files/__results___9_19.png` | 107922 | figure |
| `shap/__results___files/__results___9_7.png` | 138502 | figure |
| `shap/__results___files/__results___9_9.png` | 42275 | figure |
| `shap/modeling_tabpfn/interpretability_heldout_indices.csv` | 18358 | table |
| `shap/modeling_tabpfn/interpretability_shap_explain_indices.csv` | 28156 | table |
| `shap/modeling_tabpfn/interpretability_shap_mean_abs.csv` | 2819 | table |
| `shap/modeling_tabpfn/interpretability_train_indices.csv` | 46423 | table |
| `shap/modeling_tabpfn/k_ssi_interpretability_network_top15.png` | 366918 | figure |
| `shap/modeling_tabpfn/k_ssi_interpretability_upset.png` | 257281 | figure |
| `shap/modeling_tabpfn/k_ssi_shapiq_network_top15.png` | 469517 | figure |
| `shap/modeling_tabpfn/k_ssi_shapiq_upset.png` | 257450 | figure |
| `shap/modeling_tabpfn/sv_interpretability_shap_bar.png` | 97731 | figure |
| `shap/modeling_tabpfn/sv_interpretability_shap_beeswarm.png` | 252846 | figure |
| `shap/modeling_tabpfn/sv_interpretability_shap_scatter_f0.png` | 100833 | figure |
| `shap/modeling_tabpfn/sv_interpretability_shap_summary.png` | 329737 | figure |
| `shap/modeling_tabpfn/sv_interpretability_shap_waterfall_row0.png` | 130918 | figure |

## Leakage-contrast pair (item 5)

Both TSSI arms contain the **same artifact names** for the 7-model hold-out:

| Artifact | with-TSSI | without-TSSI |
| --- | --- | --- |
| `test_metrics.csv` (7 models × acc/f1/recall/precision/roc_auc/pr_auc) | present | present |
| `threshold_analysis.csv` (precision/recall/f1 vs threshold; **model unnamed**) | present | present |
| `confusion_matrices.png` | present | present |
| `roc_pr_curves.png` | present | present |
| `decision_tree_plot.png` | present | present |
| 7× `best_*.joblib` | present | present |
| `catboost_info/*` | present | present |
| `best_params.csv` | present (from notebook prints) | present (from notebook prints) |

GridSearch `best_params_` were always in the executed notebooks. Dump CSVs were added from those prints (2026-09-22). `threshold_analysis.csv` is not labelled by model — same gap on both arms.

## Cross-check vs paper Markdown (item 4)

Canonical report paths (requested `-(2)` / `-(3)` names not on disk): `paper_results/paper_results.md`, `docs/paper_evidence_map.md`, and the five `paper_results/0{1-5}_*/` `*_paper_figures_and_tables.md` / `feature_extraction_comparison.md`. Part 2/3 Markdown was **rewritten 2026-09-22** from the selector dump.

| Cited number (paper / evidence map) | Ingested dump | Verdict |
| --- | --- | --- |
| Part 4 thinking v3.5 PR-AUC 0.9212 / nested 5083/10/15/77 | `Kaggle_baseline_plus_tabpfn_results/baseline_plus_tabpfn_results` | **MATCH** (anti-leakage ON; freeze `nested_cv_v35_antileakage_on`) |
| Part 4 TabPFN v3.5 PR 0.8957 / nested 5082/11/20/72 | same | **MATCH** |
| Unlabeled thinking 0.9771 / local 0.9635 | older inner folder `baseline_plus_tabpfn/` | **EXCLUDED** |
| Table S-TSSI LR PR-AUC 0.9134 → 0.3431; Cat 0.9599 → 0.4942 | both TSSI `test_metrics.csv` | **MATCH** (rebuilt 2026-09-21; older 0.9575→0.5077 excluded) |
| Part 2 ML consensus n=10; Jaccard vs FDR 5/25; 87 columns; WBC dropped | `Kaggle_baseline_intrepretability_results/` | **MATCH** (reports rebuilt 2026-09-22; older 13-name / 5/28 / 88-col reconstruction excluded) |
| Part 5 Table 2 CaI/LV/eGFR 8/8 | `interpretability_feature_stability.csv` | **MATCH** |
| Part 5 Table 1 CaI MI 0.020536 | `interpretability_mutual_info_ranking.csv` | **MATCH** |
| Part 5 Table 4 eGFR 1.2288 / CaI 1.0867 | `interpretability_shap_mean_abs.csv` | **MATCH** |
| Part 5 PDP Previous PCI +0.001294 | `interpretability_pdp_binary.csv` | **MATCH** |
| Part 5 Table 5 3/3 `{CaI, eGFR, LV}` | Borda merge of MI+SFS+SHAP dumps | **MATCH** |
| Part 5 k-SII row 5176; split 3629/64 vs 1556/28 | index CSVs | **MATCH** |

## SFS 8/8 (item 3)

`fs_pdp_MI/modeling_tabpfn/interpretability_feature_stability_summary.csv`: times_selected=8 → `CaI; LV; eGFR` (n_features=3). `STABILITY_N_SEEDS=8` in `tabpfn_interpretability_fs_pdp.ipynb`. Promoted in `paper/frozen_results.yaml` `interpretability.ffs.part5_stability_sfs` and `kaggle_interpretability.sfs_stability`.

