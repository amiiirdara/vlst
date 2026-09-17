# File inventory

Inspection-only freeze audit. No notebooks were re-run. No analytical code was modified. Internet was not used. `frozen_results.yaml` was not created.

**Scope rule.** Only the filenames listed in the audit request were inspected. Canonical report copies live under `paper_results/`. Dual-tree `.md` copies under `code/**` with the same basenames were **not** present on disk at audit time. `docs/paper_evidence_map.md` is an audit ledger, not a manuscript source. Root `README.md` is a project slogan, not a results claim.

**Notebook cell numbers** are 0-based JSON indices in the `.ipynb` files (the same numbering used by `nbformat`). `execution_count` is noted only when it differs from that index in a way that matters.

**Legend.** Source type: report / notebook / Python script. Item class: numerical result / methodological detail / interpretation / provenance detail.

---

## 1. Reports

| File | Absolute path | Source type | Role | Cells / headings used |
| --- | --- | --- | --- | --- |
| `00_front_matter.md` | `/home/fadia/Documents/vlst/paper_results/00_front_matter.md` | report | Terminology (W5), clinical motivation (W2), EPV (W4), limitations (W3). Not a figure pack. | Headings: Terminology; Clinical motivation; Events per variable; Limitations; Sources |
| `EDA_paper_figures_and_tables.md` | `/home/fadia/Documents/vlst/paper_results/01_eda/EDA_paper_figures_and_tables.md` | report | Part 1 association catalogue; clinical Table C; FDR screens; Table 4 / 4b; stent encoder Figure 5 | §§0–8; Tables C, R, 1–4, 4b, 3, S4; Figures 1–6, S1–S4 |
| `baseline_feature_selections_paper_figures_and_tables.md` | `/home/fadia/Documents/vlst/paper_results/02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md` | report | Part 2 classic-model LOCO / SHAP / FFS catalogues (not prediction) | §§1–8; Tables 0–5; Figures 1–7, S1–S2 |
| `feature_extraction_comparison.md` | `/home/fadia/Documents/vlst/paper_results/03_stats_vs_ml/feature_extraction_comparison.md` | report | Part 3 stats-vs-ML overlap (methods comparison) | §§1–8; Jaccard 5/28; Figures 1–2; Table 1 |
| `baseline_plus_tabpfn_paper_figures_and_tables.md` | `/home/fadia/Documents/vlst/paper_results/04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md` | report | Part 4 nested-CV **prediction** results; TSSI supplement; Wang frozen score | §§1–8; Tables 0–3, S-CI, S-Δ, S-folds, S-TSSI, S-Wang, S-Wang-bins; Figures 1–3, S-TSSI |
| `tabpfn_interpretability_paper_figures_and_tables.md` | `/home/fadia/Documents/vlst/paper_results/05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md` | report | Part 5 TabPFN **attribution** (Version 5, 70/30) | §§1–8; Tables 0–5; Figures 1–13 |
| `paper_results.md` | `/home/fadia/Documents/vlst/paper_results/paper_results.md` | report | Concatenation of `00_front_matter.md` + Parts 1–5 | Same headings as the part files; **Part 5 footer/captions are stale relative to the standalone Part 5 file** (see conflict ledger) |
| `paper_evidence_map.md` | `/home/fadia/Documents/vlst/docs/paper_evidence_map.md` | report (audit ledger) | Historical Revisions 2–13; provenance and open items | Header Revisions 2–10+; do not quote historical revision numbers as live results |
| `README.md` (pack) | `/home/fadia/Documents/vlst/paper_results/README.md` | report | How to open the portable pack | Layout tree only |
| `README.md` (repo) | `/home/fadia/Documents/vlst/README.md` | report | One-line project title | Line 2 slogan only |

---

## 2. Notebooks

| File | Absolute path | Source type | n cells | Role | Key cells |
| --- | --- | --- | --- | --- | --- |
| `eda.ipynb` | `/home/fadia/Documents/vlst/code/analyzes/eda.ipynb` | notebook | 81 (indices 0–80) | Univariate FDR, multivariable logits, stent encoder, domain screens. Last cells still call `run_b4` / `run_b7` hygiene helpers | Cell 4 load / missingness / TSSI handling (source); cells 79–80 hygiene (`run_b7` / `run_b4`) |
| `stats_vs_ml_comparison.ipynb` | `/home/fadia/Documents/vlst/code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb` | notebook | 5 (0–4) | Hard-coded FDR-20 and ML-13 catalogues; prints Jaccard | Cell 0 markdown protocol; cells 2–4 catalogue + Jaccard print |
| `baseline_feature_selections.ipynb` | `/home/fadia/Documents/vlst/code/modeling/interpretability/baseline_feature_selections.ipynb` | notebook | 12 | Part 2 selectors; `INNER_VAL_SIZE=0.2`; TSSI dropped | Cell 1 constants; cell 2 drop; cell 3 split; cells 4–9 selectors |
| `tabpfn_interpretability.ipynb` | `/home/fadia/Documents/vlst/code/modeling/interpretability/tabpfn_interpretability.ipynb` | notebook | 17 | Part 5 Version 5 (`e356bb1`) | Cell 4 env flags; cell 6 load/split/encoder; cell 8 MI/SFS; cell 10 PDP; cell 12 SHAP; cell 14 SHAP-IQ; cell 16 consensus |
| `baseline_plus_tabpfn.ipynb` | `/home/fadia/Documents/vlst/code/modeling/rating/baseline_plus_tabpfn.ipynb` | notebook | 18 | Part 4 nested 5×4 CV, Version 4 (`139d143`) | Cell 5 models + `OUTER_SPLITS=5` / `INNER_SPLITS=4` / TSSI drop; cells 7, 11, 15 evaluation / nested operating point |
| `baseline_tssi_leakage.ipynb` | `/home/fadia/Documents/vlst/code/modeling/rating/baseline_tssi_leakage.ipynb` | notebook | 44 | 70/30 GridSearch **with** TSSI (leakage demonstration) | Cell 0 overview; cell 2 load `data/processed/` npy; cell 4 `USE_SMOTE=True` |
| `baseline_without_tssi.ipynb` | `/home/fadia/Documents/vlst/code/modeling/rating/baseline_without_tssi.ipynb` | notebook | 46 | Same 70/30 family **after dropping TSSI** | Cell 0 overview; cell 3–4 drop TSSI; cell 6 `USE_SMOTE=False` |
| `wang_vlst_score.ipynb` | `/home/fadia/Documents/vlst/code/modeling/rating/wang_vlst_score.ipynb` | notebook | 6 | Frozen Wang 2020 integer points on `VLST.csv` | Cell 0 markdown; cell 2 formula + ROC/PR; cell 3 risk bins |

`preprocessing.ipynb` is **cited** by the TSSI notebooks but was **not** in the inspect list; its scaler / imputer / encoding steps are therefore **not** inventoried here.

---

## 3. Python script

| File | Absolute path | Source type | Role |
| --- | --- | --- | --- |
| `rebuild_tssi_leakage_table.py` | `/home/fadia/Documents/vlst/code/modeling/rating/rebuild_tssi_leakage_table.py` | Python script | Hard-coded `ROWS` for Supplementary Table S-TSSI; writes `paper_table_s_tssi_leakage.csv` / PNG. Does not refit models. |

Related hygiene (cited by reports, **not** in the inspect list): `code/modeling/tools/paper_hygiene_b3_b4_b7.py` (`run_b3` bootstrap CIs; `run_b4` Table 4b; `run_b7` Table C).

---

## 4. What each file is allowed to support

| Claim class | Authoritative file(s) | Must not treat as |
| --- | --- | --- |
| Nested-CV prediction metrics | `baseline_plus_tabpfn.ipynb` (D4) and Part 4 report | Part 5 70/30 SHAP; TSSI 70/30 GridSearch; Wang score |
| Association (univariate / logit) | `eda.ipynb` and Part 1 report; quote Table 4b not Table 4 | Part 4 PR-AUC |
| Selector catalogues | `baseline_feature_selections.ipynb` and Part 2 report | A feature mask for Part 4 |
| Stats vs ML overlap | `stats_vs_ml_comparison.ipynb` cell 4 print and Part 3 report | Biological ranking |
| TabPFN attribution | `tabpfn_interpretability.ipynb` and standalone Part 5 report | Nested-CV prediction; full-cohort SHAP |
| TSSI leakage demonstration | TSSI notebooks + `rebuild_tssi_leakage_table.py` | Nested-CV headline |
| Wang historical comparator | `wang_vlst_score.ipynb` + Part 4 §7 | External validation of this pack |
| Terminology / banned phrases | `00_front_matter.md` | Root README slogan |

---

## 5. Provenance snapshot (as stated in the inspected files)

| Artefact | Stated provenance | Item class |
| --- | --- | --- |
| Part 4 nested CV | Kaggle Tesla T4, notebook commit `139d143`, Version 4, papermill 2026-09-04 | provenance detail |
| Part 5 interpretability | Kaggle Tesla T4, commit `e356bb1`, Version 5, papermill 2026-09-08–09 | provenance detail |
| Part 2 selectors | 2026-08-31 Kaggle run; per-selector CSVs not in repo | provenance detail |
| Part 1 association | `eda.ipynb` on `data/raw/VLST.csv`; Table C / 4b also via hygiene script | provenance detail |
| Bootstrap CIs (Table S-CI / S-Δ) | `run_b3()` on Version 4 OOF; **not** inside `baseline_plus_tabpfn.ipynb` | provenance detail |
| Table S-TSSI | Hard-coded in `rebuild_tssi_leakage_table.py` from stored 70/30 metrics | provenance detail |

---

## 6. Files the request named that were located

All requested basenames were found, with these path resolutions:

- `stats_vs_ml_comparison.ipynb` → `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb`
- `tabpfn_interpretability.ipynb` → `code/modeling/interpretability/tabpfn_interpretability.ipynb`
- Report `.md` files → `paper_results/` (canonical). `paper_results.md` and `00_front_matter.md` are in `paper_results/`. `paper_evidence_map.md` is in `docs/`. Two `README.md` files exist (repo root and pack).
