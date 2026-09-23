# VLST paper results (portable pack)

This folder is self-contained. Send the whole `paper_results` directory (or the zip) without the rest of the repo. Part 0 (`00_front_matter.md`) is the motivation / limitations / terminology note.

**Live nested TabPFN anti-leakage-ON (9 arms):** thinking v3.5 PR-AUC **0.9212**; TabPFN v3.5 **0.8957**. Pins **`tabpfn==9.0.0`** / **`tabpfn-client==0.6.0`**. Freeze key `nested_cv_v35_antileakage_on`. Unlabeled 0.9771 / 0.9635 excluded. Classics in nested CV use defaults + class weighting; GridSearch winners are **not** imported. **Protocol (W1):** [`anti_leakage_protocol.md`](anti_leakage_protocol.md) — five flags + incentives, not “drop TSSI.” Leakage twins: `04_tabpfn_rating/leakage_contrast_paper_figures_and_tables.md`.

## Open

- **One combined file:** `paper_results.md` (front matter + anti-leakage protocol + Parts 1–5 + leakage sub-report; figures inlined via relative paths).
- **Separate files:**
  - `00_front_matter.md` (motivation, EPV, limitations, terminology — W1 protocol + W2–W5)
  - `anti_leakage_protocol.md` (five flags, incentives, companion controls)
  - `01_eda/EDA_paper_figures_and_tables.md`
  - `02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md`
  - `03_stats_vs_ml/feature_extraction_comparison.md`
  - `04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md`
  - `04_tabpfn_rating/leakage_contrast_paper_figures_and_tables.md` (ALL LEAKS ON vs ALL LEAKS OFF)
  - `05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md` (from the two split notebooks)

Each separate markdown uses `paper_figures/` **in the same folder** (Parts 4 and 4b share `04_tabpfn_rating/paper_figures/`). Do not move a `.md` file without its `paper_figures/` sibling.

## Notebooks (not in this zip; live in the repo)

| Notebook | Pack part |
| --- | --- |
| `code/analyzes/eda.ipynb` | Part 1 |
| `code/modeling/interpretability/baseline_feature_selections.ipynb` | Part 2 |
| `code/analyzes/stats_vs_ml/stats_vs_ml_comparison.ipynb` | Part 3 |
| `code/modeling/rating/baseline_plus_tabpfn.ipynb` | Part 4 nested CV |
| `code/modeling/rating/baseline_tssi_leakage.ipynb` | Part 4b ALL LEAKS ON |
| `code/modeling/rating/baseline_without_tssi.ipynb` | Part 4b ALL LEAKS OFF |
| `code/modeling/rating/wang_vlst_score.ipynb` | Part 4 Wang comparator |
| `code/modeling/interpretability/tabpfn_interpretability_fs_pdp.ipynb` | Part 5 MI / SFS / PDP |
| `code/modeling/interpretability/tabpfn_interpretability_shap.ipynb` | Part 5 SHAP / SHAP-IQ |
| `code/modeling/interpretability/tabpfn_interpretability.ipynb` | archived parent (session-limit split) |

## Layout

```
paper_results/
  paper_results.md
  README.md
  anti_leakage_protocol.md
  00_front_matter.md
  01_eda/
  02_ml_selectors/
  03_stats_vs_ml/
  04_tabpfn_rating/
    baseline_plus_tabpfn_paper_figures_and_tables.md
    leakage_contrast_paper_figures_and_tables.md
    paper_figures/
  05_tabpfn_interpretability/
    tabpfn_interpretability_paper_figures_and_tables.md
    paper_figures/
```
