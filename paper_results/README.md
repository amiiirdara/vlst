# VLST paper results (portable pack)

This folder is self-contained. Send the whole `paper_results` directory (or the zip) without the rest of the repo.

## Open

- **One combined file:** `paper_results.md` (all five reports, figures inlined via relative paths).
- **Separate files:**
  - `01_eda/EDA_paper_figures_and_tables.md`
  - `02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md`
  - `03_stats_vs_ml/feature_extraction_comparison.md`
  - `04_tabpfn_rating/baseline_plus_tabpfn_paper_figures_and_tables.md`
  - `05_tabpfn_interpretability/tabpfn_interpretability_paper_figures_and_tables.md`

Each separate markdown uses `paper_figures/` **in the same folder**. Do not move a `.md` file without its `paper_figures/` sibling.

## Layout

```
paper_results/
  paper_results.md
  README.md
  01_eda/
  02_ml_selectors/
  03_stats_vs_ml/
  04_tabpfn_rating/
  05_tabpfn_interpretability/
```

Notebooks are **not** included.
