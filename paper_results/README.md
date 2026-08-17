# VLST paper results (portable pack)

This folder is self-contained. Send the whole `paper_results` directory (or the zip) without the rest of the repo.

## Open

- **One combined file:** `paper_results.md` (all three reports, figures inlined via relative paths).
- **Separate files:**
  - `01_eda/EDA_paper_figures_and_tables.md`
  - `02_ml_selectors/baseline_feature_selections_paper_figures_and_tables.md`
  - `03_stats_vs_ml/feature_extraction_comparison.md`

Each separate markdown uses `paper_figures/` **in the same folder**. Do not move a `.md` file without its `paper_figures/` sibling.

## Layout

```
paper_results/
  paper_results.md
  README.md
  01_eda/
    EDA_paper_figures_and_tables.md
    paper_figures/
  02_ml_selectors/
    baseline_feature_selections_paper_figures_and_tables.md
    paper_figures/
  03_stats_vs_ml/
    feature_extraction_comparison.md
    paper_figures/
```

Notebooks (`eda.ipynb`, `baseline_feature_selections.ipynb`) are **not** included.
