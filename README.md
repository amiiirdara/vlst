# vlst

Personalized Risk prediction  for very late Stent Thrombosis (VLST) after PCI in ACS using Machin Learning

That slogan is the repository name only. Result claims follow `paper/frozen_results.yaml` and `paper/results.md`: association, nested-CV prediction, and TabPFN attribution on the Wang 2020 derivation cohort. Do not read the slogan as a validated, personalised, or externally tested model.

## Paper pack

Manuscript drafts: `paper/title.md`, `paper/abstract.md`, `paper/methods.md`, `paper/results.md`, `paper/discussion.md`. Numerical source of truth: `paper/frozen_results.yaml`.

Portable results pack (`paper_results/`):

```
paper_results/
  paper_results.md          # concat of Parts 0–5 + leakage sub-report
  anti_leakage_protocol.md  # five-flag protocol + incentives (W1)
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

Anti-leakage ON (nested / Part 2 / Part 5) is **ALL LEAKS OFF**: drop `NO.`/`Name`/TSSI/WBC; quantize `Cre`/`CaI`/`Fiberinogen`/`Fast-Glu` before split; stent encoder train-fold only; no SMOTE. The 70/30 twins invert all five flags together. Do not summarise as “drop TSSI.”

Interpretability is two notebooks (parent archived):

- `code/modeling/interpretability/tabpfn_interpretability_fs_pdp.ipynb` — MI, SFS, PDP
- `code/modeling/interpretability/tabpfn_interpretability_shap.ipynb` — SHAP / SHAP-IQ

See `paper_results/README.md` for the full notebook map.
