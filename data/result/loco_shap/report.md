# LOCO & SHAP multi-objective feature importance report

Generated: —  
TabPFN context: train — rows, test — rows, — features  
Top-K = 12 per method × objective (`pr_auc`, `f1`, `f2`)

## Executive summary

- **LOCO (pr_auc)** — baseline test pr_auc = **0.818**. Top drops: `eGFR`, `Cre`, `Visual thrombus`.

## Methods

| Method | Objective | What is measured | Cost |
|--------|-----------|------------------|------|
| **LOCO** | pr_auc / f1 / f2 | Drop in test metric when one column is removed and TabPFN is **refit** | O(d) fits |
| **SHAP** | pr_auc / f1 / f2 | Coalition Shapley value of the **global** metric on explained test rows (mask to background; no refit) | O(n_perm × k) forward passes |
| **FFS** (tabpfn_playground.ipynb §10) | pr_auc / f1 / f2 | Greedy forward add on holdout | O(K × d) fits |

**Threshold policy:** For `f1` and `f2`, the metric is computed after tuning the decision threshold on the same scores used for evaluation (test set for LOCO; explained cohort for SHAP). This matches the FFS holdout logic in `tabpfn_playground.ipynb` but is optimistic if you iterate on test — use for exploratory ranking, not final model selection.

## Top features by objective

### PR_AUC

**LOCO (drop-one, refit):**

| Rank | Feature | Importance | Score without |
|------|---------|------------|---------------|
| 1 | eGFR | 0.3335 | 0.4846 |
| 2 | Cre | 0.2964 | 0.5217 |
| 3 | Visual thrombus | 0.2829 | 0.5351 |
| 4 | P-LM | 0.2454 | 0.5726 |
| 5 | Moderate/severe tortuosity | 0.2434 | 0.5746 |
| 6 | Lesion location-Ostial | 0.2408 | 0.5773 |
| 7 | Moderate/severe calcification | 0.2365 | 0.5816 |
| 8 | Thrombus aspiration | 0.2301 | 0.5879 |
| 9 | Ticagrelor | 0.2285 | 0.5896 |
| 10 | stent overlap | 0.2271 | 0.5910 |
| 11 | Bifurcation | 0.2250 | 0.5931 |
| 12 | Dissection | 0.2194 | 0.5986 |

### F1

### F2

## Pairwise overlap (Jaccard, top-K sets)

_Run full pipeline to populate SHAP and multi-objective LOCO overlap._

### Same method, different objectives


## How to read this

1. **LOCO** answers: *If I delete this column and retrain TabPFN, how much does metric X fall?* High bars = features the model cannot compensate for. Correlated twins may both look weak.
2. **SHAP** answers: *When this feature is revealed vs a background patient, how much does metric X rise on the explained cohort?* Uses the §5 context model (no refit). Correlated features can share credit.
3. **Objective matters:** `pr_auc` favours ranking power; `f2` favours recall-heavy operating points; `f1` balances precision and recall. The same feature can rank high under one objective and low under another.
4. **Clinical top signals (pr_auc LOCO, typical run):** renal function (`eGFR`, `Cre`), angiographic thrombus/tortuosity/calcification, and left-main involvement — consistent with VLST biology.

## Artifacts

- LOCO CSVs: `data/result/loco_shap/loco_{metric}.csv`
- SHAP CSVs: `data/result/loco_shap/shap_{metric}.csv`
- Overlap matrix: `data/result/loco_shap/overlap_matrix.csv`
- Figures: `data/result/loco_shap/loco_by_objective.png`, `overlap_heatmap.png`
