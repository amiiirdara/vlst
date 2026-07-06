Here is a structured comparison based on the executed outputs in all three notebooks.

## Summary: is everything OK?

Mostly yes — results.ipynb is internally consistent and the big split mismatch (70/10/20 vs 70/30) is fixed. But standalone TabPFN is still not a perfect apples-to-apples match with tabpfn.ipynb because the TabPFN feature matrix is not identical. That explains the PR-AUC and F1 gap.

---

## Headline numbers (test set: 1556 rows, 28 positives)

| Model / metric | tabpfn.ipynb | results.ipynb | advanced.ipynb |
|---|---:|---:|---:|
| Standalone TabPFN — PR-AUC | 0.818 | 0.765 | *(not run)* |
| Standalone TabPFN — F1 | 0.755 | 0.636 | — |
| TabPFN OOF train PR-AUC | 0.569 | 0.696 | — |
| XGB test PR-AUC | 0.670 | 0.673 | ~0.684 |
| LGB test PR-AUC | 0.650 | 0.705 | ~0.661 |
| Best TabPFN blend test PR-AUC | ~0.816 (stacking) | 0.779 (optuna) | — |
| Best tree blend test PR-AUC | — | ~0.705 (adv_7way) | ~0.734 (cal-tuned 7-way) |

Same split geometry everywhere now: 70% train / 30% test, 28 positives on test.

---

## Why TabPFN is lower in results.ipynb

### 1. PR-AUC gap (~0.818 → ~0.765) — different TabPFN inputs

Both notebooks use 81 raw features and the same row count (5185), but `results.ipynb` does not use the same raw matrix as `tabpfn.ipynb`:

| | tabpfn.ipynb load_raw() | results.ipynb df_to_raw_tabpfn() |
|---|---|---|
| Stent column | Raw "Stent type-SES" → category codes | Canonicalized stent_brand (advanced-style) |
| Extra cleaning | Minimal | drop_duplicates, PES/ZES/EVS coercion |
| Leakage drop | "Time since stent implantation" | Same |

Tree models in results.ipynb match tabpfn.ipynb closely (XGB ~0.67 in both), but TabPFN alone drops ~0.05 PR-AUC. That points to feature engineering, not the split.

TabPFN is sensitive to how categoricals are encoded; canonical stent brands change what the model sees vs raw messy strings.

### 2. F1 gap (~0.755 → ~0.636) — threshold + score quality

F1 depends on the decision threshold, not just ranking:

| | tabpfn.ipynb | results.ipynb |
|---|---|---|
| Threshold | 0.881 | 0.790 |
| Precision | 0.80 | 0.55 |
| Recall | 0.71 | 0.75 |
| Confusion (approx.) | FP=5, FN=8 | more FPs, similar FNs |

Two drivers:

- Coarser threshold grid in results.ipynb: 97 points on [0.02, 0.98] vs 199 points on [0.01, 0.99] in tabpfn.ipynb. TabPFN operates around 0.88–0.95; the finer grid matters.
- Different OOF scores (0.57 vs 0.70 OOF PR-AUC) → different optimal F2 threshold. Lower threshold → higher recall, lower precision → lower F1.

PR-AUC is threshold-free; F1 amplifies the gap because threshold tuning differs.

### 3. OOF looks *better* in results but test is *worse*

| | OOF train PR-AUC | Test PR-AUC |
|---|---:|---:|
| tabpfn.ipynb | 0.569 | 0.818 |
| results.ipynb | 0.696 | 0.765 |

That pattern suggests a generalization shift from the cleaned features, not a split bug. Same 10-fold OOF protocol, different inputs.

### 4. GPU / run-to-run noise (secondary)

TabPFN on GPU can vary slightly between runs. Unlikely to explain the full ~0.05 PR-AUC gap alone, but it adds noise on top of preprocessing differences.

---

## How the three notebooks relate

flowchart LR
  subgraph tabpfn_nb [tabpfn.ipynb]
    T1[Raw CSV minimal clean]
    T2[TabPFN standalone 0.82]
    T3[TabPFN blends ~0.81]
  end
  subgraph results_nb [results.ipynb]
    R1[Unified clean + scaled/raw views]
    R2[TabPFN standalone 0.76]
    R3[TabPFN blends win at 0.78]
    R4[Trees + advanced blends]
  end
  subgraph adv_nb [advanced.ipynb]
    A1[Scaled OHE only]
    A2[No TabPFN]
    A3[Heavy blend search ~0.73 test]
  end
  T1 --> T2
  R1 --> R2
  R1 --> R4
  A1 --> A3

- `tabpfn.ipynb`: TabPFN reference run — raw features, best standalone TabPFN (0.82).
- `results.ipynb`: Unified leaderboard — TabPFN blends still win, but standalone TabPFN is ~0.05 below tabpfn.ipynb.
- `advanced.ipynb`: Tree/boosters only — XGB/LGB ~0.66–0.68 test; best cal-tuned blends ~0.73–0.75. No TabPFN, so not directly comparable on the TabPFN question.

`results` vs `advanced` for trees: Roughly aligned (XGB ~0.67–0.68). Small gaps from simplified blend fitting in results (single cal slice vs 3 outer seeds in advanced).

---

## Is anything wrong?

| Check | Status |
|---|---|
| Same 70/30 split, 28 test positives | OK |
| TabPFN threshold policy (OOF F2) | OK in intent; grid coarser than tabpfn.ipynb |
| Tree baselines comparable across notebooks | OK (~0.65–0.71) |
| TabPFN standalone matches tabpfn.ipynb | Not yet — preprocessing mismatch |
| TabPFN blends beat trees in results | OK (expected) |
| advanced operational policy (~0.545 test) | Different goal (deployed conservative blend); don’t compare to TabPFN standalone |

---

## What to do if you want them to match

To get standalone TabPFN in results.ipynb closer to ~0.82:

1. Use `tabpfn.ipynb`’s exact `load_raw()` for the TabPFN path (no stent canonicalization on the raw view), while keeping advanced cleaning only for scaled models.
2. Match the threshold grid: np.linspace(0.01, 0.99, 199) like tabpfn.ipynb.
3. Optionally align blend base models (lr_raw etc.) with tabpfn.ipynb §9 factories exactly.

The current ~0.05 PR-AUC gap is explainable and not a split bug; it’s mainly TabPFN seeing a different feature table than in its dedicated notebook. I can patch results.ipynb so TabPFN uses the exact load_raw() path if you want that alignment.