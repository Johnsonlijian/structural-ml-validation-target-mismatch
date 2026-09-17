# Reproducible runbook (v3.1.0)

## 1. Environment

Python 3.11 with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `pyarrow` (see `requirements.txt`).
No GPU, no paid service and no network access is required for the reported numbers.

## 2. Inputs

Download each dataset from `DATASETS_AND_LINKS.csv` into the layout that the scripts expect under a
sibling project checkout:

```
<project>/data/raw/...                    third-party tables (not redistributed here)
<project>/code/outputs/datasets/...       provider workbooks
<project>/code/outputs/reproductions/...  released reproduction artefacts
```

`analysis/legacy_r35/01_corpus_inventory.py` prints the expected paths and fails loudly when a file
is missing.

## 3. Reproducing the reported results

| Step | Command | Output |
|---|---|---|
| Review verification | `python analysis/01_verify_review_findings.py` | `derived/R36_INDEPENDENT_VERIFICATION.json` |
| Controls | `python analysis/02_missing_controls.py` | `derived/control_metrics.csv`, `derived/fold_structure.csv` |
| Fold admissibility and contrasts | `python analysis/03_control_summary.py` | `derived/R36_fold_admissibility.csv`, `derived/R36_control_pairs.csv` |
| CFST boundary and repair | `python analysis/04_cfst_repair.py` | `derived/cfst_feasibility_map.csv`, `derived/cfst_repair.csv` |
| Registry | `python analysis/05_configuration_registry.py` | `derived/configuration_registry.csv` |
| Classification and weighting | `python analysis/06_b_controls.py` | `derived/classification_controls.csv`, `derived/weighting_controls.csv` |
| Decision experiment | `python analysis/07_d_decision_experiment.py` | `derived/D_outer_results.csv`, `derived/D_inner_grid.csv` |
| Data figures | `python analysis/09_rebuild_figures.py` | `figures/*.pdf`, `figures/*.png` |
| Numeric self-check | `python analysis/12_selfcheck_r36.py` | `derived/R36_MANUSCRIPT_SELFCHECK.md` (expects 34/34) |

## 4. Determinism

All learners use fixed seeds; `KFold(shuffle=True, random_state=0)`, `GroupKFold(min(5, n_groups))`
and the size-matched control (one permutation, cut to the honoured fold sizes) are fixed in the code.
Learners run serially (`n_jobs=1`) so the audit reproduces in constrained environments.

## 5. Boundaries

- The audit characterises the data asset and the evaluated claim, not the original studies' splits
  or intent; no source paper's deployment-claim wording was extracted verbatim.
- Fold-fraction bounds `[0.05, 0.50]` are a convention of this audit, and the CFST boundary is
  reported as a boundary rather than as an absolute limitation.
- Proxy relations are labelled as proxies and are never upgraded to source claims.
