# Structural ML validation target mismatch

Reproducibility package for the manuscript:

**A deployment-matched validation protocol for structural-engineering machine-learning benchmarks**

This repository contains public code, configuration, dataset/source manifests,
derived tables, generated figures, and run instructions for the SAVP
(Source-Aware Validation Protocol) benchmark suite.

## Contents

- `code/`: download, reproduction, diagnostics, and figure-generation scripts.
- `outputs/tables/`: derived supplementary tables and diagnostic CSV outputs.
- `outputs/figures_source/`: figure-source CSV exports.
- `figures/`: generated raster/vector figure exports.
- `data_manifests/` and `DATASETS_AND_LINKS.csv`: source registry, access dates,
  redistribution flags, grouping variables, and dataset links.
- `tests/`: smoke tests for the public package.

## Not included

- Raw third-party datasets whose redistribution rights are unclear or restricted.
- Active submission manuscripts, cover letters, reviewer-response drafts,
  private round folders, logs, credentials, or local virtual environments.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python tests/smoke_test.py
```

For a fuller rerun, fetch the original public datasets according to
`DATASETS_AND_LINKS.csv` and the scripts in `code/`. The bundled smoke test is
offline and checks the public package layout, derived tables, and trace hygiene.

## Archive

Zenodo DOI: https://doi.org/10.5281/zenodo.20006918

## Intended remote

`https://github.com/Johnsonlijian/structural-ml-validation-target-mismatch.git`
