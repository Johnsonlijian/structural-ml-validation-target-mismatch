# Reproducible Runbook

## 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Alternatively, use the conda-style environment file:

```bash
conda env create -f environment_reproducibility_v0.yml
```

## 2. Smoke test

```bash
python tests/smoke_test.py
python run_qa_chain.py --smoke-test
```

## 3. Derived outputs

Derived outputs are stored under:

- `outputs/tables/`
- `outputs/figures_source/`
- `figures/`

The public package intentionally ships derived tables and generated figures,
not raw third-party datasets. Dataset links, access dates, grouping variables,
and redistribution notes are in `DATASETS_AND_LINKS.csv`.

## 4. Full rerun boundary

A full raw-data rerun requires downloading source datasets from their original
hosts under the original terms. Group identifiers are used for validation
splits only and are not supplied as model features.
