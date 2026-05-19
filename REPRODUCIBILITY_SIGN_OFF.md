# Reproducibility Sign-Off

Date: 2026-05-12

## Local Verification

- `python tests\smoke_test.py`: passed.
- `python run_qa_chain.py --smoke-test`: passed.
- `uv run --with pandas --with matplotlib --with numpy python run_qa_chain.py --skip-06 --skip-24 --skip-34`: passed.

## Rebuilt Artefacts

- `code/outputs/figures/cace_ninemodule_rf_summary.csv`
- `outputs/tables/cace_ninemodule_rf_summary.csv`
- `outputs/tables/classification_metrics_random_grouped.csv`
- `outputs/tables/regression_validation_metrics_random_grouped_loso.csv`
- `figures/draft/figS5_cace_ninemodule_rf_gaps_v0.png`
- `figures/supplementary/figS5_cace_ninemodule_rf_gaps_v0.png`
- `data_manifests/checksums_sha256.csv`

## Boundary

Raw third-party data are not redistributed. The package regenerates headline tables and Figure S5 from bundled derived outputs; a full raw-data rerun requires fetching source datasets under their original access terms.
