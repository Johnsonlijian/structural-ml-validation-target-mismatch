# Reproducibility Sign-Off

Date: 2026-05-25

## Local Verification

- `python tests\smoke_test.py`: passed.
- Public tree scan for private submission files, round folders, logs, conflict filenames, local paths, and manuscript payloads: passed.
- A full raw-data rerun is intentionally not part of this public smoke test because raw third-party datasets are not redistributed.

## Rebuilt Artefacts

- `code/outputs/figures/cace_ninemodule_rf_summary.csv`
- `outputs/tables/cace_ninemodule_rf_summary.csv`
- `outputs/tables/classification_metrics_random_grouped.csv`
- `outputs/tables/regression_validation_metrics_random_grouped_loso.csv`
- `figures/generated/figS5_cace_ninemodule_rf_gaps_v0.png`
- `figures/supplementary/figS5_cace_ninemodule_rf_gaps_v0.png`
- `data_manifests/checksums_sha256.csv`

## Boundary

Raw third-party data are not redistributed. The package ships public scripts,
source manifests, derived validation tables, and generated figures; a full
raw-data rerun requires fetching source datasets under their original access
terms.
