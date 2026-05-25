# Source-Aware Validation Protocol (SAVP)

SAVP is a deployment-matched validation audit layer for structural-engineering
machine-learning benchmarks. Its reporting unit is a validation claim rather
than a trained model.

For each task-level validation record, the protocol records:

- intended deployment target;
- grouping proxy and proxy-strength label;
- random and deployment-matched validation score;
- topology and group-size diagnostics;
- decision-risk diagnostics where applicable;
- a reviewer-readable claim verdict.

The implementation used for the nine-module benchmark suite is in `code/`.
The current public summary tables are in `outputs/tables/`, and dataset/source
access information is in `DATASETS_AND_LINKS.csv`.
