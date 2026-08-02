# Public headed-stud evaluator

This directory contains the path-neutral application evaluator and the four
runtime helpers previously imported from the private confirmatory runner:

- `build_models`
- `assignments_from_splitter`
- `atomic_write_json`
- `datasail_scalar_assignment`

The evaluator accepts three explicit roots: a user-local headed-stud adapter
root, a public `provenance_cut` method root, and an empty output directory. It
never reads a cached DataSAIL assignment and never writes target vectors,
predictions, source identifiers, record identifiers, or fold assignments.
DataSAIL runs in memory; the release surface retains only a checksum, fold
counts, typed status, solver metadata, and aggregate metrics.

The optional frozen control compares 12 keyed external model/data combinations
across four metrics (RMSE, MAE, NRMSE, and signed mean error), hence 48 numeric
cells. The reference is an aggregate `external_metrics.csv`, not row-level data.

Example after the four public adapters and the public method snapshot have been
prepared:

```text
python src/application_evaluator/run_public_evaluation.py \
  --adapter-root src/headed_stud_adapter \
  --method-root src/round_snapshot/method \
  --output-root derived/headed_stud_evaluator \
  --reference-external-metrics app
```

When the last argument is a directory, the evaluator reads the three
license-separated aggregate reference files already present in the public
`app/` projection; it does not create a mixed-license reference table.

Strict mode is the default. If DataSAIL 1.3.0/SCIP cannot return a complete
assignment, the evaluator writes a fail-closed status and stops. The explicit
`--allow-core-without-datasail` option emits the three-method core while marking
the release incomplete; it must not be used to claim the 4-method application
result.

Scope boundary: this reproduces the headed-stud application evaluator only. It
does not reproduce the confirmatory synthetic pipeline, raw-data acquisition,
or every supplemental application control.
