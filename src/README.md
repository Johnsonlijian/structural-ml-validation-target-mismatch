# Source snapshot

In a built release candidate, `round_snapshot/` contains only source files
whose live SHA-256 values match the frozen source registry. The public source
ledger records destination paths and hashes without exposing canonical machine
paths.

The headed-stud adapter, evaluator and aggregate-only figure wrappers are
included as path-neutral source. The exact rc file ledger hashes each file,
while the application receipt separately binds the aggregate evidence. Their
scope is the headed-stud application; they do not widen the synthetic
confirmatory claim.
