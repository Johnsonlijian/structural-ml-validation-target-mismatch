# Contract-compiler figures

These builders are path-neutral public counterparts for manuscript Figures
1--4. They generate only `svg/`, `pdf/`, and `png/` outputs beneath an explicit
output directory.

- `build_contract_lifecycle_figure.py` and
  `build_joint_semantics_figure.py` are constructive diagrams and need no
  empirical record-level input.
- `build_compiler_gate_figure.py` accepts only a closed public `a32` aggregate
  root. Record roots, freeze roots, symlinks, and any path belonging to an
  excluded/interrupted attempt are refused before output creation.
- `build_primary_scaling_figure.py` additionally requires the frozen public
  `scalability_v1` bundle, `PRIMARY_EVIDENCE_LOCK.json`, and
  `provenance/ATTEMPT_T_RETENTION_AUDIT.json`. In closed mode the builder
  independently hashes the retain-all receipt, verifies all 38 admitted stage
  manifests, and matches the S0/S1/S3 raw summary bindings to the selected
  aggregate root. The Figure lock remains limited to the plotted S0/S3 stages;
  its receipt hash also binds S1. The controlled draft deliberately ships a
  closed T-only lock; both check and build routes fail closed if any of its
  closure, retention or aggregate bindings drift.
- `figures/core/_impl/build_figure_4.py` is a compatibility entry point that
  delegates to this canonical builder instead of maintaining a second gate.

Examples after release closure:

```powershell
python figures/contract_compiler/build_contract_lifecycle_figure.py --output-root build/figures
python figures/contract_compiler/build_joint_semantics_figure.py --output-root build/figures
python figures/contract_compiler/build_compiler_gate_figure.py --aggregate-root derived/confirmatory_aggregates_v32 --output-root build/figures
python figures/contract_compiler/build_primary_scaling_figure.py --aggregate-root derived/confirmatory_aggregates_v32 --scalability-root derived/scalability_v1 --retention-receipt provenance/ATTEMPT_T_RETENTION_AUDIT.json --output-root build/figures
```

The aggregate directory is distributed under the exact logical public path
`derived/confirmatory_aggregates_v32`, with hash-bound stage receipts under
`provenance/stages`. No builder reads raw records, fold assignments, targets,
predictions, or an excluded attempt tree. `--retention-receipt` may point to
another path only for controlled local verification; its bytes must match the
SHA-256 stored in the closed primary evidence lock.

The private T closure process sealed the lock only after all 38 aggregate
manifests and the retain-all receipt passed. The public builder revalidates
those bindings and verifies the final Figure 3--4 PDF hashes before copying.
