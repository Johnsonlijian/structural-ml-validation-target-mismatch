# Aggregate-derived outputs

The public release contains only the aggregate summary files listed in
`provenance/RELEASE_RECEIPT.json`. Replicate JSON, claims, model-row dumps,
partial stages, pilot outputs and raw third-party data are excluded.

Every copied summary is bound by relative path, SHA-256, byte count, media
type, schema or header and row count. The rc auditor recomputes those values.

The `app/` subtree is a separately audited 49-file headed-stud projection with
its own exact-set receipt and CC BY/CC BY-NC-SA licence partitions. Locally
reconstructed evaluator outputs belong under `derived/headed_stud_evaluator/`
and must pass the 48-cell aggregate check before use.
