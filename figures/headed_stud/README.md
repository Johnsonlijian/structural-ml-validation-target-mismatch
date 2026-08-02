# Headed-stud application figures

The two public wrappers read only the receipt-bound aggregate projection under
`app/` and inject those aggregates into the frozen project-authored figure
implementations under `_impl/`.

```bash
python figures/headed_stud/build_headed_stud_contract_figure.py \
  --evidence-root app --output-dir derived/figures

python figures/headed_stud/build_headed_stud_consequence_figure.py \
  --evidence-root app --output-dir derived/figures
```

The Figure 5 wrapper also verifies the committed row-free aggregate projection
`multirelation_evidence.json`. It contains only topology/material cell counts,
fold-level metadata summaries, backend states and source hashes; it contains no
row-level data, targets or predictions. An alternate verified projection can be
provided with `--multirelation-evidence PATH`.

Both commands generate editable SVG, PDF and PNG outputs. They do not read
row-level data. The wrappers verify the expected contract states, row counts,
model set and aggregate-file hashes before drawing.
