# When can validation scores support structural-engineering decisions?

Reproducibility package **v3.3.0** for the manuscript under review at *Advanced Engineering Informatics*.

Supersedes v3.2.0 after a third independent review that audited the **final PDFs** rather than the
sources. That review found clipped tables, type as small as 4.2 pt, an inventory table that did not
support the corpus numbers printed in the text, a main results table that still duplicated an aliased
evaluation and mislabelled two relation types, and a decision conclusion that reversed direction at
looser targets.

## What the corrected analysis establishes

1. **A corpus registry that generates the numbers**: eleven identified entries holding 5,035 records;
   six entries (3,745 records) supply the protocol-controlled analysis, five (1,290 records) contribute
   to the inventory and metadata audit only.
2. **Relation-stratified contrasts** against size-matched random controls. The concrete asset's
   eight-input key and its proxy label are the same evaluation, so the admissible set holds **32 distinct
   entries**; the **source-level stratum degrades in 16 of 16** contrasts (median -0.165 in R2). The
   wall-type splits fail the fold-size check and are excluded, so no proxy row appears.
3. **A feasibility boundary with an explicit narrowing**: the fold-bound value at which a CFST
   family-honouring split becomes feasible, and a restricted holdout covering 48.8 % of records
   registered as a *narrowed* claim. Scored on the same rows, its difference from a random split is
   -1.05 for ridge and at most 0.07 for the other learners.
4. **A decision experiment with defined denominators**, fold means and pooled counts reported
   separately. Random-fold selection was optimistic in every setting; grouped inner validation was
   better at the strictest target in both scenarios and **reversed direction in the classification
   scenario at looser targets**.
5. **A geometry gate** (`analysis/19_geometry_check_v3.py`) that renders each page and fails the build if
   ink enters the paper-edge band or type falls below 7 pt — the check the earlier log-based instrument
   could not provide.

## Contents

| Directory | Contents |
|---|---|
| `analysis/` | R38 scripts: corpus registry, table rebuilds, figure regeneration, the geometry gate, metadata verification and the manuscript self-check |
| `analysis/r37/`, `analysis/r36/` | The decision experiment and the control/registry scripts they build on |
| `analysis/legacy_r35/` | Corpus-level scripts (inventory, entitlement gaps, coverage, unified protocol run) |
| `tools/network/` | Identifier-resolution scripts (Crossref / DataCite / arXiv) |
| `derived/` | Aggregate tables and reports behind every number, including the corrected decision tables and the geometry report |
| `figures/` | Vector (PDF) and PNG versions of Figures 1-5, plus the TikZ source of Figure 1 |

## Not included

Raw third-party tables and per-row predictions. Redistribution rights for the underlying measurements are
not uniform across the sources; the audit is regenerated from the public identifiers in
`DATASETS_AND_LINKS.csv`.

## Quick start

```
pip install -r requirements.txt
python analysis/04_corpus_registry.py        # registry -> corpus numbers and the inventory table
python analysis/12_rebuild_all_tables.py     # every manuscript table
python analysis/14_figures_legible_type.py   # Figures 2-5
python analysis/19_geometry_check_v3.py      # rendered-page geometry gate
python analysis/27_selfcheck_r38.py          # guards every quoted number (38 checks)
```

## Licence

Project-authored code and documentation: MIT (`LICENSE`). Third-party datasets are not redistributed.
