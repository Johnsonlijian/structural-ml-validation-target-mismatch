# Release notes - v3.3.0

Supersedes v3.2.0 after a third independent review of the final PDFs.

Corrected:
- **geometry**: wide tables moved to landscape pages, long tables converted to `longtable` so they break
  across pages instead of running off the paper, all `\resizebox` scaling removed, figures redrawn on a
  fixed canvas at a 10 pt base. A rendered-page gate now fails the build on ink in the paper-edge band
  or type below 7 pt.
- **corpus registry**: the table and every count in the text are generated from one registry
  (11 entries, 5,035 records; 6 entries and 3,745 records in the analysis; 5 entries and 1,290 records
  inventory-only). The earlier "nine assets, 4,933 records" claim is withdrawn.
- **relation-stratified contrasts**: the aliased evaluation appears once (32 distinct entries), strata are
  derived from asset *and* variant so the concrete duplication key is no longer labelled a wall-type
  proxy, unevaluated cells read "not evaluated", and the honoured R2 column is restored.
- **decision experiment**: fold means and pooled counts are reported separately; the improvement claim is
  limited to the strictest target and the reversal at looser targets is stated explicitly.
- **metadata**: the two Mendeley entries re-verified through DataCite (Fitwi 2025 CC BY-NC 4.0;
  Salem 2022 CC BY 4.0); the DesignSafe coupling database recorded as PRJ-3053; the code-availability
  statement no longer mixes tenses.
- **failure-mode family**: the classification asset ships no codebook, so the family is stated as a
  scenario category (label code 1) rather than as a "brittle" family.

## v3.4.0

- Adds `verify_reported_numbers.py`, a runnable entry point that reproduces every reported number from
  the shipped aggregate tables; earlier releases documented commands that could not run from a clean
  checkout, which an independent audit correctly flagged.
- Adds the fold-influence and utility-sensitivity tables and their source CSVs, the two results added to
  the manuscript after the concentration of the beam-scenario improvement was measured.
- README corrected: the analysis scripts are archived for inspection, not offered as a pipeline, because
  they depend on raw third-party tables that are not redistributed.
