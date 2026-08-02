# Public scientific protocol projection

Protocol ID: `SAVP-CONFIRMATORY-V3.2`  
Master seed: `2026071204`  
Frozen scenarios: 38  
Replicates per scenario: 100  
Freeze-manifest SHA-256: `2e45f0fb2ceabda126c2ef54ba0eccf0aa8628d5f6090dd42e739702644c417f`

This document is the path-neutral scientific projection of the frozen internal
prespecification. It preserves the estimands, scenario order, fixed methods,
endpoints, multiplicity rule, execution gate and negative-result policy. It
omits submission strategy, internal workflow history, machine paths and
superseded-run diagnostics. The hashes of the source documents and the
redaction categories are recorded in `REDACTION_MAP.json`.

## Scientific question and outcome boundary

The compiler receives a typed deployment contract and only the provenance
columns named by that contract. Outcomes, predictions and external errors are
outside the compilation interface. Predictive evaluation begins only after the
deterministic conformance gate closes.

The contract binds an estimand to machine-checkable hard-separation, marginal
novelty, joint-pattern, conditioned-incidence, minimum-support, balance and
ordering clauses. A returned assignment is labelled split-exact only when an
independently implemented checker confirms every requested machine-checkable
split clause under the recorded provenance. This status does not establish
metadata truth, target-population coverage or external validity.

## Frozen stage order

1. compiler conformance;
2. S8 balance-infeasibility certificate;
3. S10 ordered-route decision;
4. S4 search-exhaustion state;
5. primary predictive stages S0 and S3;
6. stress stages S1, S2, S5, S6 and S7;
7. missing/corrupt-metadata stages S9-missing10, S9-missing30 and S9-corrupt5;
8. 24 core-grid cells crossing provenance effect `{0, 0.6, 1.2, 2.4}`,
   covariate shift `{0, 0.9, 1.8}` and noise `{0.35, 1.0}`.

Each stage uses two exclusive shards, replicates 0--49 and 50--99. A later
stage is ineligible until the preceding stage has exactly 100 claims, 100
results, zero failed records and a valid aggregate wrapper bound to the frozen
manifest and seed table. Overwrite and recovery into the confirmatory root are
forbidden.

## Fixed learners and comparator family

The four fixed learners are ridge regression, radial-basis-function support
vector regression, ExtraTrees and histogram gradient boosting. ClaimCut is the
reference compiler backend. The seven comparator slots are random K-fold,
GroupKFold by laboratory, source and supplier, composite laboratory-pair
grouping, the hard laboratory--supplier union when admissible, and the frozen
one-matrix DataSAIL C1e adapter. An unavailable comparator remains in the
multiplicity divisor and is reported as unavailable rather than deleted.

## Endpoints and decision rules

The primary predictive endpoint is record-weighted estimation-risk error
(ERE), the absolute difference between validation and target NRMSE. Equal-domain
and worst-domain ERE, signed and squared calibration error, rank correlation,
decision regret, relation residuals, semantic gaps, assignment state,
executability, runtime and comparator coverage are retained.

Paired differences are averaged across learners within each generated
meta-replicate. Uncertainty uses a deterministic 10,000-resample clustered
bootstrap over the 100 meta-replicates. The fixed family has seven comparators;
one-sided simultaneous bounds use familywise alpha `0.05` and Bonferroni
per-comparison alpha `0.05/7`.

- S0: ClaimCut minus random-K-fold ERE has a descriptive non-inferiority rule
  when the simultaneous upper bound is below `0.02`.
- S3: comparator-specific lower ERE is supported only when the simultaneous
  upper bound for ClaimCut minus the named comparator is below zero.

All other outcomes remain inconclusive or unfavourable. No result may change a
contract, endpoint, comparator, denominator or frozen stage after execution.

## Conformance hard gate

The 100 deterministic software cases test native split realization, plan and
assignment reproducibility, outcome-column exclusion, typed refusal accuracy,
separately implemented checker agreement, checker and production mutation
detection, checker--production agreement and exhaustive small-instance status
agreement. Any false split-exact event, escaped invalid mutation, invalid
certificate, outcome dependence, seed mismatch, unreadable/failed result,
orphan claim or envelope mismatch is a hard stop. These are fixed software
counts, not a population sample; no binomial confidence interval is attached.

## Release boundary

Only complete aggregate summaries and hash-bound source snapshots are eligible
for this capsule. Replicate-level results, partial aggregates, raw third-party
data, machine paths and active submission files are excluded.

