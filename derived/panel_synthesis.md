# Panel synthesis (R38, 2026-09-20)

Seats: DeepSeek `deepseek-v4-pro` (architect), Kimi `kimi-k3` (citation/evidence), Qwen `qwen3-max`
(skeptic), Doubao `doubao-seed-2-0-pro-260215` (methods/figure). Full transcript:
`model_panel_report.md`; condensed input: `synthesis_input.md`.

## Convergent verdict

All four seats agree on the framing: the principle that dependence structures break random
cross-validation is established, so the paper must claim the **instrument** (registered relation, separable
checks, feasibility certificates, status record) and must present its results as **certifying and bounding
a claim**, not as improving a decision. Three seats also agree the decision experiment should be demoted
from a load-bearing claim; one seat notes that the certification framing turns the negative baseline result
from a weakness into the contribution.

## Disposition of every panel item

| Panel item | Seat | Disposition in this round |
|---|---|---|
| 34 executed − 6 excluded = 28, but the text reports 32 admissible entries | Kimi P1 | **Closed.** The chain is now explicit: 34 splits → 6 fold-size exclusions → 28 admissible splits → 48 learner-level contrasts → 36 admissible → 32 distinct entries after merging the aliased evaluation. New SI Table S12 and a ledger section. |
| Stratum denominators after reclassification | Kimi P1 | **Closed.** The 32 entries split 16 source-level / 8 combination-unseen / 8 duplication, stated in the SI, so 16-of-16 has an explicit denominator. |
| Factorisation of the 90 decision settings | Kimi P2 | **Closed.** 2 scenarios x 5 folds x 2 protocols x 3 targets x action sets (1 + 2) = 30 + 60 = 90, stated in the SI. |
| Utility break-even penalties | Kimi P2 | **Already present** (0.96, 0.75, 20/9) and kept. |
| Wall ceiling needs a construction spec, a completeness statement and composition counts | Kimi C8, Qwen, Doubao P1 | **Partly closed.** The ceiling is proved exhaustively (202 cluster structures); the *retention* figure is now labelled "best found" rather than an optimum; the campaign-by-type incidence is published as SI Table S13; Figure 4 shows the argument. |
| "Relation-cluster schematic" missing | DeepSeek, Qwen, Doubao P1 | **Closed.** Figure 4 (added this round) shows why B and I force one cluster and what the three-fold protocol costs. |
| Per-fold decision evidence | DeepSeek P1, Qwen | **Already present**: SI Table S9 and Figure 7 give per-fold counts and both effects' concentration. |
| "Unsafe release" definition | DeepSeek P1 | **Checked**: defined in the methods as a release whose measured capacity is below the demand (regression) or whose measured mode is in the designated family (classification). |
| Three checks are not a theory of model validity | Qwen P1 | **Closed.** A scope sentence now states that the audit bounds split-claim validity, not model quality; distributional shift, measurement error and label noise are out of scope. |
| Novelty must be positioned against grouped-CV/leakage literature | all four | **Closed.** A positioning paragraph cites structured cross-validation, leakage formulation/detection, the leakage-crisis study and selection bias; all four entries were verified through Crossref. Two of them were already in the bibliography, which the packet did not show the panel. |
| Adversarial PDF fixtures should be described | Qwen P2 | **Already present** in the SI section on instrument checks (image-only page, corner ink, six-point body, rotated page, coloured edge, text outside the page). |
| Utility arithmetic must not read as a cost model | Doubao P2 | **Present and strengthened**: the arithmetic is declared a function of a reader-supplied penalty, with no recommended value. |
| Archive the release with a DOI | Kimi P2 | **Human action.** The release is public and tagged but a mutable tag is not an archive; a Zenodo deposit needs the author's account. |
| Raw-to-aggregate chain cannot be audited | Kimi Q2 | **Partly closed**: the release ships the aggregation scripts and the verification entry point; raw third-party tables stay undistributed with the reason stated, so table-level verification is what the paper promises. |
| Packet had no bibliography or figure/table inventory | Kimi, Qwen | **Packet limitation, not a manuscript defect.** The manuscript carries 30 references and a full inventory; the packet supplied claims without them. |
| Venue: AiC needs explicit construction-domain value, otherwise AEI | DeepSeek, Qwen | **Author decision.** The manuscript is written to satisfy both envelopes; the framing change (certification rather than decision improvement) is the constructive part of this finding. |

## Claims that were narrowed or demoted as a result

1. The wall retention figure is "the best protocol found", with the ceiling alone proved.
2. The abstract now closes on bounds and certification rather than on supporting decisions.
3. The decision experiment is presented as the protocol comparison it is, with the concentration and the
   reversal in tables and a figure, not as a safety improvement.

## What this round did not do, and why

- No new experiment was run: every panel item was either a reporting/accounting gap, a framing change or a
  documentation gap. The one experiment-shaped suggestion (extend the oracle tests to more edge cases) is
  recorded as an optional strengthening, not a defect.
- The Zenodo DOI remains a human action.
