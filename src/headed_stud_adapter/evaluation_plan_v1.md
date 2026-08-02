# Frozen headed-stud multi-parent evaluation plan v1

Status: frozen before numerical target access. The target column names and physical definition were verified from headers and published documentation; no target cell was used to choose parents, mappings, exclusions, models or metrics.

## Engineering question

Can a validation compiler distinguish a deployment that can be emulated by internal source holdout from one that is structurally unsupported by the development population, and does that distinction matter when an ML resistance model is deployed from normal-weight solid slabs to profiled-deck, lightweight-concrete or recycled-aggregate headed-stud tests?

The main application is `NWC solid slab (242 tests) -> profiled steel deck`. The source-quarantined external parent has 439 rows before complete-case filtering and 27 exact citation keys. `NWC -> LWC` is a near-domain material control after 30 overlapping rows are quarantined. `NWC -> RAC-SCC` is a 27-specimen, one-programme descriptive confirmation and carries no source-cluster population inference.

## Frozen contracts

Each contract is compiled from provenance and deployment metadata only.

1. `C_source`: new source/publication at deployment. Five-fold source GroupKFold is the exact simple specialization; the native compiler must also compile or return a typed feasibility result.
2. `C_deck`: new source plus `slab_topology=profiled_deck`, whereas all development rows have `slab_topology=solid`. A random/source-only split may execute but cannot be labeled an exact deployment match. The full contract is expected to refuse because the topology transition has no internal support.
3. `C_lwc`: new source plus unseen `concrete_family=lightweight`; expected full-contract refusal from NWC-only development.
4. `C_rac`: new source plus unseen `concrete_family=recycled_aggregate_scc`; expected full-contract refusal from NWC-only development.

A refusal is a successful safety output, not missing performance. The external target is evaluated after the refusal to quantify the consequence of using an executable but semantically incomplete validation proxy.

## Frozen learners and preprocessing

The input vector is the seven measured variables frozen in `adapter_freeze_v1.json`, in that exact order. Four learners match the synthetic study:

- standardized Ridge with `alpha=1`;
- standardized RBF-SVR with `C=10`, `epsilon=0.05`, `gamma=scale`;
- ExtraTrees with 160 trees, minimum leaf 3 and maximum feature fraction 0.9;
- histogram gradient boosting with 180 iterations, learning rate 0.06, 31 leaf nodes and L2 regularization 0.1.

All preprocessing is fitted inside each training fold. There is no hyperparameter search, feature selection, target transform, winsorization, clipping, reweighting or target-informed exclusion after unseal.

## Frozen validation proxies

- shuffled five-fold KFold, seed `2026071201`;
- five-fold GroupKFold by adjudicated source programme, shuffled with seed `2026071201`;
- native ClaimCut for `C_source` if feasible;
- DataSAIL-C1e scalar split only if the frozen adapter executes without changing its semantics or solver limits; solver status and semantic gaps are retained;
- typed outcomes for `C_deck`, `C_lwc` and `C_rac` before any model fitting.

The primary comparison is not splitter superiority. It is whether an executable proxy is honestly labeled exact, lossy or unsupported.

## Frozen outcomes

For each learner and executable validation proxy:

- fold-size-weighted CV RMSE and MAE;
- external RMSE, MAE and signed mean error;
- all errors normalized by the development-target sample standard deviation;
- external-risk estimation error `abs(CV_NRMSE - external_NRMSE)`;
- selected learner and full-refit external model-selection regret;
- realized source novelty, contract residual and typed compiler status.

For the deck parent, 10,000 source-cluster bootstrap resamples give 95% two-sided intervals for full-refit external RMSE/MAE and the paired difference between CV-estimated and external risk. LWC uses the same source-cluster procedure but is explicitly labeled small-cluster (six retained sources before complete-case filtering). RAC reports point estimates and an optional specimen bootstrap sensitivity only; it does not claim population-level independence because all 27 rows arise from one programme.

Feature-support shift is reported using development-only scaling and external proportions outside the development min-max range. It is descriptive and cannot trigger row deletion.

## Hard gates

- Abort a parent if the target does not match measured per-stud resistance in kN.
- Abort if frozen source quarantine cannot be reproduced from row identifiers.
- Deck remains primary only with at least 100 complete rows and 10 retained source groups.
- Do not interpret RAC as an independent multi-source validation.
- Do not claim that the compiler guarantees external calibration or that ClaimCut universally beats GroupKFold/DataSAIL.
- Any mapping, model, exclusion or metric change after target unseal creates v2 and reports this deviation before rerunning.

## Permitted headline if gates pass

The deployment compiler made an otherwise hidden structural mismatch explicit: source holdout was executable for the solid-slab development data, but profiled-deck and concrete-family deployments were not internally emulable. Source-quarantined external tests then quantified the calibration consequence of proceeding with those lossy proxies. This supports auditable validation design and typed refusal, not universal splitter superiority.
