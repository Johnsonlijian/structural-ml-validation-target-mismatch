# Formal specification v3: typed deployment-validation contract compiler

Status: active post-adversarial-audit specification. This document defines implemented semantics and downgrade boundaries; it does not assert that confirmatory gates have passed.

## 1. Contract object

A contract has schema `deployment-contract-v3` and contains two distinct layers.

The estimand layer declares:

- prediction unit;
- outcome;
- named loss and canonical loss parameters;
- target population;
- record, equal-domain, or explicitly specified custom weighting;
- mean or worst-domain risk aggregation;
- decision estimand and, when required, a declared domain relation;
- metadata source and endpoint-binding status.

The split-semantics layer declares:

- typed provenance relations;
- hard train-validation separation clauses;
- per-fold marginal novelty targets and tolerances;
- joint novelty-pattern targets and tolerances;
- conditional cross-relation incidence targets and tolerances;
- minimum observable support for every marginal, joint, and incidence clause;
- fold-size bounds and ordered relations.

Strict confirmatory contracts reject legacy-defaulted estimand metadata. The complete contract, including estimand metadata and support minima, enters the contract hash.

## 2. Outcome exclusion

Compilation accepts a provenance table and selects only relation columns named in the contract. Feature values, observed outcomes, external errors, and model ranks are outside the compiler interface. Outcome exclusion is tested by supplying differently transformed unreferenced outcome columns and requiring identical plans and assignments. This is an architectural conformance property, not a population-level statistical claim.

## 3. Relation novelty and observable support

For fold \(k\), record \(i\), and observed relation \(r\),

\[
u_{irk}=\mathbb{1}\!\left[p_{ir}\notin
\{p_{jr}:z_j\neq k,\ p_{jr}\neq\varnothing\}\right].
\]

Each marginal target specifies a target novelty \(t_r\), tolerance \(\epsilon_r\), and minimum observed records \(m_r\) per fold. A fold with fewer than \(m_r\) observed records is unevaluable. It never contributes an implicit zero residual. Every fold must be evaluable and satisfy

\[
|\widehat\nu_{rk}-t_r|\le \epsilon_r.
\]

Missing provenance remains missing and is reported. It is not converted to a shared entity.

## 4. Joint novelty and incidence topology

For relations \(\mathcal R_J\), a joint target gives

\[
\pi_C(b)=\Pr\{u_{ik,\mathcal R_J}=b\},
\qquad b\in\{0,1\}^{|\mathcal R_J|}.
\]

The realized distribution is checked by total variation in every fold. A fold below its minimum joint-support count is unevaluable and fails the clause.

For relations \(A,B\), an incidence target may specify the mean distinct \(B\)-degree per \(A\) entity, optionally conditioned on novelty of a third relation. A fold below its minimum number of left entities, including a condition-empty fold, is unevaluable and fails the clause.

Equal marginals do not identify a joint distribution, and equal joint novelty patterns do not identify cross-relation incidence. The implementation therefore preserves these clauses separately. It does not claim to exhaust all possible graph motifs or provenance semantics.

## 5. Hard components and analytic certificates

Entities declared `hard_unseen` induce equivalence edges between records. Connected components of the union graph are indivisible. Before heuristic search, the implementation checks analytic necessary conditions:

- component count below the requested number of nonempty folds;
- globally contradictory integer fold bounds;
- largest hard component exceeding the integer upper bound of one fold.

Only these proven conditions use a `certified_*` status. Each certificate has schema `infeasibility-certificate-v1` and binds:

- contract and provenance hashes;
- record and fold counts;
- hard relations;
- canonical component-membership hash;
- integer facts supporting the predicate;
- certificate hash and independent verifier version.

The independent verifier recomputes the graph, hashes, integer bounds, and predicate without importing production solver or audit code.

## 6. Search and realized-plan taxonomy

The native backend is a seeded multistart heuristic. Failure to find an admissible assignment within the frozen search budget is not an infeasibility proof.

Active states are:

- `certified_partition_infeasible`: hard components cannot form the requested number of nonempty folds;
- `certified_balance_infeasible`: an analytic integer/component certificate proves the fold bounds impossible;
- `backend_search_exhausted`: the frozen heuristic did not find a candidate satisfying all clauses; global feasibility remains unresolved;
- `realized_plan_invalid_hard`, `realized_plan_invalid_balance`, or `realized_plan_invalid_targets`: a returned assignment failed the named audit;
- `ordered_route_required`: an unordered cut is inappropriate and a predeclared forward route is required;
- `backend_semantic_gap`: the selected backend cannot express all contract clauses;
- `compiled_exact`: the backend has no declared semantic gap and the realized assignment passes the complete audit.

Candidate audit fields are named `candidate_hard_satisfied`, `candidate_balance_satisfied`, and `candidate_targets_satisfied`. They are not statements of global feasibility. `feasibility_status` distinguishes `admissible`, `certified_infeasible`, `unresolved`, `invalid_candidate`, `invalid_realized_plan`, `proxy_admissible`, and `route_required`.

## 7. Plan, fidelity, assignment, and execution

One status string is not used to imply all lifecycle states. A compilation result also reports:

- `plan_state`: emitted or not emitted;
- `fidelity`: exact, lossy, or unknown;
- `assignment_state`: not requested, not returned, returned unaudited, audited exact, audited lossy, or audit failed;
- complete-contract satisfaction;
- backend-projection satisfaction;
- comparator executability and contract admissibility.

For the frozen one-matrix DataSAIL adapter, `plan_emitted_lossy` means only that an outcome-blind adapter specification was emitted. It is not executable. If DataSAIL later returns an assignment, the compiler verifies plan integrity, validates fold labels, audits the assignment, and may return `assignment_audited_lossy`. The label describes a usable comparator under an acknowledged projection; it does not mean the full deployment contract is satisfied.

DataSAIL is a strong similarity-aware method with a different task interface. A gap reported by this adapter is a mismatch between the frozen adapter projection and the present contract, not a defect in DataSAIL.

## 8. Independent verification

The post-audit conformance suite includes an implementation-diverse oracle that imports neither production audit nor solver internals. It:

- verifies every `compiled_exact` assignment from raw provenance and a plain contract dictionary;
- exhaustively enumerates random small instances to obtain feasibility truth;
- independently verifies every analytic certificate;
- mutates hard entities, fold occupancy, relation support, joint support, incidence support, plan hashes, semantic gaps, and backend specifications;
- requires both the production audit and independent oracle to reject invalid assignments;
- preserves a fixed counterexample where the heuristic returns `backend_search_exhausted` although exhaustive search finds a feasible witness.

Repeated conformance cases are software stress tests. They are not independent observations from a scientific population and receive no binomial confidence interval in the manuscript.

## 9. Predictive evidence boundary

Compiler correctness does not imply predictive superiority. Predictive experiments separately estimate internal-to-external error, calibration, model-selection regret, runtime, and refusal frequency under frozen denominators. Comparator-specific pairing prevents one failed method from deleting valid comparisons among others.

An exact validation contract can still be uninformative if important deployment variables are unrecorded. Conversely, typed refusal states that the available development evidence cannot internally instantiate the declared target; it does not predict that every external model will fail.

## 10. Headed-stud application boundary

The headed-stud application is a source-audited external-parent evaluation. It is not a prospective blind field trial. The solid-slab development parent, profiled-deck parent, lightweight-concrete parent, and recycled-aggregate programme remain distinct. Shared experimental sources are quarantined without consulting resistance outcomes.

The application may show that a stronger deck-topology or concrete-family deployment cannot be represented internally, and may quantify the consequence of overriding that refusal with executable proxies. It cannot attribute the external error uniquely to topology, claim first natural-to-recycled aggregate transfer, or support population inference from a single recycled-aggregate programme.

## 11. Confirmation and downgrade rules

Any change to schema, status semantics, audit support rules, certificate logic, oracle, generator, endpoint, pairing, or multiplicity creates a new prespecification, source freeze, and output root before affected seeds are opened. Historical v2/v2.1/v2.1.1 outputs remain immutable audit artifacts and are not pooled with the post-audit version.

A single false-exact event, invalid certificate accepted, production-oracle disagreement, or undeclared outcome dependence blocks the compiler headline. Search exhaustion is reported as unresolved, never upgraded to infeasible. Predictive results cannot override a compiler hard-gate failure.
