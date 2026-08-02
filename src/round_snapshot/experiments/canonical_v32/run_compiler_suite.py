"""Resumable primary confirmatory suite for compiler semantics and refusal."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_batch import (  # local frozen experiment module
    PROVENANCE_COLUMNS,
    SCHEMA_VERSION,
    atomic_write_json,
    build_claim,
    create_confirmatory_claim,
    frozen_seed_table_sha256,
    replicate_seeds,
    run_replicate,
    scenario_parameters,
    verify_confirmatory_freeze,
)

ROUND_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = ROUND_ROOT / "method"
import sys

if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from provenance_cut import (  # noqa: E402
    DeploymentClaim,
    DeploymentEstimand,
    IncidenceTarget,
    JointPatternTarget,
    RelationTarget,
    ValidationContractCompiler,
)
from provenance_cut.audit import audit_assignment  # noqa: E402
from provenance_cut.synthetic import generate_contract_population_v2  # noqa: E402
from independent_contract_oracle import (  # noqa: E402
    brute_force_feasible,
    verify_assignment,
    verify_infeasibility_certificate,
)

SUITE_SCHEMA = "compiler-semantic-v3"


def _conformance_estimand() -> DeploymentEstimand:
    return DeploymentEstimand(
        prediction_unit="synthetic structural record",
        outcome="continuous structural response",
        loss="squared_error",
        target_population="frozen compiler conformance population",
        target_weighting="record_weighted",
        decision_estimand="contract realization status",
        metadata_source="prespecified",
    )


def _sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _joint_claim(patterns: tuple[tuple[str, float], ...]) -> DeploymentClaim:
    return DeploymentClaim(
        estimand=_conformance_estimand(),
        relations=(
            RelationTarget("lab", 0.5, mode="target", tolerance=0.10),
            RelationTarget("supplier", 0.5, mode="target", tolerance=0.10),
        ),
        joint_patterns=(
            JointPatternTarget(
                name="same_marginal_joint",
                relations=("lab", "supplier"),
                probabilities=patterns,
                tolerance=0.10,
            ),
        ),
        n_splits=5,
        strict_metadata=True,
    )


def _incidence_claim(degree: float) -> DeploymentClaim:
    return DeploymentClaim(
        estimand=_conformance_estimand(),
        relations=(
            RelationTarget("lab", 0.5, mode="target", tolerance=0.10),
            RelationTarget("supplier", 0.5, mode="target", tolerance=0.10),
        ),
        incidence_targets=(
            IncidenceTarget(
                name="same_marginal_incidence",
                left_relation="supplier",
                right_relation="lab",
                target_mean_degree=degree,
                tolerance=0.25,
            ),
        ),
        n_splits=5,
        strict_metadata=True,
    )


def _semantic_pair(
    provenance,
    left: DeploymentClaim,
    right: DeploymentClaim,
) -> dict[str, Any]:
    left_compiler = ValidationContractCompiler(left)
    right_compiler = ValidationContractCompiler(right)
    left_refused = left_compiler.compile(
        provenance,
        backend="datasail_c1e_scalar",
    )
    right_refused = right_compiler.compile(
        provenance,
        backend="datasail_c1e_scalar",
    )
    left_lossy = left_compiler.compile(
        provenance,
        backend="datasail_c1e_scalar",
        allow_lossy=True,
    )
    right_lossy = right_compiler.compile(
        provenance,
        backend="datasail_c1e_scalar",
        allow_lossy=True,
    )
    false_exact = int(left_refused.exact) + int(right_refused.exact)
    return {
        "left_contract_sha256": left_lossy.contract_sha256,
        "right_contract_sha256": right_lossy.contract_sha256,
        "contract_hashes_distinct": bool(
            left_lossy.contract_sha256 != right_lossy.contract_sha256
        ),
        "backend_specifications_identical": bool(
            left_lossy.backend_specification == right_lossy.backend_specification
        ),
        "plan_hashes_distinct": bool(
            left_lossy.plan_sha256 != right_lossy.plan_sha256
        ),
        "left_exact_status": left_refused.status,
        "right_exact_status": right_refused.status,
        "left_lossy_status": left_lossy.status,
        "right_lossy_status": right_lossy.status,
        "false_exact_events": false_exact,
        "semantic_gaps": sorted(
            set(left_lossy.semantic_gaps + right_lossy.semantic_gaps)
        ),
    }


def _native_realization_ok(result) -> bool:
    if result.status != "compiled_exact" or result.cut_result is None:
        return False
    cut = result.cut_result
    audit = cut.audit
    if not cut.feasible or not audit.get("contract_satisfied", False):
        return False
    if not all(
        item.get("satisfied", False)
        for item in audit.get("joint_patterns", {}).values()
    ):
        return False
    if not all(
        item.get("satisfied", False)
        for item in audit.get("incidence_targets", {}).values()
    ):
        return False
    return bool(
        cut.candidate_balance_satisfied
        and cut.candidate_targets_satisfied
    )


def _small_oracle_cases(seed: int) -> dict[str, Any]:
    """Compare compiler states with exhaustive truth on tiny instances."""

    rng = np.random.default_rng(seed)
    feasible_sizes = rng.integers(1, 4, size=3)
    labels = np.asarray(
        [
            label
            for label, size in zip(("A", "B", "C"), feasible_sizes)
            for _ in range(int(size))
        ],
        dtype=object,
    )
    rng.shuffle(labels)
    feasible_frame = pd.DataFrame({"lab": labels})
    feasible_claim = DeploymentClaim(
        estimand=_conformance_estimand(),
        relations=(
            RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),
        ),
        n_splits=3,
        random_state=seed,
        min_fold_fraction=0.10,
        max_fold_fraction=0.67,
        strict_metadata=True,
    )
    feasible_result = ValidationContractCompiler(feasible_claim).compile(
        feasible_frame,
        backend="group_kfold",
    )
    feasible_truth = brute_force_feasible(
        feasible_frame,
        feasible_claim.to_dict(),
    )

    dominant_n = int(rng.integers(4, 6))
    small_labels = np.asarray(
        ["dominant"] * dominant_n + ["small_a", "small_b"],
        dtype=object,
    )
    rng.shuffle(small_labels)
    impossible_frame = pd.DataFrame({"lab": small_labels})
    impossible_claim = DeploymentClaim(
        estimand=_conformance_estimand(),
        relations=(
            RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),
        ),
        n_splits=3,
        random_state=seed,
        min_fold_fraction=0.15,
        max_fold_fraction=0.50,
        strict_metadata=True,
    )
    impossible_result = ValidationContractCompiler(impossible_claim).compile(
        impossible_frame,
        backend="claimcut",
    )
    impossible_truth = brute_force_feasible(
        impossible_frame,
        impossible_claim.to_dict(),
    )
    certificate_check = verify_infeasibility_certificate(
        impossible_frame,
        impossible_claim.to_dict(),
        impossible_result.validation_card["infeasibility_certificate"],
    )
    agreement = bool(
        feasible_result.status == "compiled_exact"
        and feasible_truth["feasible"]
        and impossible_result.status == "certified_balance_infeasible"
        and not impossible_truth["feasible"]
        and certificate_check["valid"]
    )
    return {
        "agreement": agreement,
        "feasible_compiler_status": feasible_result.status,
        "feasible_oracle": feasible_truth,
        "impossible_compiler_status": impossible_result.status,
        "impossible_oracle": impossible_truth,
        "certificate_verification": certificate_check,
    }


def run_suite_replicate(replicate: int) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = replicate_seeds("COMPILER_SEMANTIC", replicate)
    params = scenario_parameters("S3")
    task = generate_contract_population_v2(
        seed=seeds["generator"],
        **params,
    )
    provenance = task.development.loc[:, PROVENANCE_COLUMNS]

    joint_pair = _semantic_pair(
        provenance,
        _joint_claim((("00", 0.5), ("11", 0.5))),
        _joint_claim((("01", 0.5), ("10", 0.5))),
    )
    incidence_pair = _semantic_pair(
        provenance,
        _incidence_claim(1.0),
        _incidence_claim(2.0),
    )

    native_claim = build_claim(
        split_seed=seeds["split"],
        target_supplier_degree=2.0,
    )
    compiler = ValidationContractCompiler(native_claim)
    native_first = compiler.compile(provenance, backend="claimcut")
    native_second = compiler.compile(provenance, backend="claimcut")
    if native_first.assignments is None:
        native_oracle = {
            "assignment_valid": False,
            "invalid_reason": "compiler_returned_no_assignment",
            "contract_satisfied": False,
        }
        mutation_detection = False
        mutation_oracles: dict[str, Any] = {}
    else:
        native_oracle = verify_assignment(
            provenance,
            native_first.assignments,
            native_claim.to_dict(),
        )
        assignment = native_first.assignments.copy()
        lab_values = provenance["lab"].reset_index(drop=True)
        first_lab_indices = np.flatnonzero(
            lab_values.to_numpy() == lab_values.iloc[0]
        )
        split_entity = assignment.copy()
        split_entity[first_lab_indices[0]] = int(
            (split_entity[first_lab_indices[0]] + 1) % native_claim.n_splits
        )
        empty_fold = assignment.copy()
        last_fold = int(native_claim.n_splits - 1)
        empty_fold[empty_fold == last_fold] = 0
        missing_joint = provenance.copy()
        missing_joint.loc[assignment == 0, "supplier"] = np.nan
        mutation_oracles = {
            "split_hard_entity": verify_assignment(
                provenance,
                split_entity,
                native_claim.to_dict(),
            ),
            "empty_fold": verify_assignment(
                provenance,
                empty_fold,
                native_claim.to_dict(),
            ),
            "missing_joint_support": verify_assignment(
                missing_joint,
                assignment,
                native_claim.to_dict(),
            ),
        }
        production_mutation_audits = {
            "split_hard_entity": audit_assignment(
                provenance,
                split_entity,
                native_claim,
            ),
            "empty_fold": audit_assignment(
                provenance,
                empty_fold,
                native_claim,
            ),
            "missing_joint_support": audit_assignment(
                missing_joint,
                assignment,
                native_claim,
            ),
        }
        mutation_detection = bool(
            all(
                not item.get("contract_satisfied", False)
                for item in mutation_oracles.values()
            )
        )
        production_mutation_detection = bool(
            all(
                not item.get("contract_satisfied", False)
                for item in production_mutation_audits.values()
            )
        )
        oracle_production_mutation_agreement = bool(
            all(
                bool(
                    mutation_oracles[name].get("contract_satisfied", False)
                )
                == bool(
                    production_mutation_audits[name].get(
                        "contract_satisfied",
                        False,
                    )
                )
                for name in mutation_oracles
            )
        )
    if native_first.assignments is None:
        production_mutation_audits = {}
        production_mutation_detection = False
        oracle_production_mutation_agreement = False
    small_oracle_cases = _small_oracle_cases(seeds["split"])

    y = task.development[task.target_column].to_numpy(dtype=float)
    rng = np.random.default_rng(seeds["learner"])
    transformations = {
        "original": y,
        "permuted": y[rng.permutation(len(y))],
        "negated": -y,
        "noise": rng.normal(size=len(y)),
        "rescaled": 7.0 * y + 13.0,
    }
    transformed_outcome_hashes = {
        name: _sha_array(values) for name, values in transformations.items()
    }
    outcome_plan_hashes = []
    outcome_assignment_hashes = []
    for name, transformed in transformations.items():
        outcome_probe = provenance.copy()
        outcome_probe["__OUTCOME_PROBE__"] = transformed
        compiled = compiler.compile(outcome_probe, backend="claimcut")
        outcome_plan_hashes.append(compiled.plan_sha256)
        outcome_assignment_hashes.append(
            compiled.validation_card.get("assignment_checksum")
        )

    refusal_outputs = {
        scenario: run_replicate(
            scenario=scenario,
            replicate=replicate,
            skip_datasail=True,
            datasail_max_sec=60,
        )
        for scenario in ("S4", "S8", "S10")
    }
    observed_refusals = {
        scenario: output["claimcut_status"]
        for scenario, output in refusal_outputs.items()
    }
    allowed = {
        "S4": {"backend_search_exhausted"},
        "S8": {"certified_balance_infeasible"},
        "S10": {"ordered_route_required"},
    }
    refusal_correct = {
        scenario: observed_refusals[scenario] in allowed[scenario]
        for scenario in allowed
    }
    interface_false_exact_events = int(joint_pair["false_exact_events"]) + int(
        incidence_pair["false_exact_events"]
    )
    oracle_false_exact_events = int(
        native_first.exact
        and not bool(native_oracle.get("contract_satisfied", False))
    )
    false_exact_events = interface_false_exact_events + oracle_false_exact_events
    return {
        "schema_version": SUITE_SCHEMA,
        "risk_output_schema_version": SCHEMA_VERSION,
        "replicate": int(replicate),
        "seeds": seeds,
        "seed_table_sha256": frozen_seed_table_sha256(),
        "joint_pair": joint_pair,
        "incidence_pair": incidence_pair,
        "false_exact_events": false_exact_events,
        "interface_false_exact_events": interface_false_exact_events,
        "oracle_false_exact_events": oracle_false_exact_events,
        "native_status": native_first.status,
        "native_realization_ok": _native_realization_ok(native_first),
        "independent_oracle": native_oracle,
        "independent_oracle_agreement": bool(
            native_first.exact
            and native_oracle.get("contract_satisfied", False)
        ),
        "mutation_oracles": mutation_oracles,
        "mutation_detection": mutation_detection,
        "production_mutation_audits": production_mutation_audits,
        "production_mutation_detection": production_mutation_detection,
        "oracle_production_mutation_agreement": (
            oracle_production_mutation_agreement
        ),
        "small_oracle_cases": small_oracle_cases,
        "bruteforce_status_agreement": bool(small_oracle_cases["agreement"]),
        "plan_reproducible": bool(
            native_first.plan_sha256 == native_second.plan_sha256
        ),
        "assignment_reproducible": bool(
            native_first.validation_card.get("assignment_checksum")
            == native_second.validation_card.get("assignment_checksum")
        ),
        "outcome_invariant": bool(
            len(set(outcome_plan_hashes)) == 1
            and len(set(outcome_assignment_hashes)) == 1
        ),
        "outcome_hashes_are_distinct": bool(
            len(set(transformed_outcome_hashes.values()))
            == len(transformed_outcome_hashes)
        ),
        "transformed_outcome_sha256": transformed_outcome_hashes,
        "observed_refusals": observed_refusals,
        "refusal_correct": refusal_correct,
        "all_refusals_correct": bool(all(refusal_correct.values())),
        "native_validation_card": native_first.validation_card,
        "runtime_seconds": float(time.perf_counter() - started),
        "status": "complete",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--confirmatory", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.confirmatory and args.overwrite:
        parser.error("--overwrite is forbidden in --confirmatory mode")
    if args.confirmatory:
        expected = (
            Path(__file__).resolve().parent
            / "confirmatory_v3"
            / "compiler_semantic"
        ).resolve()
        if output != expected:
            parser.error(f"--confirmatory output must be {expected}")
    output.mkdir(parents=True, exist_ok=True)
    for replicate in range(args.start, args.start + args.count):
        if args.confirmatory:
            verify_confirmatory_freeze("COMPILER_SEMANTIC", replicate)
        path = output / f"rep_{replicate:04d}.json"
        if path.exists() and args.confirmatory:
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"unreadable confirmatory result must not be replaced: {path}"
                ) from exc
            if existing.get("status") == "failed":
                raise RuntimeError(
                    f"preserved confirmatory failure requires a new output root: {path}"
                )
            print(f"SKIP_PRESERVED {path}")
            continue
        if path.exists() and not args.overwrite:
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("status") == "complete":
                    print(f"SKIP {path}")
                    continue
            except Exception:
                pass
        if args.confirmatory:
            create_confirmatory_claim(
                path,
                scenario="COMPILER_SEMANTIC",
                replicate=replicate,
            )
        try:
            payload = run_suite_replicate(replicate)
        except Exception as exc:
            payload = {
                "schema_version": SUITE_SCHEMA,
                "replicate": int(replicate),
                "seeds": replicate_seeds("COMPILER_SEMANTIC", replicate),
                "seed_table_sha256": frozen_seed_table_sha256(),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        atomic_write_json(path, payload)
        print(f"{payload['status'].upper()} replicate={replicate} output={path}")
        if args.confirmatory and payload["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
