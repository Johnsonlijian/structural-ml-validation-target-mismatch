from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ROOT = Path(__file__).resolve().parents[1]
CV1_ROOT = METHOD_ROOT.parent / "experiments" / "cv1"
for path in (METHOD_ROOT, CV1_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from independent_contract_oracle import (  # noqa: E402
    brute_force_feasible,
    verify_assignment,
    verify_infeasibility_certificate,
)
from provenance_cut import (  # noqa: E402
    DeploymentClaim,
    IncidenceTarget,
    JointPatternTarget,
    RelationTarget,
    ValidationContractCompiler,
)


def test_compiled_exact_assignment_passes_independent_oracle() -> None:
    frame = pd.DataFrame(
        {
            "lab": ["A", "A", "B", "B", "C", "C"],
        }
    )
    claim = DeploymentClaim(
        relations=(RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),),
        n_splits=3,
        random_state=11,
        min_fold_fraction=0.30,
        max_fold_fraction=0.36,
    )
    result = ValidationContractCompiler(claim).compile(
        frame,
        backend="group_kfold",
    )
    assert result.status == "compiled_exact"
    oracle = verify_assignment(frame, result.assignments, claim.to_dict())
    assert oracle["contract_satisfied"]


def test_mutations_are_rejected_by_independent_oracle() -> None:
    frame = pd.DataFrame(
        {
            "lab": ["A", "A", "B", "B", "C", "C"],
        }
    )
    claim = DeploymentClaim(
        relations=(RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),),
        n_splits=3,
        random_state=11,
        min_fold_fraction=0.30,
        max_fold_fraction=0.36,
    )
    result = ValidationContractCompiler(claim).compile(frame)
    broken_group = result.assignments.copy()
    broken_group[0] = int((broken_group[0] + 1) % claim.n_splits)
    assert not verify_assignment(
        frame,
        broken_group,
        claim.to_dict(),
    )["contract_satisfied"]

    empty_fold = result.assignments.copy()
    empty_fold[empty_fold == 2] = 1
    invalid = verify_assignment(frame, empty_fold, claim.to_dict())
    assert not invalid["assignment_valid"]
    assert invalid["invalid_reason"] == "empty_fold"


def test_brute_force_truth_agrees_on_feasible_and_certified_cases() -> None:
    feasible_frame = pd.DataFrame({"lab": ["A", "A", "B", "B", "C", "C"]})
    feasible_claim = DeploymentClaim(
        relations=(RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),),
        n_splits=3,
        min_fold_fraction=0.30,
        max_fold_fraction=0.36,
    )
    truth = brute_force_feasible(feasible_frame, feasible_claim.to_dict())
    assert truth["feasible"]

    impossible_frame = pd.DataFrame(
        {"lab": ["dominant"] * 4 + ["small_a", "small_b"]}
    )
    impossible_claim = DeploymentClaim(
        relations=(RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),),
        n_splits=3,
        min_fold_fraction=0.15,
        max_fold_fraction=0.50,
    )
    result = ValidationContractCompiler(impossible_claim).compile(
        impossible_frame,
        backend="claimcut",
    )
    assert result.status == "certified_balance_infeasible"
    truth = brute_force_feasible(impossible_frame, impossible_claim.to_dict())
    assert not truth["feasible"]


def test_search_exhausted_is_not_mislabeled_infeasible() -> None:
    frame = pd.DataFrame(
        {
            "lab": ["L3", "L0", "L3", "L2", "L0", "L1", "L2", "L1"],
            "supplier": ["S1", "S0", "S0", "S1", "S2", "S1", "S1", "S1"],
        }
    )
    claim = DeploymentClaim(
        relations=(
            RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),
            RelationTarget("supplier", 0.75, mode="target", tolerance=0.26),
        ),
        n_splits=2,
        random_state=0,
        n_starts=1,
        max_iter=1,
        min_fold_fraction=0.25,
        max_fold_fraction=0.75,
    )
    result = ValidationContractCompiler(claim).compile(
        frame,
        backend="claimcut",
    )
    truth = brute_force_feasible(frame, claim.to_dict())
    witness = np.asarray([0, 1, 0, 0, 1, 0, 0, 0], dtype=int)

    assert result.status == "backend_search_exhausted"
    assert result.cut_result.feasibility_status == "unresolved"
    assert truth["feasible"]
    assert verify_assignment(
        frame,
        witness,
        claim.to_dict(),
    )["contract_satisfied"]


def test_certified_infeasibility_has_independently_verifiable_binding() -> None:
    frame = pd.DataFrame(
        {"lab": ["dominant"] * 4 + ["small_a", "small_b"]}
    )
    claim = DeploymentClaim(
        relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
        n_splits=3,
        min_fold_fraction=0.15,
        max_fold_fraction=0.50,
    )
    result = ValidationContractCompiler(claim).compile(
        frame,
        backend="claimcut",
    )
    certificate = result.validation_card["infeasibility_certificate"]
    verification = verify_infeasibility_certificate(
        frame,
        claim.to_dict(),
        certificate,
    )
    assert verification["valid"]

    tampered = dict(certificate)
    tampered["provenance_sha256"] = "0" * 64
    assert not verify_infeasibility_certificate(
        frame,
        claim.to_dict(),
        tampered,
    )["valid"]


def test_oracle_rejects_empty_joint_and_conditioned_incidence_support() -> None:
    frame = pd.DataFrame(
        {
            "lab": ["L0", "L1", None, None],
            "supplier": ["S0", "S1", None, None],
            "source": ["A", "B", None, None],
        }
    )
    claim = DeploymentClaim(
        relations=(
            RelationTarget("lab", 0.5, tolerance=1.0),
            RelationTarget("supplier", 0.5, tolerance=1.0),
            RelationTarget("source", 0.5, tolerance=1.0),
        ),
        joint_patterns=(
            JointPatternTarget(
                name="joint",
                relations=("lab", "supplier"),
                probabilities=(("00", 1.0),),
                tolerance=1.0,
            ),
        ),
        incidence_targets=(
            IncidenceTarget(
                name="conditioned",
                left_relation="supplier",
                right_relation="lab",
                target_mean_degree=1.0,
                tolerance=10.0,
                condition_relation="source",
                condition_novelty=0,
            ),
        ),
        n_splits=2,
        min_fold_fraction=0.25,
        max_fold_fraction=0.75,
    )
    oracle = verify_assignment(
        frame,
        np.asarray([0, 0, 1, 1], dtype=int),
        claim.to_dict(),
    )
    assert not oracle["joint_patterns_satisfied"]
    assert not oracle["incidence_targets_satisfied"]
    assert not oracle["contract_satisfied"]
