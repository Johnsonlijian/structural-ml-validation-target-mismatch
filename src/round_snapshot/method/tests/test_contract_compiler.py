from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ROOT = Path(__file__).resolve().parents[1]
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from provenance_cut import (  # noqa: E402
    DeploymentClaim,
    DeploymentEstimand,
    JointPatternTarget,
    RelationTarget,
    ValidationContractCompiler,
)


def provenance_frame() -> pd.DataFrame:
    rows = []
    for lab in range(12):
        for record in range(6):
            rows.append(
                {
                    "lab": f"L{lab:02d}",
                    "supplier": f"S{lab // 2:02d}",
                    "family": f"F{record % 2}",
                }
            )
    return pd.DataFrame(rows)


class ContractCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provenance = provenance_frame()

    def test_auto_specializes_single_hard_relation_to_group_kfold(self) -> None:
        claim = DeploymentClaim(
            relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
            n_splits=4,
            random_state=11,
        )
        result = ValidationContractCompiler(claim).compile(self.provenance)
        self.assertTrue(result.exact)
        self.assertEqual(result.backend, "group_kfold")
        self.assertIsNotNone(result.assignments)
        self.assertTrue(result.validation_card["outcome_blind_compilation"])
        self.assertIsNotNone(result.validation_card["assignment_checksum"])

    def test_estimand_is_hashed_and_emitted_in_validation_card(self) -> None:
        first = DeploymentClaim(
            relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
            estimand=DeploymentEstimand(
                prediction_unit="headed stud",
                outcome="ultimate shear resistance per stud",
                loss="squared_error",
                target_population="profiled-deck push tests",
                target_weighting="equal_domain",
                decision_estimand="mean source-balanced prediction error",
                domain_relation="lab",
                metadata_source="author_declared",
            ),
            n_splits=4,
        )
        second = DeploymentClaim(
            relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
            estimand=DeploymentEstimand(
                prediction_unit="headed stud",
                outcome="ultimate shear resistance per stud",
                loss="absolute_error",
                target_population="profiled-deck push tests",
                target_weighting="equal_domain",
                decision_estimand="mean source-balanced prediction error",
                domain_relation="lab",
                metadata_source="author_declared",
            ),
            n_splits=4,
        )
        first_result = ValidationContractCompiler(first).compile(self.provenance)
        second_result = ValidationContractCompiler(second).compile(self.provenance)
        self.assertNotEqual(
            first_result.contract_sha256,
            second_result.contract_sha256,
        )
        self.assertEqual(
            first_result.validation_card["estimand"]["prediction_unit"],
            "headed stud",
        )

    def test_strict_metadata_rejects_legacy_defaults(self) -> None:
        with self.assertRaisesRegex(ValueError, "strict_metadata"):
            DeploymentClaim(
                relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
                strict_metadata=True,
            )

    def test_native_backend_compiles_full_contract_exactly(self) -> None:
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, mode="hard_unseen"),
                RelationTarget("family", 0.0, mode="support_seen"),
            ),
            n_splits=4,
            random_state=17,
            n_starts=6,
            max_iter=20,
            min_fold_fraction=0.15,
            max_fold_fraction=0.35,
        )
        result = ValidationContractCompiler(claim).compile(
            self.provenance,
            backend="claimcut",
        )
        self.assertEqual(result.status, "compiled_exact")
        self.assertEqual(result.backend, "claimcut")
        self.assertTrue(result.cut_result.feasible)
        result.require_exact()

    def test_group_backend_cannot_be_exact_when_fold_bounds_fail(self) -> None:
        frame = pd.DataFrame(
            {"source": ["dominant"] * 70 + ["a"] * 10 + ["b"] * 10 + ["c"] * 10}
        )
        claim = DeploymentClaim(
            relations=(RelationTarget("source", 1.0, mode="hard_unseen"),),
            n_splits=4,
            random_state=19,
            min_fold_fraction=0.10,
            max_fold_fraction=0.40,
        )

        result = ValidationContractCompiler(claim).compile(
            frame,
            backend="group_kfold",
        )

        self.assertEqual(result.status, "realized_plan_invalid_balance")
        self.assertFalse(result.exact)
        self.assertFalse(result.cut_result.feasible)
        self.assertFalse(result.cut_result.audit["fold_bounds_satisfied"])
        self.assertFalse(result.cut_result.audit["contract_satisfied"])

    def test_certified_refusal_does_not_report_assignment_returned(self) -> None:
        frame = pd.DataFrame(
            {"source": ["dominant"] * 70 + ["a"] * 10 + ["b"] * 10 + ["c"] * 10}
        )
        claim = DeploymentClaim(
            relations=(RelationTarget("source", 1.0, mode="hard_unseen"),),
            n_splits=4,
            max_fold_fraction=0.40,
        )
        result = ValidationContractCompiler(claim).compile(
            frame,
            backend="claimcut",
        )
        self.assertEqual(result.status, "certified_balance_infeasible")
        self.assertFalse(result.assignment_returned)
        self.assertIsNone(result.assignments)
        self.assertIsNotNone(result.cut_result)

    def test_lossy_group_backend_is_not_mislabeled_exact(self) -> None:
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, mode="hard_unseen"),
                RelationTarget("supplier", 1.0, mode="hard_unseen"),
            ),
            n_splits=4,
            random_state=23,
            min_fold_fraction=0.15,
            max_fold_fraction=0.35,
        )

        result = ValidationContractCompiler(claim).compile(
            self.provenance,
            backend="group_kfold",
            allow_lossy=True,
        )

        self.assertEqual(result.status, "assignment_audited_lossy")
        self.assertFalse(result.exact)
        self.assertTrue(result.executable)
        self.assertTrue(result.cut_result.feasible)
        self.assertFalse(result.cut_result.audit["contract_satisfied"])

    def test_scalar_backend_exposes_semantic_loss(self) -> None:
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, mode="hard_unseen"),
                RelationTarget("supplier", 0.5, mode="target"),
            ),
            joint_patterns=(
                JointPatternTarget(
                    name="lab_supplier",
                    relations=("lab", "supplier"),
                    probabilities=(("10", 0.5), ("11", 0.5)),
                ),
            ),
            n_splits=4,
        )
        compiler = ValidationContractCompiler(claim)
        refused = compiler.compile(
            self.provenance,
            backend="datasail_c1e_scalar",
        )
        self.assertEqual(refused.status, "backend_semantic_gap")
        with self.assertRaises(ValueError):
            refused.require_exact()
        lossy = compiler.compile(
            self.provenance,
            backend="datasail_c1e_scalar",
            allow_lossy=True,
        )
        self.assertEqual(lossy.status, "plan_emitted_lossy")
        self.assertIn(
            "joint_pattern_targets_not_representable",
            lossy.semantic_gaps,
        )
        self.assertIsNone(lossy.assignments)
        self.assertFalse(lossy.executable)
        self.assertFalse(lossy.assignment_returned)
        self.assertFalse(lossy.assignment_audited)

        exact_source = ValidationContractCompiler(
            DeploymentClaim(
                relations=(
                    RelationTarget("lab", 1.0, mode="hard_unseen"),
                ),
                n_splits=4,
                random_state=7,
            )
        ).compile(self.provenance, backend="group_kfold")
        audited = compiler.audit_emitted_assignment(
            self.provenance,
            lossy,
            exact_source.assignments,
        )
        self.assertEqual(audited.status, "assignment_audited_lossy")
        self.assertTrue(audited.assignment_returned)
        self.assertTrue(audited.assignment_audited)
        self.assertTrue(audited.executable)

    def test_invalid_adapter_assignment_is_not_executable(self) -> None:
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, mode="hard_unseen"),
                RelationTarget("supplier", 0.5, mode="target"),
            ),
            n_splits=4,
        )
        compiler = ValidationContractCompiler(claim)
        plan = compiler.compile(
            self.provenance,
            backend="datasail_c1e_scalar",
            allow_lossy=True,
        )
        invalid = compiler.audit_emitted_assignment(
            self.provenance,
            plan,
            np.zeros(len(self.provenance), dtype=int),
        )
        self.assertEqual(invalid.status, "assignment_returned_invalid")
        self.assertFalse(invalid.executable)

    def test_tampered_lossy_plan_is_rejected_before_assignment_audit(self) -> None:
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, mode="hard_unseen"),
                RelationTarget("supplier", 0.5, mode="target"),
            ),
            n_splits=4,
        )
        compiler = ValidationContractCompiler(claim)
        plan = compiler.compile(
            self.provenance,
            backend="datasail_c1e_scalar",
            allow_lossy=True,
        )
        plan.backend_specification["epsilon"] = 0.99
        assignment = ValidationContractCompiler(
            DeploymentClaim(
                relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
                n_splits=4,
            )
        ).compile(self.provenance).assignments
        with self.assertRaisesRegex(ValueError, "backend specification"):
            compiler.audit_emitted_assignment(
                self.provenance,
                plan,
                assignment,
            )

    def test_equal_marginals_do_not_identify_joint_contract(self) -> None:
        relations = (
            RelationTarget("lab", 0.5, mode="target"),
            RelationTarget("supplier", 0.5, mode="target"),
        )
        claim_aligned = DeploymentClaim(
            relations=relations,
            joint_patterns=(
                JointPatternTarget(
                    name="joint",
                    relations=("lab", "supplier"),
                    probabilities=(("00", 0.5), ("11", 0.5)),
                ),
            ),
            n_splits=4,
        )
        claim_crossed = DeploymentClaim(
            relations=relations,
            joint_patterns=(
                JointPatternTarget(
                    name="joint",
                    relations=("lab", "supplier"),
                    probabilities=(("01", 0.5), ("10", 0.5)),
                ),
            ),
            n_splits=4,
        )
        first = ValidationContractCompiler(claim_aligned).compile(
            self.provenance,
            backend="datasail_c1e_scalar",
            allow_lossy=True,
        )
        second = ValidationContractCompiler(claim_crossed).compile(
            self.provenance,
            backend="datasail_c1e_scalar",
            allow_lossy=True,
        )
        self.assertNotEqual(first.contract_sha256, second.contract_sha256)
        self.assertEqual(
            first.backend_specification,
            second.backend_specification,
        )
        self.assertNotEqual(first.plan_sha256, second.plan_sha256)

    def test_ordered_contract_is_typed_before_backend_execution(self) -> None:
        frame = pd.DataFrame({"time": np.arange(24)})
        claim = DeploymentClaim(
            relations=(RelationTarget("time", 1.0, mode="target"),),
            ordered_relation="time",
            n_splits=4,
        )
        result = ValidationContractCompiler(claim).compile(
            frame,
            backend="claimcut",
        )
        self.assertEqual(result.status, "ordered_route_required")
        self.assertFalse(result.executable)


if __name__ == "__main__":
    unittest.main()
