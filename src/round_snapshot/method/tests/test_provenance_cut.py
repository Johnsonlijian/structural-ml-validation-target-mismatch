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
    ClaimConditionedProvenanceCut,
    DeploymentClaim,
    IncidenceTarget,
    JointPatternTarget,
    RelationTarget,
)
from provenance_cut.audit import audit_assignment  # noqa: E402
from provenance_cut.synthetic import generate_partial_supplier_deployment  # noqa: E402


def make_crossed_provenance(n_labs: int = 12, records_per_lab: int = 8) -> pd.DataFrame:
    rows = []
    for lab in range(n_labs):
        for record in range(records_per_lab):
            rows.append(
                {
                    "lab": f"L{lab:02d}",
                    "source": f"S{lab:02d}_{record // 4}",
                    "family": f"F{record % 2}",
                    "material": f"M{(lab + record) % 3}",
                }
            )
    return pd.DataFrame(rows)


class ProvenanceCutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provenance = make_crossed_provenance()
        self.claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, mode="hard_unseen"),
                RelationTarget("source", 1.0, mode="hard_unseen"),
                RelationTarget("family", 0.0, mode="support_seen"),
                RelationTarget("material", 0.0, mode="support_seen"),
            ),
            n_splits=4,
            random_state=42,
            n_starts=6,
            max_iter=20,
            min_fold_fraction=0.15,
            max_fold_fraction=0.35,
        )

    def test_hard_entities_never_span_folds(self) -> None:
        result = ClaimConditionedProvenanceCut(self.claim).assign(self.provenance)
        self.assertTrue(result.feasible)
        self.assertTrue(result.audit["hard_constraints_satisfied"])
        for relation in ("lab", "source"):
            leakage = result.audit["relation_entity_leakage"][relation]
            self.assertEqual(leakage["n_spanning_entities"], 0)
            for fold in range(self.claim.n_splits):
                novelty = result.audit["novelty"][relation][f"fold_{fold}"]
                self.assertAlmostEqual(float(novelty), 1.0)

    def test_seen_support_relations_are_represented_in_training(self) -> None:
        result = ClaimConditionedProvenanceCut(self.claim).assign(self.provenance)
        for relation in ("family", "material"):
            values = [
                result.audit["novelty"][relation][f"fold_{fold}"]
                for fold in range(self.claim.n_splits)
            ]
            self.assertLessEqual(max(float(value) for value in values), 0.05)

    def test_assignment_is_deterministic(self) -> None:
        first = ClaimConditionedProvenanceCut(self.claim).assign(self.provenance)
        second = ClaimConditionedProvenanceCut(self.claim).assign(self.provenance)
        np.testing.assert_array_equal(first.assignments, second.assignments)
        self.assertEqual(
            first.audit["assignment_checksum"],
            second.audit["assignment_checksum"],
        )

    def test_sklearn_split_contract(self) -> None:
        splitter = ClaimConditionedProvenanceCut(self.claim)
        covered = np.zeros(len(self.provenance), dtype=int)
        for train, test in splitter.split(np.zeros((len(self.provenance), 1)), groups=self.provenance):
            self.assertEqual(len(set(train).intersection(test)), 0)
            covered[test] += 1
        np.testing.assert_array_equal(covered, np.ones(len(self.provenance), dtype=int))

    def test_infeasible_claim_is_reported(self) -> None:
        provenance = self.provenance.copy()
        provenance["lab"] = "one_connected_lab"
        claim = DeploymentClaim(
            relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
            n_splits=3,
        )
        splitter = ClaimConditionedProvenanceCut(claim)
        result = splitter.assign(provenance)
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, "certified_partition_infeasible")
        self.assertIn("connected components", result.diagnostics["reason"])
        with self.assertRaises(ValueError):
            list(splitter.split(np.zeros((len(provenance), 1)), groups=provenance))

    def test_balance_infeasibility_is_typed_and_refused(self) -> None:
        provenance = pd.DataFrame(
            {
                "lab": (
                    ["dominant"] * 70
                    + ["small_1"] * 10
                    + ["small_2"] * 10
                    + ["small_3"] * 10
                )
            }
        )
        claim = DeploymentClaim(
            relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
            n_splits=4,
            min_fold_fraction=0.10,
            max_fold_fraction=0.40,
            n_starts=3,
            max_iter=5,
        )
        result = ClaimConditionedProvenanceCut(claim).assign(provenance)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.candidate_hard_satisfied)
        self.assertIsNone(result.candidate_balance_satisfied)
        self.assertEqual(result.feasibility_status, "certified_infeasible")
        self.assertEqual(result.status, "certified_balance_infeasible")
        self.assertEqual(
            result.diagnostics["certificate"]["kind"],
            "largest_hard_component_exceeds_fold_upper_bound",
        )

    def test_unattained_soft_target_is_typed_and_refused(self) -> None:
        rows = []
        for lab in range(8):
            rows.extend(
                {"lab": f"L{lab}", "supplier": f"UNIQUE_{lab}"}
                for _ in range(5)
            )
        provenance = pd.DataFrame(rows)
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, mode="hard_unseen"),
                RelationTarget(
                    "supplier",
                    0.0,
                    mode="support_seen",
                    tolerance=0.05,
                ),
            ),
            n_splits=4,
            n_starts=4,
            max_iter=10,
            min_fold_fraction=0.15,
            max_fold_fraction=0.35,
        )
        result = ClaimConditionedProvenanceCut(claim).assign(provenance)
        self.assertFalse(result.feasible)
        self.assertTrue(result.candidate_hard_satisfied)
        self.assertTrue(result.candidate_balance_satisfied)
        self.assertFalse(result.candidate_targets_satisfied)
        self.assertEqual(result.feasibility_status, "unresolved")
        self.assertEqual(result.status, "backend_search_exhausted")
        self.assertEqual(
            result.diagnostics["candidate_status"],
            "realized_plan_invalid_targets",
        )

    def test_ordered_claim_requires_forward_split(self) -> None:
        provenance = pd.DataFrame({"time": np.arange(20)})
        claim = DeploymentClaim(
            relations=(RelationTarget("time", 1.0, mode="target"),),
            n_splits=4,
            ordered_relation="time",
        )
        result = ClaimConditionedProvenanceCut(claim).assign(provenance)
        self.assertFalse(result.feasible)
        self.assertEqual(result.status, "ordered_route_required")

    def test_joint_pattern_and_incidence_contract(self) -> None:
        task = generate_partial_supplier_deployment(
            seed=7,
            n_development_labs=10,
            n_external_labs=2,
            sources_per_supplier=1,
            records_per_source=4,
        )
        columns = (
            "lab",
            "supplier",
            "source",
            "campaign",
            "material_family",
            "structural_family",
        )
        provenance = task.development.loc[:, columns]
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, mode="hard_unseen"),
                RelationTarget("supplier", 0.5, weight=2.0, mode="target"),
                RelationTarget("source", 1.0, mode="target"),
                RelationTarget("campaign", 1.0, mode="target"),
                RelationTarget("material_family", 0.0, mode="support_seen"),
                RelationTarget("structural_family", 0.0, mode="support_seen"),
            ),
            joint_patterns=(
                JointPatternTarget(
                    name="lab_supplier_joint",
                    relations=("lab", "supplier"),
                    probabilities=(("10", 0.5), ("11", 0.5)),
                    tolerance=0.01,
                ),
            ),
            incidence_targets=(
                IncidenceTarget(
                    name="unseen_supplier_lab_degree",
                    left_relation="supplier",
                    right_relation="lab",
                    target_mean_degree=2.0,
                    condition_relation="supplier",
                    condition_novelty=1,
                    tolerance=0.01,
                ),
            ),
            n_splits=5,
            random_state=7,
            n_starts=12,
            max_iter=30,
            min_fold_fraction=0.15,
            max_fold_fraction=0.25,
        )
        result = ClaimConditionedProvenanceCut(claim).assign(provenance)
        self.assertTrue(result.feasible)
        self.assertEqual(result.status, "feasible")
        joint = result.audit["joint_patterns"]["lab_supplier_joint"]
        incidence = result.audit["incidence_targets"]["unseen_supplier_lab_degree"]
        self.assertTrue(joint["satisfied"])
        self.assertTrue(incidence["satisfied"])

    def test_joint_pattern_with_empty_fold_is_not_satisfied(self) -> None:
        provenance = pd.DataFrame(
            {
                "lab": ["L0", "L1", None, None],
                "supplier": ["S0", "S1", None, None],
            }
        )
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 0.5, tolerance=1.0),
                RelationTarget("supplier", 0.5, tolerance=1.0),
            ),
            joint_patterns=(
                JointPatternTarget(
                    name="joint",
                    relations=("lab", "supplier"),
                    probabilities=(("00", 1.0),),
                    tolerance=1.0,
                ),
            ),
            n_splits=2,
            min_fold_fraction=0.25,
            max_fold_fraction=0.75,
        )
        audit = audit_assignment(
            provenance,
            np.asarray([0, 0, 1, 1], dtype=int),
            claim,
        )
        joint = audit["joint_patterns"]["joint"]
        self.assertFalse(joint["all_folds_evaluable"])
        self.assertEqual(joint["n_evaluable_folds"], 1)
        self.assertFalse(joint["satisfied"])
        self.assertFalse(audit["contract_satisfied"])

    def test_relation_target_with_empty_fold_is_not_satisfied(self) -> None:
        provenance = pd.DataFrame({"relation": [None, None, "a", "b"]})
        claim = DeploymentClaim(
            relations=(
                RelationTarget(
                    "relation",
                    1.0,
                    tolerance=0.0,
                ),
            ),
            n_splits=2,
            min_fold_fraction=0.25,
            max_fold_fraction=0.75,
        )
        audit = audit_assignment(
            provenance,
            np.asarray([0, 0, 1, 1], dtype=int),
            claim,
        )
        support = audit["relation_target_support"]["relation"]
        self.assertEqual(support["n_observed_by_fold"], [0, 2])
        self.assertFalse(support["all_folds_evaluable"])
        self.assertFalse(audit["relation_targets_satisfied"])
        self.assertFalse(audit["contract_satisfied"])

    def test_conditioned_empty_incidence_is_not_satisfied(self) -> None:
        provenance = pd.DataFrame(
            {
                "supplier": ["S0", "S1", "S0", "S1"],
                "lab": ["L0", "L1", "L2", "L3"],
                "source": ["A", "B", "C", "D"],
            }
        )
        claim = DeploymentClaim(
            relations=(
                RelationTarget("supplier", 0.0, tolerance=1.0),
                RelationTarget("lab", 0.5, tolerance=1.0),
                RelationTarget("source", 0.5, tolerance=1.0),
            ),
            incidence_targets=(
                IncidenceTarget(
                    name="conditioned_degree",
                    left_relation="supplier",
                    right_relation="lab",
                    target_mean_degree=1.0,
                    tolerance=10.0,
                    condition_relation="supplier",
                    condition_novelty=1,
                ),
            ),
            n_splits=2,
            min_fold_fraction=0.25,
            max_fold_fraction=0.75,
        )
        audit = audit_assignment(
            provenance,
            np.asarray([0, 0, 1, 1], dtype=int),
            claim,
        )
        incidence = audit["incidence_targets"]["conditioned_degree"]
        self.assertFalse(incidence["all_folds_evaluable"])
        self.assertEqual(incidence["n_evaluable_folds"], 0)
        self.assertFalse(incidence["satisfied"])
        self.assertFalse(audit["contract_satisfied"])

    def test_minimum_support_threshold_is_enforced(self) -> None:
        provenance = pd.DataFrame(
            {
                "lab": ["L0", "L1", "L2", "L3"],
                "supplier": ["S0", "S1", "S2", "S3"],
            }
        )
        claim = DeploymentClaim(
            relations=(
                RelationTarget("lab", 1.0, tolerance=1.0),
                RelationTarget("supplier", 1.0, tolerance=1.0),
            ),
            joint_patterns=(
                JointPatternTarget(
                    name="joint_minimum",
                    relations=("lab", "supplier"),
                    probabilities=(("11", 1.0),),
                    tolerance=1.0,
                    min_observations_per_fold=3,
                ),
            ),
            n_splits=2,
            min_fold_fraction=0.25,
            max_fold_fraction=0.75,
        )
        audit = audit_assignment(
            provenance,
            np.asarray([0, 0, 1, 1], dtype=int),
            claim,
        )
        joint = audit["joint_patterns"]["joint_minimum"]
        self.assertEqual(joint["n_evaluable_folds"], 0)
        self.assertFalse(joint["satisfied"])

    def test_duplicate_complex_target_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "joint-pattern target names"):
            DeploymentClaim(
                relations=(
                    RelationTarget("lab", 0.5),
                    RelationTarget("supplier", 0.5),
                ),
                joint_patterns=(
                    JointPatternTarget(
                        name="duplicate",
                        relations=("lab", "supplier"),
                        probabilities=(("00", 1.0),),
                    ),
                    JointPatternTarget(
                        name="duplicate",
                        relations=("lab", "supplier"),
                        probabilities=(("11", 1.0),),
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
