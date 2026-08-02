"""Freeze-time smoke check for the independent contract oracle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHOD_ROOT = Path(__file__).resolve().parent
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
    DeploymentEstimand,
    RelationTarget,
    ValidationContractCompiler,
)


def estimand() -> DeploymentEstimand:
    return DeploymentEstimand(
        prediction_unit="oracle test record",
        outcome="continuous test response",
        loss="squared_error",
        target_population="freeze-time oracle corpus",
        target_weighting="record_weighted",
        decision_estimand="contract realization status",
        metadata_source="prespecified",
    )


def main() -> None:
    feasible = pd.DataFrame({"lab": ["A", "A", "B", "B", "C", "C"]})
    feasible_claim = DeploymentClaim(
        estimand=estimand(),
        relations=(RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),),
        n_splits=3,
        min_fold_fraction=0.30,
        max_fold_fraction=0.36,
        strict_metadata=True,
    )
    feasible_result = ValidationContractCompiler(feasible_claim).compile(feasible)
    feasible_audit = verify_assignment(
        feasible,
        feasible_result.assignments,
        feasible_claim.to_dict(),
    )
    feasible_truth = brute_force_feasible(feasible, feasible_claim.to_dict())

    impossible = pd.DataFrame(
        {"lab": ["dominant"] * 4 + ["small_a", "small_b"]}
    )
    impossible_claim = DeploymentClaim(
        estimand=estimand(),
        relations=(RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),),
        n_splits=3,
        min_fold_fraction=0.15,
        max_fold_fraction=0.50,
        strict_metadata=True,
    )
    impossible_result = ValidationContractCompiler(impossible_claim).compile(
        impossible,
        backend="claimcut",
    )
    certificate = impossible_result.validation_card["infeasibility_certificate"]
    certificate_audit = verify_infeasibility_certificate(
        impossible,
        impossible_claim.to_dict(),
        certificate,
    )

    exhausted = pd.DataFrame(
        {
            "lab": ["L3", "L0", "L3", "L2", "L0", "L1", "L2", "L1"],
            "supplier": ["S1", "S0", "S0", "S1", "S2", "S1", "S1", "S1"],
        }
    )
    exhausted_claim = DeploymentClaim(
        estimand=estimand(),
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
        strict_metadata=True,
    )
    exhausted_result = ValidationContractCompiler(exhausted_claim).compile(
        exhausted,
        backend="claimcut",
    )
    exhausted_truth = brute_force_feasible(exhausted, exhausted_claim.to_dict())
    witness = np.asarray([0, 1, 0, 0, 1, 0, 0, 0], dtype=int)
    witness_audit = verify_assignment(
        exhausted,
        witness,
        exhausted_claim.to_dict(),
    )

    checks = {
        "compiled_exact_passes_oracle": bool(
            feasible_result.status == "compiled_exact"
            and feasible_audit["contract_satisfied"]
        ),
        "feasible_truth_found": bool(feasible_truth["feasible"]),
        "certificate_verified": bool(certificate_audit["valid"]),
        "certified_case_has_no_truth_witness": bool(
            not brute_force_feasible(impossible, impossible_claim.to_dict())["feasible"]
        ),
        "search_exhausted_not_infeasible": bool(
            exhausted_result.status == "backend_search_exhausted"
            and exhausted_truth["feasible"]
            and witness_audit["contract_satisfied"]
        ),
    }
    payload = {
        "schema_version": "independent-oracle-check-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "feasible_status": feasible_result.status,
        "impossible_status": impossible_result.status,
        "search_exhausted_status": exhausted_result.status,
        "certificate_audit": certificate_audit,
    }
    output = METHOD_ROOT / "independent_oracle_check_result.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

