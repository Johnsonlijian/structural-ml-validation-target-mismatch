"""Preflight tests for the active confirmatory runner contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

CV1_ROOT = Path(__file__).resolve().parents[2] / "experiments" / "cv1"
if str(CV1_ROOT) not in sys.path:
    sys.path.insert(0, str(CV1_ROOT))

from run_batch import (  # noqa: E402
    MASTER_SEED,
    assignments_from_splitter,
    atomic_write_json,
    baseline_group_values,
    create_confirmatory_claim,
    parse_datasail_solver_status,
    replicate_seeds,
    verify_frozen_seed_row,
)


def test_unknown_baseline_groups_are_type_stable_and_record_unique() -> None:
    values = pd.Series(["source_a", np.nan, 7, None, "source_b"])

    groups = baseline_group_values(values)

    assert groups.dtype.kind in {"U", "S"}
    assert len(np.unique(groups)) == len(groups)
    assert groups[1].startswith("__UNKNOWN_RECORD_")
    assert groups[3].startswith("__UNKNOWN_RECORD_")
    assert groups[1] != groups[3]


def test_missing_provenance_is_accepted_by_groupkfold_baseline() -> None:
    groups = baseline_group_values(
        pd.Series(["a", "a", np.nan, "b", "b", None, "c", "c"])
    )
    X = np.arange(16, dtype=float).reshape(8, 2)
    y = np.arange(8, dtype=float)

    assignments = assignments_from_splitter(
        GroupKFold(n_splits=3, shuffle=True, random_state=17),
        X,
        y,
        groups,
    )

    assert assignments.shape == (8,)
    assert set(assignments) == {0, 1, 2}


def test_datasail_solver_status_parser_is_conservative() -> None:
    assert (
        parse_datasail_solver_status(
            "WARNING: solution may be inaccurate", assignment_returned=True
        )
        == "optimal_inaccurate"
    )
    assert (
        parse_datasail_solver_status("status: user_limit", assignment_returned=True)
        == "user_limit"
    )
    assert (
        parse_datasail_solver_status("status: optimal", assignment_returned=True)
        == "optimal"
    )
    assert (
        parse_datasail_solver_status("opaque solver output", assignment_returned=True)
        == "solution_returned_status_unparsed"
    )
    assert (
        parse_datasail_solver_status("opaque solver output", assignment_returned=False)
        == "no_assignment_status_unparsed"
    )


def test_atomic_json_write_replaces_only_at_commit(tmp_path: Path) -> None:
    output = tmp_path / "rep_0000.json"
    atomic_write_json(output, {"status": "failed", "error": "frozen"})

    assert json.loads(output.read_text(encoding="utf-8"))["error"] == "frozen"
    assert not list(tmp_path.glob("*.tmp.*"))


def test_confirmatory_seed_must_be_an_exact_frozen_row() -> None:
    scenario = "S3"
    replicate = 4
    seeds = replicate_seeds(scenario, replicate)
    table = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "replicate": replicate,
                "master_seed": MASTER_SEED,
                **{f"{name}_seed": value for name, value in seeds.items()},
            }
        ]
    )

    verify_frozen_seed_row(table, scenario, replicate)

    try:
        verify_frozen_seed_row(table, scenario, replicate + 1)
    except RuntimeError as exc:
        assert "absent or duplicated" in str(exc)
    else:
        raise AssertionError("out-of-table confirmatory seed was accepted")


def test_confirmatory_claim_is_exclusive_and_persistent(tmp_path: Path) -> None:
    output = tmp_path / "rep_0003.json"

    claim = create_confirmatory_claim(
        output,
        scenario="S3",
        replicate=3,
    )

    assert claim.exists()
    payload = json.loads(claim.read_text(encoding="utf-8"))
    assert payload["scenario"] == "S3"
    assert payload["replicate"] == 3
    try:
        create_confirmatory_claim(output, scenario="S3", replicate=3)
    except RuntimeError as exc:
        assert "already claimed" in str(exc)
    else:
        raise AssertionError("duplicate confirmatory claim was accepted")
