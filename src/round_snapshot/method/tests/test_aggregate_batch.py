from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROUND_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROUND_ROOT / "experiments" / "cv1" / "aggregate_batch.py"
SPEC = importlib.util.spec_from_file_location("cv1_aggregate_batch", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
aggregate_batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate_batch)


def _row(replicate: int, model: str, method: str, value: float) -> dict:
    return {
        "scenario": "S3",
        "replicate": replicate,
        "model": model,
        "method": method,
        "ere_record_weighted": value,
    }


def test_comparator_specific_pairing_does_not_apply_global_dropna() -> None:
    rows = []
    for replicate in (0, 1):
        for model, reference in (("ridge", 0.10), ("svr", 0.20)):
            rows.append(_row(replicate, model, "ClaimCut", reference))
            rows.append(_row(replicate, model, "RandomKFold", reference + 0.05))
    for model, value in (("ridge", 0.09), ("svr", 0.19)):
        rows.append(_row(0, model, "GroupKFold_Supplier", value))

    pairs, coverage = aggregate_batch.build_comparator_pairs(pd.DataFrame(rows))

    counts = pairs.groupby("comparator").size().to_dict()
    assert counts["RandomKFold"] == 4
    assert counts["GroupKFold_Supplier"] == 2
    random_coverage = coverage.set_index("comparator").loc["RandomKFold"]
    supplier_coverage = coverage.set_index("comparator").loc[
        "GroupKFold_Supplier"
    ]
    datasail_coverage = coverage.set_index("comparator").loc[
        "DataSAIL_C1e_Scalar"
    ]
    assert random_coverage["model_pairing_coverage_vs_reference"] == 1.0
    assert supplier_coverage["model_pairing_coverage_vs_reference"] == 0.5
    assert datasail_coverage["pairing_status"] == "missing_comparator"


def test_bootstrap_uses_fixed_seven_comparator_bonferroni_family() -> None:
    rows = []
    for replicate, reference in ((0, 0.10), (1, 0.20), (2, 0.15)):
        rows.append(_row(replicate, "ridge", "ClaimCut", reference))
        rows.append(_row(replicate, "ridge", "RandomKFold", reference + 0.1))
    pairs, coverage = aggregate_batch.build_comparator_pairs(pd.DataFrame(rows))

    result = aggregate_batch.clustered_bonferroni_bootstrap(
        pairs,
        coverage,
        scenario="S3",
        n_bootstrap=500,
        seed=7,
    )

    assert list(result["comparator"]) == list(
        aggregate_batch.FROZEN_COMPARATORS
    )
    assert (result["bonferroni_family_size"] == 7).all()
    assert np.allclose(result["familywise_alpha"], 0.05)
    assert np.allclose(result["per_comparison_alpha"], 0.05 / 7)
    random_row = result.set_index("comparator").loc["RandomKFold"]
    datasail_row = result.set_index("comparator").loc["DataSAIL_C1e_Scalar"]
    assert random_row["n_cluster_replicates"] == 3
    assert np.isclose(random_row["observed_mean_difference"], -0.1)
    assert np.isclose(
        random_row["simultaneous_upper_one_sided_95"], -0.1
    )
    assert datasail_row["n_cluster_replicates"] == 0
    assert np.isnan(datasail_row["simultaneous_upper_one_sided_95"])


def test_status_denominators_and_datasail_solver_status_are_explicit() -> None:
    payloads = [
        {
            "source_file": "rep_0000.json",
            "scenario": "S3",
            "replicate": 0,
            "status": "complete",
            "method_model_rows": [{"method": "ClaimCut"}],
            "datasail_metadata": {
                "attempted": True,
                "assignment_returned": True,
                "solver_status": "optimal_inaccurate",
                "solver_time_limit_likely": True,
            },
        },
        {
            "source_file": "rep_0001.json",
            "scenario": "S3",
            "replicate": 1,
            "status": "complete",
            "claimcut_status": "backend_search_exhausted",
            "expected_status": ["backend_search_exhausted"],
        },
        {
            "source_file": "rep_0002.json",
            "scenario": "S3",
            "replicate": 2,
            "status": "failed",
            "error": "boom",
        },
    ]
    statuses = pd.DataFrame(
        [aggregate_batch.status_record(payload) for payload in payloads]
    )
    summary = aggregate_batch.summarize_replicate_outcomes(statuses).iloc[0]
    solver = aggregate_batch.summarize_datasail_status(statuses)
    claimcut = aggregate_batch.summarize_claimcut_outcomes(statuses).iloc[0]

    assert summary["attempted"] == 3
    assert summary["success"] == 1
    assert summary["refusal"] == 1
    assert summary["failure"] == 1
    assert claimcut["attempted"] == 3
    assert claimcut["success"] == 1
    assert claimcut["refusal"] == 1
    assert claimcut["failure"] == 1
    assert "optimal_inaccurate" in set(solver["datasail_solver_status"])
    assert "missing" in set(solver["datasail_solver_status"])


def test_claimcut_refusal_is_retained_when_baseline_risk_rows_exist() -> None:
    payload = {
        "scenario": "S9_missing10",
        "replicate": 0,
        "status": "complete",
        "claimcut_status": "backend_search_exhausted",
        "claimcut_feasible": False,
        "methods": ["RandomKFold", "GroupKFold_Lab"],
        "method_model_rows": [{"method": "RandomKFold"}],
    }

    status = aggregate_batch.status_record(payload)

    assert status["outcome"] == "success"
    assert status["claimcut_outcome"] == "refusal"


def test_missing_secondary_and_mechanistic_fields_are_reported() -> None:
    frame = pd.DataFrame(
        [
            {
                "scenario": "S3",
                "replicate": 0,
                "model": "ridge",
                "method": "ClaimCut",
                "ere_record_weighted": 0.1,
            }
        ]
    )
    availability = aggregate_batch.metric_field_availability(frame).set_index(
        "field"
    )
    summary = aggregate_batch.aggregate_method_metrics(frame).iloc[0]

    field = "squared_calibration_error_record_weighted"
    assert not bool(availability.loc[field, "present_in_input_schema"])
    assert availability.loc[field, "n_nonmissing"] == 0
    assert summary[f"n_nonmissing_{field}"] == 0
    assert np.isnan(
        summary["mean_squared_calibration_error_record_weighted"]
    )
