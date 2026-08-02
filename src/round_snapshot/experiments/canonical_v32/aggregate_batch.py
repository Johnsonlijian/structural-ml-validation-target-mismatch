"""Aggregate CV1 replicate JSON files without cross-comparator deletion.

Each comparator is paired with ClaimCut independently within
``(scenario, replicate, learner)``.  Bootstrap clusters are meta-replicates:
learner-level paired differences are first averaged inside a replicate and
then replicates are sampled with replacement.  The frozen confirmatory family
contains seven comparators, so one-sided simultaneous 95% upper bounds use a
fixed Bonferroni alpha of ``0.05 / 7`` even when a comparator is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REFERENCE_METHOD = "ClaimCut"
FROZEN_COMPARATORS: tuple[str, ...] = (
    "RandomKFold",
    "GroupKFold_Lab",
    "GroupKFold_Source",
    "GroupKFold_Supplier",
    "CompositeGroup_LabPair",
    "HardUnion_LabSupplier",
    "DataSAIL_C1e_Scalar",
)
FAMILYWISE_ALPHA = 0.05
BONFERRONI_FAMILY_SIZE = 7
PER_COMPARISON_ALPHA = FAMILYWISE_ALPHA / BONFERRONI_FAMILY_SIZE

TYPED_REFUSAL_STATUSES = {
    "certified_balance_infeasible",
    "certified_partition_infeasible",
    "certified_hard_infeasible",
    "backend_search_exhausted",
    "ordered_route_required",
    "realized_plan_invalid_balance",
    "realized_plan_invalid_hard",
    "realized_plan_invalid_targets",
    "unsupported_contract",
    "unsupported_target",
}

METHOD_METRICS: dict[str, tuple[str, ...]] = {
    "ere_record_weighted": ("mean_ere", "median_ere", "sd_ere"),
    "signed_bias_record_weighted": ("mean_signed_bias",),
    "ere_equal_domain": ("mean_equal_domain_ere",),
    "ere_worst_domain": ("mean_worst_domain_ere",),
    "squared_calibration_error_record_weighted": (
        "mean_squared_calibration_error_record_weighted",
    ),
    "squared_calibration_error_equal_domain": (
        "mean_squared_calibration_error_equal_domain",
    ),
    "squared_calibration_error_worst_domain": (
        "mean_squared_calibration_error_worst_domain",
    ),
    "aggregate_external_worst_domain_nrmse": (
        "mean_external_worst_domain_nrmse",
    ),
    "foldwise_absolute_error_secondary": ("mean_foldwise_secondary",),
    "contract_signature_rmse": ("mean_contract_signature_rmse",),
    "max_relation_target_abs_error": (
        "mean_max_relation_target_abs_error",
    ),
    "semantic_loss_count": ("mean_semantic_loss_count",),
    "backend_compile_exact": ("rate_backend_compile_exact",),
}


def load_payloads(input_dir: Path) -> list[dict[str, Any]]:
    """Load every replicate file while preserving unreadable-file failures."""

    payloads: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("rep_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload = {
                "status": "unreadable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        payload["source_file"] = path.name
        payloads.append(payload)
    return payloads


def _as_bool_or_none(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return bool(value)


def classify_replicate(payload: dict[str, Any]) -> str:
    """Return one mutually exclusive denominator outcome.

    A typed refusal is a successful protocol outcome but is kept separate from
    an ordinary successful evaluation.  Complete non-risk semantic records
    without model rows remain ``success`` unless they explicitly carry a typed
    refusal status.
    """

    if str(payload.get("status", "missing")).lower() != "complete":
        return "failure"
    claimcut_status = str(payload.get("claimcut_status") or "")
    expected = payload.get("expected_status") or []
    if isinstance(expected, str):
        expected = [expected]
    no_model_rows = not bool(payload.get("method_model_rows"))
    if no_model_rows and (
        claimcut_status in TYPED_REFUSAL_STATUSES
        or claimcut_status in {str(item) for item in expected}
    ):
        return "refusal"
    return "success"


def classify_claimcut_attempt(payload: dict[str, Any]) -> str:
    """Classify the native compiler separately from replicate generation."""

    if str(payload.get("status", "missing")).lower() != "complete":
        return "failure"
    status = str(payload.get("claimcut_status") or "")
    if status in TYPED_REFUSAL_STATUSES:
        return "refusal"
    methods = set(payload.get("methods") or [])
    methods.update(
        str(row.get("method"))
        for row in payload.get("method_model_rows", [])
        if row.get("method")
    )
    if "ClaimCut" in methods or bool(payload.get("claimcut_feasible")):
        return "success"
    if status:
        return "failure"
    return "not_applicable"


def status_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten replicate and DataSAIL status without discarding solver detail."""

    metadata = payload.get("datasail_metadata") or {}
    datasail_rows_present = any(
        row.get("method") == "DataSAIL_C1e_Scalar"
        for row in payload.get("method_model_rows", [])
    )
    attempted = bool(metadata.get("attempted", datasail_rows_present))
    assignment_returned = bool(
        metadata.get("assignment_returned", datasail_rows_present)
    )
    if not attempted:
        datasail_outcome = "not_attempted"
    elif assignment_returned:
        datasail_outcome = "success"
    else:
        datasail_outcome = "failure"
    semantic_gaps = metadata.get("semantic_gaps")
    return {
        "source_file": payload.get("source_file"),
        "scenario": payload.get("scenario"),
        "replicate": payload.get("replicate"),
        "status": payload.get("status"),
        "outcome": classify_replicate(payload),
        "claimcut_outcome": classify_claimcut_attempt(payload),
        "error": payload.get("error"),
        "claimcut_status": payload.get("claimcut_status"),
        "claimcut_feasible": payload.get("claimcut_feasible"),
        "runtime_seconds": payload.get("runtime_seconds"),
        "datasail_attempted": attempted,
        "datasail_outcome": datasail_outcome,
        "datasail_solver_status": metadata.get("solver_status"),
        "datasail_assignment_returned": assignment_returned,
        "datasail_solver_time_limit_likely": metadata.get(
            "solver_time_limit_likely"
        ),
        "datasail_solver_log_sha256": metadata.get("solver_log_sha256"),
        "datasail_runtime_seconds": metadata.get("runtime_seconds"),
        "datasail_compiler_status": metadata.get("compiler_status"),
        "datasail_semantic_loss_count": (
            len(semantic_gaps) if isinstance(semantic_gaps, list) else None
        ),
        "datasail_error": payload.get("datasail_error")
        or metadata.get("error"),
    }


def summarize_replicate_outcomes(statuses: pd.DataFrame) -> pd.DataFrame:
    """Report attempted/success/refusal/failure denominators by scenario."""

    columns = [
        "scenario",
        "attempted",
        "success",
        "refusal",
        "failure",
        "success_rate",
        "refusal_rate",
        "failure_rate",
    ]
    if statuses.empty:
        return pd.DataFrame(columns=columns)
    work = statuses.copy()
    work["scenario"] = work["scenario"].fillna("<unspecified>")
    rows: list[dict[str, Any]] = []
    for scenario, frame in work.groupby("scenario", sort=False, dropna=False):
        attempted = int(len(frame))
        counts = frame["outcome"].value_counts()
        row: dict[str, Any] = {"scenario": scenario, "attempted": attempted}
        for outcome in ("success", "refusal", "failure"):
            count = int(counts.get(outcome, 0))
            row[outcome] = count
            row[f"{outcome}_rate"] = count / attempted if attempted else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def summarize_claimcut_outcomes(statuses: pd.DataFrame) -> pd.DataFrame:
    """Report native-compiler success/refusal/failure without hiding baselines."""

    columns = [
        "scenario",
        "attempted",
        "success",
        "refusal",
        "failure",
        "not_applicable",
        "success_rate",
        "refusal_rate",
        "failure_rate",
    ]
    if statuses.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    work = statuses.copy()
    work["scenario"] = work["scenario"].fillna("<unspecified>")
    for scenario, frame in work.groupby("scenario", sort=False, dropna=False):
        applicable = frame.loc[frame["claimcut_outcome"] != "not_applicable"]
        attempted = int(len(applicable))
        counts = applicable["claimcut_outcome"].value_counts()
        row: dict[str, Any] = {
            "scenario": scenario,
            "attempted": attempted,
            "not_applicable": int(
                (frame["claimcut_outcome"] == "not_applicable").sum()
            ),
        }
        for outcome in ("success", "refusal", "failure"):
            count = int(counts.get(outcome, 0))
            row[outcome] = count
            row[f"{outcome}_rate"] = _safe_ratio(count, attempted)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def summarize_datasail_status(statuses: pd.DataFrame) -> pd.DataFrame:
    """Retain exact solver-status labels and their explicit denominators."""

    columns = [
        "scenario",
        "datasail_outcome",
        "datasail_solver_status",
        "n_replicates",
        "n_assignment_returned",
        "n_time_limit_likely",
    ]
    if statuses.empty:
        return pd.DataFrame(columns=columns)
    work = statuses.copy()
    work["scenario"] = work["scenario"].fillna("<unspecified>")
    work["datasail_solver_status"] = work[
        "datasail_solver_status"
    ].fillna("missing")
    grouped = (
        work.groupby(
            ["scenario", "datasail_outcome", "datasail_solver_status"],
            sort=False,
            dropna=False,
        )
        .agg(
            n_replicates=("source_file", "size"),
            n_assignment_returned=("datasail_assignment_returned", "sum"),
            n_time_limit_likely=("datasail_solver_time_limit_likely", "sum"),
        )
        .reset_index()
    )
    return grouped.loc[:, columns]


def metric_field_availability(
    frame: pd.DataFrame,
    *,
    metrics: Iterable[str] = METHOD_METRICS,
) -> pd.DataFrame:
    """Make absent and partially missing secondary/mechanistic fields explicit."""

    rows: list[dict[str, Any]] = []
    n_rows = int(len(frame))
    for field in metrics:
        present = field in frame.columns
        n_nonmissing = int(frame[field].notna().sum()) if present else 0
        rows.append(
            {
                "field": field,
                "present_in_input_schema": bool(present),
                "n_nonmissing": n_nonmissing,
                "n_rows": n_rows,
                "nonmissing_coverage": (
                    n_nonmissing / n_rows if n_rows else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_method_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all prespecified risk and mechanism fields when available."""

    base_columns = ["scenario", "method", "n_replicates", "n_model_rows"]
    if frame.empty:
        return pd.DataFrame(columns=base_columns)
    rows: list[dict[str, Any]] = []
    for (scenario, method), group in frame.groupby(
        ["scenario", "method"], sort=False, dropna=False
    ):
        row: dict[str, Any] = {
            "scenario": scenario,
            "method": method,
            "n_replicates": int(group["replicate"].nunique()),
            "n_model_rows": int(len(group)),
        }
        for field, output_names in METHOD_METRICS.items():
            if field in group.columns:
                values = pd.to_numeric(group[field], errors="coerce")
            else:
                values = pd.Series(np.nan, index=group.index, dtype=float)
            row[f"n_nonmissing_{field}"] = int(values.notna().sum())
            for output_name in output_names:
                if output_name.startswith("median_"):
                    value = values.median()
                elif output_name.startswith("sd_"):
                    value = values.std()
                else:
                    value = values.mean()
                row[output_name] = float(value) if pd.notna(value) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def build_comparator_pairs(
    frame: pd.DataFrame,
    *,
    comparisons: Iterable[str] = FROZEN_COMPARATORS,
    reference_method: str = REFERENCE_METHOD,
    value_col: str = "ere_record_weighted",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build comparator-specific pairs and coverage; never globally drop rows."""

    required = {"replicate", "model", "method", value_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"pairing input missing columns: {sorted(missing)}")
    comparisons = tuple(comparisons)
    pivot = frame.pivot_table(
        index=["replicate", "model"],
        columns="method",
        values=value_col,
        aggfunc="first",
        dropna=False,
    )
    reference = (
        pd.to_numeric(pivot[reference_method], errors="coerce")
        if reference_method in pivot.columns
        else pd.Series(np.nan, index=pivot.index, dtype=float)
    )
    pair_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    reference_mask = reference.notna()
    reference_replicates = set(
        pivot.index.get_level_values("replicate")[reference_mask]
    )
    for comparator in comparisons:
        candidate = (
            pd.to_numeric(pivot[comparator], errors="coerce")
            if comparator in pivot.columns
            else pd.Series(np.nan, index=pivot.index, dtype=float)
        )
        comparator_mask = candidate.notna()
        paired_mask = reference_mask & comparator_mask
        paired = pivot.index.to_frame(index=False).loc[paired_mask.to_numpy()]
        if not paired.empty:
            paired["reference_value"] = reference[paired_mask].to_numpy()
            paired["comparator_value"] = candidate[paired_mask].to_numpy()
            paired["difference"] = (
                paired["reference_value"] - paired["comparator_value"]
            )
            paired["reference_method"] = reference_method
            paired["comparator"] = comparator
            pair_frames.append(paired)
        comparator_replicates = set(
            pivot.index.get_level_values("replicate")[comparator_mask]
        )
        paired_replicates = set(
            pivot.index.get_level_values("replicate")[paired_mask]
        )
        if not reference_mask.any():
            pairing_status = "missing_reference"
        elif not comparator_mask.any():
            pairing_status = "missing_comparator"
        elif not paired_mask.any():
            pairing_status = "no_pair_overlap"
        else:
            pairing_status = "paired"
        n_reference_rows = int(reference_mask.sum())
        n_comparator_rows = int(comparator_mask.sum())
        n_paired_rows = int(paired_mask.sum())
        coverage_rows.append(
            {
                "reference_method": reference_method,
                "comparator": comparator,
                "pairing_status": pairing_status,
                "n_reference_model_rows": n_reference_rows,
                "n_comparator_model_rows": n_comparator_rows,
                "n_paired_model_rows": n_paired_rows,
                "model_pairing_coverage_vs_reference": _safe_ratio(
                    n_paired_rows, n_reference_rows
                ),
                "n_reference_replicates": len(reference_replicates),
                "n_comparator_replicates": len(comparator_replicates),
                "n_paired_replicates": len(paired_replicates),
                "replicate_pairing_coverage_vs_reference": _safe_ratio(
                    len(paired_replicates), len(reference_replicates)
                ),
            }
        )
    pairs = (
        pd.concat(pair_frames, ignore_index=True)
        if pair_frames
        else pd.DataFrame(
            columns=[
                "replicate",
                "model",
                "reference_value",
                "comparator_value",
                "difference",
                "reference_method",
                "comparator",
            ]
        )
    )
    return pairs, pd.DataFrame(coverage_rows)


def _stable_seed(base_seed: int, scenario: str, comparator: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{scenario}|{comparator}".encode("utf-8")
    ).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**32)


def clustered_bonferroni_bootstrap(
    pairs: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    scenario: str,
    comparisons: Iterable[str] = FROZEN_COMPARATORS,
    n_bootstrap: int,
    seed: int,
    familywise_alpha: float = FAMILYWISE_ALPHA,
    family_size: int = BONFERRONI_FAMILY_SIZE,
) -> pd.DataFrame:
    """Replicate-clustered, paired bootstrap with fixed-family Bonferroni UCBs."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not (0.0 < familywise_alpha < 1.0):
        raise ValueError("familywise_alpha must be between zero and one")
    if family_size <= 0:
        raise ValueError("family_size must be positive")
    comparisons = tuple(comparisons)
    per_comparison_alpha = familywise_alpha / family_size
    coverage_index = coverage.set_index("comparator", drop=False)
    rows: list[dict[str, Any]] = []
    for comparator in comparisons:
        coverage_row = (
            coverage_index.loc[comparator].to_dict()
            if comparator in coverage_index.index
            else {
                "reference_method": REFERENCE_METHOD,
                "comparator": comparator,
                "pairing_status": "missing_comparator",
            }
        )
        comparator_pairs = pairs.loc[pairs["comparator"] == comparator]
        replicate_differences = (
            comparator_pairs.groupby("replicate", sort=True)["difference"]
            .mean()
            .dropna()
        )
        result: dict[str, Any] = {
            "scenario": scenario,
            **coverage_row,
            "familywise_alpha": familywise_alpha,
            "bonferroni_family_size": family_size,
            "per_comparison_alpha": per_comparison_alpha,
            "one_sided_simultaneous_confidence": 1.0 - familywise_alpha,
            "one_sided_per_comparison_quantile": 1.0 - per_comparison_alpha,
            "n_bootstrap": int(n_bootstrap),
            "n_cluster_replicates": int(len(replicate_differences)),
            "n_replicates": int(len(replicate_differences)),
        }
        if replicate_differences.empty:
            result.update(
                observed_mean_difference=np.nan,
                bootstrap_ci_2_5=np.nan,
                bootstrap_ci_97_5=np.nan,
                bootstrap_upper_one_sided_95=np.nan,
                simultaneous_upper_one_sided_95=np.nan,
                simultaneous_upper_one_sided_95_bonferroni=np.nan,
            )
            rows.append(result)
            continue
        values = replicate_differences.to_numpy(dtype=float)
        rng = np.random.default_rng(_stable_seed(seed, scenario, comparator))
        sampled_indices = rng.integers(
            0, len(values), size=(n_bootstrap, len(values))
        )
        bootstrap_means = values[sampled_indices].mean(axis=1)
        observed = float(values.mean())
        bonferroni_upper = float(
            np.quantile(bootstrap_means, 1.0 - per_comparison_alpha)
        )
        result.update(
            observed_mean_difference=observed,
            bootstrap_ci_2_5=float(np.quantile(bootstrap_means, 0.025)),
            bootstrap_ci_97_5=float(np.quantile(bootstrap_means, 0.975)),
            bootstrap_upper_one_sided_95=float(
                np.quantile(bootstrap_means, 0.95)
            ),
            simultaneous_upper_one_sided_95=bonferroni_upper,
            simultaneous_upper_one_sided_95_bonferroni=bonferroni_upper,
        )
        rows.append(result)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026071102)
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads(input_dir)

    statuses = pd.DataFrame([status_record(payload) for payload in payloads])
    method_model_rows: list[dict[str, Any]] = []
    method_summary_rows: list[dict[str, Any]] = []
    for payload in payloads:
        for row in payload.get("method_model_rows", []):
            method_model_rows.append(
                {
                    "scenario": payload.get("scenario"),
                    "replicate": payload.get("replicate"),
                    **row,
                }
            )
        for row in payload.get("method_summaries", []):
            method_summary_rows.append(
                {
                    "scenario": payload.get("scenario"),
                    "replicate": payload.get("replicate"),
                    **row,
                }
            )

    method_models = pd.DataFrame(method_model_rows)
    replicate_summaries = pd.DataFrame(method_summary_rows)
    outcome_summary = summarize_replicate_outcomes(statuses)
    claimcut_summary = summarize_claimcut_outcomes(statuses)
    datasail_summary = summarize_datasail_status(statuses)
    field_availability = metric_field_availability(method_models)

    statuses.to_csv(output_dir / "replicate_status.csv", index=False)
    outcome_summary.to_csv(
        output_dir / "replicate_outcome_summary.csv", index=False
    )
    claimcut_summary.to_csv(
        output_dir / "claimcut_outcome_summary.csv", index=False
    )
    datasail_summary.to_csv(
        output_dir / "datasail_solver_status_summary.csv", index=False
    )
    method_models.to_csv(output_dir / "method_model_rows.csv", index=False)
    replicate_summaries.to_csv(
        output_dir / "replicate_method_summaries.csv", index=False
    )
    field_availability.to_csv(
        output_dir / "metric_field_availability.csv", index=False
    )

    grouped = aggregate_method_metrics(method_models)
    grouped.to_csv(output_dir / "method_aggregate_summary.csv", index=False)

    uncertainty_outputs: list[pd.DataFrame] = []
    coverage_outputs: list[pd.DataFrame] = []
    if not method_models.empty:
        for scenario, frame in method_models.groupby(
            "scenario", sort=False, dropna=False
        ):
            scenario_label = (
                "<unspecified>" if pd.isna(scenario) else str(scenario)
            )
            pairs, coverage = build_comparator_pairs(frame)
            coverage.insert(0, "scenario", scenario_label)
            coverage_outputs.append(coverage)
            uncertainty_outputs.append(
                clustered_bonferroni_bootstrap(
                    pairs,
                    coverage.drop(columns="scenario"),
                    scenario=scenario_label,
                    n_bootstrap=args.bootstrap,
                    seed=args.seed,
                )
            )

    coverage_frame = (
        pd.concat(coverage_outputs, ignore_index=True)
        if coverage_outputs
        else pd.DataFrame()
    )
    uncertainty_frame = (
        pd.concat(uncertainty_outputs, ignore_index=True)
        if uncertainty_outputs
        else pd.DataFrame()
    )
    coverage_frame.to_csv(
        output_dir / "comparator_pairing_coverage.csv", index=False
    )
    uncertainty_frame.to_csv(
        output_dir / "paired_cluster_bootstrap.csv", index=False
    )

    totals = (
        statuses["outcome"].value_counts().to_dict()
        if not statuses.empty
        else {}
    )
    field_records = field_availability.to_dict(orient="records")
    manifest = {
        "n_files": int(len(payloads)),
        "n_complete": int(
            (statuses["status"] == "complete").sum()
            if not statuses.empty
            else 0
        ),
        "n_failed": int(
            (statuses["status"] == "failed").sum()
            if not statuses.empty
            else 0
        ),
        "n_files_attempted": int(len(payloads)),
        "n_success": int(totals.get("success", 0)),
        "n_refusal": int(totals.get("refusal", 0)),
        "n_failure": int(totals.get("failure", 0)),
        "n_method_model_rows": int(len(method_models)),
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_seed": int(args.seed),
        "pairing_unit": ["scenario", "replicate", "model"],
        "bootstrap_cluster": "replicate",
        "within_cluster_summary": "mean paired learner difference",
        "reference_method": REFERENCE_METHOD,
        "frozen_comparators": list(FROZEN_COMPARATORS),
        "multiple_comparison_method": "one-sided Bonferroni upper bound",
        "familywise_alpha": FAMILYWISE_ALPHA,
        "bonferroni_family_size": BONFERRONI_FAMILY_SIZE,
        "per_comparison_alpha": PER_COMPARISON_ALPHA,
        "fixed_family_when_comparator_missing": True,
        "global_complete_case_deletion": False,
        "metric_field_availability": field_records,
        "replicate_outcomes": outcome_summary.to_dict(orient="records"),
        "claimcut_outcomes": claimcut_summary.to_dict(orient="records"),
        "datasail_solver_status": datasail_summary.to_dict(orient="records"),
    }
    (output_dir / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not grouped.empty:
        print(grouped.to_string(index=False))
    if not uncertainty_frame.empty:
        print(uncertainty_frame.to_string(index=False))


if __name__ == "__main__":
    main()
