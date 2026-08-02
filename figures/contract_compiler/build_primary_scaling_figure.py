"""Build Figure 4 from two closed predictive aggregates and scalability v1.

The active release path is intentionally narrow and fail-closed.  It requires
the Attempt-S post-run evidence lock and retain-all S0/S1/S3 receipt before
validating the public S0/S3 plotting aggregates.  A legacy comparison gate is
loaded only when an explicitly supplied historical lock requests it; the
Attempt-S public package does not ship or invoke that module.  The scalability
panel is independently bound to the frozen manifest and hashed outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from figure_aggregate_gate import (
    AggregateGateError,
    EXPECTED_FREEZE_MANIFEST_SHA256,
    PREDICTIVE_SCHEMA,
    read_stage_csv,
    validate_stage_manifests,
)
PUBLIC_FIGURE_ROOT = Path(__file__).resolve().parent
PUBLIC_PACKAGE_ROOT = PUBLIC_FIGURE_ROOT.parents[1]
DEFAULT_COMPARISON_RECEIPT = (
    PUBLIC_PACKAGE_ROOT / "provenance" / "ATTEMPT_A_CANONICAL_COMPARISON.json"
)
DEFAULT_RETENTION_RECEIPT = (
    PUBLIC_PACKAGE_ROOT / "provenance" / "ATTEMPT_S_RETENTION_AUDIT.json"
)
ATTEMPT_S_LOCK_SCHEMA = "figure4-primary-evidence-lock-s-v1"
EXPECTED_SCALABILITY_FREEZE_MANIFEST_SHA256 = (
    "e7c879da18689e850c0498b47b45333e02b67e3e1b31ee5b819f493ce4bc3bb9"
)
EXPECTED_SCALABILITY_AGGREGATE_SUMMARY_SHA256 = (
    "78ba3a62055185fa469e9d696622880f45a838c4b9ce625ec71c64668735f2b1"
)
EXPECTED_SCALABILITY_BUNDLE_SHA256 = (
    "9e5ce48232d1a34256eb86dace8ece29da6ef0ceae51df76bc16cb304d0fc383"
)

COMPARATORS = (
    "RandomKFold",
    "GroupKFold_Lab",
    "GroupKFold_Source",
    "GroupKFold_Supplier",
    "CompositeGroup_LabPair",
    "HardUnion_LabSupplier",
    "DataSAIL_C1e_Scalar",
)
DISPLAY_LABELS = {
    "RandomKFold": "Random K-fold",
    "GroupKFold_Lab": "Group: laboratory",
    "GroupKFold_Source": "Group: source",
    "GroupKFold_Supplier": "Group: supplier",
    "CompositeGroup_LabPair": "Composite: lab-pair",
    "HardUnion_LabSupplier": "Hard union",
    "DataSAIL_C1e_Scalar": "DataSAIL C1e",
}

NAVY = "#2F5597"
TEAL = "#16828C"
TEAL_LIGHT = "#B9DDE0"
AMBER = "#D89216"
PURPLE = "#76528A"
VERMILION = "#C44E3B"
TEXT = "#1D2731"
MUTED = "#5E6973"
GRID = "#DDE3E7"
PALE_TEAL = "#EDF6F6"
PALE_AMBER = "#FFF5E4"


class Figure4GateError(RuntimeError):
    """Raised before any output is created when an input gate fails."""


@dataclass(frozen=True)
class Comparison:
    scenario: str
    comparator: str
    pairing_status: str
    mean_difference: float | None
    simultaneous_ucb: float | None
    n_replicates: int


@dataclass(frozen=True)
class ScalingPoint:
    axis: str
    x: float
    label: str
    baseline: bool
    median_seconds: float
    q1_seconds: float
    q3_seconds: float


@dataclass(frozen=True)
class Figure4Data:
    s0: tuple[Comparison, ...]
    s3: tuple[Comparison, ...]
    scaling: tuple[ScalingPoint, ...]
    status_counts: Mapping[str, int]
    high_combined: Mapping[str, float]
    receipt: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Figure4GateError(f"required immutable JSON is absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Figure4GateError(f"unreadable immutable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Figure4GateError(f"immutable JSON must contain an object: {path}")
    return value


def _finite(value: str, *, label: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise Figure4GateError(f"{label}: expected a number") from exc
    if not math.isfinite(number):
        raise Figure4GateError(f"{label}: expected a finite number")
    return number


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_primary_evidence_lock(
    path: Path,
    selection_receipt: Path,
    aggregate_root: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    lock = _read_json(path)
    schema = lock.get("schema_version")
    if schema == ATTEMPT_S_LOCK_SCHEMA:
        module = importlib.import_module("attempt_s_retention_gate")
        receipt_keyword = "retention_receipt"
        error_type = module.AttemptSRetentionGateError
    else:
        # Backward-compatible only in a historical draft that still ships the
        # optional comparison module.  Attempt-S finalization removes it.
        try:
            module = importlib.import_module("attempt_comparison_gate")
        except ModuleNotFoundError as exc:
            raise Figure4GateError(
                "evidence lock is not the Attempt-S retention schema"
            ) from exc
        receipt_keyword = "comparison_receipt"
        error_type = module.AttemptComparisonGateError
    try:
        binding = module.validate_figure4_evidence(
            **{
                "primary_evidence_lock": path,
                receipt_keyword: selection_receipt,
                "aggregate_root": aggregate_root,
                "expected_freeze_manifest_sha256": (
                    EXPECTED_FREEZE_MANIFEST_SHA256
                ),
            }
        )
    except error_type as exc:
        raise Figure4GateError(str(exc)) from exc
    return dict(binding.primary_hashes), dict(binding.receipt)


def _validate_public_primary_root(a32_root: Path) -> str:
    root = a32_root.resolve()
    if (
        root.name != "confirmatory_aggregates_v32"
        or root.parent.name != "derived"
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise Figure4GateError(
            "primary input must be the public derived/confirmatory_aggregates_v32 root"
        )
    lowered = tuple(part.lower() for part in root.parts)
    if any(part.startswith("excluded_") or "interrupted" in part for part in lowered):
        raise Figure4GateError("excluded-attempt aggregates are never admissible")
    return EXPECTED_FREEZE_MANIFEST_SHA256


def _load_primary_rows(
    a32_root: Path,
    expected_primary_hashes: Mapping[str, str],
) -> tuple[tuple[Comparison, ...], tuple[Comparison, ...], dict[str, Any]]:
    freeze_hash = _validate_public_primary_root(a32_root)
    try:
        stages = validate_stage_manifests(
            a32_root, ("S0", "S3"), expected_schema=PREDICTIVE_SCHEMA
        )
    except AggregateGateError as exc:
        raise Figure4GateError(str(exc)) from exc

    by_stage: dict[str, tuple[Comparison, ...]] = {}
    receipt: dict[str, Any] = {"v32_freeze_manifest_sha256": freeze_hash, "stages": {}}
    required_columns = (
        "scenario",
        "reference_method",
        "comparator",
        "pairing_status",
        "n_paired_replicates",
        "familywise_alpha",
        "bonferroni_family_size",
        "n_bootstrap",
        "observed_mean_difference",
        "simultaneous_upper_one_sided_95_bonferroni",
    )
    for stage_name in ("S0", "S3"):
        stage = stages[stage_name]
        summary_path = stage.directory / "paired_cluster_bootstrap.csv"
        observed_hash = _sha256(summary_path)
        if observed_hash != expected_primary_hashes[stage_name]:
            raise Figure4GateError(
                f"{stage_name}: paired aggregate summary hash drifted"
            )
        base_hash = stage.manifest.get("base_manifest_sha256")
        if not _hex64(base_hash):
            raise Figure4GateError(f"{stage_name}: base aggregate binding is absent")
        try:
            raw_rows = read_stage_csv(
                stage,
                "paired_cluster_bootstrap.csv",
                required_columns=required_columns,
                expected_rows=len(COMPARATORS),
            )
        except AggregateGateError as exc:
            raise Figure4GateError(str(exc)) from exc
        if tuple(row["comparator"] for row in raw_rows) != COMPARATORS:
            raise Figure4GateError(f"{stage_name}: comparator family/order drifted")

        rows: list[Comparison] = []
        for row in raw_rows:
            comparator = row["comparator"]
            if row["scenario"] != stage_name or row["reference_method"] != "ClaimCut":
                raise Figure4GateError(f"{stage_name}: reference/scenario drifted")
            if _finite(row["familywise_alpha"], label="familywise_alpha") != 0.05:
                raise Figure4GateError(f"{stage_name}: familywise alpha drifted")
            if int(row["bonferroni_family_size"]) != 7 or int(row["n_bootstrap"]) != 10000:
                raise Figure4GateError(f"{stage_name}: simultaneous-bound design drifted")
            status = row["pairing_status"]
            if comparator == "HardUnion_LabSupplier":
                if status != "missing_comparator" or int(row["n_paired_replicates"]) != 0:
                    raise Figure4GateError(f"{stage_name}: HardUnion availability drifted")
                mean = ucb = None
                n_replicates = 0
            else:
                if status != "paired" or int(row["n_paired_replicates"]) != 100:
                    raise Figure4GateError(f"{stage_name}/{comparator}: incomplete pairing")
                mean = _finite(row["observed_mean_difference"], label=f"{stage_name}/{comparator}/mean")
                ucb = _finite(
                    row["simultaneous_upper_one_sided_95_bonferroni"],
                    label=f"{stage_name}/{comparator}/UCB",
                )
                if ucb < mean:
                    raise Figure4GateError(f"{stage_name}/{comparator}: UCB is below the mean")
                n_replicates = 100
            rows.append(
                Comparison(stage_name, comparator, status, mean, ucb, n_replicates)
            )
        by_stage[stage_name] = tuple(rows)
        receipt["stages"][stage_name] = {
            "paired_summary_sha256": observed_hash,
            "aggregate_manifest_sha256": base_hash,
            "wrapper_sha256": _sha256(stage.manifest_path),
            "n_results": len(stage.manifest["input_records"]),
        }

    return by_stage["S0"], by_stage["S3"], receipt


def _load_scalability(scalability_root: Path) -> tuple[tuple[ScalingPoint, ...], dict[str, int], dict[str, float], dict[str, Any]]:
    root = scalability_root.resolve()
    if root.name != "scalability_v1" or not root.is_dir() or root.is_symlink():
        raise Figure4GateError("scalability input must be the canonical frozen scalability_v1 root")
    freeze_path = root / "PROTOCOL_FREEZE_MANIFEST.json"
    aggregate_path = root / "o" / "aggregate_summary.json"
    if _sha256(freeze_path) != EXPECTED_SCALABILITY_FREEZE_MANIFEST_SHA256:
        raise Figure4GateError("scalability freeze manifest hash drifted")
    if _sha256(aggregate_path) != EXPECTED_SCALABILITY_AGGREGATE_SUMMARY_SHA256:
        raise Figure4GateError("scalability aggregate summary hash drifted")
    freeze = _read_json(freeze_path)
    aggregate = _read_json(aggregate_path)
    if (
        freeze.get("protocol_id") != "CLAIMCUT-SCALABILITY-V1"
        or freeze.get("bundle_sha256") != EXPECTED_SCALABILITY_BUNDLE_SHA256
        or aggregate.get("schema_version") != "claimcut-scalability-aggregate-v1"
        or aggregate.get("freeze_bundle_sha256") != EXPECTED_SCALABILITY_BUNDLE_SHA256
        or aggregate.get("freeze_manifest_sha256") != EXPECTED_SCALABILITY_FREEZE_MANIFEST_SHA256
    ):
        raise Figure4GateError("scalability protocol/freeze binding drifted")

    headline = aggregate.get("headline")
    if not isinstance(headline, dict):
        raise Figure4GateError("scalability headline gate is absent")
    required_headline = {
        "completion_gate": True,
        "scientific_conformance_gate": True,
        "n_expected": 68,
        "n_observed": 68,
        "n_false_exact": 0,
        "n_invalid_certificates": 0,
        "n_auditor_disagreement": 0,
        "n_prespecified_status_mismatch": 0,
    }
    for field, expected in required_headline.items():
        if headline.get(field) != expected:
            raise Figure4GateError(f"scalability headline gate failed: {field}")
    for field in ("duplicate_run_ids", "missing_artifact_ids", "unexpected_artifact_ids"):
        if headline.get(field) != []:
            raise Figure4GateError(f"scalability artifact set is not closed: {field}")

    output_hashes = aggregate.get("output_hashes")
    if not isinstance(output_hashes, dict) or set(output_hashes) != {
        "cell_summary.csv", "figure4_scalability.csv", "results_raw.csv", "trend_models.csv"
    }:
        raise Figure4GateError("scalability aggregate output registry drifted")
    for filename, metadata in output_hashes.items():
        path = root / "o" / filename
        if not isinstance(metadata, dict) or _sha256(path) != metadata.get("sha256"):
            raise Figure4GateError(f"scalability aggregate output hash drifted: {filename}")
        if path.stat().st_size != int(metadata.get("bytes", -1)):
            raise Figure4GateError(f"scalability aggregate output size drifted: {filename}")

    figure_csv = root / "o" / "figure4_scalability.csv"
    with figure_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        required = {
            "cell_id", "display_axis", "x_numeric", "x_label", "baseline_reused",
            "compile_ms_median", "compile_ms_q1", "compile_ms_q3",
            "checker_ms_median", "peak_rss_increment_mb_median",
        }
        if not required.issubset(set(reader.fieldnames or [])):
            raise Figure4GateError("scalability figure table schema drifted")
    if len(rows) != 21:
        raise Figure4GateError("scalability figure table row count drifted")

    points: list[ScalingPoint] = []
    expected_axis_sizes = {"records": 4, "relations": 4, "folds": 3, "clauses": 3}
    for axis, expected_n in expected_axis_sizes.items():
        axis_rows = [row for row in rows if row["display_axis"] == axis]
        if len(axis_rows) != expected_n:
            raise Figure4GateError(f"scalability one-factor path drifted: {axis}")
        for row in axis_rows:
            median = _finite(row["compile_ms_median"], label=f"{axis}/median") / 1000.0
            q1 = _finite(row["compile_ms_q1"], label=f"{axis}/q1") / 1000.0
            q3 = _finite(row["compile_ms_q3"], label=f"{axis}/q3") / 1000.0
            if not q1 <= median <= q3:
                raise Figure4GateError(f"{axis}: median is outside its IQR")
            points.append(
                ScalingPoint(
                    axis=axis,
                    x=_finite(row["x_numeric"], label=f"{axis}/x"),
                    label=row["x_label"],
                    baseline=row["baseline_reused"].lower() == "true",
                    median_seconds=median,
                    q1_seconds=q1,
                    q3_seconds=q3,
                )
            )

    status_counts = headline.get("terminal_class_counts")
    expected_statuses = {
        "split_exact": 56,
        "certified_refusal": 8,
        "search_exhaustion_unresolved": 4,
    }
    if status_counts != expected_statuses:
        raise Figure4GateError("scalability terminal-class counts drifted")

    high_rows = [row for row in rows if row["cell_id"] == "HIGH_COMBINED"]
    if len(high_rows) != 1:
        raise Figure4GateError("HIGH_COMBINED scalability row is absent or duplicated")
    high = high_rows[0]
    high_combined = {
        "compile_seconds": _finite(high["compile_ms_median"], label="HIGH_COMBINED/compile") / 1000.0,
        "checker_ms": _finite(high["checker_ms_median"], label="HIGH_COMBINED/checker"),
        "rss_mib": _finite(high["peak_rss_increment_mb_median"], label="HIGH_COMBINED/rss"),
    }
    targets = {"compile_seconds": 11.92595835, "checker_ms": 43.3187, "rss_mib": 2.904296875}
    for field, expected in targets.items():
        if not math.isclose(high_combined[field], expected, rel_tol=0.0, abs_tol=1e-10):
            raise Figure4GateError(f"HIGH_COMBINED headline metric drifted: {field}")
    receipt = {
        "freeze_manifest_sha256": EXPECTED_SCALABILITY_FREEZE_MANIFEST_SHA256,
        "aggregate_summary_sha256": EXPECTED_SCALABILITY_AGGREGATE_SUMMARY_SHA256,
        "figure4_scalability_sha256": output_hashes["figure4_scalability.csv"]["sha256"],
        "n_observed": 68,
        "n_false_exact": 0,
    }
    return tuple(points), dict(status_counts), high_combined, receipt


def load_verified_data(
    a32_root: Path,
    scalability_root: Path,
    primary_evidence_lock: Path,
    selection_receipt: Path,
) -> Figure4Data:
    _validate_public_primary_root(a32_root)
    primary_hashes, lock_receipt = _load_primary_evidence_lock(
        primary_evidence_lock,
        selection_receipt,
        a32_root,
    )
    s0, s3, primary_receipt = _load_primary_rows(a32_root, primary_hashes)
    primary_receipt["post_run_lock"] = lock_receipt
    scaling, statuses, high, scaling_receipt = _load_scalability(scalability_root)
    refreshed_hashes, refreshed_receipt = _load_primary_evidence_lock(
        primary_evidence_lock,
        selection_receipt,
        a32_root,
    )
    if refreshed_hashes != primary_hashes or refreshed_receipt != lock_receipt:
        raise Figure4GateError("primary evidence changed during Figure 4 validation")
    return Figure4Data(
        s0=s0,
        s3=s3,
        scaling=scaling,
        status_counts=statuses,
        high_combined=high,
        receipt={"primary": primary_receipt, "scalability": scaling_receipt},
    )


def _atomic_export(fig, output_root: Path) -> list[Path]:
    targets = {
        "svg": output_root / "svg" / "Figure_4.svg",
        "pdf": output_root / "pdf" / "Figure_4.pdf",
        "png": output_root / "png" / "Figure_4.png",
    }
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    temporaries: dict[str, Path] = {}
    try:
        for kind, target in targets.items():
            handle, temporary_name = tempfile.mkstemp(
                prefix=".__Figure_4_primary_", suffix=f".{kind}", dir=target.parent
            )
            os.close(handle)
            temporary = Path(temporary_name)
            temporaries[kind] = temporary
            fig.savefig(
                temporary,
                format=kind,
                dpi=600 if kind == "png" else None,
                facecolor="white",
            )
        for kind, target in targets.items():
            os.replace(temporaries[kind], target)
        return list(targets.values())
    finally:
        for temporary in temporaries.values():
            if temporary.exists():
                temporary.unlink()


def _draw_one_sided_interval(ax, y: float, mean: float, ucb: float, color: str) -> None:
    ax.hlines(y, mean, ucb, color=color, linewidth=1.45, zorder=3)
    ax.vlines(ucb, y - 0.105, y + 0.105, color=color, linewidth=1.15, zorder=3)
    ax.scatter([mean], [y], s=25, color=color, edgecolor="white", linewidth=0.55, zorder=4)


def build_figure(data: Figure4Data, output_root: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.4,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "axes.linewidth": 0.65,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )

    fig = plt.figure(figsize=(7.20, 6.35), facecolor="white")
    outer = fig.add_gridspec(
        2, 1, height_ratios=(0.93, 1.07), left=0.075, right=0.985,
        top=0.965, bottom=0.145, hspace=0.38,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=(0.90, 1.38), wspace=0.34)
    bottom = outer[1].subgridspec(1, 2, width_ratios=(1.62, 1.00), wspace=0.29)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_d = fig.add_subplot(bottom[0, 1])

    # (a) Prespecified S0 noninferiority: the 0.02 margin is deliberately visible.
    s0_random = next(row for row in data.s0 if row.comparator == "RandomKFold")
    assert s0_random.mean_difference is not None and s0_random.simultaneous_ucb is not None
    s0_low = min(-0.004, s0_random.mean_difference, s0_random.simultaneous_ucb)
    s0_high = max(0.024, s0_random.mean_difference, s0_random.simultaneous_ucb)
    s0_padding = max(0.001, 0.06 * (s0_high - s0_low))
    s0_low -= s0_padding
    s0_high += s0_padding
    noninferiority_closes = s0_random.simultaneous_ucb < 0.02
    s0_color = TEAL if noninferiority_closes else AMBER
    ax_a.axvspan(s0_low, 0.02, color=PALE_TEAL, zorder=0)
    ax_a.axvspan(0.02, s0_high, color=PALE_AMBER, zorder=0)
    ax_a.axvline(0.0, color=MUTED, linewidth=0.75, linestyle=(0, (2, 2)), zorder=1)
    ax_a.axvline(0.02, color=AMBER, linewidth=1.35, zorder=1)
    _draw_one_sided_interval(
        ax_a, 0.0, s0_random.mean_difference, s0_random.simultaneous_ucb, s0_color
    )
    ax_a.set_xlim(s0_low, s0_high)
    ax_a.set_ylim(-0.72, 0.72)
    ax_a.set_yticks([])
    ax_a.set_xticks((-0.004, 0.0, 0.01, 0.02))
    ax_a.set_xlabel("ClaimCut − random ERE (lower is better)")
    ax_a.set_title(
        "a  S0: prespecified noninferiority "
        + ("closes" if noninferiority_closes else "does not close"),
        loc="left", fontweight="bold", color=TEXT, pad=5,
    )
    ax_a.text(
        0.02, 0.56, "0.02 margin", ha="center", va="top", color=AMBER,
        fontweight="bold", fontsize=5.9,
    )
    ax_a.text(
        0.03, 0.18,
        f"mean  {s0_random.mean_difference:+.5f}\n"
        f"simultaneous UCB  {s0_random.simultaneous_ucb:+.5f}",
        transform=ax_a.transAxes, color=TEXT, fontsize=6.1, linespacing=1.35,
    )
    ax_a.text(
        0.03, 0.82,
        "UCB < margin" if noninferiority_closes else "UCB ≥ margin",
        transform=ax_a.transAxes,
        color=s0_color, fontsize=6.4, fontweight="bold",
    )
    ax_a.spines[["top", "right", "left"]].set_visible(False)

    # (b) S3: show all seven frozen comparators, including the unavailable one.
    y_positions = np.arange(len(COMPARATORS))[::-1]
    s3_values = [
        value
        for row in data.s3
        for value in (row.mean_difference, row.simultaneous_ucb)
        if value is not None
    ]
    s3_low = min([-0.034, 0.0, *s3_values])
    s3_high = max([0.007, 0.0, *s3_values])
    s3_padding = max(0.001, 0.05 * (s3_high - s3_low))
    s3_low -= s3_padding
    s3_high += s3_padding
    ax_b.axvspan(0.0, s3_high, color=PALE_AMBER, zorder=0)
    ax_b.axvline(0.0, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    for y, row in zip(y_positions, data.s3):
        if row.mean_difference is None or row.simultaneous_ucb is None:
            ax_b.scatter([0.0015], [y], marker="x", s=24, color=MUTED, linewidth=1.1, zorder=3)
            ax_b.text(0.0023, y, "unavailable", va="center", fontsize=5.4, color=MUTED)
            continue
        color = NAVY if row.simultaneous_ucb < 0 else AMBER
        _draw_one_sided_interval(ax_b, float(y), row.mean_difference, row.simultaneous_ucb, color)
    ax_b.set_yticks(y_positions, [DISPLAY_LABELS[name] for name in COMPARATORS])
    ax_b.set_xlim(s3_low, s3_high)
    ax_b.set_xticks((-0.03, -0.02, -0.01, 0.0))
    ax_b.set_ylim(-0.7, 6.75)
    ax_b.set_xlabel("ClaimCut − comparator ERE (lower is better)")
    ax_b.set_title(
        "b  S3: simultaneous evidence is comparator-specific",
        loc="left", fontweight="bold", color=TEXT, pad=5,
    )
    ax_b.text(
        0.99, 0.98, "positive UCB: superiority rule not met",
        transform=ax_b.transAxes, ha="right", va="top", color=AMBER,
        fontsize=5.9, fontweight="bold",
    )
    ax_b.text(
        0.015, 0.02, "dot: mean    cap: simultaneous UCB",
        transform=ax_b.transAxes, color=MUTED, fontsize=5.3,
    )
    ax_b.grid(axis="y", color=GRID, linewidth=0.45)
    ax_b.spines[["top", "right", "left"]].set_visible(False)
    ax_b.tick_params(axis="y", length=0)

    # (c) Four one-factor paths; one shared y range prevents visual exaggeration.
    ax_c.axis("off")
    ax_c.text(
        0.0, 1.05, "c  Bounded one-factor compile-time paths",
        transform=ax_c.transAxes, fontweight="bold", color=TEXT, fontsize=7.4,
    )
    ax_c.text(
        1.0, 1.05, "median and IQR; n=4 per cell",
        transform=ax_c.transAxes, ha="right", color=MUTED, fontsize=5.5,
    )
    inset_positions = {
        "records": (0.00, 0.55, 0.47, 0.36),
        "relations": (0.53, 0.55, 0.47, 0.36),
        "folds": (0.00, 0.06, 0.47, 0.36),
        "clauses": (0.53, 0.06, 0.47, 0.36),
    }
    factor_titles = {
        "records": "Records",
        "relations": "Relation types",
        "folds": "Folds",
        "clauses": "Declared clauses",
    }
    for axis_name, position in inset_positions.items():
        mini = ax_c.inset_axes(position)
        points = sorted(
            (point for point in data.scaling if point.axis == axis_name),
            key=lambda point: point.x,
        )
        x = np.array([point.x for point in points], dtype=float)
        median = np.array([point.median_seconds for point in points], dtype=float)
        q1 = np.array([point.q1_seconds for point in points], dtype=float)
        q3 = np.array([point.q3_seconds for point in points], dtype=float)
        mini.fill_between(x, q1, q3, color=TEAL_LIGHT, alpha=0.78, linewidth=0)
        mini.plot(x, median, color=TEAL, linewidth=1.25, marker="o", markersize=3.0)
        for point in points:
            if point.baseline:
                mini.scatter(
                    [point.x], [point.median_seconds], marker="s", s=22,
                    color=TEXT, edgecolor="white", linewidth=0.45, zorder=4,
                )
        mini.set_title(factor_titles[axis_name], loc="left", fontsize=5.9, color=TEXT, pad=2)
        mini.set_ylim(0.0, 3.55)
        mini.set_yticks((0, 1, 2, 3))
        mini.set_xticks(x, [f"{int(value)}{'*' if point.baseline else ''}" for value, point in zip(x, points)])
        mini.grid(axis="y", color=GRID, linewidth=0.42)
        mini.spines[["top", "right"]].set_visible(False)
        mini.tick_params(length=2.0, width=0.55, pad=1.5)
        if axis_name in {"records", "folds"}:
            mini.set_ylabel("Compile time (s)", labelpad=1.2)
        else:
            mini.set_yticklabels([])
    ax_c.text(
        0.50, -0.09, "* same frozen baseline reused across paths",
        transform=ax_c.transAxes, ha="center", va="bottom", fontsize=5.2, color=MUTED,
    )

    # (d) Terminal outcomes and the deliberately bounded high-combined case.
    ax_d.axis("off")
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.text(
        0.0, 1.05, "d  Terminal classes and high-combined cost",
        transform=ax_d.transAxes, fontweight="bold", color=TEXT, fontsize=7.4,
    )
    counts = [
        ("split-exact", data.status_counts["split_exact"], TEAL),
        ("certified", data.status_counts["certified_refusal"], PURPLE),
        ("unresolved", data.status_counts["search_exhaustion_unresolved"], AMBER),
    ]
    x0, total, bar_y, bar_h = 0.02, sum(item[1] for item in counts), 0.78, 0.105
    cursor = x0
    usable = 0.96
    for label, count, color in counts:
        width = usable * count / total
        ax_d.add_patch(Rectangle((cursor, bar_y), width, bar_h, facecolor=color, edgecolor="white", linewidth=0.7))
        ax_d.text(cursor + width / 2, bar_y + bar_h / 2, str(count), ha="center", va="center", color="white", fontweight="bold", fontsize=6.2)
        cursor += width
    legend_y = (0.68, 0.61, 0.54)
    for (label, count, color), y in zip(counts, legend_y):
        ax_d.scatter([0.045], [y], s=18, color=color)
        ax_d.text(0.085, y, f"{label}  {count}/68", va="center", fontsize=5.8, color=TEXT)
    ax_d.text(0.98, 0.68, "0 false-exact", ha="right", va="center", fontsize=5.8, color=TEAL, fontweight="bold")

    # A ruled, directly labelled evidence strip replaces the former dashboard card.
    ax_d.plot([0.02, 0.98], [0.44, 0.44], color=GRID, linewidth=0.85)
    ax_d.plot([0.02, 0.98], [0.08, 0.08], color=GRID, linewidth=0.65)
    ax_d.text(0.02, 0.405, "HIGH COMBINED", color=MUTED, fontsize=5.4, fontweight="bold")
    ax_d.text(0.02, 0.335, "960 records · 8 relations · 8 folds · 14 clauses", color=TEXT, fontsize=5.2)
    metrics = [
        ("Compile", f"{data.high_combined['compile_seconds']:.3f} s"),
        ("Checker", f"{data.high_combined['checker_ms']:.1f} ms"),
        ("Peak RSS +", f"{data.high_combined['rss_mib']:.2f} MiB"),
    ]
    column_centres = (0.18, 0.50, 0.82)
    for separator in (0.34, 0.66):
        ax_d.plot([separator, separator], [0.115, 0.285], color=GRID, linewidth=0.65)
    for index, (label, value) in enumerate(metrics):
        xx = column_centres[index]
        ax_d.text(xx, 0.245, label, ha="center", color=MUTED, fontsize=5.2)
        ax_d.text(xx, 0.145, value, ha="center", color=TEXT, fontsize=7.5, fontweight="bold")

    fig.text(
        0.075, 0.035,
        "Dots are mean ClaimCut−comparator ERE; right caps are simultaneous one-sided 95% Bonferroni UCBs.\n"
        "Family: 7 comparisons; 100 clustered replicates; 10,000 resamples. The 0.02 boundary applies only to "
        "S0 ClaimCut−random noninferiority.\n"
        "Scaling paths describe the frozen grid and are not asymptotic complexity estimates.",
        ha="left", va="bottom", fontsize=5.35, color=MUTED, linespacing=1.35,
    )
    return _atomic_export(fig, output_root)


def _write_receipt(data: Figure4Data, output_root: Path, outputs: Sequence[Path]) -> Path:
    review_root = output_root.parent / "reviews"
    review_root.mkdir(parents=True, exist_ok=True)
    target = review_root / "Figure_4_build_receipt.json"
    payload = {
        "schema_version": "figure4-primary-scaling-receipt-v1",
        "input_receipt": data.receipt,
        "scientific_values": {
            "s0_claimcut_minus_random": {
                "mean": next(row.mean_difference for row in data.s0 if row.comparator == "RandomKFold"),
                "simultaneous_ucb": next(row.simultaneous_ucb for row in data.s0 if row.comparator == "RandomKFold"),
                "noninferiority_margin": 0.02,
            },
            "s3_supplier_ucb": next(row.simultaneous_ucb for row in data.s3 if row.comparator == "GroupKFold_Supplier"),
            "s3_datasail_ucb": next(row.simultaneous_ucb for row in data.s3 if row.comparator == "DataSAIL_C1e_Scalar"),
            "terminal_counts": dict(data.status_counts),
            "high_combined": dict(data.high_combined),
        },
        "outputs": {
            path.suffix.lstrip("."): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in outputs
        },
    }
    temporary = target.with_name(f".__{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--aggregate-root",
        type=Path,
        required=True,
        help="Closed public a32 aggregate directory; excluded attempts are refused.",
    )
    parser.add_argument(
        "--scalability-root",
        type=Path,
        required=True,
        help="Frozen public scalability_v1 directory.",
    )
    parser.add_argument(
        "--primary-evidence-lock",
        type=Path,
        default=PUBLIC_FIGURE_ROOT / "PRIMARY_EVIDENCE_LOCK.json",
        help="Post-run lock created only after complete execution and the registered selection gate.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--retention-receipt",
        type=Path,
        help="Attempt-S retain-all S0/S1/S3 receipt.",
    )
    selection.add_argument(
        "--comparison-receipt",
        type=Path,
        help=(
            "Historical comparison receipt; unavailable in the Attempt-S final RC."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PUBLIC_FIGURE_ROOT / "output",
    )
    args = parser.parse_args()
    try:
        lock_preview = _read_json(args.primary_evidence_lock)
        attempt_s_mode = (
            lock_preview.get("schema_version") == ATTEMPT_S_LOCK_SCHEMA
        )
        selection_receipt = args.retention_receipt or args.comparison_receipt
        if selection_receipt is None:
            selection_receipt = (
                DEFAULT_RETENTION_RECEIPT
                if attempt_s_mode
                else DEFAULT_COMPARISON_RECEIPT
            )
        data = load_verified_data(
            args.aggregate_root,
            args.scalability_root,
            args.primary_evidence_lock,
            selection_receipt,
        )
        if args.check_only:
            if attempt_s_mode:
                print(
                    "Figure 4 gate PASS: Attempt-S retain-all S0/S1/S3 gate, "
                    "S0/S3 plotting aggregates, and scalability-v1 evidence are closed"
                )
            else:
                print(
                    "Figure 4 gate PASS: registered S0/S1/S3 selection gate, "
                    "S0/S3 plotting aggregates, and scalability-v1 evidence are closed"
                )
            return
        outputs = build_figure(data, args.output_root)
        receipt = _write_receipt(data, args.output_root, outputs)
    except Figure4GateError as exc:
        raise SystemExit(f"REFUSED Figure 4: {exc}") from exc
    print("\n".join(str(path) for path in (*outputs, receipt)))


if __name__ == "__main__":
    main()
