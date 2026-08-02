"""Build Figure 3 from the final compiler-semantic a32 aggregate only."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

from figure_aggregate_gate import (
    AggregateGateError,
    COMPILER_SCHEMA,
    read_stage_json,
    read_stage_csv,
    validate_stage_manifests,
)


EXPECTED_METRICS = (
    "zero_false_exact",
    "native_exact_realization",
    "plan_reproducibility",
    "assignment_reproducibility",
    "outcome_invariance",
    "typed_refusal_accuracy",
    "independent_oracle_agreement",
    "mutation_detection",
    "production_mutation_detection",
    "oracle_production_mutation_agreement",
    "bruteforce_status_agreement",
)

METRIC_LABELS = {
    "zero_false_exact": "Zero false-exact events",
    "native_exact_realization": "Native split-exact realization",
    "plan_reproducibility": "Plan reproducibility",
    "assignment_reproducibility": "Assignment reproducibility",
    "outcome_invariance": "Outcome-column exclusion",
    "typed_refusal_accuracy": "Typed refusal accuracy",
    "independent_oracle_agreement": "Separate-implementation agreement",
    "mutation_detection": "Checker mutation detection",
    "production_mutation_detection": "Production mutation detection",
    "oracle_production_mutation_agreement": "Checker-production agreement",
    "bruteforce_status_agreement": "Exhaustive-status agreement",
}

NAVY = "#163A5F"
TEAL = "#147D82"
AMBER = "#D98E04"
VERMILION = "#C23B22"
VIOLET = "#7057A3"
TEXT = "#17212B"
MUTED = "#5F6B76"


def load_verified_counts(aggregate_root: Path):
    stages = validate_stage_manifests(
        aggregate_root, ("COMPILER_SEMANTIC",), expected_schema=COMPILER_SCHEMA
    )
    stage = stages["COMPILER_SEMANTIC"]
    hard_gate = read_stage_json(stage, "compiler_gate.json")
    required_hard_gate = {
        "n_complete": 100,
        "n_failed": 0,
        "false_exact_events_total": 0,
        "all_primary_observations_pass": True,
    }
    if not isinstance(hard_gate, dict):
        raise AggregateGateError("COMPILER_SEMANTIC: hard gate is absent")
    for field, expected in required_hard_gate.items():
        if hard_gate.get(field) != expected:
            raise AggregateGateError(
                f"COMPILER_SEMANTIC: hard gate failed for {field}"
            )
    rows = read_stage_csv(
        stage,
        "compiler_conformance_counts.csv",
        required_columns=("metric", "successes", "total", "rate"),
        expected_rows=len(EXPECTED_METRICS),
    )
    by_metric = {row["metric"]: row for row in rows}
    if set(by_metric) != set(EXPECTED_METRICS):
        raise AggregateGateError(
            "COMPILER_SEMANTIC: deterministic metric set is incomplete or changed"
        )
    for metric in EXPECTED_METRICS:
        row = by_metric[metric]
        try:
            successes = int(row["successes"])
            total = int(row["total"])
            rate = float(row["rate"])
        except Exception as exc:
            raise AggregateGateError(
                f"COMPILER_SEMANTIC: malformed metric row for {metric}"
            ) from exc
        if successes != 100 or total != 100 or not math.isclose(rate, 1.0):
            raise AggregateGateError(
                f"COMPILER_SEMANTIC: conformance did not close at 100/100 for {metric}"
            )
    return hard_gate, [by_metric[name] for name in EXPECTED_METRICS]


def _atomic_export(fig, output_root: Path) -> list[Path]:
    targets = {
        "svg": output_root / "svg" / "Figure_3.svg",
        "pdf": output_root / "pdf" / "Figure_3.pdf",
        "png": output_root / "png" / "Figure_3.png",
    }
    for target in targets.values():
        target.parent.mkdir(parents=True, exist_ok=True)
    temporaries: dict[str, Path] = {}
    try:
        for kind, target in targets.items():
            handle, temporary_name = tempfile.mkstemp(
                prefix=".__Figure_3_", suffix=f".{kind}", dir=target.parent
            )
            os.close(handle)
            temporary = Path(temporary_name)
            temporaries[kind] = temporary
            fig.savefig(
                temporary,
                format=kind,
                dpi=360 if kind == "png" else None,
                bbox_inches="tight",
                facecolor="white",
            )
        for kind, target in targets.items():
            os.replace(temporaries[kind], target)
        return list(targets.values())
    finally:
        for temporary in temporaries.values():
            if temporary.exists():
                temporary.unlink()


def build_figure(hard_gate, rows, output_root: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.2,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(13.2, 7.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.77, 1.65), width_ratios=(1.7, 1.0))
    lifecycle = fig.add_subplot(grid[0, :])
    metrics = fig.add_subplot(grid[1, 0])
    gate = fig.add_subplot(grid[1, 1])

    lifecycle.set_xlim(0, 1)
    lifecycle.set_ylim(0, 1)
    lifecycle.axis("off")
    lifecycle.set_title(
        "(a) Lifecycle conformance is checked outside the compiler",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
    )
    stages = [
        (0.07, "Typed contract", "outcome excluded", NAVY),
        (0.285, "Compiler", "plan + status", TEAL),
        (0.505, "Realization audit", "assignment predicates", VIOLET),
        (0.735, "Separate implementation", "certificate + brute force", AMBER),
        (0.945, "Card", "hashes", TEXT),
    ]
    lifecycle.plot(
        [stages[0][0], stages[-1][0]], [0.49, 0.49],
        color="#BCC6CE", linewidth=1.0, zorder=0,
    )
    for left, right in zip(stages[:-1], stages[1:]):
        lifecycle.add_patch(
            FancyArrowPatch(
                (left[0] + 0.022, 0.49),
                (right[0] - 0.022, 0.49),
                arrowstyle="-|>", mutation_scale=10, linewidth=1.05,
                color="#7C8994", zorder=1,
            )
        )
    for x, title, subtitle, color in stages:
        lifecycle.plot([x, x], [0.415, 0.565], color=color, linewidth=1.7, zorder=2)
        lifecycle.scatter(
            [x], [0.49], s=32, facecolor="white", edgecolor=color,
            linewidth=1.4, zorder=3,
        )
        lifecycle.text(
            x, 0.68, title, ha="center", va="center", fontweight="bold",
            color=color, fontsize=8.8,
        )
        lifecycle.text(
            x, 0.29, subtitle, ha="center", va="center", fontsize=8.0,
            color=MUTED,
        )
    lifecycle.text(
        0.5, 0.11,
        "Deterministic software stress cases; arrows encode transformation and separately implemented checks.",
        ha="center", color=MUTED, fontsize=8.4,
    )

    labels = [METRIC_LABELS[row["metric"]] for row in rows]
    values = [int(row["successes"]) for row in rows]
    y = list(range(len(labels)))
    colors = [TEAL, TEAL, NAVY, NAVY, NAVY, AMBER, VIOLET, VIOLET, VIOLET, VIOLET, AMBER]
    metrics.barh(y, values, color=colors, alpha=0.88, height=0.62)
    metrics.set_yticks(y, labels)
    metrics.invert_yaxis()
    metrics.set_xlim(0, 108)
    metrics.set_xticks((0, 25, 50, 75, 100))
    metrics.set_xlabel("Deterministic cases satisfying the property (of 100)")
    metrics.set_title("(b) Separately checked conformance counts", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    metrics.axvline(100, color=TEXT, linewidth=0.9, linestyle="--")
    for yy, value in zip(y, values):
        metrics.text(101.0, yy, f"{value}/100", va="center", fontsize=8.2, color=TEXT)
    metrics.spines[["top", "right", "left"]].set_visible(False)
    metrics.grid(axis="x", color="#E4E9ED", linewidth=0.7)
    metrics.set_axisbelow(True)

    gate.axis("off")
    gate.set_xlim(0, 1)
    gate.set_ylim(0, 1)
    gate.set_title("(c) Frozen hard gate", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    criteria = [
        ("COMPLETE", str(hard_gate["n_complete"]), TEAL),
        ("FAILED", str(hard_gate["n_failed"]), TEXT),
        ("FALSE EXACT", str(hard_gate["false_exact_events_total"]), VERMILION),
        ("PRIMARY PROPERTIES", "PASS", VIOLET),
    ]
    gate.text(0.075, 0.90, "AUDIT CRITERION", fontsize=7.5, color=MUTED, fontweight="bold")
    gate.text(0.94, 0.90, "ADMITTED STATE", ha="right", fontsize=7.5, color=MUTED, fontweight="bold")
    gate.plot([0.055, 0.95], [0.86, 0.86], color="#BFC8CF", linewidth=0.85)
    gate.plot([0.69, 0.69], [0.22, 0.86], color="#D9E0E5", linewidth=0.7)
    row_centres = (0.76, 0.61, 0.46, 0.31)
    for (title, value, color), yy in zip(criteria, row_centres):
        gate.plot([0.055, 0.95], [yy - 0.075, yy - 0.075], color="#E2E7EA", linewidth=0.7)
        gate.plot([0.055, 0.055], [yy - 0.045, yy + 0.045], color=color, linewidth=2.2)
        gate.text(0.085, yy, title, ha="left", va="center", fontsize=8.5, color=TEXT, fontweight="bold")
        gate.text(0.94, yy, value, ha="right", va="center", fontsize=13.2, color=color, fontweight="bold")
    gate.text(
        0.5, 0.045,
        "Counts are conformance checks, not a sampled population; no binomial confidence intervals are used.",
        transform=gate.transAxes, ha="center", va="bottom", wrap=True, fontsize=8.4, color=MUTED,
    )

    return _atomic_export(fig, output_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate-root",
        type=Path,
        required=True,
        help="Closed public a32 aggregate directory; excluded attempt trees are refused.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        hard_gate, rows = load_verified_counts(args.aggregate_root)
        if args.check_only:
            print("Figure 3 gate PASS: final compiler aggregate is 100/100 and hash-bound")
            return
        outputs = build_figure(hard_gate, rows, args.output_root)
    except AggregateGateError as exc:
        raise SystemExit(f"REFUSED Figure 3: {exc}") from exc
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
