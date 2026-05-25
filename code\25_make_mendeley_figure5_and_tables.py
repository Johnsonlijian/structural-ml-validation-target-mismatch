from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "code" / "outputs" / "reproductions" / "mendeley_beam_column_joint"
FIG = ROOT / "figures" / "generated"
TABLES = ROOT / "manuscript" / "tables"

RESULTS = REPRO / "results.csv"
GROUPS = REPRO / "group_sizes.csv"

TASK_LABELS = {
    "shear_strength_regression": "Exterior joint\nshear strength",
    "failure_mode_classification": "Exterior joint\nfailure mode",
    "joint_shear_strength_regression": "Cyclic joint\nshear strength",
}

METRIC_LABELS = {
    "shear_strength_regression": "Pooled R2",
    "failure_mode_classification": "Accuracy",
    "joint_shear_strength_regression": "Pooled R2",
}

SPLIT_LABELS = {
    "RandomKFold": "Random",
    "GroupKFold": "GroupKFold",
    "LeaveOneSourceOut": "LOSO",
}

COLORS = {
    "RandomKFold": "#4C78A8",
    "GroupKFold": "#E45756",
    "LeaveOneSourceOut": "#72B7B2",
}


def setup() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def metric_value(row: pd.Series) -> float:
    if pd.notna(row.get("pooled_r2")):
        return float(row["pooled_r2"])
    return float(row["accuracy"])


def make_summary_tables(results: pd.DataFrame, groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rf = results[results["model"].eq("Random Forest")].copy()
    rows = []
    for task, sub in rf.groupby("task", sort=False):
        random = sub[sub["split"].eq("RandomKFold")].iloc[0]
        group = sub[sub["split"].eq("GroupKFold")].iloc[0]
        logo = sub[sub["split"].eq("LeaveOneSourceOut")].iloc[0]
        rows.append(
            {
                "task": task,
                "display_task": TASK_LABELS[task].replace("\n", " "),
                "metric": METRIC_LABELS[task],
                "random": metric_value(random),
                "group_kfold": metric_value(group),
                "leave_one_source_out": metric_value(logo),
                "gap_random_minus_group": float(group["gap_vs_random"]),
                "n_samples": int(group["n_samples"]),
                "n_groups": int(group["n_groups"]),
                "median_group_size": float(group["median_group_size"]),
                "max_group_size": int(group["max_group_size"]),
            }
        )
    metric_summary = pd.DataFrame(rows)

    topology = (
        groups.groupby(["case", "task"], as_index=False)
        .agg(
            n_samples=("n", "sum"),
            n_groups=("group", "count"),
            median_group_size=("n", "median"),
            max_group_size=("n", "max"),
            singleton_groups=("n", lambda s: int((s == 1).sum())),
            groups_le_2=("n", lambda s: int((s <= 2).sum())),
            groups_le_3=("n", lambda s: int((s <= 3).sum())),
        )
        .assign(singleton_share=lambda d: d["singleton_groups"] / d["n_groups"])
    )

    metric_summary.to_csv(TABLES / "mendeley_beam_column_joint_metrics_v0.csv", index=False)
    topology.to_csv(TABLES / "mendeley_beam_column_joint_group_topology_v0.csv", index=False)

    write_markdown(
        TABLES / "mendeley_beam_column_joint_metrics_v0.md",
        "# Mendeley Beam-Column Joint Metrics v0",
        metric_summary,
    )
    write_markdown(
        TABLES / "mendeley_beam_column_joint_group_topology_v0.md",
        "# Mendeley Beam-Column Joint Group Topology v0",
        topology,
    )
    return metric_summary, topology


def write_markdown(path: Path, title: str, df: pd.DataFrame) -> None:
    lines = [title, ""]
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, row in df.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(results: pd.DataFrame, topology: pd.DataFrame) -> None:
    rf = results[results["model"].eq("Random Forest")].copy()
    tasks = [
        "shear_strength_regression",
        "failure_mode_classification",
        "joint_shear_strength_regression",
    ]
    splits = ["RandomKFold", "GroupKFold", "LeaveOneSourceOut"]

    # Middle column = empty gutter so B/C barh y-tick labels do not overlap panel A (wspace alone was insufficient).
    fig = plt.figure(figsize=(14.2, 7.4))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.45, 0.22, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.12,
        hspace=0.38,
    )
    ax_perf = fig.add_subplot(gs[:, 0])
    ax_gap = fig.add_subplot(gs[0, 2])
    ax_top = fig.add_subplot(gs[1, 2])

    x = np.arange(len(tasks))
    width = 0.23
    for i, split in enumerate(splits):
        vals = []
        for task in tasks:
            row = rf[(rf["task"].eq(task)) & (rf["split"].eq(split))].iloc[0]
            vals.append(metric_value(row))
        ax_perf.bar(x + (i - 1) * width, vals, width=width, label=SPLIT_LABELS[split], color=COLORS[split])
    ax_perf.axhline(0, color="#333333", linewidth=0.8)
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels([TASK_LABELS[t] for t in tasks])
    ax_perf.set_ylabel("Headline metric")
    ax_perf.set_title("A. Random validation overstates joint-model performance")
    ax_perf.legend(frameon=False, loc="lower left")
    for i, task in enumerate(tasks):
        metric = METRIC_LABELS[task]
        ax_perf.text(i, ax_perf.get_ylim()[1] * 0.97, metric, ha="center", va="top", fontsize=8, color="#444444")

    gaps = []
    labels = []
    for task in tasks:
        row = rf[(rf["task"].eq(task)) & (rf["split"].eq("GroupKFold"))].iloc[0]
        gaps.append(float(row["gap_vs_random"]))
        labels.append(TASK_LABELS[task].replace("\n", " "))
    ax_gap.barh(np.arange(len(gaps)), gaps, color="#E45756")
    ax_gap.set_yticks(np.arange(len(gaps)))
    ax_gap.set_yticklabels(labels, fontsize=8)
    ax_gap.invert_yaxis()
    ax_gap.axvline(0, color="#333333", linewidth=0.8)
    ax_gap.set_xlabel("Random minus GroupKFold")
    ax_gap.set_title("B. Validation gap")
    for i, gap in enumerate(gaps):
        ax_gap.text(gap + 0.03, i, f"{gap:.3f}", va="center", fontsize=8)

    top_rows = []
    for task in tasks:
        row = topology[topology["task"].eq(task)].iloc[0]
        top_rows.append(row)
    y = np.arange(len(tasks))
    med = [float(r["median_group_size"]) for r in top_rows]
    max_size = [float(r["max_group_size"]) for r in top_rows]
    singleton = [float(r["singleton_groups"]) for r in top_rows]
    ax_top.barh(y - 0.18, med, height=0.18, label="Median group size", color="#4C78A8")
    ax_top.barh(y, max_size, height=0.18, label="Max group size", color="#72B7B2")
    ax_top.barh(y + 0.18, singleton, height=0.18, label="Singleton groups", color="#F58518")
    ax_top.set_yticks(y)
    ax_top.set_yticklabels(labels, fontsize=8)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Count")
    ax_top.set_title("C. Source-group topology")
    ax_top.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Dataset-first beam-column joint reproductions expose source-aware validation gaps", fontsize=14, y=0.99)
    fig.text(
        0.01,
        0.01,
        "Source proxies: reference-coded Authors for the 203-test exterior joint dataset; Research Team for the 98-test cyclic joint dataset. "
        "LOSO = leave-one-source-out.",
        fontsize=8,
        color="#444444",
    )
    fig.savefig(FIG / "fig05_mendeley_beam_column_joint_validation_v0.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig05_mendeley_beam_column_joint_validation_v0.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    results = pd.read_csv(RESULTS)
    groups = pd.read_csv(GROUPS)
    _, topology = make_summary_tables(results, groups)
    make_figure(results, topology)


if __name__ == "__main__":
    main()
