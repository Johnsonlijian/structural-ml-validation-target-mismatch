from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "draft"
AUDIT_SUMMARY = ROOT / "code" / "outputs" / "literature" / "reproducible_candidate_audit_summary.csv"
MANUAL = ROOT / "code" / "outputs" / "literature" / "manual_reproducibility_verification_v0.csv"


def box(ax, xy, width, height, title, body, color, title_color="#222222"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=color,
        facecolor="#FFFFFF",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    x, y = xy
    ax.text(x + 0.025, y + height - 0.055, title, transform=ax.transAxes, fontsize=10.5, weight="bold", color=title_color)
    ax.text(x + 0.025, y + height - 0.105, body, transform=ax.transAxes, fontsize=8.5, color="#333333", va="top", linespacing=1.25)


def arrow(ax, start, end, color="#666666", rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.2,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(AUDIT_SUMMARY)
    manual = pd.read_csv(MANUAL)

    priority_counts = summary.groupby("reproduction_priority")["n"].sum().to_dict()
    openalex_high = manual[manual["source"].eq("OpenAlex high")]
    opportunistic = manual[manual["source"].eq("Opportunistic")]

    total_openalex = 3919
    screened = 332
    completed_metadata = int(priority_counts.get("completed", 0))
    high = int(priority_counts.get("high", 0))
    medium = int(priority_counts.get("medium", 0))
    low = int(priority_counts.get("low", 0))

    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.axis("off")

    ax.text(
        0.02,
        0.965,
        "Study design: source-aware meta-reproduction separates mechanism evidence from field-wide inference",
        transform=ax.transAxes,
        fontsize=14,
        weight="bold",
    )
    ax.text(
        0.02,
        0.925,
        "The workflow first builds a literature sampling frame, then tests executable public cases, then audits which candidates can enter the next reproduction wave.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#444444",
    )

    box(
        ax,
        (0.03, 0.69),
        0.23,
        0.17,
        "OpenAlex search",
        f"{total_openalex:,} metadata records\nstructural/civil ML query\nbroad, noisy discovery layer",
        "#4C78A8",
        "#4C78A8",
    )
    box(
        ax,
        (0.34, 0.69),
        0.25,
        0.17,
        "Strict screened frame",
        f"{screened} structural-ML candidates\ncompleted={completed_metadata}, high={high}\nmedium={medium}, low={low}",
        "#4C78A8",
        "#4C78A8",
    )
    box(
        ax,
        (0.68, 0.69),
        0.27,
        0.17,
        "Manual verification queue",
        f"{len(openalex_high)} metadata-high records checked\nrepository hint != raw data\nattrition reported instead of hidden",
        "#E45756",
        "#E45756",
    )
    arrow(ax, (0.265, 0.775), (0.335, 0.775), "#4C78A8")
    arrow(ax, (0.595, 0.775), (0.675, 0.775), "#E45756")

    box(
        ax,
        (0.03, 0.39),
        0.28,
        0.19,
        "Executable reproduction modules",
        "6 public case-study modules\nclassification + regression\nsource/mix/family grouping tested",
        "#54A24B",
        "#54A24B",
    )
    box(
        ax,
        (0.37, 0.39),
        0.25,
        0.19,
        "Validation targets",
        "RandomKFold: within-source interpolation\nGroupKFold / LOSO: unseen source\nfamily split: unseen structural family",
        "#54A24B",
        "#54A24B",
    )
    box(
        ax,
        (0.68, 0.39),
        0.27,
        0.19,
        "Observed pattern",
        "random scores are equal or higher\noptimism is heterogeneous\ncase evidence, not prevalence estimate",
        "#54A24B",
        "#54A24B",
    )
    arrow(ax, (0.31, 0.485), (0.365, 0.485), "#54A24B")
    arrow(ax, (0.62, 0.485), (0.675, 0.485), "#54A24B")
    arrow(ax, (0.46, 0.69), (0.20, 0.585), "#777777", rad=0.10)

    box(
        ax,
        (0.03, 0.105),
        0.28,
        0.18,
        "Dataset-first expansion",
        f"{len(opportunistic)} promoted datasets\nMendeley resolved and reproduced\nDesignSafe remains token/browser-gated",
        "#F58518",
        "#F58518",
    )
    box(
        ax,
        (0.37, 0.105),
        0.25,
        0.18,
        "Reporting outputs",
        "Figure 2: validation gaps\nFigure 3: reproducibility audit\nFigure 4: checklist\nFigure 5: Mendeley joint cases",
        "#F58518",
        "#F58518",
    )
    box(
        ax,
        (0.68, 0.105),
        0.27,
        0.18,
        "Submission discipline",
        "do not infer field-wide prevalence yet\nreport failed access and source topology\nexpand reproductions before NMI/NC claim",
        "#B279A2",
        "#B279A2",
    )
    arrow(ax, (0.815, 0.69), (0.19, 0.29), "#F58518", rad=0.16)
    arrow(ax, (0.31, 0.195), (0.365, 0.195), "#F58518")
    arrow(ax, (0.62, 0.195), (0.675, 0.195), "#B279A2")

    ax.text(
        0.02,
        0.035,
        "Abbreviations: LOSO, leave-one-source-out; NMI/NC, Nature Machine Intelligence / Nature Communications. "
        "Counts reflect the current v0 audit and are intentionally separated from the manually discovered case-study layer.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )

    fig.savefig(OUT / "fig01_study_design_v0.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig01_study_design_v0.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
