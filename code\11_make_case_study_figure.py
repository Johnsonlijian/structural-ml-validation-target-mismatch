"""
Build manuscript Figure 2 v0 from the first three real-data reproductions.

Outputs:
    figures/generated/fig02_case_study_matrix_v0.png
    figures/generated/fig02_case_study_matrix_v0.pdf
    code/outputs/figures/fig02_case_study_matrix_data.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
OUT_DATA = CODE_DIR / "outputs" / "figures"
OUT_FIG = ROOT / "figures" / "generated"


def load_case_data() -> pd.DataFrame:
    """Use the same headline tree model for each real-data case."""
    mangalathu = pd.read_csv(
        CODE_DIR / "outputs" / "reproductions" / "10-1016-j-engstruct-2019-110331" / "results.csv"
    )
    rahman = pd.read_csv(
        CODE_DIR / "outputs" / "reproductions" / "10-1016-j-engstruct-2020-111743" / "results.csv"
    )
    nguyen = pd.read_csv(
        CODE_DIR / "outputs" / "reproductions" / "10-1016-j-conbuildmat-2020-120950" / "results.csv"
    )

    m_rf = mangalathu[mangalathu["model"] == "RandomForest"].set_index("cv")
    r_rf = rahman[rahman["model"] == "RandomForest"].set_index("validation")
    n_rf = nguyen[nguyen["model"] == "RandomForest"].set_index("validation")

    rows = [
        {
            "case_id": 1,
            "case_label": "Mangalathu 2020\nRC shear wall",
            "short_label": "Shear wall\nclassification",
            "task": "classification",
            "metric_name": "accuracy",
            "group_proxy": "Author/source family",
            "random_metric": m_rf.loc["RandomStratifiedKFold-5", "accuracy_mean"],
            "group_metric": m_rf.loc["GroupKFold-by-Author-5", "accuracy_mean"],
            "gap": m_rf.loc["RandomStratifiedKFold-5", "accuracy_mean"]
            - m_rf.loc["GroupKFold-by-Author-5", "accuracy_mean"],
            "n_samples": 393,
            "n_groups": 75,
            "interpretation": "strong collapse",
        },
        {
            "case_id": 2,
            "case_label": "Rahman 2021 /\nLantsoght SFRC",
            "short_label": "SFRC shear\nregression",
            "task": "regression",
            "metric_name": "R2",
            "group_proxy": "Source reference",
            "random_metric": r_rf.loc["RandomKFold", "r2_mean"],
            "group_metric": r_rf.loc["GroupKFold_Source", "r2_mean"],
            "gap": r_rf.loc["RandomKFold", "r2_mean"] - r_rf.loc["GroupKFold_Source", "r2_mean"],
            "n_samples": int(r_rf.loc["RandomKFold", "n_samples"]),
            "n_groups": int(r_rf.loc["LeaveOneSourceOut", "n_splits"]),
            "interpretation": "strong collapse",
        },
        {
            "case_id": 3,
            "case_label": "Nguyen 2021 /\nUCI concrete",
            "short_label": "Concrete strength\nregression",
            "task": "regression",
            "metric_name": "pooled R2",
            "group_proxy": "Mix proportions",
            "random_metric": n_rf.loc["RandomKFold", "pooled_r2"],
            "group_metric": n_rf.loc["GroupKFold_Mix", "pooled_r2"],
            "gap": n_rf.loc["RandomKFold", "pooled_r2"] - n_rf.loc["GroupKFold_Mix", "pooled_r2"],
            "n_samples": int(n_rf.loc["RandomKFold", "n_samples"]),
            "n_groups": int(n_rf.loc["RandomKFold", "n_groups"]),
            "interpretation": "mild optimism",
        },
    ]
    return pd.DataFrame(rows)


def draw_validation_schematic(ax) -> None:
    ax.set_axis_off()
    ax.set_title("A  Validation split logic", loc="left", fontweight="bold")

    colors = ["#4C78A8", "#F58518", "#54A24B"]
    x0 = 0.05
    y0 = 0.62
    w = 0.08
    h = 0.18

    for i in range(9):
        ax.add_patch(Rectangle((x0 + i * (w + 0.01), y0), w, h, color=colors[i % 3], alpha=0.85))
    ax.text(0.05, 0.84, "Random CV: same source families can cross folds", fontsize=9)
    ax.text(0.05, 0.53, "train / test mixed within source", fontsize=8, color="#555555")

    y1 = 0.18
    for i, c in enumerate(colors):
        ax.add_patch(Rectangle((0.05 + i * 0.24, y1), 0.18, h, color=c, alpha=0.85))
    ax.text(0.05, 0.40, "Group-aware CV: source family held out together", fontsize=9)
    ax.text(0.05, 0.09, "test = unseen source or mix family", fontsize=8, color="#555555")

    arrow = FancyArrowPatch((0.88, 0.71), (0.88, 0.28), arrowstyle="-|>", mutation_scale=14, color="#333333")
    ax.add_patch(arrow)
    ax.text(0.80, 0.48, "harder\ngeneralization", fontsize=8, ha="center", va="center")


def draw_performance_bars(ax, data: pd.DataFrame) -> None:
    ax.set_title("B  Random vs group-aware performance", loc="left", fontweight="bold")
    x = np.arange(len(data))
    width = 0.34
    ax.bar(x - width / 2, data["random_metric"], width, label="Random CV", color="#4C78A8")
    ax.bar(x + width / 2, data["group_metric"], width, label="Group-aware CV", color="#F58518")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["short_label"], fontsize=8)
    ax.set_ylabel("Headline metric")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    for i, row in data.iterrows():
        ax.text(i, max(row["random_metric"], row["group_metric"]) + 0.06, row["metric_name"], ha="center", fontsize=8)


def draw_gap_bars(ax, data: pd.DataFrame) -> None:
    ax.set_title("C  Validation optimism gap", loc="left", fontweight="bold")
    colors = ["#E45756" if gap > 0.25 else "#72B7B2" for gap in data["gap"]]
    ax.bar(np.arange(len(data)), data["gap"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data["short_label"], fontsize=8)
    ax.set_ylabel("Random - group metric")
    for i, gap in enumerate(data["gap"]):
        ax.text(i, gap + 0.04, f"{gap:.3f}", ha="center", fontsize=8)


def draw_topology(ax, data: pd.DataFrame) -> None:
    ax.set_title("D  Dataset topology", loc="left", fontweight="bold")
    x = np.arange(len(data))
    ax.bar(x, data["n_samples"], color="#B279A2", label="samples")
    ax.bar(x, data["n_groups"], color="#59A14F", label="groups")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(data["short_label"], fontsize=8)
    ax.set_ylabel("Count (log scale)")
    ax.legend(frameon=False, fontsize=8)
    for i, row in data.iterrows():
        ax.text(i, row["n_samples"] * 1.18, f"n={int(row['n_samples'])}\ng={int(row['n_groups'])}", ha="center", fontsize=8)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    data = load_case_data()
    data.to_csv(OUT_DATA / "fig02_case_study_matrix_data.csv", index=False)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    draw_validation_schematic(axes[0, 0])
    draw_performance_bars(axes[0, 1], data)
    draw_gap_bars(axes[1, 0], data)
    draw_topology(axes[1, 1], data)

    fig.suptitle(
        "Random validation optimism is heterogeneous across civil-engineering ML datasets",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_FIG / "fig02_case_study_matrix_v0.png", dpi=300)
    fig.savefig(OUT_FIG / "fig02_case_study_matrix_v0.pdf")
    plt.close(fig)

    print(f"[OK] {OUT_DATA / 'fig02_case_study_matrix_data.csv'}")
    print(f"[OK] {OUT_FIG / 'fig02_case_study_matrix_v0.png'}")
    print(f"[OK] {OUT_FIG / 'fig02_case_study_matrix_v0.pdf'}")


if __name__ == "__main__":
    main()
