"""
Build Figure 2 v1 for the P1 manuscript.

Main Figure 2 uses four source/mix validation cases:
1. Mangalathu 2020 RC shear-wall classification
2. Rahman 2021 / Lantsoght 2019 SFRC shear regression
3. Nguyen 2021 / UCI concrete strength regression
4. Matthews et al. 2023 corroded RC beam moment regression

Stub-CFST is exported as a supplementary deployment-target figure because it
tests unseen section-family extrapolation rather than source-reference leakage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
REPRO_DIR = CODE_DIR / "outputs" / "reproductions"
OUT_DATA = CODE_DIR / "outputs" / "figures"
OUT_FIG = ROOT / "figures" / "draft"

BLUE = "#4C78A8"
ORANGE = "#F58518"
RED = "#E45756"
TEAL = "#72B7B2"
PURPLE = "#B279A2"
GREEN = "#59A14F"
GREY = "#6B6B6B"
LIGHT_GREY = "#E6E6E6"


def read_results(doi_slug: str) -> pd.DataFrame:
    return pd.read_csv(REPRO_DIR / doi_slug / "results.csv")


def load_main_cases() -> pd.DataFrame:
    mangalathu = read_results("10-1016-j-engstruct-2019-110331")
    rahman = read_results("10-1016-j-engstruct-2020-111743")
    nguyen = read_results("10-1016-j-conbuildmat-2020-120950")
    corroded = read_results("10-5281-zenodo-8062007")

    m_rf = mangalathu[mangalathu["model"] == "RandomForest"].set_index("cv")
    r_rf = rahman[rahman["model"] == "RandomForest"].set_index("validation")
    n_rf = nguyen[nguyen["model"] == "RandomForest"].set_index("validation")
    c_rf = corroded[corroded["model"] == "RandomForest"].set_index("validation")

    rows = [
        {
            "case_id": 1,
            "case_label": "Mangalathu 2020\nRC shear wall",
            "short_label": "Shear wall\nfailure mode",
            "domain": "RC walls",
            "task": "classification",
            "metric_name": "Accuracy",
            "random_metric": m_rf.loc["RandomStratifiedKFold-5", "accuracy_mean"],
            "group_metric": m_rf.loc["GroupKFold-by-Author-5", "accuracy_mean"],
            "group_proxy": "Author/source",
            "n_samples": 393,
            "n_groups": 75,
            "topology_note": "median group n=3; 26 singletons",
            "severity": "large",
        },
        {
            "case_id": 2,
            "case_label": "Rahman 2021 /\nLantsoght SFRC",
            "short_label": "SFRC shear\ncapacity",
            "domain": "SFRC beams",
            "task": "regression",
            "metric_name": "Fold-mean R^2",
            "random_metric": r_rf.loc["RandomKFold", "r2_mean"],
            "group_metric": r_rf.loc["GroupKFold_Source", "r2_mean"],
            "group_proxy": "Source reference",
            "n_samples": int(r_rf.loc["RandomKFold", "n_samples"]),
            "n_groups": int(r_rf.loc["LeaveOneSourceOut", "n_splits"]),
            "topology_note": "many small source groups",
            "severity": "large",
        },
        {
            "case_id": 3,
            "case_label": "Nguyen 2021 /\nUCI concrete",
            "short_label": "Concrete\nstrength",
            "domain": "Concrete mixtures",
            "task": "regression",
            "metric_name": "Pooled R^2",
            "random_metric": n_rf.loc["RandomKFold", "pooled_r2"],
            "group_metric": n_rf.loc["GroupKFold_Mix", "pooled_r2"],
            "group_proxy": "Mix proportions",
            "n_samples": int(n_rf.loc["RandomKFold", "n_samples"]),
            "n_groups": int(n_rf.loc["RandomKFold", "n_groups"]),
            "topology_note": "standardized material dataset",
            "severity": "small",
        },
        {
            "case_id": 4,
            "case_label": "Matthews 2023 /\ncorroded RC beams",
            "short_label": "Corroded beam\nmoment",
            "domain": "RC deterioration",
            "task": "regression",
            "metric_name": "Pooled R^2",
            "random_metric": c_rf.loc["RandomKFold", "pooled_r2"],
            "group_metric": c_rf.loc["GroupKFold_Source", "pooled_r2"],
            "group_proxy": "Experimental program",
            "n_samples": int(c_rf.loc["RandomKFold", "n_samples"]),
            "n_groups": int(c_rf.loc["RandomKFold", "n_groups"]),
            "topology_note": "54 programs; median group n=10.5",
            "severity": "moderate",
        },
    ]
    data = pd.DataFrame(rows)
    data["gap"] = data["random_metric"] - data["group_metric"]
    return data


def load_stub_case() -> pd.DataFrame:
    stub = read_results("10-1038-s41598-024-53352-1")
    rf = stub[stub["model"] == "RandomForest"].set_index("validation")
    gb = stub[stub["model"] == "GradientBoosting"].set_index("validation")
    ridge = stub[stub["model"] == "Ridge"].set_index("validation")
    rows = []
    for label, model_df in [("Ridge", ridge), ("Random Forest", rf), ("Gradient Boosting", gb)]:
        random_r2 = model_df.loc["RandomKFold", "pooled_r2"]
        shape_r2 = model_df.loc["LeaveOneShapeOut", "pooled_r2"]
        rows.append(
            {
                "model": label,
                "random_metric": random_r2,
                "shape_metric": shape_r2,
                "gap": random_r2 - shape_r2,
            }
        )
    return pd.DataFrame(rows)


def draw_panel_a(ax) -> None:
    ax.set_axis_off()
    ax.set_title("a  What random folds can hide", loc="left", fontweight="bold", pad=8)

    family_colors = [BLUE, ORANGE, GREEN, PURPLE]
    left_x, top_y = 0.05, 0.62
    cell_w, cell_h = 0.065, 0.14

    ax.text(0.05, 0.88, "Random CV", fontsize=10, fontweight="bold")
    ax.text(0.05, 0.82, "same source families split across train and test", fontsize=8, color=GREY)
    for i in range(12):
        ax.add_patch(
            Rectangle(
                (left_x + i * (cell_w + 0.008), top_y),
                cell_w,
                cell_h,
                facecolor=family_colors[i % 4],
                edgecolor="white",
                linewidth=0.7,
            )
        )
    ax.text(0.05, 0.56, "test resembles training sources", fontsize=8, color=GREY)

    bottom_y = 0.20
    ax.text(0.05, 0.43, "Group-aware CV", fontsize=10, fontweight="bold")
    ax.text(0.05, 0.37, "entire source, mix or program held out", fontsize=8, color=GREY)
    for i, color in enumerate(family_colors[:4]):
        ax.add_patch(
            Rectangle(
                (left_x + i * 0.19, bottom_y),
                0.15,
                cell_h,
                facecolor=color,
                edgecolor="white",
                linewidth=0.7,
            )
        )
    ax.text(0.05, 0.12, "test probes out-of-source generalization", fontsize=8, color=GREY)

    arrow = FancyArrowPatch((0.88, 0.69), (0.88, 0.27), arrowstyle="-|>", mutation_scale=14, color="#333333")
    ax.add_patch(arrow)
    ax.text(0.82, 0.48, "harder\nclaim", fontsize=8, ha="center", va="center")


def draw_panel_b(ax, data: pd.DataFrame) -> None:
    ax.set_title("b  Random validation gives higher headline scores", loc="left", fontweight="bold", pad=8)
    x = np.arange(len(data))
    width = 0.34
    ax.bar(x - width / 2, data["random_metric"], width, label="Random CV", color=BLUE)
    ax.bar(x + width / 2, data["group_metric"], width, label="Group-aware CV", color=ORANGE)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["short_label"], fontsize=8)
    ax.set_ylabel("Headline metric")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_ylim(min(-1.2, data["group_metric"].min() - 0.15), 1.23)

    for i, row in data.iterrows():
        y = max(row["random_metric"], row["group_metric"]) + 0.045
        ax.text(i, y, row["metric_name"], ha="center", va="bottom", fontsize=7, color=GREY)


def draw_panel_c(ax, data: pd.DataFrame) -> None:
    ax.set_title("c  Optimism is a distribution, not a constant", loc="left", fontweight="bold", pad=8)
    colors = [RED if s == "large" else ORANGE if s == "moderate" else TEAL for s in data["severity"]]
    x = np.arange(len(data))
    ax.bar(x, data["gap"], color=colors)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["short_label"], fontsize=8)
    ax.set_ylabel("Random - group metric")
    ax.set_ylim(0, max(1.85, data["gap"].max() + 0.18))
    for i, gap in enumerate(data["gap"]):
        ax.text(i, gap + 0.05, f"{gap:.3f}", ha="center", va="bottom", fontsize=8)


def draw_panel_d(ax, data: pd.DataFrame) -> None:
    ax.set_title("d  Dataset topology explains part of the heterogeneity", loc="left", fontweight="bold", pad=8)
    x = np.arange(len(data))
    width = 0.34
    ax.bar(x - width / 2, data["n_samples"], width, color=PURPLE, label="Specimens/records")
    ax.bar(x + width / 2, data["n_groups"], width, color=GREEN, label="Groups")
    ax.set_yscale("log")
    ax.set_ylabel("Count (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(data["short_label"], fontsize=8)
    ax.set_ylim(45, data["n_samples"].max() * 3.0)
    # Legend: see _legend_panel_d_upper_left() after tight_layout (log y + savefig tight).
    for i, row in data.iterrows():
        ax.text(i, row["n_samples"] * 1.25, f"n={int(row['n_samples'])}\ng={int(row['n_groups'])}", ha="center", fontsize=7)


def style_axes(axes) -> None:
    for ax in np.ravel(axes):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)


def _legend_panel_d_upper_left(fig: plt.Figure, ax_d: plt.Axes):
    """Panel d is log-scaled; ax.legend(..., transAxes) is still often pushed to lower-right
    when rcParams savefig.bbox='tight' runs. Anchor a figure-level legend to the subplot's
    bbox in figure coordinates (upper-left inset)."""
    while fig.legends:
        fig.legends[0].remove()
    pos = ax_d.get_position()
    leg_h = [
        Patch(facecolor=PURPLE, edgecolor="#333333", linewidth=0.5, label="Specimens/records"),
        Patch(facecolor=GREEN, edgecolor="#333333", linewidth=0.5, label="Groups"),
    ]
    leg_l = [h.get_label() for h in leg_h]
    leg = fig.legend(
        leg_h,
        leg_l,
        bbox_to_anchor=(pos.x0 + 0.012 * pos.width, pos.y1 - 0.018 * pos.height),
        bbox_transform=fig.transFigure,
        loc="upper left",
        frameon=False,
        fontsize=8,
        borderaxespad=0,
    )
    leg.set_zorder(30)
    return leg


def make_main_figure(data: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )
    fig = plt.figure(figsize=(12.0, 7.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[1.0, 1.0])
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )

    draw_panel_a(axes[0, 0])
    draw_panel_b(axes[0, 1], data)
    draw_panel_c(axes[1, 0], data)
    draw_panel_d(axes[1, 1], data)
    style_axes(axes)
    axes[0, 0].grid(False)

    fig.suptitle(
        "Random validation optimism is heterogeneous across civil-engineering machine-learning datasets",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=2.6, w_pad=2.2)
    leg_d = _legend_panel_d_upper_left(fig, axes[1, 1])
    fig.savefig(
        OUT_FIG / "fig02_case_study_matrix_v1.png",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[leg_d],
        pad_inches=0.12,
    )
    fig.savefig(
        OUT_FIG / "fig02_case_study_matrix_v1.pdf",
        bbox_inches="tight",
        bbox_extra_artists=[leg_d],
        pad_inches=0.12,
    )
    plt.close(fig)


def make_supplementary_stub_figure(stub: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.8, 3.9),
        gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.44},
    )
    x = np.arange(len(stub))
    width = 0.34
    axes[0].bar(x - width / 2, stub["random_metric"], width, color=BLUE, label="Random CV")
    axes[0].bar(x + width / 2, stub["shape_metric"], width, color=ORANGE, label="Leave-one-shape-out")
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stub["model"], rotation=15, ha="right")
    axes[0].set_ylabel("Pooled R^2")
    axes[0].set_title("a  Generalization to unseen section families", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, stub["gap"], color=RED)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stub["model"], rotation=15, ha="right")
    axes[1].set_ylabel("Random - shape-held-out R^2")
    axes[1].set_title("b  Deployment-target gap", loc="left", fontweight="bold")
    for i, gap in enumerate(stub["gap"]):
        axes[1].text(i, gap + 0.05, f"{gap:.3f}", ha="center", fontsize=8)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "figS1_stub_cfst_shape_generalization_v1.png", dpi=300)
    fig.savefig(OUT_FIG / "figS1_stub_cfst_shape_generalization_v1.pdf")
    plt.close(fig)


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    main_cases = load_main_cases()
    stub_case = load_stub_case()
    main_cases.to_csv(OUT_DATA / "fig02_case_study_matrix_v1_data.csv", index=False)
    stub_case.to_csv(OUT_DATA / "figS1_stub_cfst_shape_generalization_v1_data.csv", index=False)

    make_main_figure(main_cases)
    make_supplementary_stub_figure(stub_case)

    print(f"[OK] {OUT_DATA / 'fig02_case_study_matrix_v1_data.csv'}")
    print(f"[OK] {OUT_FIG / 'fig02_case_study_matrix_v1.png'}")
    print(f"[OK] {OUT_FIG / 'fig02_case_study_matrix_v1.pdf'}")
    print(f"[OK] {OUT_DATA / 'figS1_stub_cfst_shape_generalization_v1_data.csv'}")
    print(f"[OK] {OUT_FIG / 'figS1_stub_cfst_shape_generalization_v1.png'}")
    print(f"[OK] {OUT_FIG / 'figS1_stub_cfst_shape_generalization_v1.pdf'}")


if __name__ == "__main__":
    main()
