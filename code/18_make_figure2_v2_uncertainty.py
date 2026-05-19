"""
Build Figure 2 v2 with uncertainty diagnostics.

Uncertainty is intentionally conservative and case-specific:
- Mangalathu shear-wall classification: fold-summary normal approximation,
  because prediction-level outputs were not saved by the original reproduction.
- Rahman/SFRC: fold-summary normal approximation, matching the Figure 2
  headline fold-mean R^2 gap.
- UCI concrete and corroded RC beams: paired group bootstrap over mix/source
  groups using prediction-level out-of-fold predictions.

The goal is not to claim a final inferential interval for the field. The goal is
to prevent point-estimate over-reading in the motivating case-study figure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "code"
REPRO_DIR = CODE_DIR / "outputs" / "reproductions"
OUT_DATA = CODE_DIR / "outputs" / "figures"
OUT_FIG = ROOT / "figures" / "draft"
FRAG_DIR = ROOT / "manuscript_fragments"

BLUE = "#4C78A8"
ORANGE = "#F58518"
RED = "#E45756"
TEAL = "#72B7B2"
PURPLE = "#B279A2"
GREEN = "#59A14F"
GREY = "#6B6B6B"
LIGHT_GREY = "#E6E6E6"


@dataclass(frozen=True)
class Interval:
    estimate: float
    lo: float
    hi: float
    method: str
    note: str


def read_results(doi_slug: str) -> pd.DataFrame:
    return pd.read_csv(REPRO_DIR / doi_slug / "results.csv")


def normal_ci(mean: float, sd: float, n: int, note: str) -> Interval:
    se = sd / np.sqrt(n)
    return Interval(
        estimate=float(mean),
        lo=float(mean - 1.96 * se),
        hi=float(mean + 1.96 * se),
        method="fold_summary_normal_approx",
        note=note,
    )


def difference_ci(a: Interval, b: Interval, a_sd: float, b_sd: float, a_n: int, b_n: int, note: str) -> Interval:
    gap = a.estimate - b.estimate
    se = np.sqrt((a_sd**2 / a_n) + (b_sd**2 / b_n))
    return Interval(
        estimate=float(gap),
        lo=float(gap - 1.96 * se),
        hi=float(gap + 1.96 * se),
        method="fold_summary_normal_approx_unpaired",
        note=note,
    )


def paired_group_bootstrap(
    predictions_path: Path,
    model: str,
    random_validation: str,
    group_validation: str,
    group_col: str,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[Interval, Interval, Interval, dict[str, object]]:
    pred = pd.read_csv(predictions_path)
    pred = pred[pred["model"] == model].copy()
    group_pred = pred[pred["validation"] == group_validation].copy()
    random_pred = pred[pred["validation"] == random_validation].copy()

    if group_col not in group_pred.columns:
        raise KeyError(f"{group_col} not found in {predictions_path}")

    group_map = group_pred[["sample_index", group_col]].drop_duplicates()
    group_map = group_map.rename(columns={group_col: "boot_group"})
    groups = group_map["boot_group"].dropna().unique()
    rng = np.random.default_rng(seed)

    random_metrics: list[float] = []
    group_metrics: list[float] = []
    gap_metrics: list[float] = []
    skipped = 0

    by_group = {g: group_map.loc[group_map["boot_group"] == g, ["sample_index"]] for g in groups}

    for _ in range(n_boot):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        selected = pd.concat([by_group[g] for g in sampled_groups], ignore_index=True)

        r = selected.merge(random_pred, on="sample_index", how="left")
        g = selected.merge(group_pred, on="sample_index", how="left")
        if r["observed"].nunique() < 2 or g["observed"].nunique() < 2:
            skipped += 1
            continue
        try:
            r2_random = r2_score(r["observed"], r["predicted"])
            r2_group = r2_score(g["observed"], g["predicted"])
        except ValueError:
            skipped += 1
            continue
        random_metrics.append(float(r2_random))
        group_metrics.append(float(r2_group))
        gap_metrics.append(float(r2_random - r2_group))

    def interval(values: list[float], estimate: float, label: str) -> Interval:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return Interval(
            estimate=float(estimate),
            lo=float(lo),
            hi=float(hi),
            method="paired_group_bootstrap_prediction_level",
            note=label,
        )

    point_random = r2_score(random_pred["observed"], random_pred["predicted"])
    point_group = r2_score(group_pred["observed"], group_pred["predicted"])
    diagnostics = {
        "n_boot_requested": n_boot,
        "n_boot_used": len(gap_metrics),
        "n_boot_skipped": skipped,
        "n_groups": int(len(groups)),
        "group_col": group_col,
        "model": model,
        "random_validation": random_validation,
        "group_validation": group_validation,
    }
    return (
        interval(random_metrics, point_random, "random pooled R^2"),
        interval(group_metrics, point_group, "group-aware pooled R^2"),
        interval(gap_metrics, point_random - point_group, "paired gap in pooled R^2"),
        diagnostics,
    )


def build_case_data() -> tuple[pd.DataFrame, list[dict[str, object]]]:
    diagnostics: list[dict[str, object]] = []

    # Case 1: classification, fold-summary only.
    mangalathu = read_results("10-1016-j-engstruct-2019-110331")
    m_rf = mangalathu[mangalathu["model"] == "RandomForest"].set_index("cv")
    m_random = normal_ci(
        m_rf.loc["RandomStratifiedKFold-5", "accuracy_mean"],
        m_rf.loc["RandomStratifiedKFold-5", "accuracy_sd"],
        int(m_rf.loc["RandomStratifiedKFold-5", "n_splits"]),
        "classification; prediction-level outputs unavailable",
    )
    m_group = normal_ci(
        m_rf.loc["GroupKFold-by-Author-5", "accuracy_mean"],
        m_rf.loc["GroupKFold-by-Author-5", "accuracy_sd"],
        int(m_rf.loc["GroupKFold-by-Author-5", "n_splits"]),
        "classification; prediction-level outputs unavailable",
    )
    m_gap = difference_ci(
        m_random,
        m_group,
        m_rf.loc["RandomStratifiedKFold-5", "accuracy_sd"],
        m_rf.loc["GroupKFold-by-Author-5", "accuracy_sd"],
        int(m_rf.loc["RandomStratifiedKFold-5", "n_splits"]),
        int(m_rf.loc["GroupKFold-by-Author-5", "n_splits"]),
        "approximate unpaired fold-summary CI; folds are not independent experiments",
    )

    # Case 2: SFRC, fold-summary interval because Figure 2 uses fold-mean R^2.
    rahman = read_results("10-1016-j-engstruct-2020-111743")
    r_rf = rahman[rahman["model"] == "RandomForest"].set_index("validation")
    r_random = normal_ci(
        r_rf.loc["RandomKFold", "r2_mean"],
        r_rf.loc["RandomKFold", "r2_std"],
        int(r_rf.loc["RandomKFold", "n_splits"]),
        "regression; headline uses fold-mean R^2",
    )
    r_group = normal_ci(
        r_rf.loc["GroupKFold_Source", "r2_mean"],
        r_rf.loc["GroupKFold_Source", "r2_std"],
        int(r_rf.loc["GroupKFold_Source", "n_splits"]),
        "regression; headline uses fold-mean R^2",
    )
    r_gap = difference_ci(
        r_random,
        r_group,
        r_rf.loc["RandomKFold", "r2_std"],
        r_rf.loc["GroupKFold_Source", "r2_std"],
        int(r_rf.loc["RandomKFold", "n_splits"]),
        int(r_rf.loc["GroupKFold_Source", "n_splits"]),
        "approximate unpaired fold-summary CI; reflects fold instability",
    )

    # Case 3: UCI concrete, prediction-level paired group bootstrap.
    n_random, n_group, n_gap, n_diag = paired_group_bootstrap(
        REPRO_DIR / "10-1016-j-conbuildmat-2020-120950" / "predictions.csv",
        model="RandomForest",
        random_validation="RandomKFold",
        group_validation="GroupKFold_Mix",
        group_col="mix_group",
    )
    diagnostics.append({"case": "UCI concrete", **n_diag})

    # Case 4: corroded RC beams, prediction-level paired source bootstrap.
    c_random, c_group, c_gap, c_diag = paired_group_bootstrap(
        REPRO_DIR / "10-5281-zenodo-8062007" / "predictions.csv",
        model="RandomForest",
        random_validation="RandomKFold",
        group_validation="GroupKFold_Source",
        group_col="source_group",
    )
    diagnostics.append({"case": "Corroded RC beams", **c_diag})

    rows = [
        case_row(
            1,
            "Shear wall\nfailure mode",
            "Accuracy",
            m_random,
            m_group,
            m_gap,
            "Author/source",
            393,
            75,
            "large",
        ),
        case_row(
            2,
            "SFRC shear\ncapacity",
            "Fold-mean R^2",
            r_random,
            r_group,
            r_gap,
            "Source reference",
            int(r_rf.loc["RandomKFold", "n_samples"]),
            int(r_rf.loc["LeaveOneSourceOut", "n_splits"]),
            "large",
        ),
        case_row(
            3,
            "Concrete\nstrength",
            "Pooled R^2",
            n_random,
            n_group,
            n_gap,
            "Mix proportions",
            1030,
            428,
            "small",
        ),
        case_row(
            4,
            "Corroded beam\nmoment",
            "Pooled R^2",
            c_random,
            c_group,
            c_gap,
            "Experimental program",
            804,
            54,
            "moderate",
        ),
    ]
    return pd.DataFrame(rows), diagnostics


def case_row(
    case_id: int,
    short_label: str,
    metric_name: str,
    random: Interval,
    group: Interval,
    gap: Interval,
    group_proxy: str,
    n_samples: int,
    n_groups: int,
    severity: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "short_label": short_label,
        "metric_name": metric_name,
        "random_metric": random.estimate,
        "random_lo": random.lo,
        "random_hi": random.hi,
        "group_metric": group.estimate,
        "group_lo": group.lo,
        "group_hi": group.hi,
        "gap": gap.estimate,
        "gap_lo": gap.lo,
        "gap_hi": gap.hi,
        "group_proxy": group_proxy,
        "n_samples": n_samples,
        "n_groups": n_groups,
        "severity": severity,
        "uncertainty_method": gap.method,
        "uncertainty_note": gap.note,
    }


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
    for i, color in enumerate(family_colors):
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
    ax.add_patch(FancyArrowPatch((0.88, 0.69), (0.88, 0.27), arrowstyle="-|>", mutation_scale=14, color="#333333"))
    ax.text(0.82, 0.48, "harder\nclaim", fontsize=8, ha="center", va="center")


def asym_yerr(point: pd.Series, lo: pd.Series, hi: pd.Series) -> np.ndarray:
    return np.vstack([(point - lo).clip(lower=0), (hi - point).clip(lower=0)])


def draw_panel_b(ax, data: pd.DataFrame) -> None:
    ax.set_title("b  Headline scores with case-specific uncertainty", loc="left", fontweight="bold", pad=8)
    x = np.arange(len(data))
    width = 0.34
    ax.bar(x - width / 2, data["random_metric"], width, label="Random CV", color=BLUE)
    ax.errorbar(
        x - width / 2,
        data["random_metric"],
        yerr=asym_yerr(data["random_metric"], data["random_lo"], data["random_hi"]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2,
    )
    ax.bar(x + width / 2, data["group_metric"], width, label="Group-aware CV", color=ORANGE)
    ax.errorbar(
        x + width / 2,
        data["group_metric"],
        yerr=asym_yerr(data["group_metric"], data["group_lo"], data["group_hi"]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2,
    )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["short_label"], fontsize=8)
    ax.set_ylabel("Headline metric")
    ax.set_ylim(min(-2.2, data["group_lo"].min() - 0.15), 1.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    for i, row in data.iterrows():
        ax.text(i, min(1.18, max(row["random_metric"], row["group_metric"]) + 0.08), row["metric_name"], ha="center", fontsize=7, color=GREY)


def draw_panel_c(ax, data: pd.DataFrame) -> None:
    ax.set_title("c  Optimism gaps remain heterogeneous", loc="left", fontweight="bold", pad=8)
    colors = [RED if s == "large" else ORANGE if s == "moderate" else TEAL for s in data["severity"]]
    x = np.arange(len(data))
    ax.bar(x, data["gap"], color=colors)
    ax.errorbar(
        x,
        data["gap"],
        yerr=asym_yerr(data["gap"], data["gap_lo"], data["gap_hi"]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.9,
        capsize=2,
    )
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["short_label"], fontsize=8)
    ax.set_ylabel("Random - group metric")
    ax.set_ylim(min(0, data["gap_lo"].min() - 0.15), max(2.8, data["gap_hi"].max() + 0.2))
    for i, gap in enumerate(data["gap"]):
        ax.text(i, gap + 0.08, f"{gap:.3f}", ha="center", va="bottom", fontsize=8)


def draw_panel_d(ax, data: pd.DataFrame) -> None:
    ax.set_title("d  Dataset topology and uncertainty method", loc="left", fontweight="bold", pad=8)
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
        method = "fold approx." if "fold_summary" in row["uncertainty_method"] else "group bootstrap"
        ax.text(i, row["n_samples"] * 1.25, f"n={int(row['n_samples'])}\ng={int(row['n_groups'])}\n{method}", ha="center", fontsize=7)


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


def make_figure(data: pd.DataFrame) -> None:
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
        "Random validation optimism is heterogeneous and topology-dependent",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955], h_pad=2.6, w_pad=2.2)
    leg_d = _legend_panel_d_upper_left(fig, axes[1, 1])
    fig.savefig(
        OUT_FIG / "fig02_case_study_matrix_v2_uncertainty.png",
        dpi=300,
        bbox_inches="tight",
        bbox_extra_artists=[leg_d],
        pad_inches=0.12,
    )
    fig.savefig(
        OUT_FIG / "fig02_case_study_matrix_v2_uncertainty.pdf",
        bbox_inches="tight",
        bbox_extra_artists=[leg_d],
        pad_inches=0.12,
    )
    plt.close(fig)


def write_diagnostics(data: pd.DataFrame, diagnostics: list[dict[str, object]]) -> None:
    rows = []
    for _, row in data.iterrows():
        rows.append(
            {
                "case_id": int(row["case_id"]),
                "case": row["short_label"].replace("\n", " "),
                "uncertainty_method": row["uncertainty_method"],
                "note": row["uncertainty_note"],
                "random_metric": row["random_metric"],
                "random_lo": row["random_lo"],
                "random_hi": row["random_hi"],
                "group_metric": row["group_metric"],
                "group_lo": row["group_lo"],
                "group_hi": row["group_hi"],
                "gap": row["gap"],
                "gap_lo": row["gap_lo"],
                "gap_hi": row["gap_hi"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT_DATA / "fig02_case_study_matrix_v2_uncertainty.csv", index=False)
    (OUT_DATA / "fig02_case_study_matrix_v2_bootstrap_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Figure 2 v2 Uncertainty Diagnostics",
        "",
        "Figure 2 v2 adds case-specific uncertainty intervals to the four main case-study gaps.",
        "",
        "## Methods",
        "",
        "- Mangalathu shear-wall classification: fold-summary normal approximation because prediction-level outputs were not saved.",
        "- Rahman/SFRC: fold-summary normal approximation because Figure 2 uses fold-mean `R²` as the severe source-reference headline.",
        "- UCI concrete and corroded RC beams: paired group bootstrap over mix/source groups using prediction-level out-of-fold predictions.",
        "",
        "These intervals are not field-wide inferential estimates. They diagnose whether the case-study point estimates are visually over-read.",
        "",
        "## Bootstrap Diagnostics",
        "",
    ]
    for item in diagnostics:
        lines.append(
            f"- {item['case']}: requested {item['n_boot_requested']} bootstraps, used {item['n_boot_used']}, "
            f"skipped {item['n_boot_skipped']}, groups={item['n_groups']}."
        )
    lines.extend(
        [
            "",
            "## Reviewer-Sensitive Interpretation",
            "",
            "The wide SFRC interval reflects fold instability rather than uncertainty about whether the gap exists. The UCI and corroded-beam intervals are narrower because they use pooled prediction-level metrics with many mix/source groups. Mangalathu remains a fold-summary approximation and should not be overinterpreted as a precise confidence interval.",
        ]
    )
    (FRAG_DIR / "figure2_v2_uncertainty_diagnostics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    FRAG_DIR.mkdir(parents=True, exist_ok=True)

    data, diagnostics = build_case_data()
    data.to_csv(OUT_DATA / "fig02_case_study_matrix_v2_data.csv", index=False)
    write_diagnostics(data, diagnostics)
    make_figure(data)

    print(f"[OK] {OUT_DATA / 'fig02_case_study_matrix_v2_data.csv'}")
    print(f"[OK] {OUT_DATA / 'fig02_case_study_matrix_v2_uncertainty.csv'}")
    print(f"[OK] {OUT_DATA / 'fig02_case_study_matrix_v2_bootstrap_diagnostics.json'}")
    print(f"[OK] {FRAG_DIR / 'figure2_v2_uncertainty_diagnostics.md'}")
    print(f"[OK] {OUT_FIG / 'fig02_case_study_matrix_v2_uncertainty.png'}")
    print(f"[OK] {OUT_FIG / 'fig02_case_study_matrix_v2_uncertainty.pdf'}")


if __name__ == "__main__":
    main()
