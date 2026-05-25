from __future__ import annotations

"""
Figure 1: validation-target taxonomy (Panel A) + source-aware workflow (Panel B).

Layout rules (quality):
- No connector segments pass through unrelated boxes (dedicated gutter bands + margin tracks).
- Panel A arrows run *under* the taxonomy boxes (y below box bottoms), not through text.
- zorder: cross-row feeds < box faces < in-row arrows < text.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"
AUDIT_SUMMARY = ROOT / "code" / "outputs" / "literature" / "reproducible_candidate_audit_summary.csv"
MANUAL = ROOT / "code" / "outputs" / "literature" / "manual_reproducibility_verification_v0.csv"

Z_FEED = 1  # cross-row polylines (drawn before boxes)
Z_BOX = 2
Z_INROW = 3  # horizontal arrows on row midlines
Z_TEXT = 4

BLUE = "#4C78A8"
ORANGE = "#F58518"
RED = "#E45756"
GREEN = "#54A24B"
PURPLE = "#B279A2"
GREY = "#6B6B6B"


def _rounded_box(ax, xy, width, height, *, edgecolor: str, lw: float = 1.15) -> None:
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            linewidth=lw,
            edgecolor=edgecolor,
            facecolor="#FFFFFF",
            transform=ax.transAxes,
            zorder=Z_BOX,
        )
    )


def _box_text(ax, xy, width, height, title: str, body: str, *, title_color: str) -> None:
    x, y = xy
    ax.text(
        x + 0.018,
        y + height - 0.038,
        title,
        transform=ax.transAxes,
        fontsize=10,
        weight="bold",
        color=title_color,
        zorder=Z_TEXT,
        clip_on=False,
    )
    ax.text(
        x + 0.018,
        y + height - 0.078,
        body,
        transform=ax.transAxes,
        fontsize=8.1,
        color="#333333",
        va="top",
        linespacing=1.22,
        zorder=Z_TEXT,
        clip_on=False,
    )


def box(ax, xy, width, height, title, body, color, title_color="#222222") -> None:
    _rounded_box(ax, xy, width, height, edgecolor=color)
    _box_text(ax, xy, width, height, title, body, title_color=title_color)


def arrow_h(ax, start, end, color: str, lw: float = 1.25) -> None:
    arr = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        connectionstyle="arc3,rad=0",
        zorder=Z_INROW,
        clip_on=False,
    )
    ax.add_patch(arr)


def polyline_arrow(
    ax,
    xs: list[float],
    ys: list[float],
    color: str,
    lw: float = 1.35,
    *,
    linestyle: str = "-",
) -> None:
    """Orthogonal polyline; arrowhead on final segment only."""
    if len(xs) < 2:
        return
    ax.plot(
        xs,
        ys,
        color=color,
        lw=lw,
        ls=linestyle,
        transform=ax.transAxes,
        clip_on=False,
        zorder=Z_FEED,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.annotate(
        "",
        xy=(xs[-1], ys[-1]),
        xytext=(xs[-2], ys[-2]),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
            linestyle=linestyle,
            joinstyle="round",
        ),
        zorder=Z_FEED + 0.05,
    )


def _panel_a_taxonomy(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.02,
        0.93,
        "Panel A — Validation-target taxonomy",
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
        zorder=Z_TEXT,
    )
    ax.text(
        0.02,
        0.835,
        "Left → right: stronger alignment between split machinery and the deployment claim.",
        transform=ax.transAxes,
        fontsize=8.8,
        color="#444444",
        zorder=Z_TEXT,
    )

    # Five compact tiles; leave a clean band *below* tiles for arrows (no line through text).
    y0, h = 0.18, 0.52
    specs: list[tuple[float, float, str, str, str]] = [  # (x, width, title, body, color)
        (0.02, 0.175, "Random / K-fold", "Within-source\ninterpolation", BLUE),
        (0.202, 0.168, "GroupKFold / LOSO", "Unseen source or\nprogramme", BLUE),
        (0.378, 0.158, "Mixture groups", "Hold out mixture\nfamilies", GREEN),
        (0.544, 0.158, "Family hold-out", "Structural-family\nextrapolation", GREEN),
        (0.712, 0.268, "Deployment-matched", "Stress test aligned\nto stated deployment", PURPLE),
    ]
    for x, w, title, body, col in specs:
        box(ax, (x, y0), w, h, title, body, col, col)

    y_arrow = 0.125
    for x0, x1 in (
        (0.197, 0.2005),
        (0.371, 0.376),
        (0.537, 0.542),
        (0.703, 0.710),
    ):
        arrow_h(ax, (x0, y_arrow), (x1, y_arrow), "#333333", lw=1.1)


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

    # --- Panel B geometry (transAxes, y upward): explicit gutters so feeds never cross boxes ---
    # Row 1 (literature)
    y1b, y1t = 0.705, 0.86
    h1 = y1t - y1b
    # Gutter between row1 and row2 (horizontal feeds only here)
    g1_lo, g1_hi = 0.635, 0.705
    # Row 2 (evidence engine)
    y2b, y2t = 0.455, 0.615
    h2 = y2t - y2b
    # Gutter between row2 and row3
    g2_lo, g2_hi = 0.385, 0.455
    # Row 3 (outputs / discipline)
    y3b, y3t = 0.195, 0.355
    h3 = y3t - y3b

    x_o, w_o = 0.028, 0.228  # OpenAlex
    x_s, w_s = 0.275, 0.248  # Screened
    x_m, w_m = 0.548, 0.268  # Manual
    cx_o = x_o + w_o * 0.5
    cx_s = x_s + w_s * 0.5
    cx_m = x_m + w_m * 0.5

    x_e, w_e = 0.028, 0.278  # Executable
    x_v, w_v = 0.322, 0.248  # Validation targets
    x_p, w_p = 0.588, 0.268  # Observed pattern
    cx_e = x_e + w_e * 0.5
    cx_p = x_p + w_p * 0.5

    x_d, w_d = 0.028, 0.278  # Dataset-first
    x_r, w_r = 0.322, 0.248  # Reporting
    x_c, w_c = 0.588, 0.268  # Claim discipline
    cx_d = x_d + w_d * 0.5

    y_feed_1 = 0.5 * (g1_lo + g1_hi)  # mid gutter 1 (~0.67)
    y_feed_2 = 0.5 * (g2_lo + g2_hi)  # mid gutter 2 (~0.42)
    x_track = 0.985  # right margin track for Manual → Dataset-first (avoids row-2 panels)

    fig = plt.figure(figsize=(14.2, 9.4))
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.05, 2.45], hspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    _panel_a_taxonomy(ax_a)

    ax_b.set_axis_off()
    ax_b.text(
        0.02,
        0.97,
        "Panel B — Source-aware reproduction workflow",
        transform=ax_b.transAxes,
        fontsize=12,
        weight="bold",
        zorder=Z_TEXT,
    )
    ax_b.text(
        0.02,
        0.905,
        "Rows read left → right. Cross-row links use the shaded gutters only.",
        transform=ax_b.transAxes,
        fontsize=8.8,
        color="#444444",
        zorder=Z_TEXT,
    )

    # Gutter shading (behind everything except it should be subtle)
    for ylo, yhi, alpha in ((g1_lo, g1_hi, 0.06), (g2_lo, g2_hi, 0.06)):
        ax_b.axhspan(ylo, yhi, xmin=0, xmax=1, transform=ax_b.transAxes, facecolor="#EAECEF", alpha=alpha, zorder=0)

    # Cross-row feeds first (under boxes; segments stay in gutters / right margin track)
    polyline_arrow(
        ax_b,
        [cx_s, cx_s, cx_e, cx_e],
        [y1b, y_feed_1, y_feed_1, y2t],
        GREY,
        lw=1.35,
    )
    polyline_arrow(
        ax_b,
        [cx_m, x_track, x_track, cx_d, cx_d],
        [y1b, y1b, y_feed_2, y_feed_2, y3t],
        ORANGE,
        lw=1.35,
    )

    # Row 1
    box(
        ax_b,
        (x_o, y1b),
        w_o,
        h1,
        "OpenAlex search",
        f"{total_openalex:,} metadata records\nstructural/civil ML query\nnoisy discovery layer",
        BLUE,
        BLUE,
    )
    box(
        ax_b,
        (x_s, y1b),
        w_s,
        h1,
        "Strict screened frame",
        f"{screened} structural-ML candidates\ncompleted={completed_metadata}, high={high}\nmedium={medium}, low={low}",
        BLUE,
        BLUE,
    )
    box(
        ax_b,
        (x_m, y1b),
        w_m,
        h1,
        "Manual verification queue",
        f"{len(openalex_high)} metadata-high checks\nrepository hint ≠ raw data\nattrition reported explicitly",
        RED,
        RED,
    )
    arrow_h(ax_b, (x_o + w_o, 0.5 * (y1b + y1t)), (x_s, 0.5 * (y1b + y1t)), BLUE)
    arrow_h(ax_b, (x_s + w_s, 0.5 * (y1b + y1t)), (x_m, 0.5 * (y1b + y1t)), RED)

    # Row 2
    box(
        ax_b,
        (x_e, y2b),
        w_e,
        h2,
        "Executable reproduction modules",
        "6 public case-study modules\nclassification + regression\nsource / mix / family splits",
        GREEN,
        GREEN,
    )
    box(
        ax_b,
        (x_v, y2b),
        w_v,
        h2,
        "Validation targets",
        "RandomKFold: within-source\nGroupKFold / LOSO: unseen source\nfamily hold-out: extrapolation",
        GREEN,
        GREEN,
    )
    box(
        ax_b,
        (x_p, y2b),
        w_p,
        h2,
        "Observed pattern",
        "random ≥ grouped headline\nheterogeneous optimism\ncase evidence, not prevalence",
        GREEN,
        GREEN,
    )
    arrow_h(ax_b, (x_e + w_e, 0.5 * (y2b + y2t)), (x_v, 0.5 * (y2b + y2t)), GREEN)
    arrow_h(ax_b, (x_v + w_v, 0.5 * (y2b + y2t)), (x_p, 0.5 * (y2b + y2t)), GREEN)

    # Row 3
    box(
        ax_b,
        (x_d, y3b),
        w_d,
        h3,
        "Dataset-first expansion",
        f"{len(opportunistic)} promoted datasets\nMendeley resolved & reproduced\nDesignSafe token/browser gated",
        ORANGE,
        ORANGE,
    )
    box(
        ax_b,
        (x_r, y3b),
        w_r,
        h3,
        "Reporting outputs",
        "Fig.2 validation gaps · Fig.3 audit\nFig.4 checklist · Fig.5 Mendeley",
        ORANGE,
        ORANGE,
    )
    box(
        ax_b,
        (x_c, y3b),
        w_c,
        h3,
        "Submission discipline",
        "avoid field-wide prevalence claims\nreport access failures + topology\nexpand repro before NMI/NC framing",
        PURPLE,
        PURPLE,
    )
    arrow_h(ax_b, (x_d + w_d, 0.5 * (y3b + y3t)), (x_r, 0.5 * (y3b + y3t)), ORANGE)
    arrow_h(ax_b, (x_r + w_r, 0.5 * (y3b + y3t)), (x_c, 0.5 * (y3b + y3t)), PURPLE)

    ax_b.text(
        0.02,
        0.03,
        "Abbrev.: LOSO = leave-one-source-out; NMI/NC = Nature Machine Intelligence / Nature Communications. "
        "Counts reflect the OpenAlex screening audit; distinct from the manually curated case-study layer.",
        transform=ax_b.transAxes,
        fontsize=7.9,
        color="#555555",
        zorder=Z_TEXT,
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.96, bottom=0.035)

    fig.savefig(OUT / "fig01_study_design_v0.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fig01_study_design_v0.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
