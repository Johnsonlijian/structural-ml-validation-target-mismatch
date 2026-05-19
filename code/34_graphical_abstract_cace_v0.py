"""
Single-panel graphical abstract for CACE (landscape).

Reads: code/outputs/figures/cace_ninemodule_rf_summary.csv (from 33_cace_nine_module_summary.py).
Writes: figures/submission/graphical_abstract_cace_v0.png (300 dpi) + .pdf

Layout matches manuscript_fragments/graphical_abstract_cace_v0.md (left taxonomy, centre funnel, right RF gaps).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CODE = Path(__file__).resolve().parent
PROJECT = CODE.parent
CSV_PATH = CODE / "outputs" / "figures" / "cace_ninemodule_rf_summary.csv"
SUBMISSION = PROJECT / "figures" / "submission"
SOURCE = PROJECT / "figures" / "source"

SEV_COLOR = {
    "negligible": "#72B7B2",
    "moderate": "#F58518",
    "severe": "#E45756",
    "contrast": "#B279A2",
}


def _short_label(full: str) -> str:
    if "Shear-wall" in full:
        return "Shear-wall (acc.)"
    if "SFRC" in full:
        return "SFRC shear (R²)"
    if "UCI concrete" in full:
        return "UCI concrete (R²)"
    if "Corroded RC beam" in full:
        return "Corroded beams (R²)"
    if "Stub-CFST" in full:
        return "Stub-CFST family (R²)"
    if "Exterior joint shear" in full:
        return "Joint ext. shear (R²)"
    if "Exterior joint failure" in full:
        return "Joint ext. FM (acc.)"
    if "Cyclic joint" in full:
        return "Joint cyclic (R²)"
    if "RC columns" in full and "combined" in full:
        return "DS columns (R²)"
    if "peak lateral strength" in full:
        return "Wall Vmax (R²)"
    if "drift capacity" in full:
        return "Wall drift (R²)"
    if "peak shear V_m_avg" in full:
        return "Coupl. V (R²)"
    if "normalized shear" in full:
        return "Coupl. V′ (R²)"
    if "chord rotation" in full:
        return "Coupl. θ (R²)"
    return full[:28] + "…" if len(full) > 30 else full


def _draw_taxonomy(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Validation targets", fontsize=10, fontweight="bold", loc="left", pad=6)

    boxes = [
        (0.08, 0.68, 0.84, 0.14, "Random / shuffle\n(within-database interpolation)"),
        (0.08, 0.42, 0.84, 0.14, "Grouped / LOSO\n(out-of-source transport)"),
        (0.08, 0.16, 0.84, 0.14, "Deployment-matched\n(e.g. structural-family holdout)"),
    ]
    for x, y, w, h, txt in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.02",
                linewidth=1.0,
                edgecolor="#333333",
                facecolor="#F0F0F0",
            )
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.5, wrap=True)
    # Connect box bottoms to next box tops (y increases upward).
    for y0, y1 in ((0.68, 0.56), (0.42, 0.30)):
        ax.add_patch(
            FancyArrowPatch(
                (0.5, y0),
                (0.5, y1),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#333333",
            )
        )


def _draw_funnel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Evidence pipeline", fontsize=10, fontweight="bold", loc="center", pad=6)
    stages = [
        (0.12, 0.72, 0.76, 0.12, "OpenAlex frame\n(332 papers screened)"),
        (0.22, 0.44, 0.56, 0.12, "Dataset-first\ntriage"),
        (0.28, 0.14, 0.44, 0.14, "Nine executable\npublic modules"),
    ]
    for x, y, w, h, txt in stages:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.015,rounding_size=0.02",
                linewidth=1.0,
                edgecolor="#4E79A7",
                facecolor="#E8EEF5",
            )
        )
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7.5)
    # Stage bottoms → next tops: stage1 bottom 0.72 → stage2 top 0.56; stage2 bottom 0.44 → stage3 top 0.28
    for y0, y1 in ((0.72, 0.56), (0.44, 0.28)):
        ax.add_patch(
            FancyArrowPatch(
                (0.5, y0),
                (0.5, y1),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color="#4E79A7",
            )
        )


def _draw_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    labels = [_short_label(str(s)) for s in df["label"]]
    y = np.arange(len(labels))
    gaps = df["gap"].to_numpy(dtype=float)
    colors = [SEV_COLOR.get(str(s), "#333333") for s in df["severity"]]

    ax.barh(y, gaps, color=colors, linewidth=0.6, edgecolor="#333333")
    ax.axvline(0, color="#222222", linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlabel("Gap (random − grouped / deployment)", fontsize=7.5)
    ax.set_title("Random Forest headline contrasts", fontsize=10, fontweight="bold", loc="right", pad=6)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    if not CSV_PATH.is_file():
        raise SystemExit(f"Missing {CSV_PATH}; run code/33_cace_nine_module_summary.py first.")

    df = pd.read_csv(CSV_PATH)
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    # Landscape canvas: ≥ ~1800 px width at 300 dpi → ≥ 6 in; use 12×4.2 in for EM thumbnail legibility.
    fig = plt.figure(figsize=(12, 4.2), dpi=100)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 0.72, 1.25], wspace=0.22)
    ax_l = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[0, 2])
    _draw_taxonomy(ax_l)
    _draw_funnel(ax_m)
    _draw_bars(ax_r, df)

    fig.suptitle(
        "Source-aware validation for structural ML: taxonomy, sampling frame, heterogeneous RF gaps",
        fontsize=9.5,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.12)

    png = SUBMISSION / "graphical_abstract_cace_v0.png"
    pdf = SUBMISSION / "graphical_abstract_cace_v0.pdf"
    svg = SOURCE / "graphical_abstract_cace_v0.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {png}")
    print(f"[OK] {pdf}")
    print(f"[OK] {svg}")


if __name__ == "__main__":
    main()
