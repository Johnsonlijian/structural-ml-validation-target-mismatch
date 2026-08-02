from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROUND_ROOT = ROOT.parent
CASE_ROOT = (
    ROUND_ROOT
    / "experiments"
    / "real_data"
    / "external_stud_family"
)
CONTROL_ROOT = CASE_ROOT / "p1_controls"
SVG_PATH = ROOT / "output" / "svg" / "Figure_6.svg"
PDF_PATH = ROOT / "output" / "pdf" / "Figure_6.pdf"
PNG_PATH = ROOT / "output" / "png" / "Figure_6.png"

NAVY = "#163A5F"
TEAL = "#147D82"
AMBER = "#D98E04"
RED = "#C23B22"
VIOLET = "#7057A3"
TEXT = "#17212B"
MUTED = "#5F6B76"
GRID = "#D5DDE3"
PANEL = "#FAFBFC"
GRAY = "#7C8994"
LIGHT_GRAY = "#EEF2F4"

MODEL_ORDER = ["Ridge", "RBF_SVR", "ExtraTrees", "HistGradientBoosting"]
MODEL_LABEL = {
    "Ridge": "Ridge",
    "RBF_SVR": "RBF-SVR",
    "ExtraTrees": "ExtraTrees",
    "HistGradientBoosting": "HGB",
}
MODEL_COLOR = {
    "Ridge": NAVY,
    "RBF_SVR": VIOLET,
    "ExtraTrees": AMBER,
    "HistGradientBoosting": TEAL,
}


def long_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def sha256(path: Path) -> str:
    with open(long_path(path), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def read_json(path: Path) -> dict:
    with open(long_path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(long_path(path))


def load_evidence() -> dict[str, pd.DataFrame | dict]:
    manifest = read_json(CONTROL_ROOT / "p1_controls_manifest.json")
    used_control_files = [
        "multivariate_support.csv",
        "prediction_ratio_metrics.csv",
        "source_composition_intervals.csv",
        "deck_positive_control_comparison.csv",
    ]
    for name in used_control_files:
        if sha256(CONTROL_ROOT / name) != manifest["outputs"][name]:
            raise RuntimeError(f"frozen Figure 6 aggregate hash mismatch: {name}")

    evaluation_manifest = read_json(CASE_ROOT / "r" / "manifest.json")
    model_selection_path = CASE_ROOT / "r" / "model_selection.csv"
    if (
        sha256(model_selection_path)
        != evaluation_manifest["outputs"]["model_selection.csv"]
    ):
        raise RuntimeError("frozen model-selection table hash mismatch")

    support = read_csv(CONTROL_ROOT / "multivariate_support.csv")
    metrics = read_csv(CONTROL_ROOT / "prediction_ratio_metrics.csv")
    intervals = read_csv(CONTROL_ROOT / "source_composition_intervals.csv")
    positive = read_csv(CONTROL_ROOT / "deck_positive_control_comparison.csv")
    selection = read_csv(model_selection_path)

    assert set(support["parent"]) == {"deck464", "lwc90", "rac27"}
    deck_metrics = metrics.loc[
        (metrics["parent"] == "deck464")
        & (metrics["model"].isin(MODEL_ORDER))
    ]
    deck_intervals = intervals.loc[
        (intervals["parent"] == "deck464")
        & (intervals["model"].isin(MODEL_ORDER))
    ]
    assert set(deck_metrics["model"]) == set(MODEL_ORDER)
    assert set(deck_intervals["model"]) == set(MODEL_ORDER)
    assert set(deck_intervals["status"]) == {
        "source_composition_resampling_interval"
    }

    positive_subset = positive.loc[
        positive["model"].isin(
            ["HistGradientBoosting", "DeckSpecific_Eq11_positive_control"]
        )
    ]
    assert set(positive_subset["n"].astype(int)) == {401}

    deck_selection = selection.loc[selection["parent"] == "deck464"].copy()
    assert len(deck_selection) == 4
    assert (
        deck_selection.loc[
            deck_selection["method"] == "RandomKFold",
            "external_nrmse_regret",
        ].iloc[0]
        == 0.0
    )
    return {
        "manifest": manifest,
        "support": support,
        "metrics": deck_metrics,
        "intervals": deck_intervals,
        "positive": positive_subset,
        "selection": deck_selection,
    }


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "axes.edgecolor": "#9AA7B2",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": PANEL,
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "pdf.use14corefonts": False,
        }
    )


def panel_style(ax: mpl.axes.Axes, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, color="#7F8A93")
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.set_index("model").loc[MODEL_ORDER].reset_index()


def build_figure() -> mpl.figure.Figure:
    evidence = load_evidence()
    support = evidence["support"]
    metrics = model_rows(evidence["metrics"])
    intervals = model_rows(evidence["intervals"])
    positive = evidence["positive"].set_index("model")
    selection = evidence["selection"]

    set_style()
    fig = plt.figure(figsize=(15.8, 10.3))
    outer = fig.add_gridspec(
        2,
        12,
        left=0.055,
        right=0.985,
        top=0.875,
        bottom=0.105,
        height_ratios=[1.0, 1.02],
        hspace=0.38,
        wspace=1.20,
    )
    ax_a = fig.add_subplot(outer[0, 0:5])
    ax_b = fig.add_subplot(outer[0, 5:12])
    c_grid = outer[1, 0:5].subgridspec(1, 2, wspace=0.13)
    ax_c1 = fig.add_subplot(c_grid[0, 0])
    ax_c2 = fig.add_subplot(c_grid[0, 1], sharey=ax_c1)
    ax_d = fig.add_subplot(outer[1, 5:8])
    ax_e = fig.add_subplot(outer[1, 8:12])

    fig.suptitle(
        "Override consequence: numerical overlap does not represent a topology deployment",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=TEXT,
    )
    fig.text(
        0.055,
        0.925,
        "All four NWC learners overpredict deck resistance; RandomKFold alone selects the post hoc lowest-error external-parent learner.",
        ha="left",
        fontsize=11.5,
        color=MUTED,
    )

    # (a) Numerical support coverage.
    parent_order = ["deck464", "lwc90", "rac27"]
    support_indexed = support.set_index("parent").loc[parent_order]
    parent_labels = ["Deck\n413 / 22", "LWC\n60 / 5", "RAC\n27 / 1"]
    support_fields = [
        ("knn_fraction_within_development_q95", "5-NN", NAVY),
        (
            "robust_mahalanobis_fraction_within_development_q95",
            "Robust MCD",
            VIOLET,
        ),
        ("pca_joint_fraction_within_development_q95", "PCA joint", TEAL),
    ]
    x = np.arange(3)
    width = 0.23
    for index, (column, label, color) in enumerate(support_fields):
        values = support_indexed[column].to_numpy(float)
        bars = ax_a.bar(
            x + (index - 1) * width,
            values,
            width=width,
            color=color,
            alpha=0.90,
            label=label,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 1.025),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.7,
                color=TEXT,
            )
    ax_a.axhline(0.95, color=RED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax_a.text(2.47, 0.957, "NWC 95% envelope", color=RED, fontsize=8.2,
              ha="right", va="bottom")
    ax_a.set_ylim(-0.12, 1.08)
    ax_a.set_yticks(np.arange(0, 1.01, 0.2))
    ax_a.set_xticks(x, parent_labels)
    ax_a.set_ylabel("Fraction inside NWC envelope")
    ax_a.set_title("(a) Recorded numerical support", loc="left", color=TEXT)
    ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=3,
        frameon=False,
        fontsize=8.6,
        handlelength=1.5,
    )
    ax_a.text(
        -0.42,
        -0.082,
        "Deck topology is not one of the seven numerical features.",
        transform=ax_a.transData,
        color=RED,
        fontsize=8.6,
        fontweight="bold",
    )
    panel_style(ax_a, grid_axis="y")

    # (b) External deck NRMSE with source-composition intervals.
    y = np.arange(len(MODEL_ORDER))
    risk = metrics["nrmse_by_development_sd"].to_numpy(float)
    low = intervals["nrmse_composition_q025"].to_numpy(float)
    high = intervals["nrmse_composition_q975"].to_numpy(float)
    for index, model in enumerate(MODEL_ORDER):
        ax_b.errorbar(
            risk[index],
            y[index],
            xerr=np.array(
                [[risk[index] - low[index]], [high[index] - risk[index]]]
            ),
            fmt="o",
            markersize=8.5,
            color=MODEL_COLOR[model],
            ecolor=MODEL_COLOR[model],
            elinewidth=2.2,
            capsize=4,
            capthick=1.6,
            zorder=4,
        )
        ax_b.text(
            min(risk[index] + 0.018, 1.27),
            y[index] + 0.03,
            f"{risk[index]:.3f}",
            va="center",
            fontsize=8.7,
            color=MODEL_COLOR[model],
            fontweight="bold",
        )
    ax_b.axvline(1.0, color="#8D99A3", linestyle=(0, (3, 3)), linewidth=1.0)
    ax_b.set_xlim(0.70, 1.30)
    ax_b.set_yticks(y, [MODEL_LABEL[name] for name in MODEL_ORDER])
    ax_b.set_ylim(3.55, -0.35)
    ax_b.set_xlabel("External NRMSE (normalized by NWC target SD)")
    ax_b.set_title(
        "(b) Deck external risk | 413 rows, 22 observed clusters",
        loc="left",
        color=TEXT,
    )
    ax_b.text(
        0.71,
        3.42,
        "Whiskers: source-composition resampling interval; not a population confidence interval.",
        transform=ax_b.transData,
        fontsize=8.5,
        color=MUTED,
    )
    panel_style(ax_b, grid_axis="x")

    # (c) Directional consequence: ratio and overprediction.
    ratio = metrics["prediction_to_measurement_mean"].to_numpy(float)
    ratio_low = intervals["prediction_to_measurement_mean_q025"].to_numpy(float)
    ratio_high = intervals["prediction_to_measurement_mean_q975"].to_numpy(float)
    over = metrics["overprediction_rate"].to_numpy(float)
    for index, model in enumerate(MODEL_ORDER):
        ax_c1.errorbar(
            ratio[index],
            y[index],
            xerr=np.array(
                [[ratio[index] - ratio_low[index]], [ratio_high[index] - ratio[index]]]
            ),
            fmt="o",
            markersize=7.5,
            color=MODEL_COLOR[model],
            ecolor=MODEL_COLOR[model],
            elinewidth=1.9,
            capsize=3.5,
            zorder=4,
        )
        ax_c1.text(
            ratio[index] + 0.025,
            y[index],
            f"{ratio[index]:.2f}",
            va="center",
            fontsize=7.8,
            color=MODEL_COLOR[model],
            fontweight="bold",
        )
    ax_c1.axvline(1.0, color=RED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax_c1.set_xlim(0.90, 2.08)
    ax_c1.set_yticks(y, [MODEL_LABEL[name] for name in MODEL_ORDER])
    ax_c1.set_ylim(3.55, -0.35)
    ax_c1.set_xlabel("Mean predicted / measured")
    ax_c1.set_title("(c) Directional overprediction", loc="left", color=TEXT)
    panel_style(ax_c1, grid_axis="x")

    for index, model in enumerate(MODEL_ORDER):
        ax_c2.barh(
            y[index],
            over[index],
            height=0.56,
            color=MODEL_COLOR[model],
            alpha=0.90,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax_c2.text(
            min(over[index] - 0.02, 0.96),
            y[index],
            f"{100 * over[index]:.1f}%",
            ha="right",
            va="center",
            fontsize=7.7,
            color="white",
            fontweight="bold",
        )
    ax_c2.axvline(0.5, color=RED, linestyle=(0, (4, 3)), linewidth=1.0)
    ax_c2.set_xlim(0, 1.02)
    ax_c2.tick_params(axis="y", labelleft=False)
    ax_c2.set_xlabel("Fraction predicted > measured")
    ax_c2.set_title("Deck overprediction rate", loc="left", color=TEXT)
    panel_style(ax_c2, grid_axis="x")
    ax_c1.text(
        0.92,
        3.42,
        "Ratio > 1 denotes resistance overprediction.",
        transform=ax_c1.transData,
        fontsize=8.1,
        color=RED,
        fontweight="bold",
    )

    # (d) Non-independent positive control on a common 401-row subset.
    hgb = positive.loc["HistGradientBoosting"]
    eq11 = positive.loc["DeckSpecific_Eq11_positive_control"]
    categories = ["NRMSE", "Overprediction"]
    hgb_values = [
        float(hgb["nrmse_by_development_sd"]),
        float(hgb["overprediction_rate"]),
    ]
    eq_values = [
        float(eq11["nrmse_by_development_sd"]),
        float(eq11["overprediction_rate"]),
    ]
    xd = np.arange(2)
    bw = 0.32
    bars_hgb = ax_d.bar(
        xd - bw / 2,
        hgb_values,
        width=bw,
        color=TEAL,
        label="NWC HGB",
        edgecolor="white",
        linewidth=0.6,
    )
    bars_eq = ax_d.bar(
        xd + bw / 2,
        eq_values,
        width=bw,
        color=LIGHT_GRAY,
        edgecolor=GRAY,
        linewidth=1.2,
        hatch="////",
        label="Deck Eq. 11",
    )
    for bar in [*bars_hgb, *bars_eq]:
        value = bar.get_height()
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=TEXT,
        )
    ax_d.set_ylim(0, 1.08)
    ax_d.set_xticks(xd, categories)
    ax_d.set_title(
        "(d) Deck-aware positive control\ncommon 401-row subset",
        loc="left",
        color=TEXT,
    )
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), frameon=False,
                fontsize=8.2)
    ax_d.text(
        0.50,
        0.52,
        "Eq. 11 was calibrated on the deck database:\npositive control, not independent.",
        transform=ax_d.transAxes,
        ha="center",
        fontsize=8.0,
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white",
              "edgecolor": GRID, "alpha": 0.94},
    )
    panel_style(ax_d, grid_axis="y")

    # (e) Model-selection regret: unfavorable result shown coequally.
    method_order = [
        "RandomKFold",
        "GroupKFold_Source",
        "ClaimCut_C_source",
        "DataSAIL_C1e_Scalar_C_deck",
    ]
    method_label = {
        "RandomKFold": "RandomKFold",
        "GroupKFold_Source": "Source GroupKFold",
        "ClaimCut_C_source": "Source ClaimCut",
        "DataSAIL_C1e_Scalar_C_deck": "DataSAIL C1e proxy",
    }
    selected = selection.set_index("method").loc[method_order]
    regret = selected["external_nrmse_regret"].to_numpy(float)
    chosen = selected["selected_model"].to_list()
    ye = np.arange(4)
    for index, (value, model) in enumerate(zip(regret, chosen)):
        color = MODEL_COLOR[model]
        ax_e.hlines(index, 0, value, color="#AAB4BD", linewidth=3.0, zorder=2)
        ax_e.scatter(
            [value],
            [index],
            s=95,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )
        offset = 0.003 if value > 0 else 0.006
        ax_e.text(
            value + offset,
            index,
            f"{MODEL_LABEL[model]} | {value:.3f}",
            va="center",
            fontsize=8.2,
            color=color,
            fontweight="bold",
        )
    ax_e.axvline(0, color="#8D99A3", linewidth=1.0)
    ax_e.set_xlim(-0.004, 0.145)
    ax_e.set_yticks(ye, [method_label[name] for name in method_order])
    ax_e.invert_yaxis()
    ax_e.set_xlabel("External NRMSE regret (lower is better)")
    ax_e.set_title(
        "(e) Model-selection regret (lower is better)",
        loc="left",
        color=TEXT,
    )
    panel_style(ax_e, grid_axis="x")

    fig.text(
        0.055,
        0.047,
        "Evidence boundary: aggregate-only external-parent evaluation; support is descriptive, intervals resample observed source composition, and RAC contains one programme.",
        fontsize=9.0,
        color=MUTED,
        ha="left",
    )
    fig.text(
        0.055,
        0.022,
        "The result supports typed refusal and explicit override consequences, not universal splitter superiority.",
        fontsize=9.2,
        color=TEXT,
        fontweight="bold",
        ha="left",
    )
    return fig


def main() -> None:
    for path in (SVG_PATH.parent, PDF_PATH.parent, PNG_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    fig.savefig(SVG_PATH, format="svg")
    fig.savefig(PDF_PATH, format="pdf")
    fig.savefig(PNG_PATH, format="png", dpi=210)
    plt.close(fig)
    print(f"SVG: {SVG_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"PNG: {PNG_PATH}")


if __name__ == "__main__":
    main()
