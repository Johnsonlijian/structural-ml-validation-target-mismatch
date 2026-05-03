from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "draft"
TABLE_OUT = ROOT / "manuscript" / "tables"
AUDIT_SUMMARY = ROOT / "code" / "outputs" / "literature" / "reproducible_candidate_audit_summary.csv"
MANUAL_VERIFICATION = ROOT / "code" / "outputs" / "literature" / "manual_reproducibility_verification_v0.csv"


CHECKLIST = [
    ("Target", "State the intended deployment target", "within-source, unseen source, unseen mix, unseen section family, time block, spatial block"),
    ("Grouping", "Define the grouping variable before modelling", "source reference, experimental program, mixture family, author/source family, section family"),
    ("Random baseline", "Report the conventional random split score", "so readers can compare with prior literature"),
    ("Grouped validation", "Report a target-matched grouped split", "GroupKFold, leave-one-source-out, leave-one-family-out"),
    ("Topology", "Report group topology diagnostics", "group count, median size, singleton rate, imbalance"),
    ("Classification balance", "Report class balance within groups", "required for failure-mode classification"),
    ("Regression metrics", "Report pooled and fold-level regression metrics", "pooled R2 plus fold-level diagnostics; MAE/RMSE when available"),
    ("Uncertainty", "Attach uncertainty or sensitivity diagnostics", "bootstrap, fold-summary interval, leave-one-group sensitivity"),
    ("Data provenance", "List original data sources and licenses", "do not treat a PDF repository record as raw data"),
    ("Code release", "Release scripts and environment files", "download, preprocessing, validation, figures"),
    ("Failure log", "Report failed or partial reproductions", "embargo, metadata-only, no source labels, blocked access"),
    ("Claim discipline", "Match claims to the validation target", "do not call within-source interpolation out-of-source generalization"),
]


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.mkdir(parents=True, exist_ok=True)


def make_figure3() -> None:
    summary = pd.read_csv(AUDIT_SUMMARY)
    manual = pd.read_csv(MANUAL_VERIFICATION)

    priority_counts = (
        summary.groupby("reproduction_priority", as_index=False)["n"]
        .sum()
        .set_index("reproduction_priority")
        .reindex(["completed", "high", "medium", "low"])
        .fillna(0)
        .reset_index()
    )

    openalex_high = manual[manual["source"].eq("OpenAlex high")].copy()
    decision_counts = (
        openalex_high.groupby("next_decision", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=True)
    )

    promoted = manual[manual["source"].eq("Opportunistic")].copy()
    promoted_counts = (
        promoted.groupby("next_decision", as_index=False)
        .size()
        .rename(columns={"size": "n"})
        .sort_values("n", ascending=True)
    )

    colors = {
        "completed": "#4C78A8",
        "high": "#F58518",
        "medium": "#54A24B",
        "low": "#B279A2",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1.1, 1.25]})

    ax = axes[0]
    ax.bar(
        priority_counts["reproduction_priority"],
        priority_counts["n"],
        color=[colors.get(x, "#777777") for x in priority_counts["reproduction_priority"]],
    )
    ax.set_title("OpenAlex triage frame")
    ax.set_ylabel("Number of papers")
    ax.set_xlabel("Metadata priority")
    ax.tick_params(axis="x", rotation=25)
    for i, row in priority_counts.iterrows():
        ax.text(i, row["n"] + 4, str(int(row["n"])), ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    ax.barh(decision_counts["next_decision"], decision_counts["n"], color="#E45756")
    ax.set_title("Manual check of metadata-high records")
    ax.set_xlabel("Number of records")
    ax.set_xlim(0, max(1.2, decision_counts["n"].max() + 0.4))
    for i, row in decision_counts.reset_index(drop=True).iterrows():
        ax.text(row["n"] + 0.05, i, str(int(row["n"])), va="center", fontsize=9)

    ax = axes[2]
    ax.barh(promoted_counts["next_decision"], promoted_counts["n"], color="#72B7B2")
    ax.set_title("Dataset-first candidates promoted")
    ax.set_xlabel("Number of datasets")
    ax.set_xlim(0, max(1.2, promoted_counts["n"].max() + 0.4))
    for i, row in promoted_counts.reset_index(drop=True).iterrows():
        ax.text(row["n"] + 0.05, i, str(int(row["n"])), va="center", fontsize=9)

    fig.suptitle("Metadata reproducibility signals require manual verification", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig03_reproducibility_audit_v0.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig03_reproducibility_audit_v0.pdf", bbox_inches="tight")
    plt.close(fig)


def make_checklist_table() -> pd.DataFrame:
    checklist = pd.DataFrame(CHECKLIST, columns=["domain", "requirement", "evidence_to_report"])
    checklist.insert(0, "item", range(1, len(checklist) + 1))
    checklist.to_csv(TABLE_OUT / "reporting_checklist_v0.csv", index=False)

    lines = [
        "# Reporting Checklist v0",
        "",
        "| # | Domain | Requirement | Evidence to report |",
        "|---:|---|---|---|",
    ]
    for row in checklist.itertuples(index=False):
        lines.append(f"| {row.item} | {row.domain} | {row.requirement} | {row.evidence_to_report} |")
    (TABLE_OUT / "reporting_checklist_v0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checklist


def make_figure4(checklist: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.4))
    ax.axis("off")

    ax.text(0.0, 1.02, "Reporting checklist for deployment-matched structural ML validation", fontsize=14, weight="bold")
    ax.text(
        0.0,
        0.965,
        "Each item forces the validation score to identify what kind of generalization it supports.",
        fontsize=10,
        color="#444444",
    )

    y = 0.9
    row_h = 0.068
    for row in checklist.itertuples(index=False):
        bg = "#F5F7FA" if row.item % 2 else "#FFFFFF"
        ax.add_patch(plt.Rectangle((0.0, y - row_h + 0.006), 1.0, row_h, color=bg, transform=ax.transAxes, zorder=0))
        ax.text(0.015, y - 0.018, f"{row.item:02d}", fontsize=9, weight="bold", color="#4C78A8", transform=ax.transAxes)
        ax.text(0.085, y - 0.018, row.domain, fontsize=9, weight="bold", transform=ax.transAxes)
        ax.text(0.25, y - 0.018, row.requirement, fontsize=9, transform=ax.transAxes)
        ax.text(0.25, y - 0.044, row.evidence_to_report, fontsize=8, color="#555555", transform=ax.transAxes)
        y -= row_h

    ax.text(
        0.0,
        0.035,
        "Use this checklist with random and grouped validation scores; a high random score alone is not evidence of out-of-source generalization.",
        fontsize=9,
        color="#333333",
        transform=ax.transAxes,
    )
    fig.savefig(OUT / "fig04_reporting_checklist_v0.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig04_reporting_checklist_v0.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup()
    make_figure3()
    checklist = make_checklist_table()
    make_figure4(checklist)


if __name__ == "__main__":
    main()
