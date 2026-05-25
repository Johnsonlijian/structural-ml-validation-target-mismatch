"""Create compact R05 diagnostic figures from generated CSV tables."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
R05 = ROOT / "code" / "outputs" / "r05"
FIG = ROOT / "figures" / "generated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def f(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return float("nan")


def short(label: str) -> str:
    repl = {
        "UCI concrete strength": "UCI concrete",
        "SFRC shear capacity": "SFRC shear",
        "Shear-wall failure classification": "Shear-wall cls.",
        "Stub-CFST section-family holdout": "Stub-CFST",
        "Corroded RC beam moment capacity": "Corroded beam",
        "PRJ-2430 RC wall peak lateral strength|Vmax": "RC wall Vmax",
        "PRJ-2430 RC wall drift capacity|driftCap": "RC wall drift",
        "Cyclic beam-column joint shear": "Cyclic joint",
    }
    return repl.get(label, label)


def topology_null() -> None:
    rows = read_csv(R05 / "topology_null_retraining.csv")
    labels = [short(row["task"]) for row in rows]
    true = np.array([f(row["true_gap"]) for row in rows])
    med = np.array([f(row["pseudo_gap_median"]) for row in rows])
    lo = np.array([f(row["pseudo_gap_95_low"]) for row in rows])
    hi = np.array([f(row["pseudo_gap_95_high"]) for row in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hlines(y, lo, hi, color="#9aa0a6", linewidth=3, label="pseudo-group 95% interval")
    ax.scatter(med, y, color="#5f6368", s=28, label="pseudo median", zorder=3)
    ax.scatter(true, y, color="#b3261e", s=36, label="true grouping gap", zorder=4)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Random minus grouped score gap")
    ax.set_title("R05 topology-preserving retraining null", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"fig_r05_topology_null.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def mitigation_tradeoff() -> None:
    rows = [row for row in read_csv(R05 / "source_balanced_mitigation.csv") if row["variant"] != "vanilla_grouped"]
    labels = [short(row["task"]) + "\n" + row["variant"].replace("_", " ") for row in rows]
    x = np.array([f(row["delta_score_vs_vanilla"]) for row in rows])
    y = np.array([f(row["delta_cvar20_vs_vanilla"]) for row in rows])
    colors = ["#1a73e8" if row["variant"] == "source_balanced_erm" else "#f29900" for row in rows]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.scatter(x, y, c=colors, s=42)
    for xi, yi, label in zip(x, y, labels):
        ax.text(xi, yi, label, fontsize=6.5, ha="left", va="bottom")
    ax.set_xlabel("Grouped score change vs vanilla")
    ax.set_ylabel("CVaR20 error change vs vanilla")
    ax.set_title("R05 mitigation trade-off diagnostics", loc="left")
    ax.text(
        0.02,
        0.02,
        "upper-left: score improves but tail risk worsens\nlower-right: score and tail risk both improve",
        transform=ax.transAxes,
        fontsize=7,
        color="#555555",
    )
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG / f"fig_r05_mitigation_tradeoff.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    topology_null()
    mitigation_tradeoff()
    print(f"[OK] {FIG / 'fig_r05_topology_null.svg'}")
    print(f"[OK] {FIG / 'fig_r05_mitigation_tradeoff.svg'}")


if __name__ == "__main__":
    main()
