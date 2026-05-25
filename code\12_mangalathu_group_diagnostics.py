"""
Diagnostics for the Mangalathu 2020 shear-wall classification case.

The first reproduction showed a large RandomKFold vs GroupKFold accuracy gap.
This script checks whether the author/source groups are tiny or class-imbalanced,
which is the most likely reviewer challenge to the interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = (
    ROOT
    / "outputs"
    / "datasets"
    / "mangalathu_2020_shear_wall"
    / "Shear_Wall_Database.xlsx"
)
OUT = ROOT / "outputs" / "reproductions" / "10-1016-j-engstruct-2019-110331"
FIG_DIR = ROOT.parent / "figures" / "generated"

TARGET = "FailureMode"
GROUP = "Author"


def load_data() -> pd.DataFrame:
    df = pd.read_excel(DATA, sheet_name="Database")
    keep = [GROUP, TARGET, "Specimen"]
    df = df[keep].dropna().copy()
    df[GROUP] = df[GROUP].astype(str).str.strip()
    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df


def normalized_entropy(counts: pd.Series) -> float:
    values = counts[counts > 0].astype(float).to_numpy()
    if len(values) <= 1:
        return 0.0
    p = values / values.sum()
    entropy = -float(np.sum(p * np.log(p)))
    return entropy / np.log(len(values))


def build_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_balance = (
        df.pivot_table(index=GROUP, columns=TARGET, values="Specimen", aggfunc="count", fill_value=0)
        .sort_index()
    )
    class_balance["n"] = class_balance.sum(axis=1)
    class_cols = [col for col in class_balance.columns if col != "n"]
    class_balance["majority_class"] = class_balance[class_cols].idxmax(axis=1)
    class_balance["majority_share"] = class_balance[class_cols].max(axis=1) / class_balance["n"]
    class_balance["n_classes"] = (class_balance[class_cols] > 0).sum(axis=1)
    class_balance["class_entropy"] = class_balance[class_cols].apply(normalized_entropy, axis=1)
    class_balance = class_balance.sort_values("n", ascending=False)

    group_sizes = class_balance[["n", "majority_class", "majority_share", "n_classes", "class_entropy"]].copy()
    group_sizes = group_sizes.reset_index()

    overall = df[TARGET].value_counts().rename_axis(TARGET).reset_index(name="n")
    overall["share"] = overall["n"] / overall["n"].sum()

    return group_sizes, class_balance.reset_index(), overall


def make_figure(group_sizes: pd.DataFrame, class_balance: pd.DataFrame, overall: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    sizes = np.sort(group_sizes["n"].to_numpy())[::-1]
    ax.bar(np.arange(1, len(sizes) + 1), sizes, color="#4C78A8")
    ax.set_title("A  Author/source group sizes", loc="left", fontweight="bold")
    ax.set_xlabel("Group rank")
    ax.set_ylabel("Number of specimens")
    ax.set_yscale("log")

    ax = axes[0, 1]
    ax.bar(overall[TARGET], overall["share"], color="#F58518")
    ax.set_title("B  Overall class distribution", loc="left", fontweight="bold")
    ax.set_ylabel("Share")
    ax.tick_params(axis="x", rotation=25)

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 11)
    ax.hist(group_sizes["majority_share"], bins=bins, color="#E45756", edgecolor="white")
    ax.set_title("C  Within-group majority-class share", loc="left", fontweight="bold")
    ax.set_xlabel("Majority-class share")
    ax.set_ylabel("Number of groups")

    ax = axes[1, 1]
    class_cols = [
        col
        for col in class_balance.columns
        if col not in {GROUP, "n", "majority_class", "majority_share", "n_classes", "class_entropy"}
    ]
    top = class_balance.sort_values("n", ascending=False).head(20).copy()
    heat = top[class_cols].div(top[class_cols].sum(axis=1), axis=0).to_numpy()
    im = ax.imshow(heat, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_title("D  Top-20 groups: class shares", loc="left", fontweight="bold")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top[GROUP].astype(str).str.slice(0, 16), fontsize=7)
    ax.set_xticks(np.arange(len(class_cols)))
    ax.set_xticklabels(class_cols, rotation=25, ha="right")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class share")

    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle(
        "Mangalathu 2020 diagnostic: author groups are uneven and often class-skewed",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig02_mangalathu_group_diagnostics.png", dpi=300)
    fig.savefig(FIG_DIR / "fig03_mangalathu_group_diagnostics.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    group_sizes, class_balance, overall = build_tables(df)

    group_sizes.to_csv(OUT / "author_group_sizes.csv", index=False)
    class_balance.to_csv(OUT / "author_class_balance.csv", index=False)
    overall.to_csv(OUT / "failure_mode_distribution.csv", index=False)

    summary = {
        "n_samples": int(len(df)),
        "n_author_groups": int(group_sizes[GROUP].nunique()),
        "median_group_size": float(group_sizes["n"].median()),
        "max_group_size": int(group_sizes["n"].max()),
        "singleton_groups": int((group_sizes["n"] == 1).sum()),
        "groups_with_single_class": int((group_sizes["n_classes"] == 1).sum()),
        "median_majority_share": float(group_sizes["majority_share"].median()),
        "overall_class_distribution": overall.to_dict(orient="records"),
    }
    (OUT / "group_diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(group_sizes, class_balance, overall)

    print(json.dumps(summary, indent=2))
    print(f"[OK] {OUT / 'author_group_sizes.csv'}")
    print(f"[OK] {OUT / 'author_class_balance.csv'}")
    print(f"[OK] {OUT / 'failure_mode_distribution.csv'}")
    print(f"[OK] {OUT / 'group_diagnostics.json'}")
    print(f"[OK] {OUT / 'fig02_mangalathu_group_diagnostics.png'}")
    print(f"[OK] {FIG_DIR / 'fig03_mangalathu_group_diagnostics.png'}")


if __name__ == "__main__":
    main()
