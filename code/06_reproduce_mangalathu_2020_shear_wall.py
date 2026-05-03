"""
First executable real-data reproduction for P1.

Case:
  Mangalathu et al. 2020, Engineering Structures
  "Data-driven machine-learning-based seismic failure mode identification of
  reinforced concrete shear walls"

Question:
  How much does random validation overestimate failure-mode classification
  accuracy compared with author-grouped validation?

Output:
  outputs/reproductions/10-1016-j-engstruct-2019-110331/
    - reproduction.json
    - results.csv
    - fig01_validation_gap.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = (
    ROOT
    / "outputs"
    / "datasets"
    / "mangalathu_2020_shear_wall"
    / "Shear_Wall_Database.xlsx"
)
OUT = ROOT / "outputs" / "reproductions" / "10-1016-j-engstruct-2019-110331"
OUT.mkdir(parents=True, exist_ok=True)


NUMERIC_FEATURES = [
    "M/Vlw",
    "lw/tw",
    "ρvwFy,vw/fc",
    "ρhwFy,vw/fc",
    "ρvcFy,vc/fc",
    "ρhcFy,hc/fc",
    "P/fcAg",
    "Ab/Ag",
]
CATEGORICAL_FEATURES = ["Section"]
TARGET = "FailureMode"
GROUP = "Author"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Excel column encoding is garbled as "��"; recover intended rho symbol.
    rename = {
        "��vwFy,vw/fc": "ρvwFy,vw/fc",
        "��hwFy,vw/fc": "ρhwFy,vw/fc",
        "��vcFy,vc/fc": "ρvcFy,vc/fc",
        "��hcFy,hc/fc": "ρhcFy,hc/fc",
    }
    return df.rename(columns=rename)


def model_factory(name: str) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    if name == "LogisticRegression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    elif name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=-1,
        )
    elif name == "GradientBoosting":
        clf = GradientBoostingClassifier(random_state=0)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate_splitter(df: pd.DataFrame, model_name: str, splitter, split_name: str) -> dict:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(str).to_numpy()
    groups = df[GROUP].astype(str).to_numpy()

    accs, bals, f1s = [], [], []
    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        model = model_factory(model_name)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        bals.append(balanced_accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))

    return {
        "model": model_name,
        "cv": split_name,
        "n_splits": len(accs),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_sd": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(bals)),
        "balanced_accuracy_sd": float(np.std(bals)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_sd": float(np.std(f1s)),
    }


def run() -> tuple[pd.DataFrame, dict]:
    df = normalize_columns(pd.read_excel(DATA, sheet_name="Database"))
    keep_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, GROUP, "Specimen"]
    df = df[keep_cols].dropna().copy()

    splitters = {
        "RandomStratifiedKFold-5": StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
        "GroupKFold-by-Author-5": GroupKFold(n_splits=5),
        "LeaveOneAuthorOut": LeaveOneGroupOut(),
    }
    models = ["LogisticRegression", "RandomForest", "GradientBoosting"]

    rows = []
    for model in models:
        for split_name, splitter in splitters.items():
            res = evaluate_splitter(df, model, splitter, split_name)
            rows.append(res)
            print(
                f"{model:18s} | {split_name:24s} | "
                f"acc={res['accuracy_mean']:.3f} | bacc={res['balanced_accuracy_mean']:.3f} | f1={res['macro_f1_mean']:.3f}"
            )

    results = pd.DataFrame(rows)
    summary = {
        "doi": "10.1016/j.engstruct.2019.110331",
        "paper": "Data-driven machine-learning-based seismic failure mode identification of reinforced concrete shear walls",
        "dataset": str(DATA),
        "n_samples": int(len(df)),
        "n_authors": int(df[GROUP].nunique()),
        "target_classes": sorted(df[TARGET].astype(str).unique().tolist()),
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "group_col": GROUP,
        "models": models,
        "result_csv": str(OUT / "results.csv"),
    }
    return results, summary


def make_figure(results: pd.DataFrame) -> None:
    pivot = results.pivot(index="model", columns="cv", values="accuracy_mean")
    order = ["RandomStratifiedKFold-5", "GroupKFold-by-Author-5", "LeaveOneAuthorOut"]
    pivot = pivot[order]
    delta_group = pivot["RandomStratifiedKFold-5"] - pivot["GroupKFold-by-Author-5"]
    delta_loso = pivot["RandomStratifiedKFold-5"] - pivot["LeaveOneAuthorOut"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    x = np.arange(len(pivot.index))
    width = 0.26
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    ax = axes[0]
    for i, cv in enumerate(order):
        ax.bar(x + (i - 1) * width, pivot[cv].values, width=width, label=cv, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=10, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("a · Mangalathu 2020 shear-wall data", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax2 = axes[1]
    ax2.bar(x - 0.2, delta_group.values, width=0.36, color="#B279A2", label="Random - Group")
    ax2.bar(x + 0.2, delta_loso.values, width=0.36, color="#9C755F", label="Random - LOSO")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot.index, rotation=10, ha="right")
    ax2.set_ylabel("Delta accuracy")
    ax2.set_title("b · overstatement from random validation", loc="left")
    ax2.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.suptitle("First real-data reproduction: validation strategy changes reported accuracy", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig01_validation_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results, summary = run()
    results.to_csv(OUT / "results.csv", index=False)
    make_figure(results)

    pivot = results.pivot(index="model", columns="cv", values="accuracy_mean")
    summary["delta_accuracy_random_minus_group"] = (
        pivot["RandomStratifiedKFold-5"] - pivot["GroupKFold-by-Author-5"]
    ).to_dict()
    summary["delta_accuracy_random_minus_loso"] = (
        pivot["RandomStratifiedKFold-5"] - pivot["LeaveOneAuthorOut"]
    ).to_dict()
    (OUT / "reproduction.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] {OUT / 'results.csv'}")
    print(f"[OK] {OUT / 'reproduction.json'}")
    print(f"[OK] {OUT / 'fig01_validation_gap.png'}")


if __name__ == "__main__":
    main()

# Canonical filename: keep this path stable; `_Conflict*` copies are accidental sync artefacts.
