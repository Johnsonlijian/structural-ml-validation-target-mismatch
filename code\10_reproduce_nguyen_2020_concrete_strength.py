"""
Regression reproduction: Nguyen et al. 2021 concrete strength ML paper.

Original paper:
    Nguyen et al. "Efficient machine learning models for prediction of
    concrete strengths", Construction and Building Materials 266 (2021) 120950.

Reproduction logic:
    The paper reports efficient ML models on public concrete-strength datasets.
    This script uses the UCI Concrete Compressive Strength dataset and compares:
        1. RandomKFold
        2. GroupKFold by concrete mix proportions
        3. Leave-one-repeated-mix-out, restricted to mix families with repeats

The key P1 question is whether random validation is optimistic when the same
concrete mix proportions appear in both train and test folds at different ages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "outputs" / "datasets" / "uci_concrete_compressive_strength" / "Concrete_Data.xls"
OUT_DIR = ROOT / "outputs" / "reproductions" / "10-1016-j-conbuildmat-2020-120950"


@dataclass(frozen=True)
class EvalResult:
    model: str
    validation: str
    n_splits: int
    n_samples: int
    n_groups: int
    r2_mean: float
    r2_std: float
    mae_mean: float
    mae_std: float


def clean_column(name: str) -> str:
    mapping = {
        "Cement": "cement",
        "Blast Furnace Slag": "slag",
        "Fly Ash": "fly_ash",
        "Water": "water",
        "Superplasticizer": "superplasticizer",
        "Coarse Aggregate": "coarse_aggregate",
        "Fine Aggregate": "fine_aggregate",
        "Age": "age",
        "Concrete compressive strength": "strength_mpa",
    }
    for key, value in mapping.items():
        if name.startswith(key):
            return value
    raise ValueError(f"Unexpected column: {name}")


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    raw = pd.read_excel(DATASET)
    raw = raw.rename(columns={col: clean_column(col) for col in raw.columns})

    feature_cols = [
        "cement",
        "slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "coarse_aggregate",
        "fine_aggregate",
        "age",
    ]
    mix_cols = feature_cols[:-1]

    data = raw[feature_cols + ["strength_mpa"]].copy()
    for col in feature_cols + ["strength_mpa"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna().reset_index(drop=True)

    # Same mix proportions measured at multiple ages are treated as one group.
    data["mix_group"] = (
        data[mix_cols]
        .round(3)
        .astype(str)
        .agg("|".join, axis=1)
    )
    group_sizes = (
        data["mix_group"]
        .value_counts()
        .rename_axis("mix_group")
        .reset_index(name="n")
    )

    return data[feature_cols], data["strength_mpa"], data["mix_group"], group_sizes


def make_models() -> dict[str, Pipeline]:
    return {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline(
            [
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "GradientBoosting": Pipeline(
            [("model", GradientBoostingRegressor(random_state=42))]
        ),
    }


def iter_splits(splitter, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None):
    if groups is None:
        yield from splitter.split(X, y)
    else:
        yield from splitter.split(X, y, groups)


def collect_predictions(
    model_name: str,
    model: Pipeline,
    validation_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    splitter,
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold, (train_idx, test_idx) in enumerate(iter_splits(splitter, X, y, groups), start=1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "validation": validation_name,
                    "fold": fold,
                    "sample_index": test_idx,
                    "mix_group": groups.iloc[test_idx].to_numpy() if groups is not None else "random_fold",
                    "observed": y.iloc[test_idx].to_numpy(),
                    "predicted": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize_predictions(predictions: pd.DataFrame, n_samples: int, n_groups: int) -> pd.DataFrame:
    rows: list[EvalResult] = []
    for (model, validation), part in predictions.groupby(["model", "validation"]):
        fold_metrics = []
        for _, fold_part in part.groupby("fold"):
            r2 = r2_score(fold_part["observed"], fold_part["predicted"]) if len(fold_part) >= 2 else np.nan
            mae = mean_absolute_error(fold_part["observed"], fold_part["predicted"])
            fold_metrics.append((r2, mae))

        r2_values = np.array([r2 for r2, _ in fold_metrics], dtype=float)
        mae_values = np.array([mae for _, mae in fold_metrics], dtype=float)
        rows.append(
            EvalResult(
                model=model,
                validation=validation,
                n_splits=int(part["fold"].nunique()),
                n_samples=n_samples,
                n_groups=n_groups,
                r2_mean=float(np.nanmean(r2_values)),
                r2_std=float(np.nanstd(r2_values)),
                mae_mean=float(np.nanmean(mae_values)),
                mae_std=float(np.nanstd(mae_values)),
            )
        )

    results = pd.DataFrame([row.__dict__ for row in rows])
    pooled = []
    for (model, validation), part in predictions.groupby(["model", "validation"]):
        pooled.append(
            {
                "model": model,
                "validation": validation,
                "pooled_r2": r2_score(part["observed"], part["predicted"]),
                "pooled_mae": mean_absolute_error(part["observed"], part["predicted"]),
                "n_predictions": len(part),
            }
        )
    return results.merge(pd.DataFrame(pooled), on=["model", "validation"], how="left")


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y, groups, group_sizes = load_data()
    repeated_groups = group_sizes[group_sizes["n"] > 1]["mix_group"]
    repeated_mask = groups.isin(repeated_groups)
    X_repeated = X[repeated_mask].reset_index(drop=True)
    y_repeated = y[repeated_mask].reset_index(drop=True)
    groups_repeated = groups[repeated_mask].reset_index(drop=True)

    validations = {
        "RandomKFold": (X, y, None, KFold(n_splits=10, shuffle=True, random_state=42)),
        "GroupKFold_Mix": (X, y, groups, GroupKFold(n_splits=10)),
        "LeaveOneRepeatedMixOut": (
            X_repeated,
            y_repeated,
            groups_repeated,
            LeaveOneGroupOut(),
        ),
    }

    prediction_frames: list[pd.DataFrame] = []
    for model_name, model in make_models().items():
        for validation_name, (Xv, yv, gv, splitter) in validations.items():
            prediction_frames.append(
                collect_predictions(model_name, model, validation_name, Xv, yv, splitter, gv)
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    results = summarize_predictions(predictions, n_samples=len(X), n_groups=groups.nunique())
    return results, predictions, group_sizes


def make_validation_figure(results: pd.DataFrame) -> None:
    order = ["RandomKFold", "GroupKFold_Mix", "LeaveOneRepeatedMixOut"]
    models = ["Ridge", "RandomForest", "GradientBoosting"]
    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, validation in enumerate(order):
        values = [
            results[(results["model"] == model) & (results["validation"] == validation)]["pooled_r2"].iloc[0]
            for model in models
        ]
        ax.bar(x + (i - 1) * width, values, width=width, label=validation)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Pooled R^2")
    ax.set_title("Concrete strength regression: random vs mix-aware validation")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig01_concrete_mix_validation_gap.png", dpi=300)
    plt.close(fig)


def make_group_size_figure(group_sizes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sizes = np.sort(group_sizes["n"].to_numpy())[::-1]
    ax.bar(np.arange(1, len(sizes) + 1), sizes, color="#F58518")
    ax.set_xlabel("Mix group rank")
    ax.set_ylabel("Number of records")
    ax.set_title("UCI concrete mix-family size distribution")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig02_mix_group_size_distribution.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, predictions, group_sizes = run()
    results.to_csv(OUT_DIR / "results.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    group_sizes.to_csv(OUT_DIR / "mix_group_sizes.csv", index=False)
    make_validation_figure(results)
    make_group_size_figure(group_sizes)

    pivot = results.pivot(index="model", columns="validation", values="pooled_r2")
    summary = {
        "doi": "10.1016/j.conbuildmat.2020.120950",
        "paper": "Nguyen et al. 2021 Construction and Building Materials concrete strength regression",
        "dataset": "UCI Concrete Compressive Strength dataset, DOI 10.24432/C5PK67",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(results["n_samples"].iloc[0]),
        "n_mix_groups": int(results["n_groups"].iloc[0]),
        "n_repeated_mix_groups": int((group_sizes["n"] > 1).sum()),
        "delta_pooled_r2_random_minus_group": {
            model: float(pivot.loc[model, "RandomKFold"] - pivot.loc[model, "GroupKFold_Mix"])
            for model in pivot.index
        },
        "pooled_r2": {
            model: {validation: float(pivot.loc[model, validation]) for validation in pivot.columns}
            for model in pivot.index
        },
        "results": results.to_dict(orient="records"),
    }
    (OUT_DIR / "reproduction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(results.sort_values(["model", "validation"]).to_string(index=False))
    print(f"[OK] {OUT_DIR / 'results.csv'}")
    print(f"[OK] {OUT_DIR / 'predictions.csv'}")
    print(f"[OK] {OUT_DIR / 'mix_group_sizes.csv'}")
    print(f"[OK] {OUT_DIR / 'fig01_concrete_mix_validation_gap.png'}")
    print(f"[OK] {OUT_DIR / 'fig02_mix_group_size_distribution.png'}")
    print(f"[OK] {OUT_DIR / 'reproduction.json'}")


if __name__ == "__main__":
    main()
