"""
Regression reproduction: Rahman et al. 2021 SFRC shear-strength ML paper.

Original paper:
    Rahman et al. "Data-driven shear strength prediction of steel fiber
    reinforced concrete beams using machine learning approach",
    Engineering Structures 233 (2021) 111743.

Reproduction logic:
    The paper reports 10-fold cross-validation on a compiled SFRC beam database.
    This script uses the public Lantsoght 2019 Zenodo database that is cited by
    related SFRC ML papers and evaluates the same broad task under:
        1. RandomKFold
        2. GroupKFold by source reference
        3. LeaveOneSourceOut

The key P1 question is whether random cross-validation is optimistic when rows
from the same experimental source family appear in both train and test folds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "outputs" / "datasets" / "lantsoght_2019_sfrc_shear" / "database_SFRC.xlsx"
OUT_DIR = ROOT / "outputs" / "reproductions" / "10-1016-j-engstruct-2020-111743"


@dataclass(frozen=True)
class EvalResult:
    model: str
    validation: str
    n_splits: int
    n_samples: int
    r2_mean: float
    r2_std: float
    mae_mean: float
    mae_std: float


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both older and newer scikit-learn keyword names."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def source_groups(reference: pd.Series) -> pd.Series:
    """Construct stable source groups from partially merged reference cells."""
    groups: list[str] = []
    current_author = "unknown"
    current_source = "unknown"
    for value in reference:
        if pd.isna(value):
            groups.append(current_source)
            continue

        text = str(value).strip()
        if not text:
            groups.append(current_source)
            continue

        if any(ch.isalpha() for ch in text):
            current_author = text
            current_source = text
        elif text.replace(".0", "").isdigit():
            current_source = f"{current_author} {text.replace('.0', '')}"
        else:
            current_source = text
        groups.append(current_source)

    return pd.Series(groups, index=reference.index, name="source_group")


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    raw = pd.read_excel(DATASET, header=1)

    # Drop the units row and normalize the few columns used in Rahman-like models.
    data = raw.iloc[1:].copy()
    rename_by_position = {
        data.columns[16]: "rho_l",
        data.columns[18]: "a_over_d",
        data.columns[21]: "fc",
        data.columns[22]: "fiber_type",
        data.columns[25]: "Vf",
        data.columns[27]: "L_over_d",
        data.columns[29]: "F",
        data.columns[35]: "Vu_kN",
    }
    data = data.rename(columns=rename_by_position)
    data["source_group"] = source_groups(data["Reference"])

    numeric_features = ["a_over_d", "fc", "rho_l", "Vf", "L_over_d", "F"]
    categorical_features = ["fiber_type"]
    target = "Vu_kN"

    for col in numeric_features + [target]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["fiber_type"] = data["fiber_type"].fillna("unknown").astype(str).str.strip().str.lower()

    keep = numeric_features + categorical_features + [target, "source_group", "Reference", "ID"]
    data = data[keep].dropna(subset=numeric_features + [target, "source_group"]).reset_index(drop=True)

    X = data[numeric_features + categorical_features]
    y = data[target]
    groups = data["source_group"]
    return X, y, groups, numeric_features, categorical_features


def make_models(numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
        ]
    )
    return {
        "Ridge": Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline(
            [
                ("preprocess", preprocess),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("preprocess", preprocess),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
    }


def evaluate_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series, splitter, groups=None) -> tuple[float, float, float, float, int]:
    r2_values: list[float] = []
    mae_values: list[float] = []
    splits = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)

    for train_idx, test_idx in splits:
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        if len(test_idx) >= 2:
            r2_values.append(r2_score(y.iloc[test_idx], pred))
        mae_values.append(mean_absolute_error(y.iloc[test_idx], pred))

    return (
        float(np.mean(r2_values)),
        float(np.std(r2_values)),
        float(np.mean(mae_values)),
        float(np.std(mae_values)),
        len(mae_values),
    )


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
    splits = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        fold_rows = pd.DataFrame(
            {
                "model": model_name,
                "validation": validation_name,
                "fold": fold,
                "sample_index": test_idx,
                "source_group": groups.iloc[test_idx].to_numpy() if groups is not None else "random_fold",
                "observed": y.iloc[test_idx].to_numpy(),
                "predicted": pred,
            }
        )
        rows.append(fold_rows)

    return pd.concat(rows, ignore_index=True)


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, validation), part in predictions.groupby(["model", "validation"]):
        rows.append(
            {
                "model": model,
                "validation": validation,
                "pooled_r2": r2_score(part["observed"], part["predicted"]),
                "pooled_mae": mean_absolute_error(part["observed"], part["predicted"]),
                "n_predictions": len(part),
            }
        )
    return pd.DataFrame(rows)


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y, groups, numeric_features, categorical_features = load_data()
    models = make_models(numeric_features, categorical_features)

    n_groups = groups.nunique()
    validations = {
        "RandomKFold": (KFold(n_splits=10, shuffle=True, random_state=42), None),
        "GroupKFold_Source": (GroupKFold(n_splits=min(10, n_groups)), groups),
        "LeaveOneSourceOut": (LeaveOneGroupOut(), groups),
    }

    rows: list[EvalResult] = []
    prediction_frames: list[pd.DataFrame] = []
    for model_name, model in models.items():
        for validation_name, (splitter, split_groups) in validations.items():
            r2_mean, r2_std, mae_mean, mae_std, n_splits = evaluate_cv(
                model, X, y, splitter, split_groups
            )
            prediction_frames.append(
                collect_predictions(model_name, model, validation_name, X, y, splitter, split_groups)
            )
            rows.append(
                EvalResult(
                    model=model_name,
                    validation=validation_name,
                    n_splits=n_splits,
                    n_samples=len(X),
                    r2_mean=r2_mean,
                    r2_std=r2_std,
                    mae_mean=mae_mean,
                    mae_std=mae_std,
                )
            )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = pooled_metrics(predictions)
    results = pd.DataFrame([row.__dict__ for row in rows]).merge(
        pooled, on=["model", "validation"], how="left"
    )
    return results, predictions, groups.value_counts().rename_axis("source_group").reset_index(name="n")


def make_figure(results: pd.DataFrame) -> None:
    order = ["RandomKFold", "GroupKFold_Source", "LeaveOneSourceOut"]
    models = list(results["model"].unique())
    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, validation in enumerate(order):
        values = [
            results[(results["model"] == model) & (results["validation"] == validation)]["r2_mean"].iloc[0]
            for model in models
        ]
        ax.bar(x + (i - 1) * width, values, width=width, label=validation)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Mean R^2 across folds")
    ax.set_title("SFRC shear-strength regression: random vs source-aware validation")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig01_sfrc_regression_validation_gap.png", dpi=300)
    plt.close(fig)


def make_group_size_figure(group_sizes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sizes = np.sort(group_sizes["n"].to_numpy())[::-1]
    ax.bar(np.arange(1, len(sizes) + 1), sizes, color="#4C78A8")
    ax.set_xlabel("Source group rank")
    ax.set_ylabel("Number of specimens")
    ax.set_title("SFRC source-group size distribution")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig02_source_group_size_distribution.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, predictions, group_sizes = run()
    results.to_csv(OUT_DIR / "results.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    group_sizes.to_csv(OUT_DIR / "source_group_sizes.csv", index=False)
    make_figure(results)
    make_group_size_figure(group_sizes)

    pivot = results.pivot(index="model", columns="validation", values="r2_mean")
    pooled_pivot = results.pivot(index="model", columns="validation", values="pooled_r2")
    summary = {
        "doi": "10.1016/j.engstruct.2020.111743",
        "paper": "Rahman et al. 2021 Engineering Structures SFRC shear-strength regression",
        "dataset": "Lantsoght 2019 Zenodo SFRC shear database, DOI 10.5281/zenodo.2578061",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(results["n_samples"].iloc[0]),
        "n_source_groups": int(len(group_sizes)),
        "delta_r2_random_minus_group": {
            model: float(pivot.loc[model, "RandomKFold"] - pivot.loc[model, "GroupKFold_Source"])
            for model in pivot.index
        },
        "pooled_r2": {
            model: {validation: float(pooled_pivot.loc[model, validation]) for validation in pooled_pivot.columns}
            for model in pooled_pivot.index
        },
        "delta_r2_random_minus_logo": {
            model: float(pivot.loc[model, "RandomKFold"] - pivot.loc[model, "LeaveOneSourceOut"])
            for model in pivot.index
        },
        "results": results.to_dict(orient="records"),
    }
    (OUT_DIR / "reproduction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(results.to_string(index=False))
    print(f"[OK] {OUT_DIR / 'results.csv'}")
    print(f"[OK] {OUT_DIR / 'predictions.csv'}")
    print(f"[OK] {OUT_DIR / 'source_group_sizes.csv'}")
    print(f"[OK] {OUT_DIR / 'fig01_sfrc_regression_validation_gap.png'}")
    print(f"[OK] {OUT_DIR / 'fig02_source_group_size_distribution.png'}")
    print(f"[OK] {OUT_DIR / 'reproduction.json'}")


if __name__ == "__main__":
    main()
