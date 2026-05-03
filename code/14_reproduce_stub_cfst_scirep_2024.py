"""
Fourth real-data reproduction: Stub-CFST Scientific Reports 2024.

Paper:
    "Prediction of the axial compression capacity of stub CFST columns using
    machine learning techniques", Scientific Reports 2024.

Question:
    How much does random validation overstate performance when the model must
    generalize across CFST section families: circular, rectangular, double-skin?

Important caveat:
    The public workbooks do not include paper/source-reference labels. This
    script therefore uses section family as the group proxy. It tests structural
    family generalization, not literature-source leakage.
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
DATA_DIR = ROOT / "outputs" / "datasets" / "stub_cfst_scirep_2024"
OUT_DIR = ROOT / "outputs" / "reproductions" / "10-1038-s41598-024-53352-1"
FIG_DIR = ROOT.parent / "figures" / "draft"


FEATURES = [
    "shape",
    "outer_dim_1",
    "outer_dim_2",
    "outer_thickness",
    "length",
    "fy_outer",
    "fc",
    "inner_dim",
    "inner_thickness",
    "fy_inner",
    "steel_area",
    "concrete_area",
    "nominal_capacity",
    "slenderness",
]
NUMERIC_FEATURES = [f for f in FEATURES if f != "shape"]
CATEGORICAL_FEATURES = ["shape"]
TARGET = "strength_index"
GROUP = "shape"


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
    pooled_r2: float
    pooled_mae: float


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def circular_area(diameter: pd.Series, thickness: pd.Series) -> tuple[pd.Series, pd.Series]:
    outer = np.pi / 4.0 * diameter**2
    inner_d = diameter - 2.0 * thickness
    inner = np.pi / 4.0 * inner_d**2
    return outer - inner, inner


def load_circular() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "PC.xlsx")
    steel_area, concrete_area = circular_area(df["D"], df["t"])
    nominal = (steel_area * df["Fy"] + concrete_area * df["fc"]) / 1000.0
    return pd.DataFrame(
        {
            "shape": "circular",
            "P": df["P"],
            "outer_dim_1": df["D"],
            "outer_dim_2": df["D"],
            "outer_thickness": df["t"],
            "length": df["L"],
            "fy_outer": df["Fy"],
            "fc": df["fc"],
            "inner_dim": 0.0,
            "inner_thickness": 0.0,
            "fy_inner": 0.0,
            "steel_area": steel_area,
            "concrete_area": concrete_area,
            "nominal_capacity": nominal,
            "slenderness": df["D"] / df["t"],
        }
    )


def load_rectangular() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "PR.xlsx")
    outer = df["H"] * df["B"]
    inner_h = df["H"] - 2.0 * df["t"]
    inner_b = df["B"] - 2.0 * df["t"]
    concrete_area = inner_h * inner_b
    steel_area = outer - concrete_area
    nominal = (steel_area * df["Fy"] + concrete_area * df["fc"]) / 1000.0
    return pd.DataFrame(
        {
            "shape": "rectangular",
            "P": df["P"],
            "outer_dim_1": df["H"],
            "outer_dim_2": df["B"],
            "outer_thickness": df["t"],
            "length": df["L"],
            "fy_outer": df["Fy"],
            "fc": df["fc"],
            "inner_dim": 0.0,
            "inner_thickness": 0.0,
            "fy_inner": 0.0,
            "steel_area": steel_area,
            "concrete_area": concrete_area,
            "nominal_capacity": nominal,
            "slenderness": np.maximum(df["H"], df["B"]) / df["t"],
        }
    )


def load_double_skin() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "PDS.xlsx")
    outer_steel, outer_inner_area = circular_area(df["Do"], df["to"])
    inner_steel, _ = circular_area(df["Di"], df["ti"])
    concrete_area = outer_inner_area - np.pi / 4.0 * df["Di"] ** 2
    total_steel = outer_steel + inner_steel
    nominal = (outer_steel * df["fyo"] + inner_steel * df["fyi"] + concrete_area * df["fc"]) / 1000.0
    return pd.DataFrame(
        {
            "shape": "double_skin",
            "P": df["P"],
            "outer_dim_1": df["Do"],
            "outer_dim_2": df["Do"],
            "outer_thickness": df["to"],
            "length": df["L"],
            "fy_outer": df["fyo"],
            "fc": df["fc"],
            "inner_dim": df["Di"],
            "inner_thickness": df["ti"],
            "fy_inner": df["fyi"],
            "steel_area": total_steel,
            "concrete_area": concrete_area,
            "nominal_capacity": nominal,
            "slenderness": df["Do"] / df["to"],
        }
    )


def load_data() -> pd.DataFrame:
    df = pd.concat([load_circular(), load_rectangular(), load_double_skin()], ignore_index=True)
    df[TARGET] = df["P"] / df["nominal_capacity"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
    return df.reset_index(drop=True)


def make_models() -> dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", make_one_hot_encoder(), CATEGORICAL_FEATURES),
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
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [("preprocess", preprocess), ("model", GradientBoostingRegressor(random_state=42))]
        ),
    }


def collect_predictions(model_name: str, model: Pipeline, validation: str, X, y, splitter, groups=None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    splits = splitter.split(X, y, groups) if groups is not None else splitter.split(X, y)
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "validation": validation,
                    "fold": fold,
                    "sample_index": test_idx,
                    "shape": groups.iloc[test_idx].to_numpy() if groups is not None else "random_fold",
                    "observed": y.iloc[test_idx].to_numpy(),
                    "predicted": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize(predictions: pd.DataFrame, n_samples: int, n_groups: int) -> pd.DataFrame:
    rows: list[EvalResult] = []
    for (model, validation), part in predictions.groupby(["model", "validation"]):
        fold_rows = []
        for _, fold in part.groupby("fold"):
            fold_rows.append(
                (
                    r2_score(fold["observed"], fold["predicted"]) if len(fold) >= 2 else np.nan,
                    mean_absolute_error(fold["observed"], fold["predicted"]),
                )
            )
        r2_values = np.array([x[0] for x in fold_rows], dtype=float)
        mae_values = np.array([x[1] for x in fold_rows], dtype=float)
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
                pooled_r2=float(r2_score(part["observed"], part["predicted"])),
                pooled_mae=float(mean_absolute_error(part["observed"], part["predicted"])),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def make_figures(results: pd.DataFrame, group_sizes: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    models = ["Ridge", "RandomForest", "GradientBoosting"]
    validations = ["RandomKFold", "LeaveOneShapeOut"]
    x = np.arange(len(models))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for i, validation in enumerate(validations):
        values = [
            results[(results["model"] == model) & (results["validation"] == validation)]["pooled_r2"].iloc[0]
            for model in models
        ]
        ax.bar(x + (i - 0.5) * width, values, width=width, label=validation)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Pooled R^2")
    ax.set_title("A  Random vs leave-one-shape-out", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar(group_sizes["shape"], group_sizes["n"], color="#59A14F")
    ax.set_ylabel("Number of specimens")
    ax.set_title("B  Section-family topology", loc="left", fontweight="bold")
    for i, row in group_sizes.reset_index(drop=True).iterrows():
        ax.text(i, row["n"] + 20, str(int(row["n"])), ha="center")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig01_stub_cfst_shape_generalization.png", dpi=300)
    fig.savefig(FIG_DIR / "fig04_stub_cfst_shape_generalization.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP]

    validations = {
        "RandomKFold": (KFold(n_splits=10, shuffle=True, random_state=42), None),
        "LeaveOneShapeOut": (LeaveOneGroupOut(), groups),
        "GroupKFold_Shape": (GroupKFold(n_splits=3), groups),
    }

    frames = []
    for model_name, model in make_models().items():
        for validation, (splitter, split_groups) in validations.items():
            frames.append(collect_predictions(model_name, model, validation, X, y, splitter, split_groups))

    predictions = pd.concat(frames, ignore_index=True)
    results = summarize(predictions, len(df), groups.nunique())
    group_sizes = groups.value_counts().rename_axis("shape").reset_index(name="n")

    df.to_csv(OUT_DIR / "combined_stub_cfst_data.csv", index=False)
    results.to_csv(OUT_DIR / "results.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    group_sizes.to_csv(OUT_DIR / "shape_group_sizes.csv", index=False)
    make_figures(results, group_sizes)

    pivot = results.pivot(index="model", columns="validation", values="pooled_r2")
    summary = {
        "doi": "10.1038/s41598-024-53352-1",
        "paper": "Prediction of the axial compression capacity of stub CFST columns using machine learning techniques",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(df)),
        "n_shape_groups": int(groups.nunique()),
        "target": TARGET,
        "group_proxy": "section family (circular, rectangular, double_skin)",
        "caveat": "No source-reference labels are available in the public workbooks; this tests cross-section-family generalization, not literature-source leakage.",
        "delta_pooled_r2_random_minus_shape": {
            model: float(pivot.loc[model, "RandomKFold"] - pivot.loc[model, "LeaveOneShapeOut"])
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
    print(f"[OK] {OUT_DIR / 'shape_group_sizes.csv'}")
    print(f"[OK] {OUT_DIR / 'fig01_stub_cfst_shape_generalization.png'}")
    print(f"[OK] {FIG_DIR / 'fig04_stub_cfst_shape_generalization.png'}")
    print(f"[OK] {OUT_DIR / 'reproduction.json'}")


if __name__ == "__main__":
    main()
