"""
Fifth real-data reproduction: corroded RC beam residual moment capacity.

Data:
    Monotonic Flexural Testing of Corroded Reinforced Concrete Beams Database
    Zenodo DOI: 10.5281/zenodo.8062007

Question:
    Does random validation overstate regression performance relative to
    leave-one-experimental-program/source validation?
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs" / "datasets" / "corroded_rc_beam_moment_capacity" / "Zenodo_Database.csv"
OUT_DIR = ROOT / "outputs" / "reproductions" / "10-5281-zenodo-8062007"
FIG_DIR = ROOT.parent / "figures" / "draft"

SOURCE_COL = "Author, Specimen ID "
TARGET_RAW = "Mmax,exp (kNm)"
TARGET = "log_mmax_exp"

FEATURES = [
    "Cross-section",
    "Test Type and Configuration",
    "Sustained Loading (% Ultimate)",
    "Width (mm)",
    "Depth (mm)",
    "Length (mm)",
    "Test Length (mm)",
    "Side Cover to Ctr of Tension Bar (mm)",
    "Bottom Cover to Ctr of Tension Bar (mm)",
    "Top Cover to Ctr of Compression Bar (mm)",
    "Reinforcement Design ",
    "# Tensile Bars",
    "Diameter Tensile Bars, db,t (mm)",
    "Tension Reinforcement Ratio, pten (%)",
    "# Compression Bars",
    "Diameter Comp Bars, db,c (mm)",
    "Longitudinal Bar Type",
    "Tension Bar End Anchorage",
    "Lsplice (mm)",
    "fy Longitudinal Bars (Tensile), (MPa) ",
    "fsu Long Bars, (MPa)",
    "Comp Reinforcement Ratio, pcom (%)",
    "# Stirrup Legs",
    "Stirrup Bar Type",
    "Stirrup Spacing, s (mm) ",
    "Stirrup Diameter, ds (mm)",
    "fy,s Stirrup Bars",
    "stirrup_volumetric_ratio",
    "Cement Type",
    "W/C Ratio",
    "Max Aggregate Size (mm)",
    "Comp. Test Method",
    "f'c (MPa) ",
    "Corrosion Method",
    "Cathode Type",
    "Corrosion Zone Length, Lc (mm)",
    "corrosion_current_density",
    "Duration, (days)",
    "Solution Concetration (% NaCl)",
    "Immersion Depth (mm)",
    "Wet/Dry Cyclic Ratio (days)",
    "mass_loss_tensile_bars",
    "Average Sample Length, (mm)",
    "Corrosion Penetration Depth, Xaver (mm)",
    "Shear Span, x (mm)",
]


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


def parse_source(value: object) -> str:
    text = str(value).strip()
    return text.split(",")[0].strip() if "," in text else text


def normalize_problem_columns(raw: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for col in raw.columns:
        if col.startswith("Stirrup Volumetric Ratio"):
            rename[col] = "stirrup_volumetric_ratio"
        elif col.startswith("Corrosion Current Density"):
            rename[col] = "corrosion_current_density"
        elif col.startswith("Mass Loss (Tensile bars)"):
            rename[col] = "mass_loss_tensile_bars"
    return raw.rename(columns=rename)


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, list[str], list[str]]:
    raw = normalize_problem_columns(pd.read_csv(DATA))
    raw = raw[raw[TARGET_RAW].notna()].copy()
    raw[TARGET] = np.log(pd.to_numeric(raw[TARGET_RAW], errors="coerce"))
    raw["source_group"] = raw[SOURCE_COL].apply(parse_source)

    data = raw[FEATURES + [TARGET, "source_group", SOURCE_COL]].copy()
    numeric_features = [col for col in FEATURES if pd.api.types.is_numeric_dtype(data[col])]
    categorical_features = [col for col in FEATURES if col not in numeric_features]
    for col in categorical_features:
        data[col] = data[col].fillna("missing").astype(str).str.strip()

    data = data.dropna(subset=[TARGET, "source_group"]).reset_index(drop=True)
    group_sizes = data["source_group"].value_counts().rename_axis("source_group").reset_index(name="n")
    return data[FEATURES], data[TARGET], data["source_group"], group_sizes, numeric_features, categorical_features


def make_models(numeric_features: list[str], categorical_features: list[str]) -> dict[str, Pipeline]:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", make_one_hot_encoder())]),
                categorical_features,
            ),
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
            [("preprocess", preprocess), ("model", GradientBoostingRegressor(random_state=42))]
        ),
    }


def collect_predictions(model_name: str, model: Pipeline, validation: str, X, y, splitter, groups=None) -> pd.DataFrame:
    rows = []
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
                    "source_group": groups.iloc[test_idx].to_numpy() if groups is not None else "random_fold",
                    "observed": y.iloc[test_idx].to_numpy(),
                    "predicted": pred,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize(predictions: pd.DataFrame, n_samples: int, n_groups: int) -> pd.DataFrame:
    rows = []
    for (model, validation), part in predictions.groupby(["model", "validation"]):
        fold_metrics = []
        for _, fold in part.groupby("fold"):
            r2 = r2_score(fold["observed"], fold["predicted"]) if len(fold) >= 2 else np.nan
            mae = mean_absolute_error(fold["observed"], fold["predicted"])
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
                pooled_r2=float(r2_score(part["observed"], part["predicted"])),
                pooled_mae=float(mean_absolute_error(part["observed"], part["predicted"])),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def make_figures(results: pd.DataFrame, group_sizes: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    models = ["Ridge", "RandomForest", "GradientBoosting"]
    validations = ["RandomKFold", "GroupKFold_Source", "LeaveOneSourceOut"]
    x = np.arange(len(models))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for i, validation in enumerate(validations):
        values = [
            results[(results["model"] == model) & (results["validation"] == validation)]["pooled_r2"].iloc[0]
            for model in models
        ]
        ax.bar(x + (i - 1) * width, values, width=width, label=validation)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Pooled R^2 on log moment capacity")
    ax.set_title("A  Random vs source-aware validation", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    sizes = np.sort(group_sizes["n"].to_numpy())[::-1]
    ax.bar(np.arange(1, len(sizes) + 1), sizes, color="#59A14F")
    ax.set_yscale("log")
    ax.set_xlabel("Source group rank")
    ax.set_ylabel("Number of beam tests")
    ax.set_title("B  Experimental-program topology", loc="left", fontweight="bold")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig01_corroded_rc_beam_source_validation.png", dpi=300)
    fig.savefig(FIG_DIR / "fig05_corroded_rc_beam_source_validation.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups, group_sizes, numeric_features, categorical_features = load_data()
    n_groups = groups.nunique()
    validations = {
        "RandomKFold": (KFold(n_splits=10, shuffle=True, random_state=42), None),
        "GroupKFold_Source": (GroupKFold(n_splits=min(10, n_groups)), groups),
        "LeaveOneSourceOut": (LeaveOneGroupOut(), groups),
    }

    frames = []
    for model_name, model in make_models(numeric_features, categorical_features).items():
        for validation, (splitter, split_groups) in validations.items():
            frames.append(collect_predictions(model_name, model, validation, X, y, splitter, split_groups))

    predictions = pd.concat(frames, ignore_index=True)
    results = summarize(predictions, len(X), n_groups)
    X.assign(**{TARGET: y, "source_group": groups}).to_csv(OUT_DIR / "analysis_data.csv", index=False)
    results.to_csv(OUT_DIR / "results.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    group_sizes.to_csv(OUT_DIR / "source_group_sizes.csv", index=False)
    make_figures(results, group_sizes)

    pivot = results.pivot(index="model", columns="validation", values="pooled_r2")
    summary = {
        "doi": "10.5281/zenodo.8062007",
        "paper": "Corroded RC beam residual moment capacity ML example",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
        "n_source_groups": int(n_groups),
        "target": TARGET,
        "group_proxy": "experimental program parsed from Author, Specimen ID",
        "delta_pooled_r2_random_minus_group": {
            model: float(pivot.loc[model, "RandomKFold"] - pivot.loc[model, "GroupKFold_Source"])
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
    print(f"[OK] {OUT_DIR / 'source_group_sizes.csv'}")
    print(f"[OK] {OUT_DIR / 'fig01_corroded_rc_beam_source_validation.png'}")
    print(f"[OK] {FIG_DIR / 'fig05_corroded_rc_beam_source_validation.png'}")
    print(f"[OK] {OUT_DIR / 'reproduction.json'}")


if __name__ == "__main__":
    main()
