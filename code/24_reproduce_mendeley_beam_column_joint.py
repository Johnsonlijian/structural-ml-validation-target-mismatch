from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "mendeley_beam_column_joint"
OUT = ROOT / "code" / "outputs" / "reproductions" / "mendeley_beam_column_joint"
FIG = ROOT / "figures" / "draft"

TEKLEWOIN = DATA / "10.17632-8ndgpm7zw7.1" / "Teklewoin_Joint_Dataset.xlsx"
SALEM = DATA / "10.17632-rbhfnz32sy.1" / "Salem_Beam_Column_Joint_Cyclic_Shear.xlsx"
RANDOM_STATE = 42


@dataclass(frozen=True)
class Task:
    case: str
    task: str
    target: str
    group_col: str
    features: list[str]
    metric_kind: str
    data: pd.DataFrame


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\n", " ") for c in out.columns]
    return out


def numericize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_tekklewoin_tasks() -> list[Task]:
    df = normalize_columns(pd.read_excel(TEKLEWOIN, sheet_name="Dataset"))
    df["Authors"] = df["Authors"].ffill()
    numeric_cols = [
        "Hc [mm]",
        "L [mm]",
        "hc [mm]",
        "bc [mm]",
        "hb [mm]",
        "bb [mm]",
        "fcm [MPa]",
        "ρsvfyv[MPa]",
        "ρshfyh [MPa]",
        "NEd [kN]",
        "VExp [kN]",
    ]
    df = numericize(df, numeric_cols).dropna(subset=["Authors", "VExp [kN]", "FM"])
    base_features = [
        "Hc [mm]",
        "L [mm]",
        "hc [mm]",
        "bc [mm]",
        "hb [mm]",
        "bb [mm]",
        "fcm [MPa]",
        "ρsvfyv[MPa]",
        "ρshfyh [MPa]",
        "NEd [kN]",
    ]
    return [
        Task(
            case="Mendeley exterior RC beam-column joints",
            task="shear_strength_regression",
            target="VExp [kN]",
            group_col="Authors",
            features=base_features,
            metric_kind="regression",
            data=df,
        ),
        Task(
            case="Mendeley exterior RC beam-column joints",
            task="failure_mode_classification",
            target="FM",
            group_col="Authors",
            features=base_features,
            metric_kind="classification",
            data=df,
        ),
    ]


def load_salem_task() -> Task:
    df = normalize_columns(pd.read_excel(SALEM, sheet_name="Sheet1"))
    df["Research Team"] = df["Research Team"].ffill()
    drop_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    df = df.drop(columns=drop_cols)
    numeric_cols = [
        "joint shear strength (MPa)",
        "ρcore (%)",
        "beam Long. RFT ratio %",
        "e (mm)",
        "N (kN)",
        "hb (mm)",
        "bb (mm)",
        "hc (mm)",
        "bc (mm)",
        "fy (MPa)",
        "fc (MPa)",
    ]
    df = numericize(df, numeric_cols).dropna(subset=["Research Team", "joint shear strength (MPa)"])
    features = [
        "ρcore (%)",
        "beam Long. RFT ratio %",
        "e (mm)",
        "N (kN)",
        "hb (mm)",
        "bb (mm)",
        "hc (mm)",
        "bc (mm)",
        "fy (MPa)",
        "fc (MPa)",
        "Joint Type, JT",
    ]
    return Task(
        case="Mendeley cyclic beam-column joint shear",
        task="joint_shear_strength_regression",
        target="joint shear strength (MPa)",
        group_col="Research Team",
        features=features,
        metric_kind="regression",
        data=df,
    )


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical_features = [c for c in x.columns if c not in numeric_features]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())]), categorical_features),
        ],
        remainder="drop",
    )


def models(metric_kind: str) -> dict[str, object]:
    if metric_kind == "regression":
        return {
            "Ridge": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=2),
            "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        }
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def splitters(task: Task, y: pd.Series, groups: pd.Series) -> dict[str, object]:
    n_groups = groups.nunique()
    n_splits = min(5, n_groups)
    if task.metric_kind == "classification":
        random_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    else:
        random_splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return {
        "RandomKFold": random_splitter,
        "GroupKFold": GroupKFold(n_splits=n_splits),
        "LeaveOneSourceOut": LeaveOneGroupOut(),
    }


def evaluate_predictions(metric_kind: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    if metric_kind == "regression":
        return {
            "pooled_r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def run_cv(task: Task) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = task.data.dropna(subset=[task.target, task.group_col]).copy()
    x = df[task.features].copy()
    y = df[task.target].copy()
    groups = df[task.group_col].astype(str).copy()

    results = []
    predictions = []
    for model_name, estimator in models(task.metric_kind).items():
        pipeline = Pipeline([("preprocess", build_preprocessor(x)), ("model", estimator)])
        for split_name, splitter in splitters(task, y, groups).items():
            y_pred = pd.Series(index=y.index, dtype=object if task.metric_kind == "classification" else float)
            fold_scores = []
            if split_name == "RandomKFold":
                split_iter = splitter.split(x, y)
            else:
                split_iter = splitter.split(x, y, groups)
            for fold, (train_idx, test_idx) in enumerate(split_iter):
                fitted = clone(pipeline)
                fitted.fit(x.iloc[train_idx], y.iloc[train_idx])
                pred = fitted.predict(x.iloc[test_idx])
                y_pred.iloc[test_idx] = pred
                fold_scores.append(evaluate_predictions(task.metric_kind, y.iloc[test_idx], pred))
                for idx, p in zip(y.index[test_idx], pred):
                    predictions.append(
                        {
                            "case": task.case,
                            "task": task.task,
                            "model": model_name,
                            "split": split_name,
                            "fold": fold,
                            "row_index": int(idx),
                            "group": groups.loc[idx],
                            "y_true": y.loc[idx],
                            "y_pred": p,
                        }
                    )
            pooled = evaluate_predictions(task.metric_kind, y, y_pred.to_numpy())
            row = {
                "case": task.case,
                "task": task.task,
                "model": model_name,
                "split": split_name,
                "n_samples": int(len(df)),
                "n_groups": int(groups.nunique()),
                "median_group_size": float(groups.value_counts().median()),
                "max_group_size": int(groups.value_counts().max()),
            }
            row.update(pooled)
            for metric in fold_scores[0]:
                values = [score[metric] for score in fold_scores]
                row[f"fold_mean_{metric}"] = float(np.mean(values))
                row[f"fold_sd_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            results.append(row)

    group_sizes = (
        groups.value_counts()
        .rename_axis("group")
        .reset_index(name="n")
        .assign(case=task.case, task=task.task)
    )
    return pd.DataFrame(results), pd.DataFrame(predictions), group_sizes


def add_gaps(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in results.groupby(["case", "task", "model"], sort=False):
        random = sub[sub["split"].eq("RandomKFold")]
        if random.empty:
            continue
        random_row = random.iloc[0]
        for _, row in sub.iterrows():
            out = row.to_dict()
            if row["split"] == "RandomKFold":
                out["gap_vs_random"] = 0.0
                out["gap_vs_random_balanced_accuracy"] = 0.0
                out["gap_vs_random_f1_macro"] = 0.0
            elif "pooled_r2" in sub.columns and pd.notna(row.get("pooled_r2")):
                out["gap_vs_random"] = float(random_row["pooled_r2"] - row["pooled_r2"])
                out["gap_vs_random_balanced_accuracy"] = np.nan
                out["gap_vs_random_f1_macro"] = np.nan
            elif "accuracy" in sub.columns and pd.notna(row.get("accuracy")):
                out["gap_vs_random"] = float(random_row["accuracy"] - row["accuracy"])
                rb = random_row.get("balanced_accuracy")
                gb = row.get("balanced_accuracy")
                out["gap_vs_random_balanced_accuracy"] = (
                    float(rb - gb) if pd.notna(rb) and pd.notna(gb) else np.nan
                )
                rf1 = random_row.get("f1_macro")
                gf1 = row.get("f1_macro")
                out["gap_vs_random_f1_macro"] = (
                    float(rf1 - gf1) if pd.notna(rf1) and pd.notna(gf1) else np.nan
                )
            else:
                out["gap_vs_random"] = np.nan
                out["gap_vs_random_balanced_accuracy"] = np.nan
                out["gap_vs_random_f1_macro"] = np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def make_figure(results: pd.DataFrame) -> None:
    best = results[results["model"].eq("Random Forest")].copy()
    order = [
        "shear_strength_regression",
        "failure_mode_classification",
        "joint_shear_strength_regression",
    ]
    labels = {
        "shear_strength_regression": "Exterior joint\nshear strength",
        "failure_mode_classification": "Exterior joint\nfailure mode",
        "joint_shear_strength_regression": "Cyclic joint\nshear strength",
    }
    metric_col = {
        "shear_strength_regression": "pooled_r2",
        "failure_mode_classification": "accuracy",
        "joint_shear_strength_regression": "pooled_r2",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    x = np.arange(len(order))
    width = 0.25
    for i, split in enumerate(["RandomKFold", "GroupKFold", "LeaveOneSourceOut"]):
        vals = []
        for task in order:
            row = best[(best["task"].eq(task)) & (best["split"].eq(split))]
            vals.append(float(row.iloc[0][metric_col[task]]) if not row.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=split)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[t] for t in order])
    ax.set_ylabel("Headline metric (R2 or accuracy)")
    ax.set_title("Random vs source-aware validation")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    gap_rows = best[best["split"].isin(["GroupKFold", "LeaveOneSourceOut"])].copy()
    ylabels = [f"{labels[t].replace(chr(10), ' ')}\n{split}" for t, split in zip(gap_rows["task"], gap_rows["split"])]
    ax.barh(np.arange(len(gap_rows)), gap_rows["gap_vs_random"], color="#E45756")
    ax.set_yticks(np.arange(len(gap_rows)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Random minus stricter validation")
    ax.set_title("Validation gap")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_mendeley_beam_column_joint_validation_v0.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "fig06_mendeley_beam_column_joint_validation_v0.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    tasks = [*load_tekklewoin_tasks(), load_salem_task()]

    result_frames = []
    prediction_frames = []
    group_frames = []
    for task in tasks:
        results, predictions, group_sizes = run_cv(task)
        result_frames.append(results)
        prediction_frames.append(predictions)
        group_frames.append(group_sizes)

    results = add_gaps(pd.concat(result_frames, ignore_index=True))
    predictions = pd.concat(prediction_frames, ignore_index=True)
    group_sizes = pd.concat(group_frames, ignore_index=True)

    results.to_csv(OUT / "results.csv", index=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    group_sizes.to_csv(OUT / "group_sizes.csv", index=False)
    make_figure(results)

    summary = {
        "datasets": [
            "10.17632/8ndgpm7zw7.1",
            "10.17632/rbhfnz32sy.1",
        ],
        "tasks": sorted(results["task"].unique()),
        "outputs": {
            "results": str((OUT / "results.csv").relative_to(ROOT)),
            "predictions": str((OUT / "predictions.csv").relative_to(ROOT)),
            "group_sizes": str((OUT / "group_sizes.csv").relative_to(ROOT)),
            "figure": str((FIG / "fig06_mendeley_beam_column_joint_validation_v0.png").relative_to(ROOT)),
        },
        "interpretive_note": "Source-aware grouping uses author/reference codes in the 203-test dataset and research team blocks in the 98-test dataset. This is a candidate sixth reproduction, pending manuscript integration.",
    }
    (OUT / "reproduction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

# Canonical filename: keep this path stable; `_Conflict*` copies are accidental sync artefacts.
