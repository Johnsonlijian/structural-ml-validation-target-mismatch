from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "designsafe_prj2430_wall" / "WallData.mat"
OUT = ROOT / "code" / "outputs" / "reproductions" / "designsafe_prj2430_wall"
FIG = ROOT / "figures" / "draft"
RANDOM_STATE = 42


@dataclass(frozen=True)
class Task:
    name: str
    target: str
    features: list[str]


TASKS = [
    Task(name="PRJ-2430 RC wall peak lateral strength", target="Vmax", features=[]),
    Task(name="PRJ-2430 RC wall drift capacity", target="driftCap", features=[]),
]


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def to_python_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return np.nan
        if value.size == 1:
            return to_python_scalar(value.item())
        return np.nan
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if value is None:
        return np.nan
    return value


def finite_or_nan(value):
    value = to_python_scalar(value)
    if isinstance(value, str):
        return value
    try:
        if value is None or not math.isfinite(float(value)):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return value


def array_value(arr, idx):
    try:
        return finite_or_nan(arr[idx])
    except Exception:
        return np.nan


def struct_field_by_id(struct, unique_id: str, field: str):
    if not hasattr(struct, unique_id):
        return np.nan
    item = getattr(struct, unique_id)
    if not hasattr(item, field):
        return np.nan
    return finite_or_nan(getattr(item, field))


def add_array_fields(rows: list[dict[str, object]], prefix: str, struct, fields: list[str]) -> None:
    for field in fields:
        if not hasattr(struct, field):
            continue
        values = getattr(struct, field)
        if not isinstance(values, np.ndarray) or values.shape[0] != len(rows):
            continue
        for idx, row in enumerate(rows):
            row[f"{prefix}_{field}"] = array_value(values, idx)


def extract_dataframe() -> pd.DataFrame:
    wall_data = loadmat(DATA, squeeze_me=True, struct_as_record=False)["WallData"]
    rows: list[dict[str, object]] = []
    for idx, unique_id in enumerate(wall_data.UniqueID):
        uid = str(unique_id)
        rows.append(
            {
                "row_index": idx,
                "Authors": array_value(wall_data.Authors, idx),
                "SpecimenID": array_value(wall_data.SpecimenID, idx),
                "UniqueID": uid,
                "Vmax": struct_field_by_id(wall_data.ExperimentalData, uid, "Vmax"),
                "Dmax": struct_field_by_id(wall_data.ExperimentalData, uid, "Dmax"),
                "driftCap": struct_field_by_id(wall_data.ExperimentalData, uid, "driftCap"),
                "dispCap": struct_field_by_id(wall_data.ExperimentalData, uid, "dispCap"),
                "numStorys_exp": struct_field_by_id(wall_data.ExperimentalData, uid, "numStorys"),
                "total_axial_load": struct_field_by_id(wall_data.Loading.TotalAxialLoad, uid, ""),
            }
        )

    # Struct fields keyed by specimen id do not use field names inside TotalAxialLoad.
    for row in rows:
        uid = str(row["UniqueID"])
        if hasattr(wall_data.Loading.TotalAxialLoad, uid):
            row["total_axial_load"] = finite_or_nan(getattr(wall_data.Loading.TotalAxialLoad, uid))
        hist = getattr(wall_data.Loading.CyclicHistory, uid, None)
        if isinstance(hist, np.ndarray) and hist.size:
            arr = np.asarray(hist, dtype=float)
            row["cyclic_history_points"] = int(arr.shape[0])
            row["cyclic_history_max_abs"] = float(np.nanmax(np.abs(arr[:, 0]))) if arr.ndim == 2 else np.nan
        else:
            row["cyclic_history_points"] = np.nan
            row["cyclic_history_max_abs"] = np.nan

    add_array_fields(rows, "walltype", wall_data.WallType, ["Shape", "SteelLayout"])
    add_array_fields(
        rows,
        "geom",
        wall_data.Geometry,
        [
            "t",
            "h",
            "l",
            "h_v",
            "l_be",
            "endCover",
            "sideCover",
            "t_f",
            "b_f",
            "AspectRatio",
            "ShearSpan",
            "SlenderRatio",
            "BeRatio",
            "WallVolume",
            "WallSurfaceArea",
            "Area",
            "numStory",
            "Igross",
            "h_measured",
            "l_bef",
            "LoadingDirection",
            "BuildingAspectRatio",
        ],
    )
    add_array_fields(
        rows,
        "reinf",
        wall_data.Reinf,
        [
            "rho_be",
            "rho_v",
            "rho_v_all",
            "rho_h",
            "rho_z",
            "rho_h_be",
            "NoCH",
            "NoCV",
            "s_h",
            "s_v",
            "s_hoop",
            "d_be",
            "d_v",
            "d_h",
            "d_hoop",
            "rho_vol",
            "s_hoop_e",
            "rho_be_e",
            "d_be_e",
            "d_v_e",
            "d_h_e",
            "d_hoop_e",
            "rho_vol_e",
        ],
    )
    add_array_fields(
        rows,
        "mat",
        wall_data.Material,
        [
            "age",
            "fc",
            "eps",
            "Ec",
            "fr",
            "fsp",
            "fy_be",
            "fu_be",
            "fy_v",
            "fu_v",
            "fy_h",
            "fu_h",
            "fy_hoop",
            "fu_hoop",
            "eps_ult_be",
            "eps_ult_v",
            "eps_max",
            "fyt_web",
            "fut_web",
            "fy_be_e",
            "fu_be_e",
            "eps_ult_be_e",
            "eps_ult_hoop",
        ],
    )
    add_array_fields(rows, "loading", wall_data.Loading, ["Mtop_Vtop", "LoadingType", "AxialLoad"])
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col not in ["Authors", "SpecimenID", "UniqueID", "walltype_Shape", "walltype_SteelLayout", "loading_LoadingType"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    forbidden = {
        "row_index",
        "Authors",
        "SpecimenID",
        "UniqueID",
        "Vmax",
        "Dmax",
        "driftCap",
        "dispCap",
        "numStorys_exp",
    }
    return [c for c in df.columns if c not in forbidden]


def build_pipeline(x: pd.DataFrame, estimator) -> Pipeline:
    numeric_features = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical_features = [c for c in x.columns if c not in numeric_features]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())]), categorical_features),
        ]
    )
    return Pipeline([("preprocess", pre), ("model", estimator)])


def models() -> dict[str, object]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, min_samples_leaf=2),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {"pooled_r2": float(r2_score(y_true, y_pred)), "mae": float(mean_absolute_error(y_true, y_pred))}


def add_gaps(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, model), sub in results.groupby(["task", "model"], sort=False):
        random = sub[sub["split"].eq("RandomKFold")].iloc[0]
        for _, row in sub.iterrows():
            rec = row.to_dict()
            rec["gap_vs_random"] = float(random["pooled_r2"] - row["pooled_r2"])
            rows.append(rec)
    return pd.DataFrame(rows)


def run_task(df: pd.DataFrame, task: Task) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = feature_columns(df)
    clean = df.dropna(subset=[task.target, "Authors"]).copy()
    x = clean[features].copy()
    y = clean[task.target].astype(float).copy()
    groups = clean["Authors"].astype(str).copy()
    n_group_splits = min(5, groups.nunique())
    splitters = {
        "RandomKFold": KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        "GroupKFold": GroupKFold(n_splits=n_group_splits),
        "LeaveOneSourceOut": LeaveOneGroupOut(),
    }
    rows = []
    pred_rows = []
    for model_name, estimator in models().items():
        pipeline = build_pipeline(x, estimator)
        for split_name, splitter in splitters.items():
            y_pred = pd.Series(index=y.index, dtype=float)
            fold_scores = []
            split_iter = splitter.split(x, y) if split_name == "RandomKFold" else splitter.split(x, y, groups)
            for fold, (train_idx, test_idx) in enumerate(split_iter):
                fitted = clone(pipeline)
                fitted.fit(x.iloc[train_idx], y.iloc[train_idx])
                pred = fitted.predict(x.iloc[test_idx])
                y_pred.iloc[test_idx] = pred
                if len(test_idx) > 1:
                    fold_scores.append(evaluate(y.iloc[test_idx], pred))
                else:
                    fold_scores.append({"pooled_r2": np.nan, "mae": float(mean_absolute_error(y.iloc[test_idx], pred))})
                for idx, p in zip(y.index[test_idx], pred):
                    pred_rows.append(
                        {
                            "task": task.name,
                            "target": task.target,
                            "model": model_name,
                            "split": split_name,
                            "fold": fold,
                            "row_index": int(clean.loc[idx, "row_index"]),
                            "unique_id": clean.loc[idx, "UniqueID"],
                            "group": groups.loc[idx],
                            "y_true": float(y.loc[idx]),
                            "y_pred": float(p),
                        }
                    )
            pooled = evaluate(y, y_pred)
            row = {
                "task": task.name,
                "target": task.target,
                "model": model_name,
                "split": split_name,
                "n_samples": int(len(clean)),
                "n_groups": int(groups.nunique()),
                "median_group_size": float(groups.value_counts().median()),
                "max_group_size": int(groups.value_counts().max()),
            }
            row.update(pooled)
            for metric in fold_scores[0]:
                vals = [s[metric] for s in fold_scores]
                row[f"fold_mean_{metric}"] = float(np.nanmean(vals))
                row[f"fold_sd_{metric}"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
            rows.append(row)
    return add_gaps(pd.DataFrame(rows)), pd.DataFrame(pred_rows)


def make_figure(results: pd.DataFrame, groups: pd.DataFrame) -> None:
    rf = results[results["model"].eq("Random Forest")].copy()
    tasks = [task.name for task in TASKS]
    labels = ["Peak lateral\nstrength", "Drift\ncapacity"]
    splits = ["RandomKFold", "GroupKFold", "LeaveOneSourceOut"]
    colors = {"RandomKFold": "#4C78A8", "GroupKFold": "#E45756", "LeaveOneSourceOut": "#72B7B2"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    ax = axes[0]
    x = np.arange(len(tasks))
    width = 0.23
    for i, split in enumerate(splits):
        vals = [float(rf[(rf["task"].eq(task)) & (rf["split"].eq(split))].iloc[0]["pooled_r2"]) for task in tasks]
        ax.bar(x + (i - 1) * width, vals, width=width, label=split, color=colors[split])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pooled R2")
    ax.set_title("PRJ-2430 source-aware validation")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    gap_rows = rf[rf["split"].eq("GroupKFold")].set_index("task").loc[tasks].reset_index()
    ax.barh(np.arange(len(gap_rows)), gap_rows["gap_vs_random"], color="#E45756")
    ax.set_yticks(np.arange(len(gap_rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Random minus GroupKFold R2")
    ax.set_title("Validation gap")
    for i, gap in enumerate(gap_rows["gap_vs_random"]):
        ax.text(gap + 0.02, i, f"{gap:.3f}", va="center", fontsize=8)

    ax = axes[2]
    top_groups = groups.sort_values("n", ascending=False).head(12)
    ax.barh(np.arange(len(top_groups)), top_groups["n"], color="#777777")
    ax.set_yticks(np.arange(len(top_groups)))
    ax.set_yticklabels(top_groups["group"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Specimens")
    ax.set_title("Largest source groups")
    fig.tight_layout()
    fig.savefig(FIG / "figS3_designsafe_prj2430_wall_validation_v0.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "figS3_designsafe_prj2430_wall_validation_v0.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = extract_dataframe()
    features = feature_columns(df)
    for task in TASKS:
        object.__setattr__(task, "features", features)
    df.to_csv(OUT / "extracted_wall_data.csv", index=False)
    result_frames = []
    pred_frames = []
    for task in TASKS:
        results, predictions = run_task(df, task)
        result_frames.append(results)
        pred_frames.append(predictions)
    results = pd.concat(result_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    groups = df["Authors"].astype(str).value_counts().rename_axis("group").reset_index(name="n")
    results.to_csv(OUT / "results.csv", index=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    groups.to_csv(OUT / "group_sizes.csv", index=False)
    make_figure(results, groups)
    summary = {
        "dataset": "DesignSafe PRJ-2430 UoA-UW Reinforced Concrete Wall Database",
        "doi": "10.17603/ds2-r12q-t415",
        "n_rows": int(len(df)),
        "n_source_groups": int(df["Authors"].nunique()),
        "group_proxy": "Authors",
        "targets": [task.target for task in TASKS],
        "n_features": len(features),
        "outputs": {
            "extracted_data": str((OUT / "extracted_wall_data.csv").relative_to(ROOT)),
            "results": str((OUT / "results.csv").relative_to(ROOT)),
            "predictions": str((OUT / "predictions.csv").relative_to(ROOT)),
            "group_sizes": str((OUT / "group_sizes.csv").relative_to(ROOT)),
            "figure": str((FIG / "figS3_designsafe_prj2430_wall_validation_v0.png").relative_to(ROOT)),
        },
    }
    (OUT / "reproduction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
