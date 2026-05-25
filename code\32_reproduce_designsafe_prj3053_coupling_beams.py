from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
DATA = ROOT / "data" / "raw" / "designsafe_prj3053_coupling_beams" / "Diagonal Database.csv"
OUT = ROOT / "code" / "outputs" / "reproductions" / "designsafe_prj3053_coupling_beams"
FIG = ROOT / "figures" / "generated"
RANDOM_STATE = 42


@dataclass(frozen=True)
class Task:
    name: str
    target: str


TASKS = [
    Task(name="PRJ-3053 coupling beam peak shear", target="V_m_avg_kips"),
    Task(name="PRJ-3053 coupling beam normalized shear", target="v_m_normalized"),
    Task(name="PRJ-3053 coupling beam chord-rotation capacity", target="CR_capacity_average"),
]


RENAME = {
    "Database Number": "database_number",
    "Reference Number": "reference_number",
    "Reference": "reference",
    "Specimen Identification": "specimen_id",
    "b_w in.": "b_w_in",
    "h in.": "h_in",
    "Confinement ": "confinement",
    "b_c to b_w in.": "b_c_to_b_w_in",
    "b_c to h in.": "b_c_to_h_in",
    "l_n in.": "l_n_in",
    "l_n/h ": "l_n_over_h",
    "f_cm psi": "f_cm_psi",
    "Quantity_diagonal": "quantity_diagonal",
    "alpha deg": "alpha_deg",
    "Diagonal d_b in.": "diagonal_d_b_in",
    "Diagonal f_ym ksi": "diagonal_f_ym_ksi",
    "Quantity_parallel": "quantity_parallel",
    "Parallel d_b in.": "parallel_d_b_in",
    "Parallel f_ym ksi": "parallel_f_ym_ksi",
    "Condition": "condition",
    "Transverse d_b in.": "transverse_d_b_in",
    " f_ytm ksi": "f_ytm_ksi",
    "s in.": "s_in",
    "s/d_b": "s_over_d_b",
    "(s/d_b)_normalized": "s_over_d_b_normalized",
    "A_sh_provided/(b_c s) to b_w %": "ash_bc_s_to_bw_pct",
    "A_sh_provided/(b_c s) to h %": "ash_bc_s_to_h_pct",
    "A_sh_provided/A_sh_required to b_w ": "ash_provided_required_bw",
    "A_sh_provided/A_sh_required to h ": "ash_provided_required_h",
    "V_m- kips": "V_m_minus_kips",
    "V_m+ kips": "V_m_plus_kips",
    "v_m_normalized": "v_m_normalized",
    "CR_capacity - %": "CR_capacity_minus",
    "CR_capacity + %": "CR_capacity_plus",
    "CR_capacity_average %": "CR_capacity_average",
    "Axial Restraint": "axial_restraint",
    "Axial Measured": "axial_measured",
    "P/f_cm/A_g -": "P_fc_Ag_minus",
    "P/f_cm/A_g +": "P_fc_Ag_plus",
    "(P/f_cm/A_g)_average kips": "P_fc_Ag_average",
}


RESPONSE_COLUMNS = {
    "V_m_minus_kips",
    "V_m_plus_kips",
    "V_m_avg_kips",
    "v_m_normalized",
    "CR_capacity_minus",
    "CR_capacity_plus",
    "CR_capacity_average",
}


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_database() -> pd.DataFrame:
    raw = pd.read_csv(DATA, header=None, dtype=str, encoding="utf-8", encoding_errors="replace")
    raw = raw.set_index(0).T.reset_index(drop=True)
    raw = raw.rename(columns=RENAME)
    raw = raw.replace({"x": np.nan, "": np.nan})
    for col in raw.columns:
        if col not in {"reference", "specimen_id", "confinement", "condition", "axial_restraint", "axial_measured"}:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["V_m_avg_kips"] = raw[["V_m_minus_kips", "V_m_plus_kips"]].mean(axis=1)
    raw["source_group"] = raw["reference"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    raw["row_index"] = np.arange(len(raw))
    return raw


def feature_columns(df: pd.DataFrame) -> list[str]:
    forbidden = {"row_index", "reference", "reference_number", "specimen_id", "database_number", "source_group"}
    return [c for c in df.columns if c not in forbidden and c not in RESPONSE_COLUMNS]


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
    clean = df.dropna(subset=[task.target, "source_group"]).copy()
    features = feature_columns(clean)
    x = clean[features].copy()
    y = clean[task.target].astype(float).copy()
    groups = clean["source_group"].astype(str).copy()
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
                            "specimen_id": clean.loc[idx, "specimen_id"],
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
    labels = ["Peak shear", "Normalized\nshear", "Chord-rotation\ncapacity"]
    splits = ["RandomKFold", "GroupKFold", "LeaveOneSourceOut"]
    colors = {"RandomKFold": "#4C78A8", "GroupKFold": "#E45756", "LeaveOneSourceOut": "#72B7B2"}
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})
    x = np.arange(len(tasks))
    width = 0.23
    ax = axes[0]
    for i, split in enumerate(splits):
        vals = [float(rf[(rf["task"].eq(task)) & (rf["split"].eq(split))].iloc[0]["pooled_r2"]) for task in tasks]
        ax.bar(x + (i - 1) * width, vals, width=width, label=split, color=colors[split])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pooled R2")
    ax.set_title("PRJ-3053 coupling beam validation")
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
    fig.savefig(FIG / "figS4_designsafe_prj3053_coupling_beam_validation_v0.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "figS4_designsafe_prj3053_coupling_beam_validation_v0.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_database()
    df.to_csv(OUT / "extracted_coupling_beam_data.csv", index=False)
    result_frames = []
    pred_frames = []
    for task in TASKS:
        results, predictions = run_task(df, task)
        result_frames.append(results)
        pred_frames.append(predictions)
    results = pd.concat(result_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    groups = df["source_group"].value_counts().rename_axis("group").reset_index(name="n")
    results.to_csv(OUT / "results.csv", index=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    groups.to_csv(OUT / "group_sizes.csv", index=False)
    make_figure(results, groups)
    summary = {
        "dataset": "DesignSafe PRJ-3053 Database of Diagonally-Reinforced Concrete Coupling Beams",
        "doi": "10.17603/ds2-g5n8-4p74",
        "n_rows": int(len(df)),
        "n_source_groups": int(df["source_group"].nunique()),
        "group_proxy": "Reference",
        "targets": [task.target for task in TASKS],
        "n_features": len(feature_columns(df)),
        "outputs": {
            "extracted_data": str((OUT / "extracted_coupling_beam_data.csv").relative_to(ROOT)),
            "results": str((OUT / "results.csv").relative_to(ROOT)),
            "predictions": str((OUT / "predictions.csv").relative_to(ROOT)),
            "group_sizes": str((OUT / "group_sizes.csv").relative_to(ROOT)),
            "figure": str((FIG / "figS4_designsafe_prj3053_coupling_beam_validation_v0.png").relative_to(ROOT)),
        },
    }
    (OUT / "reproduction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
