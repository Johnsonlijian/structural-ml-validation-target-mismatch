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
DATA = ROOT / "data" / "raw" / "designsafe_rc_columns"
OUT = ROOT / "code" / "outputs" / "reproductions" / "designsafe_rc_columns"
FIG = ROOT / "figures" / "draft"
RANDOM_STATE = 42

CIRCULAR = DATA / "10.17603-ds2-52bz-0n63" / "Database_Circular_Columns.csv"
RECTANGULAR = DATA / "10.17603-ds2-7qg0-4303" / "Database_Rectangular_Columns.csv"


@dataclass(frozen=True)
class Task:
    case: str
    data: pd.DataFrame
    target: str
    group_col: str
    features: list[str]


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def read_designsafe_csv(path: Path) -> pd.DataFrame:
    # Row 2 contains UI metadata and row 3 marks DATASTART.
    return pd.read_csv(path, skiprows=[1, 2])


def numericize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def prepare_circular() -> Task:
    df = read_designsafe_csv(CIRCULAR)
    df["section_family"] = "circular"
    df["source_group"] = df["Authors"].fillna("unknown").astype(str)
    features = [
        "Section diameter D [in.]",
        "d [in.]",
        "Clear Cover cc [in.]",
        "lc  [in.]",
        "a [in.]",
        "a/d",
        "No.of bars",
        "Angle [degrees]",
        "Bar dia. [in.]",
        "fy (longi. reinf.) [psi]",
        "pL (longi. reinf.)",
        "Trans. reinf. legs perp. to load",
        "Trans. reinf. legs parl. to load",
        "Trans. bar dia. [in.]",
        "Spacing of trans. reinf. (s) [in.]",
        "fy (trans. reinf.) [psi]",
        "pt (trans. reinf. volumetric ratio)",
        "pv (trans. reinf. ratio)",
        "s/d",
        "Seismic hoops or spirals",
        "f'c [psi]",
        "Axial load(P) [kips]",
        "Axial load ratio",
        "Spliced longi. bars",
        "Splice length [in]",
        "Splice height [in]",
        "Test setup",
        "Number of loading directions",
        "section_family",
    ]
    numeric_features = [f for f in features if f != "section_family"]
    df = numericize(df, numeric_features + ["Maximum lateral load (primary) (Vmax1) [kips]"])
    df = df.dropna(subset=["Maximum lateral load (primary) (Vmax1) [kips]", "source_group"])
    return Task(
        case="DesignSafe circular RC columns",
        data=df,
        target="Maximum lateral load (primary) (Vmax1) [kips]",
        group_col="source_group",
        features=features,
    )


def prepare_rectangular() -> Task:
    df = read_designsafe_csv(RECTANGULAR)
    df["section_family"] = "rectangular"
    df["source_group"] = df["First Author"].fillna("unknown").astype(str)
    features = [
        "Section depth (h) [in.]",
        "Section width (b) [in.]",
        "d1 [in.]",
        "d2 [in.]",
        "Clear Cover cc [in.]",
        "lc [in.]",
        "a [in.]",
        "a/d1",
        "Longi. bars along first face (perp.)",
        "Bar dia. [in.]",
        "Longi. bars along second face (perp.)",
        "Longi. bars in middle layers(perp.)",
        "Longi. bars in middle layers (parl.)",
        "fy (longi. reinf.) [psi]",
        "pL (longi. reinf.)",
        "Trans. reinf. legs perp. to load",
        "Trans. reinf. legs parl. to load",
        "Trans. bar dia. [in.]",
        "Spacing of trans. reinf. (s) [in.]",
        "fy (trans. reinf.) [psi]",
        "pt (trans. reinf. volumetric ratio)",
        "pv (trans. reinf. ratio)",
        "s/d1 (primary)",
        "s/d2 (secondary)",
        "Seismic hoops",
        "f'c [psi]",
        "Axial load(P) [kips]",
        "Axial load ratio",
        "Spliced longi. bars",
        "Splice length [in.]",
        "Splice height [in.]",
        "Test configuration",
        "Number of loading directions",
        "section_family",
    ]
    numeric_features = [f for f in features if f != "section_family"]
    df = numericize(df, numeric_features + ["Maximum lateral load (primary) (Vmax1) [kips]"])
    df = df.dropna(subset=["Maximum lateral load (primary) (Vmax1) [kips]", "source_group"])
    return Task(
        case="DesignSafe rectangular RC columns",
        data=df,
        target="Maximum lateral load (primary) (Vmax1) [kips]",
        group_col="source_group",
        features=features,
    )


def make_combined(circular: Task, rectangular: Task) -> Task:
    circ = circular.data.rename(
        columns={
            "Section diameter D [in.]": "section_depth_in",
            "d [in.]": "effective_depth_1_in",
            "lc  [in.]": "clear_length_in",
            "a [in.]": "shear_span_in",
            "a/d": "shear_span_ratio",
            "fy (longi. reinf.) [psi]": "fy_longitudinal_psi",
            "pL (longi. reinf.)": "rho_longitudinal",
            "Spacing of trans. reinf. (s) [in.]": "transverse_spacing_in",
            "fy (trans. reinf.) [psi]": "fy_transverse_psi",
            "pt (trans. reinf. volumetric ratio)": "rho_transverse_vol",
            "pv (trans. reinf. ratio)": "rho_transverse",
            "f'c [psi]": "fc_psi",
            "Axial load(P) [kips]": "axial_load_kips",
            "Axial load ratio": "axial_load_ratio",
            "Maximum lateral load (primary) (Vmax1) [kips]": "vmax1_kips",
        }
    )
    circ["section_width_in"] = circ["section_depth_in"]
    rect = rectangular.data.rename(
        columns={
            "Section depth (h) [in.]": "section_depth_in",
            "Section width (b) [in.]": "section_width_in",
            "d1 [in.]": "effective_depth_1_in",
            "lc [in.]": "clear_length_in",
            "a [in.]": "shear_span_in",
            "a/d1": "shear_span_ratio",
            "fy (longi. reinf.) [psi]": "fy_longitudinal_psi",
            "pL (longi. reinf.)": "rho_longitudinal",
            "Spacing of trans. reinf. (s) [in.]": "transverse_spacing_in",
            "fy (trans. reinf.) [psi]": "fy_transverse_psi",
            "pt (trans. reinf. volumetric ratio)": "rho_transverse_vol",
            "pv (trans. reinf. ratio)": "rho_transverse",
            "f'c [psi]": "fc_psi",
            "Axial load(P) [kips]": "axial_load_kips",
            "Axial load ratio": "axial_load_ratio",
            "Maximum lateral load (primary) (Vmax1) [kips]": "vmax1_kips",
        }
    )
    features = [
        "section_depth_in",
        "section_width_in",
        "effective_depth_1_in",
        "clear_length_in",
        "shear_span_in",
        "shear_span_ratio",
        "fy_longitudinal_psi",
        "rho_longitudinal",
        "transverse_spacing_in",
        "fy_transverse_psi",
        "rho_transverse_vol",
        "rho_transverse",
        "fc_psi",
        "axial_load_kips",
        "axial_load_ratio",
        "section_family",
    ]
    combined = pd.concat([circ, rect], ignore_index=True, sort=False)
    return Task(
        case="DesignSafe RC columns combined",
        data=combined.dropna(subset=["vmax1_kips", "source_group"]),
        target="vmax1_kips",
        group_col="source_group",
        features=features,
    )


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


def run_task(task: Task) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = task.data.copy()
    x = df[task.features].copy()
    y = df[task.target].copy()
    groups = df[task.group_col].astype(str).copy()
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
                fold_scores.append(evaluate(y.iloc[test_idx], pred))
                for idx, p in zip(y.index[test_idx], pred):
                    pred_rows.append(
                        {
                            "case": task.case,
                            "model": model_name,
                            "split": split_name,
                            "fold": fold,
                            "row_index": int(idx),
                            "group": groups.loc[idx],
                            "y_true": float(y.loc[idx]),
                            "y_pred": float(p),
                        }
                    )
            pooled = evaluate(y, y_pred)
            row = {
                "case": task.case,
                "model": model_name,
                "split": split_name,
                "n_samples": int(len(df)),
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
    results = pd.DataFrame(rows)
    results = add_gaps(results)
    group_sizes = groups.value_counts().rename_axis("group").reset_index(name="n").assign(case=task.case)
    return results, pd.DataFrame(pred_rows), group_sizes


def add_gaps(results: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (case, model), sub in results.groupby(["case", "model"], sort=False):
        random = sub[sub["split"].eq("RandomKFold")].iloc[0]
        for _, row in sub.iterrows():
            rec = row.to_dict()
            rec["gap_vs_random"] = float(random["pooled_r2"] - row["pooled_r2"])
            out.append(rec)
    return pd.DataFrame(out)


def make_figure(results: pd.DataFrame) -> None:
    rf = results[results["model"].eq("Random Forest")].copy()
    cases = [
        "DesignSafe circular RC columns",
        "DesignSafe rectangular RC columns",
        "DesignSafe RC columns combined",
    ]
    labels = ["Circular\ncolumns", "Rectangular\ncolumns", "Combined\ncolumns"]
    splits = ["RandomKFold", "GroupKFold", "LeaveOneSourceOut"]
    colors = {"RandomKFold": "#4C78A8", "GroupKFold": "#E45756", "LeaveOneSourceOut": "#72B7B2"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), gridspec_kw={"width_ratios": [1.3, 1.0]})
    ax = axes[0]
    x = np.arange(len(cases))
    width = 0.23
    for i, split in enumerate(splits):
        vals = [float(rf[(rf["case"].eq(case)) & (rf["split"].eq(split))].iloc[0]["pooled_r2"]) for case in cases]
        ax.bar(x + (i - 1) * width, vals, width=width, label=split, color=colors[split])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pooled R2")
    ax.set_title("DesignSafe RC column lateral-strength validation")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    gap_rows = rf[rf["split"].eq("GroupKFold")].set_index("case").loc[cases].reset_index()
    ax.barh(np.arange(len(gap_rows)), gap_rows["gap_vs_random"], color="#E45756")
    ax.set_yticks(np.arange(len(gap_rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Random minus GroupKFold R2")
    ax.set_title("Validation gap")
    for i, gap in enumerate(gap_rows["gap_vs_random"]):
        ax.text(gap + 0.03, i, f"{gap:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figS2_designsafe_rc_column_validation_v0.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "figS2_designsafe_rc_column_validation_v0.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    circular = prepare_circular()
    rectangular = prepare_rectangular()
    combined = make_combined(circular, rectangular)
    result_frames = []
    pred_frames = []
    group_frames = []
    for task in [circular, rectangular, combined]:
        results, predictions, groups = run_task(task)
        result_frames.append(results)
        pred_frames.append(predictions)
        group_frames.append(groups)
    results = pd.concat(result_frames, ignore_index=True)
    predictions = pd.concat(pred_frames, ignore_index=True)
    groups = pd.concat(group_frames, ignore_index=True)
    results.to_csv(OUT / "results.csv", index=False)
    predictions.to_csv(OUT / "predictions.csv", index=False)
    groups.to_csv(OUT / "group_sizes.csv", index=False)
    make_figure(results)
    summary = {
        "datasets": ["10.17603/ds2-52bz-0n63", "10.17603/ds2-7qg0-4303"],
        "target": "Maximum lateral load (primary) (Vmax1) [kips]",
        "group_proxy": "Authors / First Author",
        "outputs": {
            "results": str((OUT / "results.csv").relative_to(ROOT)),
            "predictions": str((OUT / "predictions.csv").relative_to(ROOT)),
            "group_sizes": str((OUT / "group_sizes.csv").relative_to(ROOT)),
            "figure": str((FIG / "figS2_designsafe_rc_column_validation_v0.png").relative_to(ROOT)),
        },
    }
    (OUT / "reproduction.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
