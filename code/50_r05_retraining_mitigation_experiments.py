"""R05 retraining and mitigation experiments for the SAVP manuscript.

R04 added prediction-level diagnostics. R05 upgrades the strongest remaining
review risks into true computational checks:

1. topology-preserving pseudo-group *retraining* nulls;
2. retrained source-balanced ERM and group-loss-reweighted ERM baselines;
3. group-aware conformal coverage diagnostics;
4. physics monotonicity counterfactual checks;
5. simple empirical/baseline comparisons and severity-threshold sensitivity.

The experiments are deliberately diagnostic. They do not claim state-of-the-art
mitigation, design-code calibration, or field-wide prevalence.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
FIG_OUT = ROOT / "code" / "outputs" / "figures"
R05_OUT = ROOT / "code" / "outputs" / "r05"
TABLE_OUT = ROOT / "manuscript" / "tables"
RANDOM_STATE = 20260513
PSEUDO_REPS = 30
EPS = 1e-12


@dataclass
class Task:
    module: str
    name: str
    metric_kind: str
    x: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    monotonic: dict[str, int]
    baseline_features: list[str]
    empirical_constant: float | None = None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def cvar(values: list[float], tail: float = 0.20) -> float | None:
    if not values:
        return None
    n_tail = max(1, math.ceil(len(values) * tail))
    return mean(sorted(values, reverse=True)[:n_tail])


def scale_for(y: Iterable[float]) -> float:
    values = [abs(float(v)) for v in y]
    return mean(values) if values and mean(values) > EPS else 1.0


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clean_xyg(x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = x.copy()
    df["_target"] = pd.Series(y).to_numpy()
    df["_group"] = pd.Series(groups).astype(str).to_numpy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["_target", "_group"]).reset_index(drop=True)
    y_out = pd.to_numeric(df.pop("_target"), errors="coerce")
    g_out = df.pop("_group").astype(str)
    keep = y_out.notna() & g_out.notna()
    return df.loc[keep].reset_index(drop=True), y_out.loc[keep].reset_index(drop=True), g_out.loc[keep].reset_index(drop=True)


def load_tasks() -> list[Task]:
    tasks: list[Task] = []

    uci = load_module(CODE / "10_reproduce_nguyen_2020_concrete_strength.py", "r05_uci")
    x, y, groups, _ = uci.load_data()
    x, y, groups = clean_xyg(x, y, groups)
    tasks.append(
        Task(
            module="UCI concrete",
            name="UCI concrete strength",
            metric_kind="regression",
            x=x,
            y=y,
            groups=groups,
            monotonic={"cement": 1, "water": -1, "age": 1},
            baseline_features=["cement", "water", "age", "slag", "fly_ash"],
        )
    )

    sfrc = load_module(CODE / "08_reproduce_rahman_2021_sfrc_regression.py", "r05_sfrc")
    x, y, groups, _, _ = sfrc.load_data()
    x, y, groups = clean_xyg(x, y, groups)
    tasks.append(
        Task(
            module="SFRC shear",
            name="SFRC shear capacity",
            metric_kind="regression",
            x=x,
            y=y,
            groups=groups,
            monotonic={"fc": 1, "rho_l": 1, "Vf": 1, "F": 1},
            baseline_features=["a_over_d", "fc", "rho_l", "Vf", "F"],
        )
    )

    mang = load_module(CODE / "06_reproduce_mangalathu_2020_shear_wall.py", "r05_mangalathu")
    df = mang.normalize_columns(pd.read_excel(mang.DATA, sheet_name="Database"))
    df = df[mang.NUMERIC_FEATURES + mang.CATEGORICAL_FEATURES + [mang.TARGET, mang.GROUP]].dropna().reset_index(drop=True)
    x, y, groups = clean_xyg(df[mang.NUMERIC_FEATURES + mang.CATEGORICAL_FEATURES], df[mang.TARGET].astype(str), df[mang.GROUP].astype(str))
    tasks.append(
        Task(
            module="Mangalathu shear-wall",
            name="Shear-wall failure classification",
            metric_kind="classification",
            x=x,
            y=y.astype(str),
            groups=groups,
            monotonic={},
            baseline_features=[],
        )
    )

    stub = load_module(CODE / "14_reproduce_stub_cfst_scirep_2024.py", "r05_stub")
    df = stub.load_data()
    x, y, groups = clean_xyg(df[stub.FEATURES], df[stub.TARGET], df[stub.GROUP])
    tasks.append(
        Task(
            module="Stub-CFST",
            name="Stub-CFST section-family holdout",
            metric_kind="regression",
            x=x,
            y=y,
            groups=groups,
            # The modeled target is strength_index = measured / nominal capacity,
            # not raw axial capacity. Do not impose capacity monotonicity here.
            monotonic={},
            baseline_features=["fc", "fy_outer", "steel_area", "concrete_area", "slenderness"],
            empirical_constant=1.0,
        )
    )

    cor = load_module(CODE / "16_reproduce_corroded_rc_beam_moment.py", "r05_corroded")
    x, y, groups, _, _, _ = cor.load_data()
    x, y, groups = clean_xyg(x, y, groups)
    tasks.append(
        Task(
            module="Corroded RC beams",
            name="Corroded RC beam moment capacity",
            metric_kind="regression",
            x=x,
            y=y,
            groups=groups,
            monotonic={"f'c (MPa) ": 1, "Width (mm)": 1, "Depth (mm)": 1, "mass_loss_tensile_bars": -1},
            baseline_features=["Width (mm)", "Depth (mm)", "f'c (MPa) ", "mass_loss_tensile_bars"],
        )
    )

    try:
        wall = load_module(CODE / "30_reproduce_designsafe_prj2430_wall.py", "r05_wall")
        df = wall.extract_dataframe()
        for task_def in wall.TASKS:
            clean = df.dropna(subset=[task_def.target, "Authors"]).copy()
            features = wall.feature_columns(clean)
            x, y, groups = clean_xyg(clean[features], clean[task_def.target], clean["Authors"].astype(str))
            monotonic = {"mat_fc": 1, "geom_Area": 1, "reinf_rho_v": 1, "reinf_rho_h": 1} if task_def.target == "Vmax" else {}
            baseline_features = [c for c in ["mat_fc", "geom_Area", "geom_ShearSpan", "reinf_rho_v", "loading_AxialLoad"] if c in x.columns]
            tasks.append(
                Task(
                    module="PRJ-2430 RC walls",
                    name=f"{task_def.name}|{task_def.target}",
                    metric_kind="regression",
                    x=x,
                    y=y,
                    groups=groups,
                    monotonic=monotonic,
                    baseline_features=baseline_features,
                )
            )
    except Exception as exc:
        print(f"[warn] skip PRJ-2430 loader: {exc}")

    try:
        mend = load_module(CODE / "24_reproduce_mendeley_beam_column_joint.py", "r05_mendeley")
        salem = mend.load_salem_task()
        x, y, groups = clean_xyg(salem.data[salem.features], salem.data[salem.target], salem.data[salem.group_col].astype(str))
        tasks.append(
            Task(
                module="Mendeley cyclic joint",
                name="Cyclic beam-column joint shear",
                metric_kind="regression",
                x=x,
                y=y,
                groups=groups,
                monotonic={"fc (MPa)": 1, "fy (MPa)": 1, "beam Long. RFT ratio %": 1},
                baseline_features=["fc (MPa)", "fy (MPa)", "beam Long. RFT ratio %", "N (kN)"],
            )
        )
    except Exception as exc:
        print(f"[warn] skip Mendeley loader: {exc}")

    return tasks


def build_pipeline(x: pd.DataFrame, metric_kind: str, learner: str = "Random Forest") -> Pipeline:
    numeric = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    categorical = [c for c in x.columns if c not in numeric]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", make_ohe())]), categorical),
        ],
        remainder="drop",
    )
    if metric_kind == "regression":
        if learner == "Ridge":
            estimator: BaseEstimator = Ridge(alpha=1.0)
        elif learner == "Gradient Boosting":
            estimator = GradientBoostingRegressor(random_state=RANDOM_STATE)
        else:
            estimator = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        if learner == "Logistic Regression":
            estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        elif learner == "Gradient Boosting":
            estimator = GradientBoostingClassifier(random_state=RANDOM_STATE)
        else:
            estimator = RandomForestClassifier(
                n_estimators=120,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def split_iter(task: Task, split: str, groups: pd.Series | None = None):
    if split == "random":
        if task.metric_kind == "classification":
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(task.x, task.y)
        return KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(task.x, task.y)
    split_groups = task.groups if groups is None else groups
    n_splits = min(5, int(pd.Series(split_groups).nunique()))
    return GroupKFold(n_splits=n_splits).split(task.x, task.y, split_groups)


def group_weights(groups: pd.Series) -> np.ndarray:
    counts = groups.value_counts()
    weights = groups.map(lambda g: 1.0 / counts.loc[g]).astype(float).to_numpy()
    return weights / np.mean(weights)


def group_loss_reweighted_weights(task: Task, model: Pipeline, train_idx: np.ndarray, base_weights: np.ndarray, eta: float = 1.0) -> np.ndarray:
    x_train = task.x.iloc[train_idx]
    y_train = task.y.iloc[train_idx]
    groups_train = task.groups.iloc[train_idx].reset_index(drop=True)
    local_weights = base_weights[train_idx].astype(float).copy()
    local_weights = local_weights / np.mean(local_weights)
    fitted = clone(model)
    for _ in range(3):
        fitted.fit(x_train, y_train, model__sample_weight=local_weights)
        pred = fitted.predict(x_train)
        if task.metric_kind == "classification":
            losses = pd.Series((pred.astype(str) != y_train.astype(str).to_numpy()).astype(float))
        else:
            scale = scale_for(y_train.astype(float))
            losses = pd.Series(np.abs(pred.astype(float) - y_train.astype(float).to_numpy()) / scale)
        group_loss = losses.groupby(groups_train).mean()
        centered = group_loss - group_loss.mean()
        group_factor = np.exp(np.clip(eta * centered, -3.0, 3.0))
        group_factor = group_factor / group_factor.mean()
        local_weights = groups_train.map(group_factor).to_numpy() * base_weights[train_idx]
        local_weights = local_weights / np.mean(local_weights)
    return local_weights


def evaluate_oof(
    task: Task,
    split: str,
    learner: str = "Random Forest",
    groups: pd.Series | None = None,
    weighting: str = "vanilla",
) -> dict[str, object]:
    model = build_pipeline(task.x, task.metric_kind, learner)
    y_pred = pd.Series(index=task.y.index, dtype=object)
    prob_rows: list[dict[str, float]] = [{} for _ in range(len(task.y))]
    fold_ids = pd.Series(index=task.y.index, dtype=int)
    base_weights = group_weights(task.groups)
    for fold, (train_idx, test_idx) in enumerate(split_iter(task, split, groups), start=1):
        fitted = clone(model)
        fit_kwargs: dict[str, object] = {}
        if weighting == "source_balanced":
            fit_kwargs["model__sample_weight"] = base_weights[train_idx]
        elif weighting == "group_loss_reweighted":
            fit_kwargs["model__sample_weight"] = group_loss_reweighted_weights(task, model, train_idx, base_weights)
        fitted.fit(task.x.iloc[train_idx], task.y.iloc[train_idx], **fit_kwargs)
        pred = fitted.predict(task.x.iloc[test_idx])
        y_pred.iloc[test_idx] = pred
        fold_ids.iloc[test_idx] = fold
        if task.metric_kind == "classification" and hasattr(fitted, "predict_proba"):
            probs = fitted.predict_proba(task.x.iloc[test_idx])
            classes = [str(c) for c in fitted.named_steps["model"].classes_]
            for local, sample in enumerate(test_idx):
                prob_rows[sample] = {label: float(probs[local, j]) for j, label in enumerate(classes)}
    return summarize_oof(task, y_pred, prob_rows, fold_ids, split, learner, weighting)


def summarize_oof(
    task: Task,
    y_pred: pd.Series,
    probs: list[dict[str, float]],
    fold_ids: pd.Series,
    split: str,
    learner: str,
    weighting: str,
) -> dict[str, object]:
    y_true = task.y.reset_index(drop=True)
    pred = y_pred.reset_index(drop=True)
    groups = task.groups.reset_index(drop=True)
    if task.metric_kind == "classification":
        y_s = y_true.astype(str).to_numpy()
        p_s = pred.astype(str).to_numpy()
        metric = accuracy_score(y_s, p_s)
        secondary = balanced_accuracy_score(y_s, p_s)
        macro = f1_score(y_s, p_s, average="macro")
        group_losses = []
        for _, idx in groups.groupby(groups).groups.items():
            idx_list = list(idx)
            group_losses.append(mean((p_s[idx_list] != y_s[idx_list]).astype(float)))
        return {
            "module": task.module,
            "task": task.name,
            "metric_kind": task.metric_kind,
            "learner": learner,
            "split": split,
            "weighting": weighting,
            "n": len(task.y),
            "n_groups": int(groups.nunique()),
            "score": float(metric),
            "secondary_score": float(secondary),
            "macro_f1": float(macro),
            "mae": "",
            "nmae": "",
            "worst_group_nmae_or_error": float(max(group_losses)),
            "cvar20_group_nmae_or_error": float(cvar(group_losses) or 0.0),
            "overprediction_rate": "",
            "y_true": y_s,
            "y_pred": p_s,
            "groups": groups.to_numpy(),
            "probs": probs,
        }
    y = y_true.astype(float).to_numpy()
    p = pred.astype(float).to_numpy()
    scale = scale_for(y)
    group_nmae = []
    group_over = []
    for _, idx in groups.groupby(groups).groups.items():
        idx_list = list(idx)
        group_nmae.append(mean(np.abs(p[idx_list] - y[idx_list]) / scale))
        group_over.append(mean((p[idx_list] > y[idx_list]).astype(float)))
    return {
        "module": task.module,
        "task": task.name,
        "metric_kind": task.metric_kind,
        "learner": learner,
        "split": split,
        "weighting": weighting,
        "n": len(task.y),
        "n_groups": int(groups.nunique()),
        "score": float(r2_score(y, p)),
        "secondary_score": "",
        "macro_f1": "",
        "mae": float(mean_absolute_error(y, p)),
        "nmae": float(mean(np.abs(p - y) / scale)),
        "worst_group_nmae_or_error": float(max(group_nmae)),
        "cvar20_group_nmae_or_error": float(cvar(group_nmae) or 0.0),
        "overprediction_rate": float(mean((p > y).astype(float))),
        "y_true": y,
        "y_pred": p,
        "groups": groups.to_numpy(),
        "probs": probs,
    }


def score_gap(random_summary: dict[str, object], grouped_summary: dict[str, object]) -> float:
    return float(random_summary["score"]) - float(grouped_summary["score"])


def pseudo_groups_like(groups: pd.Series, rng: random.Random) -> pd.Series:
    sizes = list(groups.value_counts().sort_values(ascending=False))
    indices = list(range(len(groups)))
    rng.shuffle(indices)
    out = [""] * len(groups)
    cursor = 0
    for gid, size in enumerate(sizes):
        for idx in indices[cursor : cursor + int(size)]:
            out[idx] = f"pseudo_{gid:03d}"
        cursor += int(size)
    return pd.Series(out, index=groups.index)


def topology_null_retraining(tasks: list[Task]) -> list[dict[str, object]]:
    rng = random.Random(RANDOM_STATE)
    rows: list[dict[str, object]] = []
    for task in tasks:
        if task.groups.nunique() < 2:
            continue
        random_summary = evaluate_oof(task, "random")
        true_grouped = evaluate_oof(task, "grouped")
        true_gap = score_gap(random_summary, true_grouped)
        pseudo_gaps: list[float] = []
        for _ in range(PSEUDO_REPS):
            pseudo = pseudo_groups_like(task.groups, rng)
            pseudo_grouped = evaluate_oof(task, "grouped", groups=pseudo)
            pseudo_gaps.append(score_gap(random_summary, pseudo_grouped))
        pseudo_median = median(pseudo_gaps)
        pseudo_lo = percentile(pseudo_gaps, 0.025)
        pseudo_hi = percentile(pseudo_gaps, 0.975)
        rank = sum(1 for gap in pseudo_gaps if gap <= true_gap) / len(pseudo_gaps)
        if pseudo_hi is not None and true_gap > pseudo_hi:
            interpretation = "true_gap_exceeds_topology_retraining_null"
        elif pseudo_lo is not None and true_gap < pseudo_lo:
            interpretation = "true_gap_below_topology_retraining_null"
        else:
            interpretation = "compatible_with_topology_retraining_null"
        rows.append(
            {
                "module": task.module,
                "task": task.name,
                "metric_kind": task.metric_kind,
                "learner": "Random Forest",
                "true_random_score": fmt(float(random_summary["score"]), 6),
                "true_grouped_score": fmt(float(true_grouped["score"]), 6),
                "true_gap": fmt(true_gap, 6),
                "pseudo_gap_median": fmt(pseudo_median, 6),
                "pseudo_gap_95_low": fmt(pseudo_lo, 6),
                "pseudo_gap_95_high": fmt(pseudo_hi, 6),
                "true_gap_percentile": fmt(rank, 3),
                "excess_over_topology_null": fmt(true_gap - pseudo_median, 6),
                "pseudo_reps": len(pseudo_gaps),
                "interpretation": interpretation,
            }
        )
    return rows


def mitigation_baselines(tasks: list[Task]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        random_summary = evaluate_oof(task, "random")
        vanilla = evaluate_oof(task, "grouped", weighting="vanilla")
        sb = evaluate_oof(task, "grouped", weighting="source_balanced")
        glr = evaluate_oof(task, "grouped", weighting="group_loss_reweighted")
        for label, summary in (("vanilla_grouped", vanilla), ("source_balanced_erm", sb), ("group_loss_reweighted_erm", glr)):
            rows.append(
                {
                    "module": task.module,
                    "task": task.name,
                    "metric_kind": task.metric_kind,
                    "learner": "Random Forest",
                    "variant": label,
                    "random_score": fmt(float(random_summary["score"]), 6),
                    "grouped_score": fmt(float(summary["score"]), 6),
                    "gap_vs_random": fmt(float(random_summary["score"]) - float(summary["score"]), 6),
                    "nmae_or_blank": fmt(float(summary["nmae"]), 6) if summary["nmae"] != "" else "",
                    "worst_group_nmae_or_error": fmt(float(summary["worst_group_nmae_or_error"]), 6),
                    "cvar20_group_nmae_or_error": fmt(float(summary["cvar20_group_nmae_or_error"]), 6),
                    "overprediction_rate": fmt(float(summary["overprediction_rate"]), 6) if summary["overprediction_rate"] != "" else "",
                    "delta_score_vs_vanilla": fmt(float(summary["score"]) - float(vanilla["score"]), 6),
                    "delta_cvar20_vs_vanilla": fmt(float(summary["cvar20_group_nmae_or_error"]) - float(vanilla["cvar20_group_nmae_or_error"]), 6),
                }
            )
    return rows


def ece(y: np.ndarray, pred: np.ndarray, probs: list[dict[str, float]], bins: int = 10) -> tuple[float | None, float | None]:
    if not probs or not probs[0]:
        return None, None
    conf = np.array([max(p.values()) if p else 0.0 for p in probs], dtype=float)
    correct = (y.astype(str) == pred.astype(str)).astype(float)
    total = len(y)
    out = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (conf >= lo) & ((conf < hi) if b < bins - 1 else (conf <= hi))
        if mask.any():
            out += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return out, float(conf.mean())


def conformal_calibration(tasks: list[Task]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        random_summary = evaluate_oof(task, "random")
        grouped_summary = evaluate_oof(task, "grouped")
        if task.metric_kind == "regression":
            y_random = np.asarray(random_summary["y_true"], dtype=float)
            p_random = np.asarray(random_summary["y_pred"], dtype=float)
            y_grouped = np.asarray(grouped_summary["y_true"], dtype=float)
            p_grouped = np.asarray(grouped_summary["y_pred"], dtype=float)
            scale = scale_for(y_random)
            for nominal in (0.80, 0.90, 0.95):
                random_q = float(np.quantile(np.abs(y_random - p_random) / scale, nominal))
                grouped_q = float(np.quantile(np.abs(y_grouped - p_grouped) / scale, nominal))
                grouped_abs = np.abs(y_grouped - p_grouped) / scale
                group_cov = []
                groups = pd.Series(grouped_summary["groups"])
                for _, idx in groups.groupby(groups).groups.items():
                    idx_list = list(idx)
                    group_cov.append(float(np.mean(grouped_abs[idx_list] <= random_q)))
                rows.append(
                    {
                        "module": task.module,
                        "task": task.name,
                        "metric_kind": task.metric_kind,
                        "nominal": fmt(nominal, 2),
                        "random_calibrated_grouped_coverage": fmt(float(np.mean(grouped_abs <= random_q)), 6),
                        "grouped_calibrated_grouped_coverage": fmt(float(np.mean(grouped_abs <= grouped_q)), 6),
                        "width_inflation_grouped_over_random": fmt(grouped_q / random_q if random_q > EPS else None, 6),
                        "worst_group_coverage_random_width": fmt(min(group_cov), 6),
                        "undercovered_groups_random_width": sum(1 for cov in group_cov if cov < nominal),
                        "random_ece": "",
                        "grouped_ece": "",
                        "random_confidence": "",
                        "grouped_confidence": "",
                    }
                )
        else:
            y_random = np.asarray(random_summary["y_true"]).astype(str)
            p_random = np.asarray(random_summary["y_pred"]).astype(str)
            y_grouped = np.asarray(grouped_summary["y_true"]).astype(str)
            p_grouped = np.asarray(grouped_summary["y_pred"]).astype(str)
            random_ece, random_conf = ece(y_random, p_random, random_summary["probs"])  # type: ignore[arg-type]
            grouped_ece, grouped_conf = ece(y_grouped, p_grouped, grouped_summary["probs"])  # type: ignore[arg-type]
            rows.append(
                {
                    "module": task.module,
                    "task": task.name,
                    "metric_kind": task.metric_kind,
                    "nominal": "",
                    "random_calibrated_grouped_coverage": "",
                    "grouped_calibrated_grouped_coverage": "",
                    "width_inflation_grouped_over_random": "",
                    "worst_group_coverage_random_width": "",
                    "undercovered_groups_random_width": "",
                    "random_ece": fmt(random_ece, 6),
                    "grouped_ece": fmt(grouped_ece, 6),
                    "random_confidence": fmt(random_conf, 6),
                    "grouped_confidence": fmt(grouped_conf, 6),
                }
            )
    return rows


def physics_monotonicity(tasks: list[Task]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RANDOM_STATE)
    for task in tasks:
        if task.metric_kind != "regression" or not task.monotonic:
            continue
        model = build_pipeline(task.x, task.metric_kind, "Random Forest")
        model.fit(task.x, task.y)
        n = min(500, len(task.x))
        sample_idx = rng.choice(np.arange(len(task.x)), size=n, replace=False)
        base_x = task.x.iloc[sample_idx].copy()
        base_pred = model.predict(base_x).astype(float)
        for feature, direction in task.monotonic.items():
            if feature not in base_x.columns or not pd.api.types.is_numeric_dtype(base_x[feature]):
                continue
            perturbed = base_x.copy()
            values = pd.to_numeric(perturbed[feature], errors="coerce")
            delta = np.where(np.abs(values.to_numpy(dtype=float)) > EPS, 0.10 * values.to_numpy(dtype=float), 0.10)
            perturbed[feature] = values + delta
            pred = model.predict(perturbed).astype(float)
            diff = pred - base_pred
            if direction > 0:
                violations = diff < -1e-9
                magnitude = np.maximum(-diff, 0.0)
                expected = "nondecreasing"
            else:
                violations = diff > 1e-9
                magnitude = np.maximum(diff, 0.0)
                expected = "nonincreasing"
            rows.append(
                {
                    "module": task.module,
                    "task": task.name,
                    "feature": feature,
                    "expected_direction": expected,
                    "learner": "Random Forest full-data diagnostic",
                    "n_counterfactuals": n,
                    "violation_rate": fmt(float(np.mean(violations)), 6),
                    "mean_violation_magnitude": fmt(float(np.mean(magnitude)), 6),
                    "p90_violation_magnitude": fmt(float(np.quantile(magnitude, 0.90)), 6),
                }
            )
    return rows


def baseline_comparison(tasks: list[Task]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        if task.metric_kind != "regression":
            continue
        rf_grouped = evaluate_oof(task, "grouped")
        variants: list[tuple[str, pd.Series]] = []
        if task.empirical_constant is not None:
            variants.append(("empirical_constant", pd.Series(np.full(len(task.y), task.empirical_constant), index=task.y.index)))
        # Training-mean baseline under GroupKFold.
        pred_mean = pd.Series(index=task.y.index, dtype=float)
        for train_idx, test_idx in split_iter(task, "grouped"):
            pred_mean.iloc[test_idx] = float(task.y.iloc[train_idx].astype(float).mean())
        variants.append(("grouped_training_mean", pred_mean))
        if task.baseline_features:
            usable = [c for c in task.baseline_features if c in task.x.columns]
            if usable:
                simple = Task(
                    module=task.module,
                    name=task.name,
                    metric_kind="regression",
                    x=task.x[usable].copy(),
                    y=task.y.copy(),
                    groups=task.groups.copy(),
                    monotonic={},
                    baseline_features=[],
                )
                simple_summary = evaluate_oof(simple, "grouped", learner="Ridge")
                variants.append(("simplified_feature_ridge", pd.Series(simple_summary["y_pred"])))
        y = task.y.astype(float).to_numpy()
        scale = scale_for(y)
        for label, pred_series in variants:
            pred = pred_series.astype(float).to_numpy()
            rows.append(
                {
                    "module": task.module,
                    "task": task.name,
                    "baseline": label,
                    "baseline_grouped_r2": fmt(float(r2_score(y, pred)), 6),
                    "baseline_nmae": fmt(float(np.mean(np.abs(pred - y) / scale)), 6),
                    "baseline_overprediction_rate": fmt(float(np.mean(pred > y)), 6),
                    "rf_grouped_r2": fmt(float(rf_grouped["score"]), 6),
                    "rf_grouped_nmae": fmt(float(rf_grouped["nmae"]), 6),
                    "rf_minus_baseline_r2": fmt(float(rf_grouped["score"]) - float(r2_score(y, pred)), 6),
                }
            )
    return rows


def severity_sensitivity() -> list[dict[str, object]]:
    path = FIG_OUT / "cace_ninemodule_rf_summary.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    out: list[dict[str, object]] = []
    threshold_grid = [
        ("regression", threshold) for threshold in [0.30, 0.35, 0.40, 0.50]
    ] + [
        ("classification", threshold) for threshold in [0.20, 0.25, 0.30]
    ]
    for kind, threshold in threshold_grid:
        counts = Counter()
        labels: list[str] = []
        for row in rows:
            task = row.get("task", "")
            gap_text = row.get("gap") or row.get("accuracy_gap") or ""
            try:
                gap = float(gap_text)
            except ValueError:
                continue
            if kind == "classification" and not task.startswith("classification"):
                continue
            if kind == "regression" and task.startswith("classification"):
                continue
            if gap < 0:
                severity = "contrast"
            elif gap < 0.10:
                severity = "negligible"
            elif gap < threshold:
                severity = "moderate"
            else:
                severity = "severe"
            counts[severity] += 1
            labels.append(f"{row.get('label','')}: {severity}")
        out.append(
            {
                "metric_kind": kind,
                "severe_threshold": fmt(threshold, 2),
                "rows": len(labels),
                "negligible": counts["negligible"],
                "moderate": counts["moderate"],
                "severe": counts["severe"],
                "contrast": counts["contrast"],
            }
        )
    return out


def write_markdown_tables(
    topology_rows: list[dict[str, object]],
    mitigation_rows: list[dict[str, object]],
    conformal_rows: list[dict[str, object]],
    physics_rows: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
    severity_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Table S21. R05 topology-preserving retraining null experiment",
        "",
        f"Pseudo-groups preserve the empirical group-size distribution and trigger full GroupKFold retraining. Repetitions per task: {PSEUDO_REPS}.",
        "",
        "| Module | true gap | pseudo median | pseudo 95% interval | percentile | interpretation |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in topology_rows:
        lines.append(
            f"| {row['task']} | {row['true_gap']} | {row['pseudo_gap_median']} | {row['pseudo_gap_95_low']} to {row['pseudo_gap_95_high']} | {row['true_gap_percentile']} | {row['interpretation']} |"
        )
    write_text(TABLE_OUT / "r05_topology_null_retraining.md", "\n".join(lines) + "\n")

    lines = [
        "# Table S22. R05 retrained source-balanced mitigation baselines",
        "",
        "Random Forest baselines are retrained under vanilla GroupKFold, source-balanced ERM, and group-loss-reweighted ERM. These baselines test mitigation response rather than state-of-the-art performance.",
        "",
        "| Module | variant | grouped score | gap vs random | worst group | CVaR20 | delta score vs vanilla |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in mitigation_rows:
        lines.append(
            f"| {row['task']} | {row['variant']} | {row['grouped_score']} | {row['gap_vs_random']} | {row['worst_group_nmae_or_error']} | {row['cvar20_group_nmae_or_error']} | {row['delta_score_vs_vanilla']} |"
        )
    write_text(TABLE_OUT / "r05_source_balanced_mitigation.md", "\n".join(lines) + "\n")

    lines = [
        "# Table S23. R05 group-aware conformal/calibration coverage",
        "",
        "Regression rows compare random-calibrated absolute residual widths against grouped residual coverage; classification rows report probability ECE/confidence.",
        "",
        "| Module | nominal | random-width grouped coverage | grouped-width coverage | width inflation | worst group coverage | grouped ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in conformal_rows:
        lines.append(
            f"| {row['task']} | {row['nominal']} | {row['random_calibrated_grouped_coverage']} | {row['grouped_calibrated_grouped_coverage']} | {row['width_inflation_grouped_over_random']} | {row['worst_group_coverage_random_width']} | {row['grouped_ece']} |"
        )
    write_text(TABLE_OUT / "r05_group_conformal_calibration.md", "\n".join(lines) + "\n")

    lines = [
        "# Table S24. R05 physics-monotonicity counterfactual diagnostics",
        "",
        "Full-data Random Forest diagnostics perturb selected physically interpretable features by +10%. Directions are preliminary sanity checks, not design-code compatibility tests.",
        "",
        "| Module | feature | expected | violation rate | mean violation |",
        "|---|---|---|---:|---:|",
    ]
    for row in physics_rows:
        lines.append(
            f"| {row['task']} | {row['feature']} | {row['expected_direction']} | {row['violation_rate']} | {row['mean_violation_magnitude']} |"
        )
    write_text(TABLE_OUT / "r05_physics_monotonicity.md", "\n".join(lines) + "\n")

    lines = [
        "# Table S25. R05 empirical and simplified baseline comparison",
        "",
        "Baselines include training-mean predictors, simplified-feature ridge models, and an empirical constant for Stub-CFST strength index. They are context baselines, not design-code calibrations.",
        "",
        "| Module | baseline | baseline R2 | baseline nMAE | RF grouped R2 | RF minus baseline R2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in baseline_rows:
        lines.append(
            f"| {row['task']} | {row['baseline']} | {row['baseline_grouped_r2']} | {row['baseline_nmae']} | {row['rf_grouped_r2']} | {row['rf_minus_baseline_r2']} |"
        )
    write_text(TABLE_OUT / "r05_empirical_baselines.md", "\n".join(lines) + "\n")

    lines = [
        "# Table S26. R05 severity-threshold sensitivity",
        "",
        "Severity tags are readability labels. This table varies the severe threshold to check whether the broad tiering is fragile.",
        "",
        "| Metric kind | severe threshold | rows | contrast | negligible | moderate | severe |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in severity_rows:
        lines.append(
            f"| {row['metric_kind']} | {row['severe_threshold']} | {row['rows']} | {row['contrast']} | {row['negligible']} | {row['moderate']} | {row['severe']} |"
        )
    write_text(TABLE_OUT / "r05_severity_sensitivity.md", "\n".join(lines) + "\n")


def main() -> None:
    R05_OUT.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    print(f"[info] loaded {len(tasks)} R05 tasks")

    topology_rows = topology_null_retraining(tasks)
    mitigation_rows = mitigation_baselines(tasks)
    conformal_rows = conformal_calibration(tasks)
    physics_rows = physics_monotonicity(tasks)
    baseline_rows = baseline_comparison(tasks)
    severity_rows = severity_sensitivity()

    write_csv(
        R05_OUT / "topology_null_retraining.csv",
        topology_rows,
        [
            "module",
            "task",
            "metric_kind",
            "learner",
            "true_random_score",
            "true_grouped_score",
            "true_gap",
            "pseudo_gap_median",
            "pseudo_gap_95_low",
            "pseudo_gap_95_high",
            "true_gap_percentile",
            "excess_over_topology_null",
            "pseudo_reps",
            "interpretation",
        ],
    )
    write_csv(
        R05_OUT / "source_balanced_mitigation.csv",
        mitigation_rows,
        [
            "module",
            "task",
            "metric_kind",
            "learner",
            "variant",
            "random_score",
            "grouped_score",
            "gap_vs_random",
            "nmae_or_blank",
            "worst_group_nmae_or_error",
            "cvar20_group_nmae_or_error",
            "overprediction_rate",
            "delta_score_vs_vanilla",
            "delta_cvar20_vs_vanilla",
        ],
    )
    write_csv(
        R05_OUT / "group_conformal_calibration.csv",
        conformal_rows,
        [
            "module",
            "task",
            "metric_kind",
            "nominal",
            "random_calibrated_grouped_coverage",
            "grouped_calibrated_grouped_coverage",
            "width_inflation_grouped_over_random",
            "worst_group_coverage_random_width",
            "undercovered_groups_random_width",
            "random_ece",
            "grouped_ece",
            "random_confidence",
            "grouped_confidence",
        ],
    )
    write_csv(
        R05_OUT / "physics_monotonicity.csv",
        physics_rows,
        [
            "module",
            "task",
            "feature",
            "expected_direction",
            "learner",
            "n_counterfactuals",
            "violation_rate",
            "mean_violation_magnitude",
            "p90_violation_magnitude",
        ],
    )
    write_csv(
        R05_OUT / "empirical_baselines.csv",
        baseline_rows,
        [
            "module",
            "task",
            "baseline",
            "baseline_grouped_r2",
            "baseline_nmae",
            "baseline_overprediction_rate",
            "rf_grouped_r2",
            "rf_grouped_nmae",
            "rf_minus_baseline_r2",
        ],
    )
    write_csv(
        R05_OUT / "severity_sensitivity.csv",
        severity_rows,
        ["metric_kind", "severe_threshold", "rows", "negligible", "moderate", "severe", "contrast"],
    )
    write_markdown_tables(topology_rows, mitigation_rows, conformal_rows, physics_rows, baseline_rows, severity_rows)
    for path in sorted(R05_OUT.glob("*.csv")):
        print(f"[OK] {path}")
    for path in sorted(TABLE_OUT.glob("r05_*.md")):
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
