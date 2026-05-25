"""R06 shift and source-proxy diagnostics for the SAVP manuscript.

R05 showed that true grouped gaps exceed topology-preserving pseudo-group
retraining nulls. R06 asks a different, reviewer-facing question: do grouped
folds also show measurable feature/target distribution shift, and how fragile
are grouped scores when the available source proxy is noisy?

The diagnostics are computer-only and use the same public tasks loaded by the
R05 script. They do not claim causal identification or field-wide prevalence.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import random
import sys
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
R06_OUT = CODE / "outputs" / "r06"
TABLE_OUT = ROOT / "manuscript" / "tables"
FIG_OUT = ROOT / "figures" / "generated"
RANDOM_STATE = 20260513
MAX_PAIRWISE = 350
PROXY_LEVELS = [0.0, 0.10, 0.25, 0.50, 0.75, 1.0]
PROXY_REPS = 5


def load_r05():
    path = CODE / "50_r05_retraining_mitigation_experiments.py"
    spec = importlib.util.spec_from_file_location("r06_r05", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r05 = load_r05()


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


def fmt(value: object, digits: int = 3) -> str:
    if value == "" or value is None:
        return ""
    try:
        x = float(value)
    except Exception:
        return str(value)
    if math.isnan(x):
        return ""
    return f"{x:.{digits}f}"


def dense(a):
    if hasattr(a, "toarray"):
        return a.toarray()
    return np.asarray(a)


def sample_rows(x: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= max_n:
        return x
    idx = rng.choice(len(x), size=max_n, replace=False)
    return x[idx]


def median_bandwidth(x: np.ndarray, y: np.ndarray) -> float:
    z = np.vstack([x, y])
    if len(z) > 500:
        rng = np.random.default_rng(RANDOM_STATE)
        z = sample_rows(z, 500, rng)
    d = pdist(z, metric="euclidean")
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return 1.0
    return float(np.median(d))


def rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    sigma = median_bandwidth(x, y)
    gamma = 1.0 / (2.0 * sigma * sigma + 1e-12)
    kxx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    kyy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    kxy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def energy_distance_nd(x: np.ndarray, y: np.ndarray) -> float:
    return float(2.0 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean())


def domain_auc(x: np.ndarray, y: np.ndarray) -> float:
    n0, n1 = len(x), len(y)
    if min(n0, n1) < 8:
        return float("nan")
    z = np.vstack([x, y])
    labels = np.array([0] * n0 + [1] * n1)
    n_splits = min(3, int(np.bincount(labels).min()))
    if n_splits < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    prob = cross_val_predict(clf, z, labels, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, prob))


def target_or_class_shift(task, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    if task.metric_kind == "regression":
        y = task.y.astype(float).to_numpy()
        return float(abs(y[test_idx].mean() - y[train_idx].mean()) / (r05.scale_for(y) + 1e-12))
    y = task.y.astype(str)
    classes = sorted(y.unique())
    train_counts = y.iloc[train_idx].value_counts(normalize=True).reindex(classes, fill_value=0.0)
    test_counts = y.iloc[test_idx].value_counts(normalize=True).reindex(classes, fill_value=0.0)
    return float(0.5 * np.abs(train_counts.to_numpy() - test_counts.to_numpy()).sum())


def fold_loss(task, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    model = r05.build_pipeline(task.x, task.metric_kind, "Random Forest")
    model.fit(task.x.iloc[train_idx], task.y.iloc[train_idx])
    pred = model.predict(task.x.iloc[test_idx])
    if task.metric_kind == "classification":
        return float(1.0 - accuracy_score(task.y.iloc[test_idx].astype(str), pd.Series(pred).astype(str)))
    y = task.y.iloc[test_idx].astype(float).to_numpy()
    return float(mean_absolute_error(y, pred.astype(float)) / (r05.scale_for(task.y.astype(float).to_numpy()) + 1e-12))


def fold_shift_rows(tasks) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(RANDOM_STATE)
    for task in tasks:
        for fold, (train_idx, test_idx) in enumerate(r05.split_iter(task, "grouped"), start=1):
            prep = clone(r05.build_pipeline(task.x, task.metric_kind, "Random Forest").named_steps["preprocess"])
            prep.fit(task.x.iloc[train_idx], task.y.iloc[train_idx])
            x_train = dense(prep.transform(task.x.iloc[train_idx])).astype(float)
            x_test = dense(prep.transform(task.x.iloc[test_idx])).astype(float)
            x_train_s = sample_rows(x_train, MAX_PAIRWISE, rng)
            x_test_s = sample_rows(x_test, MAX_PAIRWISE, rng)
            mmd = rbf_mmd2(x_train_s, x_test_s)
            energy = energy_distance_nd(x_train_s, x_test_s)
            auc = domain_auc(x_train_s, x_test_s)
            shift = target_or_class_shift(task, np.asarray(train_idx), np.asarray(test_idx))
            loss = fold_loss(task, np.asarray(train_idx), np.asarray(test_idx))
            rows.append(
                {
                    "module": task.module,
                    "task": task.name,
                    "metric_kind": task.metric_kind,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "heldout_groups": int(task.groups.iloc[test_idx].nunique()),
                    "mmd2": mmd,
                    "energy_distance": energy,
                    "domain_auc": auc,
                    "target_or_class_shift": shift,
                    "grouped_fold_loss": loss,
                }
            )
    return rows


def summarize_shift(rows: list[dict[str, object]], gap_lookup: dict[str, float]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    df = pd.DataFrame(rows)
    for (module, task), g in df.groupby(["module", "task"], sort=False):
        def med(col: str) -> float:
            return float(pd.to_numeric(g[col], errors="coerce").median())

        def rho(col: str) -> float:
            x = pd.to_numeric(g[col], errors="coerce")
            y = pd.to_numeric(g["grouped_fold_loss"], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                return float("nan")
            return float(spearmanr(x[mask], y[mask]).statistic)

        mmd_med = med("mmd2")
        auc_med = med("domain_auc")
        shift_label = "large" if (auc_med >= 0.80 or mmd_med >= 0.08) else "moderate" if (auc_med >= 0.65 or mmd_med >= 0.03) else "low"
        out.append(
            {
                "module": module,
                "task": task,
                "metric_kind": g["metric_kind"].iloc[0],
                "n_folds": len(g),
                "gap_from_r05": gap_lookup.get(task, float("nan")),
                "median_mmd2": mmd_med,
                "median_energy": med("energy_distance"),
                "median_domain_auc": auc_med,
                "median_target_or_class_shift": med("target_or_class_shift"),
                "median_grouped_fold_loss": med("grouped_fold_loss"),
                "spearman_mmd_loss": rho("mmd2"),
                "spearman_auc_loss": rho("domain_auc"),
                "shift_label": shift_label,
            }
        )
    return out


def r05_gap_lookup() -> dict[str, float]:
    path = CODE / "outputs" / "r05" / "topology_null_retraining.csv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["task"], df["true_gap"]))


def corrupt_groups(groups: pd.Series, fraction: float, rng: random.Random) -> pd.Series:
    labels = groups.astype(str).reset_index(drop=True).to_list()
    if fraction <= 0:
        return pd.Series(labels)
    n = len(labels)
    k = int(round(fraction * n))
    idx = list(range(n))
    chosen = rng.sample(idx, k=min(k, n))
    shuffled = [labels[i] for i in chosen]
    rng.shuffle(shuffled)
    out = labels[:]
    for i, new_label in zip(chosen, shuffled):
        out[i] = new_label
    return pd.Series(out)


def proxy_degradation_rows(tasks) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = random.Random(RANDOM_STATE)
    for task in tasks:
        random_summary = r05.evaluate_oof(task, "random", learner="Random Forest")
        true_grouped = r05.evaluate_oof(task, "grouped", learner="Random Forest")
        true_gap = r05.score_gap(random_summary, true_grouped)
        for fraction in PROXY_LEVELS:
            reps = 1 if fraction == 0 else PROXY_REPS
            for rep in range(reps):
                proxy_groups = corrupt_groups(task.groups, fraction, rng)
                grouped = r05.evaluate_oof(task, "grouped", learner="Random Forest", groups=proxy_groups)
                gap = r05.score_gap(random_summary, grouped)
                rows.append(
                    {
                        "module": task.module,
                        "task": task.name,
                        "metric_kind": task.metric_kind,
                        "proxy_corruption_fraction": fraction,
                        "replicate": rep + 1,
                        "random_score": random_summary["score"],
                        "grouped_score_under_proxy": grouped["score"],
                        "true_grouped_score": true_grouped["score"],
                        "gap_under_proxy": gap,
                        "true_gap": true_gap,
                        "gap_retention": gap / true_gap if abs(true_gap) > 1e-12 else float("nan"),
                        "proxy_n_groups": int(proxy_groups.nunique()),
                    }
                )
    return rows


def write_markdown_tables(summary_rows: list[dict[str, object]], proxy_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Table S27. R06 fold-level feature/target shift diagnostics",
        "",
        "Grouped folds are transformed using preprocessing fitted on the training fold only. MMD, energy distance, domain-classifier AUC, and target/class shift are diagnostics of train-versus-held-out distribution difference, not causal tests.",
        "",
        "| Module | Task | Gap | Median MMD2 | Domain AUC | Target/class shift | Fold loss | rho(MMD, loss) | Label |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['module']} | {r['task']} | {fmt(r['gap_from_r05'])} | {fmt(r['median_mmd2'])} | "
            f"{fmt(r['median_domain_auc'])} | {fmt(r['median_target_or_class_shift'])} | "
            f"{fmt(r['median_grouped_fold_loss'])} | {fmt(r['spearman_mmd_loss'])} | {r['shift_label']} |"
        )
    write_text(TABLE_OUT / "r06_shift_diagnostics.md", "\n".join(lines) + "\n")

    p = pd.DataFrame(proxy_rows)
    agg = (
        p.groupby(["module", "task", "metric_kind", "proxy_corruption_fraction"], sort=False)
        .agg(
            grouped_score_median=("grouped_score_under_proxy", "median"),
            gap_median=("gap_under_proxy", "median"),
            gap_retention_median=("gap_retention", "median"),
            gap_retention_min=("gap_retention", "min"),
            gap_retention_max=("gap_retention", "max"),
        )
        .reset_index()
    )
    lines = [
        "# Table S28. R06 source-proxy degradation sensitivity",
        "",
        "A fraction of specimen-level group labels is permuted before GroupKFold evaluation. Lower gap retention means that the proxy is no longer preserving the original deployment grouping, so grouped validation becomes less diagnostic of the original source/family claim.",
        "",
        "| Module | Task | Corruption | Grouped score | Gap | Gap retention | Range |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r['module']} | {r['task']} | {fmt(r['proxy_corruption_fraction'], 2)} | "
            f"{fmt(r['grouped_score_median'])} | {fmt(r['gap_median'])} | "
            f"{fmt(r['gap_retention_median'])} | {fmt(r['gap_retention_min'])}-{fmt(r['gap_retention_max'])} |"
        )
    write_text(TABLE_OUT / "r06_proxy_degradation.md", "\n".join(lines) + "\n")


def make_figures(summary_rows: list[dict[str, object]], proxy_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = pd.to_numeric(s["median_domain_auc"], errors="coerce")
    y = pd.to_numeric(s["gap_from_r05"], errors="coerce")
    ax.scatter(x, y, s=58, color="#365c8d", alpha=0.86)
    for _, r in s.iterrows():
        if pd.notna(r["median_domain_auc"]) and pd.notna(r["gap_from_r05"]):
            ax.annotate(str(r["module"]).replace("Mangalathu ", "").replace(" RC ", " "), (r["median_domain_auc"], r["gap_from_r05"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(0.5, color="0.70", lw=1, ls="--")
    ax.set_xlabel("Median domain-classifier AUC (train vs held-out fold)")
    ax.set_ylabel("R05 true random-minus-grouped gap")
    ax.set_title("R06: deployment gap versus measurable feature-shift signal")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG_OUT / f"fig_r06_shift_gap.{ext}", dpi=220)
    plt.close(fig)

    p = pd.DataFrame(proxy_rows)
    agg = p.groupby(["module", "proxy_corruption_fraction"], sort=False)["gap_retention"].median().reset_index()
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for module, g in agg.groupby("module", sort=False):
        ax.plot(g["proxy_corruption_fraction"], g["gap_retention"], marker="o", lw=1.4, label=module)
    ax.axhline(1.0, color="0.25", lw=1, ls=":")
    ax.axhline(0.0, color="0.70", lw=1)
    ax.set_xlabel("Fraction of group labels permuted")
    ax.set_ylabel("Median gap retention relative to original grouping")
    ax.set_title("R06: source-proxy degradation sensitivity")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(FIG_OUT / f"fig_r06_proxy_degradation.{ext}", dpi=220)
    plt.close(fig)


def main() -> None:
    tasks = r05.load_tasks()
    print(f"[info] loaded {len(tasks)} R06 tasks")
    gap_lookup = r05_gap_lookup()

    fold_rows = fold_shift_rows(tasks)
    summary_rows = summarize_shift(fold_rows, gap_lookup)
    proxy_rows = proxy_degradation_rows(tasks)

    write_csv(
        R06_OUT / "shift_diagnostics_folds.csv",
        fold_rows,
        [
            "module",
            "task",
            "metric_kind",
            "fold",
            "n_train",
            "n_test",
            "heldout_groups",
            "mmd2",
            "energy_distance",
            "domain_auc",
            "target_or_class_shift",
            "grouped_fold_loss",
        ],
    )
    write_csv(
        R06_OUT / "shift_diagnostics_summary.csv",
        summary_rows,
        [
            "module",
            "task",
            "metric_kind",
            "n_folds",
            "gap_from_r05",
            "median_mmd2",
            "median_energy",
            "median_domain_auc",
            "median_target_or_class_shift",
            "median_grouped_fold_loss",
            "spearman_mmd_loss",
            "spearman_auc_loss",
            "shift_label",
        ],
    )
    write_csv(
        R06_OUT / "proxy_degradation_sensitivity.csv",
        proxy_rows,
        [
            "module",
            "task",
            "metric_kind",
            "proxy_corruption_fraction",
            "replicate",
            "random_score",
            "grouped_score_under_proxy",
            "true_grouped_score",
            "gap_under_proxy",
            "true_gap",
            "gap_retention",
            "proxy_n_groups",
        ],
    )
    write_markdown_tables(summary_rows, proxy_rows)
    make_figures(summary_rows, proxy_rows)
    for path in [
        R06_OUT / "shift_diagnostics_folds.csv",
        R06_OUT / "shift_diagnostics_summary.csv",
        R06_OUT / "proxy_degradation_sensitivity.csv",
        TABLE_OUT / "r06_shift_diagnostics.md",
        TABLE_OUT / "r06_proxy_degradation.md",
        FIG_OUT / "fig_r06_shift_gap.svg",
        FIG_OUT / "fig_r06_proxy_degradation.svg",
    ]:
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()

