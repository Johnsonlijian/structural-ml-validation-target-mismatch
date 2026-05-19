"""R02 hardening diagnostics for the structural ML validation manuscript.

The script only uses existing reproduction outputs. It upgrades the external
red-team response from qualitative judgement to auditable diagnostics:

1. topology meta-diagnostic with group-size stress features;
2. normalized multi-learner robustness summary;
3. paired group-bootstrap uncertainty for prediction-level exports;
4. engineering-decision proxy metrics for regression targets.

The outputs are still diagnostic, not proof of causality or design-code safety.
"""

from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "code" / "outputs" / "reproductions"
FIG_OUT = ROOT / "code" / "outputs" / "figures"
TABLE_OUT = ROOT / "manuscript" / "tables"
BOOTSTRAP_REPS = 1000
RANDOM_SEED = 20260513


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


def f(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def clean_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00b2": "2",
        "鈥?": "-",
        "閳?": "-",
        "R虏": "R2",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return " ".join(out.split())


def model_name(name: str) -> str:
    value = clean_text(name).replace("_", " ").strip().lower()
    if value in {"randomforest", "random forest", "rf"}:
        return "Random Forest"
    if value in {"gradientboosting", "gradient boosting", "gbdt"}:
        return "Gradient Boosting"
    if value in {"ridge", "ridge regression"}:
        return "Ridge"
    if value in {"logisticregression", "logistic regression"}:
        return "Logistic Regression"
    return clean_text(name).strip()


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def ranks(values: list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and order[j][1] == order[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k][0]] = rank
        i = j
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys))


def r2_score(y_true: list[float], y_pred: list[float]) -> float | None:
    if len(y_true) < 2 or len(y_true) != len(y_pred):
        return None
    ybar = mean(y_true)
    ss_tot = sum((y - ybar) ** 2 for y in y_true)
    if ss_tot <= 0:
        return None
    ss_res = sum((y - p) ** 2 for y, p in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot


def mae(y_true: list[float], y_pred: list[float]) -> float | None:
    if not y_true or len(y_true) != len(y_pred):
        return None
    return sum(abs(y - p) for y, p in zip(y_true, y_pred)) / len(y_true)


def accuracy(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true or len(y_true) != len(y_pred):
        return None
    return sum(1 for y, p in zip(y_true, y_pred) if str(y) == str(p)) / len(y_true)


def balanced_accuracy(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true or len(y_true) != len(y_pred):
        return None
    labels = sorted(set(y_true))
    recalls: list[float] = []
    for label in labels:
        positives = sum(1 for y in y_true if y == label)
        if positives:
            recalls.append(sum(1 for y, p in zip(y_true, y_pred) if y == label and p == label) / positives)
    return mean(recalls) if recalls else None


def macro_f1(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true or len(y_true) != len(y_pred):
        return None
    labels = sorted(set(y_true) | set(y_pred))
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == label and p == label)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != label and p == label)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == label and p != label)
        denom = (2 * tp) + fp + fn
        scores.append((2 * tp) / denom if denom else 0.0)
    return mean(scores) if scores else None


def group_stats(path: Path, task_filter: str | None = None, case_filter: str | None = None) -> dict[str, float]:
    rows = read_csv(path)
    values: list[int] = []
    for row in rows:
        if task_filter and row.get("task") != task_filter:
            continue
        if case_filter and row.get("case") != case_filter:
            continue
        value = f(row.get("n"))
        if value is not None:
            values.append(int(value))
    if not values:
        return {}
    total = sum(values)
    return {
        "groups_from_file": float(len(values)),
        "median_group_size": float(median(values)),
        "max_group_size": float(max(values)),
        "singleton_groups": float(sum(1 for value in values if value == 1)),
        "singleton_rate": float(sum(1 for value in values if value == 1)) / len(values),
        "top_group_share": float(max(values)) / total if total else 0.0,
    }


def stats_for_summary_label(label: str) -> dict[str, float]:
    label = clean_text(label)
    if "Mangalathu" in label:
        return group_stats(REPRO / "10-1016-j-engstruct-2019-110331" / "author_group_sizes.csv")
    if "SFRC" in label:
        return group_stats(REPRO / "10-1016-j-engstruct-2020-111743" / "source_group_sizes.csv")
    if "UCI concrete" in label:
        return group_stats(REPRO / "10-1016-j-conbuildmat-2020-120950" / "mix_group_sizes.csv")
    if "Corroded RC" in label:
        return group_stats(REPRO / "10-5281-zenodo-8062007" / "source_group_sizes.csv")
    if "Stub-CFST" in label:
        return group_stats(REPRO / "10-1038-s41598-024-53352-1" / "shape_group_sizes.csv")
    if "Exterior joint shear" in label:
        return group_stats(REPRO / "mendeley_beam_column_joint" / "group_sizes.csv", task_filter="shear_strength_regression")
    if "Exterior joint failure" in label:
        return group_stats(REPRO / "mendeley_beam_column_joint" / "group_sizes.csv", task_filter="failure_mode_classification")
    if "Cyclic joint shear" in label:
        return group_stats(REPRO / "mendeley_beam_column_joint" / "group_sizes.csv", task_filter="joint_shear_strength_regression")
    if "DesignSafe RC columns" in label:
        return group_stats(REPRO / "designsafe_rc_columns" / "group_sizes.csv")
    if "PRJ-2430" in label:
        return group_stats(REPRO / "designsafe_prj2430_wall" / "group_sizes.csv")
    if "PRJ-3053" in label:
        return group_stats(REPRO / "designsafe_prj3053_coupling_beams" / "group_sizes.csv")
    return {}


def topology_family(group_variable: str, task: str) -> str:
    gv = group_variable.lower()
    if "section family" in gv:
        return "structural-family extrapolation"
    if "mixture" in gv:
        return "mixture-family constraint"
    if "experimental programme" in gv:
        return "experimental-programme grouping"
    if "reference" in gv or "author" in gv or "source" in gv:
        return "source/reference grouping"
    if task.startswith("classification"):
        return "classification grouping"
    return "other deployment grouping"


def topology_meta_diagnostic() -> None:
    rows = read_csv(FIG_OUT / "cace_ninemodule_rf_summary.csv")
    out: list[dict[str, object]] = []
    for row in rows:
        label = clean_text(row.get("label", ""))
        n = f(row.get("n_samples")) or 0.0
        groups = f(row.get("n_groups")) or 0.0
        gap = f(row.get("gap")) or 0.0
        stats = stats_for_summary_label(label)
        median_group = stats.get("median_group_size")
        singleton_rate = stats.get("singleton_rate")
        top_share = stats.get("top_group_share")
        flags: list[str] = []
        if groups < 25:
            flags.append("few_groups")
        if median_group is not None and median_group <= 3:
            flags.append("low_median_group_size")
        if singleton_rate is not None and singleton_rate >= 0.25:
            flags.append("high_singleton_rate")
        if top_share is not None and top_share >= 0.20:
            flags.append("concentrated_group")
        if "section" in row.get("group_variable", "").lower():
            flags.append("structural_family_holdout")
        if row.get("task", "").startswith("classification"):
            flags.append("classification")
        if not flags:
            flags.append("no_large_topology_flag")
        out.append(
            {
                "label": label,
                "topology_family": topology_family(row.get("group_variable", ""), row.get("task", "")),
                "task": clean_text(row.get("task", "")),
                "n_samples": int(n),
                "n_groups": int(groups),
                "samples_per_group": fmt(n / groups if groups else None, 2),
                "median_group_size": fmt(median_group, 2),
                "singleton_rate": fmt(singleton_rate, 3),
                "top_group_share": fmt(top_share, 3),
                "gap": fmt(gap, 6),
                "abs_gap": fmt(abs(gap), 6),
                "severity": row.get("severity", ""),
                "topology_flags": ";".join(flags),
            }
        )

    fields = [
        "label",
        "topology_family",
        "task",
        "n_samples",
        "n_groups",
        "samples_per_group",
        "median_group_size",
        "singleton_rate",
        "top_group_share",
        "gap",
        "abs_gap",
        "severity",
        "topology_flags",
    ]
    write_csv(FIG_OUT / "r02_topology_meta_diagnostic.csv", out, fields)

    abs_gap = [float(row["abs_gap"]) for row in out if row["abs_gap"]]
    n_groups = [float(row["n_groups"]) for row in out if row["abs_gap"]]
    median_groups = [float(row["median_group_size"]) for row in out if row["median_group_size"] and row["abs_gap"]]
    median_group_gaps = [float(row["abs_gap"]) for row in out if row["median_group_size"] and row["abs_gap"]]
    singleton_rates = [float(row["singleton_rate"]) for row in out if row["singleton_rate"] and row["abs_gap"]]
    singleton_gaps = [float(row["abs_gap"]) for row in out if row["singleton_rate"] and row["abs_gap"]]

    by_flag: dict[str, list[float]] = defaultdict(list)
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in out:
        gap = float(row["abs_gap"])
        by_family[str(row["topology_family"])].append(gap)
        for flag in str(row["topology_flags"]).split(";"):
            by_flag[flag].append(gap)

    lines = [
        "# Table S11. R02 topology meta-diagnostic",
        "",
        "This diagnostic adds group-size stress features to the Table S7 headline rows. It is deliberately framed as a small-sample meta-diagnostic rather than a causal model.",
        "",
        f"- Headline rows: {len(out)}.",
        f"- Median absolute gap: {median(abs_gap):.3f}.",
        f"- Spearman(abs gap, n_groups): {fmt(spearman(n_groups, abs_gap), 2)}.",
        f"- Spearman(abs gap, median group size): {fmt(spearman(median_groups, median_group_gaps), 2)}.",
        f"- Spearman(abs gap, singleton rate): {fmt(spearman(singleton_rates, singleton_gaps), 2)}.",
        "",
        "Family-level median absolute gaps:",
        "",
        "| Grouping family | rows | median abs gap |",
        "|---|---:|---:|",
    ]
    for family, gaps in sorted(by_family.items()):
        lines.append(f"| {family} | {len(gaps)} | {median(gaps):.3f} |")
    lines.extend(["", "Topology-flag median absolute gaps:", "", "| Flag | rows | median abs gap |", "|---|---:|---:|"])
    for flag, gaps in sorted(by_flag.items()):
        lines.append(f"| {flag} | {len(gaps)} | {median(gaps):.3f} |")
    lines.extend(
        [
            "",
            "Interpretation: severe gaps are not a single mechanism. The section-family holdout is an extrapolation stress test; low group sizes and singleton-heavy rows require conservative uncertainty language; source/reference rows remain the main evidence for deployment-target mismatch.",
        ]
    )
    write_text(TABLE_OUT / "r02_topology_meta_diagnostic.md", "\n".join(lines) + "\n")


def split_kind(row: dict[str, str]) -> str | None:
    value = row.get("validation", row.get("split", row.get("cv", "")))
    if "Random" in value:
        return "random"
    if "GroupKFold" in value:
        return "grouped"
    return None


def metric_value(row: dict[str, str]) -> tuple[str | None, float | None]:
    for key in ("pooled_r2", "accuracy", "accuracy_mean"):
        value = f(row.get(key))
        if value is not None:
            return key, value
    return None, None


def descriptor_for_result(module: str, row: dict[str, str]) -> str:
    parts = [row.get(key, "") for key in ("case", "task", "target") if row.get(key, "")]
    if not parts:
        return module
    return clean_text("|".join(parts))


def collect_gap_inventory() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(REPRO.glob("*/results.csv")):
        module = path.parent.name
        rows = read_csv(path)
        buckets: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
        for row in rows:
            kind = split_kind(row)
            if kind is None:
                continue
            metric, value = metric_value(row)
            if metric is None or value is None:
                continue
            descriptor = descriptor_for_result(module, row)
            key = (module, descriptor, model_name(row.get("model", "")), metric)
            buckets[key][kind] = value
        for (module, descriptor, model, metric), values in buckets.items():
            if "random" in values and "grouped" in values:
                records.append(
                    {
                        "module": module,
                        "descriptor": descriptor,
                        "model": model,
                        "metric": metric,
                        "random_score": values["random"],
                        "grouped_score": values["grouped"],
                        "gap": values["random"] - values["grouped"],
                    }
                )
    return records


def multi_learner_robustness() -> None:
    records = collect_gap_inventory()
    by_descriptor: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for row in records:
        key = (str(row["module"]), str(row["descriptor"]), str(row["metric"]))
        by_descriptor[key][str(row["model"])] = float(row["gap"])

    out: list[dict[str, object]] = []
    for (module, descriptor, metric), model_gaps in sorted(by_descriptor.items()):
        if len(model_gaps) < 2:
            continue
        gaps = list(model_gaps.values())
        positive = sum(1 for gap in gaps if gap > 0.05)
        negative = sum(1 for gap in gaps if gap < -0.05)
        near_zero = sum(1 for gap in gaps if abs(gap) <= 0.05)
        if all(abs(gap) < 0.10 for gap in gaps):
            classification = "low_or_negligible_across_models"
        elif positive == len(gaps) and median(gaps) >= 0.30:
            classification = "robust_large_positive_gap"
        elif positive == len(gaps):
            classification = "robust_positive_gap"
        elif positive and negative:
            classification = "learner_sensitive_mixed_sign"
        else:
            classification = "learner_sensitive_magnitude"
        rf_gap = model_gaps.get("Random Forest")
        out.append(
            {
                "module": module,
                "descriptor": descriptor,
                "metric": metric,
                "n_models": len(model_gaps),
                "models": ";".join(sorted(model_gaps)),
                "median_gap": fmt(median(gaps), 6),
                "min_gap": fmt(min(gaps), 6),
                "max_gap": fmt(max(gaps), 6),
                "positive_models": positive,
                "negative_models": negative,
                "near_zero_models": near_zero,
                "random_forest_gap": fmt(rf_gap, 6),
                "robustness_class": classification,
            }
        )

    fields = [
        "module",
        "descriptor",
        "metric",
        "n_models",
        "models",
        "median_gap",
        "min_gap",
        "max_gap",
        "positive_models",
        "negative_models",
        "near_zero_models",
        "random_forest_gap",
        "robustness_class",
    ]
    write_csv(FIG_OUT / "r02_multi_learner_robustness.csv", out, fields)

    class_counts: dict[str, int] = defaultdict(int)
    for row in out:
        class_counts[str(row["robustness_class"])] += 1
    lines = [
        "# Table S12. R02 multi-learner robustness summary",
        "",
        "Model names are normalized across reproduction modules. Rows summarize descriptors with at least two learners and paired random/grouped results.",
        "",
        f"- Descriptor rows with at least two learners: {len(out)}.",
        f"- Rows classified as robust large positive gaps: {class_counts.get('robust_large_positive_gap', 0)}.",
        f"- Rows classified as low/negligible across models: {class_counts.get('low_or_negligible_across_models', 0)}.",
        "",
        "| Descriptor | models | median gap | range | class |",
        "|---|---:|---:|---|---|",
    ]
    for row in out:
        lines.append(
            f"| {row['descriptor']} | {row['n_models']} | {row['median_gap']} | {row['min_gap']} to {row['max_gap']} | {row['robustness_class']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: Random Forest is no longer the sole robustness story. Several large gaps persist across learners, while the coupling-beam peak-shear row remains learner-sensitive and should be retained as a contrast rather than as a leakage example.",
        ]
    )
    write_text(TABLE_OUT / "r02_multi_learner_robustness.md", "\n".join(lines) + "\n")


def prediction_descriptor(module: str, row: dict[str, str]) -> str:
    parts = [row.get(key, "") for key in ("case", "task", "target") if row.get(key, "")]
    return clean_text("|".join(parts)) if parts else module


def prediction_columns(row: dict[str, str]) -> tuple[str, str, str, str]:
    y_col = "observed" if "observed" in row else "y_true"
    p_col = "predicted" if "predicted" in row else "y_pred"
    id_col = "sample_index" if "sample_index" in row else "row_index"
    group_col = "source_group" if "source_group" in row else "mix_group" if "mix_group" in row else "shape" if "shape" in row else "group"
    return y_col, p_col, id_col, group_col


def collect_prediction_pairs() -> list[dict[str, object]]:
    paired_sets: list[dict[str, object]] = []
    for path in sorted(REPRO.glob("*/predictions.csv")):
        module = path.parent.name
        rows = read_csv(path)
        buckets: dict[tuple[str, str], dict[str, dict[str, dict[str, str]]]] = defaultdict(lambda: defaultdict(dict))
        group_labels: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        for row in rows:
            kind = split_kind(row)
            if kind is None:
                continue
            y_col, p_col, id_col, group_col = prediction_columns(row)
            descriptor = prediction_descriptor(module, row)
            key = (descriptor, model_name(row.get("model", "")))
            sample_id = row.get(id_col, "")
            if sample_id == "":
                continue
            buckets[key][kind][sample_id] = row
            if kind == "grouped":
                group_labels[key][sample_id] = row.get(group_col, "")
        for (descriptor, model), split_rows in buckets.items():
            if "random" not in split_rows or "grouped" not in split_rows:
                continue
            common = sorted(set(split_rows["random"]) & set(split_rows["grouped"]))
            if len(common) < 10:
                continue
            random_rows = [split_rows["random"][sample] for sample in common]
            grouped_rows = [split_rows["grouped"][sample] for sample in common]
            y_col, p_col, _, _ = prediction_columns(grouped_rows[0])
            has_class_probabilities = any(key.startswith("prob_class_") for key in grouped_rows[0])
            numeric = (
                not has_class_probabilities
                and all(f(row.get(y_col)) is not None and f(row.get(p_col)) is not None for row in random_rows + grouped_rows)
            )
            paired_sets.append(
                {
                    "module": module,
                    "descriptor": descriptor,
                    "model": model,
                    "metric_type": "regression" if numeric else "classification",
                    "sample_ids": common,
                    "random_rows": random_rows,
                    "grouped_rows": grouped_rows,
                    "group_labels": [group_labels[(descriptor, model)].get(sample, "ungrouped") for sample in common],
                    "y_col": y_col,
                    "p_col": p_col,
                }
            )
    return paired_sets


def score_for_indices(
    data: dict[str, object], indices: list[int], metric_name: str
) -> tuple[float | None, float | None, float | None]:
    random_rows = data["random_rows"]  # type: ignore[index]
    grouped_rows = data["grouped_rows"]  # type: ignore[index]
    y_col = str(data["y_col"])
    p_col = str(data["p_col"])
    if metric_name == "pooled_r2":
        y = [f(random_rows[i].get(y_col)) for i in indices]  # type: ignore[index]
        pr = [f(random_rows[i].get(p_col)) for i in indices]  # type: ignore[index]
        pg = [f(grouped_rows[i].get(p_col)) for i in indices]  # type: ignore[index]
        if any(value is None for value in y + pr + pg):
            return None, None, None
        yy = [float(value) for value in y if value is not None]
        rr = [float(value) for value in pr if value is not None]
        gg = [float(value) for value in pg if value is not None]
        random_score = r2_score(yy, rr)
        grouped_score = r2_score(yy, gg)
    else:
        y = [str(random_rows[i].get(y_col)) for i in indices]  # type: ignore[index]
        pr = [str(random_rows[i].get(p_col)) for i in indices]  # type: ignore[index]
        pg = [str(grouped_rows[i].get(p_col)) for i in indices]  # type: ignore[index]
        if metric_name == "accuracy":
            random_score = accuracy(y, pr)
            grouped_score = accuracy(y, pg)
        elif metric_name == "balanced_accuracy":
            random_score = balanced_accuracy(y, pr)
            grouped_score = balanced_accuracy(y, pg)
        elif metric_name == "macro_f1":
            random_score = macro_f1(y, pr)
            grouped_score = macro_f1(y, pg)
        else:
            raise ValueError(metric_name)
    if random_score is None or grouped_score is None:
        return None, None, None
    return random_score, grouped_score, random_score - grouped_score


def bootstrap_uncertainty_and_engineering() -> None:
    rng = random.Random(RANDOM_SEED)
    paired_sets = collect_prediction_pairs()
    uncertainty_rows: list[dict[str, object]] = []
    engineering_rows: list[dict[str, object]] = []
    for data in paired_sets:
        n = len(data["sample_ids"])  # type: ignore[arg-type]
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, group in enumerate(data["group_labels"]):  # type: ignore[union-attr]
            groups[str(group)].append(idx)
        if len(groups) < 2:
            continue
        all_indices = list(range(n))
        group_keys = list(groups)
        metric_names = ["pooled_r2"] if data["metric_type"] == "regression" else ["accuracy", "balanced_accuracy", "macro_f1"]
        for metric_name in metric_names:
            random_score, grouped_score, gap = score_for_indices(data, all_indices, metric_name)
            if gap is None:
                continue
            boot_gaps: list[float] = []
            for _ in range(BOOTSTRAP_REPS):
                sampled: list[int] = []
                for group in (rng.choice(group_keys) for _ in group_keys):
                    sampled.extend(groups[group])
                _, _, boot_gap = score_for_indices(data, sampled, metric_name)
                if boot_gap is not None and math.isfinite(boot_gap):
                    boot_gaps.append(boot_gap)
            lo = percentile(boot_gaps, 0.025)
            hi = percentile(boot_gaps, 0.975)
            uncertainty_rows.append(
                {
                    "module": data["module"],
                    "descriptor": data["descriptor"],
                    "model": data["model"],
                    "metric_type": data["metric_type"],
                    "metric": metric_name,
                    "n_pairs": n,
                    "n_bootstrap_groups": len(groups),
                    "random_score": fmt(random_score, 6),
                    "grouped_score": fmt(grouped_score, 6),
                    "gap": fmt(gap, 6),
                    "gap_ci_low": fmt(lo, 6),
                    "gap_ci_high": fmt(hi, 6),
                    "ci_excludes_zero": bool(lo is not None and hi is not None and (lo > 0 or hi < 0)),
                    "bootstrap_reps_used": len(boot_gaps),
                }
            )

        if data["metric_type"] != "regression":
            continue
        random_rows = data["random_rows"]  # type: ignore[index]
        grouped_rows = data["grouped_rows"]  # type: ignore[index]
        y_col = str(data["y_col"])
        p_col = str(data["p_col"])
        y = [float(f(row.get(y_col)) or 0.0) for row in random_rows]
        pr = [float(f(row.get(p_col)) or 0.0) for row in random_rows]
        pg = [float(f(row.get(p_col)) or 0.0) for row in grouped_rows]
        scale = mean(abs(value) for value in y) or 1.0
        random_mae = mae(y, pr)
        grouped_mae = mae(y, pg)
        random_over = sum(1 for yy, pp in zip(y, pr) if pp > yy) / len(y)
        grouped_over = sum(1 for yy, pp in zip(y, pg) if pp > yy) / len(y)
        random_abs = [abs(yy - pp) / scale for yy, pp in zip(y, pr)]
        grouped_abs = [abs(yy - pp) / scale for yy, pp in zip(y, pg)]
        random_pos = [max(pp - yy, 0.0) / scale for yy, pp in zip(y, pr)]
        grouped_pos = [max(pp - yy, 0.0) / scale for yy, pp in zip(y, pg)]
        engineering_rows.append(
            {
                "module": data["module"],
                "descriptor": data["descriptor"],
                "model": data["model"],
                "n_pairs": n,
                "scale_mean_abs_y": fmt(scale, 6),
                "random_nmae": fmt((random_mae or 0.0) / scale, 6),
                "grouped_nmae": fmt((grouped_mae or 0.0) / scale, 6),
                "delta_nmae_grouped_minus_random": fmt(((grouped_mae or 0.0) - (random_mae or 0.0)) / scale, 6),
                "random_overprediction_rate": fmt(random_over, 6),
                "grouped_overprediction_rate": fmt(grouped_over, 6),
                "delta_overprediction_rate_grouped_minus_random": fmt(grouped_over - random_over, 6),
                "random_p90_abs_error_norm": fmt(percentile(random_abs, 0.90), 6),
                "grouped_p90_abs_error_norm": fmt(percentile(grouped_abs, 0.90), 6),
                "random_mean_positive_error_norm": fmt(mean(random_pos), 6),
                "grouped_mean_positive_error_norm": fmt(mean(grouped_pos), 6),
            }
        )

    uncertainty_fields = [
        "module",
        "descriptor",
        "model",
        "metric_type",
        "metric",
        "n_pairs",
        "n_bootstrap_groups",
        "random_score",
        "grouped_score",
        "gap",
        "gap_ci_low",
        "gap_ci_high",
        "ci_excludes_zero",
        "bootstrap_reps_used",
    ]
    write_csv(FIG_OUT / "r02_paired_group_bootstrap_uncertainty.csv", uncertainty_rows, uncertainty_fields)

    engineering_fields = [
        "module",
        "descriptor",
        "model",
        "n_pairs",
        "scale_mean_abs_y",
        "random_nmae",
        "grouped_nmae",
        "delta_nmae_grouped_minus_random",
        "random_overprediction_rate",
        "grouped_overprediction_rate",
        "delta_overprediction_rate_grouped_minus_random",
        "random_p90_abs_error_norm",
        "grouped_p90_abs_error_norm",
        "random_mean_positive_error_norm",
        "grouped_mean_positive_error_norm",
    ]
    write_csv(FIG_OUT / "r02_engineering_decision_metrics.csv", engineering_rows, engineering_fields)

    severe_uncertain = [
        row
        for row in uncertainty_rows
        if row["model"] == "Random Forest" and row["ci_excludes_zero"] in {True, "True"}
    ]
    lines = [
        "# Table S13. R02 paired group-bootstrap uncertainty",
        "",
        "Prediction-level exports are re-analysed with one paired group-bootstrap routine. For each descriptor/model, random and grouped predictions are paired on the same sample identifiers, and bootstrap resampling is performed over grouped-validation groups.",
        "",
        f"- Prediction-level descriptor/model rows analysed: {len(uncertainty_rows)}.",
        f"- Rows with 95% bootstrap interval excluding zero: {sum(1 for row in uncertainty_rows if row['ci_excludes_zero'] in {True, 'True'})}.",
        f"- Random Forest rows with intervals excluding zero: {len(severe_uncertain)}.",
        "",
        "| Descriptor | model | metric | gap | 95% CI | groups |",
        "|---|---|---|---:|---|---:|",
    ]
    for row in uncertainty_rows:
        if row["model"] == "Random Forest" or row["metric_type"] == "classification":
            lines.append(
                f"| {row['descriptor']} | {row['model']} | {row['metric']} | {row['gap']} | {row['gap_ci_low']} to {row['gap_ci_high']} | {row['n_bootstrap_groups']} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: this table replaces mixed uncertainty language where prediction-level exports exist. Classification rows report accuracy, balanced accuracy, and macro-F1; regression rows report pooled R2.",
        ]
    )
    write_text(TABLE_OUT / "r02_paired_group_bootstrap_uncertainty.md", "\n".join(lines) + "\n")

    lines = [
        "# Table S14. R02 engineering-decision proxy metrics",
        "",
        "These are target-scale diagnostic proxies, not design-code safety factors. For capacity-like regression targets, overprediction is reported as an unsafe-side proxy because it can imply non-conservative capacity estimates.",
        "",
        f"- Regression descriptor/model rows analysed: {len(engineering_rows)}.",
        "",
        "| Descriptor | model | delta nMAE | random overprediction | grouped overprediction | grouped p90 norm abs error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in engineering_rows:
        if row["model"] == "Random Forest":
            lines.append(
                f"| {row['descriptor']} | {row['model']} | {row['delta_nmae_grouped_minus_random']} | {row['random_overprediction_rate']} | {row['grouped_overprediction_rate']} | {row['grouped_p90_abs_error_norm']} |"
            )
    lines.extend(
        [
            "",
            "Interpretation: grouped/deployment-matched evaluation usually increases normalized error and changes unsafe-side overprediction rates, but this still does not replace code-based resistance-factor, reliability, or design-check analyses.",
        ]
    )
    write_text(TABLE_OUT / "r02_engineering_decision_metrics.md", "\n".join(lines) + "\n")


def main() -> None:
    topology_meta_diagnostic()
    multi_learner_robustness()
    bootstrap_uncertainty_and_engineering()
    print(f"[OK] {FIG_OUT / 'r02_topology_meta_diagnostic.csv'}")
    print(f"[OK] {FIG_OUT / 'r02_multi_learner_robustness.csv'}")
    print(f"[OK] {FIG_OUT / 'r02_paired_group_bootstrap_uncertainty.csv'}")
    print(f"[OK] {FIG_OUT / 'r02_engineering_decision_metrics.csv'}")
    print(f"[OK] {TABLE_OUT / 'r02_topology_meta_diagnostic.md'}")
    print(f"[OK] {TABLE_OUT / 'r02_multi_learner_robustness.md'}")
    print(f"[OK] {TABLE_OUT / 'r02_paired_group_bootstrap_uncertainty.md'}")
    print(f"[OK] {TABLE_OUT / 'r02_engineering_decision_metrics.md'}")


if __name__ == "__main__":
    main()
