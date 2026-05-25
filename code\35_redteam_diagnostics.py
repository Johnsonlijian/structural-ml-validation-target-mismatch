"""Generate red-team diagnostic tables from existing reproduction outputs.

This script does not add new experiments. It consolidates two reviewer-risk
checks that can be run from the current archive:

1. A topology-mechanism diagnostic table for the Table S7 headline rows.
2. A multi-learner gap inventory from per-module results.csv files.

The outputs are descriptive, not causal; they are meant to prevent overclaiming
and to define the next analysis round.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "code" / "outputs" / "reproductions"
FIG_OUT = ROOT / "code" / "outputs" / "figures"
TABLE_OUT = ROOT / "manuscript" / "tables"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _f(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in ("", "nan", "None", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _clean_text(value: str) -> str:
    return (
        value.replace("—", "-")
        .replace("–", "-")
        .replace("鈥?", "-")
        .replace("R²", "R2")
    )


def topology_diagnostics() -> None:
    summary = FIG_OUT / "cace_ninemodule_rf_summary.csv"
    rows = _read_csv(summary)
    out: list[dict[str, object]] = []
    for row in rows:
        n = _f(row, "n_samples") or 0.0
        groups = _f(row, "n_groups") or 0.0
        gap = _f(row, "gap") or 0.0
        label = _clean_text(row["label"])
        samples_per_group = n / groups if groups else None
        flags: list[str] = []
        if groups and groups < 25:
            flags.append("few_groups")
        if samples_per_group is not None and samples_per_group < 5:
            flags.append("low_samples_per_group")
        if row["task"].startswith("classification"):
            flags.append("classification_topology")
        if "Stub-CFST" in label:
            flags.append("structural_family_holdout")
        if "PRJ-3053" in label:
            flags.append("singleton_sensitive_reference_groups")
        if not flags:
            flags.append("no_obvious_topology_flag_from_headline")
        out.append(
            {
                "label": label,
                "task": row["task"],
                "n_samples": int(n),
                "n_groups": int(groups),
                "samples_per_group": f"{samples_per_group:.2f}" if samples_per_group else "",
                "gap": f"{gap:.6f}",
                "abs_gap": f"{abs(gap):.6f}",
                "severity": row["severity"],
                "diagnostic_flags": ";".join(flags),
                "interpretation_boundary": "descriptive_only_not_causal",
            }
        )

    fields = [
        "label",
        "task",
        "n_samples",
        "n_groups",
        "samples_per_group",
        "gap",
        "abs_gap",
        "severity",
        "diagnostic_flags",
        "interpretation_boundary",
    ]
    _write_csv(FIG_OUT / "redteam_topology_mechanism_audit.csv", out, fields)

    severe = [r for r in out if str(r["severity"]) == "severe"]
    med_abs = median(float(r["abs_gap"]) for r in out)
    md = [
        "# Table S9. Red-team topology-mechanism diagnostic",
        "",
        "This table is a descriptive diagnostic over the Table S7 headline rows. It does not estimate a causal model. The purpose is to separate source dependence, structural-family extrapolation, small-group topology, and classification stress before drawing engineering conclusions.",
        "",
        f"- Headline rows: {len(out)}.",
        f"- Median absolute random-minus-grouped gap: {med_abs:.3f}.",
        f"- Severe rows: {len(severe)}.",
        "",
        "| Label | n | groups | n/group | gap | severity | diagnostic flags |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in out:
        md.append(
            f"| {_clean_text(str(r['label']))} | {r['n_samples']} | {r['n_groups']} | {r['samples_per_group']} | {r['gap']} | {r['severity']} | {r['diagnostic_flags']} |"
        )
    md.append("")
    md.append("Interpretation: large gaps should be read with topology diagnostics, not as a single leakage mechanism. Rows flagged as structural-family holdout or low samples per group require especially conservative engineering interpretation.")
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUT / "redteam_topology_mechanism_audit_v0.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _random_group_split(row: dict[str, str]) -> tuple[str | None, str | None]:
    keys = ("validation", "split", "cv")
    for key in keys:
        if key in row:
            value = row[key]
            if "Random" in value:
                return key, "random"
            if "GroupKFold" in value:
                return key, "grouped"
    return None, None


def _metric(row: dict[str, str]) -> tuple[str | None, float | None]:
    for name in ("pooled_r2", "accuracy", "accuracy_mean"):
        if name in row and row.get(name) not in ("", None):
            return name, _f(row, name)
    return None, None


def multi_learner_inventory() -> None:
    records: list[dict[str, object]] = []
    for path in sorted(REPRO.glob("*/results.csv")):
        rows = _read_csv(path)
        buckets: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
        for row in rows:
            split_key, split_kind = _random_group_split(row)
            if split_kind is None:
                continue
            metric_name, metric_value = _metric(row)
            if metric_name is None or metric_value is None:
                continue
            model = row.get("model", "")
            descriptor = "|".join(
                row.get(k, "")
                for k in ("case", "task", "target")
                if row.get(k, "")
            )
            if not descriptor:
                descriptor = path.parent.name
            descriptor = _clean_text(descriptor)
            key = (path.parent.name, descriptor, model, metric_name)
            buckets[key][split_kind] = row

        for (module, descriptor, model, metric_name), split_rows in buckets.items():
            if "random" not in split_rows or "grouped" not in split_rows:
                continue
            _, random_value = _metric(split_rows["random"])
            _, grouped_value = _metric(split_rows["grouped"])
            if random_value is None or grouped_value is None:
                continue
            records.append(
                {
                    "module": module,
                    "descriptor": descriptor,
                    "model": model,
                    "metric": metric_name,
                    "random_score": f"{random_value:.6f}",
                    "grouped_score": f"{grouped_value:.6f}",
                    "gap": f"{random_value - grouped_value:.6f}",
                }
            )

    fields = ["module", "descriptor", "model", "metric", "random_score", "grouped_score", "gap"]
    _write_csv(FIG_OUT / "redteam_multi_learner_gap_inventory.csv", records, fields)

    by_model: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_model[str(r["model"])].append(float(r["gap"]))

    md = [
        "# Table S10. Red-team multi-learner gap inventory",
        "",
        "This inventory consolidates random-versus-grouped gaps across the available per-module result files. It is not yet a harmonized robustness analysis because models and tasks differ by dataset; it defines the minimum evidence base for a future learner-robustness figure.",
        "",
        "| Model | rows | median gap | positive gaps | negative gaps |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, gaps in sorted(by_model.items()):
        positive = sum(1 for g in gaps if g > 0)
        negative = sum(1 for g in gaps if g < 0)
        md.append(f"| {model} | {len(gaps)} | {median(gaps):.3f} | {positive} | {negative} |")
    md.extend(
        [
            "",
            "Interpretation: the current manuscript's compact Table S7 uses Random Forest headline rows for readability. This inventory shows where additional learner-robustness analysis is already available and where harmonization is still required before making stronger CACE-level claims.",
        ]
    )
    TABLE_OUT.mkdir(parents=True, exist_ok=True)
    (TABLE_OUT / "redteam_multi_learner_gap_inventory_v0.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    topology_diagnostics()
    multi_learner_inventory()
    print(f"[OK] {FIG_OUT / 'redteam_topology_mechanism_audit.csv'}")
    print(f"[OK] {FIG_OUT / 'redteam_multi_learner_gap_inventory.csv'}")
    print(f"[OK] {TABLE_OUT / 'redteam_topology_mechanism_audit_v0.md'}")
    print(f"[OK] {TABLE_OUT / 'redteam_multi_learner_gap_inventory_v0.md'}")


if __name__ == "__main__":
    main()
