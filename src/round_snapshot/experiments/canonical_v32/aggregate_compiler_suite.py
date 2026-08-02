"""Aggregate primary compiler-suite rates with exact binomial intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for path in sorted(source.glob("rep_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        status_rows.append(
            {
                "file": path.name,
                "replicate": payload.get("replicate"),
                "status": payload.get("status"),
                "error": payload.get("error"),
            }
        )
        if payload.get("status") == "complete":
            payloads.append(payload)
    pd.DataFrame(status_rows).to_csv(output / "replicate_status.csv", index=False)

    metrics = {
        "zero_false_exact": [item["false_exact_events"] == 0 for item in payloads],
        "native_exact_realization": [item["native_realization_ok"] for item in payloads],
        "plan_reproducibility": [item["plan_reproducible"] for item in payloads],
        "assignment_reproducibility": [item["assignment_reproducible"] for item in payloads],
        "outcome_invariance": [item["outcome_invariant"] for item in payloads],
        "typed_refusal_accuracy": [item["all_refusals_correct"] for item in payloads],
        "independent_oracle_agreement": [
            item["independent_oracle_agreement"] for item in payloads
        ],
        "mutation_detection": [item["mutation_detection"] for item in payloads],
        "production_mutation_detection": [
            item["production_mutation_detection"] for item in payloads
        ],
        "oracle_production_mutation_agreement": [
            item["oracle_production_mutation_agreement"] for item in payloads
        ],
        "bruteforce_status_agreement": [
            item["bruteforce_status_agreement"] for item in payloads
        ],
    }
    rows = []
    for metric, values in metrics.items():
        total = len(values)
        successes = int(sum(bool(value) for value in values))
        rows.append(
            {
                "metric": metric,
                "successes": successes,
                "total": total,
                "rate": successes / total if total else float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "compiler_conformance_counts.csv", index=False)
    hard_gate = {
        "n_complete": len(payloads),
        "n_failed": sum(row["status"] != "complete" for row in status_rows),
        "false_exact_events_total": int(
            sum(item["false_exact_events"] for item in payloads)
        ),
        "all_primary_observations_pass": bool(
            payloads and all(all(values) for values in metrics.values())
        ),
    }
    (output / "compiler_gate.json").write_text(
        json.dumps(hard_gate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(json.dumps(hard_gate, indent=2))


if __name__ == "__main__":
    main()
