"""Strict v3.2 compiler preflight followed by the frozen base aggregation."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from protocol_v32 import (
    PROTOCOL_ID,
    ProtocolPaths,
    aggregate_temporary_path,
    atomic_create_json,
    runtime_preflight,
    reject_aggregate_temporary_pollution,
    sha256_bytes,
    validate_stage_complete,
)


def main() -> None:
    paths = ProtocolPaths.canonical()
    context = runtime_preflight(paths)
    reject_aggregate_temporary_pollution(paths)
    scenario = "COMPILER_SEMANTIC"
    records = validate_stage_complete(context, scenario)
    if len(records) != 100:
        raise RuntimeError("v3.2 compiler aggregation requires exactly 100 records")

    final = paths.aggregate_root / "compiler_semantic"
    temporary = aggregate_temporary_path(paths)
    if final.exists() or temporary.exists():
        raise RuntimeError("v3.2 compiler aggregate is immutable or has a stale temp")
    paths.aggregate_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(paths.cv_root / "aggregate_compiler_suite.py"),
        "--input",
        str(paths.scenario_root(scenario)),
        "--output",
        str(temporary),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"frozen compiler aggregation failed: {completed.stderr[-2000:]}"
        )
    gate = json.loads((temporary / "compiler_gate.json").read_text(encoding="utf-8"))
    required_gate = {
        "n_complete": 100,
        "n_failed": 0,
        "false_exact_events_total": 0,
        "all_primary_observations_pass": True,
    }
    for field, expected in required_gate.items():
        if gate.get(field) != expected:
            raise RuntimeError(f"v3.2 compiler hard gate failed: {field}")
    with (temporary / "compiler_conformance_counts.csv").open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    expected_metrics = {
        "zero_false_exact",
        "native_exact_realization",
        "plan_reproducibility",
        "assignment_reproducibility",
        "outcome_invariance",
        "typed_refusal_accuracy",
        "independent_oracle_agreement",
        "mutation_detection",
        "production_mutation_detection",
        "oracle_production_mutation_agreement",
        "bruteforce_status_agreement",
    }
    if {row.get("metric") for row in rows} != expected_metrics or any(
        int(row["successes"]) != 100
        or int(row["total"]) != 100
        or float(row["rate"]) != 1.0
        for row in rows
    ):
        raise RuntimeError("v3.2 compiler metric counts are not 100/100")
    registry_hash = sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    manifest = {
        "aggregate_schema": "v32-compiler-aggregate-v1",
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": context.manifest_sha256,
        "seed_table_sha256": context.seed_table_sha256,
        "scenario": scenario,
        "n_claims": 100,
        "n_results": 100,
        "input_record_registry_sha256": registry_hash,
        "input_records": records,
        "base_command": command,
        "base_stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "hard_gate": gate,
    }
    atomic_create_json(temporary / "v32_aggregate_manifest.json", manifest)
    reject_aggregate_temporary_pollution(paths, allowed=temporary)
    temporary.rename(final)
    print(json.dumps(required_gate, sort_keys=True))


if __name__ == "__main__":
    main()
