"""Strict v3.2 predictive preflight followed by frozen base aggregation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from protocol_v32 import (
    PROTOCOL_ID,
    ProtocolPaths,
    SCENARIOS,
    aggregate_temporary_path,
    atomic_create_json,
    runtime_preflight,
    reject_aggregate_temporary_pollution,
    sha256_bytes,
    sha256_file,
    validate_predecessors,
    validate_scenario_name,
    validate_stage_complete,
)


PREDICTIVE_SCENARIOS = tuple(sorted(SCENARIOS.difference({"COMPILER_SEMANTIC"})))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=PREDICTIVE_SCENARIOS)
    args = parser.parse_args()
    scenario = str(args.scenario)
    validate_scenario_name(scenario)
    paths = ProtocolPaths.canonical()
    context = runtime_preflight(paths)
    reject_aggregate_temporary_pollution(paths)
    validate_predecessors(context, scenario)
    records = validate_stage_complete(context, scenario)
    if len(records) != 100:
        raise RuntimeError("v3.2 predictive aggregation requires exactly 100 records")

    final = paths.aggregate_root / scenario
    temporary = aggregate_temporary_path(paths)
    if final.exists() or temporary.exists():
        raise RuntimeError("v3.2 predictive aggregate is immutable or has a stale temp")
    paths.aggregate_root.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(paths.cv_root / "aggregate_batch.py"),
        "--input",
        str(paths.scenario_root(scenario)),
        "--output",
        str(temporary),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"frozen predictive aggregation failed: {completed.stderr[-2000:]}"
        )
    base_manifest_path = temporary / "aggregation_manifest.json"
    if not base_manifest_path.is_file():
        raise RuntimeError("frozen predictive aggregation produced no manifest")
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if int(base_manifest.get("n_files", -1)) != 100:
        raise RuntimeError("frozen predictive aggregate did not read exactly 100 files")
    if int(base_manifest.get("n_complete", -1)) != 100 or int(
        base_manifest.get("n_failed", -1)
    ) != 0:
        raise RuntimeError("frozen predictive aggregate contains incomplete/failed files")
    registry_hash = sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    manifest = {
        "aggregate_schema": "v32-predictive-aggregate-v1",
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
        "base_manifest_sha256": sha256_file(base_manifest_path),
    }
    atomic_create_json(temporary / "v32_aggregate_manifest.json", manifest)
    reject_aggregate_temporary_pollution(paths, allowed=temporary)
    temporary.rename(final)
    print(json.dumps({"scenario": scenario, "n_results": 100}, sort_keys=True))


if __name__ == "__main__":
    main()
