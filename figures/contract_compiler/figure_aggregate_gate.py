"""Fail-closed readers for submission figures backed by final v3.2 aggregates.

The figure builders intentionally do not inspect ``confirmatory_v32`` records.
They first validate every required ``a32`` aggregate manifest and only then
open aggregate CSV files.  A missing stage, hash mismatch, non-100 denominator,
or record-registry drift is a hard refusal and must occur before output paths
are created.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_FREEZE_MANIFEST_SHA256 = (
    "800bc0560955b8c2d86492599b1d5c02a4655d0d0f2f76feebaf74b681282464"
)
EXPECTED_SEED_TABLE_SHA256 = (
    "bcc4ec1c43f780bb64e417fce81f21df52db1202db8e7fdbe79104295c18d4ad"
)
EXPECTED_PROTOCOL_ID = "SAVP-CONFIRMATORY-V3.2"
COMPILER_SCHEMA = "v32-compiler-aggregate-v1"
PREDICTIVE_SCHEMA = "v32-predictive-aggregate-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")

CORE_SCENARIOS = tuple(
    f"CORE_{effect}_{covariate}_{noise}"
    for effect in ("0", "0.6", "1.2", "2.4")
    for covariate in ("0", "0.9", "1.8")
    for noise in ("0.35", "1.0")
)

PREDICTIVE_STAGE_ORDER = (
    "S8",
    "S10",
    "S4",
    "S0",
    "S3",
    "S1",
    "S2",
    "S5",
    "S6",
    "S7",
    "S9_missing10",
    "S9_missing30",
    "S9_corrupt5",
    *CORE_SCENARIOS,
)


class AggregateGateError(RuntimeError):
    """Raised when a final aggregate cannot support a figure."""


@dataclass(frozen=True)
class AggregateStage:
    scenario: str
    directory: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_record_registry_hash(records: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(
        json.dumps(
            list(records), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def _directory_name(scenario: str) -> str:
    return "compiler_semantic" if scenario == "COMPILER_SEMANTIC" else scenario


def _validate_root(aggregate_root: Path) -> Path:
    root = aggregate_root.resolve()
    if root.name != "confirmatory_aggregates_v32" or root.parent.name != "derived":
        raise AggregateGateError(
            "aggregate root must be the public derived/confirmatory_aggregates_v32 directory"
        )
    forbidden = {"confirmatory_v32", "freeze_v32", "confirmatory", "freeze"}
    lowered_parts = tuple(part.lower() for part in root.parts)
    if any(part in forbidden for part in lowered_parts) or any(
        part.startswith("excluded_") or "interrupted" in part
        for part in lowered_parts
    ):
        raise AggregateGateError(
            "figure readers may not be routed through record, freeze, or excluded-attempt roots"
        )
    if not root.is_dir():
        raise AggregateGateError(f"final a32 aggregate root is absent: {root}")
    if root.is_symlink():
        raise AggregateGateError("final a32 aggregate root may not be a symlink")
    polluted = sorted(
        path.name for path in root.iterdir() if path.name.startswith(".__")
    )
    if polluted:
        raise AggregateGateError(
            f"a32 contains a temporary aggregate artifact: {polluted[0]}"
        )
    return root


def _read_manifest(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise AggregateGateError(f"required aggregate manifest is absent: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AggregateGateError(f"aggregate manifest is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise AggregateGateError(f"aggregate manifest is not an object: {path}")
    return payload


def _validate_input_records(
    manifest: Mapping[str, Any], scenario: str
) -> None:
    records = manifest.get("input_records")
    if not isinstance(records, list) or len(records) != 100:
        raise AggregateGateError(
            f"{scenario}: aggregate must bind exactly 100 input records"
        )
    try:
        replicates = [int(row["replicate"]) for row in records]
    except Exception as exc:
        raise AggregateGateError(
            f"{scenario}: aggregate record registry has no canonical replicate ids"
        ) from exc
    if replicates != list(range(100)):
        raise AggregateGateError(
            f"{scenario}: aggregate record registry is missing, reordered, or duplicated"
        )
    for row in records:
        for field in ("claim_sha256", "result_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or not HEX64.fullmatch(value):
                raise AggregateGateError(
                    f"{scenario}: invalid {field} in aggregate record registry"
                )
    observed = _canonical_record_registry_hash(records)
    if manifest.get("input_record_registry_sha256") != observed:
        raise AggregateGateError(
            f"{scenario}: aggregate input-record registry hash drifted"
        )


def validate_stage_manifests(
    aggregate_root: Path,
    scenarios: Iterable[str],
    *,
    expected_schema: str,
) -> dict[str, AggregateStage]:
    """Validate all requested manifests before any caller reads an aggregate CSV."""

    root = _validate_root(aggregate_root)
    requested = tuple(scenarios)
    if not requested or len(set(requested)) != len(requested):
        raise AggregateGateError("required aggregate stage list is empty or duplicated")

    receipt_root = root.parents[1] / "provenance" / "stages"
    if not receipt_root.is_dir() or receipt_root.is_symlink():
        raise AggregateGateError("public stage-receipt directory is absent or unsafe")

    # First collect every path-neutral receipt. This makes incomplete stage families fail
    # before the builder can inspect a single aggregate CSV.
    raw: dict[str, tuple[Path, Path, Mapping[str, Any]]] = {}
    for scenario in requested:
        directory = root / _directory_name(scenario)
        manifest_path = receipt_root / f"{_directory_name(scenario)}.json"
        raw[scenario] = (directory, manifest_path, _read_manifest(manifest_path))

    seed_hash: str | None = None
    validated: dict[str, AggregateStage] = {}
    for scenario in requested:
        directory, manifest_path, manifest = raw[scenario]
        expected = {
            "schema_version": "path-neutral-stage-binding-v1",
            "aggregate_schema": expected_schema,
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "freeze_manifest_sha256": EXPECTED_FREEZE_MANIFEST_SHA256,
            "seed_table_sha256": EXPECTED_SEED_TABLE_SHA256,
            "stage": scenario,
            "directory": _directory_name(scenario),
        }
        for field, value in expected.items():
            if manifest.get(field) != value:
                raise AggregateGateError(
                    f"{scenario}: aggregate manifest mismatch for {field}"
                )
        current_seed_hash = manifest.get("seed_table_sha256")
        if not isinstance(current_seed_hash, str) or not HEX64.fullmatch(
            current_seed_hash
        ):
            raise AggregateGateError(f"{scenario}: invalid seed-table hash")
        if seed_hash is None:
            seed_hash = current_seed_hash
        elif current_seed_hash != seed_hash:
            raise AggregateGateError(
                f"{scenario}: seed-table hash differs across final aggregates"
            )
        _validate_input_records(manifest, scenario)
        summary_files = manifest.get("summary_files")
        if not isinstance(summary_files, list) or not summary_files:
            raise AggregateGateError(f"{scenario}: public summary registry is absent")
        logical_paths: set[str] = set()
        for row in summary_files:
            logical = row.get("path") if isinstance(row, dict) else None
            digest = row.get("sha256") if isinstance(row, dict) else None
            size = row.get("bytes") if isinstance(row, dict) else None
            if (
                not isinstance(logical, str)
                or not isinstance(digest, str)
                or not HEX64.fullmatch(digest)
                or not isinstance(size, int)
                or size < 0
                or logical in logical_paths
            ):
                raise AggregateGateError(f"{scenario}: invalid public summary registry")
            logical_paths.add(logical)
        if expected_schema == PREDICTIVE_SCHEMA:
            base_hash = manifest.get("base_manifest_sha256")
            if not isinstance(base_hash, str) or not HEX64.fullmatch(base_hash):
                raise AggregateGateError(
                    f"{scenario}: predictive aggregate has no bound base manifest"
                )
        validated[scenario] = AggregateStage(
            scenario=scenario,
            directory=directory,
            manifest_path=manifest_path,
            manifest=manifest,
        )
    return validated


def read_stage_csv(
    stage: AggregateStage,
    filename: str,
    *,
    required_columns: Iterable[str],
    expected_rows: int | None = None,
) -> list[dict[str, str]]:
    """Read one CSV only after its containing stage passed the manifest gate."""

    if not filename.endswith(".csv") or Path(filename).name != filename:
        raise AggregateGateError(f"unsafe aggregate CSV name: {filename!r}")
    path = stage.directory / filename
    if not path.is_file() or path.is_symlink():
        raise AggregateGateError(
            f"{stage.scenario}: required aggregate CSV is absent: {filename}"
        )
    logical = (
        Path("derived")
        / "confirmatory_aggregates_v32"
        / _directory_name(stage.scenario)
        / filename
    ).as_posix()
    rows_by_path = {
        str(row.get("path")): row
        for row in stage.manifest.get("summary_files", [])
        if isinstance(row, dict)
    }
    binding = rows_by_path.get(logical)
    if binding is None:
        raise AggregateGateError(
            f"{stage.scenario}: CSV is outside the public summary allowlist: {filename}"
        )
    observed = _sha256_bytes(path.read_bytes())
    if observed != binding.get("sha256") or path.stat().st_size != int(
        binding.get("bytes", -1)
    ):
        raise AggregateGateError(
            f"{stage.scenario}: public summary hash/size drifted: {filename}"
        )
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = set(required_columns).difference(fieldnames)
            if missing:
                raise AggregateGateError(
                    f"{stage.scenario}: {filename} misses columns {sorted(missing)}"
                )
            rows = list(reader)
    except AggregateGateError:
        raise
    except Exception as exc:
        raise AggregateGateError(
            f"{stage.scenario}: aggregate CSV is unreadable: {filename}"
        ) from exc
    if expected_rows is not None and len(rows) != expected_rows:
        raise AggregateGateError(
            f"{stage.scenario}: {filename} has {len(rows)} rows, expected {expected_rows}"
        )
    return rows


def read_stage_json(stage: AggregateStage, filename: str) -> Mapping[str, Any]:
    """Read one JSON summary only after its public receipt binding passes."""

    if not filename.endswith(".json") or Path(filename).name != filename:
        raise AggregateGateError(f"unsafe aggregate JSON name: {filename!r}")
    path = stage.directory / filename
    if not path.is_file() or path.is_symlink():
        raise AggregateGateError(
            f"{stage.scenario}: required aggregate JSON is absent: {filename}"
        )
    logical = (
        Path("derived")
        / "confirmatory_aggregates_v32"
        / _directory_name(stage.scenario)
        / filename
    ).as_posix()
    rows_by_path = {
        str(row.get("path")): row
        for row in stage.manifest.get("summary_files", [])
        if isinstance(row, dict)
    }
    binding = rows_by_path.get(logical)
    if binding is None:
        raise AggregateGateError(
            f"{stage.scenario}: JSON is outside the public summary allowlist: {filename}"
        )
    observed = _sha256_bytes(path.read_bytes())
    if observed != binding.get("sha256") or path.stat().st_size != int(
        binding.get("bytes", -1)
    ):
        raise AggregateGateError(
            f"{stage.scenario}: public summary hash/size drifted: {filename}"
        )
    value = _read_manifest(path)
    return value


def parse_finite_nonnegative(value: str, *, label: str) -> float:
    try:
        number = float(value)
    except Exception as exc:
        raise AggregateGateError(f"{label}: expected a numeric value") from exc
    if not math.isfinite(number) or number < 0:
        raise AggregateGateError(f"{label}: expected a finite nonnegative value")
    return number
