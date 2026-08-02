"""Fail-closed, standalone Attempt-T retention gate for Figure 4.

The gate never opens or compares an excluded attempt. It verifies the closed
T-only evidence lock, the retain-all receipt, every retained stage-manifest
binding, and the closure-bound S0/S1/S3 paired summaries. Only S0 and S3
hashes are returned to the plotting reader; S1 is checked explicitly to make
the no-favourable-selection boundary auditable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


LOCK_SCHEMA = "figure4-primary-evidence-lock-t-v1"
RETENTION_SCHEMA = "attempt-t-retention-audit-v1"
PROTOCOL_ID = "SAVP-CONFIRMATORY-V3.2"
CANONICAL_ATTEMPT = "T"
RETENTION_STATUS = "pass-all-prespecified-t-retained"
RETENTION_ROOT_MODE = "production-pinned-t-only"
ROOT_BINDING_FIELD = "canonical_attempt_T_logical_root"
RETENTION_RECEIPT_NAME = "ATTEMPT_T_RETENTION_AUDIT.json"
PRIMARY_STAGES = ("S0", "S1", "S3")
FIGURE_STAGES = ("S0", "S3")
ROW_KEY_FIELDS = ("scenario", "reference_method", "comparator")
SUMMARY_NAME = "paired_cluster_bootstrap.csv"
EXCLUDED_ATTEMPTS = tuple(chr(code) for code in range(ord("A"), ord("S") + 1))
CORE_SCENARIOS = tuple(
    f"CORE_{effect}_{covariate}_{noise}"
    for effect in ("0", "0.6", "1.2", "2.4")
    for covariate in ("0", "0.9", "1.8")
    for noise in ("0.35", "1.0")
)
STAGE_ORDER = (
    "COMPILER_SEMANTIC",
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
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AttemptTRetentionGateError(RuntimeError):
    """Raised before Figure 4 may read or draw Attempt-T results."""


@dataclass(frozen=True)
class Figure4EvidenceBinding:
    primary_hashes: Mapping[str, str]
    receipt: Mapping[str, Any]


def _fs(path: Path) -> str:
    return str(path.absolute())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AttemptTRetentionGateError(
            f"required evidence file is unreadable: {path.name}"
        ) from exc
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json_and_hash(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise AttemptTRetentionGateError(f"{label} is absent or not a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttemptTRetentionGateError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise AttemptTRetentionGateError(f"{label} must contain a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _require_hex(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise AttemptTRetentionGateError(f"{label} is not a SHA-256 digest")
    return value


def _stage_directory(stage: str) -> str:
    return "compiler_semantic" if stage == "COMPILER_SEMANTIC" else stage


def _root_mode(aggregate_root: Path) -> tuple[Path, str]:
    supplied = aggregate_root.absolute()
    if not supplied.is_dir() or supplied.is_symlink():
        raise AttemptTRetentionGateError("aggregate root is absent or is a symlink")
    root = supplied.resolve()
    lowered = tuple(part.lower() for part in root.parts)
    if any(part.startswith("excluded_") or "interrupted" in part for part in lowered):
        raise AttemptTRetentionGateError("excluded-attempt aggregates are inadmissible")
    if root.name == "a32":
        live_lock = root.parent / "pipeline_logs_v32" / ".pipeline.lock"
        if os.path.exists(_fs(live_lock)):
            raise AttemptTRetentionGateError(
                "pipeline lock is present; Attempt T remains sealed"
            )
        return root, "local-a32"
    if root.name == "confirmatory_aggregates_v32" and root.parent.name == "derived":
        return root, "public-projection"
    raise AttemptTRetentionGateError(
        "aggregate root must be local a32 or public derived/confirmatory_aggregates_v32"
    )


def _observed_manifest_hashes(root: Path, mode: str) -> dict[str, str]:
    observed: dict[str, str] = {}
    if mode == "local-a32":
        for stage in STAGE_ORDER:
            path = root / _stage_directory(stage) / "v32_aggregate_manifest.json"
            observed[stage] = _sha256(path)
        return observed

    receipt_root = root.parents[1] / "provenance" / "stages"
    if not receipt_root.is_dir() or receipt_root.is_symlink():
        raise AttemptTRetentionGateError("public stage-receipt directory is absent")
    for stage in STAGE_ORDER:
        directory = _stage_directory(stage)
        payload, _ = _read_json_and_hash(
            receipt_root / f"{directory}.json", label=f"{stage} stage receipt"
        )
        digest = payload.get("original_wrapper_sha256")
        if (
            payload.get("schema_version") != "path-neutral-stage-binding-v1"
            or payload.get("stage") != stage
            or payload.get("directory") != directory
            or not isinstance(digest, str)
            or not HEX64.fullmatch(digest)
        ):
            raise AttemptTRetentionGateError(f"{stage}: public stage receipt is invalid")
        observed[stage] = digest
    return observed


def _summary_profile(root: Path, stage: str) -> dict[str, Any]:
    path = root / stage / SUMMARY_NAME
    if not path.is_file() or path.is_symlink():
        raise AttemptTRetentionGateError(f"{stage}: paired summary is absent")
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = sum(1 for _ in reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise AttemptTRetentionGateError(f"{stage}: paired summary is unreadable") from exc
    if not set(ROW_KEY_FIELDS).issubset(columns) or rows <= 0:
        raise AttemptTRetentionGateError(f"{stage}: paired summary schema is incomplete")
    return {
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": columns,
    }


def validate_figure4_evidence(
    *,
    primary_evidence_lock: Path,
    retention_receipt: Path,
    aggregate_root: Path,
    expected_freeze_manifest_sha256: str,
) -> Figure4EvidenceBinding:
    """Validate the T lock, retain-all receipt, and S0/S1/S3 summaries."""

    root, mode = _root_mode(aggregate_root)
    lock, lock_hash = _read_json_and_hash(
        primary_evidence_lock.absolute(), label="primary evidence lock"
    )
    expected_lock = {
        "schema_version": LOCK_SCHEMA,
        "status": "closed",
        "protocol_id": PROTOCOL_ID,
        "freeze_manifest_sha256": expected_freeze_manifest_sha256,
        "full_pipeline_stage_count": len(STAGE_ORDER),
        "selection_gate": "t-only-retention",
        "retention_status": RETENTION_STATUS,
        "canonical_attempt": CANONICAL_ATTEMPT,
    }
    for field, expected in expected_lock.items():
        if lock.get(field) != expected:
            raise AttemptTRetentionGateError(
                f"primary evidence lock is not release-ready for {field}"
            )
    pipeline_hash = _require_hex(
        lock.get("pipeline_completion_sha256"), label="pipeline completion binding"
    )
    closure_hash = _require_hex(
        lock.get("pipeline_closure_receipt_sha256"), label="pipeline closure binding"
    )
    closure_evidence_hash = _require_hex(
        lock.get("pipeline_closure_evidence_set_sha256"),
        label="pipeline closure evidence-set binding",
    )
    expected_retention_hash = _require_hex(
        lock.get("retention_receipt_sha256"), label="Attempt-T retention binding"
    )
    lock_stages = lock.get("stages")
    if not isinstance(lock_stages, dict) or set(lock_stages) != set(FIGURE_STAGES):
        raise AttemptTRetentionGateError(
            "primary evidence lock must bind exactly S0 and S3"
        )

    retention, observed_retention_hash = _read_json_and_hash(
        retention_receipt.absolute(), label="Attempt-T retention receipt"
    )
    if observed_retention_hash != expected_retention_hash:
        raise AttemptTRetentionGateError(
            "Attempt-T retention receipt does not match the primary evidence lock"
        )
    expected_retention = {
        "schema_version": RETENTION_SCHEMA,
        "status": "PASS",
        "canonical_attempt": CANONICAL_ATTEMPT,
        "retention_status": RETENTION_STATUS,
        "selection_rule": "retain-all-prespecified-attempt-t",
        "protocol_id": PROTOCOL_ID,
        "excluded_attempts": list(EXCLUDED_ATTEMPTS),
        "pipeline_completion_sha256": pipeline_hash,
        "pipeline_closure_receipt_sha256": closure_hash,
        "pipeline_closure_evidence_set_sha256": closure_evidence_hash,
        "retained_stage_count": len(STAGE_ORDER),
        "retained_stage_order_sha256": _canonical_sha256(list(STAGE_ORDER)),
    }
    for field, expected in expected_retention.items():
        if retention.get(field) != expected:
            raise AttemptTRetentionGateError(f"Attempt-T retention mismatch: {field}")

    binding = retention.get("root_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("mode") != RETENTION_ROOT_MODE
        or binding.get("production_admissible") is not True
        or binding.get(ROOT_BINDING_FIELD) != "a32"
        or binding.get("receipt_logical_path") != RETENTION_RECEIPT_NAME
    ):
        raise AttemptTRetentionGateError("Attempt-T retention root binding is invalid")
    safety = retention.get("safety")
    if (
        not isinstance(safety, Mapping)
        or safety.get("all_prespecified_t_outputs_retained") is not True
        or safety.get("excluded_attempt_compared") is not False
        or safety.get("excluded_attempt_opened") is not False
        or safety.get("value_dependent_filtering_performed") is not False
        or safety.get("scientific_interpretation_performed") is not False
        or safety.get("selection_candidate_set") != ["T"]
        or safety.get("outcome_based_attempt_choice_possible") is not False
        or safety.get("non_overwriting") is not True
    ):
        raise AttemptTRetentionGateError("Attempt-T retention safety boundary is invalid")

    retained_manifests = retention.get("all_stage_manifest_sha256")
    if (
        not isinstance(retained_manifests, Mapping)
        or set(retained_manifests) != set(STAGE_ORDER)
        or any(not HEX64.fullmatch(str(value)) for value in retained_manifests.values())
    ):
        raise AttemptTRetentionGateError("Attempt-T retained manifest set is incomplete")
    if dict(retained_manifests) != _observed_manifest_hashes(root, mode):
        raise AttemptTRetentionGateError("Attempt-T retained manifest binding drifted")

    primary = retention.get("primary_completeness_stages")
    if not isinstance(primary, Mapping) or set(primary) != set(PRIMARY_STAGES):
        raise AttemptTRetentionGateError("Attempt-T S0/S1/S3 registry is incomplete")
    primary_hashes: dict[str, str] = {}
    stage_receipt: dict[str, dict[str, Any]] = {}
    for stage in PRIMARY_STAGES:
        observed = _summary_profile(root, stage)
        row = primary.get(stage)
        if (
            not isinstance(row, Mapping)
            or row.get("logical_path")
            != f"canonical_attempt_T/{stage}/{SUMMARY_NAME}"
            or row.get("closure_path") != f"a32/{stage}/{SUMMARY_NAME}"
            or row.get("raw_sha256") != observed["sha256"]
            or row.get("bytes") != observed["bytes"]
            or row.get("rows") != observed["rows"]
            or row.get("columns") != observed["columns"]
            or row.get("row_key_fields") != list(ROW_KEY_FIELDS)
            or row.get("scientific_values_interpreted") is not False
        ):
            raise AttemptTRetentionGateError(
                f"{stage}: Attempt-T retained summary binding mismatch"
            )
        stage_receipt[stage] = {
            "paired_summary_sha256": observed["sha256"],
            "rows": observed["rows"],
        }
        if stage in FIGURE_STAGES:
            lock_row = lock_stages.get(stage)
            if not isinstance(lock_row, Mapping) or lock_row.get(
                "paired_summary_sha256"
            ) != observed["sha256"]:
                raise AttemptTRetentionGateError(
                    f"primary evidence lock {stage} summary hash drifted"
                )
            primary_hashes[stage] = observed["sha256"]

    return Figure4EvidenceBinding(
        primary_hashes=primary_hashes,
        receipt={
            "evidence_lock_sha256": lock_hash,
            "canonical_attempt": CANONICAL_ATTEMPT,
            "selection_gate": "t-only-retention",
            "retention_status": RETENTION_STATUS,
            "pipeline_completion_sha256": pipeline_hash,
            "pipeline_closure_receipt_sha256": closure_hash,
            "pipeline_closure_evidence_set_sha256": closure_evidence_hash,
            "retention_receipt_sha256": observed_retention_hash,
            "retention_stages": stage_receipt,
        },
    )
