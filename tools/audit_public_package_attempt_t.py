#!/usr/bin/env python3
"""Standalone read-only audit for the Attempt-T public release."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


PROTOCOL_ID = "SAVP-CONFIRMATORY-V3.2"
FREEZE_SHA256 = "0b6e3498b425b5cddb725bdc7b8cf3477ce8461f3677c3283de667c1a4464760"
APPLICATION_SHA256 = "a86843f8518ef6effb7a97b47349ae9820bea7412f0fd77be0ec23a868611407"
CLOSURE_PATH = "provenance/ATTEMPT_T_PIPELINE_CLOSURE.json"
RETENTION_PATH = "provenance/ATTEMPT_T_RETENTION_AUDIT.json"
LOCK_PATH = "figures/contract_compiler/PRIMARY_EVIDENCE_LOCK.json"
PRIMARY_STAGES = ("S0", "S1", "S3")
ROW_KEYS = ("scenario", "reference_method", "comparator")
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
MAX_BYTES = 25 * 1024 * 1024
FORBIDDEN_PARTS = {
    "raw",
    "downloads",
    "archives",
    "manuscript",
    "manuscripts",
    "rounds",
    "logs",
    "pipeline_logs_v32",
    "confirmatory_v32",
    "a32",
    "cover_letter",
    "reviewer_response",
}
IGNORED_LOCAL_PARTS = {".git"}
PRIVATE_PATHS = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\|/)(?!/)[^\s'\"<>]+"),
    re.compile(r"(?<![\\A-Za-z0-9])\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9$_.-]+"),
    re.compile(r"(?<![A-Za-z0-9])/(?:home|Users|mnt|tmp)/[^\s'\"<>]+"),
)
SECRETS = (
    re.compile(r"(?i)(?:api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON: {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path.name}")
    return value


def resolve(root: Path, logical: str) -> Path:
    if not logical or "\\" in logical:
        raise RuntimeError(f"unsafe logical path: {logical!r}")
    pure = PurePosixPath(logical)
    if pure.is_absolute() or any(part in {".", ".."} for part in pure.parts):
        raise RuntimeError(f"unsafe logical path: {logical!r}")
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"path escapes package root: {logical}") from exc
    return path


def check_file(path: Path, record: dict[str, Any]) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256(path) != record.get("sha256")
        or path.stat().st_size != int(record.get("bytes", -1))
    ):
        raise RuntimeError(f"file binding mismatch: {path.name}")


def scan_public_tree(root: Path) -> int:
    count = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.lower() in IGNORED_LOCAL_PARTS for part in relative.parts):
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden: {relative}")
        if any(part.lower() in FORBIDDEN_PARTS for part in relative.parts):
            raise RuntimeError(f"private path component: {relative}")
        if not path.is_file():
            continue
        count += 1
        if path.stat().st_size > MAX_BYTES:
            raise RuntimeError(f"public file exceeds 25 MiB: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"non-UTF-8 public file: {relative}") from exc
        if any(pattern.search(text) for pattern in PRIVATE_PATHS):
            raise RuntimeError(f"private absolute path leaked: {relative}")
        if any(pattern.search(text) for pattern in SECRETS):
            raise RuntimeError(f"credential-like text detected: {relative}")
        if path.suffix.lower() == ".py":
            ast.parse(text, filename=str(relative))
    return count


def paired_summary(root: Path, stage: dict[str, Any]) -> dict[str, Any]:
    matches = [
        item
        for item in stage.get("summary_files", [])
        if PurePosixPath(str(item.get("path", ""))).name
        == "paired_cluster_bootstrap.csv"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"no unique paired summary: {stage.get('stage')}")
    record = matches[0]
    path = resolve(root, str(record.get("path", "")))
    check_file(path, record)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    return {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "columns": list(reader.fieldnames or []),
    }


def audit(root: Path) -> dict[str, Any]:
    required = (
        "figures/contract_compiler/attempt_t_retention_gate.py",
        "figures/contract_compiler/build_primary_scaling_figure.py",
        "tests/figures/test_contract_compiler_figures_attempt_t.py",
        "tools/audit_public_package_attempt_t.py",
    )
    for logical in required:
        if not resolve(root, logical).is_file():
            raise RuntimeError(f"Attempt-T scaffold is absent: {logical}")
    for logical in (
        "figures/contract_compiler/attempt_comparison_gate.py",
        "tests/figures/test_contract_compiler_figures.py",
        "tests/figures/test_contract_compiler_figures_attempt_s.py",
    ):
        if resolve(root, logical).exists():
            raise RuntimeError(f"excluded selection scaffold is present: {logical}")

    receipt = read_json(root / "provenance" / "RELEASE_RECEIPT.json")
    if (
        receipt.get("schema_version") != "public-release-receipt-v1"
        or receipt.get("release_status") != "public-authorized"
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("freeze_manifest_sha256") != FREEZE_SHA256
        or receipt.get("stage_order") != list(STAGE_ORDER)
    ):
        raise RuntimeError("public release receipt identity mismatch")
    stages = receipt.get("stages")
    if not isinstance(stages, list) or [row.get("stage") for row in stages] != list(STAGE_ORDER):
        raise RuntimeError("public release stage registry mismatch")
    stage_by_name = {str(row["stage"]): row for row in stages}
    for stage in stages:
        check_file(resolve(root, str(stage["projection"]["path"])), stage["projection"])
        for summary in stage.get("summary_files", []):
            check_file(resolve(root, str(summary["path"])), summary)

    closure_binding = receipt.get("pipeline_closure")
    closure_path = resolve(root, CLOSURE_PATH)
    if not isinstance(closure_binding, dict):
        raise RuntimeError("pipeline closure binding is absent")
    check_file(closure_path, closure_binding)
    closure = read_json(closure_path)
    order_hash = canonical_sha256(list(STAGE_ORDER))
    for field, value in {
        "schema_version": "attempt-t-pipeline-closure-v1",
        "status": "PASS",
        "canonical_attempt": "T",
        "protocol_id": PROTOCOL_ID,
        "stage_count": 38,
        "expected_stage_order": list(STAGE_ORDER),
        "expected_stage_order_sha256": order_hash,
    }.items():
        if closure.get(field) != value:
            raise RuntimeError(f"pipeline closure mismatch: {field}")
    pipeline = closure.get("pipeline_log")
    if not isinstance(pipeline, dict):
        raise RuntimeError("pipeline lifecycle binding is absent")
    for field, value in {
        "pipeline_start_count": 1,
        "pipeline_complete_count": 1,
        "stage_start_count": 38,
        "stage_complete_count": 38,
        "worker_exit_count": 76,
        "aggregator_exit_count": 38,
        "nonzero_exit_count": 0,
        "failure_event_count": 0,
    }.items():
        if pipeline.get(field) != value:
            raise RuntimeError(f"pipeline lifecycle mismatch: {field}")
    verification = closure.get("verification")
    finalizer = verification.get("finalizer") if isinstance(verification, dict) else None
    if (
        not isinstance(finalizer, dict)
        or finalizer.get("path")
        != "repro_package_AEI_public_draft/finalize_package_attempt_t.py"
        or finalizer.get("aggregate_verifier") != "verify_aggregates"
        or not HEX64.fullmatch(str(finalizer.get("sha256", "")))
        or verification.get("post_verification_stability") != "PASS"
        or verification.get("scientific_interpretation_performed") is not False
        or verification.get("confirmatory_root_opened") is not False
    ):
        raise RuntimeError("pipeline closure verification boundary is invalid")
    evidence_payload = {
        "expected_stage_order_sha256": order_hash,
        "pipeline_log": {key: pipeline[key] for key in ("path", "sha256", "bytes")},
        "finalizer": dict(finalizer),
        "stages": closure.get("stages"),
    }
    if closure.get("evidence_set_sha256") != canonical_sha256(evidence_payload):
        raise RuntimeError("pipeline closure evidence-set hash mismatch")

    lock = read_json(resolve(root, LOCK_PATH))
    for field, value in {
        "schema_version": "figure4-primary-evidence-lock-t-v1",
        "status": "closed",
        "protocol_id": PROTOCOL_ID,
        "freeze_manifest_sha256": FREEZE_SHA256,
        "full_pipeline_stage_count": 38,
        "selection_gate": "t-only-retention",
        "retention_status": "pass-all-prespecified-t-retained",
        "canonical_attempt": "T",
        "pipeline_completion_sha256": pipeline.get("sha256"),
        "pipeline_closure_receipt_sha256": sha256(closure_path),
        "pipeline_closure_evidence_set_sha256": closure.get("evidence_set_sha256"),
    }.items():
        if lock.get(field) != value:
            raise RuntimeError(f"primary-evidence lock mismatch: {field}")

    retention_binding = receipt.get("attempt_t_retention")
    retention_path = resolve(root, RETENTION_PATH)
    if not isinstance(retention_binding, dict):
        raise RuntimeError("Attempt-T retention binding is absent")
    check_file(retention_path, retention_binding)
    retention = read_json(retention_path)
    if lock.get("retention_receipt_sha256") != sha256(retention_path):
        raise RuntimeError("primary-evidence lock does not bind retention receipt")
    for field, value in {
        "schema_version": "attempt-t-retention-audit-v1",
        "status": "PASS",
        "canonical_attempt": "T",
        "retention_status": "pass-all-prespecified-t-retained",
        "selection_rule": "retain-all-prespecified-attempt-t",
        "protocol_id": PROTOCOL_ID,
        "excluded_attempts": list(EXCLUDED_ATTEMPTS),
        "pipeline_completion_sha256": pipeline.get("sha256"),
        "pipeline_closure_receipt_sha256": sha256(closure_path),
        "pipeline_closure_evidence_set_sha256": closure.get("evidence_set_sha256"),
        "retained_stage_count": 38,
        "retained_stage_order_sha256": order_hash,
    }.items():
        if retention.get(field) != value:
            raise RuntimeError(f"Attempt-T retention mismatch: {field}")
    safety = retention.get("safety")
    if safety != {
        "all_prespecified_t_outputs_retained": True,
        "excluded_attempt_compared": False,
        "excluded_attempt_opened": False,
        "non_overwriting": True,
        "outcome_based_attempt_choice_possible": False,
        "scientific_interpretation_performed": False,
        "selection_candidate_set": ["T"],
        "value_dependent_filtering_performed": False,
    }:
        raise RuntimeError("Attempt-T retention safety boundary mismatch")
    wrapper_hashes = {
        stage: str(stage_by_name[stage].get("original_wrapper_sha256", ""))
        for stage in STAGE_ORDER
    }
    if (
        any(not HEX64.fullmatch(value) for value in wrapper_hashes.values())
        or retention.get("all_stage_manifest_sha256") != wrapper_hashes
    ):
        raise RuntimeError("Attempt-T retained manifest set is incomplete")
    primary = retention.get("primary_completeness_stages")
    if not isinstance(primary, dict) or set(primary) != set(PRIMARY_STAGES):
        raise RuntimeError("Attempt-T primary completeness set is incomplete")
    for stage in PRIMARY_STAGES:
        observed = paired_summary(root, stage_by_name[stage])
        row = primary[stage]
        if (
            row.get("logical_path")
            != f"canonical_attempt_T/{stage}/paired_cluster_bootstrap.csv"
            or row.get("closure_path") != f"a32/{stage}/paired_cluster_bootstrap.csv"
            or row.get("raw_sha256") != observed["sha256"]
            or row.get("bytes") != observed["bytes"]
            or row.get("rows") != observed["rows"]
            or row.get("columns") != observed["columns"]
            or row.get("row_key_fields") != list(ROW_KEYS)
            or row.get("scientific_values_interpreted") is not False
        ):
            raise RuntimeError(f"Attempt-T primary completeness mismatch: {stage}")
    for stage in ("S0", "S3"):
        if lock.get("stages", {}).get(stage) != {
            "paired_summary_sha256": primary[stage]["raw_sha256"]
        }:
            raise RuntimeError(f"primary-evidence lock stage mismatch: {stage}")

    application = receipt.get("application_projection")
    app_path = root / "app" / "RELEASE_RECEIPT.json"
    if (
        not isinstance(application, dict)
        or application.get("path") != "app/RELEASE_RECEIPT.json"
        or application.get("sha256") != APPLICATION_SHA256
        or not app_path.is_file()
        or sha256(app_path) != APPLICATION_SHA256
    ):
        raise RuntimeError("headed-stud projection binding mismatch")

    ledger_path = root / "provenance" / "FILE_HASHES.csv"
    with ledger_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    recorded = {
        str(row["path"]): (str(row["sha256"]), int(row["bytes"])) for row in rows
    }
    if len(recorded) != len(rows) or any(
        not HEX64.fullmatch(digest) for digest, _ in recorded.values()
    ):
        raise RuntimeError("public file ledger contains a duplicate or invalid hash")
    actual = {
        path.relative_to(root).as_posix(): (sha256(path), path.stat().st_size)
        for path in root.rglob("*")
        if path.is_file()
        and path != ledger_path
        and not any(
            part.lower() in IGNORED_LOCAL_PARTS
            for part in path.relative_to(root).parts
        )
    }
    if recorded != actual:
        raise RuntimeError("public file ledger does not match the exact file set")
    return {
        "status": "PUBLIC_PACKAGE_AUDIT_PASS",
        "profile": "attempt-t-public-v2",
        "release_status": receipt["release_status"],
        "canonical_attempt": "T",
        "stages": len(stages),
        "files_checked": scan_public_tree(root),
        "excluded_attempt_opened": False,
        "remote_action_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise RuntimeError(f"package root is not a directory: {root}")
    print(json.dumps(audit(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
