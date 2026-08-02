from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = PACKAGE_ROOT / "figures" / "contract_compiler"
CORE_FIGURE_4 = PACKAGE_ROOT / "figures" / "core" / "_impl" / "build_figure_4.py"
FINALIZER_T = PACKAGE_ROOT / "finalize_package_attempt_t.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path):
    gate = _load(
        "attempt_t_retention_gate_fixture",
        FIGURE_ROOT / "attempt_t_retention_gate.py",
    )
    root = tmp_path / "cvT" / "a32"
    root.mkdir(parents=True)
    manifest_hashes: dict[str, str] = {}
    for stage in gate.STAGE_ORDER:
        directory = root / gate._stage_directory(stage)
        directory.mkdir(parents=True)
        manifest = directory / "v32_aggregate_manifest.json"
        manifest.write_text(
            json.dumps({"schema_version": "synthetic-test", "stage": stage}),
            encoding="utf-8",
        )
        manifest_hashes[stage] = _sha256(manifest)

    primary: dict[str, dict[str, object]] = {}
    for stage in gate.PRIMARY_STAGES:
        summary = root / stage / gate.SUMMARY_NAME
        summary.write_text(
            "scenario,reference_method,comparator,synthetic_value\n"
            f"{stage},ClaimCut,RandomKFold,0\n",
            encoding="utf-8",
        )
        primary[stage] = {
            "logical_path": f"canonical_attempt_T/{stage}/{gate.SUMMARY_NAME}",
            "closure_path": f"a32/{stage}/{gate.SUMMARY_NAME}",
            "raw_sha256": _sha256(summary),
            "bytes": summary.stat().st_size,
            "rows": 1,
            "columns": [
                "scenario",
                "reference_method",
                "comparator",
                "synthetic_value",
            ],
            "row_key_fields": list(gate.ROW_KEY_FIELDS),
            "scientific_values_interpreted": False,
        }

    pipeline_hash = "a" * 64
    closure_hash = "b" * 64
    evidence_hash = "c" * 64
    retention = {
        "schema_version": gate.RETENTION_SCHEMA,
        "status": "PASS",
        "canonical_attempt": gate.CANONICAL_ATTEMPT,
        "retention_status": gate.RETENTION_STATUS,
        "selection_rule": "retain-all-prespecified-attempt-t",
        "protocol_id": gate.PROTOCOL_ID,
        "excluded_attempts": list(gate.EXCLUDED_ATTEMPTS),
        "pipeline_completion_sha256": pipeline_hash,
        "pipeline_closure_receipt_sha256": closure_hash,
        "pipeline_closure_evidence_set_sha256": evidence_hash,
        "retained_stage_count": len(gate.STAGE_ORDER),
        "retained_stage_order_sha256": _canonical_sha256(list(gate.STAGE_ORDER)),
        "root_binding": {
            "mode": gate.RETENTION_ROOT_MODE,
            "production_admissible": True,
            gate.ROOT_BINDING_FIELD: "a32",
            "receipt_logical_path": gate.RETENTION_RECEIPT_NAME,
        },
        "all_stage_manifest_sha256": manifest_hashes,
        "primary_completeness_stages": primary,
        "safety": {
            "all_prespecified_t_outputs_retained": True,
            "excluded_attempt_compared": False,
            "excluded_attempt_opened": False,
            "value_dependent_filtering_performed": False,
            "scientific_interpretation_performed": False,
            "selection_candidate_set": ["T"],
            "outcome_based_attempt_choice_possible": False,
            "non_overwriting": True,
        },
    }
    retention_path = tmp_path / gate.RETENTION_RECEIPT_NAME
    retention_path.write_text(
        json.dumps(retention, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze_hash = "d" * 64
    lock = {
        "schema_version": gate.LOCK_SCHEMA,
        "status": "closed",
        "protocol_id": gate.PROTOCOL_ID,
        "freeze_manifest_sha256": freeze_hash,
        "full_pipeline_stage_count": len(gate.STAGE_ORDER),
        "selection_gate": "t-only-retention",
        "retention_status": gate.RETENTION_STATUS,
        "canonical_attempt": gate.CANONICAL_ATTEMPT,
        "pipeline_completion_sha256": pipeline_hash,
        "pipeline_closure_receipt_sha256": closure_hash,
        "pipeline_closure_evidence_set_sha256": evidence_hash,
        "retention_receipt_sha256": _sha256(retention_path),
        "stages": {
            stage: {"paired_summary_sha256": primary[stage]["raw_sha256"]}
            for stage in gate.FIGURE_STAGES
        },
    }
    lock_path = tmp_path / "PRIMARY_EVIDENCE_LOCK.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return gate, root, retention_path, lock_path, freeze_hash


def _validate(gate, root, retention_path, lock_path, freeze_hash):
    return gate.validate_figure4_evidence(
        primary_evidence_lock=lock_path,
        retention_receipt=retention_path,
        aggregate_root=root,
        expected_freeze_manifest_sha256=freeze_hash,
    )


def test_attempt_t_gate_binds_retained_primary_stages(tmp_path: Path) -> None:
    binding = _validate(*_fixture(tmp_path))
    assert set(binding.primary_hashes) == {"S0", "S3"}
    assert set(binding.receipt["retention_stages"]) == {"S0", "S1", "S3"}
    assert binding.receipt["canonical_attempt"] == "T"
    assert binding.receipt["selection_gate"] == "t-only-retention"


def test_attempt_t_gate_refuses_manifest_drift(tmp_path: Path) -> None:
    gate, root, retention_path, lock_path, freeze_hash = _fixture(tmp_path)
    manifest = root / gate._stage_directory("S8") / "v32_aggregate_manifest.json"
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(gate.AttemptTRetentionGateError, match="manifest binding"):
        _validate(gate, root, retention_path, lock_path, freeze_hash)


def test_attempt_t_gate_refuses_s1_summary_drift(tmp_path: Path) -> None:
    gate, root, retention_path, lock_path, freeze_hash = _fixture(tmp_path)
    with (root / "S1" / gate.SUMMARY_NAME).open("a", encoding="utf-8") as handle:
        handle.write("S1,ClaimCut,RandomKFold,1\n")
    with pytest.raises(gate.AttemptTRetentionGateError, match="S1"):
        _validate(gate, root, retention_path, lock_path, freeze_hash)


def test_attempt_t_public_entry_points_expose_retention_receipt_cli() -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for script in (FIGURE_ROOT / "build_primary_scaling_figure.py", CORE_FIGURE_4):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
        assert "--retention-receipt" in result.stdout


def test_attempt_t_finalizer_scaffold_is_t_only() -> None:
    if not FINALIZER_T.is_file():
        pytest.skip("the immutable finalizer is a controlled-draft build tool")
    finalizer = _load("attempt_t_finalizer_scaffold", FINALIZER_T)
    scaffold = finalizer._impl.SCAFFOLD_FILES
    assert "figures/contract_compiler/attempt_t_retention_gate.py" in scaffold
    assert "tests/figures/test_contract_compiler_figures_attempt_t.py" in scaffold
    assert "tools/audit_public_package_attempt_t.py" in scaffold
    assert "figures/contract_compiler/attempt_comparison_gate.py" not in scaffold
    assert "tests/figures/test_contract_compiler_figures_attempt_s.py" not in scaffold
