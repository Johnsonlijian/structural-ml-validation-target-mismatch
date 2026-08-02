"""Temporary-root tests for v3.2 integrity; canonical roots are never touched."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from protocol_v32 import (
    CLAIM_SCHEMA,
    COMPILER_SCHEMA,
    FREEZE_SCHEMA,
    MASTER_SEED,
    PREDICTIVE_SCHEMA,
    PROTOCOL_ID,
    SEED_NAMES,
    STAGE_ORDER,
    ProtocolPaths,
    aggregate_temporary_path,
    atomic_create_json,
    atomic_temporary_path,
    begin_worker,
    bind_result,
    create_claim,
    derive_seeds,
    expected_envelope,
    finish_worker,
    reject_aggregate_temporary_pollution,
    runtime_preflight,
    sha256_bytes,
    sha256_file,
    validate_record_pair,
    validate_scenario_name,
    validate_stage_complete,
    validate_worker_done,
)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_fake_freeze(tmp_path: Path) -> tuple[ProtocolPaths, object]:
    cv_root = tmp_path / "cv1"
    cv_root.mkdir()
    paths = ProtocolPaths(cv_root=cv_root, project_root=tmp_path)
    paths.prespec.write_text("v3.2 test prespec\n", encoding="utf-8")
    scientific_source = (
        Path(__file__).resolve().parent.parent
        / "confirmatory_prespecification_v3.md"
    )
    scientific = tmp_path / "confirmatory_prespecification_v3.md"
    scientific.write_bytes(scientific_source.read_bytes())
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"artifact")
    freeze = paths.freeze_root
    freeze.mkdir()

    seed_path = freeze / "confirmatory_seed_table.csv"
    fields = [
        "scenario",
        "replicate",
        "master_seed",
        "generator_seed",
        "split_seed",
        "learner_seed",
        "datasail_seed",
    ]
    with seed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in STAGE_ORDER:
            for replicate in range(100):
                seeds = derive_seeds(scenario, replicate)
                writer.writerow(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "master_seed": MASTER_SEED,
                        **{f"{name}_seed": seeds[name] for name in SEED_NAMES},
                    }
                )

    sources = [
        {
            "path": str(source.relative_to(tmp_path)),
            "exists": True,
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    ]
    source_registry = freeze / "source_hashes.json"
    _write_json(source_registry, sources)
    excluded = freeze / "excluded_pilot_registry.json"
    _write_json(excluded, [])
    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "status": "final",
        "protocol_id": PROTOCOL_ID,
        "master_seed": MASTER_SEED,
        "output_schema_version": PREDICTIVE_SCHEMA,
        "compiler_schema_version": COMPILER_SCHEMA,
        "contract_schema_version": "deployment-contract-v3",
        "confirmatory_root_clean": True,
        "all_required_exist": True,
        "all_checks_pass": True,
        "confirmatory_files_found": [],
        "freeze_root": paths.relative(paths.freeze_root),
        "confirmatory_root": paths.relative(paths.confirmatory_root),
        "aggregate_root": paths.relative(paths.aggregate_root),
        "stage_order": list(STAGE_ORDER),
        "shards": [{"start": 0, "count": 50}, {"start": 50, "count": 50}],
        "seed_table": {
            "path": paths.relative(seed_path),
            "sha256": sha256_file(seed_path),
            "n_rows": 3800,
            "n_scenarios": 38,
        },
        "source_registry": {
            "path": paths.relative(source_registry),
            "sha256": sha256_file(source_registry),
        },
        "sources": sources,
        "artifacts": [
            {
                "path": str(artifact.relative_to(tmp_path)),
                "exists": True,
                "bytes": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
        ],
        "excluded_registry": {
            "path": paths.relative(excluded),
            "sha256": sha256_file(excluded),
        },
        "prespecification": {
            "path": paths.relative(paths.prespec),
            "sha256": sha256_file(paths.prespec),
        },
        "scientific_prespecification": {
            "path": paths.relative(scientific),
            "sha256": sha256_file(scientific),
        },
    }
    manifest_path = freeze / "freeze_manifest.json"
    _write_json(manifest_path, manifest)
    (freeze / "freeze_manifest.sha256").write_text(
        sha256_file(manifest_path) + "\n", encoding="ascii"
    )
    return paths, runtime_preflight(paths)


def _complete_record(context, scenario: str, replicate: int, start: int) -> None:
    _, claim_hash = create_claim(
        context, scenario, replicate, start=start, count=50
    )
    envelope = expected_envelope(context, scenario, replicate)
    payload = {
        "schema_version": envelope["result_schema"],
        "replicate": replicate,
        "seeds": envelope["seeds"],
        "seed_table_sha256": envelope["seed_table_sha256"],
        "status": "complete",
    }
    if scenario != "COMPILER_SEMANTIC":
        payload["scenario"] = scenario
    bound = bind_result(payload, envelope, claim_hash)
    atomic_create_json(context.paths.result_path(scenario, replicate), bound)


def test_seed_table_and_envelope_are_fully_bound(tmp_path: Path) -> None:
    paths, context = _build_fake_freeze(tmp_path)
    row = context.row("COMPILER_SEMANTIC", 0)
    assert row.master_seed == MASTER_SEED
    assert row.seeds == derive_seeds("COMPILER_SEMANTIC", 0)
    paths.scenario_root("COMPILER_SEMANTIC").mkdir(parents=True)
    _complete_record(context, "COMPILER_SEMANTIC", 0, 0)
    hashes = validate_record_pair(context, "COMPILER_SEMANTIC", 0)
    claim = json.loads(paths.claim_path("COMPILER_SEMANTIC", 0).read_text())
    result = json.loads(paths.result_path("COMPILER_SEMANTIC", 0).read_text())
    assert claim["claim_schema"] == CLAIM_SCHEMA
    assert claim["execution_envelope"]["master_seed"] == MASTER_SEED
    assert claim["execution_envelope"]["result_schema"] == COMPILER_SCHEMA
    assert result["scenario"] == "COMPILER_SEMANTIC"
    assert result["master_seed"] == MASTER_SEED
    assert result["claim_sha256"] == hashes["claim_sha256"]


def test_manifest_and_live_source_drift_are_rejected(tmp_path: Path) -> None:
    paths, _ = _build_fake_freeze(tmp_path)
    source = tmp_path / "source.py"
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash drifted"):
        runtime_preflight(paths)


def test_manifest_sidecar_is_required(tmp_path: Path) -> None:
    paths, _ = _build_fake_freeze(tmp_path)
    (paths.freeze_root / "freeze_manifest.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(RuntimeError, match="immutable sidecar"):
        runtime_preflight(paths)


@pytest.mark.parametrize("scenario", ["../S8", "S8/x", "S8\\x", "UNKNOWN"])
def test_scenario_whitelist_precedes_path_construction(scenario: str) -> None:
    with pytest.raises(RuntimeError):
        validate_scenario_name(scenario)


def test_orphan_and_pollution_are_hard_stops(tmp_path: Path) -> None:
    paths, context = _build_fake_freeze(tmp_path)
    stage = paths.scenario_root("COMPILER_SEMANTIC")
    stage.mkdir(parents=True)
    begin_worker(context, "COMPILER_SEMANTIC", 0, 50)
    create_claim(context, "COMPILER_SEMANTIC", 0, start=0, count=50)
    with pytest.raises(RuntimeError, match="orphan"):
        validate_record_pair(context, "COMPILER_SEMANTIC", 0)
    with pytest.raises(RuntimeError, match="active"):
        begin_worker(context, "COMPILER_SEMANTIC", 0, 50)


@pytest.mark.parametrize("prefix", [".__v32_tmp_", ".__freeze_v32_"])
def test_begin_worker_rejects_stale_temporary_artifacts(
    tmp_path: Path, prefix: str
) -> None:
    paths, context = _build_fake_freeze(tmp_path)
    stage = paths.scenario_root("COMPILER_SEMANTIC")
    stage.mkdir(parents=True)
    (stage / f"{prefix}crash").write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="temporary artifact"):
        begin_worker(context, "COMPILER_SEMANTIC", 0, 50)


def test_worker_done_requires_fifty_valid_pairs(tmp_path: Path) -> None:
    _, context = _build_fake_freeze(tmp_path)
    assert begin_worker(context, "COMPILER_SEMANTIC", 0, 50) == "started"
    for replicate in range(50):
        _complete_record(context, "COMPILER_SEMANTIC", replicate, 0)
    finish_worker(context, "COMPILER_SEMANTIC", 0, 50)
    validate_worker_done(context, "COMPILER_SEMANTIC", 0, 50)


def test_stage_completion_rejects_extra_worker_record_and_temp_files(
    tmp_path: Path,
) -> None:
    paths, context = _build_fake_freeze(tmp_path)
    for start in (0, 50):
        assert begin_worker(context, "COMPILER_SEMANTIC", start, 50) == "started"
        for replicate in range(start, start + 50):
            _complete_record(context, "COMPILER_SEMANTIC", replicate, start)
        finish_worker(context, "COMPILER_SEMANTIC", start, 50)
    stage = paths.scenario_root("COMPILER_SEMANTIC")
    validate_stage_complete(context, "COMPILER_SEMANTIC")

    extra_worker = stage / "worker_100_149.done"
    extra_worker.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="worker sentinel set"):
        validate_stage_complete(context, "COMPILER_SEMANTIC")
    extra_worker.unlink()

    extra_active = stage / "worker_100_149.active"
    extra_active.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="worker sentinel set"):
        validate_stage_complete(context, "COMPILER_SEMANTIC")
    extra_active.unlink()

    noncanonical_record = stage / "rep_0000.json.backup"
    noncanonical_record.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="record set"):
        validate_stage_complete(context, "COMPILER_SEMANTIC")
    noncanonical_record.unlink()

    for prefix in (".__v32_tmp_", ".__freeze_v32_"):
        stale = stage / f"{prefix}interrupted"
        stale.write_text("stale", encoding="utf-8")
        with pytest.raises(RuntimeError, match="temporary artifact"):
            validate_stage_complete(context, "COMPILER_SEMANTIC")
        stale.unlink()


def test_next_stage_requires_bound_predecessor_aggregate(tmp_path: Path) -> None:
    paths, context = _build_fake_freeze(tmp_path)
    for start in (0, 50):
        assert begin_worker(context, "COMPILER_SEMANTIC", start, 50) == "started"
        for replicate in range(start, start + 50):
            _complete_record(context, "COMPILER_SEMANTIC", replicate, start)
        finish_worker(context, "COMPILER_SEMANTIC", start, 50)
    with pytest.raises(RuntimeError, match="aggregate is absent"):
        begin_worker(context, "S8", 0, 50)
    records = validate_stage_complete(context, "COMPILER_SEMANTIC")
    registry_hash = sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    )
    aggregate = paths.aggregate_root / "compiler_semantic"
    aggregate.mkdir(parents=True)
    _write_json(
        aggregate / "v32_aggregate_manifest.json",
        {
            "aggregate_schema": "v32-compiler-aggregate-v1",
            "protocol_id": PROTOCOL_ID,
            "manifest_sha256": context.manifest_sha256,
            "seed_table_sha256": context.seed_table_sha256,
            "scenario": "COMPILER_SEMANTIC",
            "n_claims": 100,
            "n_results": 100,
            "input_record_registry_sha256": registry_hash,
            "input_records": records,
            "hard_gate": {
                "n_complete": 100,
                "n_failed": 0,
                "false_exact_events_total": 0,
                "all_primary_observations_pass": True,
            },
        },
    )
    aggregate_temp = paths.aggregate_root / ".__tmp_interrupted_aggregate"
    aggregate_temp.mkdir()
    with pytest.raises(RuntimeError, match="aggregate temporary artifact"):
        begin_worker(context, "S8", 0, 50)
    aggregate_temp.rmdir()
    assert begin_worker(context, "S8", 0, 50) == "started"


def test_wrong_result_seed_is_rejected(tmp_path: Path) -> None:
    paths, context = _build_fake_freeze(tmp_path)
    paths.scenario_root("COMPILER_SEMANTIC").mkdir(parents=True)
    _, claim_hash = create_claim(
        context, "COMPILER_SEMANTIC", 0, start=0, count=50
    )
    envelope = expected_envelope(context, "COMPILER_SEMANTIC", 0)
    bad = {
        "schema_version": COMPILER_SCHEMA,
        "replicate": 0,
        "seeds": {**envelope["seeds"], "split": -1},
        "seed_table_sha256": envelope["seed_table_sha256"],
        "status": "complete",
    }
    with pytest.raises(RuntimeError, match="component seeds"):
        bind_result(bad, envelope, claim_hash)


def test_official_cli_surfaces_expose_no_unsafe_options() -> None:
    root = Path(__file__).resolve().parent
    for name in ("run_batch_v32.py", "run_compiler_suite_v32.py"):
        text = (root / name).read_text(encoding="utf-8")
        for forbidden in (
            '"--output"',
            '"--confirmatory"',
            '"--overwrite"',
            '"--skip-datasail"',
            '"--datasail-max-sec"',
            '"--count"',
        ):
            assert forbidden not in text
    for name in ("aggregate_batch_v32.py", "aggregate_compiler_suite_v32.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "reject_aggregate_temporary_pollution(paths)" in text


def test_aggregate_temp_and_v3_prespec_prewrite_guards(tmp_path: Path) -> None:
    from freeze_experiment_v32 import (
        assert_v3_scientific_prespecification,
        assert_windows_path_budget,
        ast_check_command,
        freeze_temporary_path,
    )

    paths, _ = _build_fake_freeze(tmp_path)
    paths.aggregate_root.mkdir()
    stale = paths.aggregate_root / ".__tmp_crashed"
    stale.mkdir()
    with pytest.raises(RuntimeError, match="aggregate temporary artifact"):
        reject_aggregate_temporary_pollution(paths)
    stale.rmdir()
    scientific = Path(__file__).resolve().parent.parent / "confirmatory_prespecification_v3.md"
    assert_v3_scientific_prespecification(scientific)
    wrong = tmp_path / "wrong_v3.md"
    wrong.write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="before v3.2 freeze writes"):
        assert_v3_scientific_prespecification(wrong)
    assert_windows_path_budget()

    command = ast_check_command()
    rendered = " ".join(command)
    assert "py_compile" not in rendered
    assert "ast.parse" in rendered
    assert "-B" in command
    cache = Path(__file__).resolve().parent / "__pycache__"
    before = {path.name for path in cache.glob("*v32*.pyc")} if cache.exists() else set()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        text=True,
        capture_output=True,
        check=False,
    )
    after = {path.name for path in cache.glob("*v32*.pyc")} if cache.exists() else set()
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "AST_PASS 7"
    assert after == before

    canonical = ProtocolPaths.canonical()
    longest_stage = max(STAGE_ORDER, key=lambda name: (len(name), name))
    assert longest_stage == "CORE_2.4_1.8_0.35"
    longest_result = canonical.result_path(longest_stage, 99)
    longest_claim = canonical.claim_path(longest_stage, 99)
    worker_active = canonical.scenario_root(longest_stage) / "worker_050_099.active"
    worker_done = (
        canonical.scenario_root(longest_stage)
        / "worker_050_099.done"
    )
    freeze_manifest = canonical.freeze_root / "freeze_manifest.json"
    finals = (
        longest_result,
        longest_claim,
        worker_active,
        worker_done,
        freeze_manifest,
    )
    assert all(len(str(path)) < 260 for path in finals)
    maximum_pid = 4294967295
    protocol_temp = atomic_temporary_path(longest_result, process_id=maximum_pid)
    freeze_temp = freeze_temporary_path(freeze_manifest, process_id=maximum_pid)
    aggregate_temp = aggregate_temporary_path(canonical, process_id=maximum_pid)
    deepest_aggregate_temp = (
        aggregate_temp / "datasail_solver_status_summary.csv"
    )
    deepest_aggregate_final = (
        canonical.aggregate_root
        / longest_stage
        / "datasail_solver_status_summary.csv"
    )
    assert protocol_temp.name == ".__v32_tmp_4294967295"
    assert freeze_temp.name == ".__freeze_v32_4294967295"
    assert aggregate_temp.name == ".__tmp_4294967295"
    assert len(str(protocol_temp)) < 260
    assert len(str(freeze_temp)) < 260
    assert len(str(deepest_aggregate_temp)) < 260
    assert len(str(deepest_aggregate_final)) < 260


def test_official_compiler_cli_writes_only_temp_canonical_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import run_compiler_suite_v32 as cli

    paths, context = _build_fake_freeze(tmp_path)

    def fake_replicate(replicate: int):
        row = context.row("COMPILER_SEMANTIC", replicate)
        return {
            "schema_version": COMPILER_SCHEMA,
            "replicate": replicate,
            "seeds": row.seeds,
            "seed_table_sha256": context.seed_table_sha256,
            "status": "complete",
        }

    fake_suite = types.SimpleNamespace(run_suite_replicate=fake_replicate)
    monkeypatch.setitem(sys.modules, "run_compiler_suite", fake_suite)
    monkeypatch.setattr(
        cli.ProtocolPaths,
        "canonical",
        classmethod(lambda cls: paths),
    )
    monkeypatch.setattr(cli, "runtime_preflight", lambda received: context)
    monkeypatch.setattr(cli, "activate_base_runner", lambda received: None)
    monkeypatch.setattr(sys, "argv", ["run_compiler_suite_v32.py", "--start", "0"])
    cli.main()
    validate_worker_done(context, "COMPILER_SEMANTIC", 0, 50)
    assert len(list(paths.scenario_root("COMPILER_SEMANTIC").glob("rep_*.json"))) == 50


def test_official_predictive_cli_freezes_datasail_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import run_batch_v32 as cli

    paths, context = _build_fake_freeze(tmp_path)
    calls = []

    class FakeRunner:
        def run_replicate(self, **kwargs):
            calls.append(kwargs)
            replicate = int(kwargs["replicate"])
            row = context.row("S8", replicate)
            return {
                "schema_version": PREDICTIVE_SCHEMA,
                "scenario": "S8",
                "replicate": replicate,
                "seeds": row.seeds,
                "seed_table_sha256": context.seed_table_sha256,
                "status": "complete",
            }

    monkeypatch.setattr(
        cli.ProtocolPaths,
        "canonical",
        classmethod(lambda cls: paths),
    )
    monkeypatch.setattr(cli, "runtime_preflight", lambda received: context)
    monkeypatch.setattr(
        cli,
        "begin_worker",
        lambda *args: (
            paths.scenario_root("S8").mkdir(parents=True, exist_ok=True)
            or "started"
        ),
    )
    monkeypatch.setattr(cli, "finish_worker", lambda *args: None)
    monkeypatch.setattr(cli, "activate_base_runner", lambda received: FakeRunner())
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_batch_v32.py", "--scenario", "S8", "--start", "0"],
    )
    cli.main()
    assert len(calls) == 50
    assert all(call["skip_datasail"] is False for call in calls)
    assert all(call["datasail_max_sec"] == 60 for call in calls)
    validate_record_pair(context, "S8", 0)
    validate_record_pair(context, "S8", 49)
