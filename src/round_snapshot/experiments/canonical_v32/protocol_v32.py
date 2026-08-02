"""Strict execution and provenance binding for confirmatory protocol v3.2.

This module owns only execution integrity.  The scientific implementation
remains in the frozen v3 base modules.  Official v3.2 entry points derive all
paths, reject recovery/overwrite, and bind every claim and result to one final
freeze and one unique seed-table row.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


PROTOCOL_ID = "SAVP-CONFIRMATORY-V3.2"
FREEZE_SCHEMA = "confirmatory-freeze-v3.2"
ENVELOPE_SCHEMA = "confirmatory-execution-envelope-v1"
CLAIM_SCHEMA = "confirmatory-exclusive-claim-v2"
WORKER_SCHEMA = "confirmatory-worker-v1"
MASTER_SEED = 2026071204
PREDICTIVE_SCHEMA = "cv1-replicate-v4"
COMPILER_SCHEMA = "compiler-semantic-v3"
CONTRACT_SCHEMA = "deployment-contract-v3"
SEED_NAMES = ("generator", "split", "learner", "datasail")
SEED_TABLE_SHA_RE = re.compile(r"^[0-9a-f]{64}$")

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
SCENARIOS = frozenset(STAGE_ORDER)
SHARDS = ((0, 50), (50, 50))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_scenario_code(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def derive_seeds(scenario: str, replicate: int) -> dict[str, int]:
    root = np.random.SeedSequence(
        [MASTER_SEED, stable_scenario_code(scenario), int(replicate)]
    )
    return {
        name: int(child.generate_state(1, dtype=np.uint32)[0])
        for name, child in zip(SEED_NAMES, root.spawn(len(SEED_NAMES)))
    }


def validate_scenario_name(scenario: str) -> None:
    if scenario not in SCENARIOS:
        raise RuntimeError(f"scenario is not in the frozen v3.2 whitelist: {scenario!r}")
    if any(token in scenario for token in ("/", "\\", "..")):
        raise RuntimeError(f"unsafe scenario token: {scenario!r}")


def validate_shard(start: int, count: int) -> None:
    if (int(start), int(count)) not in SHARDS:
        raise RuntimeError("v3.2 permits only the frozen shards 0/50 and 50/50")


@dataclass(frozen=True)
class ProtocolPaths:
    cv_root: Path
    project_root: Path

    @classmethod
    def canonical(cls) -> "ProtocolPaths":
        cv_root = Path(__file__).resolve().parent
        round_root = cv_root.parents[1]
        project_root = round_root.parents[1]
        return cls(cv_root=cv_root, project_root=project_root)

    @property
    def freeze_root(self) -> Path:
        return self.cv_root / "freeze_v32"

    @property
    def confirmatory_root(self) -> Path:
        return self.cv_root / "confirmatory_v32"

    @property
    def aggregate_root(self) -> Path:
        return self.cv_root / "a32"

    @property
    def prespec(self) -> Path:
        return self.cv_root.parent / "confirmatory_prespecification_v32.md"

    def scenario_root(self, scenario: str) -> Path:
        validate_scenario_name(scenario)
        name = "compiler_semantic" if scenario == "COMPILER_SEMANTIC" else scenario
        result = (self.confirmatory_root / name).resolve()
        expected_parent = self.confirmatory_root.resolve()
        if result.parent != expected_parent:
            raise RuntimeError("scenario path escaped the canonical confirmatory root")
        return result

    def result_path(self, scenario: str, replicate: int) -> Path:
        return self.scenario_root(scenario) / f"rep_{int(replicate):04d}.json"

    def claim_path(self, scenario: str, replicate: int) -> Path:
        return self.scenario_root(scenario) / f"rep_{int(replicate):04d}.claim"

    def relative(self, path: Path) -> str:
        resolved = path.resolve()
        root = self.project_root.resolve()
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"path is outside the project root: {resolved}")
        return str(resolved.relative_to(root))


@dataclass(frozen=True)
class SeedRow:
    scenario: str
    replicate: int
    master_seed: int
    seeds: dict[str, int]


@dataclass(frozen=True)
class FreezeContext:
    paths: ProtocolPaths
    manifest: dict[str, Any]
    manifest_sha256: str
    seed_table_sha256: str
    source_registry_sha256: str
    prespec_sha256: str
    rows: Mapping[tuple[str, int], SeedRow]

    def row(self, scenario: str, replicate: int) -> SeedRow:
        validate_scenario_name(scenario)
        key = (scenario, int(replicate))
        if key not in self.rows:
            raise RuntimeError(f"seed row is absent from v3.2 freeze: {scenario}/{replicate}")
        row = self.rows[key]
        expected = derive_seeds(scenario, replicate)
        if row.master_seed != MASTER_SEED or row.seeds != expected:
            raise RuntimeError(f"seed row does not match v3.2 derivation: {scenario}/{replicate}")
        return row


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"unreadable protocol JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"protocol JSON must contain an object: {path}")
    return value


def _read_sha_sidecar(path: Path) -> str:
    try:
        value = path.read_text(encoding="ascii").strip().lower()
    except Exception as exc:
        raise RuntimeError(f"missing or unreadable SHA-256 sidecar: {path}") from exc
    if not SEED_TABLE_SHA_RE.fullmatch(value):
        raise RuntimeError(f"invalid SHA-256 sidecar: {path}")
    return value


def _resolve_registry_path(paths: ProtocolPaths, relative: str) -> Path:
    candidate = (paths.project_root / relative).resolve()
    if not candidate.is_relative_to(paths.project_root.resolve()):
        raise RuntimeError(f"freeze registry path escaped project root: {relative}")
    return candidate


def _verify_registry(paths: ProtocolPaths, entries: Iterable[dict[str, Any]]) -> None:
    for entry in entries:
        path = _resolve_registry_path(paths, str(entry.get("path", "")))
        if not path.is_file():
            raise RuntimeError(f"frozen dependency is missing: {path}")
        if sha256_file(path) != entry.get("sha256"):
            raise RuntimeError(f"frozen dependency hash drifted: {path}")
        if int(entry.get("bytes", -1)) != path.stat().st_size:
            raise RuntimeError(f"frozen dependency size drifted: {path}")


def _read_seed_table(path: Path) -> dict[tuple[str, int], SeedRow]:
    rows: dict[tuple[str, int], SeedRow] = {}
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_fields = [
                "scenario",
                "replicate",
                "master_seed",
                "generator_seed",
                "split_seed",
                "learner_seed",
                "datasail_seed",
            ]
            if reader.fieldnames != expected_fields:
                raise RuntimeError("v3.2 seed-table columns differ from the frozen schema")
            for raw in reader:
                scenario = str(raw["scenario"])
                validate_scenario_name(scenario)
                replicate = int(raw["replicate"])
                key = (scenario, replicate)
                if key in rows:
                    raise RuntimeError(f"duplicate v3.2 seed row: {key}")
                rows[key] = SeedRow(
                    scenario=scenario,
                    replicate=replicate,
                    master_seed=int(raw["master_seed"]),
                    seeds={name: int(raw[f"{name}_seed"]) for name in SEED_NAMES},
                )
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"cannot parse v3.2 seed table: {path}") from exc
    expected_keys = {(scenario, rep) for scenario in STAGE_ORDER for rep in range(100)}
    if set(rows) != expected_keys:
        missing = len(expected_keys.difference(rows))
        extra = len(set(rows).difference(expected_keys))
        raise RuntimeError(f"v3.2 seed-table key set mismatch: missing={missing}, extra={extra}")
    for row in rows.values():
        if row.master_seed != MASTER_SEED or row.seeds != derive_seeds(
            row.scenario, row.replicate
        ):
            raise RuntimeError(
                f"v3.2 seed derivation mismatch: {row.scenario}/{row.replicate}"
            )
    return rows


def runtime_preflight(paths: ProtocolPaths | None = None) -> FreezeContext:
    paths = paths or ProtocolPaths.canonical()
    manifest_path = paths.freeze_root / "freeze_manifest.json"
    sidecar_path = paths.freeze_root / "freeze_manifest.sha256"
    manifest_hash = _read_sha_sidecar(sidecar_path)
    if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_hash:
        raise RuntimeError("v3.2 freeze manifest does not match its immutable sidecar")
    manifest = _load_json(manifest_path)
    required = {
        "schema_version": FREEZE_SCHEMA,
        "status": "final",
        "protocol_id": PROTOCOL_ID,
        "master_seed": MASTER_SEED,
        "output_schema_version": PREDICTIVE_SCHEMA,
        "compiler_schema_version": COMPILER_SCHEMA,
        "contract_schema_version": CONTRACT_SCHEMA,
        "confirmatory_root_clean": True,
        "all_required_exist": True,
        "all_checks_pass": True,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"v3.2 manifest field mismatch: {field}")
    if manifest.get("confirmatory_files_found") != []:
        raise RuntimeError("v3.2 freeze was not created over an empty result root")
    if manifest.get("freeze_root") != paths.relative(paths.freeze_root):
        raise RuntimeError("v3.2 manifest names a noncanonical freeze root")
    if manifest.get("confirmatory_root") != paths.relative(paths.confirmatory_root):
        raise RuntimeError("v3.2 manifest names a noncanonical confirmatory root")
    if manifest.get("aggregate_root") != paths.relative(paths.aggregate_root):
        raise RuntimeError("v3.2 manifest names a noncanonical aggregate root")
    if manifest.get("stage_order") != list(STAGE_ORDER):
        raise RuntimeError("v3.2 manifest stage order differs from the protocol")
    if manifest.get("shards") != [
        {"start": start, "count": count} for start, count in SHARDS
    ]:
        raise RuntimeError("v3.2 manifest shard plan differs from the protocol")

    seed_meta = manifest.get("seed_table") or {}
    seed_path = _resolve_registry_path(paths, str(seed_meta.get("path", "")))
    if seed_path != (paths.freeze_root / "confirmatory_seed_table.csv").resolve():
        raise RuntimeError("v3.2 seed table is outside the canonical freeze root")
    seed_hash = sha256_file(seed_path) if seed_path.is_file() else None
    if seed_hash != seed_meta.get("sha256"):
        raise RuntimeError("v3.2 seed table does not match the final manifest")
    if seed_meta.get("n_rows") != 3800 or seed_meta.get("n_scenarios") != 38:
        raise RuntimeError("v3.2 seed-table dimensions are not 38 x 100")
    rows = _read_seed_table(seed_path)

    source_meta = manifest.get("source_registry") or {}
    source_path = _resolve_registry_path(paths, str(source_meta.get("path", "")))
    if source_path != (paths.freeze_root / "source_hashes.json").resolve():
        raise RuntimeError("v3.2 source registry is outside the canonical freeze root")
    source_hash = sha256_file(source_path) if source_path.is_file() else None
    if source_hash != source_meta.get("sha256"):
        raise RuntimeError("v3.2 source registry hash mismatch")
    source_entries = json.loads(source_path.read_text(encoding="utf-8"))
    if source_entries != manifest.get("sources"):
        raise RuntimeError("v3.2 source registry differs from the manifest")
    _verify_registry(paths, source_entries)
    _verify_registry(paths, manifest.get("artifacts") or [])

    exclusion_meta = manifest.get("excluded_registry") or {}
    exclusion_path = _resolve_registry_path(paths, str(exclusion_meta.get("path", "")))
    if exclusion_path != (
        paths.freeze_root / "excluded_pilot_registry.json"
    ).resolve():
        raise RuntimeError("v3.2 exclusion registry is outside the canonical freeze root")
    if not exclusion_path.is_file() or sha256_file(exclusion_path) != exclusion_meta.get(
        "sha256"
    ):
        raise RuntimeError("v3.2 exclusion registry hash mismatch")

    prespec_meta = manifest.get("prespecification") or {}
    prespec_path = _resolve_registry_path(paths, str(prespec_meta.get("path", "")))
    prespec_hash = sha256_file(prespec_path) if prespec_path.is_file() else None
    if prespec_hash != prespec_meta.get("sha256") or prespec_path != paths.prespec.resolve():
        raise RuntimeError("v3.2 prespecification hash or path mismatch")
    scientific = manifest.get("scientific_prespecification") or {}
    scientific_path = _resolve_registry_path(paths, str(scientific.get("path", "")))
    expected_scientific = paths.cv_root.parent / "confirmatory_prespecification_v3.md"
    if (
        scientific_path != expected_scientific.resolve()
        or not scientific_path.is_file()
        or sha256_file(scientific_path)
        != "c664f66742966c82b638463b8139e251e0bcdd5508209151982d7685adb44034"
        or scientific.get("sha256")
        != "c664f66742966c82b638463b8139e251e0bcdd5508209151982d7685adb44034"
    ):
        raise RuntimeError("v3 scientific prespecification binding mismatch")
    return FreezeContext(
        paths=paths,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        seed_table_sha256=str(seed_hash),
        source_registry_sha256=str(source_hash),
        prespec_sha256=str(prespec_hash),
        rows=rows,
    )


def result_schema_for(scenario: str) -> str:
    validate_scenario_name(scenario)
    return COMPILER_SCHEMA if scenario == "COMPILER_SEMANTIC" else PREDICTIVE_SCHEMA


def expected_envelope(
    context: FreezeContext,
    scenario: str,
    replicate: int,
) -> dict[str, Any]:
    row = context.row(scenario, replicate)
    result_path = context.paths.result_path(scenario, replicate)
    return {
        "envelope_schema": ENVELOPE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "freeze_schema": FREEZE_SCHEMA,
        "manifest_sha256": context.manifest_sha256,
        "source_registry_sha256": context.source_registry_sha256,
        "prespec_sha256": context.prespec_sha256,
        "seed_table_sha256": context.seed_table_sha256,
        "master_seed": row.master_seed,
        "seeds": dict(row.seeds),
        "scenario": scenario,
        "replicate": int(replicate),
        "result_schema": result_schema_for(scenario),
        "canonical_confirmatory_root": context.paths.relative(
            context.paths.confirmatory_root
        ),
        "canonical_result_path": context.paths.relative(result_path),
    }


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"immutable protocol file already exists: {path}") from exc


def atomic_create_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create a JSON file without replacing an existing target."""

    temporary = atomic_temporary_path(path)
    if temporary.exists():
        raise RuntimeError(f"stale v3.2 temporary file: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"immutable protocol file already exists: {path}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_temporary_path(path: Path, *, process_id: int | None = None) -> Path:
    """Return the short same-directory temp path used on long Windows roots."""

    pid = os.getpid() if process_id is None else int(process_id)
    return path.with_name(f".__v32_tmp_{pid}")


def create_claim(
    context: FreezeContext,
    scenario: str,
    replicate: int,
    *,
    start: int,
    count: int,
) -> tuple[Path, str]:
    validate_shard(start, count)
    claim_path = context.paths.claim_path(scenario, replicate)
    result_path = context.paths.result_path(scenario, replicate)
    if claim_path.exists() or result_path.exists():
        raise RuntimeError(f"v3.2 record path is already occupied: {result_path.stem}")
    payload = {
        "claim_schema": CLAIM_SCHEMA,
        "execution_envelope": expected_envelope(context, scenario, replicate),
        "worker": {"start": int(start), "count": int(count)},
        "process_id": int(os.getpid()),
        "claimed_utc": utc_now(),
    }
    _exclusive_json(claim_path, payload)
    return claim_path, sha256_file(claim_path)


def validate_base_payload(
    payload: dict[str, Any],
    envelope: dict[str, Any],
) -> None:
    scenario = str(envelope["scenario"])
    expected_schema = str(envelope["result_schema"])
    if payload.get("schema_version") != expected_schema:
        raise RuntimeError("base result schema differs from the v3.2 envelope")
    if int(payload.get("replicate", -1)) != int(envelope["replicate"]):
        raise RuntimeError("base result replicate differs from the v3.2 envelope")
    if payload.get("seed_table_sha256") != envelope["seed_table_sha256"]:
        raise RuntimeError("base result recorded the wrong v3.2 seed-table hash")
    if payload.get("seeds") != envelope["seeds"]:
        raise RuntimeError("base result component seeds differ from the v3.2 envelope")
    if scenario != "COMPILER_SEMANTIC" and payload.get("scenario") != scenario:
        raise RuntimeError("base predictive result scenario differs from the v3.2 envelope")


def bind_result(
    payload: dict[str, Any],
    envelope: dict[str, Any],
    claim_sha256: str,
) -> dict[str, Any]:
    validate_base_payload(payload, envelope)
    output = dict(payload)
    output.update(
        {
            "protocol_id": PROTOCOL_ID,
            "scenario": envelope["scenario"],
            "master_seed": envelope["master_seed"],
            "seeds": dict(envelope["seeds"]),
            "seed_table_sha256": envelope["seed_table_sha256"],
            "execution_envelope": dict(envelope),
            "claim_sha256": claim_sha256,
        }
    )
    return output


def validate_record_pair(
    context: FreezeContext,
    scenario: str,
    replicate: int,
    *,
    require_complete: bool = True,
) -> dict[str, str]:
    claim_path = context.paths.claim_path(scenario, replicate)
    result_path = context.paths.result_path(scenario, replicate)
    if claim_path.is_file() != result_path.is_file():
        raise RuntimeError(f"orphan v3.2 claim/result: {scenario}/{replicate}")
    if not claim_path.is_file():
        raise RuntimeError(f"missing v3.2 claim/result: {scenario}/{replicate}")
    claim = _load_json(claim_path)
    result = _load_json(result_path)
    envelope = expected_envelope(context, scenario, replicate)
    if claim.get("claim_schema") != CLAIM_SCHEMA:
        raise RuntimeError(f"claim schema mismatch: {claim_path}")
    if claim.get("execution_envelope") != envelope:
        raise RuntimeError(f"claim envelope mismatch: {claim_path}")
    worker = claim.get("worker") or {}
    try:
        worker_start, worker_count = int(worker["start"]), int(worker["count"])
        validate_shard(worker_start, worker_count)
    except Exception as exc:
        raise RuntimeError(f"claim worker range mismatch: {claim_path}") from exc
    if not worker_start <= int(replicate) < worker_start + worker_count:
        raise RuntimeError(f"claim replicate is outside its worker shard: {claim_path}")
    if not isinstance(claim.get("claimed_utc"), str):
        raise RuntimeError(f"claim timestamp is absent: {claim_path}")
    claim_hash = sha256_file(claim_path)
    if result.get("execution_envelope") != envelope:
        raise RuntimeError(f"result envelope mismatch: {result_path}")
    if result.get("claim_sha256") != claim_hash:
        raise RuntimeError(f"result-to-claim hash mismatch: {result_path}")
    validate_base_payload(result, envelope)
    if result.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"result protocol mismatch: {result_path}")
    if result.get("scenario") != scenario:
        raise RuntimeError(f"result scenario mismatch: {result_path}")
    if int(result.get("master_seed", -1)) != MASTER_SEED:
        raise RuntimeError(f"result master seed mismatch: {result_path}")
    if result.get("seeds") != envelope["seeds"]:
        raise RuntimeError(f"result component seeds mismatch: {result_path}")
    if result.get("seed_table_sha256") != context.seed_table_sha256:
        raise RuntimeError(f"result seed-table hash mismatch: {result_path}")
    if require_complete and result.get("status") != "complete":
        raise RuntimeError(f"non-complete v3.2 result is a hard stop: {result_path}")
    return {"claim_sha256": claim_hash, "result_sha256": sha256_file(result_path)}


def worker_stem(start: int, count: int) -> str:
    validate_shard(start, count)
    return f"worker_{start:03d}_{start + count - 1:03d}"


def worker_paths(stage_root: Path, start: int, count: int) -> tuple[Path, Path]:
    stem = worker_stem(start, count)
    return stage_root / f"{stem}.active", stage_root / f"{stem}.done"


def _worker_payload(
    context: FreezeContext,
    scenario: str,
    start: int,
    count: int,
    status: str,
) -> dict[str, Any]:
    return {
        "worker_schema": WORKER_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": context.manifest_sha256,
        "seed_table_sha256": context.seed_table_sha256,
        "master_seed": MASTER_SEED,
        "scenario": scenario,
        "start": int(start),
        "count": int(count),
        "status": status,
        "process_id": int(os.getpid()),
    }


def _validate_worker_payload(
    payload: dict[str, Any],
    context: FreezeContext,
    scenario: str,
    start: int,
    count: int,
    status: str,
) -> None:
    expected = _worker_payload(context, scenario, start, count, status)
    for field in (
        "worker_schema",
        "protocol_id",
        "manifest_sha256",
        "seed_table_sha256",
        "master_seed",
        "scenario",
        "start",
        "count",
        "status",
    ):
        if payload.get(field) != expected[field]:
            raise RuntimeError(f"worker sentinel mismatch: {scenario}/{start}/{field}")


def validate_shard_records(
    context: FreezeContext,
    scenario: str,
    start: int,
    count: int,
) -> list[dict[str, Any]]:
    validate_shard(start, count)
    records = []
    for replicate in range(start, start + count):
        hashes = validate_record_pair(context, scenario, replicate)
        records.append({"replicate": replicate, **hashes})
    return records


def validate_worker_done(
    context: FreezeContext,
    scenario: str,
    start: int,
    count: int,
) -> None:
    stage_root = context.paths.scenario_root(scenario)
    active, done = worker_paths(stage_root, start, count)
    if active.exists() or not done.is_file():
        raise RuntimeError(f"worker is not cleanly complete: {scenario}/{start}")
    payload = _load_json(done)
    _validate_worker_payload(payload, context, scenario, start, count, "done")
    records = validate_shard_records(context, scenario, start, count)
    registry_hash = sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if payload.get("record_registry_sha256") != registry_hash:
        raise RuntimeError(f"worker record registry drifted: {done}")


def reject_stage_temporary_pollution(stage_root: Path) -> None:
    """Treat interrupted atomic/freeze temporaries as an immutable hard stop."""

    if not stage_root.exists():
        return
    temporary = sorted(
        path
        for path in stage_root.iterdir()
        if path.name.startswith((".__v32_tmp_", ".__freeze_v32_"))
    )
    if temporary:
        raise RuntimeError(f"stale v3.2 temporary artifact is a hard stop: {temporary[0]}")


def validate_stage_complete(
    context: FreezeContext,
    scenario: str,
) -> list[dict[str, Any]]:
    validate_scenario_name(scenario)
    stage_root = context.paths.scenario_root(scenario)
    if not stage_root.is_dir():
        raise RuntimeError(f"v3.2 stage directory is absent: {stage_root}")
    reject_stage_temporary_pollution(stage_root)
    expected_done = {
        worker_paths(stage_root, start, count)[1].name for start, count in SHARDS
    }
    observed_workers = {
        path.name for path in stage_root.iterdir() if path.name.startswith("worker_")
    }
    if observed_workers != expected_done:
        raise RuntimeError(
            f"v3.2 stage worker sentinel set is polluted or incomplete: {scenario}"
        )
    expected_claims = {f"rep_{rep:04d}.claim" for rep in range(100)}
    expected_results = {f"rep_{rep:04d}.json" for rep in range(100)}
    expected_records = expected_claims | expected_results
    observed_records = {
        path.name for path in stage_root.iterdir() if path.name.startswith("rep_")
    }
    if observed_records != expected_records:
        raise RuntimeError(
            f"v3.2 stage record set contains missing, extra or noncanonical entries: {scenario}"
        )
    for start, count in SHARDS:
        validate_worker_done(context, scenario, start, count)
    observed_claims = {path.name for path in stage_root.glob("rep_*.claim")}
    observed_results = {path.name for path in stage_root.glob("rep_*.json")}
    if observed_claims != expected_claims or observed_results != expected_results:
        raise RuntimeError(f"v3.2 stage does not contain exactly 100 matched records: {scenario}")
    return [
        {"replicate": rep, **validate_record_pair(context, scenario, rep)}
        for rep in range(100)
    ]


def validate_aggregate_complete(context: FreezeContext, scenario: str) -> None:
    reject_aggregate_temporary_pollution(context.paths)
    name = "compiler_semantic" if scenario == "COMPILER_SEMANTIC" else scenario
    path = context.paths.aggregate_root / name / "v32_aggregate_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"v3.2 predecessor aggregate is absent: {scenario}")
    payload = _load_json(path)
    expected_schema = (
        "v32-compiler-aggregate-v1"
        if scenario == "COMPILER_SEMANTIC"
        else "v32-predictive-aggregate-v1"
    )
    required = {
        "aggregate_schema": expected_schema,
        "protocol_id": PROTOCOL_ID,
        "manifest_sha256": context.manifest_sha256,
        "seed_table_sha256": context.seed_table_sha256,
        "scenario": scenario,
        "n_claims": 100,
        "n_results": 100,
    }
    for field, expected in required.items():
        if payload.get(field) != expected:
            raise RuntimeError(f"v3.2 aggregate binding mismatch: {scenario}/{field}")
    records = validate_stage_complete(context, scenario)
    registry_hash = sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    if payload.get("input_record_registry_sha256") != registry_hash:
        raise RuntimeError(f"v3.2 aggregate input registry drifted: {scenario}")
    if payload.get("input_records") != records:
        raise RuntimeError(f"v3.2 aggregate input record list drifted: {scenario}")
    if scenario == "COMPILER_SEMANTIC":
        gate = payload.get("hard_gate") or {}
        expected_gate = {
            "n_complete": 100,
            "n_failed": 0,
            "false_exact_events_total": 0,
            "all_primary_observations_pass": True,
        }
        for field, expected in expected_gate.items():
            if gate.get(field) != expected:
                raise RuntimeError(f"v3.2 compiler aggregate gate failed: {field}")


def reject_aggregate_temporary_pollution(
    paths: ProtocolPaths,
    *,
    allowed: Path | None = None,
) -> None:
    """Reject every stale aggregate staging entry outside one active builder."""

    root = paths.aggregate_root
    if not root.exists():
        return
    allowed_resolved = None if allowed is None else allowed.resolve()
    temporary = sorted(
        path
        for path in root.iterdir()
        if path.name.startswith(".__tmp_")
        and (allowed_resolved is None or path.resolve() != allowed_resolved)
    )
    if temporary:
        raise RuntimeError(
            f"stale v3.2 aggregate temporary artifact is a hard stop: {temporary[0]}"
        )


def aggregate_temporary_path(
    paths: ProtocolPaths,
    *,
    process_id: int | None = None,
) -> Path:
    """Return the single short aggregate staging path for one strict stage."""

    pid = os.getpid() if process_id is None else int(process_id)
    return paths.aggregate_root / f".__tmp_{pid}"


def validate_predecessors(context: FreezeContext, scenario: str) -> None:
    validate_scenario_name(scenario)
    index = STAGE_ORDER.index(scenario)
    for predecessor in STAGE_ORDER[:index]:
        validate_stage_complete(context, predecessor)
        validate_aggregate_complete(context, predecessor)


def begin_worker(
    context: FreezeContext,
    scenario: str,
    start: int,
    count: int,
) -> str:
    validate_shard(start, count)
    validate_predecessors(context, scenario)
    stage_root = context.paths.scenario_root(scenario)
    stage_root.mkdir(parents=True, exist_ok=True)
    reject_stage_temporary_pollution(stage_root)
    active, done = worker_paths(stage_root, start, count)
    if done.exists():
        validate_worker_done(context, scenario, start, count)
        return "skip_done"
    if active.exists():
        raise RuntimeError(f"orphan/active v3.2 worker is a hard stop: {active}")
    for replicate in range(start, start + count):
        claim = context.paths.claim_path(scenario, replicate)
        result = context.paths.result_path(scenario, replicate)
        if claim.exists() or result.exists():
            raise RuntimeError(
                f"pre-existing v3.2 record without a done worker is pollution: {scenario}/{replicate}"
            )
    active_workers = list(stage_root.glob("worker_*.active"))
    if len(active_workers) >= 2:
        raise RuntimeError(f"v3.2 stage already has two active workers: {scenario}")
    payload = _worker_payload(context, scenario, start, count, "active")
    payload["started_utc"] = utc_now()
    _exclusive_json(active, payload)
    return "started"


def finish_worker(
    context: FreezeContext,
    scenario: str,
    start: int,
    count: int,
) -> None:
    stage_root = context.paths.scenario_root(scenario)
    active, done = worker_paths(stage_root, start, count)
    if not active.is_file() or done.exists():
        raise RuntimeError(f"worker sentinel state is invalid at close: {scenario}/{start}")
    active_payload = _load_json(active)
    _validate_worker_payload(active_payload, context, scenario, start, count, "active")
    records = validate_shard_records(context, scenario, start, count)
    registry_hash = sha256_bytes(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    payload = _worker_payload(context, scenario, start, count, "done")
    payload.update(
        completed_utc=utc_now(),
        n_records=len(records),
        record_registry_sha256=registry_hash,
    )
    atomic_create_json(done, payload)
    active.unlink()


def activate_base_runner(context: FreezeContext):
    """Bind the unchanged v3 scientific runner to the active v3.2 freeze."""

    import run_batch as runner

    runner.MASTER_SEED = MASTER_SEED
    runner.frozen_seed_table_sha256 = lambda: context.seed_table_sha256
    return runner
