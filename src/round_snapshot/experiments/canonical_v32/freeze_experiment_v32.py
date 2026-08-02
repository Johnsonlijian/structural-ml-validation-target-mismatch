"""Build the single-creation final freeze for confirmatory protocol v3.2.

The builder is intentionally self-contained: it does not import any historical
freeze builder.  It stages all files outside ``freeze_v32`` and promotes the
directory only after checks pass and the canonical result/aggregate roots are
still empty.  It must not be run until execution is separately authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from protocol_v32 import (
    COMPILER_SCHEMA,
    CONTRACT_SCHEMA,
    FREEZE_SCHEMA,
    MASTER_SEED,
    PREDICTIVE_SCHEMA,
    PROTOCOL_ID,
    SEED_NAMES,
    SHARDS,
    STAGE_ORDER,
    ProtocolPaths,
    aggregate_temporary_path,
    atomic_temporary_path,
    derive_seeds,
)


PATHS = ProtocolPaths.canonical()
CV_ROOT = PATHS.cv_root
PROJECT_ROOT = PATHS.project_root
ROUND_ROOT = CV_ROOT.parents[1]
EXPERIMENTS_ROOT = CV_ROOT.parent
METHOD_ROOT = ROUND_ROOT / "method"
FREEZE_ROOT = PATHS.freeze_root
CONFIRMATORY_ROOT = PATHS.confirmatory_root
AGGREGATE_ROOT = PATHS.aggregate_root
VENV_PYTHON = PROJECT_ROOT / "r30ds" / "Scripts" / "python.exe"
DATASAIL_WHEEL = METHOD_ROOT / "vendor_wheels" / "datasail-1.3.0-py3-none-any.whl"
DATASAIL_PATCH = (
    PROJECT_ROOT
    / "r30ds"
    / "Lib"
    / "site-packages"
    / "datasail"
    / "cluster"
    / "wlk.py"
)
V3_SCIENTIFIC_PRESPEC_SHA256 = (
    "c664f66742966c82b638463b8139e251e0bcdd5508209151982d7685adb44034"
)
V32_AST_FILES = (
    "protocol_v32.py",
    "run_batch_v32.py",
    "run_compiler_suite_v32.py",
    "aggregate_batch_v32.py",
    "aggregate_compiler_suite_v32.py",
    "freeze_experiment_v32.py",
    "test_v32_protocol.py",
)
MAX_WINDOWS_PATH = 260
MAX_PROCESS_ID = 4294967295


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = freeze_temporary_path(path)
    if path.exists() or temporary.exists():
        raise RuntimeError(f"v3.2 freeze file already exists: {path}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def freeze_temporary_path(path: Path, *, process_id: int | None = None) -> Path:
    """Return a short same-directory freeze temp safe below MAX_PATH."""

    pid = os.getpid() if process_id is None else int(process_id)
    return path.with_name(f".__freeze_v32_{pid}")


def run_checked(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(cwd),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-20000:],
        "stderr": completed.stderr[-20000:],
        "passed": completed.returncode == 0,
    }


def ast_check_command(files: tuple[str, ...] = V32_AST_FILES) -> list[str]:
    """Build a no-bytecode AST check safe for long Windows paths."""

    script = (
        "import ast\n"
        "from pathlib import Path\n"
        f"files = {tuple(files)!r}\n"
        "for name in files:\n"
        "    ast.parse(Path(name).read_text(encoding='utf-8'), filename=name)\n"
        "print(f'AST_PASS {len(files)}')\n"
    )
    return [sys.executable, "-B", "-c", script]


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def registry_entry(path: Path, *, logical_path: Path | None = None) -> dict[str, Any]:
    return {
        "path": relative(logical_path or path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def hash_registry(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [registry_entry(path) for path in paths]


def required_sources() -> list[Path]:
    paths = sorted((METHOD_ROOT / "provenance_cut").glob("*.py"))
    paths.extend(sorted((METHOD_ROOT / "tests").glob("*.py")))
    paths.extend(
        [
            METHOD_ROOT / "independent_oracle_check.py",
            METHOD_ROOT / "datasail_smoke_test.py",
            METHOD_ROOT / "formal_specification_v3.md",
            CV_ROOT / "independent_contract_oracle.py",
            CV_ROOT / "run_batch.py",
            CV_ROOT / "run_compiler_suite.py",
            CV_ROOT / "aggregate_batch.py",
            CV_ROOT / "aggregate_compiler_suite.py",
            CV_ROOT / "freeze_experiment_v3.py",
            CV_ROOT / "protocol_v32.py",
            CV_ROOT / "run_batch_v32.py",
            CV_ROOT / "run_compiler_suite_v32.py",
            CV_ROOT / "aggregate_batch_v32.py",
            CV_ROOT / "aggregate_compiler_suite_v32.py",
            CV_ROOT / "freeze_experiment_v32.py",
            CV_ROOT / "test_v32_protocol.py",
            CV_ROOT / "run_pipeline_v32.ps1",
            CV_ROOT / "RUN_PIPELINE_V32.md",
            EXPERIMENTS_ROOT / "confirmatory_prespecification_v3.md",
            EXPERIMENTS_ROOT / "confirmatory_prespecification_v32.md",
            CV_ROOT / "CONFIRMATORY_EXECUTION_DEVIATION_001.md",
            CV_ROOT / "CONFIRMATORY_SEMANTIC_AUDIT_002.md",
            CV_ROOT / "CONFIRMATORY_EXECUTION_DEVIATION_003.md",
            CV_ROOT / "CONFIRMATORY_EXECUTION_DEVIATION_004.md",
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def excluded_roots() -> list[Path]:
    explicit = [
        CV_ROOT / "confirmatory",
        CV_ROOT / "confirmatory_v21",
        CV_ROOT / "confirmatory_v211",
        CV_ROOT / "confirmatory_v3",
        CV_ROOT / "confirmatory_v31",
        ROUND_ROOT / "experiments" / "poc_synthetic",
    ]
    prefixes = ("pilot_", "t0", "t_", "schema_", "compiler_suite_")
    explicit.extend(
        path
        for path in sorted(CV_ROOT.iterdir())
        if path.is_dir() and path.name.startswith(prefixes)
    )
    output: list[Path] = []
    seen: set[Path] = set()
    for path in explicit:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            output.append(path)
    return output


def build_exclusion_registry() -> list[dict[str, Any]]:
    output = []
    for directory in excluded_roots():
        files = (
            sorted(path for path in directory.rglob("*") if path.is_file())
            if directory.is_dir()
            else []
        )
        output.append(
            {
                "path": relative(directory),
                "exists": directory.is_dir(),
                "excluded_from_v32_inference": True,
                "n_files": len(files),
                "files": hash_registry(files),
            }
        )
    return output


def root_entries(root: Path) -> list[Path]:
    return list(root.iterdir()) if root.exists() else []


def assert_result_roots_clean() -> None:
    if root_entries(CONFIRMATORY_ROOT):
        raise RuntimeError("confirmatory_v32 must be absent or empty before freeze")
    if root_entries(AGGREGATE_ROOT):
        raise RuntimeError("a32 must be absent or empty before freeze")


def assert_v3_scientific_prespecification(path: Path | None = None) -> None:
    scientific = path or (
        EXPERIMENTS_ROOT / "confirmatory_prespecification_v3.md"
    )
    if (
        not scientific.is_file()
        or sha256_file(scientific) != V3_SCIENTIFIC_PRESPEC_SHA256
    ):
        raise RuntimeError(
            "v3 scientific prespecification hash differs before v3.2 freeze writes"
        )


def assert_clean_before_writes(staging: Path) -> None:
    if FREEZE_ROOT.exists():
        raise RuntimeError("freeze_v32 already exists; final freeze cannot be rebuilt")
    stale = list(CV_ROOT.glob(".__f32_*")) + list(
        CV_ROOT.glob("freeze_v32.__building_*")
    )
    if stale:
        raise RuntimeError(f"stale v3.2 freeze staging directory exists: {stale[0]}")
    if staging.exists():
        raise RuntimeError(f"v3.2 freeze staging path already exists: {staging}")
    assert_result_roots_clean()


def assert_windows_path_budget() -> None:
    """Fail before staging if any canonical final/temp exceeds MAX_PATH."""

    scenario = max(STAGE_ORDER, key=lambda name: (len(name), name))
    staging = CV_ROOT / f".__f32_{MAX_PROCESS_ID}"
    aggregate_temp = aggregate_temporary_path(
        PATHS, process_id=MAX_PROCESS_ID
    )
    freeze_target = staging / "freeze_manifest.json"
    candidates = [
        PATHS.result_path(scenario, 99),
        PATHS.claim_path(scenario, 99),
        PATHS.scenario_root(scenario) / "worker_050_099.active",
        PATHS.scenario_root(scenario) / "worker_050_099.done",
        AGGREGATE_ROOT / scenario / "datasail_solver_status_summary.csv",
        AGGREGATE_ROOT / scenario / "v32_aggregate_manifest.json",
        aggregate_temp / "datasail_solver_status_summary.csv",
        FREEZE_ROOT / "environment_r30ds_pip_freeze.txt",
        staging / "environment_r30ds_pip_freeze.txt",
        staging / "excluded_pilot_registry.json",
        staging / "confirmatory_seed_table.csv",
        freeze_target,
        freeze_temporary_path(freeze_target, process_id=MAX_PROCESS_ID),
        atomic_temporary_path(
            PATHS.result_path(scenario, 99), process_id=MAX_PROCESS_ID
        ),
    ]
    over = [(len(str(path)), path) for path in candidates if len(str(path)) >= MAX_WINDOWS_PATH]
    if over:
        length, path = max(over, key=lambda item: item[0])
        raise RuntimeError(f"v3.2 canonical path exceeds MAX_PATH ({length}): {path}")


def write_seed_table(path: Path) -> dict[str, Any]:
    fields = [
        "scenario",
        "replicate",
        "master_seed",
        "generator_seed",
        "split_seed",
        "learner_seed",
        "datasail_seed",
    ]
    with path.open("x", newline="", encoding="utf-8") as handle:
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
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "path": relative(FREEZE_ROOT / path.name),
        "sha256": sha256_file(path),
        "n_scenarios": len(STAGE_ORDER),
        "replicates_per_scenario": 100,
        "n_rows": len(STAGE_ORDER) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", required=True)
    args = parser.parse_args()
    if not args.final:
        raise RuntimeError("v3.2 supports only a single final freeze")
    staging = CV_ROOT / f".__f32_{os.getpid()}"
    assert_clean_before_writes(staging)
    assert_v3_scientific_prespecification()
    assert_windows_path_budget()
    staging.mkdir(parents=False, exist_ok=False)

    checks = {
        "ast": run_checked(
            ast_check_command(),
            CV_ROOT,
        ),
        "pytest": run_checked(
            [
                sys.executable,
                "-m",
                "pytest",
                str(METHOD_ROOT / "tests"),
                str(CV_ROOT / "test_v32_protocol.py"),
                "-q",
            ],
            PROJECT_ROOT,
        ),
    }
    package_lock = None
    if VENV_PYTHON.is_file():
        checks["datasail_import"] = run_checked(
            [
                str(VENV_PYTHON),
                "-c",
                (
                    "import cvxpy, datasail, importlib.metadata; "
                    "assert importlib.metadata.version('datasail') == '1.3.0'; "
                    "assert 'SCIP' in cvxpy.installed_solvers()"
                ),
            ],
            PROJECT_ROOT,
        )
        package_lock = run_checked(
            [str(VENV_PYTHON), "-m", "pip", "freeze", "--all"],
            PROJECT_ROOT,
        )
        checks["package_lock"] = package_lock
    else:
        checks["datasail_import"] = {
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"missing v3.2 interpreter: {VENV_PYTHON}",
            "command": [],
            "cwd": str(PROJECT_ROOT),
        }

    seed_table = write_seed_table(staging / "confirmatory_seed_table.csv")
    if package_lock and package_lock["passed"]:
        (staging / "environment_r30ds_pip_freeze.txt").write_text(
            package_lock["stdout"], encoding="utf-8", newline="\n"
        )

    sources = hash_registry(required_sources())
    if not all(item["exists"] for item in sources):
        raise RuntimeError("one or more v3.2 frozen source dependencies are missing")
    atomic_json(staging / "source_hashes.json", sources)
    exclusions = build_exclusion_registry()
    atomic_json(staging / "excluded_pilot_registry.json", exclusions)

    artifacts = [
        registry_entry(DATASAIL_WHEEL),
        registry_entry(DATASAIL_PATCH),
        registry_entry(METHOD_ROOT / "independent_oracle_check_result.json"),
        registry_entry(METHOD_ROOT / "datasail_smoke_test_result.json"),
        registry_entry(
            staging / "environment_r30ds_pip_freeze.txt",
            logical_path=FREEZE_ROOT / "environment_r30ds_pip_freeze.txt",
        ),
    ]
    all_checks = all(bool(item.get("passed")) for item in checks.values())
    all_artifacts = all(bool(item.get("exists")) for item in artifacts)
    if not all_checks or not all_artifacts:
        raise RuntimeError("v3.2 freeze checks or artifacts did not pass")

    if FREEZE_ROOT.exists():
        raise RuntimeError("freeze_v32 appeared during freeze construction")
    assert_result_roots_clean()
    manifest = {
        "schema_version": FREEZE_SCHEMA,
        "status": "final",
        "protocol_id": PROTOCOL_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": MASTER_SEED,
        "output_schema_version": PREDICTIVE_SCHEMA,
        "compiler_schema_version": COMPILER_SCHEMA,
        "contract_schema_version": CONTRACT_SCHEMA,
        "freeze_root": relative(FREEZE_ROOT),
        "confirmatory_root": relative(CONFIRMATORY_ROOT),
        "aggregate_root": relative(AGGREGATE_ROOT),
        "confirmatory_root_clean": True,
        "confirmatory_files_found": [],
        "stage_order": list(STAGE_ORDER),
        "shards": [{"start": start, "count": count} for start, count in SHARDS],
        "prespecification": {
            "path": relative(EXPERIMENTS_ROOT / "confirmatory_prespecification_v32.md"),
            "sha256": sha256_file(
                EXPERIMENTS_ROOT / "confirmatory_prespecification_v32.md"
            ),
        },
        "scientific_prespecification": {
            "path": relative(EXPERIMENTS_ROOT / "confirmatory_prespecification_v3.md"),
            "sha256": sha256_file(
                EXPERIMENTS_ROOT / "confirmatory_prespecification_v3.md"
            ),
        },
        "seed_table": seed_table,
        "source_registry": {
            "path": relative(FREEZE_ROOT / "source_hashes.json"),
            "sha256": sha256_file(staging / "source_hashes.json"),
        },
        "sources": sources,
        "artifacts": artifacts,
        "excluded_registry": {
            "path": relative(FREEZE_ROOT / "excluded_pilot_registry.json"),
            "sha256": sha256_file(staging / "excluded_pilot_registry.json"),
        },
        "checks": checks,
        "all_required_exist": True,
        "all_checks_pass": True,
    }
    atomic_json(staging / "freeze_manifest.json", manifest)
    manifest_hash = sha256_file(staging / "freeze_manifest.json")
    with (staging / "freeze_manifest.sha256").open(
        "x", encoding="ascii", newline="\n"
    ) as handle:
        handle.write(manifest_hash + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    if FREEZE_ROOT.exists() or root_entries(CONFIRMATORY_ROOT) or root_entries(
        AGGREGATE_ROOT
    ):
        raise RuntimeError("v3.2 roots changed during freeze construction")
    staging.rename(FREEZE_ROOT)
    print(json.dumps({"status": "final", "manifest_sha256": manifest_hash}))


if __name__ == "__main__":
    main()
