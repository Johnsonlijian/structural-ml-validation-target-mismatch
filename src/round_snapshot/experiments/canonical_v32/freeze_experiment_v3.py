"""Create the post-audit v3 confirmatory source and seed freeze."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

CV_ROOT = Path(__file__).resolve().parent
ROUND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROUND_ROOT.parents[1]
METHOD_ROOT = ROUND_ROOT / "method"
FREEZE_ROOT = CV_ROOT / "freeze_v3"
PRESPEC = ROUND_ROOT / "experiments" / "confirmatory_prespecification_v3.md"
CONFIRMATORY_ROOT = CV_ROOT / "confirmatory_v3"
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

if str(CV_ROOT) not in sys.path:
    sys.path.insert(0, str(CV_ROOT))

from run_batch import MASTER_SEED, SCHEMA_VERSION, replicate_seeds  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


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


def scenario_names() -> list[str]:
    core = [
        f"CORE_{effect}_{covariate}_{noise}"
        for effect in ("0", "0.6", "1.2", "2.4")
        for covariate in ("0", "0.9", "1.8")
        for noise in ("0.35", "1.0")
    ]
    stress = [
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "S9_missing10",
        "S9_missing30",
        "S9_corrupt5",
        "S10",
    ]
    return core + stress + ["COMPILER_SEMANTIC"]


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
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario in scenario_names():
            for replicate in range(100):
                seeds = replicate_seeds(scenario, replicate)
                writer.writerow(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "master_seed": MASTER_SEED,
                        **{f"{name}_seed": value for name, value in seeds.items()},
                    }
                )
    temporary.replace(path)
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
        "n_scenarios": len(scenario_names()),
        "replicates_per_scenario": 100,
        "n_rows": len(scenario_names()) * 100,
    }


def hash_registry(paths: Iterable[Path]) -> list[dict[str, Any]]:
    output = []
    for path in paths:
        output.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return output


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
            CV_ROOT / "aggregate_batch.py",
            CV_ROOT / "run_compiler_suite.py",
            CV_ROOT / "aggregate_compiler_suite.py",
            CV_ROOT / "freeze_experiment_v3.py",
            PRESPEC,
            CV_ROOT / "CONFIRMATORY_EXECUTION_DEVIATION_001.md",
            CV_ROOT / "CONFIRMATORY_SEMANTIC_AUDIT_002.md",
        ]
    )
    return paths


def excluded_registry() -> list[dict[str, Any]]:
    explicit = [
        ROUND_ROOT / "experiments" / "poc_synthetic",
        CV_ROOT / "confirmatory",
        CV_ROOT / "confirmatory_v21",
        CV_ROOT / "confirmatory_v211",
    ]
    prefixes = (
        "pilot_",
        "t0",
        "t_",
        "schema_",
        "compiler_suite_",
    )
    explicit.extend(
        path
        for path in sorted(CV_ROOT.iterdir())
        if path.is_dir() and path.name.startswith(prefixes)
    )
    output = []
    seen: set[Path] = set()
    for directory in explicit:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        files = (
            sorted(path for path in directory.rglob("*") if path.is_file())
            if directory.is_dir()
            else []
        )
        output.append(
            {
                "path": str(directory.relative_to(PROJECT_ROOT)),
                "exists": directory.is_dir(),
                "excluded_from_v3_inference": True,
                "n_files": len(files),
                "files": hash_registry(files),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    FREEZE_ROOT.mkdir(parents=True, exist_ok=True)

    seed_table = write_seed_table(FREEZE_ROOT / "confirmatory_seed_table.csv")
    checks = {
        "pytest": run_checked(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            METHOD_ROOT,
        ),
        "independent_oracle": run_checked(
            [sys.executable, "independent_oracle_check.py"],
            METHOD_ROOT,
        ),
        "compiler_smoke": run_checked(
            [
                sys.executable,
                "run_compiler_suite.py",
                "--start",
                "999",
                "--count",
                "1",
                "--output",
                str(FREEZE_ROOT / "compiler_smoke"),
            ],
            CV_ROOT,
        ),
    }
    if VENV_PYTHON.is_file():
        checks["datasail_smoke"] = run_checked(
            [str(VENV_PYTHON), "datasail_smoke_test.py"],
            METHOD_ROOT,
        )
        package_lock = run_checked(
            [str(VENV_PYTHON), "-m", "pip", "freeze", "--all"],
            PROJECT_ROOT,
        )
        if package_lock["passed"]:
            (FREEZE_ROOT / "environment_r30ds_pip_freeze.txt").write_text(
                package_lock["stdout"],
                encoding="utf-8",
            )
    else:
        checks["datasail_smoke"] = {
            "passed": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"missing interpreter: {VENV_PYTHON}",
            "command": [],
            "cwd": str(PROJECT_ROOT),
        }

    sources = hash_registry(required_sources())
    artifacts = hash_registry(
        [
            DATASAIL_WHEEL,
            DATASAIL_PATCH,
            METHOD_ROOT / "independent_oracle_check_result.json",
            METHOD_ROOT / "datasail_smoke_test_result.json",
            FREEZE_ROOT / "compiler_smoke" / "rep_0999.json",
            FREEZE_ROOT / "environment_r30ds_pip_freeze.txt",
        ]
    )
    exclusions = excluded_registry()
    confirmatory_files = (
        [
            path
            for pattern in ("rep_*.json", "rep_*.claim")
            for path in CONFIRMATORY_ROOT.rglob(pattern)
        ]
        if CONFIRMATORY_ROOT.exists()
        else []
    )
    clean_root = not confirmatory_files
    all_required = all(item["exists"] for item in sources + artifacts)
    all_checks = all(item["passed"] for item in checks.values())
    requested_status = "final" if args.final else "candidate"
    status = (
        requested_status
        if all_required and all_checks and clean_root
        else "failed"
    )
    manifest = {
        "schema_version": "confirmatory-freeze-v3",
        "status": status,
        "requested_status": requested_status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": MASTER_SEED,
        "output_schema_version": SCHEMA_VERSION,
        "compiler_schema_version": "compiler-semantic-v3",
        "contract_schema_version": "deployment-contract-v3",
        "prespecification": {
            "path": str(PRESPEC.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(PRESPEC) if PRESPEC.is_file() else None,
        },
        "seed_table": seed_table,
        "checks": checks,
        "sources": sources,
        "artifacts": artifacts,
        "excluded_registry_path": str(
            (FREEZE_ROOT / "excluded_pilot_registry.json").relative_to(
                PROJECT_ROOT
            )
        ),
        "confirmatory_root": str(CONFIRMATORY_ROOT.relative_to(PROJECT_ROOT)),
        "confirmatory_root_clean": clean_root,
        "confirmatory_files_found": [
            str(path.relative_to(PROJECT_ROOT)) for path in confirmatory_files
        ],
        "all_required_exist": all_required,
        "all_checks_pass": all_checks,
    }
    atomic_json(FREEZE_ROOT / "excluded_pilot_registry.json", exclusions)
    atomic_json(FREEZE_ROOT / "source_hashes.json", sources)
    atomic_json(FREEZE_ROOT / "freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

