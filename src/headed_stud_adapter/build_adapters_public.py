"""Path-neutral public wrapper for the frozen headed-stud adapter builder.

Place this file beside ``build_adapters.py`` and
``public_input_registry.json``.  The wrapper replaces the original
execution-receipt byte locks (which contain timestamps and runtimes) with
stable scientific locks: exact source archives, frozen protocol files, the
manual overlap decision file, and the deterministic adjudicated source table.

The generated canonical row tables remain local under ``derived_private`` and
must not be committed or redistributed by the paper repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import build_adapters as implementation


HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "public_input_registry.json"
EXPECTED = {
    "freeze_v1.json": "463e0b039a01b748e1b479bb4584767c631053cd6c28aa4a7b4d15dbc719bce6",
    "adapter_freeze_v1.json": "42ac1a0bed31307884ce19bc38eb629cff198cd67c06b6ea1347af1f3d919515",
    "evaluation_plan_v1.md": "9dabb8178a8f1d96fe6278adb8ce4e38d9e1e0e8dc7699c12a4a001b00d09a76",
    "manual_overlap_review_v1.json": "d8a196cb33a7741395e83ab932495c887f1490d1dae9c08142eb9fdf819188d9",
    "source_audit_adjudicated.csv": "6607e1a319f4bdf5056f77dd17c5ad86b46d1ec98007d352720ef9c6eeb30565",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_public_verification() -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for filename, expected in EXPECTED.items():
        path = HERE / filename
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"stable frozen input mismatch: {filename}")

    for item in registry["archives"]:
        path = HERE / "raw" / item["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"downloaded archive missing: {item['filename']}")
        if path.stat().st_size != int(item["bytes"]):
            raise RuntimeError(f"archive size mismatch: {item['filename']}")
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"archive hash mismatch: {item['filename']}")

    freeze = json.loads((HERE / "adapter_freeze_v1.json").read_text(encoding="utf-8"))
    if freeze.get("target_access_status") != "sealed_at_freeze":
        raise RuntimeError("adapter protocol no longer records its original sealed status")
    return freeze


def main() -> None:
    if implementation.HERE.resolve() != HERE.resolve():
        raise RuntimeError("build_adapters.py must be beside this wrapper")
    implementation.verify_freeze = stable_public_verification
    implementation.main()


if __name__ == "__main__":
    main()
