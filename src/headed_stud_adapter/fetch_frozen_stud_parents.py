"""Download frozen headed-stud parents without opening archive contents."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "freeze_v1.json"
RAW = HERE / "raw"
SHORT_NAMES = {
    "nwc_solid_242": "nwc242",
    "profiled_deck_464": "deck464",
    "lwc_solid_90": "lwc90",
    "rac_scc_independent_27": "rac27",
}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(parent: dict[str, Any]) -> Path:
    return RAW / f"{SHORT_NAMES[parent['id']]}.zip"


def download(parent: dict[str, Any]) -> dict[str, Any]:
    output = destination(parent)
    partial = output.with_suffix(".zip.part")
    if output.is_file():
        return {
            "id": parent["id"],
            "path": str(output.relative_to(HERE)),
            "bytes": output.stat().st_size,
            "sha256": file_hash(output),
            "md5": file_hash(output, "md5"),
            "status": "verified_existing",
        }

    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        parent["download_url"],
        headers={"User-Agent": "R30-headed-stud-freeze/1.0"},
    )
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        status = int(getattr(response, "status", 200))
        if offset and status != 206:
            offset = 0
            partial.unlink(missing_ok=True)
        with partial.open("ab" if offset else "wb") as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)

    expected_size = parent.get("archive_bytes")
    expected_md5 = parent.get("archive_md5")
    actual_size = partial.stat().st_size
    actual_md5 = file_hash(partial, "md5")
    if expected_size is not None and actual_size != int(expected_size):
        raise RuntimeError(
            f"size mismatch for {parent['id']}: {actual_size}/{expected_size}"
        )
    if expected_md5 is not None and actual_md5 != str(expected_md5):
        raise RuntimeError(
            f"MD5 mismatch for {parent['id']}: {actual_md5}/{expected_md5}"
        )
    partial.replace(output)
    return {
        "id": parent["id"],
        "path": str(output.relative_to(HERE)),
        "bytes": actual_size,
        "sha256": file_hash(output),
        "md5": actual_md5,
        "status": "downloaded_and_verified",
        "runtime_seconds": float(time.perf_counter() - started),
    }


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("target_access_at_freeze") != "not_opened":
        raise RuntimeError("registry does not preserve the sealed-target boundary")
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for parent in registry["parents"]:
        row = download(parent)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    manifest = {
        "schema_version": "r30-headed-stud-download-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "registry_sha256": file_hash(REGISTRY),
        "total_bytes": int(sum(row["bytes"] for row in rows)),
        "all_verified": len(rows) == len(registry["parents"]),
        "targets_opened": False,
        "archives": rows,
    }
    (HERE / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
