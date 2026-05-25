from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "mendeley_beam_column_joint"

DATASETS = {
    "10.17632-8ndgpm7zw7.1": {
        "doi": "10.17632/8ndgpm7zw7.1",
        "dataset_id": "8ndgpm7zw7",
        "version": 1,
        "title": "Dataset of Exterior Reinforced Concrete Beam-Column Joint Experiments",
        "license": "CC BY-NC 4.0",
        "filename": "Teklewoin_Joint_Dataset.xlsx",
        "url": "https://data.mendeley.com/public-files/datasets/8ndgpm7zw7/files/7b39f6f8-31fd-4969-b7f5-e5f868ec0405/file_downloaded",
        "expected_sha256": "737f2d34875de2c5b73bcf2d0f4db47de9494730d5d1e7ed5d1ff0c9f15b26a8",
        "expected_size": 37550,
    },
    "10.17632-rbhfnz32sy.1": {
        "doi": "10.17632/rbhfnz32sy.1",
        "dataset_id": "rbhfnz32sy",
        "version": 1,
        "title": "Dataset for joint shear strength of beam-column joints subject to cyclic loading",
        "license": "CC BY 4.0",
        "filename": "Salem_Beam_Column_Joint_Cyclic_Shear.xlsx",
        "url": "https://data.mendeley.com/public-files/datasets/rbhfnz32sy/files/73b93c76-4a9b-4985-9a10-2d1d5df1224f/file_downloaded",
        "expected_sha256": "35fb8f3815bff92cf194f19d388f1b99b291aeec4149d479a784429f431f69e8",
        "expected_size": 28593,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Mendeley Data public API download URLs",
        "datasets": [],
    }

    for key, meta in DATASETS.items():
        dataset_dir = OUT / key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target = dataset_dir / meta["filename"]
        download(meta["url"], target)
        observed_sha = sha256(target)
        observed_size = target.stat().st_size
        ok = observed_sha == meta["expected_sha256"] and observed_size == meta["expected_size"]
        record = {
            **meta,
            "local_path": str(target.relative_to(ROOT)),
            "observed_sha256": observed_sha,
            "observed_size": observed_size,
            "verified": ok,
        }
        if not ok:
            raise RuntimeError(f"Download verification failed for {key}: {record}")
        manifest["datasets"].append(record)

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Mendeley Beam-Column Joint Data Manifest",
        "",
        f"Created UTC: `{manifest['created_utc']}`",
        "",
    ]
    for record in manifest["datasets"]:
        lines.extend(
            [
                f"## {record['doi']}",
                "",
                f"- Title: {record['title']}",
                f"- License: {record['license']}",
                f"- Local path: `{record['local_path']}`",
                f"- SHA256: `{record['observed_sha256']}`",
                f"- Size: {record['observed_size']} bytes",
                f"- Verified: {record['verified']}",
                "",
            ]
        )
    (OUT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
