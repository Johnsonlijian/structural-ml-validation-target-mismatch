"""
Download the UCI Concrete Compressive Strength dataset.

This dataset is used by Nguyen et al. 2021, "Efficient machine learning
models for prediction of concrete strengths" (Construction and Building
Materials, DOI 10.1016/j.conbuildmat.2020.120950).
"""

from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path


URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "datasets" / "uci_concrete_compressive_strength"
ZIP_FILE = OUT_DIR / "concrete_compressive_strength.zip"
MANIFEST = OUT_DIR / "MANIFEST.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_FILE.exists():
        print(f"[download] {URL}")
        urllib.request.urlretrieve(URL, ZIP_FILE)
    else:
        print(f"[skip] exists: {ZIP_FILE}")

    with zipfile.ZipFile(ZIP_FILE) as zf:
        zf.extractall(OUT_DIR)
        files = sorted(zf.namelist())

    lines = [
        "# UCI Concrete Compressive Strength Dataset",
        "",
        "- Dataset: Concrete Compressive Strength",
        "- UCI dataset ID: 165",
        "- Dataset DOI: 10.24432/C5PK67",
        "- License: CC-BY-4.0",
        "- Source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength",
        f"- Download URL: {URL}",
        f"- Local zip: `{ZIP_FILE.name}`",
        f"- Zip size bytes: {ZIP_FILE.stat().st_size}",
        f"- Zip SHA256: `{sha256(ZIP_FILE)}`",
        "",
        "## Extracted files",
        "",
    ]
    for name in files:
        path = OUT_DIR / name
        lines.append(f"- `{name}` · {path.stat().st_size} bytes · SHA256 `{sha256(path)}`")

    lines.extend(
        [
            "",
            "Use: third P1 real-data reproduction candidate; regression target is concrete compressive strength.",
            "Grouping hypothesis: the same mix proportions can appear at multiple ages, so random folds may leak mix-family information.",
        ]
    )
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {ZIP_FILE}")
    print(f"[OK] {MANIFEST}")


if __name__ == "__main__":
    main()
