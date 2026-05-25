"""
Download Lantsoght 2019 SFRC shear database from Zenodo.

This dataset underlies multiple SFRC shear-strength ML papers, including the
Rahman et al. 2021 Engineering Structures paper. It is CC-BY-4.0 and public,
making it a strong candidate for the first real regression reproduction.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


URL = "https://zenodo.org/api/records/2578061/files/database%20SFRC.xlsx/content"

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "datasets" / "lantsoght_2019_sfrc_shear"
OUT_FILE = OUT_DIR / "database_SFRC.xlsx"
MANIFEST = OUT_DIR / "MANIFEST.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not OUT_FILE.exists():
        print(f"[download] {URL}")
        urllib.request.urlretrieve(URL, OUT_FILE)
    else:
        print(f"[skip] exists: {OUT_FILE}")

    digest = sha256(OUT_FILE)
    MANIFEST.write_text(
        "\n".join(
            [
                "# Lantsoght 2019 SFRC Shear Database",
                "",
                "- Dataset: Database of experiments on SFRC beams without stirrups failing in shear",
                "- DOI: 10.5281/zenodo.2578061",
                "- License: CC-BY-4.0",
                f"- Source: {URL}",
                f"- Local file: `{OUT_FILE.name}`",
                f"- Size bytes: {OUT_FILE.stat().st_size}",
                f"- SHA256: `{digest}`",
                "",
                "Use: second P1 real-data reproduction candidate; regression target is shear capacity.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] {OUT_FILE}")
    print(f"[OK] {MANIFEST}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
