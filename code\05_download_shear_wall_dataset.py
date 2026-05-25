"""
Download the open shear-wall dataset from Sujith Mangalathu's GitHub.

This is not the 2018 beam-column joint paper, whose database is described on the
author site but no direct download link was found. The 2020 shear-wall paper is
from the same author line and includes both data and code, making it a suitable
first executable reproduction target for P1.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


URL = (
    "https://raw.githubusercontent.com/sujithmangalathu/"
    "Shear-Wall-Failure-Mode/master/Shear_Wall_Database.xlsx"
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "datasets" / "mangalathu_2020_shear_wall"
OUT_FILE = OUT_DIR / "Shear_Wall_Database.xlsx"
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
                "# Mangalathu 2020 Shear Wall Dataset",
                "",
                f"- Source: {URL}",
                "- Repository: https://github.com/sujithmangalathu/Shear-Wall-Failure-Mode",
                "- Paper: Data-driven machine-learning-based seismic failure mode identification of reinforced concrete shear walls, Engineering Structures, 2020",
                "- DOI: 10.1016/j.engstruct.2019.110331",
                f"- Local file: `{OUT_FILE.name}`",
                f"- Size bytes: {OUT_FILE.stat().st_size}",
                f"- SHA256: `{digest}`",
                "",
                "Use: first true executable P1 reproduction case, because data and notebook are public.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] {OUT_FILE}")
    print(f"[OK] {MANIFEST}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
