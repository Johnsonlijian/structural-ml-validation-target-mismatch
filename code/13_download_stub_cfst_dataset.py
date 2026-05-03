"""
Download public data for the Stub-CFST Scientific Reports 2024 paper.

Paper:
    "Prediction of the axial compression capacity of stub CFST columns using
    machine learning techniques", Scientific Reports 2024.

Repository:
    https://github.com/kmegahed/Stub-CFST-Machine-learning
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


FILES = {
    "README.md": "https://raw.githubusercontent.com/kmegahed/Stub-CFST-Machine-learning/main/README.md",
    "PC.xlsx": "https://raw.githubusercontent.com/kmegahed/Stub-CFST-Machine-learning/main/PC.xlsx",
    "PDS.xlsx": "https://raw.githubusercontent.com/kmegahed/Stub-CFST-Machine-learning/main/PDS.xlsx",
    "PR.xlsx": "https://raw.githubusercontent.com/kmegahed/Stub-CFST-Machine-learning/main/PR.xlsx",
    "SR_DS.xlsx": "https://raw.githubusercontent.com/kmegahed/Stub-CFST-Machine-learning/main/SR_DS.xlsx",
    "supplementary_data.zip": "https://raw.githubusercontent.com/kmegahed/Stub-CFST-Machine-learning/main/supplementary%20data.zip",
}

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "datasets" / "stub_cfst_scirep_2024"
MANIFEST = OUT_DIR / "MANIFEST.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stub-CFST Scientific Reports 2024 Dataset",
        "",
        "- Paper: Prediction of the axial compression capacity of stub CFST columns using machine learning techniques",
        "- Journal: Scientific Reports, 2024",
        "- Article URL: https://www.nature.com/articles/s41598-024-53352-1",
        "- Repository: https://github.com/kmegahed/Stub-CFST-Machine-learning",
        "",
        "## Files",
        "",
    ]

    for name, url in FILES.items():
        path = OUT_DIR / name
        if not path.exists():
            print(f"[download] {name}")
            urllib.request.urlretrieve(url, path)
        else:
            print(f"[skip] {name}")
        lines.append(f"- `{name}` · {path.stat().st_size} bytes · SHA256 `{sha256(path)}` · {url}")

    lines.extend(
        [
            "",
            "Use: fourth P1 real-data reproduction candidate; regression target is axial compression capacity or strength index.",
            "Next diagnostic: inspect workbook columns for section type, source/dataset labels, and target variables.",
        ]
    )
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {MANIFEST}")


if __name__ == "__main__":
    main()
