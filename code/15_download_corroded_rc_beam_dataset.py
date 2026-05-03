"""
Download public data for corroded RC beam moment-capacity ML example.

Repository:
    https://github.com/bma114/corroded-RC-beam-moment-capacity

Zenodo database:
    Monotonic Flexural Testing of Corroded Reinforced Concrete Beams Database
    DOI: 10.5281/zenodo.8062007
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


FILES = {
    "README.md": "https://raw.githubusercontent.com/bma114/corroded-RC-beam-moment-capacity/main/README.md",
    "Data.csv": "https://raw.githubusercontent.com/bma114/corroded-RC-beam-moment-capacity/main/Data.csv",
    "run.py": "https://raw.githubusercontent.com/bma114/corroded-RC-beam-moment-capacity/main/run.py",
    "Functions.py": "https://raw.githubusercontent.com/bma114/corroded-RC-beam-moment-capacity/main/Functions.py",
    "Zenodo_Database.csv": "https://zenodo.org/api/records/8062007/files/Database.csv/content",
    "Database_Key.pdf": "https://zenodo.org/api/records/8062007/files/Database%20Key.pdf/content",
}

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "datasets" / "corroded_rc_beam_moment_capacity"
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
        "# Corroded RC Beam Moment Capacity Dataset",
        "",
        "- Repository: https://github.com/bma114/corroded-RC-beam-moment-capacity",
        "- Zenodo DOI: 10.5281/zenodo.8062007",
        "- Description: 804 corroded/uncorroded RC beams from 54 experimental programs.",
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
            "Use: fifth P1 real-data reproduction candidate; target is residual bending moment capacity.",
            "Critical requirement: inspect for experimental-program or source-reference labels before reproduction.",
        ]
    )
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {MANIFEST}")


if __name__ == "__main__":
    main()
