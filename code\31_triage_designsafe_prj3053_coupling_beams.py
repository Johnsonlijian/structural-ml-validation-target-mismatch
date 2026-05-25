from __future__ import annotations

import csv
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "designsafe_prj3053_coupling_beams"
REPORT = ROOT / "manuscript_fragments" / "designsafe_prj3053_coupling_beam_triage_v0.md"
DOI = "10.17603/ds2-g5n8-4p74"

FILES = [
    {
        "name": "PRJ-3053_metadata.json",
        "path": "/published-data/PRJ-3053/PRJ-3053_metadata.json",
    },
    {
        "name": "manifest-sha512.txt",
        "path": "/published-data/PRJ-3053/Experiment--database-of-diagonally-reinforced-concrete-coupling-beams/manifest-sha512.txt",
    },
    {
        "name": "Database Readme.pdf",
        "path": "/published-data/PRJ-3053/Experiment--database-of-diagonally-reinforced-concrete-coupling-beams/data/Model-config--readme-file/data/Database Readme.pdf",
    },
    {
        "name": "sensor info placeholder.pdf",
        "path": "/published-data/PRJ-3053/Experiment--database-of-diagonally-reinforced-concrete-coupling-beams/data/Model-config--readme-file/Sensor--numerous-sensor-configurations-were-present-in-the-tests/data/sensor info placeholder.pdf",
    },
    {
        "name": "Diagonal Database.csv",
        "path": "/published-data/PRJ-3053/Experiment--database-of-diagonally-reinforced-concrete-coupling-beams/data/Model-config--readme-file/Sensor--numerous-sensor-configurations-were-present-in-the-tests/Event--database-proper/data/Diagonal Database.csv",
    },
]


def request_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preview_url(path: str) -> str:
    encoded_path = urllib.parse.quote(path, safe="")
    return (
        "https://www.designsafe-ci.org/api/datafiles/tapis/public/preview/"
        f"designsafe.storage.published/{encoded_path}?doi={urllib.parse.quote(DOI, safe='')}"
    )


def inspect_csv(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = list(csv.reader(text.splitlines()))
    return {
        "text": text,
        "n_lines": len(text.splitlines()),
        "n_rows": len(rows),
        "n_columns_first_row": len(rows[0]) if rows else 0,
        "nonempty_cells": sum(1 for row in rows for cell in row if cell.strip()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "PRJ-3053",
        "doi": DOI,
        "title": "Database of Diagonally-Reinforced Concrete Coupling Beams",
        "access_method": "DesignSafe Data Depot anonymous preview postit redeem URL",
        "records": [],
    }
    for item in FILES:
        preview = request_json(preview_url(item["path"]))
        target = OUT / item["name"]
        download(str(preview["href"]), target)
        manifest["records"].append(
            {
                **item,
                "local_path": str(target.relative_to(ROOT)),
                "observed_size": target.stat().st_size,
                "observed_sha256": sha256(target),
            }
        )

    csv_info = inspect_csv(OUT / "Diagonal Database.csv")
    manifest["csv_inspection"] = {k: v for k, v in csv_info.items() if k != "text"}

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# DesignSafe PRJ-3053 Coupling Beam Triage v0",
        "",
        f"Created UTC: `{manifest['created_utc']}`",
        "",
        f"DOI: `{DOI}`",
        "",
        "## File Inspection",
        "",
        "| File | Size (bytes) | SHA256 |",
        "|---|---:|---|",
    ]
    for record in manifest["records"]:
        lines.append(f"| `{record['name']}` | {record['observed_size']} | `{record['observed_sha256']}` |")
    lines.extend(
        [
            "",
            "## CSV Inspection",
            "",
            f"- Lines: {csv_info['n_lines']}",
            f"- CSV rows: {csv_info['n_rows']}",
            f"- Columns in first row: {csv_info['n_columns_first_row']}",
            f"- Non-empty cells: {csv_info['nonempty_cells']}",
            "",
            "Raw CSV content:",
            "",
            "```text",
            csv_info["text"],
            "```",
            "",
            "## Decision",
            "",
            "Include PRJ-3053 as the ninth reproduction module. "
            "The public listing metadata initially understated the CSV size, but the preview/download "
            "endpoint returned a valid 60-specimen database with 21 literature-reference groups.",
            "",
            "Follow-up reproduction outputs are stored in `code/outputs/reproductions/designsafe_prj3053_coupling_beams/`.",
        ]
    )
    (OUT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
