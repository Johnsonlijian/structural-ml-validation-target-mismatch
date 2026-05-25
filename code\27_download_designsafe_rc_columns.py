from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "designsafe_rc_columns"

FILES = {
    "10.17603-ds2-52bz-0n63": {
        "doi": "10.17603/ds2-52bz-0n63",
        "project": "PRJ-2525",
        "title": "Circular Column Database",
        "filename": "Database_Circular_Columns.csv",
        "path": "/published-data/PRJ-2525/Project--circular-column-database/data/Database_Circular_Columns.csv",
    },
    "10.17603-ds2-7qg0-4303": {
        "doi": "10.17603/ds2-7qg0-4303",
        "project": "PRJ-2526",
        "title": "Rectangular Column Database",
        "filename": "Database_Rectangular_Columns.csv",
        "path": "/published-data/PRJ-2526/Project--rectangular-column-database/data/Database_Rectangular_Columns.csv",
    },
}


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


def preview_url(path: str, doi: str) -> str:
    encoded_path = urllib.parse.quote(path, safe="")
    return (
        "https://www.designsafe-ci.org/api/datafiles/tapis/public/preview/"
        f"designsafe.storage.published/{encoded_path}?doi={urllib.parse.quote(doi, safe='')}"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "access_method": "DesignSafe Data Depot anonymous listing + preview postit redeem URL",
        "records": [],
    }
    for key, meta in FILES.items():
        folder = OUT / key
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / meta["filename"]
        preview = request_json(preview_url(meta["path"], meta["doi"]))
        href = str(preview["href"])
        download(href, target)
        record = {
            **meta,
            "preview_href_host": urllib.parse.urlparse(href).netloc,
            "local_path": str(target.relative_to(ROOT)),
            "observed_size": target.stat().st_size,
            "observed_sha256": sha256(target),
        }
        manifest["records"].append(record)

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# DesignSafe RC Columns Data Manifest",
        "",
        f"Created UTC: `{manifest['created_utc']}`",
        "",
        "Access method: DesignSafe Data Depot anonymous listing plus preview postit redeem URL.",
        "",
    ]
    for record in manifest["records"]:
        lines.extend(
            [
                f"## {record['doi']}",
                "",
                f"- Project: {record['project']}",
                f"- Title: {record['title']}",
                f"- Local path: `{record['local_path']}`",
                f"- Size: {record['observed_size']} bytes",
                f"- SHA256: `{record['observed_sha256']}`",
                "",
            ]
        )
    (OUT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
