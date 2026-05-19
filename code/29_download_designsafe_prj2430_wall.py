from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "designsafe_prj2430_wall"
DOI = "10.17603/ds2-r12q-t415"

FILES = [
    {
        "name": "PRJ-2430_metadata.json",
        "path": "/published-data/PRJ-2430/PRJ-2430_metadata.json",
    },
    {
        "name": "1. ReadMe.txt",
        "path": "/published-data/PRJ-2430/Project--uoa-uw-reinforced-concrete-wall-database/data/1. ReadMe.txt",
    },
    {
        "name": "2. Database references.docx",
        "path": "/published-data/PRJ-2430/Project--uoa-uw-reinforced-concrete-wall-database/data/2. Database references.docx",
    },
    {
        "name": "3. UoA_UW_WallDatabase_ReadMe.xlsx",
        "path": "/published-data/PRJ-2430/Project--uoa-uw-reinforced-concrete-wall-database/data/3. UoA_UW_WallDatabase_ReadMe.xlsx",
    },
    {
        "name": "WallData.mat",
        "path": "/published-data/PRJ-2430/Project--uoa-uw-reinforced-concrete-wall-database/data/WallData.mat",
    },
]


def request_json(url: str) -> dict[str, object]:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
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


def normalize_preview_href(href: str) -> str:
    parsed = urllib.parse.urlparse(href)
    if "view.officeapps.live.com" not in parsed.netloc:
        return href
    params = urllib.parse.parse_qs(parsed.query)
    src = params.get("src", [href])[0]
    return urllib.parse.unquote(src)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "PRJ-2430",
        "doi": DOI,
        "title": "UoA-UW Reinforced Concrete Wall Database",
        "access_method": "DesignSafe Data Depot anonymous preview postit redeem URL",
        "records": [],
    }
    for item in FILES:
        preview = request_json(preview_url(item["path"]))
        href = normalize_preview_href(str(preview["href"]))
        target = OUT / item["name"]
        download(href, target)
        manifest["records"].append(
            {
                **item,
                "local_path": str(target.relative_to(ROOT)),
                "observed_size": target.stat().st_size,
                "observed_sha256": sha256(target),
            }
        )

    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "# DesignSafe PRJ-2430 Wall Data Manifest",
        "",
        f"Created UTC: `{manifest['created_utc']}`",
        "",
        f"DOI: `{DOI}`",
        "",
        "Access method: DesignSafe Data Depot anonymous preview postit redeem URL.",
        "",
    ]
    for record in manifest["records"]:
        lines.extend(
            [
                f"## {record['name']}",
                "",
                f"- Remote path: `{record['path']}`",
                f"- Local path: `{record['local_path']}`",
                f"- Size: {record['observed_size']} bytes",
                f"- SHA256: `{record['observed_sha256']}`",
                "",
            ]
        )
    (OUT / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
