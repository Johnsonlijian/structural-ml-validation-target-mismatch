"""R38-11: the standard geometry check, replacing the log-based overfull scan.

Log-based checks cannot see a table that was scaled with \\resizebox or one whose columns simply run
past the paper edge; only the rendered page shows what a reader sees. This instrument reports, per
page: text outside an explicit safe area, the worst right/bottom edge, and the smallest font size.

Usage:  python analysis/11_geometry_check.py [main|si|both]
Exit code is non-zero when a document fails, so the check can gate a rebuild.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manuscript"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# A4 with the elsarticle preprint margins: require 40 pt of margin on every side, and no text below
# 7 pt (the point at which table type stops being legible in print).
SIDE_MARGIN = 40.0
FONT_FLOOR = 7.0


def check(pdf: Path) -> dict:
    doc = pymupdf.open(pdf)
    pages = []
    for pno, page in enumerate(doc, 1):
        w, h = page.rect.width, page.rect.height
        worst_right, worst_left, worst_bottom = 0.0, w, 0.0
        small = []
        over = []
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    if not s["text"].strip():
                        continue
                    x0, y0, x1, y1 = s["bbox"]
                    worst_right = max(worst_right, x1)
                    worst_left = min(worst_left, x0)
                    worst_bottom = max(worst_bottom, y1)
                    if x1 > w - SIDE_MARGIN or x0 < SIDE_MARGIN or y1 > h - SIDE_MARGIN:
                        over.append(s["text"][:24])
                    if round(s["size"], 1) < FONT_FLOOR:
                        small.append(round(s["size"], 1))
        if over or small:
            pages.append({
                "page": pno, "width": round(w, 1),
                "text_outside_safe_area": len(over),
                "worst_right": round(worst_right, 1),
                "min_left": round(worst_left, 1),
                "worst_bottom": round(worst_bottom, 1),
                "fonts_below_floor": sorted(set(small)),
                "examples": over[:6],
            })
    doc.close()
    return {"file": pdf.name, "pages_with_problems": pages,
            "ok": not pages}


targets = sys.argv[1] if len(sys.argv) > 1 else "both"
files = {"main": ["main.pdf"], "si": ["si.pdf"], "both": ["main.pdf", "si.pdf"]}[targets]
report = {"side_margin_pt": SIDE_MARGIN, "font_floor_pt": FONT_FLOOR, "documents": []}
fails = 0
for name in files:
    r = check(MAN / name)
    report["documents"].append(r)
    flag = "OK" if r["ok"] else f"{len(r['pages_with_problems'])} page(s) with problems"
    print(f"{name}: {flag}")
    for p in r["pages_with_problems"]:
        print(f"  page {p['page']}: outside safe area {p['text_outside_safe_area']} spans, "
              f"worst_right {p['worst_right']} (limit {p['width'] - SIDE_MARGIN:.0f}), "
              f"fonts {p['fonts_below_floor']}")
        if p["examples"]:
            print(f"      e.g. {p['examples']}")
    fails += 0 if r["ok"] else 1
(OUT / "R38_geometry_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nwrote {OUT/'R38_geometry_check.json'} | documents failing: {fails}")
sys.exit(1 if fails else 0)
