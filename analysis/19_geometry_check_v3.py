"""R38-19: geometry check v3 — ignore what is legitimately small or legitimately at the edge.

Two false positives appeared in v2: mathematical sub/superscripts (`$R^2$` renders its exponent at
6 pt on a 7 pt base) and page numbers, which on landscape pages land inside the edge band. The check
now measures the *dominant* body size per page, treats short low-height spans as scripts, and excludes
the header/footer lines when looking for ink in the paper-edge band.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pymupdf

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manuscript"
OUT = ROOT / "analysis" / "out"

EDGE_PT = 20.0
BODY_FLOOR = 7.0          # dominant type must be at least this
HARD_FLOOR = 5.5          # nothing may be smaller than this, scripts included
FOOTER_PT = 70.0          # page numbers and running heads live within this of the edge
DPI = 100


def page_report(page) -> dict | None:
    spans = [s for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"]
             if s["text"].strip()]
    if not spans:
        return None
    sizes = [round(s["size"], 1) for s in spans]
    dominant = Counter(sizes).most_common(1)[0][0]
    scripts, too_small = [], []
    for s in spans:
        h = s["bbox"][3] - s["bbox"][1]
        if s["size"] >= BODY_FLOOR:
            continue
        is_script = len(s["text"]) <= 4 and h < 5.0
        (scripts if is_script else too_small).append((round(s["size"], 1), s["text"][:16]))
    hard = [t for t in too_small if t[0] < HARD_FLOOR]
    pix = page.get_pixmap(dpi=DPI)
    w, h, n, stride = pix.width, pix.height, pix.n, pix.stride
    band = max(2, int(EDGE_PT / 72 * DPI))
    footer = int(FOOTER_PT / 72 * DPI)
    samples = pix.samples
    edge_ink = 0
    for y in range(footer, h - footer, 2):
        row = y * stride
        for x in list(range(0, band, 2)) + list(range(max(0, w - band), w, 2)):
            off = row + x * n
            if samples[off] < 200:
                edge_ink += 1
    for x in range(footer, w - footer, 2):
        for y in list(range(0, band, 2)) + list(range(max(0, h - band), h, 2)):
            off = y * stride + x * n
            if samples[off] < 200:
                edge_ink += 1
    problems = {}
    if dominant < BODY_FLOOR:
        problems["dominant_font"] = dominant
    if hard:
        problems["type_below_hard_floor"] = hard[:5]
    if edge_ink > 12:
        problems["edge_ink"] = edge_ink
    return {"page": page.number + 1, **problems} if problems else None


report = {"edge_pt": EDGE_PT, "body_floor_pt": BODY_FLOOR, "hard_floor_pt": HARD_FLOOR,
          "documents": []}
fails = 0
for name in ("main.pdf", "si.pdf"):
    doc = pymupdf.open(MAN / name)
    pages = [r for p in doc if (r := page_report(p))]
    doc.close()
    report["documents"].append({"file": name, "ok": not pages, "pages": pages})
    print(f"{name}: {'OK' if not pages else str(len(pages)) + ' page(s) flagged'}")
    for p in pages:
        print(f"  page {p['page']}: " + ", ".join(f"{k}={v}" for k, v in p.items() if k != "page"))
    fails += 0 if not pages else 1
(OUT / "R38_geometry_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nwrote {OUT/'R38_geometry_check.json'} | failing documents: {fails}")
sys.exit(1 if fails else 0)
