# -*- coding: utf-8 -*-
"""Smoke checks for the public reproducibility package (no network)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
# ASCII-only taboo tokens (avoid encoding issues on Windows consoles).
FORBIDDEN = ("DESKTOP-", "/Users/", "/home/")


def main() -> int:
    failures = []
    if not (ROOT / "code" / "33_cace_nine_module_summary.py").is_file():
        failures.append("missing code/33_cace_nine_module_summary.py")
    if not (ROOT / "outputs" / "tables" / "table_s8_group_topology_master.csv").is_file():
        failures.append("missing outputs/tables/table_s8_group_topology_master.csv")
    s7 = ROOT / "outputs" / "tables" / "cace_ninemodule_rf_summary.csv"
    if not s7.is_file():
        failures.append("missing outputs/tables/cace_ninemodule_rf_summary.csv (regenerate with run_qa_chain)")

    for p in ROOT.rglob("*.py"):
        if p.resolve() == SELF:
            continue
        if "Conflict" in p.name:
            failures.append(f"Conflict-tagged filename (use canonical script names): {p.relative_to(ROOT)}")
            continue
        t = p.read_text(encoding="utf-8", errors="ignore")
        for s in FORBIDDEN:
            if s in t:
                failures.append(f"forbidden substring in {p.relative_to(ROOT)}: {s}")
                break
        if re.search(r"[A-Za-z]:\\Users\\", t):
            failures.append(f"possible absolute Windows user path in {p.relative_to(ROOT)}")

    rep = ROOT / "outputs" / "reproducibility_smoke_test_report.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    nl = chr(10)
    body = "## Smoke test" + nl + nl
    if not failures:
        body += "- **OK**" + nl
    else:
        body += nl.join(f"- FAIL: {x}" for x in failures) + nl
    rep.write_text(body + nl, encoding="utf-8")
    print(rep.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
