"""R38-27: numeric and consistency self-check for the R38 manuscript.

Guards every number and wording decision the third review forced: the corpus counts generated from the
registry, the de-aliased contrast table, the decision experiment's fold means against its pooled counts
(including the reversal at looser targets), the figure corrections, the metadata verified through
DataCite, and the page geometry of the final PDFs.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pymupdf

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
R36 = ROOT.parent / "R36_independent_review_rebuild_2026-09-16"
R37 = ROOT.parent / "R37_second_review_repair_2026-09-16"
MAN = ROOT / "manuscript"
AUD = ROOT / "audit" / "r38"
OUT = ROOT / "analysis" / "out"

checks: list[tuple[str, bool, str]] = []


def ck(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def pdf_text(path: Path) -> str:
    doc = pymupdf.open(path)
    t = "\n".join(p.get_text() for p in doc)
    doc.close()
    return re.sub(r"\s+", " ", t)


main_pdf, si_pdf = pdf_text(MAN / "main.pdf"), pdf_text(MAN / "si.pdf")
src = "\n".join(p.read_text(encoding="utf-8") for p in list((MAN / "sections").glob("*.tex")) +
                [MAN / "main.tex", MAN / "si.tex"])

# ---------------------------------------------------------------- 1. corpus counts
nums = json.loads((AUD / "corpus_numbers.json").read_text(encoding="utf-8"))
ck("registry: 11 entries, 5,035 records, 6 in analysis (3,745), 5 inventory-only (1,290)",
   nums == {"assets_total": 11, "assets_in_analysis": 6, "assets_inventory_only": 5,
            "records_total": 5035, "records_in_analysis": 3745, "records_inventory_only": 1290,
            "regression_assets": 5, "regression_configurations": 6}, str(nums))
for token in ("5{,}035", "3{,}745", "1{,}290", "eleven"):
    ck(f"main text states {token}", token in (MAN / "main.tex").read_text(encoding="utf-8")
       or token in (MAN / "sections" / "03_results.tex").read_text(encoding="utf-8"))
ck("no residue of the nine-asset / 4,933 claim", "4{,}933" not in src and "nine assets" not in src)
ck("classification called the seventh task configuration, not a seventh dataset",
   "seventh \\emph{task configuration}" in src or "seventh task configuration" in src)

# ---------------------------------------------------------------- 2. contrast table and stratum claims
pairs = pd.read_csv(R36 / "analysis/out/R36_control_pairs.csv")
adm = pairs[pairs["split_admissible"].fillna(False)]
adm = adm[~((adm.variant == "uci_996_allinputs") & (adm.dataset == "concrete_strength"))]
src_only = adm[adm.variant == "honoured"]
ck("32 distinct admissible entries", len(adm) == 32, str(len(adm)))
ck("source stratum 16/16 below the size-matched control, median -0.1654",
   len(src_only) == 16 and int((src_only.delta_vs_sizematch < 0).sum()) == 16
   and abs(src_only.delta_vs_sizematch.median() + 0.1654) < 0.0005)
ck("extremes quoted as differences: -3.722 vs random, -3.306 vs size-matched",
   "-3.722" in src and "-3.306" in src)
ck("pooled '32 of 36' sentence removed", "pooled over all admissible strata" not in src)
ck("per-stratum '4 of 4' claim removed", "4 of 4 below their size-matched control" not in src)
t4 = (MAN / "tables" / "table2_contrasts.tex").read_text(encoding="utf-8")
ck("contrast table prints 'not evaluated' instead of nan", "nan" not in t4.lower()
   and "not evaluated" in t4)
ck("contrast table has no duplicate alias row", "alias of the proxy row" not in t4)
data_rows = [ln for ln in t4.splitlines()
             if ln.rstrip().endswith(r"\\") and "&" in ln and not ln.startswith("Relation stratum")]
strata = {ln.split("&")[0].strip() for ln in data_rows}
ck("contrast table strata are exactly the four relation kinds",
   strata == {"source-level", "combination-unseen", "duplication (7-ingredient key)",
              "duplication (8-input key)"}, str(sorted(strata)))
wall_rows = [ln for ln in data_rows if "RC wall" in ln]
ck("no wall row is mislabelled as a duplication relation",
   all("duplication" not in ln for ln in wall_rows), f"{len(wall_rows)} wall rows")
ck("no data row is labelled as a proxy stratum",
   all("proxy" not in ln.split("&")[0] for ln in data_rows))

# ---------------------------------------------------------------- 3. decision experiment
outer = pd.read_csv(R37 / "audit/r37/decision_outer.csv")
beam = outer[outer.scenario == "beam_release"]
fail = outer[(outer.scenario == "failure_mode") & (outer.action_set == "release_when_not_dangerous")]


def pair(df, alpha):
    r = df[(df.protocol == "random") & (df.alpha == alpha)]
    s = df[(df.protocol == "source") & (df.alpha == alpha)]
    return r.realised_r_all.mean(), s.realised_r_all.mean()


b05, b10, b20 = (pair(beam, a) for a in (0.05, 0.10, 0.20))
f05, f10, f20 = (pair(fail, a) for a in (0.05, 0.10, 0.20))
ck("beam: source better at all three targets",
   all(s < r for r, s in (b05, b10, b20)), f"{b05[0]:.4f}/{b05[1]:.4f}")
ck("failure mode: source better at alpha=0.05 only",
   f05[1] < f05[0] and f10[1] > f10[0] and f20[1] > f20[0],
   f"0.05 {f05[0]:.4f}->{f05[1]:.4f}; 0.10 {f10[0]:.4f}->{f10[1]:.4f}")
ck("text states the reversal", "reverses" in src or "reversal" in src)
ck("failure-mode reversal quantified in the text", "0.0873" in src and "0.0741" in src)
ck("pooled conditional risk reported beside the fold mean",
   "9.6" in src and "10.9" in src and "4.0" in src and "3.1" in src)
for label, df, expect in (("beam/random", beam[beam.protocol == "random"], (804, 416, 40)),
                          ("beam/source", beam[beam.protocol == "source"], (804, 396, 16))):
    g = df[df.alpha == 0.05]
    got = (int(g.n_test.sum()), int(g.n_released.sum()), int(g.n_unsafe.sum()))
    ck(f"pooled counts {label} = {expect}", got == expect, str(got))

# ---------------------------------------------------------------- 4. figures
d1 = (MAN / "fig1_scenario_drawing.tex").read_text(encoding="utf-8")
ck("figure 1 carries no run-specific numbers",
   not any(t in d1 for t in ("0.021", "0.050", "0.038", "0.026")))
f3 = (ROOT / "analysis" / "14_figures_legible_type.py").read_text(encoding="utf-8")
ck("feasibility map treats K>m as structurally impossible",
   "elif K > m_families" in f3 and "no: $m<K$" in f3)

# ---------------------------------------------------------------- 5. metadata and wording
reg_tbl = (MAN / "tables" / "table1_assets.tex").read_text(encoding="utf-8")
bib = (MAN / "references.bib").read_text(encoding="utf-8")
ck("Mendeley licences as verified", "CC BY-NC 4.0 (DataCite)" in reg_tbl
   and "CC BY 4.0 (DataCite)" in reg_tbl)
ck("Mendeley years and creators corrected in the bibliography",
   "year = {2025}" in bib and "year = {2022}" in bib and "Fitwi" in bib and "Salem" in bib)
ck("designated family replaces the brittle-family wording",
   "brittle" not in src and "designated family" in src)
ck("SI carries the failure-mode mapping table", "Failure-mode labels of the classification asset" in si_pdf)
ck("no claim that the designated family is a physical mechanism",
   "scenario category" in si_pdf or "scenario category" in src)
ck("code availability tense is consistent",
   "prepared for release" in src and "has not been published" in src)

# ---------------------------------------------------------------- 6. abstract, highlights, geometry
main_tex = (MAN / "main.tex").read_text(encoding="utf-8")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, re.S).group(1)
plain = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", abstract)
plain = re.sub(r"[\\$]", "", plain)
words = [w for w in re.split(r"\s+", plain.strip()) if re.search(r"[A-Za-z0-9]", w)]
ck("abstract at or below 245 words", len(words) <= 245, f"{len(words)} words")
hl = (MAN / "sections" / "00_highlights.tex").read_text(encoding="utf-8")
lens = [len(" ".join(m.split())) for m in re.findall(r"\\item\s+(.*?)(?=\n\\item|\n\\end)", hl, re.S)]
ck("every highlight at or below 85 characters", all(n <= 85 for n in lens), str(lens))
ck("five highlights", len(lens) == 5, str(len(lens)))

geo = json.loads((OUT / "R38_geometry_check.json").read_text(encoding="utf-8"))
ck("final PDFs pass the rendered-geometry check",
   all(d["ok"] for d in geo["documents"]),
   str([(d["file"], d["ok"]) for d in geo["documents"]]))
ck("main tables 1-6 and figures 1-5 all present",
   len(re.findall(r"Table\s+\d+:", main_pdf)) == 6 and len(re.findall(r"Figure\s+\d+:", main_pdf)) == 5)
ck("SI carries seven numbered tables",
   len(re.findall(r"Table\s+\d+:", si_pdf)) == 7, str(len(re.findall(r"Table\s+\d+:", si_pdf))))

failed = [c for c in checks if not c[1]]
lines = ["# R38 numeric and consistency self-check", "",
         f"checks: {len(checks)} | passed: {len(checks) - len(failed)} | failed: {len(failed)}", ""]
lines += [f"- {'PASS' if ok else '**FAIL**'} — {n}" + (f"  (`{d}`)" if d else "") for n, ok, d in checks]
(OUT / "R38_MANUSCRIPT_SELFCHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:4]))
for n, ok, d in failed:
    print(f"FAIL: {n} ({d})")
print(f"wrote {OUT/'R38_MANUSCRIPT_SELFCHECK.md'}")
