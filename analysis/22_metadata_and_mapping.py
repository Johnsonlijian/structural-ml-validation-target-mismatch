"""R38-22: metadata corrections verified through DataCite, plus the failure-mode mapping statement.

Verified here (DataCite REST API, 2026-09-16):
  10.17632/8ndgpm7zw7.1  creator Fitwi, Teklewoin Haile; issued 2025-04-02; CC BY-NC 4.0
  10.17632/rbhfnz32sy.1  creator Hanaa Salem;          issued 2022-08-02; CC BY 4.0
Both were previously listed as CC BY 4.0 / 2020 in the registry, which the third review flagged.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "manuscript"

# ---------------------------------------------------------------- 1. registry: licences and years
reg = ROOT / "analysis" / "04_corpus_registry.py"
t = reg.read_text(encoding="utf-8")
t = t.replace('"Mendeley Data", "CC BY-NC 4.0 (verify at source)", 203',
              '"Mendeley Data", "CC BY-NC 4.0 (DataCite)", 203')
t = t.replace('"Mendeley Data", "CC BY 4.0 (verify at source)", 98',
              '"Mendeley Data", "CC BY 4.0 (DataCite)", 98')
t = t.replace('"DesignSafe-CI", "published project terms", 60',
              '"DesignSafe-CI", "published project (PRJ-3053)", 60')
reg.write_text(t, encoding="utf-8")
print("registry licences updated")

# ---------------------------------------------------------------- 2. bibliography years and creators
bib = MAN / "references.bib"
b = bib.read_text(encoding="utf-8")
b = b.replace("""@misc{data_joints_mendeley,
  author = {Mendeley Data},
  title = {Exterior reinforced concrete beam-column joint tests},
  year = {2020},""",
              """@misc{data_joints_mendeley,
  author = {Fitwi, Teklewoin Haile},
  title = {Dataset of Exterior Reinforced Concrete Beam-Column Joint Experiments},
  year = {2025},""")
b = b.replace("""@misc{data_joints_cyclic,
  author = {Mendeley Data},
  title = {Beam-column joint shear strength under cyclic loading},
  year = {2020},""",
              """@misc{data_joints_cyclic,
  author = {Salem, Hanaa},
  title = {The dataset for the joint shear strength of beam-column joints subject to cyclic loading},
  year = {2022},""")
bib.write_text(b, encoding="utf-8")
print("bibliography entries updated:",
      "Fitwi" in b, "Salem" in b, "year = {2025}" in b, "year = {2022}" in b)

# ---------------------------------------------------------------- 3. dangerous family -> designated family
si = MAN / "si.tex"
s = si.read_text(encoding="utf-8")
old = re.search(r"\\paragraph\{Two action sets for the classification scenario\.\}.*?(?=\n\\paragraph)", s, re.S)
new = ("\\paragraph{Two action sets for the classification scenario.} The main statement uses "
       "\\emph{release only when the predicted mode is outside the designated family}, which matches an "
       "engineering release decision; the alternative \\emph{act on any high-confidence prediction} is "
       "also reported, and there coverage collapses to $0.003$--$0.203$ with near-zero realised rates, "
       "which is why it is not used.\n\n"
       "\\begin{table}[htbp]\\centering\n"
       "\\caption{Failure-mode labels of the classification asset. The released file contains numeric "
       "codes only: it ships no codebook, and the authors' repository README documents the database but "
       "not the label meanings. The designated family used in the decision experiment is therefore "
       "recorded as a \\emph{scenario category} --- label code 1, the largest class --- and no claim is "
       "made that it corresponds to a particular physical mechanism.}\\label{tab:modes}\\footnotesize\n"
       "\\begin{tabular}{crl}\\toprule\n"
       "Label code & Records & Treatment in this study \\\\\\midrule\n"
       "1 & 152 & designated family (an unsafe release is one whose measured label is 1) \\\\\n"
       "2 & 96 & outside the designated family \\\\\n"
       "3 & 122 & outside the designated family \\\\\n"
       "4 & 23 & outside the designated family \\\\\n"
       "\\bottomrule\\end{tabular}\\end{table}\n\n")
if old:
    s = s[:old.start()] + new + s[old.end():]
    si.write_text(s, encoding="utf-8")
    print("SI: two-action-set paragraph replaced with the mapping table")
else:
    print("SI: paragraph not found — check manually")

# ---------------------------------------------------------------- 4. main text wording
for name in ("sections/03_results.tex", "sections/02_methods.tex"):
    p = MAN / name
    x = p.read_text(encoding="utf-8")
    x = x.replace("belongs to the brittle family", "belongs to the designated family (label code 1)")
    x = x.replace("lies in the brittle family", "lies in the designated family (label code 1)")
    x = x.replace("lies outside the brittle family", "lies outside the designated family")
    x = x.replace("the brittle family", "the designated family")
    p.write_text(x, encoding="utf-8")
print("main text: 'brittle family' replaced by the designated-family wording")

# ---------------------------------------------------------------- 5. code availability tense
dec = MAN / "sections" / "06_declarations.tex"
d = dec.read_text(encoding="utf-8")
d = d.replace("experiment and the figure sources are released at",
              "experiment and the figure sources are prepared for release at")
d = d.replace("(release \\texttt{v3.2.0}, commit \\texttt{b3cde03}); the repository is prepared locally "
              "and is published\non acceptance.",
              "(release candidate \\texttt{v3.2.0}, commit \\texttt{b3cde03}); the repository exists "
              "locally, has not been published yet, and will be published on acceptance, with the "
              "release name updated here if the tag changes.")
dec.write_text(d, encoding="utf-8")
print("code availability statement: tense now matches reality")
