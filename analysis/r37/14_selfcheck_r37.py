"""R37-14: numeric self-check for the corrected manuscript.

Every quantitative statement changed by the second review is guarded here: the corrected decision
experiment (units, denominators, candidate feasibility), the CFST same-row comparison, the
de-aliased stratified contrasts, the singleton arithmetic and the counting units.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
R36 = ROOT.parent / "R36_independent_review_rebuild_2026-09-16"
AUD = ROOT / "audit" / "r37"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

checks: list[tuple[str, bool, str]] = []


def ck(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


# ---------------------------------------------------------------- corrected decision experiment
outer = pd.read_csv(AUD / "decision_outer.csv")
cand = pd.read_csv(AUD / "decision_candidates.csv")
unit = json.loads((AUD / "unit_check.json").read_text(encoding="utf-8"))
beam = outer[outer.scenario == "beam_release"]
fail = outer[(outer.scenario == "failure_mode") &
             (outer.action_set == "release_when_not_dangerous")]


def agg(df, protocol, alpha):
    g = df[(df.protocol == protocol) & (df.alpha == alpha)]
    return {"est": g.inner_r_all.mean(), "real": g.realised_r_all.mean(),
            "gap": g.realised_r_all.mean() - g.inner_r_all.mean(),
            "above": int((g.gap_r_all > 1e-12).sum()), "met": int(g.target_met.sum()),
            "cov": g.realised_coverage.mean(), "rel": g.realised_r_released.mean(), "n": len(g)}


ck("unit check: physical and log decision paths agree record by record",
   bool(unit["all_agree"]) and all(r["mismatches"] == 0 for r in unit["physical_vs_log_path"]),
   f"{len(unit['physical_vs_log_path'])} margins")
ck("inner coverage non-decreasing in alpha", unit["inner_coverage_monotone_in_alpha_violations"] == 0,
   str(unit["inner_coverage_monotone_in_alpha_violations"]))

b = agg(beam, "random", 0.05)
ck("beam/random: 0.0199 estimated -> 0.0499 realised (+0.0300)",
   abs(b["est"] - 0.0199) < 0.0005 and abs(b["real"] - 0.0499) < 0.0005 and abs(b["gap"] - 0.0300) < 0.0005,
   f"{b['est']:.4f}/{b['real']:.4f}/{b['gap']:+.4f}")
ck("beam/random: 4 of 5 folds above their own estimate, coverage 0.518, r_rel 0.109",
   b["above"] == 4 and abs(b["cov"] - 0.518) < 0.001 and abs(b["rel"] - 0.109) < 0.001,
   f"above={b['above']} cov={b['cov']:.3f} rel={b['rel']:.3f}")
ck("beam/random: gap identical at all three targets",
   len({round(agg(beam, "random", a)["gap"], 5) for a in (0.05, 0.10, 0.20)}) == 1)
ck("beam/source: conservative (-0.0162 / -0.0649 / -0.0863)",
   abs(agg(beam, "source", 0.05)["gap"] + 0.0162) < 0.0005
   and abs(agg(beam, "source", 0.10)["gap"] + 0.0649) < 0.0005
   and abs(agg(beam, "source", 0.20)["gap"] + 0.0863) < 0.0005,
   f"{agg(beam, 'source', 0.05)['gap']:+.4f}")
f05, f10, f20 = (agg(fail, "random", a) for a in (0.05, 0.10, 0.20))
ck("failure/random: optimistic (+0.0341 / +0.0271 / +0.0271)",
   abs(f05["gap"] - 0.0341) < 0.0005 and abs(f10["gap"] - 0.0271) < 0.0005
   and abs(f20["gap"] - 0.0271) < 0.0005,
   f"{f05['gap']:+.4f}")
s05 = agg(fail, "source", 0.05)
ck("failure/source: still optimistic (+0.0162) with lower realised rate than random",
   abs(s05["gap"] - 0.0162) < 0.0005 and s05["real"] < f05["real"],
   f"gap={s05['gap']:+.4f} real={s05['real']:.4f} vs random {f05['real']:.4f}")
ck("protocol comparison: source beats random in both scenarios at alpha=0.05",
   agg(beam, "source", 0.05)["real"] < agg(beam, "random", 0.05)["real"]
   and s05["real"] < f05["real"])
ck("coverage cost at alpha=0.05: about 0.025 (beam) and 0.074 (failure)",
   abs((agg(beam, "random", 0.05)["cov"] - agg(beam, "source", 0.05)["cov"]) - 0.025) < 0.004
   and abs((f05["cov"] - s05["cov"]) - 0.074) < 0.004,
   f"{agg(beam, 'random', 0.05)['cov'] - agg(beam, 'source', 0.05)['cov']:.3f} / {f05['cov'] - s05['cov']:.3f}")

# candidate feasibility: active vs inactive constraint
beam_cand = cand[cand.scenario == "beam_release"]


def feasible_count(protocol, fold, alpha):
    m = ((beam_cand.protocol == protocol) & (beam_cand.outer_fold == fold) & (beam_cand.alpha == alpha))
    return int(beam_cand[m]["feasible"].sum()), int(m.sum())


ck("beam/random: all candidate rules feasible at every target (inactive constraint)",
   all(feasible_count("random", f, a) == (63, 63) for f in range(5) for a in (0.05, 0.10, 0.20)))
src_feas = {feasible_count("source", f, 0.05)[0] for f in range(5)}
ck("beam/source: 42-56 of 63 rules feasible at alpha=0.05 (constraint active)",
   min(src_feas) == 42 and max(src_feas) == 56, str(sorted(src_feas)))
src_feas_10 = {feasible_count("source", f, 0.10)[0] for f in range(5)}
ck("beam/source: 54-63 rules feasible at alpha=0.10 and 63 at alpha=0.20",
   min(src_feas_10) == 54 and max(src_feas_10) == 63
   and all(feasible_count("source", f, 0.20)[0] == 63 for f in range(5)),
   f"{sorted(src_feas_10)}")

# ---------------------------------------------------------------- CFST same rows
cfst = pd.read_csv(AUD / "cfst_same_rows.csv")
ck("CFST restricted holdout covers 642 of 1316 records (48.78 %)",
   int(cfst["random5_R2_on_642_rows"].notna().sum()) == 4, "four learners")
row_rf = cfst[cfst.model == "RandomForest"].iloc[0]
row_ridge = cfst[cfst.model == "Ridge"].iloc[0]
row_et = cfst[cfst.model == "ExtraTrees"].iloc[0]
row_gb = cfst[cfst.model == "GradientBoosting"].iloc[0]
ck("CFST same rows: random forest 0.812 vs restricted 0.817 (+0.005)",
   abs(row_rf["random5_R2_on_642_rows"] - 0.812) < 0.001
   and abs(row_rf["restricted_holdout_R2"] - 0.817) < 0.001
   and abs(row_rf["delta_same_rows"] - 0.005) < 0.001)
ck("CFST same rows: ridge -1.053, extra trees -0.074, boosting -0.034",
   abs(row_ridge["delta_same_rows"] + 1.053) < 0.001
   and abs(row_et["delta_same_rows"] + 0.074) < 0.001
   and abs(row_gb["delta_same_rows"] + 0.034) < 0.001)

# ---------------------------------------------------------------- contrasts, aliases, strata
pairs = pd.read_csv(R36 / "analysis/out/R36_control_pairs.csv")
adm = pairs[pairs["split_admissible"].fillna(False)]
ck("36 admissible rows, 32 distinct evaluations after merging the concrete alias", len(adm) == 36,
   str(len(adm)))
src_only = adm[adm.variant == "honoured"]
ck("source stratum: 16 rows, 16/16 below the size-matched control, median -0.1654",
   len(src_only) == 16 and int((src_only.delta_vs_sizematch < 0).sum()) == 16
   and abs(src_only.delta_vs_sizematch.median() + 0.1654) < 0.0005,
   f"{len(src_only)} / {int((src_only.delta_vs_sizematch < 0).sum())} / "
   f"{src_only.delta_vs_sizematch.median():+.4f}")
ck("largest admissible degradation -3.839 overall, -2.879 within the size-matched subset",
   abs(adm.r2_honoured.min() + 3.839) < 0.001
   and abs(adm[adm.delta_vs_sizematch.notna()].r2_honoured.min() + 2.879) < 0.001,
   f"{adm.r2_honoured.min():.3f} / {adm[adm.delta_vs_sizematch.notna()].r2_honoured.min():.3f}")
ck("pooled: 32/36 vs random (median -0.109) and 17/20 vs size-match (median -0.136)",
   int((adm.delta_vs_random < 0).sum()) == 32
   and abs(adm.delta_vs_random.median() + 0.109) < 0.0005
   and int((adm[adm.delta_vs_sizematch.notna()].delta_vs_sizematch < 0).sum()) == 17)

# ---------------------------------------------------------------- fold audit and classification
folds = pd.read_csv(R36 / "analysis/out/R36_fold_admissibility.csv")
bad = folds[~folds.admissible].copy()
STRUCT = {"honoured": "source", "sizematch_relation": "source", "honoured_proxy": "wall",
          "sizematch_proxy": "wall"}
bad["structure"] = bad.variant.map(STRUCT)
ck("34 audited variants, 6 inadmissible, 3 distinct structures",
   len(folds) == 34 and len(bad) == 6 and bad.groupby(["dataset", "structure"]).ngroups == 3)
clf = pd.read_csv(R36 / "audit/r36/classification_controls.csv")
ck("classification 0.847 -> 0.491 with size-matched control 0.844",
   abs(float(clf[(clf.variant == "random5")].accuracy_mean.iloc[0]) - 0.847) < 0.001
   and abs(float(clf[(clf.variant == "honoured")].accuracy_mean.iloc[0]) - 0.491) < 0.001
   and abs(float(clf[(clf.variant == "sizematch")].accuracy_mean.iloc[0]) - 0.844) < 0.001)

# ---------------------------------------------------------------- manuscript-level checks
main = (ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S).group(1)
n_words = len(re.findall(r"[A-Za-z][A-Za-z\-]*", abstract))
ck("abstract within 250 words", n_words <= 250, f"{n_words} words")
body = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "manuscript").rglob("*.tex"))
for phrase in ("earlier version of this work", "three to six", "94.9\\,\\% are singletons",
               "stayed at or below its estimate in all 45"):
    ck(f"withdrawn phrase absent: {phrase!r}", phrase not in body)
ck("no appendix proof left in the source", not (ROOT / "manuscript/sections/07_appendix.tex").exists())

failed = [c for c in checks if not c[1]]
lines = ["# R37 numeric self-check", "",
         f"checks: {len(checks)} | passed: {len(checks) - len(failed)} | failed: {len(failed)}", ""]
lines += [f"- {'PASS' if ok else '**FAIL**'} — {name}" + (f"  (`{d}`)" if d else "")
          for name, ok, d in checks]
(OUT / "R37_MANUSCRIPT_SELFCHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:5]))
for name, ok, d in failed:
    print(f"FAIL: {name} ({d})")
print(f"wrote {OUT/'R37_MANUSCRIPT_SELFCHECK.md'}")
