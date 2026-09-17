"""Reproduce every number reported in the manuscript from the shipped aggregate tables.

Run from the package root:  python verify_reported_numbers.py

Inputs are the derived tables only; no raw third-party data and no network access are required. Each
check prints PASS or FAIL and the command exits non-zero if any check fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / "derived"
checks: list[tuple[str, bool, str]] = []


def ck(name, ok, detail=""):
    checks.append((name, bool(ok), detail))


# ---------------------------------------------------------------- decision experiment
outer = pd.read_csv(D / "decision_outer.csv")
beam = outer[outer.scenario == "beam_release"]
cat = outer[(outer.scenario == "failure_mode") &
            (outer.action_set == "release_when_not_dangerous")]


def pooled(df, protocol, alpha):
    g = df[(df.protocol == protocol) & (df.alpha == alpha)]
    return int(g.n_released.sum()), int(g.n_unsafe.sum()), int(g.n_test.sum()), len(g)


for alpha, want in ((0.05, (416, 40, 396, 16)), (0.10, (416, 40, 363, 13)),
                    (0.20, (416, 40, 374, 16))):
    ar, er, as_, es, = (*pooled(beam, "random", alpha)[:2], *pooled(beam, "source", alpha)[:2])
    ck(f"beam alpha={alpha:.2f}: random {want[0]}/{want[1]}, source {want[2]}/{want[3]}",
       (ar, er, as_, es) == want, f"got {ar}/{er}, {as_}/{es}")

for alpha, want in ((0.05, (193, 31, 164, 22)), (0.10, (193, 30, 197, 35)),
                    (0.20, (193, 30, 197, 35))):
    ar, er, as_, es = (*pooled(cat, "random", alpha)[:2], *pooled(cat, "source", alpha)[:2])
    ck(f"coded-category alpha={alpha:.2f}: random {want[0]}/{want[1]}, source {want[2]}/{want[3]}",
       (ar, er, as_, es) == want, f"got {ar}/{er}, {as_}/{es}")

# ---------------------------------------------------------------- fold concentration
b05 = beam[beam.alpha == 0.05]
piv = b05.pivot_table(index="outer_fold", columns="protocol", values="n_unsafe", aggfunc="sum")
avoided = (piv["random"] - piv["source"])
ck("beam alpha=0.05 avoids 24 unsafe releases", int(avoided.sum()) == 24, str(int(avoided.sum())))
ck("22 of the 24 avoided events sit in one outer fold",
   int(avoided.max()) == 22, f"max fold {int(avoided.idxmax())} = {int(avoided.max())}")

# ---------------------------------------------------------------- utility sensitivity
util = pd.read_csv(D / "utility_sensitivity.csv")
want = {("beam_release", 0.05): (4, 24, 804), ("beam_release", 0.10): (-26, 27, 804),
        ("beam_release", 0.20): (-18, 24, 804), ("failure_mode", 0.05): (-20, 9, 393),
        ("failure_mode", 0.10): (-1, -5, 393), ("failure_mode", 0.20): (-1, -5, 393)}
for _, row in util.iterrows():
    key = (row["scenario"], round(float(row["alpha"]), 2))
    base, lam, n = want[key]
    ok = row["utility"] == f"({base} + {lam}*lambda)/{n}"
    ck(f"utility difference {key[0]} alpha={key[1]:.2f} is ({base} + {lam}*lambda)/{n}", ok,
       row["utility"])

# ---------------------------------------------------------------- registry and contrasts
reg = pd.read_csv(D / "corpus_registry.csv")
ck("registry holds eleven entries", len(reg) == 11, str(len(reg)))
ck("registry records total 5,035", int(reg["records"].sum()) == 5035, str(int(reg["records"].sum())))
adm = pd.read_csv(D / "control_pairs.csv")
adm = adm[adm["split_admissible"].fillna(False)]
adm = adm[~((adm.variant == "uci_996_allinputs") & (adm.dataset == "concrete_strength"))]
ck("32 distinct admissible contrasts", len(adm) == 32, str(len(adm)))
src = adm[adm.variant == "honoured"]
ck("source stratum degrades in 16 of 16", len(src) == 16 and int((src.delta_vs_sizematch < 0).sum()) == 16)
ck("source-stratum median difference is -0.165",
   abs(src.delta_vs_sizematch.median() + 0.1654) < 0.0005, f"{src.delta_vs_sizematch.median():.4f}")

failed = [c for c in checks if not c[1]]
print(f"checks: {len(checks)} | passed: {len(checks) - len(failed)} | failed: {len(failed)}")
for name, ok, detail in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not ok else ""))
sys.exit(1 if failed else 0)
