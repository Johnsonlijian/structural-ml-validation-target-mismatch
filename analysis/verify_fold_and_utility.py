"""R38-57: verify the enhancement package's two headline claims from my own records.

Claim 1 (fold concentration): of the 24 events avoided by source-honouring selection at the strict beam
target, 22 fall in a single outer fold.
Claim 2 (cost sensitivity): with unit reward for an event-free release, penalty lambda for an event
release and zero for a refusal, the source-minus-random utility difference in the coded-category
scenario at alpha=0.05 is (-20 + 9 lambda)/393, changing sign at lambda = 20/9; at the looser targets it
is (-1 - 5 lambda)/393; and in the beam scenario it is (4 + 24 lambda)/804 at alpha=0.05.

Both are recomputed here from the fold-level decisions I produced, not taken from the package.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

R37 = Path(__file__).resolve().parents[2] / "R37_second_review_repair_2026-09-16"
outer = pd.read_csv(R37 / "audit/r37/decision_outer.csv")
OUT = Path(__file__).resolve().parents[1] / "analysis" / "out"

print(f"columns: {list(outer.columns)}")
print(f"rows: {len(outer)} | scenarios: {sorted(outer.scenario.unique())}")
print(f"protocols: {sorted(outer.protocol.unique())} | alphas: {sorted(outer.alpha.unique())}")


def pooled(df, protocol, alpha):
    g = df[(df.protocol == protocol) & (df.alpha == alpha)]
    return int(g.n_test.sum()), int(g.n_released.sum()), int(g.n_unsafe.sum()), len(g)


print("\n=== pooled counts (A = releases, E = events) ===")
rows = []
for scen, action in (("beam_release", None),
                     ("failure_mode", "release_when_not_dangerous")):
    df = outer[outer.scenario == scen]
    if action:
        df = df[df.action_set == action]
    for alpha in sorted(df.alpha.unique()):
        n_r, a_r, e_r, k = pooled(df, "random", alpha)
        n_s, a_s, e_s, _ = pooled(df, "source", alpha)
        # utility difference of source minus random, U = (A - E - lambda E)/N
        base = (a_s - e_s) - (a_r - e_r)
        lam = -(e_s - e_r)
        rows.append({"scenario": scen, "alpha": alpha, "N": n_r, "A_rand": a_r, "E_rand": e_r,
                     "A_src": a_s, "E_src": e_s, "dA": a_s - a_r, "dE": e_s - e_r,
                     "utility": f"({base} + {lam}*lambda)/{n_r}", "folds": k})
        print(f"  {scen:<14} a={alpha:.2f}  random {a_r:>4}/{e_r:>3}  source {a_s:>4}/{e_s:>3}  "
              f"dA={a_s-a_r:+4d} dE={e_s-e_r:+4d}  U_src-rand = ({base} + {lam}*lambda)/{n_r}")
pd.DataFrame(rows).to_csv(OUT / "R38_cost_sensitivity.csv", index=False)

print("\n=== fold concentration of the avoided events (beam, alpha=0.05) ===")
beam = outer[(outer.scenario == "beam_release") & (outer.alpha == 0.05)]
piv = beam.pivot_table(index="outer_fold", columns="protocol", values="n_unsafe", aggfunc="sum")
piv["avoided"] = piv["random"] - piv["source"]
piv["released_random"] = beam[beam.protocol == "random"].groupby("outer_fold").n_released.sum()
piv["released_source"] = beam[beam.protocol == "source"].groupby("outer_fold").n_released.sum()
print(piv.to_string())
print(f"\n  total events random={int(piv['random'].sum())} source={int(piv['source'].sum())} "
      f"avoided={int(piv['avoided'].sum())}")
print(f"  largest single-fold share of the avoided events: "
      f"{int(piv['avoided'].max())} of {int(piv['avoided'].sum())} "
      f"(fold {int(piv['avoided'].idxmax())})")
piv.to_csv(OUT / "R38_fold_concentration.csv")
