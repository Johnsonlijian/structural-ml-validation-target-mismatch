"""R38-88: corrected verification of the wall discard result.

The first verification attempt stated the wrong reason: wall type B is not confined to a single campaign
(it appears in Oesterle {B,I,Rect} and Oh {B,Rect,T}). The property that actually caps the number of
folds is that **every campaign carrying B or I also carries Rect**, so any cluster that keeps B or I must
also contain Rect; Rect cannot appear in two clusters, so B, I and Rect always share one cluster, and only
Flanged and T can form additional ones. That gives at most three non-empty clusters, hence at most three
folds, however many records are discarded.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
WALL = PROJ / "code/outputs/reproductions/designsafe_prj2430_wall/extracted_wall_data.csv"
OUT = ROOT / "analysis" / "out"
AUTH, TYPE = "Authors", "walltype_Shape"

df = pd.read_csv(WALL, low_memory=False)
res = json.loads((OUT / "R38_wall_discard_alternative.json").read_text(encoding="utf-8"))
camp_types = {a: set(g[TYPE]) for a, g in df.groupby(AUTH)}
types = sorted(df[TYPE].unique())
fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


print("=== claim 1: B and I never appear without Rect")
carriers = {t: sorted(a for a, ts in camp_types.items() if t in ts) for t in types}
print(f"  carriers: " + " | ".join(f"{t}: {v}" for t, v in carriers.items()))
for t in ("B", "I"):
    bad = [a for a in carriers[t] if "Rect" not in camp_types[a]]
    check(f"every campaign carrying {t} also carries Rect", not bad, str(bad))

print("\n=== claim 2: therefore at most three non-empty clusters")


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for parts in set_partitions(rest):
        for i in range(len(parts)):
            yield parts[:i] + [[first] + parts[i]] + parts[i + 1:]
        yield [[first]] + parts


best = (0, None)
for r in range(1, len(types) + 1):
    for kept in itertools.combinations(types, r):
        kept = set(kept)
        for parts in set_partitions(sorted(kept)):
            clusters = [set(p) for p in parts]
            nonempty = 0
            for c in clusters:
                if any((ts & kept) and (ts & kept) <= c for ts in camp_types.values()):
                    nonempty += 1
            if nonempty > best[0]:
                best = (nonempty, (sorted(map(sorted, parts)), sorted(kept)))
check("no structure yields more than three non-empty clusters", best[0] <= 3,
      f"maximum {best[0]} at {best[1]}")
check("four folds infeasible", not res["per_K"]["4"]["feasible"])
check("five folds infeasible", not res["per_K"]["5"]["feasible"])
check("structural ceiling recorded as three", res["structural_ceiling"] == 3)

print("\n=== claim 3: the retained three-fold selection is closed under both relations")
k3 = res["per_K"]["3"]
clusters = [set(c) for c in k3["clusters"]]
kept_types = set(k3["kept_types"])
members: dict[int, list[str]] = {i: [] for i in range(len(clusters))}
for a, ts in camp_types.items():
    inter = ts & kept_types
    if not inter:
        continue
    holder = [i for i, c in enumerate(clusters) if inter <= c]
    if len(holder) == 1:
        members[holder[0]].append(a)
fold_of_campaign = {a: i for i, v in members.items() for a in v}
fold_of_type = {t: i for i, c in enumerate(clusters) for t in (c & kept_types)}
check("no retained campaign straddles two folds",
      not [a for a in fold_of_campaign if len({fold_of_campaign[a]} |
          {fold_of_type[t] for t in camp_types[a] if t in fold_of_type}) > 1])
check("no retained type level straddles two folds",
      not [t for t in fold_of_type if len({fold_of_type[t]} |
          {fold_of_campaign[a] for a in carriers[t] if a in fold_of_campaign}) > 1])
sizes = [int(((df[AUTH].isin(members[i])) & (df[TYPE].isin(clusters[i] & kept_types))).sum())
         for i in range(len(clusters)) if members[i]]
n_ret = sum(sizes)
L, U = math.ceil(0.05 * n_ret), math.floor(0.50 * n_ret)
check("fold sizes reproduce the recorded ones", sorted(sizes, reverse=True) == k3["fold_sizes"],
      f"{sorted(sizes, reverse=True)} vs {k3['fold_sizes']}")
check("every fold within [L, U]", all(L <= s <= U for s in sizes), f"[{L}, {U}]")
check("retained total matches", n_ret == k3["n_retained"], str(n_ret))
quarantined = {q["campaign"] for q in k3["quarantined_campaigns"]}
check("recorded quarantined set equals the complement",
      quarantined == set(camp_types) - set(fold_of_campaign),
      f"{len(quarantined)} quarantined")

print(f"\nretained {n_ret} of {len(df)} ({n_ret/len(df):.1%}) | quarantined {len(df)-n_ret} "
      f"| campaigns quarantined {len(quarantined)} of {len(camp_types)}")
print(f"\nALL CHECKS {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
