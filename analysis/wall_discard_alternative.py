"""R38-86: the discard-based alternative, final — the structural ceiling on the number of folds.

Why no discard-based protocol can rescue a five-fold two-relation claim on this database, and what the
best achievable protocol retains:

  * the campaign signature of the data is ('Rect',) 20 campaigns/98 records, ('B','I','Rect') 1/13,
    ('Flanged',) 2/10, ('Rect','T') 2/10, ('B','Rect','T') 1/5, ('T',) 1/4, ('Flanged','Rect') 1/2;
  * B and I occur *only* inside the single Oesterle campaign, and Oesterle also carries Rect, so any
    partition honouring both relations must place B, I and Rect in one closed cluster; otherwise the
    campaign is quarantined and B and I vanish with it;
  * Flanged and T can then form at most two further clusters, so the number of folds is capped at three
    no matter how many records are discarded.

For each structure (a partition of a kept subset of type levels) the components are formed, the dominant
component is shrunk until both fold-fraction bounds hold, and the best retention is reported for every K.
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
n = len(df)
camp_types = {a: set(g[TYPE]) for a, g in df.groupby(AUTH)}
camp_size = {a: len(g) for a, g in df.groupby(AUTH)}
types = sorted(df[TYPE].unique())


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for parts in set_partitions(rest):
        for i in range(len(parts)):
            yield parts[:i] + [[first] + parts[i]] + parts[i + 1:]
        yield [[first]] + parts


def feasible_grouping(sizes, K):
    best = None
    for parts in set_partitions(list(range(len(sizes)))):
        if len(parts) != K:
            continue
        gs = [sum(sizes[i] for i in part) for part in parts]
        n_ret = sum(gs)
        L, U = math.ceil(0.05 * n_ret), math.floor(0.50 * n_ret)
        if all(L <= s <= U for s in gs) and (best is None or n_ret > best[0]):
            best = (n_ret, sorted(gs, reverse=True), L, U)
    return best


structures = {}
for r in range(1, len(types) + 1):
    for kept in itertools.combinations(types, r):
        for parts in set_partitions(list(kept)):
            key = tuple(sorted(tuple(sorted(p)) for p in parts))
            structures.setdefault(key, set(kept))
print(f"distinct cluster structures: {len(structures)}")

results = {}
for K in (2, 3, 4, 5):
    best = None
    for key, kept_types in structures.items():
        clusters = [set(p) for p in key]
        members: dict[int, list[tuple[str, int]]] = {i: [] for i in range(len(clusters))}
        for a, ts in camp_types.items():
            inter = ts & kept_types
            if not inter:
                continue
            holder = [i for i, cts in enumerate(clusters) if inter <= cts]
            if len(holder) == 1:
                members[holder[0]].append(
                    (a, int(((df[AUTH] == a) & (df[TYPE].isin(clusters[holder[0]]))).sum())))
        if sum(1 for v in members.values() if v) < K:
            continue
        for _ in range(300):
            sizes = [sum(sz for _, sz in v) for v in members.values() if v]
            res = feasible_grouping(sizes, K)
            if res:
                if best is None or res[0] > best["n_retained"]:
                    best = {"n_retained": res[0], "folds": K, "fold_sizes": res[1],
                            "L": res[2], "U": res[3],
                            "kept_types": sorted(kept_types),
                            "clusters": [sorted(c) for c in clusters],
                            "component_sizes": sorted(sizes, reverse=True),
                            "retained_campaigns": sorted(a for v in members.values()
                                                         for a, _ in v)}
                break
            reducible = [i for i, v in members.items() if len(v) > 1]
            if not reducible:
                break
            i = max(reducible, key=lambda i: sum(sz for _, sz in members[i]))
            members[i].sort(key=lambda x: x[1])
            members[i].pop(0)
    results[K] = best

print("\n=== best discard-based protocol per fold count ===")
summary = {"records": n, "campaigns": len(camp_types), "per_K": {}}
for K in (2, 3, 4, 5):
    b = results[K]
    if b is None:
        print(f"K = {K}: infeasible — no discard-based protocol exists")
        summary["per_K"][str(K)] = {"feasible": False}
    else:
        lost = sorted(set(camp_types) - set(b["retained_campaigns"]), key=lambda a: -camp_size[a])
        print(f"K = {K}: retain {b['n_retained']}/{n} ({b['n_retained']/n:.1%}), "
              f"quarantine {n - b['n_retained']}, folds {b['fold_sizes']} "
              f"bounds [{b['L']}, {b['U']}], clusters {b['clusters']}")
        summary["per_K"][str(K)] = {"feasible": True, "n_retained": b["n_retained"],
                                    "quarantined": n - b["n_retained"],
                                    "fold_sizes": b["fold_sizes"], "L": b["L"], "U": b["U"],
                                    "clusters": b["clusters"],
                                    "kept_types": b["kept_types"],
                                    "quarantined_campaigns": [
                                        {"campaign": a, "records": camp_size[a],
                                         "type_levels": sorted(camp_types[a])} for a in lost]}
        if K == 3:
            pd.DataFrame(summary["per_K"]["3"]["quarantined_campaigns"]).to_csv(
                OUT / "R38_wall_quarantined_campaigns.csv", index=False)

summary["structural_ceiling"] = max([int(k) for k, v in summary["per_K"].items() if v["feasible"]] or [0])
(OUT / "R38_wall_discard_alternative.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"\nstructural ceiling on folds for a two-relation honouring partition: "
      f"{summary['structural_ceiling']}")
print(f"wrote {OUT/'R38_wall_discard_alternative.json'}")
