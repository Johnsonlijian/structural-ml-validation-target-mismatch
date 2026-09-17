"""R38-74: run the open comparison — the audit wrapper against a correctly configured GroupKFold.

What the two policies share: the same outer source holdout, the same candidate rule space, the same inner
estimates, and the same nominal target alpha. They differ only in the selection policy:

  * **G, the grouped-only baseline.** A correctly configured GroupKFold: grouped inner folds, and the
    candidate with the largest inner coverage among those whose inner unsafe-release rate meets the
    target. This is exactly the released "source" arm. When no candidate meets the target the released
    code falls back to the lowest-risk candidate as a diagnostic; the variant G-refuse is reported too.
  * **A, the audit-wrapped policy.** The same grouped inner folds, but the audit's gates run first: the
    inner split must satisfy the fold-fraction bounds [L, U] and the group-count certificate m < K, and if
    no candidate meets the target the policy refuses to release rather than falling back.

The comparison therefore answers a precise question: in these settings, does the audit's machinery change
any decision or any realised outcome, once the baseline is already configured correctly?

This is a re-analysis of predictions already produced, not a new training run: refusal simply means no
record is released, so no re-fitting is involved and nothing is claimed about new data.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
R37 = ROOT.parent / "R37_second_review_repair_2026-09-16"
REPRO = PROJ / "code/outputs/reproductions"
DATASETS = PROJ / "code/outputs/datasets"
OUT = ROOT / "analysis" / "out"
SEED, K_IN, K_OUT = 0, 5, 5
ALPHAS = (0.05, 0.10, 0.20)


def design(path: Path, target: str, rel: str, drop: list[str]):
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, low_memory=False)
    keep = [c for c in df.columns if c not in set(drop) | {target, rel}]
    X = df[keep].copy()
    for c in [c for c in X.columns if X[c].dtype == object]:
        X = X.drop(columns=[c]) if X[c].nunique(dropna=True) > 12 else pd.concat(
            [X.drop(columns=[c]), pd.get_dummies(X[c], prefix=c, dummy_na=True)], axis=1)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    ok = y.notna()
    return X[ok].reset_index(drop=True), y[ok].reset_index(drop=True), df.loc[ok, rel].astype(str).reset_index(drop=True)


def inner_splits(strategy, X, y, g):
    return (list(KFold(K_IN, shuffle=True, random_state=SEED).split(X, y)) if strategy == "random"
            else list(GroupKFold(min(K_IN, int(g.nunique()))).split(X, y, g)))


print("=== inner-split admissibility (the audit's first gate) ===")
admissibility = []
for scenario, path, target, rel, drop in (
        ("beam_release", REPRO / "10-5281-zenodo-8062007/analysis_data.csv", "log_mmax_exp", "source_group", []),
        ("failure_mode", DATASETS / "mangalathu_2020_shear_wall/Shear_Wall_Database.xlsx", "FailureMode", "Author", ["Specimen"])):
    X, y, g = design(path, target, rel, drop)
    outer = list(GroupKFold(min(K_OUT, int(g.nunique()))).split(X, y, g))
    for of, (tr_all, te_all) in enumerate(outer):
        n = len(tr_all)
        L = math.ceil(0.05 * n)
        U = math.floor(0.50 * n)
        for strategy in ("random", "source"):
            folds = inner_splits(strategy, X.iloc[tr_all], y.iloc[tr_all], g.iloc[tr_all])
            sizes = [len(ite) for _, ite in folds]
            groups_per_fold = [int(g.iloc[tr_all].iloc[ite].nunique()) for _, ite in folds]
            m = int(g.iloc[tr_all].nunique())
            fold_ok = all(L <= s <= U for s in sizes)
            # the certificate is m < K => infeasible, so feasibility needs m >= K (groups, not rows)
            group_ok = (strategy == "random") or (m >= len(folds))
            admissibility.append({"scenario": scenario, "outer_fold": of, "protocol": strategy,
                                  "n_train": n, "L": L, "U": U, "fold_sizes": sizes,
                                  "groups": m, "groups_per_fold": groups_per_fold,
                                  "fold_size_ok": fold_ok, "group_cert_ok": group_ok,
                                  "admissible": bool(fold_ok and group_ok)})
            print(f"  {scenario:<14} fold {of} {strategy:<7} n={n:>4} L={L:>3} U={U:>3} "
                  f"sizes={sizes} groups={m} -> {'admissible' if fold_ok and group_ok else 'INADMISSIBLE'}")
adm = pd.DataFrame(admissibility)
adm.to_csv(OUT / "R38_baseline_admissibility.csv", index=False)

# ---------------------------------------------------------------- the two selection policies
cand = pd.read_csv(R37 / "audit/r37/decision_candidates.csv")
outer = pd.read_csv(R37 / "audit/r37/decision_outer.csv")
cand["action_set"] = cand["action_set"].fillna("regression")
outer["action_set"] = outer["action_set"].fillna("regression")
keys = ["scenario", "action_set", "outer_fold", "protocol", "alpha"]

adm_map = {(r.scenario, r.outer_fold, r.protocol): bool(r.admissible) for r in adm.itertuples()}
rows = []
for key, grp in cand.groupby(keys):
    scenario, aset, fold, protocol, alpha = key
    feas = grp[grp["feasible"].fillna(False)]
    # G: max inner coverage among feasible; if none, lowest inner risk (released behaviour)
    if len(feas):
        g_sel = feas.sort_values(["inner_coverage", "inner_r_all"], ascending=[False, True]).iloc[0]
        g_mode = "release"
    else:
        g_sel = grp.sort_values("inner_r_all").iloc[0]
        g_mode = "fallback_min_risk"
    # G-refuse variant
    gr_mode = "release" if len(feas) else "refuse"
    # A: audit gates first
    admissible = adm_map.get((scenario, fold, protocol), True)
    if not admissible or not len(feas):
        a_mode = "refuse"
        a_sel = None
    else:
        a_mode = "release"
        a_sel = g_sel
    o = outer[(outer.scenario == scenario) & (outer.action_set == aset) & (outer.outer_fold == fold) &
              (outer.protocol == protocol) & (np.isclose(outer.alpha, alpha))]
    n_test = int(o.n_test.iloc[0]) if len(o) else int(grp.get("n_test", pd.Series([np.nan])).iloc[0]) if "n_test" in grp else 0
    # the realised outcome of the released baseline choice
    if len(o):
        o_row = o.iloc[0]
        g_rel, g_uns = int(o_row.n_released), int(o_row.n_unsafe)
    else:
        g_rel = g_uns = 0
    a_rel, a_uns = (0, 0) if a_mode == "refuse" else (g_rel, g_uns)
    same_rule = (a_mode == "release") or False
    rows.append({"scenario": scenario, "action_set": aset, "outer_fold": fold, "protocol": protocol,
                 "alpha": float(alpha), "n_candidates": len(grp), "n_feasible": len(feas),
                 "admissible": admissible, "G_mode": g_mode, "G_refuse_mode": gr_mode, "A_mode": a_mode,
                 "G_model": g_sel.get("model"), "G_margin": g_sel.get("margin"),
                 "G_released": g_rel, "G_unsafe": g_uns,
                 "A_released": a_rel, "A_unsafe": a_uns,
                 "same_choice": bool(a_mode == "release" and len(feas))})
cmp = pd.DataFrame(rows)
cmp.to_csv(OUT / "R38_baseline_vs_audit.csv", index=False)

print("\n=== where the two policies differ ===")
diff = cmp[cmp.G_mode != cmp.A_mode]
print(f"settings compared: {len(cmp)} | policies differ in: {len(diff)}")
if len(diff):
    print(diff[["scenario", "action_set", "outer_fold", "protocol", "alpha", "n_feasible",
                "admissible", "G_mode", "A_mode"]].to_string(index=False))
print(f"\nsettings with an empty feasible set (baseline falls back, audit refuses): "
      f"{int((cmp.G_mode == 'fallback_min_risk').sum())}")
print(f"settings where the audit's admissibility gate bites: "
      f"{int((~cmp.admissible).sum())} of {len(cmp)}")

for label, col_rel, col_uns in (("G (grouped baseline)", "G_released", "G_unsafe"),
                                ("A (audit-wrapped)", "A_released", "A_unsafe")):
    for scen in cmp.scenario.unique():
        for aset in cmp[cmp.scenario == scen].action_set.unique():
            sub = cmp[(cmp.scenario == scen) & (cmp.action_set == aset) & (cmp.protocol == "source")]
            if not len(sub):
                continue
            print(f"  {label:<20} {scen:<14} {aset:<24} released {int(sub[col_rel].sum()):>4} "
                  f"unsafe {int(sub[col_uns].sum()):>3}")

json.dump({"settings": len(cmp), "differing": int(len(diff)),
           "empty_feasible_sets": int((cmp.G_mode == 'fallback_min_risk').sum()),
           "inadmissible": int((~cmp.admissible).sum())},
          open(OUT / "R38_baseline_vs_audit_summary.json", "w"), indent=2)
print(f"\nwrote {OUT/'R38_baseline_vs_audit.csv'}")
