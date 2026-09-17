"""R36-04: admissible repair for the CFST case + the K x fold-upper-bound feasibility map.

The K=3 family split that R35 reported violates the declared upper bound U = 0.5n (674 > 658),
so it cannot be used as evidence. This script
  1) builds the admissible family-honouring evaluation (hold out each SMALLER family in turn,
     leaving the largest family in the training set), with a size-matched random control;
  2) emits the feasibility map over K x f_max so the paper can show exactly which constraint
     makes which K infeasible;
  3) reports what claim each repaired evaluation can and cannot support.
"""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
OUT = ROOT / "audit" / "r36"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
FMIN, FMAX = 0.05, 0.50

LEARNERS = {
    "Ridge": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "ExtraTrees": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "GradientBoosting": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", GradientBoostingRegressor(random_state=SEED))]),
}

csv = PROJ / "code/outputs/reproductions/10-1038-s41598-024-53352-1/combined_stub_cfst_data.csv"
df = pd.read_csv(csv, low_memory=False)
target = "P"
drop = {"strength_index", "nominal_capacity", target, "shape"}
X = df[[c for c in df.columns if c not in drop]].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[target], errors="coerce")
g = df["shape"].astype(str)
n = int(len(df))
sizes = g.value_counts().sort_values(ascending=False)
print(f"components: {sizes.to_dict()} | n={n}")

# ---------------------------------------------------------------- feasible map
rows = []
for K in (2, 3, 4, 5, 10):
    for fmax in (Fraction(1, 2), Fraction(337, 658), Fraction(11, 20), Fraction(3, 5)):
        U = (n * fmax.numerator) // fmax.denominator
        L = math.ceil(FMIN * n)
        reasons = []
        if len(sizes) < K:
            reasons.append("m<K")
        if int(sizes.max()) > U:
            reasons.append("max_w>U")
        if K * U < n:
            reasons.append("K*U<n")
        if K * L > n:
            reasons.append("K*L>n")
        rows.append({"K": K, "f_max": str(fmax), "f_max_dec": round(float(fmax), 6),
                     "U": U, "L": L, "feasible": not reasons, "blocked_by": ",".join(reasons)})
feas = pd.DataFrame(rows)
feas.to_csv(OUT / "cfst_feasibility_map.csv", index=False, encoding="utf-8")

# ---------------------------------------------------------------- admissible repair
small = list(sizes.index)[1:]                      # the two smaller families
repair_splits = []
for fam in small:
    te = np.flatnonzero(g.to_numpy() == fam)
    tr = np.flatnonzero(g.to_numpy() != fam)
    repair_splits.append((tr, te))
repair_sizes = [len(te) for _, te in repair_splits]
print(f"repair folds (hold out each smaller family): {repair_sizes} | "
      f"train fractions {[round(1 - s / n, 3) for s in repair_sizes]}")

rng = np.random.default_rng(SEED)
perm = rng.permutation(n)
sm_splits, start = [], 0
for s in repair_sizes:
    idx = perm[start:start + s]
    start += s
    sm_splits.append((np.setdiff1d(np.arange(n), idx), idx))

base_splits = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))

def run(splits, label):
    out = []
    for name, factory in LEARNERS.items():
        yt, yp, folds = [], [], []
        for k, (tr, te) in enumerate(splits):
            model = factory()
            model.fit(X.iloc[tr], y.iloc[tr])
            p = model.predict(X.iloc[te])
            yt.append(y.iloc[te].to_numpy()); yp.append(p)
            folds.append(r2_score(y.iloc[te], p) if len(te) > 1 else np.nan)
        yt, yp = np.concatenate(yt), np.concatenate(yp)
        out.append({"variant": label, "model": name, "K": len(splits),
                    "pooled_r2": round(float(r2_score(yt, yp)), 4),
                    "rmse": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 2),
                    "mae": round(float(mean_absolute_error(yt, yp)), 2),
                    "fold_mean_r2": round(float(np.nanmean(folds)), 4)})
    return out

res = []
res += run(base_splits, "random5")
res += run(repair_splits, "family_holdout_small2")
res += run(sm_splits, "sizematch_family_small2")
old = list(GroupKFold(n_splits=3).split(X, y, g))
res += run(old, "family_holdout_K3_inadmissible")
tbl = pd.DataFrame(res)
tbl.to_csv(OUT / "cfst_repair.csv", index=False, encoding="utf-8")

lines = ["# CFST: feasibility map and admissible repair", "",
         f"components {sizes.to_dict()}, n = {n}, declared bounds [L,U] = [{math.ceil(FMIN*n)}, {math.floor(FMAX*n)}]", "",
         "## K x fold-upper-bound feasibility", "",
         "| K | f_max | U | feasible | blocked by |", "|---:|---|---:|---|---|"]
for _, r in feas.iterrows():
    lines.append(f"| {r['K']} | {r['f_max']} ({r['f_max_dec']:.6f}) | {r['U']} | "
                 f"{'yes' if r['feasible'] else 'no'} | {r['blocked_by']} |")
lines += ["", "## Admissible repair vs the inadmissible K=3 split", "",
          "| variant | model | K | pooled R2 | RMSE | fold-mean R2 |", "|---|---|---:|---:|---:|---:|"]
for _, r in tbl.iterrows():
    lines.append(f"| {r['variant']} | {r['model']} | {r['K']} | {r['pooled_r2']:.3f} | "
                 f"{r['rmse']:.1f} | {r['fold_mean_r2']:.3f} |")
lines += ["",
          "Reading: the repaired evaluation holds out each of the two smaller families in turn "
          "(396 and 246 records) and keeps the 674-record family in the training set, so every fold "
          "respects [L,U] and the training fractions (0.699, 0.813) are comparable to the random "
          "five-fold baseline (0.800). The repair does NOT test transfer to the largest family; that "
          "claim remains unavailable at the declared bound and is reported as such."]

(OUT / "CFST_REPAIR.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[6:]))
