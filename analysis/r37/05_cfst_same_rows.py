"""R37-05: CFST — evaluate the random control on the *same rows* as the restricted holdout.

The review's acceptance condition: a restricted holdout covering 642 of 1316 records may not be
compared with a random-fold score computed on all 1316. This script recomputes out-of-fold
predictions from the random five-fold split and evaluates them on exactly the rows the restricted
holdout tested, so the two numbers share an evaluation set.

Outputs -> audit/r37/cfst_same_rows.csv, CFST_SAME_ROWS.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
OUT = ROOT / "audit" / "r37"
DATA = PROJ / "code/outputs/reproductions/10-1038-s41598-024-53352-1/combined_stub_cfst_data.csv"
SEED = 0

MODELS = {
    "Ridge": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                               ("m", Ridge(alpha=1.0))]),
    "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "ExtraTrees": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                    ("m", ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "GradientBoosting": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                          ("m", GradientBoostingRegressor(random_state=SEED))]),
}

df = pd.read_csv(DATA, low_memory=False)
fam = df["shape"].astype(str)
sizes = fam.value_counts()
small = sizes.sort_values().index[:2].tolist()          # the two smaller families (396 and 246)
mask_small = fam.isin(small).to_numpy()
print("family sizes:", sizes.to_dict(), "| restricted test families:", small,
      "| rows:", int(mask_small.sum()))

X = df.drop(columns=[c for c in ("P", "shape", "strength_index", "nominal_capacity") if c in df.columns])
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df["P"], errors="coerce").to_numpy()
ok = ~np.isnan(y)
X, y, mask_small = X[ok].reset_index(drop=True), y[ok], mask_small[ok]
print("analytic rows:", len(y))

rows = []
for name, factory in MODELS.items():
    oof = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
        mdl = factory()
        mdl.fit(X.iloc[tr], y[tr])
        oof[te] = mdl.predict(X.iloc[te])
    r2_all = r2_score(y, oof)
    r2_restricted_rows = r2_score(y[mask_small], oof[mask_small])
    # per-family restricted scores, mirroring the two holdouts
    per_family = {f: round(float(r2_score(y[fam[ok].to_numpy() == f], oof[fam[ok].to_numpy() == f])), 3)
                  for f in small}
    rows.append({"model": name, "random5_R2_all_1316": round(float(r2_all), 3),
                 "random5_R2_on_642_rows": round(float(r2_restricted_rows), 3),
                 "random5_R2_per_restricted_family": per_family})

out = pd.DataFrame(rows)
repair = pd.read_csv(ROOT.parent / "R36_independent_review_rebuild_2026-09-16/audit/r36/cfst_repair.csv")
rep = repair[repair.variant == "family_holdout_small2"][["model", "pooled_r2"]].rename(
    columns={"pooled_r2": "restricted_holdout_R2"})
merged = out.merge(rep, on="model", how="left")
merged["delta_same_rows"] = (merged["restricted_holdout_R2"] - merged["random5_R2_on_642_rows"]).round(3)
merged.to_csv(OUT / "cfst_same_rows.csv", index=False, encoding="utf-8")

lines = ["# CFST: comparing on the same evaluated rows", "",
         f"Restricted holdout families: {small} ({int(mask_small.sum())} of {len(y)} records, "
         f"{mask_small.mean():.2%}).", "",
         "| model | random 5-fold, all 1,316 rows | random 5-fold, the same 642 rows | "
         "restricted family holdout (642 rows) | difference on the shared rows |",
         "|---|---:|---:|---:|---:|"]
for _, r in merged.iterrows():
    lines.append(f"| {r['model']} | {r['random5_R2_all_1316']:.3f} | {r['random5_R2_on_642_rows']:.3f} | "
                 f"{r['restricted_holdout_R2']:.3f} | {r['delta_same_rows']:+.3f} |")
lines += ["", "The third and fourth columns share an evaluation set; the first column does not, and is",
          "therefore not a valid comparator for the restricted holdout."]
(OUT / "CFST_SAME_ROWS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[4:]))
