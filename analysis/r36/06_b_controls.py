"""R36-06 (work package B): classification controls, source-equal vs record-weighted error,
and the per-row predictions needed to recompute both.

Adds to the R36 evidence:
  * failure-mode classification under random folds, author holdout and a size-matched random
    control, with per-fold class support and balanced accuracy;
  * for the regression configurations: record-weighted vs source-equal MSE/RMSE (equal weight
    per provenance component), which the independent review asked for;
  * per-row out-of-fold predictions saved to audit/r36 for re-derivation.

Outputs -> audit/r36/{B_controls.md, control_predictions.parquet, classification_controls.csv,
                     weighting_controls.csv}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
AUD = ROOT / "audit" / "r36"
OUT = ROOT / "analysis" / "out"
SEED, K_MAX = 0, 5
REPRO = PROJ / "code" / "outputs" / "reproductions"
DATASETS = PROJ / "code" / "outputs" / "datasets"

REG = [
    dict(name="stub_cfst", path=REPRO / "10-1038-s41598-024-53352-1/combined_stub_cfst_data.csv",
         target="P", rel="shape", drop=["strength_index", "nominal_capacity"]),
    dict(name="corroded_rc_beam", path=REPRO / "10-5281-zenodo-8062007/analysis_data.csv",
         target="log_mmax_exp", rel="source_group", drop=[]),
    dict(name="wall_prj2430_Vmax", path=REPRO / "designsafe_prj2430_wall/extracted_wall_data.csv",
         target="Vmax", rel="Authors", drop=["Dmax", "driftCap", "SpecimenID", "UniqueID", "row_index"]),
    dict(name="wall_prj2430_drift", path=REPRO / "designsafe_prj2430_wall/extracted_wall_data.csv",
         target="driftCap", rel="Authors", drop=["Dmax", "Vmax", "SpecimenID", "UniqueID", "row_index"]),
    dict(name="coupling_beams_prj3053",
         path=REPRO / "designsafe_prj3053_coupling_beams/extracted_coupling_beam_data.csv",
         target="V_m_avg_kips", rel="source_group",
         drop=["specimen_id", "reference_number", "v_m_normalized"]),
]
CLF = dict(name="shear_wall_failure_mode",
           path=DATASETS / "mangalathu_2020_shear_wall/Shear_Wall_Database.xlsx",
           target="FailureMode", rel="Author", drop=["Specimen"])

REGRESSORS = {
    "Ridge": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "ExtraTrees": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "GradientBoosting": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", GradientBoostingRegressor(random_state=SEED))]),
}
CLASSIFIERS = {
    "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")),
                                      ("m", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1))]),
}


def frame(path: Path, target: str, rel: str | None, drop: list[str]):
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, low_memory=False)
    drop_set = set(drop) | {target} | ({rel} if rel else set())
    cols = [c for c in df.columns if c not in drop_set]
    X = df[cols].copy()
    for c in [c for c in X.columns if X[c].dtype == object]:
        if X[c].nunique(dropna=True) <= 12:
            X = pd.concat([X.drop(columns=[c]), pd.get_dummies(X[c], prefix=c, dummy_na=True)], axis=1)
        else:
            X = X.drop(columns=[c])
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    ok = y.notna()
    g = df.loc[ok, rel].astype(str).reset_index(drop=True) if rel else None
    return X[ok].reset_index(drop=True), y[ok].reset_index(drop=True), g


def size_match_splits(sizes, n, seed=SEED):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    out, start = [], 0
    for s in sizes:
        te = perm[start:start + s]
        start += s
        out.append((np.setdiff1d(np.arange(n), te), te))
    return out


pred_rows, weight_rows = [], []
for spec in REG:
    if not spec["path"].exists():
        print(f"[skip] {spec['name']}")
        continue
    X, y, g = frame(spec["path"], spec["target"], spec["rel"], spec["drop"])
    n = len(y)
    variants = {"random5": list(KFold(K_MAX, shuffle=True, random_state=SEED).split(X, y))}
    ng = int(g.nunique())
    hon = list(GroupKFold(min(K_MAX, ng)).split(X, y, g))
    variants["honoured"] = hon
    variants["sizematch"] = size_match_splits([len(te) for _, te in hon], n)
    for vname, splits in variants.items():
        for lname, factory in REGRESSORS.items():
            for k, (tr, te) in enumerate(splits):
                if len(tr) == 0 or len(te) == 0:
                    continue
                model = factory()
                model.fit(X.iloc[tr], y.iloc[tr])
                p = model.predict(X.iloc[te])
                for i, pv, tv in zip(te, p, y.iloc[te].to_numpy()):
                    pred_rows.append({"dataset": spec["name"], "variant": vname, "model": lname,
                                      "fold": k, "row": int(i), "group": str(g.iloc[i]),
                                      "y_true": float(tv), "y_pred": float(pv)})
            sub = pd.DataFrame([r for r in pred_rows if r["dataset"] == spec["name"]
                                and r["variant"] == vname and r["model"] == lname])
            if sub.empty:
                continue
            rec_mse = float(np.mean((sub.y_true - sub.y_pred) ** 2))
            per_source = sub.groupby("group").apply(
                lambda d: float(np.mean((d.y_true - d.y_pred) ** 2)), include_groups=False)
            weight_rows.append({
                "dataset": spec["name"], "variant": vname, "model": lname,
                "n": int(len(sub)), "n_sources": int(per_source.size),
                "mse_record_weighted": round(rec_mse, 5),
                "mse_source_equal": round(float(per_source.mean()), 5),
                "rmse_record_weighted": round(float(np.sqrt(rec_mse)), 4),
                "rmse_source_equal": round(float(np.sqrt(per_source.mean())), 4),
                "ratio_source_equal_over_record": round(float(per_source.mean() / rec_mse), 3),
            })

preds = pd.DataFrame(pred_rows)
preds.to_parquet(AUD / "control_predictions.parquet", index=False)
w = pd.DataFrame(weight_rows)
w.to_csv(AUD / "weighting_controls.csv", index=False, encoding="utf-8")

# ---- classification controls ---------------------------------------------------
cls_rows, per_fold = [], []
X, y, g = frame(CLF["path"], CLF["target"], CLF["rel"], CLF["drop"])
n = len(y)
ng = int(g.nunique())
variants = {"random5": list(KFold(K_MAX, shuffle=True, random_state=SEED).split(X, y))}
variants["honoured"] = list(GroupKFold(min(K_MAX, ng)).split(X, y, g))
variants["sizematch"] = size_match_splits([len(te) for _, te in variants["honoured"]], n)
for vname, splits in variants.items():
    for lname, factory in CLASSIFIERS.items():
        accs, bals = [], []
        for k, (tr, te) in enumerate(splits):
            model = factory()
            model.fit(X.iloc[tr], y.iloc[tr])
            p = model.predict(X.iloc[te])
            accs.append(accuracy_score(y.iloc[te], p))
            bals.append(balanced_accuracy_score(y.iloc[te], p))
            support = pd.Series(y.iloc[te]).value_counts().to_dict()
            per_fold.append({"dataset": CLF["name"], "variant": vname, "model": lname, "fold": k,
                             "n_test": int(len(te)), "n_train": int(len(tr)),
                             "classes_in_test": len(support),
                             "classes_in_train": int(pd.Series(y.iloc[tr]).nunique()),
                             "min_class_support_test": int(min(support.values())),
                             "accuracy": round(float(accs[-1]), 4),
                             "balanced_accuracy": round(float(bals[-1]), 4)})
        cls_rows.append({"dataset": CLF["name"], "variant": vname, "model": lname,
                         "n": int(len(y)), "n_groups": ng,
                         "accuracy_mean": round(float(np.mean(accs)), 4),
                         "accuracy_sd": round(float(np.std(accs)), 4),
                         "balanced_accuracy_mean": round(float(np.mean(bals)), 4),
                         "balanced_accuracy_sd": round(float(np.std(bals)), 4),
                         "K": len(splits)})
c = pd.DataFrame(cls_rows)
c.to_csv(AUD / "classification_controls.csv", index=False, encoding="utf-8")
pf = pd.DataFrame(per_fold)
pf.to_csv(AUD / "classification_per_fold.csv", index=False, encoding="utf-8")

lines = ["# R36 work package B controls", "",
         "## Classification: failure-mode prediction under three split families", "",
         "| variant | model | K | accuracy (mean ± sd) | balanced accuracy (mean ± sd) |",
         "|---|---|---:|---|---|"]
for _, r in c.iterrows():
    lines.append(f"| {r['variant']} | {r['model']} | {r['K']} | "
                 f"{r['accuracy_mean']:.3f} ± {r['accuracy_sd']:.3f} | "
                 f"{r['balanced_accuracy_mean']:.3f} ± {r['balanced_accuracy_sd']:.3f} |")
lines += ["", "Per-fold class support (all folds must contain every class for balanced accuracy to be",
          "meaningful):", "", "| variant | fold | n train | n test | classes in test | min class support | acc | bal acc |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
for _, r in pf.iterrows():
    lines.append(f"| {r['variant']} | {r['fold']} | {r['n_train']} | {r['n_test']} | "
                 f"{r['classes_in_test']} | {r['min_class_support_test']} | {r['accuracy']:.3f} | "
                 f"{r['balanced_accuracy']:.3f} |")
lines += ["", "## Record-weighted vs source-equal error (regression)", "",
          "| dataset | variant | model | n | sources | RMSE record-weighted | RMSE source-equal | ratio |",
          "|---|---|---|---:|---:|---:|---:|---:|"]
for _, r in w.sort_values(["dataset", "variant", "model"]).iterrows():
    lines.append(f"| {r['dataset']} | {r['variant']} | {r['model']} | {r['n']} | {r['n_sources']} | "
                 f"{r['rmse_record_weighted']:.3f} | {r['rmse_source_equal']:.3f} | "
                 f"{r['ratio_source_equal_over_record']:.2f} |")
lines += ["", "A ratio above 1 means the honoured split looks worse when every provenance component counts",
          "equally, i.e. record weighting understates the damage for small programmes."]
(AUD / "B_controls.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:16]))
print(f"\nwrote {AUD/'B_controls.md'}, control_predictions.parquet, weighting_controls.csv, classification_controls.csv")
