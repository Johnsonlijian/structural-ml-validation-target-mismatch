"""R36-02: the missing controls demanded by the independent review.

For every dataset in the protocol-controlled experiment this script now produces, under one
learner set and one metric definition:

  random5     KFold(5, shuffle, seed 0)                          -- the practice baseline
  honoured    GroupKFold on the provenance relation              -- the entitled split
  sizematch   random folds with EXACTLY the honoured fold sizes  -- isolates training-size
  dummy       per-fold training-mean predictor                   -- reference for R2 < 0

plus two case-specific controls:
  wall_combo  RC wall: test folds must not repeat a seen (source, wall-type) COMBINATION
              (the weaker claim that naive key concatenation actually implements)
  uci_428 / uci_996 / uci_shuffled  UCI concrete: seven-ingredient key, eight-input key, and a
              size-matched shuffled-label negative control

Outputs -> audit/r36/{control_metrics.csv, fold_structure.csv, CONTROL_SUMMARY.md}
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
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
K_MAX = 5

SPECS = [
    {
        "dataset": "stub_cfst",
        "path": "code/outputs/reproductions/10-1038-s41598-024-53352-1/combined_stub_cfst_data.csv",
        "target": "P", "relation": "shape", "proxy": None,
        "drop": ["strength_index", "nominal_capacity"], "relation_semantics": "section family (proxy)",
    },
    {
        "dataset": "corroded_rc_beam",
        "path": "code/outputs/reproductions/10-5281-zenodo-8062007/analysis_data.csv",
        "target": "log_mmax_exp", "relation": "source_group", "proxy": None,
        "drop": [], "relation_semantics": "experimental programme (source-level)",
    },
    {
        "dataset": "wall_prj2430_Vmax",
        "path": "code/outputs/reproductions/designsafe_prj2430_wall/extracted_wall_data.csv",
        "target": "Vmax", "relation": "Authors", "proxy": "walltype_Shape",
        "drop": ["Dmax", "driftCap", "SpecimenID", "UniqueID", "row_index"],
        "relation_semantics": "author (source-level); wall type (proxy)",
    },
    {
        "dataset": "wall_prj2430_drift",
        "path": "code/outputs/reproductions/designsafe_prj2430_wall/extracted_wall_data.csv",
        "target": "driftCap", "relation": "Authors", "proxy": "walltype_Shape",
        "drop": ["Dmax", "Vmax", "SpecimenID", "UniqueID", "row_index"],
        "relation_semantics": "author (source-level); wall type (proxy)",
    },
    {
        "dataset": "coupling_beams_prj3053",
        "path": "code/outputs/reproductions/designsafe_prj3053_coupling_beams/extracted_coupling_beam_data.csv",
        "target": "V_m_avg_kips", "relation": "source_group", "proxy": None,
        "drop": ["specimen_id", "reference_number", "v_m_normalized"],
        "relation_semantics": "reference (source-level)",
    },
    {
        "dataset": "concrete_strength",
        "path": "code/outputs/datasets/uci_concrete_compressive_strength/Concrete_Data.xls",
        "target": "__last__", "relation": None, "proxy": "__mix__",
        "drop": [], "relation_semantics": "identical mix proportions (duplication proxy)",
    },
]

LEARNERS = {
    "Ridge": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "ExtraTrees": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "GradientBoosting": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", GradientBoostingRegressor(random_state=SEED))]),
}
DUMMY = lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", DummyRegressor(strategy="mean"))])


def load(spec: dict) -> pd.DataFrame:
    path = PROJ / spec["path"]
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, low_memory=False)
    if spec["target"] == "__last__":
        spec["target"] = df.columns[-1]
    return df


def build_xy(df: pd.DataFrame, spec: dict):
    target = spec["target"]
    drop = set(spec["drop"]) | {target}
    for g in (spec.get("relation"), spec.get("proxy")):
        if g and g != "__mix__":
            drop.add(g)
    feats = [c for c in df.columns if c not in drop]
    X = df[feats].copy()
    for c in [c for c in X.columns if X[c].dtype == object]:
        if X[c].nunique(dropna=True) <= 12:
            X = pd.concat([X.drop(columns=[c]), pd.get_dummies(X[c], prefix=c, dummy_na=True)], axis=1)
        else:
            X = X.drop(columns=[c])
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target], errors="coerce")
    ok = y.notna()
    return X[ok].reset_index(drop=True), y[ok].reset_index(drop=True), ok


def group_series(df: pd.DataFrame, spec: dict, ok: pd.Series, kind: str):
    if kind == "relation":
        if not spec.get("relation"):
            return None
        return df.loc[ok, spec["relation"]].astype(str).reset_index(drop=True)
    if spec.get("proxy") == "__mix__":
        cols = [c for c in df.columns if c != spec["target"]]
        return df.loc[ok, cols].astype(str).agg("|".join, axis=1).reset_index(drop=True)
    if spec.get("proxy"):
        return df.loc[ok, spec["proxy"]].astype(str).reset_index(drop=True)
    return None


def evaluate(X, y, splits, label, dataset, rows, fold_rows, learner_factories):
    for name, factory in learner_factories.items():
        preds, fold_scores = [], []
        for k, (tr, te) in enumerate(splits):
            if len(tr) == 0 or len(te) == 0:
                continue
            model = factory()
            model.fit(X.iloc[tr], y.iloc[tr])
            p = model.predict(X.iloc[te])
            preds.append((te, p, y.iloc[te].to_numpy()))
            fold_scores.append(r2_score(y.iloc[te], p) if len(te) > 1 else np.nan)
            fold_rows.append({"dataset": dataset, "variant": label, "fold": k,
                              "n_train": int(len(tr)), "n_test": int(len(te)),
                              "train_fraction": round(len(tr) / len(y), 4)})
        if not preds:
            continue
        yt = np.concatenate([p[2] for p in preds])
        yp = np.concatenate([p[1] for p in preds])
        rows.append({
            "dataset": dataset, "variant": label, "model": name,
            "n": int(yt.size),
            "pooled_r2": round(float(r2_score(yt, yp)), 4),
            "rmse": round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4),
            "mae": round(float(mean_absolute_error(yt, yp)), 4),
            "fold_mean_r2": round(float(np.nanmean(fold_scores)), 4),
            "fold_sd_r2": round(float(np.nanstd(fold_scores)), 4),
            "n_folds": int(len(preds)),
        })


def main() -> None:
    rows: list[dict] = []
    fold_rows: list[dict] = []
    receipt: dict = {"seed": SEED, "k_max": K_MAX, "inputs": {}, "cases": {}}

    for spec in SPECS:
        path = PROJ / spec["path"]
        if not path.exists():
            print(f"[skip] {spec['dataset']}: missing input")
            continue
        receipt["inputs"][spec["dataset"]] = {"path": spec["path"],
                                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        df = load(spec)
        X, y, ok = build_xy(df, spec)
        rand_splits = list(KFold(n_splits=K_MAX, shuffle=True, random_state=SEED).split(X, y))
        evaluate(X, y, rand_splits, "random5", spec["dataset"], rows, fold_rows, LEARNERS)
        evaluate(X, y, rand_splits, "dummy_trainmean", spec["dataset"], rows, fold_rows, {"Dummy": DUMMY})

        relations = {"relation": group_series(df, spec, ok, "relation"),
                     "proxy": group_series(df, spec, ok, "proxy")}
        for kind, g in relations.items():
            if g is None:
                continue
            ng = int(g.nunique())
            if ng < 3:
                continue
            splits = list(GroupKFold(n_splits=min(K_MAX, ng)).split(X, y, g))
            label = {"relation": "honoured", "proxy": "honoured_proxy"}[kind]
            evaluate(X, y, splits, label, spec["dataset"], rows, fold_rows, LEARNERS)
            # size-matched random control: same K, identical test-fold sizes
            sizes = [len(te) for _, te in splits]
            rng = np.random.default_rng(SEED)
            perm = rng.permutation(len(y))
            sm_splits, start = [], 0
            for s in sizes:
                idx = perm[start:start + s]
                start += s
                sm_splits.append((np.setdiff1d(np.arange(len(y)), idx), idx))
            evaluate(X, y, sm_splits, f"sizematch_{kind}", spec["dataset"], rows, fold_rows, LEARNERS)
            receipt["cases"].setdefault(spec["dataset"], {})[label] = {
                "n_groups": ng, "K": len(splits), "fold_sizes": sizes,
                "train_fractions": [round(1 - s / len(y), 4) for s in sizes],
            }
            print(f"[{spec['dataset']}] {label}: groups={ng} K={len(splits)} "
                  f"fold sizes={sizes} train_frac={[round(1 - s / len(y), 3) for s in sizes]}")

        # wall: combination-level claim (the weaker claim concatenation implements)
        if spec["dataset"].startswith("wall_") and relations["relation"] is not None and relations["proxy"] is not None:
            combo = relations["relation"] + " || " + relations["proxy"]
            nc = int(combo.nunique())
            splits = list(GroupKFold(n_splits=min(K_MAX, nc)).split(X, y, combo))
            evaluate(X, y, splits, "wall_combo", spec["dataset"], rows, fold_rows, LEARNERS)
            receipt["cases"].setdefault(spec["dataset"], {})["wall_combo"] = {
                "n_combinations": nc, "K": len(splits),
                "fold_sizes": [len(te) for _, te in splits],
            }
            print(f"[{spec['dataset']}] wall_combo: combinations={nc} K={len(splits)}")

    # UCI relation-definition variants + shuffled negative control
    uci = PROJ / "code/outputs/datasets/uci_concrete_compressive_strength/Concrete_Data.xls"
    if uci.exists():
        df = pd.read_excel(uci)
        target = df.columns[-1]
        X = df[[c for c in df.columns if c != target]].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(df[target], errors="coerce")
        ingredients = [c for c in df.columns if c != target and "Age" not in c]
        keys = {
            "uci_428_ingredients": df[ingredients].astype(str).agg("|".join, axis=1),
            "uci_996_allinputs": df[[c for c in df.columns if c != target]].astype(str).agg("|".join, axis=1),
        }
        for label, g in keys.items():
            ng = int(g.nunique())
            splits = list(GroupKFold(n_splits=min(K_MAX, ng)).split(X, y, g))
            evaluate(X, y, splits, label, "concrete_strength", rows, fold_rows, LEARNERS)
            sizes = [len(te) for _, te in splits]
            rng = np.random.default_rng(SEED)
            cells = np.repeat(g.to_numpy(), 1)
            # shuffled-label control: keep the group-size multiset, destroy the link to rows
            shuffled = rng.permutation(cells)
            s_splits = list(GroupKFold(n_splits=min(K_MAX, ng)).split(X, y, shuffled))
            evaluate(X, y, s_splits, f"{label}_shuffledlabels", "concrete_strength", rows, fold_rows, LEARNERS)
            receipt["cases"].setdefault("concrete_strength", {})[label] = {
                "n_groups": ng, "K": len(splits), "fold_sizes": sizes,
            }
            print(f"[concrete_strength] {label}: groups={ng} K={len(splits)} (plus shuffled-label control)")

    metrics = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows).drop_duplicates()
    metrics.to_csv(OUT / "control_metrics.csv", index=False, encoding="utf-8")
    folds.to_csv(OUT / "fold_structure.csv", index=False, encoding="utf-8")
    (OUT / "RUN_RECEIPT.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nmetric rows: {len(metrics)} | fold rows: {len(folds)} -> {OUT}")


if __name__ == "__main__":
    main()
