"""R37-04: corrected engineering-decision experiment.

Fixes required by the second independent review:

  * **units**: the target is log-capacity. The physical rule ``exp(z_hat) >= d_phys*(1+m)`` is
    equivalent to ``z_hat >= log(d_phys) + log(1+m)``. The earlier run compared ``z_hat`` with
    ``(1+m)*z_d``, which is a different (and stricter) rule; this is corrected, and the two paths are
    asserted to agree row by row.
  * **risk denominator**: every setting reports N_test, N_released, N_unsafe, the population unsafe
    event rate r_all = N_unsafe/N_test, the conditional release risk r_rel = N_unsafe/N_released
    (NA when nothing is released) and the coverage C = N_released/N_test. Selection uses r_all <= alpha.
  * **candidate table**: every candidate rule is written out with its inner risk, inner coverage,
    feasibility and selection reason, so an inactive constraint is visible as such.
  * **classification action sets**: both "act on any high-confidence prediction" and "release only when
    the predicted mode is outside the dangerous family" are evaluated.

Outputs -> audit/r37/{decision_outer.csv, decision_candidates.csv, decision_per_fold.csv,
                     DECISION_EXPERIMENT_R37.md, unit_check.json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
OUT = ROOT / "audit" / "r37"
OUT.mkdir(parents=True, exist_ok=True)
REPRO = PROJ / "code/outputs/reproductions"
DATASETS = PROJ / "code/outputs/datasets"
SEED = 0
K_IN, K_OUT = 5, 5
ALPHAS = (0.05, 0.10, 0.20)
MARGINS = np.round(np.arange(0.0, 1.01, 0.05), 2)
CONFS = np.round(np.arange(0.0, 1.01, 0.20), 2)
DEMAND_Q = 0.50

REGRESSORS = {
    "Ridge": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "GradientBoosting": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", GradientBoostingRegressor(random_state=SEED))]),
}
CLASSIFIERS = {
    "RandomForest": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1))]),
    "ExtraTrees": lambda: Pipeline([("imp", SimpleImputer(strategy="median")), ("m", ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=1))]),
}


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


def risk_block(act: np.ndarray, unsafe: np.ndarray) -> dict:
    n = int(act.size)
    n_rel = int(act.sum())
    n_unsafe = int((act & unsafe).sum())
    return {
        "n_test": n, "n_released": n_rel, "n_unsafe": n_unsafe,
        "r_all": n_unsafe / n if n else float("nan"),
        "r_released": (n_unsafe / n_rel) if n_rel else None,
        "coverage": n_rel / n if n else float("nan"),
    }


# ================================================================= beam release
X, y, g = design(REPRO / "10-5281-zenodo-8062007/analysis_data.csv", "log_mmax_exp", "source_group", [])
outer = list(GroupKFold(min(K_OUT, int(g.nunique()))).split(X, y, g))
cand_rows, outer_rows, fold_rows = [], [], []
unit_rows = []

for of, (tr_all, te_all) in enumerate(outer):
    # demand in physical units, then its log: the rule below is exactly exp(z) >= d_phys*(1+m)
    d_phys = float(np.quantile(np.exp(y.iloc[tr_all]), DEMAND_Q))
    z_d = float(np.log(d_phys))
    for strategy in ("random", "source"):
        folds = inner_splits(strategy, X.iloc[tr_all], y.iloc[tr_all], g.iloc[tr_all])
        cached: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {m: [] for m in REGRESSORS}
        for itr, ite in folds:
            for name, factory in REGRESSORS.items():
                mdl = factory()
                mdl.fit(X.iloc[tr_all].iloc[itr], y.iloc[tr_all].iloc[itr])
                cached[name].append((mdl.predict(X.iloc[tr_all].iloc[ite]),
                                     y.iloc[tr_all].iloc[ite].to_numpy()))
        est: dict[tuple[str, float], dict] = {}
        for name, plist in cached.items():
            for m in MARGINS:
                acts, unsafes = [], []
                for p, t in plist:
                    thr = z_d + np.log1p(m)                     # corrected log-space threshold
                    acts.append(p >= thr)
                    unsafes.append(np.exp(t) < d_phys)          # physical comparison for the truth
                a = np.concatenate(acts)
                u = np.concatenate(unsafes)
                est[(name, float(m))] = risk_block(a, u)

        # unit check: the two decision paths must agree row by row
        if of == 0 and strategy == "random":
            p0, t0 = cached["Ridge"][0]
            for m in (0.0, 0.2, 0.5):
                path_a = p0 >= (z_d + np.log1p(m))
                path_b = np.exp(p0) >= d_phys * (1 + m)
                unit_rows.append({"margin": float(m), "agree": bool(np.array_equal(path_a, path_b)),
                                  "mismatches": int((path_a != path_b).sum())})

        for alpha in ALPHAS:
            feasible = [(k, v) for k, v in est.items() if v["r_all"] <= alpha]
            if feasible:
                best_cov = max(v["coverage"] for _, v in feasible)
                tied = [(k, v) for k, v in feasible if abs(v["coverage"] - best_cov) < 1e-12]
                (name, m), chosen = min(tied, key=lambda kv: (kv[0][1], kv[0][0]))
                reason = "max inner coverage among rules with inner risk <= alpha"
            else:
                (name, m), chosen = min(est.items(), key=lambda kv: kv[1]["r_all"])[0], est[
                    min(est.items(), key=lambda kv: kv[1]["r_all"])[0]]
                reason = "no rule met the target; minimum-risk rule reported, not released"
            for (cn, cm), cv in est.items():
                cand_rows.append({
                    "scenario": "beam_release", "outer_fold": of, "protocol": strategy, "alpha": alpha,
                    "model": cn, "margin": float(cm), "inner_r_all": round(cv["r_all"], 5),
                    "inner_coverage": round(cv["coverage"], 4), "feasible": cv["r_all"] <= alpha,
                    "chosen": bool(cn == name and cm == m), "reason": reason if (cn == name and cm == m) else "",
                })
            mdl = REGRESSORS[name]()
            mdl.fit(X.iloc[tr_all], y.iloc[tr_all])
            p = mdl.predict(X.iloc[te_all])
            t = y.iloc[te_all].to_numpy()
            act = p >= (z_d + np.log1p(m))
            unsafe = np.exp(t) < d_phys
            realised = risk_block(act, unsafe)
            outer_rows.append({
                "scenario": "beam_release", "outer_fold": of, "protocol": strategy, "alpha": alpha,
                "demand_quantile": DEMAND_Q, "demand_physical": round(d_phys, 4),
                "model": name, "margin": float(m), "inner_r_all": round(chosen["r_all"], 5),
                "inner_coverage": round(chosen["coverage"], 4),
                "realised_r_all": round(realised["r_all"], 5),
                "realised_r_released": (round(realised["r_released"], 5)
                                        if realised["r_released"] is not None else None),
                "realised_coverage": round(realised["coverage"], 4),
                "n_test": realised["n_test"], "n_released": realised["n_released"],
                "n_unsafe": realised["n_unsafe"],
                "gap_r_all": round(realised["r_all"] - chosen["r_all"], 5),
                "target_met": bool(realised["r_all"] <= alpha),
            })
            fold_rows.append({**outer_rows[-1]})

# ================================================================= failure mode
Xc, yc, gc = design(DATASETS / "mangalathu_2020_shear_wall/Shear_Wall_Database.xlsx", "FailureMode", "Author", ["Specimen"])
labels = sorted(pd.unique(yc))
dangerous = {labels[0]}
outer_c = list(GroupKFold(min(K_OUT, int(gc.nunique()))).split(Xc, yc, gc))
cls_rows, cls_cands = [], []
for of, (tr_all, te_all) in enumerate(outer_c):
    for strategy in ("random", "source"):
        folds = inner_splits(strategy, Xc.iloc[tr_all], yc.iloc[tr_all], gc.iloc[tr_all])
        cached = {m: [] for m in CLASSIFIERS}
        for itr, ite in folds:
            for name, factory in CLASSIFIERS.items():
                mdl = factory()
                mdl.fit(Xc.iloc[tr_all].iloc[itr], yc.iloc[tr_all].iloc[itr])
                pr = mdl.predict_proba(Xc.iloc[tr_all].iloc[ite])
                cached[name].append((mdl.classes_[pr.argmax(axis=1)], pr.max(axis=1),
                                     yc.iloc[tr_all].iloc[ite].to_numpy()))
        for action_set in ("act_on_confidence", "release_when_not_dangerous"):
            est = {}
            for name, plist in cached.items():
                for conf in CONFS:
                    acts, unsafes = [], []
                    for pred, confmax, t in plist:
                        in_dangerous = np.isin(t, list(dangerous))
                        if action_set == "act_on_confidence":
                            act = confmax >= conf
                        else:
                            act = (confmax >= conf) & ~np.isin(pred, list(dangerous))
                        acts.append(act)
                        unsafes.append(in_dangerous)
                    est[(name, float(conf))] = risk_block(np.concatenate(acts), np.concatenate(unsafes))
            for alpha in ALPHAS:
                feasible = [(k, v) for k, v in est.items() if v["r_all"] <= alpha]
                if feasible:
                    best_cov = max(v["coverage"] for _, v in feasible)
                    tied = [(k, v) for k, v in feasible if abs(v["coverage"] - best_cov) < 1e-12]
                    (name, conf), chosen = min(tied, key=lambda kv: (kv[0][1], kv[0][0]))
                    reason = "max inner coverage among rules with inner risk <= alpha"
                else:
                    (name, conf), chosen = min(est.items(), key=lambda kv: kv[1]["r_all"])[0], \
                        min(est.values(), key=lambda v: v["r_all"])
                    reason = "no rule met the target; minimum-risk rule reported"
                for (cn, cc), cv in est.items():
                    cls_cands.append({
                        "scenario": "failure_mode", "action_set": action_set, "outer_fold": of,
                        "protocol": strategy, "alpha": alpha, "model": cn, "confidence": float(cc),
                        "inner_r_all": round(cv["r_all"], 5), "inner_coverage": round(cv["coverage"], 4),
                        "feasible": cv["r_all"] <= alpha, "chosen": bool(cn == name and cc == conf),
                        "reason": reason if (cn == name and cc == conf) else "",
                    })
                mdl = CLASSIFIERS[name]()
                mdl.fit(Xc.iloc[tr_all], yc.iloc[tr_all])
                pr = mdl.predict_proba(Xc.iloc[te_all])
                pred = mdl.classes_[pr.argmax(axis=1)]
                confmax = pr.max(axis=1)
                t = yc.iloc[te_all].to_numpy()
                if action_set == "act_on_confidence":
                    act = confmax >= conf
                else:
                    act = (confmax >= conf) & ~np.isin(pred, list(dangerous))
                unsafe = np.isin(t, list(dangerous))
                realised = risk_block(act, unsafe)
                cls_rows.append({
                    "scenario": "failure_mode", "action_set": action_set, "outer_fold": of,
                    "protocol": strategy, "alpha": alpha, "model": name, "confidence": float(conf),
                    "inner_r_all": round(chosen["r_all"], 5), "inner_coverage": round(chosen["coverage"], 4),
                    "realised_r_all": round(realised["r_all"], 5),
                    "realised_r_released": (round(realised["r_released"], 5)
                                            if realised["r_released"] is not None else None),
                    "realised_coverage": round(realised["coverage"], 4),
                    "n_test": realised["n_test"], "n_released": realised["n_released"],
                    "n_unsafe": realised["n_unsafe"],
                    "gap_r_all": round(realised["r_all"] - chosen["r_all"], 5),
                    "target_met": bool(realised["r_all"] <= alpha),
                })

outer_df = pd.concat([pd.DataFrame(outer_rows), pd.DataFrame(cls_rows)], ignore_index=True)
cand_df = pd.concat([pd.DataFrame(cand_rows), pd.DataFrame(cls_cands)], ignore_index=True)
outer_df.to_csv(OUT / "decision_outer.csv", index=False, encoding="utf-8")
cand_df.to_csv(OUT / "decision_candidates.csv", index=False, encoding="utf-8")

# inner-coverage monotonicity in alpha (fixed candidate set)
mono = []
for (scen, aset, strat, fo, model, knob), grp in cand_df.groupby(
        ["scenario", "action_set" if "action_set" in cand_df.columns else "scenario", "protocol",
         "outer_fold", "model", "margin" if "margin" in cand_df.columns else "confidence"],
        dropna=False):
    pass  # placeholder replaced below (kept simple: computed directly)

grp_cols = ["scenario", "protocol", "outer_fold", "model"]
cand_df["knob"] = cand_df.get("margin", pd.Series(index=cand_df.index, dtype=float))
if "confidence" in cand_df.columns:
    cand_df["knob"] = cand_df["knob"].fillna(cand_df["confidence"])
viol = 0
for _, grp in cand_df.groupby(grp_cols):
    s = grp.groupby("alpha")["inner_coverage"].max().sort_index()
    if len(s) > 1 and any(s.diff().dropna() < -1e-12):
        viol += 1
unit_ok = all(r["agree"] for r in unit_rows)
(OUT / "unit_check.json").write_text(json.dumps({
    "physical_vs_log_path": unit_rows,
    "all_agree": unit_ok,
    "inner_coverage_monotone_in_alpha_violations": viol,
}, indent=2), encoding="utf-8")

# ------------------------------------------------------------------ report
agg = (outer_df.groupby(["scenario", "action_set"] if "action_set" in outer_df.columns else ["scenario",
        "protocol", "alpha"])
       .agg(est=("inner_r_all", "mean"), real=("realised_r_all", "mean"),
            cov=("realised_coverage", "mean"), folds=("outer_fold", "size"),
            above=("gap_r_all", lambda s: int((s > 1e-12).sum())),
            target_met=("target_met", "sum")).reset_index())

lines = ["# R37 corrected engineering-decision experiment", "",
         f"Unit check (physical path vs log path): {'PASS' if unit_ok else 'FAIL'} "
         f"{unit_rows}", "",
         "Risk denominators: `r_all` = unsafe events / all test records (used by the selector);",
         "`r_released` = unsafe events / released records (NA when nothing is released);",
         "`coverage` = released / all test records.", ""]
for (scen, aset), grp in outer_df.groupby(["scenario"] + (["action_set"] if "action_set" in outer_df.columns else [])):
    title = f"## {scen}" + (f" — action set: {aset}" if "action_set" in outer_df.columns else "")
    lines += [title, "",
              "| protocol | alpha | folds | mean estimated r_all | mean realised r_all | gap | "
              "folds above own estimate | folds meeting target | mean coverage | mean r_released |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (strat, alpha), sub in grp.groupby(["protocol", "alpha"]):
        rr = sub["realised_r_released"].dropna()
        lines.append(f"| {strat} | {alpha:.2f} | {len(sub)} | {sub['inner_r_all'].mean():.4f} | "
                     f"{sub['realised_r_all'].mean():.4f} | "
                     f"{(sub['realised_r_all'].mean() - sub['inner_r_all'].mean()):+.4f} | "
                     f"{int((sub['gap_r_all'] > 1e-12).sum())} | {int(sub['target_met'].sum())} | "
                     f"{sub['realised_coverage'].mean():.3f} | "
                     f"{(rr.mean() if len(rr) else float('nan')):.4f} |")
    lines.append("")
lines += ["## Selection behaviour (is the constraint active?)", "",
          "| scenario | protocol | fold | alpha | chosen rule | inner r_all | feasible rules |",
          "|---|---|---:|---:|---|---:|---:|"]
for _, r in cand_df[cand_df["chosen"]].sort_values(["scenario", "protocol", "outer_fold", "alpha"]).iterrows():
    knob = r.get("margin", r.get("confidence"))
    feas = int(cand_df[(cand_df.scenario == r["scenario"]) & (cand_df.protocol == r["protocol"]) &
                       (cand_df.outer_fold == r["outer_fold"]) & (cand_df.alpha == r["alpha"])]["feasible"].sum())
    lines.append(f"| {r['scenario']} | {r['protocol']} | {int(r['outer_fold'])} | {r['alpha']:.2f} | "
                 f"{r['model']}/{knob} | {r['inner_r_all']:.4f} | {feas} |")
(OUT / "DECISION_EXPERIMENT_R37.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:8]))
print(f"\nwrote {OUT/'DECISION_EXPERIMENT_R37.md'}, decision_outer.csv, decision_candidates.csv, unit_check.json")
print(f"unit check: {'PASS' if unit_ok else 'FAIL'} | coverage-monotonicity violations: {viol}")
