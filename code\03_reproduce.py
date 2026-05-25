"""Generic single-paper reproduction entry point.

Example:
    python 03_reproduce.py --doi 10.1016/j.engstruct.2018.02.020 \
        --data <path-to-public-csv> \
        --target <y-col> \
        --features <x-cols-comma-sep> \
        --group <group-col> \
        --model <Ridge|RF|GBR|MLP>

The script compares random K-fold, GroupKFold, and LeaveOneGroupOut validation
for a tabular structural-ML dataset with a user-supplied grouping column.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPRegressor

DEFAULT_OUT_ROOT = Path(__file__).resolve().parent / "outputs" / "reproductions"


MODELS = {
    "Ridge": lambda: Ridge(alpha=1.0, random_state=0),
    "RF": lambda: RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
    "GBR": lambda: GradientBoostingRegressor(n_estimators=300, random_state=0),
    "MLP": lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=0
    ),
}


def slugify(doi: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", doi.lower()).strip("-")


def cv_metrics(X, y, groups, model_factory, splitter):
    r2s, maes = [], []
    for tr, te in splitter.split(X, y, groups=groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        yp = m.predict(X[te])
        r2s.append(r2_score(y[te], yp))
        maes.append(mean_absolute_error(y[te], yp))
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doi", required=True)
    parser.add_argument("--data", required=True, help="input CSV path")
    parser.add_argument("--target", required=True)
    parser.add_argument("--features", required=True, help="comma-separated feature columns")
    parser.add_argument("--group", required=True)
    parser.add_argument("--model", default="GBR", choices=list(MODELS.keys()))
    parser.add_argument("--reported_r2", type=float, default=None)
    parser.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    feats = [c.strip() for c in args.features.split(",")]
    missing = [c for c in feats + [args.target, args.group] if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns: {missing}; available: {list(df.columns)[:30]}")

    df = df.dropna(subset=feats + [args.target, args.group]).copy()
    X = df[feats].to_numpy(dtype=float)
    y = df[args.target].to_numpy(dtype=float)
    g = df[args.group].astype(str).to_numpy()

    factory = MODELS[args.model]

    out: dict = {
        "doi": args.doi,
        "model_family": args.model,
        "n_samples": int(len(df)),
        "n_groups": int(pd.Series(g).nunique()),
        "features": feats,
        "target": args.target,
        "group_col": args.group,
        "reported_r2": args.reported_r2,
        "reproduced": {},
        "delta_r2_random_vs_group": None,
        "delta_r2_random_vs_loso": None,
        "code_runs": True,
        "issues_log": [],
        "timestamp": datetime.utcnow().isoformat(),
    }

    splitters = {
        "RandomKFold-5": KFold(n_splits=5, shuffle=True, random_state=0),
        "GroupKFold-5": GroupKFold(n_splits=5),
        "LeaveOneGroupOut": LeaveOneGroupOut(),
    }
    for name, sp in splitters.items():
        if name == "LeaveOneGroupOut" and out["n_groups"] > 60:
            out["issues_log"].append(f"{name} skipped: n_groups={out['n_groups']} > 60")
            continue
        r2_mean, r2_std, mae = cv_metrics(X, y, g, factory, sp)
        out["reproduced"][name] = {"r2_mean": r2_mean, "r2_std": r2_std, "mae": mae}
        print(f"  {name:18s} | R^2 = {r2_mean:6.3f} ± {r2_std:5.3f} | MAE = {mae:6.3f}")

    if "RandomKFold-5" in out["reproduced"] and "GroupKFold-5" in out["reproduced"]:
        out["delta_r2_random_vs_group"] = (
            out["reproduced"]["RandomKFold-5"]["r2_mean"]
            - out["reproduced"]["GroupKFold-5"]["r2_mean"]
        )
    if "RandomKFold-5" in out["reproduced"] and "LeaveOneGroupOut" in out["reproduced"]:
        out["delta_r2_random_vs_loso"] = (
            out["reproduced"]["RandomKFold-5"]["r2_mean"]
            - out["reproduced"]["LeaveOneGroupOut"]["r2_mean"]
        )

    out_dir = Path(args.out_root) / slugify(args.doi)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "reproduction.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"[OK] wrote {out_path}")
    print(f"     delta_r2 (Random - Group) = {out['delta_r2_random_vs_group']}")


if __name__ == "__main__":
    main()
