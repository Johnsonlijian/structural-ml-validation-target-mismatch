"""R36-03: summarise the control experiment and test whether the entitlement gap survives.

Produces, per dataset x honoured variant x learner:
    R2(random5), R2(sizematch), R2(honoured), and the reference dummy baseline
so the reader can separate
    (a) the practice contrast  random5 -> honoured
    (b) the size-controlled contrast sizematch -> honoured
and flags any executed split whose fold sizes violate the declared bounds [L,U].
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
AUD = ROOT / "audit" / "r36"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)
FMIN, FMAX = 0.05, 0.50

m = pd.read_csv(AUD / "control_metrics.csv")
f = pd.read_csv(AUD / "fold_structure.csv")

# ---- fold-bound admissibility of every executed split --------------------------
fold_rows = []
for (ds, variant), g in f.groupby(["dataset", "variant"]):
    n_total = int(g["n_train"].iloc[0] + g["n_test"].iloc[0])
    L, U = math.ceil(FMIN * n_total), math.floor(FMAX * n_total)
    bad_low = int((g["n_test"] < L).sum())
    bad_high = int((g["n_test"] > U).sum())
    fold_rows.append({
        "dataset": ds, "variant": variant, "K": int(len(g)),
        "n": n_total, "L": L, "U": U,
        "test_sizes": sorted(g["n_test"].tolist(), reverse=True),
        "train_fraction_range": f"{g['train_fraction'].min():.3f}-{g['train_fraction'].max():.3f}",
        "folds_below_L": bad_low, "folds_above_U": bad_high,
        "admissible": bad_low == 0 and bad_high == 0,
    })
folds = pd.DataFrame(fold_rows).sort_values(["admissible", "dataset", "variant"])
folds.to_csv(OUT / "R36_fold_admissibility.csv", index=False, encoding="utf-8")

piv = {}
for variant in ("random5", "sizematch_relation", "honoured", "sizematch_proxy", "honoured_proxy",
                "wall_combo", "uci_428_ingredients", "uci_996_allinputs",
                "uci_428_ingredients_shuffledlabels", "uci_996_allinputs_shuffledlabels"):
    sub = m[m["variant"] == variant]
    for _, r in sub.iterrows():
        piv[(r["dataset"], variant, r["model"])] = r["pooled_r2"]

rows = []
honoured_variants = ["honoured", "honoured_proxy", "wall_combo", "uci_428_ingredients", "uci_996_allinputs"]
for (ds, variant), g in m[m["variant"].isin(honoured_variants)].groupby(["dataset", "variant"]):
    sm_key = {"honoured": "sizematch_relation", "honoured_proxy": "sizematch_proxy"}.get(variant)
    for _, r in g.iterrows():
        rows.append({
            "dataset": ds, "variant": variant, "model": r["model"],
            "r2_random5": piv.get((ds, "random5", r["model"])),
            "r2_sizematch": piv.get((ds, sm_key, r["model"])) if sm_key else None,
            "r2_honoured": r["pooled_r2"],
            "fold_mean_r2": r["fold_mean_r2"], "fold_sd_r2": r["fold_sd_r2"], "K": r["n_folds"],
        })
ctl = pd.DataFrame(rows)
ctl["delta_vs_random"] = ctl["r2_honoured"] - ctl["r2_random5"]
ctl["delta_vs_sizematch"] = ctl["r2_honoured"] - ctl["r2_sizematch"]

adm = folds.set_index(["dataset", "variant"])["admissible"].to_dict()
ctl["split_admissible"] = [adm.get((d, v), None) for d, v in zip(ctl["dataset"], ctl["variant"])]
ctl.to_csv(OUT / "R36_control_pairs.csv", index=False, encoding="utf-8")

# ---- dummy reference -----------------------------------------------------------
dummy = m[m["variant"] == "dummy_trainmean"][["dataset", "model", "pooled_r2"]].rename(
    columns={"pooled_r2": "r2_dummy_trainmean"})

lines = ["# R36 control experiment: does the entitlement gap survive?", ""]
lines += ["## Fold-bound admissibility of every executed split", "",
          "| dataset | variant | K | n | L | U | test sizes | train fraction | below L | above U | admissible |",
          "|---|---|---:|---:|---:|---:|---|---|---:|---:|---|"]
for _, r in folds.iterrows():
    lines.append(f"| {r['dataset']} | {r['variant']} | {r['K']} | {r['n']} | {r['L']} | {r['U']} | "
                 f"{r['test_sizes']} | {r['train_fraction_range']} | {r['folds_below_L']} | "
                 f"{r['folds_above_U']} | {'yes' if r['admissible'] else '**NO**'} |")
lines.append("")
inadm = folds[~folds["admissible"]]
lines.append(f"inadmissible splits: **{len(inadm)}/{len(folds)}**"
             + (f" -> {list(zip(inadm['dataset'], inadm['variant']))}" if len(inadm) else ""))
lines.append("")

lines += ["## Paired contrasts (R2, pooled)", "",
          "| dataset | variant | model | R2 random5 | R2 size-match | R2 honoured | d vs random | d vs size-match | folds | admissible |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
for _, r in ctl.sort_values(["dataset", "variant", "model"]).iterrows():
    sm = "" if pd.isna(r["r2_sizematch"]) else f"{r['r2_sizematch']:.3f}"
    ds_ = "" if pd.isna(r["delta_vs_sizematch"]) else f"{r['delta_vs_sizematch']:+.3f}"
    lines.append(f"| {r['dataset']} | {r['variant']} | {r['model']} | {r['r2_random5']:.3f} | {sm} | "
                 f"{r['r2_honoured']:.3f} | {r['delta_vs_random']:+.3f} | {ds_} | {r['K']} | "
                 f"{'yes' if r['split_admissible'] else '**NO**'} |")
lines.append("")

ok = ctl[ctl["split_admissible"] == True]                                   # noqa: E712
lines += ["## Summary over admissible splits only", ""]
lines.append(f"- admissible paired contrasts: **{len(ok)}**")
if len(ok):
    lines.append(f"- degrade vs random5: **{int((ok['delta_vs_random'] < 0).sum())}/{len(ok)}**; "
                 f"median {ok['delta_vs_random'].median():+.3f}")
    sized = ok[ok["delta_vs_sizematch"].notna()]
    lines.append(f"- with a size-matched control available: **{len(sized)}**; "
                 f"degrade vs size-match: **{int((sized['delta_vs_sizematch'] < 0).sum())}/{len(sized)}**; "
                 f"median {sized['delta_vs_sizematch'].median():+.3f}")
    lines.append(f"- below zero R2 under the honoured split: **{int((ok['r2_honoured'] < 0).sum())}/{len(ok)}**")
lines.append("")

lines += ["## Dummy (training-mean) reference", "",
          "| dataset | pooled R2 of the per-fold training-mean predictor |", "|---|---:|"]
for _, r in dummy.iterrows():
    lines.append(f"| {r['dataset']} | {r['r2_dummy_trainmean']:+.3f} |")
lines.append("")
lines.append("A negative R2 under the honoured split must be compared against this reference, not "
             "against zero: the training-mean predictor is what a practitioner can actually fall "
             "back on.")

(OUT / "R36_CONTROL_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:12]))
print("...")
print("\n".join(lines[-24:]))
