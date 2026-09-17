"""R37-02: verify the second review's findings against this project's own artefacts.

Checks, in the order the review raises them:
  P0-1  source-honouring realised vs estimated: base settings and fold-level counts
  P0-2  whether the alpha constraint was ever active in the base run
  P0-3  the log-target margin: does the executed rule match the stated physical rule?
  P1-2  UCI seven-ingredient vs eight-input: are the two evaluations aliases?
  P1-3  singleton arithmetic for the eight-input key
  P1-6  which variant actually carries the largest admissible degradation
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
R36 = ROOT.parent / "R36_independent_review_rebuild_2026-09-16"
OUT = ROOT / "audit" / "r37"
OUT.mkdir(parents=True, exist_ok=True)

findings: list[str] = []


def say(line: str = "") -> None:
    print(line)
    findings.append(line)


# ---------------------------------------------------------------- P0-1 / P0-2
dec = pd.read_csv(R36 / "audit/r36/D_outer_results.csv")
base = dec.dropna(subset=["realised_unsafe"])
say("## P0-1/P0-2  base decision experiment (the run behind SI Table 4)")
agg = (base.groupby(["scenario", "alpha_target", "strategy"])
       .agg(est=("est_unsafe", "mean"), real=("realised_unsafe", "mean"),
            cov=("realised_coverage", "mean"), folds=("outer_fold", "size")).reset_index())
src = agg[agg.strategy == "source"].copy()
src["gap"] = src["real"] - src["est"]
say(f"- source settings: {len(src)}; realised above estimate: {int((src['gap'] > 0).sum())}")
for _, r in src.sort_values(["scenario", "alpha_target"]).iterrows():
    flag = "ABOVE" if r["gap"] > 0 else "below"
    say(f"    {r['scenario']:<14} alpha={r['alpha_target']:.2f}  est={r['est']:.4f} "
        f"real={r['real']:.4f}  gap={r['gap']:+.4f}  {flag}")
rnd = agg[agg.strategy == "random"].copy()
rnd["gap"] = rnd["real"] - rnd["est"]
say(f"- random settings: {len(rnd)}; realised above estimate: {int((rnd['gap'] > 0).sum())}")

fold = base[base.strategy == "source"].copy()
fold["gap"] = fold["realised_unsafe"] - fold["est_unsafe"]
say(f"- source fold x target rows: {len(fold)}; above estimate: {int((fold['gap'] > 0).sum())}; "
    f"largest gap {fold['gap'].max():+.4f}")
worst = fold.loc[fold["gap"].idxmax()]
say(f"    worst row: {worst['scenario']} alpha={worst['alpha_target']:.2f} fold={int(worst['outer_fold'])} "
    f"est={worst['est_unsafe']:.4f} real={worst['realised_unsafe']:.4f}")

sweep = pd.read_csv(R36 / "audit/r36/D_sweep_demand.csv").dropna(subset=["realised"])
sw_src = sweep[sweep.strategy == "source"]
say(f"- demand sweep (beam only): {len(sw_src)} source fold-settings; "
    f"above own estimate: {int((sw_src['realised'] > sw_src['est']).sum())}")

# was the alpha constraint ever active? compare chosen knobs across alpha
chosen = base[["scenario", "strategy", "alpha_target", "outer_fold"]].copy()
for col in ("margin", "confidence", "model"):
    if col in base.columns:
        chosen[col] = base[col].to_numpy()
say("- chosen rule identical across alpha for the same scenario/fold/protocol:")
for (scen, strat, fo), grp in chosen.groupby(["scenario", "strategy", "outer_fold"]):
    knobs = {(r.get("margin"), r.get("confidence"), r.get("model")) for _, r in grp.iterrows()}
    if len(knobs) == 1:
        say(f"    {scen:<14} {strat:<7} fold={int(fo)}: identical across alpha ({list(knobs)[0]})")
    else:
        say(f"    {scen:<14} {strat:<7} fold={int(fo)}: {len(knobs)} distinct rules")

# ---------------------------------------------------------------- P0-3 log target
say()
say("## P0-3  log-target margin")
src_code = (R36 / "analysis/07_d_decision_experiment.py").read_text(encoding="utf-8")
for needle in ("d = float(np.quantile", "d * (1 + m)"):
    hits = [ln.strip() for ln in src_code.splitlines() if needle in ln]
    say(f"- executed code contains {needle!r}: {hits if hits else 'not found'}")
say("  physical rule:  R_hat >= d_phys*(1+m)   |  log rule: z_hat >= log(d_phys) + log(1+m)")
say("  executed rule:  p >= d_log*(1+m)        <-- multiplies the log-demand, not log(1+m)")
y = pd.read_csv(PROJ / "code/outputs/reproductions/10-5281-zenodo-8062007/analysis_data.csv")["log_mmax_exp"]
d_log = float(np.quantile(y, 0.5))
for m in (0.0, 0.2, 0.5):
    executed = d_log * (1 + m)
    intended = d_log + np.log1p(m)
    say(f"    m={m:.1f}: executed threshold {executed:.4f} vs intended {intended:.4f} "
        f"(difference {executed - intended:+.4f} in log units)")

# ---------------------------------------------------------------- P1-2 UCI aliases
say()
say("## P1-2  UCI relation variants: alias check")
mt = pd.read_csv(R36 / "audit/r36/control_metrics.csv")
uci = mt[mt.dataset == "concrete_strength"]
say(f"- variants present: {sorted(uci['variant'].unique())}")
for v in sorted(uci["variant"].unique()):
    row = uci[uci.variant == v].sort_values("model")
    say(f"    {v:<26} K={int(row['n_folds'].iloc[0])} "
        f"pooledR2={[round(float(x), 4) for x in row['pooled_r2']]}")
folds = pd.read_csv(R36 / "audit/r36/fold_structure.csv")
print_cols = [c for c in ("dataset", "variant", "fold", "n_test", "test_index_hash") if c in folds.columns]
sub = folds[folds.dataset == "concrete_strength"]
say(f"- fold-structure columns available: {list(folds.columns)}")
if "test_index_hash" in sub.columns:
    for v in sorted(sub["variant"].unique()):
        h = hashlib.sha256("".join(sorted(map(str, sub[sub.variant == v]["test_index_hash"]))).encode()).hexdigest()[:16]
        say(f"    {v:<26} assignment hash {h}")

# ---------------------------------------------------------------- P1-3 singletons
say()
say("## P1-3  singleton arithmetic (eight-input key, n=1030, groups=996)")
n, g = 1030, 996
say(f"- singleton groups >= {g - (n - g)} = {g - (n - g)}; group share >= {(g - (n - g)) / g:.4%}")
say("- 94.9% is therefore a *record* share, not a group share; three numbers are required")

# ---------------------------------------------------------------- P1-6 extreme value
say()
say("## P1-6  largest admissible degradation: which set?")
pairs = pd.read_csv(R36 / "analysis/out/R36_control_pairs.csv")
adm = pairs[pairs.split_admissible == True]                                    # noqa: E712
per_variant = (adm.assign(degr=adm.r2_honoured)
               .sort_values("degr").groupby("variant").head(1)
               [["dataset", "variant", "model", "r2_honoured"]])
say(per_variant.sort_values("r2_honoured").to_string(index=False))
say(f"- all admissible: minimum R2 = {adm['r2_honoured'].min():.3f}")
sized = adm[adm.delta_vs_sizematch.notna()]
say(f"- size-matched subset: minimum R2 = {sized['r2_honoured'].min():.3f}, "
    f"{int((sized['r2_honoured'] < 0).sum())} negative")
src_only = adm[adm.variant == "honoured"]
say(f"- source-level only: n={len(src_only)}, negative vs size-match "
    f"{int((src_only['delta_vs_sizematch'] < 0).sum())}, "
    f"median {src_only['delta_vs_sizematch'].median():.4f}")

(OUT / "R37_REVIEW_VERIFICATION.md").write_text(
    "# R37 verification of the second review\n\n" + "\n".join(findings) + "\n", encoding="utf-8")
print(f"\nwrote {OUT/'R37_REVIEW_VERIFICATION.md'}")
