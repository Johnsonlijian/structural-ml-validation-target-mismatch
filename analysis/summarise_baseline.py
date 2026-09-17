"""R38-75: summarise the baseline-versus-audit comparison for the manuscript and the SI."""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "analysis" / "out"
cmp = pd.read_csv(D / "R38_baseline_vs_audit.csv")
adm = pd.read_csv(D / "R38_baseline_admissibility.csv")
adm["fold_sizes"] = adm["fold_sizes"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)

print("=== per scenario / action set / protocol ===")
g = (cmp.groupby(["scenario", "action_set", "protocol"])
     .apply(lambda d: pd.Series({"settings": len(d),
                                 "differing": int((~d.same_choice).sum()),
                                 "empty_feasible": int((d.G_mode == "fallback_min_risk").sum()),
                                 "inadmissible": int((~d.admissible).sum())}),
            include_groups=False))
print(g.to_string())

print(f"\ninner splits {len(adm)} | admissible {int(adm.admissible.sum())} | "
      f"smallest fold slack {int(min(min(s) - L for s, L in zip(adm.fold_sizes, adm.L)))} rows above L | "
      f"largest fold {int(max(max(s) for s in adm.fold_sizes))} against U "
      f"{int(min(adm.U))}-{int(max(adm.U))}")
print(f"pooled realised: baseline released {int(cmp.G_released.sum())} unsafe {int(cmp.G_unsafe.sum())} | "
      f"audit released {int(cmp.A_released.sum())} unsafe {int(cmp.A_unsafe.sum())}")
print(f"identical decisions in {int(cmp.same_choice.sum())} of {len(cmp)} settings")

summary = {
    "settings": int(len(cmp)),
    "identical_decisions": int(cmp.same_choice.sum()),
    "empty_feasible_sets": int((cmp.G_mode == "fallback_min_risk").sum()),
    "inadmissible_inner_splits": int((~cmp.admissible).sum()),
    "inner_splits_checked": int(len(adm)),
    "baseline_released": int(cmp.G_released.sum()), "baseline_unsafe": int(cmp.G_unsafe.sum()),
    "audit_released": int(cmp.A_released.sum()), "audit_unsafe": int(cmp.A_unsafe.sum()),
    "by_group": g.reset_index().to_dict(orient="records"),
}
(D / "R38_baseline_vs_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# ---- SI table rows
rows = []
label = {"beam_release": "beam release", "failure_mode": "coded-category screening"}
for (scen, aset, prot), d in cmp.groupby(["scenario", "action_set", "protocol"]):
    rows.append((f"{label[scen]} & {aset.replace('_', ' ')} & {prot} & {len(d)} & "
                 f"{int((~d.same_choice).sum())} & {int((d.G_mode == 'fallback_min_risk').sum())} & "
                 f"{int((~d.admissible).sum())} \\\\"))
(D / "R38_baseline_table_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("\nSI table rows written")
for r in rows:
    print("  " + r)
