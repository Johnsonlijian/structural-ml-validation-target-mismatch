"""R38-107: build the admissibility ledger that the panel asked for (34 executed -> 6 excluded -> ...).

The citation seat flagged a P1 arithmetic risk: the text says six of thirty-four executed split variants
fail the fold-size check, and elsewhere reports 32 admissible distinct contrasts, so a hostile reader sees
34 - 6 = 28 against 32. The chain is only defensible if the intermediate counts are shown explicitly. This
script computes them from the released artefacts and writes the ledger.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
R36 = Path(__file__).resolve().parents[2] / "R36_independent_review_rebuild_2026-09-16"
OUT = ROOT / "analysis" / "out"

reg = pd.read_csv(R36 / "audit/r36/configuration_registry.csv")
pairs = pd.read_csv(R36 / "analysis/out/R36_control_pairs.csv")
adm = pd.read_csv(R36 / "audit/r36/fold_structure.csv") if (R36 / "audit/r36/fold_structure.csv").exists() else None

print("=== configuration registry ===")
print(f"rows: {len(reg)} | columns: {list(reg.columns)}")
for col in reg.columns:
    if reg[col].nunique() <= 12:
        print(f"  {col}: {sorted(map(str, reg[col].dropna().unique()))[:12]}")

print("\n=== control pairs (the executed contrasts) ===")
print(f"rows: {len(pairs)} | columns: {list(pairs.columns)}")
print(f"  distinct (dataset, variant, model): "
      f"{pairs.groupby(['dataset', 'variant', 'model']).ngroups}")
print(f"  admissible rows: {int(pairs.split_admissible.fillna(False).sum())}")
print(f"  inadmissible rows: {int((~pairs.split_admissible.fillna(False)).sum())}")
print("\n  per variant: rows / admissible")
tab = pairs.groupby("variant").agg(rows=("split_admissible", "size"),
                                   admissible=("split_admissible", "sum"))
print(tab.to_string())

print("\n=== the alias case ===")
alias = pairs[(pairs.variant == "uci_996_allinputs") & (pairs.dataset == "concrete_strength")]
print(f"  rows matching the aliased evaluation: {len(alias)}")
adm_only = pairs[pairs.split_admissible.fillna(False)]
dedup = adm_only[~((adm_only.variant == "uci_996_allinputs") &
                   (adm_only.dataset == "concrete_strength"))]
print(f"  admissible rows: {len(adm_only)} | after removing the alias duplicate: {len(dedup)}")
print("\n  distinct entries by (dataset, variant, model):")
print(f"    admissible: {adm_only.groupby(['dataset','variant','model']).ngroups}")
print(f"    de-aliased: {dedup.groupby(['dataset','variant','model']).ngroups}")

ledger = {
    "executed_variants": int(pairs.variant.nunique()),
    "executed_rows": int(len(pairs)),
    "inadmissible_rows": int((~pairs.split_admissible.fillna(False)).sum()),
    "admissible_rows": int(len(adm_only)),
    "alias_rows": int(len(alias)),
    "distinct_admissible_entries": int(adm_only.groupby(["dataset", "variant", "model"]).ngroups),
    "distinct_dealiased_entries": int(dedup.groupby(["dataset", "variant", "model"]).ngroups),
    "variants_inadmissible": sorted(tab[tab.admissible == 0].index.tolist()),
    "variants_partially_inadmissible": sorted(
        tab[(tab.admissible > 0) & (tab.admissible < tab.rows)].index.tolist()),
}
(OUT / "R38_admissibility_ledger.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")
print("\n=== ledger ===")
for k, v in ledger.items():
    print(f"  {k}: {v}")
