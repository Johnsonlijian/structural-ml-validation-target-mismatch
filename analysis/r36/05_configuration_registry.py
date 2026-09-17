"""R36-05 (work package A): the per-configuration registry.

Single source of truth for every count, relation label and scope statement that appears in the
manuscript. Each row is one analysis configuration (dataset x target x relation), carrying

  provenance   : asset ids, repository, licence status, cross-asset relations
  counts       : raw rows, analytic rows, post-filter groups, missing-label rows
  semantics    : relation kind in engineering terms (NOT the code column name)
  features     : what is available at prediction time, and what was dropped and why
  split        : the executed split, its fold sizes, admissibility under declared bounds
  claim status : whether the audited use is the original study's own deployment claim or a
                 scenario defined by this study (default: scenario, because no original claim
                 text has been extracted verbatim)

Nothing here is inferred from a code column name.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                                  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[1]
PROJ = Path(__file__).resolve().parents[3]
AUD36 = ROOT / "audit" / "r36"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)
REPRO = PROJ / "code" / "outputs" / "reproductions"
DATASETS = PROJ / "code" / "outputs" / "datasets"
FMIN, FMAX = 0.05, 0.50

# asset -> repository metadata (identifiers were resolved through Crossref/DataCite/arXiv in R35)
ASSETS = {
    "stub_cfst": dict(asset_id="corpus_cfst_study", study="10.1038/s41598-024-53352-1",
                      repository="Scientific Reports supplementary + Mendeley",
                      cross_asset="none"),
    "corroded_rc_beam": dict(asset_id="corpus_corroded_dataset", study="10.5281/zenodo.8062007",
                             repository="Zenodo", cross_asset="none"),
    "wall_prj2430_Vmax": dict(asset_id="corpus_ds_wall_db", study="10.17603/ds2-r12q-t415",
                              repository="DesignSafe-CI", cross_asset="same wall database as drift target"),
    "wall_prj2430_drift": dict(asset_id="corpus_ds_wall_db", study="10.17603/ds2-r12q-t415",
                               repository="DesignSafe-CI", cross_asset="same wall database as Vmax target"),
    "coupling_beams_prj3053": dict(asset_id="corpus_ds_coupling_db", study="10.17603/ds2-46wc-n185",
                                   repository="DesignSafe-CI",
                                   cross_asset="identifier corrected in R35 (recorded id did not resolve)"),
    "shear_wall_failure_mode": dict(asset_id="Mangalathu2020ShearWall",
                                    study="10.1016/j.engstruct.2020.110331",
                                    repository="journal article + author database",
                                    cross_asset="study paper and its database are one asset"),
    "concrete_strength": dict(asset_id="corpus_uci_concrete", study="10.24432/C5PK67",
                              repository="UCI Machine Learning Repository",
                              cross_asset="the Nguyen et al. study (10.1016/j.conbuildmat.2020.120950) "
                                          "uses this same UCI asset; it is one asset, not two"),
    "sfrc_beam_shear": dict(asset_id="corpus_lantsoght_sfrc_db", study="10.5281/zenodo.2578061",
                            repository="Zenodo",
                            cross_asset="the Rahman et al. study (10.1016/j.engstruct.2020.111743) "
                                        "uses this same database; one asset, not two"),
    "rc_columns": dict(asset_id="corpus_ds_rc_circ + corpus_ds_rc_rect",
                       study="10.17603/ds2-52bz-0n63; 10.17603/ds2-7qg0-4303",
                       repository="DesignSafe-CI",
                       cross_asset="two databases combined; group-size file reports per-case counts"),
    "joints_mendeley": dict(asset_id="corpus_mendeley_joint1 + corpus_mendeley_joint2",
                            study="10.17632/8ndgpm7zw7.1; 10.17632/rbhfnz32sy.1",
                            repository="Mendeley Data",
                            cross_asset="two workbooks (203 and 98 rows); corpus row reports the 203-row workbook"),
}

# dataset -> artefact used by the audit, with relation semantics in engineering terms
CONFIGS = [
    dict(dataset="stub_cfst", target="P", relation_key="shape",
         relation_semantics="section family (PROXY: a section type is not a test programme or laboratory)",
         relation_kind="proxy-family", drop=["strength_index", "nominal_capacity"],
         artefact="reproductions/10-1038-s41598-024-53352-1/combined_stub_cfst_data.csv",
         experiment="regression"),
    dict(dataset="corroded_rc_beam", target="log_mmax_exp", relation_key="source_group",
         relation_semantics="experimental programme parsed from author/specimen identifiers (SOURCE-LEVEL)",
         relation_kind="source-level", drop=[],
         artefact="reproductions/10-5281-zenodo-8062007/analysis_data.csv",
         experiment="regression"),
    dict(dataset="wall_prj2430_Vmax", target="Vmax", relation_key="Authors",
         relation_semantics="author/campaign (SOURCE-LEVEL)",
         relation_kind="source-level",
         drop=["Dmax", "driftCap", "SpecimenID", "UniqueID", "row_index"],
         artefact="reproductions/designsafe_prj2430_wall/extracted_wall_data.csv",
         experiment="regression", secondary="walltype_Shape",
         secondary_semantics="wall type (PROXY structural attribute)"),
    dict(dataset="wall_prj2430_drift", target="driftCap", relation_key="Authors",
         relation_semantics="author/campaign (SOURCE-LEVEL)",
         relation_kind="source-level",
         drop=["Dmax", "Vmax", "SpecimenID", "UniqueID", "row_index"],
         artefact="reproductions/designsafe_prj2430_wall/extracted_wall_data.csv",
         experiment="regression", secondary="walltype_Shape",
         secondary_semantics="wall type (PROXY structural attribute)"),
    dict(dataset="coupling_beams_prj3053", target="V_m_avg_kips", relation_key="source_group",
         relation_semantics="reference/programme (SOURCE-LEVEL)",
         relation_kind="source-level",
         drop=["specimen_id", "reference_number", "v_m_normalized"],
         artefact="reproductions/designsafe_prj3053_coupling_beams/extracted_coupling_beam_data.csv",
         experiment="regression"),
    dict(dataset="concrete_strength", target="__last__", relation_key=None,
         relation_semantics="identical mix proportions (DUPLICATION relation, not deployment novelty)",
         relation_kind="duplication",
         drop=[], artefact="datasets/uci_concrete_compressive_strength/Concrete_Data.xls",
         experiment="regression", secondary="__mix__"),
    dict(dataset="shear_wall_failure_mode", target="FailureMode", relation_key="Author",
         relation_semantics="author/campaign (SOURCE-LEVEL)",
         relation_kind="source-level", drop=["Specimen"],
         artefact="datasets/mangalathu_2020_shear_wall/Shear_Wall_Database.xlsx",
         experiment="classification"),
]

rows = []
for cfg in CONFIGS:
    path = (REPRO / cfg["artefact"].split("reproductions/", 1)[1]) if cfg["artefact"].startswith("reproductions/") \
        else (DATASETS / cfg["artefact"].split("datasets/", 1)[1])
    rec = {
        "dataset": cfg["dataset"], "experiment": cfg["experiment"], "target": cfg["target"],
        "relation_key": cfg["relation_key"], "relation_kind": cfg["relation_kind"],
        "relation_semantics": cfg["relation_semantics"],
        "secondary_relation": cfg.get("secondary", ""),
        "secondary_semantics": cfg.get("secondary_semantics", ""),
        "artefact": str(path.relative_to(PROJ)),
        "artefact_exists": path.exists(),
    }
    rec.update({k: ASSETS[cfg["dataset"]][k] for k in ("asset_id", "study", "repository", "cross_asset")})
    if path.exists():
        df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path, low_memory=False)
        if cfg["target"] == "__last__":
            cfg["target"] = df.columns[-1]
            rec["target"] = cfg["target"]
        rec["n_raw"] = int(len(df))
        y = pd.to_numeric(df[cfg["target"]], errors="coerce")
        rec["n_analytic"] = int(y.notna().sum())
        rec["n_dropped_missing_target"] = int(len(df) - y.notna().sum())
        rel = cfg["relation_key"]
        if rel and rel in df.columns:
            lab = df.loc[y.notna(), rel].astype(str)
            role_rows = int(lab.isin(["random_fold", "held_out_source", "held_out"]).sum())
            rec["groups_postfilter"] = int(lab[~lab.isin(["random_fold", "held_out_source", "held_out"])].nunique())
            rec["rows_with_role_value_in_group_column"] = role_rows
            rec["rows_with_missing_relation"] = int(lab.isin(["nan", "None", ""]).sum())
        else:
            rec["groups_postfilter"] = None
            rec["rows_with_role_value_in_group_column"] = None
            rec["rows_with_missing_relation"] = None
        rec["dropped_features"] = " | ".join(cfg["drop"])
    else:
        rec.update({"n_raw": None, "n_analytic": None, "groups_postfilter": None})
    # declared-use status: no original claim text has been extracted verbatim in this project
    rec["audited_use"] = "scenario defined by this study (source-/family-/combination-unseen evaluation)"
    rec["original_claim_extracted"] = "no - original deployment-claim wording not yet extracted from the source papers"
    rows.append(rec)

reg = pd.DataFrame(rows)
reg.to_csv(AUD36 / "configuration_registry.csv", index=False, encoding="utf-8")

# ---- executed-split structure, joined in ---------------------------------------
folds = pd.read_csv(OUT / "R36_fold_admissibility.csv")
lines = ["# R36 configuration registry (work package A)", "",
         "Every count, relation label and scope statement in the manuscript must come from this file.",
         "", "| dataset | experiment | target | relation | kind | n raw | n analytic | groups | role rows | missing rel | admissible splits |",
         "|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]
for _, r in reg.iterrows():
    ds = r["dataset"]
    f = folds[folds["dataset"] == ds]
    adm = f"{int(f['admissible'].sum())}/{len(f)}" if len(f) else "-"
    lines.append(f"| {r['dataset']} | {r['experiment']} | {r['target']} | {r['relation_key']} | "
                 f"{r['relation_kind']} | {r['n_raw']} | {r['n_analytic']} | {r['groups_postfilter']} | "
                 f"{r['rows_with_role_value_in_group_column']} | {r['rows_with_missing_relation']} | {adm} |")
lines += ["", "## Relation semantics (code column name is not the science)", ""]
for _, r in reg.iterrows():
    lines.append(f"- **{r['dataset']}**: {r['relation_semantics']}"
                 + (f"; secondary: {r['secondary_semantics']}" if r["secondary_semantics"] else ""))
lines += ["", "## Provenance and cross-asset relations", "",
          "| dataset | asset id | study/data id | repository | cross-asset note |",
          "|---|---|---|---|---|"]
for _, r in reg.iterrows():
    lines.append(f"| {r['dataset']} | `{r['asset_id']}` | `{r['study']}` | {r['repository']} | {r['cross_asset']} |")
lines += ["", "## Declared use vs audited use", "",
          "No source paper's deployment-claim wording has been extracted verbatim in this project, so every",
          "configuration is registered as a **scenario defined by this study**. The manuscript must not say",
          "that the original authors made the audited claim until that text is extracted and recorded here.",
          "",
          f"- configurations registered: {len(reg)}",
          f"- configurations with a publicly resolvable identifier: {int(reg['study'].notna().sum())}",
          f"- configurations whose artefact exists locally: {int(reg['artefact_exists'].sum())}"]

(AUD36 / "CONFIGURATION_REGISTRY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines[:18]))
print(f"\nwrote {AUD36/'configuration_registry.csv'} and CONFIGURATION_REGISTRY.md")
