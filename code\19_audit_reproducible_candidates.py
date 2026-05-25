"""
Audit the OpenAlex structural-ML candidate pool for reproducibility triage.

This does not prove that a paper is reproducible. It creates a transparent,
reviewable sampling frame so the manuscript can move beyond hand-picked case
studies and toward a field-wide meta-reproduction.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "code" / "outputs" / "literature" / "top_structural_reproduction_candidates.csv"
OUT_DIR = ROOT / "code" / "outputs" / "literature"
FRAG_DIR = ROOT / "manuscript_fragments"

COMPLETED_DOI = {
    "10.1016/j.engstruct.2019.110331": "case_1_mangalathu_shear_wall",
    "10.1016/j.engstruct.2020.111743": "case_2_rahman_sfrc",
    "10.1016/j.conbuildmat.2020.120950": "case_3_nguyen_concrete",
    "10.1038/s41598-024-53352-1": "case_s1_stub_cfst",
    "10.5281/zenodo.8062007": "case_4_corroded_rc_beam_dataset",
}

DATA_REPOSITORY_HINTS = [
    "github",
    "zenodo",
    "figshare",
    "mendeley",
    "dryad",
    "purr",
    "dataverse",
    "osf.io",
    "data.mendeley",
    "datadryad",
]

TOPIC_RULES = [
    ("shear wall", ["shear wall", "squat wall", "wall shear"]),
    ("beam-column joint", ["beam-column", "beam column", "joint"]),
    ("concrete strength", ["concrete strength", "compressive strength"]),
    ("CFST/composite column", ["cfst", "concrete-filled", "concrete filled", "composite column"]),
    ("fiber/SFRC shear", ["steel fiber", "sfrc", "fiber reinforced"]),
    ("pile/geotechnical", ["pile", "soil", "geotechnical", "drivability"]),
    ("seismic/fragility", ["seismic", "fragility", "earthquake"]),
    ("structural deterioration", ["corrosion", "corroded", "deterioration", "durability"]),
]


def clean_doi(value: object) -> str:
    text = str(value or "").strip().lower()
    return text.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")


def infer_topic(row: pd.Series) -> str:
    text = " ".join([str(row.get("title", "")), str(row.get("abstract", "")), str(row.get("concepts", ""))]).lower()
    for topic, needles in TOPIC_RULES:
        if any(needle in text for needle in needles):
            return topic
    return "other structural ML"


def has_data_hint(row: pd.Series) -> bool:
    haystack = " ".join([str(row.get("oa_url", "")), str(row.get("abstract", "")), str(row.get("title", ""))]).lower()
    return any(hint in haystack for hint in DATA_REPOSITORY_HINTS)


def infer_priority(row: pd.Series) -> tuple[str, str]:
    doi = row["doi_clean"]
    if doi in COMPLETED_DOI:
        return "completed", COMPLETED_DOI[doi]
    if bool(row["has_data_hint"]):
        return "high", "repository/data hint in OA URL or metadata"
    if bool(row.get("is_oa", False)) and str(row.get("oa_url", "")).strip():
        return "medium", "open full text but no repository hint"
    if int(row.get("cited_by", 0)) >= 100 and bool(row.get("has_metric", False)):
        return "medium", "highly cited with quantitative metrics; data status unknown"
    return "low", "no immediate open-data/code signal"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_PATH)
    df["doi_clean"] = df["doi"].apply(clean_doi)
    df["topic_bucket"] = df.apply(infer_topic, axis=1)
    df["has_data_hint"] = df.apply(has_data_hint, axis=1)
    priority = df.apply(infer_priority, axis=1, result_type="expand")
    df["reproduction_priority"] = priority[0]
    df["priority_reason"] = priority[1]

    priority_order = {"completed": 0, "high": 1, "medium": 2, "low": 3}
    df["priority_rank"] = df["reproduction_priority"].map(priority_order)
    df = df.sort_values(["priority_rank", "cited_by"], ascending=[True, False]).reset_index(drop=True)

    audit_cols = [
        "reproduction_priority",
        "priority_reason",
        "topic_bucket",
        "doi_clean",
        "title",
        "year",
        "venue",
        "cited_by",
        "is_oa",
        "oa_url",
        "first_author",
        "has_data_hint",
    ]
    audit = df[audit_cols].copy()
    audit.to_csv(OUT_DIR / "reproducible_candidate_audit.csv", index=False)

    summary = (
        audit.groupby(["reproduction_priority", "topic_bucket"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["reproduction_priority", "n"], ascending=[True, False])
    )
    summary.to_csv(OUT_DIR / "reproducible_candidate_audit_summary.csv", index=False)

    high = audit[audit["reproduction_priority"].isin(["completed", "high", "medium"])].head(40)
    lines = [
        "# Reproducible Candidate Audit v0",
        "",
        "Purpose: convert the 332-paper OpenAlex candidate pool into a transparent sampling frame for the field-wide meta-reproduction.",
        "",
        "## Summary by priority",
        "",
    ]
    count_table = audit["reproduction_priority"].value_counts().reindex(["completed", "high", "medium", "low"]).fillna(0).astype(int)
    for priority_name, count in count_table.items():
        lines.append(f"- `{priority_name}`: {count}")
    lines.extend(
        [
            "",
            "## Top 40 audit queue",
            "",
            "| Priority | Topic | Year | Cites | DOI | Title | Reason |",
            "|---|---|---:|---:|---|---|---|",
        ]
    )
    for _, row in high.iterrows():
        title = str(row["title"]).replace("|", "/")
        reason = str(row["priority_reason"]).replace("|", "/")
        lines.append(
            f"| {row['reproduction_priority']} | {row['topic_bucket']} | {int(row['year']) if pd.notna(row['year']) else ''} | "
            f"{int(row['cited_by']) if pd.notna(row['cited_by']) else ''} | `{row['doi_clean']}` | {title} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit is a triage layer, not a final reproducibility judgment. `high` means the metadata contains a repository or data-hosting hint. `medium` means the paper is open or influential enough to inspect manually. `low` means no immediate open-data/code signal was found in OpenAlex metadata.",
            "",
            "Next manuscript use: report this file as the start of the field-wide sampling frame so the five case studies are clearly positioned as executable examples rather than a cherry-picked final sample.",
        ]
    )
    (FRAG_DIR / "reproducible_candidate_audit_v0.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] {OUT_DIR / 'reproducible_candidate_audit.csv'}")
    print(f"[OK] {OUT_DIR / 'reproducible_candidate_audit_summary.csv'}")
    print(f"[OK] {FRAG_DIR / 'reproducible_candidate_audit_v0.md'}")
    print(count_table.to_string())


if __name__ == "__main__":
    main()
