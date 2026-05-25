from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "code" / "outputs" / "reproductions"
OUT = ROOT / "manuscript" / "tables"


CASES = [
    {
        "case": "Shear wall failure mode",
        "doi_or_source": "10.1016/j.engstruct.2019.110331",
        "group_proxy": "Author/source family",
        "group_file": REPRO / "10-1016-j-engstruct-2019-110331" / "author_group_sizes.csv",
        "diagnostics": REPRO / "10-1016-j-engstruct-2019-110331" / "group_diagnostics.json",
        "interpretation": "source-family dependence plus class/topology stress",
    },
    {
        "case": "SFRC shear capacity",
        "doi_or_source": "10.1016/j.engstruct.2020.111743",
        "group_proxy": "Source reference",
        "group_file": REPRO / "10-1016-j-engstruct-2020-111743" / "source_group_sizes.csv",
        "diagnostics": None,
        "interpretation": "source-reference generalization",
    },
    {
        "case": "UCI concrete strength",
        "doi_or_source": "10.1016/j.conbuildmat.2020.120950 / UCI",
        "group_proxy": "Identical mix proportions",
        "group_file": REPRO / "10-1016-j-conbuildmat-2020-120950" / "mix_group_sizes.csv",
        "diagnostics": None,
        "interpretation": "mix-family contrast",
    },
    {
        "case": "Corroded RC beam moment",
        "doi_or_source": "10.5281/zenodo.8062007",
        "group_proxy": "Experimental program",
        "group_file": REPRO / "10-5281-zenodo-8062007" / "source_group_sizes.csv",
        "diagnostics": None,
        "interpretation": "source-reference optimism with useful absolute performance",
    },
    {
        "case": "Stub-CFST extension",
        "doi_or_source": "10.1038/s41598-024-53352-1",
        "group_proxy": "Section family",
        "group_file": REPRO / "10-1038-s41598-024-53352-1" / "shape_group_sizes.csv",
        "diagnostics": None,
        "interpretation": "deployment-matched structural-family extrapolation",
    },
]


def summarize(case: dict[str, object]) -> dict[str, object]:
    groups = pd.read_csv(case["group_file"])
    sizes = groups["n"]

    out: dict[str, object] = {
        "case": case["case"],
        "doi_or_source": case["doi_or_source"],
        "group_proxy": case["group_proxy"],
        "n_samples": int(sizes.sum()),
        "n_groups": int(len(sizes)),
        "median_group_size": float(sizes.median()),
        "max_group_size": int(sizes.max()),
        "singleton_groups": int((sizes == 1).sum()),
        "singleton_share": round(float((sizes == 1).mean()), 3),
        "groups_with_single_class": "",
        "median_majority_share": "",
        "interpretation": case["interpretation"],
    }

    diagnostics = case.get("diagnostics")
    if diagnostics:
        diag = json.loads(Path(diagnostics).read_text(encoding="utf-8"))
        out["groups_with_single_class"] = int(diag["groups_with_single_class"])
        out["median_majority_share"] = float(diag["median_majority_share"])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([summarize(case) for case in CASES])
    table.to_csv(OUT / "supplementary_group_topology_table_v0.csv", index=False)

    lines = [
        "# Supplementary Group Topology Table v0",
        "",
        "This table reports the topology behind each group-aware validation case. It is intended to prevent interpretation of every random-vs-group gap as the same mechanism.",
        "",
        "| Case | Group proxy | n | Groups | Median group size | Max group size | Singleton groups | Class/topology note | Interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in table.itertuples(index=False):
        class_note = "not applicable"
        if row.groups_with_single_class != "":
            class_note = f"{row.groups_with_single_class} single-class groups; median majority share {row.median_majority_share:.2f}"
        lines.append(
            f"| {row.case} | {row.group_proxy} | {row.n_samples} | {row.n_groups} | "
            f"{row.median_group_size:g} | {row.max_group_size} | {row.singleton_groups} | "
            f"{class_note} | {row.interpretation} |"
        )

    lines.extend(
        [
            "",
            "## Data sources",
            "",
            "- Group-size CSV files under `code/outputs/reproductions/*/`.",
            "- Shear-wall class/topology diagnostics from `group_diagnostics.json`.",
        ]
    )
    (OUT / "supplementary_group_topology_table_v0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
