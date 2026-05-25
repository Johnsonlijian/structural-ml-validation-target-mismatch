from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "mendeley_beam_column_joint"
OUT = ROOT / "code" / "outputs" / "reproductions" / "mendeley_beam_column_joint"

FILES = {
    "10.17632/8ndgpm7zw7.1": DATA
    / "10.17632-8ndgpm7zw7.1"
    / "Teklewoin_Joint_Dataset.xlsx",
    "10.17632/rbhfnz32sy.1": DATA
    / "10.17632-rbhfnz32sy.1"
    / "Salem_Beam_Column_Joint_Cyclic_Shear.xlsx",
}


def clean_columns(columns: list[object]) -> list[str]:
    return [str(col).strip().replace("\n", " ") for col in columns]


def inspect_file(doi: str, path: Path) -> list[dict[str, object]]:
    xls = pd.ExcelFile(path)
    records: list[dict[str, object]] = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        df.columns = clean_columns(list(df.columns))
        nonempty = df.dropna(how="all")
        columns = list(df.columns)
        source_like = [
            col
            for col in columns
            if any(token in col.lower() for token in ["ref", "source", "author", "study", "paper", "research"])
        ]
        target_like = [
            col
            for col in columns
            if any(token in col.lower() for token in ["shear", "strength", "failure", "mode", "capacity"])
        ]
        records.append(
            {
                "doi": doi,
                "file": path.name,
                "sheet": sheet,
                "n_rows": int(len(nonempty)),
                "n_columns": int(len(columns)),
                "columns": columns,
                "source_like_columns": source_like,
                "target_like_columns": target_like,
                "first_nonempty_rows": nonempty.head(3).astype(str).to_dict(orient="records"),
            }
        )
    return records


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, object]] = []
    for doi, path in FILES.items():
        all_records.extend(inspect_file(doi, path))

    (OUT / "inspection.json").write_text(json.dumps(all_records, indent=2, ensure_ascii=False), encoding="utf-8")

    flat = pd.DataFrame(
        [
            {
                "doi": r["doi"],
                "file": r["file"],
                "sheet": r["sheet"],
                "n_rows": r["n_rows"],
                "n_columns": r["n_columns"],
                "source_like_columns": "; ".join(r["source_like_columns"]),
                "target_like_columns": "; ".join(r["target_like_columns"]),
            }
            for r in all_records
        ]
    )
    flat.to_csv(OUT / "inspection_summary.csv", index=False)

    lines = [
        "# Mendeley Beam-Column Joint Dataset Inspection",
        "",
        "Purpose: determine whether the newly downloaded Mendeley datasets contain source labels and target columns suitable for source-aware reproduction.",
        "",
        "| DOI | Sheet | Rows | Columns | Source-like columns | Target-like columns |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in flat.itertuples(index=False):
        lines.append(
            f"| {row.doi} | {row.sheet} | {row.n_rows} | {row.n_columns} | "
            f"{row.source_like_columns or 'none detected'} | {row.target_like_columns or 'none detected'} |"
        )

    lines.extend(
        [
            "",
            "## Initial decision",
            "",
            "- `10.17632/8ndgpm7zw7.1` is prioritized if its Excel workbook exposes reference/source identifiers together with shear strength or failure-mode targets.",
            "- `10.17632/rbhfnz32sy.1` is prioritized if it exposes the 18 research-project/source grouping described on the landing page.",
            "- If a source column is missing, the dataset may still be useful as an open benchmark but cannot enter the source-reference reproduction pool without reconstructing groups from references.",
            "",
        ]
    )
    (OUT / "inspection_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
