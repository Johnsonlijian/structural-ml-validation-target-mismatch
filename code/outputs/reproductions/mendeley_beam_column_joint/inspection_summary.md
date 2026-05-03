# Mendeley Beam-Column Joint Dataset Inspection

Purpose: determine whether the newly downloaded Mendeley datasets contain source labels and target columns suitable for source-aware reproduction.

| DOI | Sheet | Rows | Columns | Source-like columns | Target-like columns |
|---|---|---:|---:|---|---|
| 10.17632/8ndgpm7zw7.1 | Dataset | 203 | 19 | Authors | none detected |
| 10.17632/8ndgpm7zw7.1 | Notations | 18 | 2 | none detected | none detected |
| 10.17632/8ndgpm7zw7.1 | Authors | 34 | 2 | Authors list | none detected |
| 10.17632/rbhfnz32sy.1 | Sheet1 | 98 | 21 | Research Team | Predicted joint shear strength (MPa) based on kernel; joint shear strength (MPa) |
| 10.17632/rbhfnz32sy.1 | Sheet2 | 7 | 6 | none detected | none detected |
| 10.17632/rbhfnz32sy.1 | Sheet3 | 0 | 0 | none detected | none detected |

## Initial decision

- `10.17632/8ndgpm7zw7.1` is prioritized if its Excel workbook exposes reference/source identifiers together with shear strength or failure-mode targets.
- `10.17632/rbhfnz32sy.1` is prioritized if it exposes the 18 research-project/source grouping described on the landing page.
- If a source column is missing, the dataset may still be useful as an open benchmark but cannot enter the source-reference reproduction pool without reconstructing groups from references.
