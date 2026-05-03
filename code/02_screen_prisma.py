"""First-pass automatic PRISMA screening of OpenAlex metadata.

Heuristics applied (all soft, can be overridden manually in the output CSV):
- exclude reviews/editorials by venue keyword
- exclude < 2014
- require at least one ML term and one structural term in title or abstract
- require at least one metric term (R2, MAE, RMSE, accuracy, etc.)

Output: data/00_metadata/prisma_screening.csv with `decision` and `reason` columns.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "00_metadata"

ML_TERMS = re.compile(
    r"\b(machine learning|deep learning|neural network|random forest|"
    r"gradient boosting|xgboost|support vector|cnn|rnn|lstm|transformer|"
    r"ensemble|catboost|lightgbm|automl)\b",
    re.I,
)
STRUCT_TERMS = re.compile(
    r"\b(cfst|concrete-filled steel|reinforced concrete|shear wall|"
    r"beam-column|moment frame|fragility|ductility|seismic|column|beam|slab|"
    r"masonry|composite (beam|column)|steel-concrete|truss)\b",
    re.I,
)
METRIC_TERMS = re.compile(
    r"\b(r\^?2|coefficient of determination|mae|mean absolute error|"
    r"rmse|root mean square error|accuracy|f1|auc|cross[- ]validation)\b",
    re.I,
)
EXCLUDE_VENUES = re.compile(r"(review|preprint server|comment|editorial)", re.I)


def screen_row(row: pd.Series) -> tuple[str, str]:
    text = f"{row.get('title') or ''}  {row.get('abstract') or ''}"
    if pd.isna(row.get("year")) or int(row.get("year") or 0) < 2014:
        return "exclude", "year<2014"
    if EXCLUDE_VENUES.search(str(row.get("venue") or "")):
        return "exclude", "venue:review"
    if not ML_TERMS.search(text):
        return "exclude", "no_ml_term"
    if not STRUCT_TERMS.search(text):
        return "exclude", "no_struct_term"
    if not METRIC_TERMS.search(text):
        return "review", "no_metric_term"
    return "include", "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(OUT_DIR / "openalex_search.parquet"))
    ap.add_argument("--out", default=str(OUT_DIR / "prisma_screening.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    df["decision"], df["reason"] = zip(*df.apply(screen_row, axis=1))
    counts = df["decision"].value_counts().to_dict()
    df.sort_values(["decision", "cited_by"], ascending=[True, False]).to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out}")
    print(f"     counts: {counts}")
    print(f"     keep {counts.get('include', 0)}  review {counts.get('review', 0)}  "
          f"exclude {counts.get('exclude', 0)}")


if __name__ == "__main__":
    main()
