"""
P1 · PRISMA 筛选

读 OpenAlex 检索结果，按以下规则筛：
- 排除：review / proceeding / book chapter / non-English
- 必含 ML 关键词
- 必含 structural / civil 关键词
- 必含 quantitative metric（accuracy / R2 / RMSE / MAE / strength prediction）
- 优先：开放获取 + 引用 ≥ 5

输出：outputs/literature/prisma_screening_<date>.csv

用法：
    python 02_screen.py --in outputs/literature/openalex_full_2026-04-28.parquet
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent / "outputs" / "literature"


ML_TERMS = re.compile(
    r"\b(?:machine learning|deep learning|neural network|random forest|"
    r"gradient boost(?:ing)?|xgboost|catboost|lightgbm|support vector|"
    r"convolutional|transformer|gaussian process|bayesian network)\b",
    re.IGNORECASE,
)

CIVIL_TERMS = re.compile(
    r"\b(?:civil engineering|structural engineering|construction engineering|"
    r"concrete-filled steel tube|concrete filled steel tube|cfst|"
    r"reinforced concrete|rc beam|rc column|shear wall|beam-column|"
    r"steel frame|rc frame|masonry|timber structure|bridge|girder|truss|"
    r"seismic fragility|earthquake engineering|structural health monitoring|"
    r"compressive strength|shear strength|axial capacity|load capacity|"
    r"pile drivability|slope stability|geotechnical|tunnel|dam|building)\b",
    re.IGNORECASE,
)

METRIC_TERMS = re.compile(
    r"\b(?:accuracy|r2|r-squared|rmse|mae|mape|f1[- ]?score|precision|recall|"
    r"auc|capacity|strength|fragility|prediction|regress(?:ion)?)\b",
    re.IGNORECASE,
)

EXCLUDE_TYPES = re.compile(
    r"\b(?:review|systematic review|meta-analysis|bibliometric|chapter|"
    r"proceedings of|conference proceedings)\b",
    re.IGNORECASE,
)

BIO_MED_EXCLUDE = re.compile(
    r"\b(?:protein|proteome|peptide|ligand|drug|cheminformatics|genetic|genome|"
    r"neuroimaging|brain|cardiac|coronary|medical|clinical|patient|tumou?r|"
    r"antimicrobial|molecular|atomistic dynamics|crystal|inorganic crystal|"
    r"materials genome|electronic structure)\b",
    re.IGNORECASE,
)

CIVIL_VENUE_TERMS = re.compile(
    r"(engineering structures|computers and structures|journal of building engineering|"
    r"construction and building materials|computer-aided civil and infrastructure engineering|"
    r"soil dynamics and earthquake engineering|earthquake engineering|structural control|"
    r"structural health monitoring|structures|thin-walled structures|"
    r"journal of structural engineering|engineering applications of artificial intelligence|"
    r"automation in construction|advances in structural engineering|geoscience frontiers|"
    r"applied sciences|materials|scientific reports|buildings|sustainability)",
    re.IGNORECASE,
)


def safe_text(value: object, max_len: int = 140) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text.encode("ascii", "replace").decode("ascii")


def screen(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text = (df["title"].fillna("") + " . " + df["abstract"].fillna("")).str.lower()

    df["has_ml"] = text.str.contains(ML_TERMS, na=False)
    venue = df["venue"].fillna("").astype(str).str.lower()
    df["has_civil_text"] = text.str.contains(CIVIL_TERMS, na=False)
    df["has_civil_venue"] = venue.str.contains(CIVIL_VENUE_TERMS, na=False)
    df["has_civil"] = df["has_civil_text"] | df["has_civil_venue"]
    df["has_metric"] = text.str.contains(METRIC_TERMS, na=False)
    df["is_review"] = text.str.contains(EXCLUDE_TYPES, na=False)
    df["is_bio_med"] = text.str.contains(BIO_MED_EXCLUDE, na=False)

    df["pass_screen"] = (
        df["has_ml"]
        & df["has_civil"]
        & df["has_metric"]
        & ~df["is_review"]
        & ~df["is_bio_med"]
        & (df["year"] >= 2014)
        & (df["cited_by"].fillna(0) >= 5)
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=None, help="输入 parquet/csv；默认取最近一个")
    args = parser.parse_args()

    if args.inp is None:
        candidates = sorted(
            list(ROOT.glob("openalex_*.parquet")) + list(ROOT.glob("openalex_*.csv")),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit("未找到 OpenAlex 检索输出。先跑 01_literature_search.py")
        in_path = candidates[-1]
    else:
        in_path = Path(args.inp)

    if in_path.suffix == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    print(f"[INFO] loaded {len(df)} works from {in_path.name}")
    out = screen(df)

    print()
    print("=== screen counts ===")
    print(f"  has_ml     : {out['has_ml'].sum()}")
    print(f"  has_civil  : {out['has_civil'].sum()}")
    print(f"  has_metric : {out['has_metric'].sum()}")
    print(f"  excluded   : {out['is_review'].sum()}")
    print(f"  bio/med ex : {out['is_bio_med'].sum()}")
    print(f"  PASS       : {out['pass_screen'].sum()}")

    out_path = ROOT / f"prisma_screening_{date.today().isoformat()}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")
    print()
    print(f"[OK] wrote {out_path}")

    pass_df = out[out["pass_screen"]].sort_values("cited_by", ascending=False)
    print()
    print("Top 10 passing screen:")
    top = pass_df.head(10)[["year", "cited_by", "venue", "first_author", "title"]].copy()
    for col in top.columns:
        top[col] = top[col].map(safe_text)
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
