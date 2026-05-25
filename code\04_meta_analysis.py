"""Summarize reproduction JSON files into compact meta-analysis tables.

Example:
    python 04_meta_analysis.py
    python 04_meta_analysis.py --inp code/outputs/reproductions --out code/outputs/meta_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_IN = Path(__file__).resolve().parent / "outputs" / "reproductions"
DEFAULT_OUT = Path(__file__).resolve().parent / "outputs" / "meta_results"


def load_all(reprod_dir: Path) -> pd.DataFrame:
    rows = []
    for p in reprod_dir.glob("*/reproduction.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] {p} unreadable: {exc}")
            continue
        row = {
            "doi": d.get("doi"),
            "model_family": d.get("model_family"),
            "n_samples": d.get("n_samples"),
            "n_groups": d.get("n_groups"),
            "reported_r2": d.get("reported_r2"),
            "delta_r2_random_vs_group": d.get("delta_r2_random_vs_group"),
            "delta_r2_random_vs_loso": d.get("delta_r2_random_vs_loso"),
            "code_runs": d.get("code_runs"),
        }
        rep = d.get("reproduced", {})
        for k, v in rep.items():
            row[f"r2_{k}"] = v.get("r2_mean")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    res: dict = {}
    if df.empty:
        return res

    valid = df.dropna(subset=["delta_r2_random_vs_group"])
    if not valid.empty:
        res["n_reproduced"] = int(len(valid))
        res["delta_r2_median"] = float(valid["delta_r2_random_vs_group"].median())
        res["delta_r2_mean"] = float(valid["delta_r2_random_vs_group"].mean())
        res["delta_r2_q25"] = float(valid["delta_r2_random_vs_group"].quantile(0.25))
        res["delta_r2_q75"] = float(valid["delta_r2_random_vs_group"].quantile(0.75))
        res["pct_inflation_above_0.20"] = float(
            (valid["delta_r2_random_vs_group"] > 0.20).mean()
        )

    high_claim = df[df["reported_r2"].notna() & (df["reported_r2"] >= 0.9)]
    if not high_claim.empty:
        survives = (high_claim["r2_GroupKFold-5"].fillna(-1) >= 0.7).mean()
        res["pct_high_claim_papers"] = float((df["reported_r2"].fillna(-1) >= 0.9).mean())
        res["pct_high_claim_surviving_group_cv"] = float(survives)

    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", default=str(DEFAULT_IN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    in_dir = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[WARN] {in_dir} 不存在；元分析输出空骨架供 W7+ 使用")
        df = pd.DataFrame()
    else:
        df = load_all(in_dir)

    df.to_csv(out_dir / "reproductions_table.csv", index=False)
    summary = summarize(df)
    (out_dir / "meta_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[OK] {len(df)} reproductions -> {out_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
