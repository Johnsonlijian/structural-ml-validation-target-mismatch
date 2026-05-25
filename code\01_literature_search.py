"""
P1 · OpenAlex 文献检索

通过 OpenAlex API 拉取 2014-至今 结构工程 ML 论文 metadata。
免认证，但建议设置 mailto 头以提高优先级。

输出：outputs/literature/openalex_<date>.parquet  +  .csv

用法：
    python 01_literature_search.py --max 5000
    python 01_literature_search.py --max 200 --quick
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError
from datetime import date
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs" / "literature"
OUT.mkdir(parents=True, exist_ok=True)

OPENALEX_BASE = "https://api.openalex.org/works"

# OpenAlex 推荐设置 mailto 提供 polite pool 优先级，免 key
USER_AGENT = "smll-literature-fetch/0.1 (mailto:sx@imut.edu.cn)"


def reconstruct_abstract(inv_idx: dict | None) -> str:
    """OpenAlex 把 abstract 倒排了；这里还原成普通字符串。"""
    if not inv_idx:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv_idx.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def safe_text(value: object, max_len: int = 140) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text.encode("ascii", "replace").decode("ascii")


def query_terms(quick: bool = False) -> list[str]:
    if quick:
        return ["machine learning concrete-filled steel tube compressive strength"]

    # OpenAlex search is not a boolean query language. Several simple searches
    # merged by DOI/OpenAlex ID are far more stable than one long OR/AND string.
    return [
        "machine learning structural strength prediction",
        "machine learning reinforced concrete shear strength",
        "machine learning concrete-filled steel tube CFST",
        "deep learning structural engineering capacity prediction",
        "random forest structural engineering strength prediction",
        "XGBoost concrete strength structural engineering",
        "machine learning seismic fragility structural engineering",
        "machine learning bridge structural performance prediction",
        "machine learning shear wall strength prediction",
        "machine learning beam column capacity prediction",
    ]


def build_query(search: str) -> dict:
    """
    PRISMA 检索式翻成 OpenAlex search 参数。
    OpenAlex 的 search 是 abstract+title+fulltext 的混合检索。
    """
    return {
        "search": search,
        "filter": ",".join(
            [
                "from_publication_date:2014-01-01",
                "type:article",
                "language:en",
            ]
        ),
        "per-page": 200,
        "sort": "cited_by_count:desc",
    }


def fetch_page(query: dict, cursor: str = "*") -> dict:
    q = dict(query)
    q["cursor"] = cursor
    url = f"{OPENALEX_BASE}?{urllib.parse.urlencode(q)}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def extract(work: dict) -> dict:
    primary_loc = (work.get("primary_location") or {}) or {}
    src = primary_loc.get("source") or {}
    authors = work.get("authorships") or []
    return {
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("title"),
        "year": work.get("publication_year"),
        "venue": src.get("display_name"),
        "venue_issn": (src.get("issn_l") or ""),
        "is_oa": (work.get("open_access") or {}).get("is_oa"),
        "oa_url": (work.get("open_access") or {}).get("oa_url"),
        "cited_by": work.get("cited_by_count", 0),
        "n_authors": len(authors),
        "first_author": (authors[0]["author"]["display_name"] if authors else ""),
        "first_aff": (
            authors[0]["institutions"][0]["display_name"]
            if authors and authors[0].get("institutions")
            else ""
        ),
        "concepts": ";".join(
            c["display_name"] for c in (work.get("concepts") or [])[:5]
        ),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
    }


def fetch_for_term(search: str, max_per_term: int) -> list[dict]:
    query = build_query(search)
    rows: list[dict] = []
    cursor = "*"
    page = 0
    while len(rows) < max_per_term and cursor:
        page += 1
        try:
            payload = fetch_page(query, cursor=cursor)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            print(f"[WARN] HTTP {exc.code} for search={safe_text(search)} page={page}: {safe_text(body)}")
            break
        except (URLError, TimeoutError) as exc:
            print(f"[WARN] network error for search={safe_text(search)} page={page}: {exc}; retrying after 5s")
            time.sleep(5)
            continue
        except Exception as exc:
            print(f"[WARN] page {page} failed for search={safe_text(search)}: {exc}")
            break

        works = payload.get("results", [])
        if not works:
            break
        for w in works:
            rows.append(extract(w))
            if len(rows) >= max_per_term:
                break

        meta = payload.get("meta", {})
        cursor = meta.get("next_cursor") or ""
        print(f"  {safe_text(search, 55):55s} | page {page:2d} | got {len(works):3d} | term_total {len(rows):4d}")
        time.sleep(0.2)

    return rows


def fetch_all(max_n: int, quick: bool = False) -> pd.DataFrame:
    terms = query_terms(quick=quick)
    max_per_term = max(50, (max_n // len(terms)) + 25)
    seen: set[str] = set()
    rows: list[dict] = []

    for search in terms:
        if len(rows) >= max_n:
            break
        for row in fetch_for_term(search, max_per_term=max_per_term):
            key = row.get("doi") or row.get("openalex_id") or row.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(row)
            if len(rows) >= max_n:
                break

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=5000)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    df = fetch_all(args.max, quick=args.quick)
    if df.empty:
        print("[ERR] no results", file=sys.stderr)
        sys.exit(1)

    today = date.today().isoformat()
    suffix = "quick" if args.quick else "full"
    out_csv = OUT / f"openalex_{suffix}_{today}.csv"
    out_parquet = OUT / f"openalex_{suffix}_{today}.parquet"

    df.to_csv(out_csv, index=False, encoding="utf-8")
    try:
        df.to_parquet(out_parquet, index=False)
        parquet_msg = f", {out_parquet.name}"
    except Exception as exc:
        parquet_msg = f" (parquet skipped: {exc})"

    print()
    print(f"[OK] {len(df):d} works -> {out_csv.name}{parquet_msg}")
    print()
    print("Top 5 by citations:")
    cols = ["year", "cited_by", "venue", "first_author", "title"]
    top = df.sort_values("cited_by", ascending=False).head(5)[cols].copy()
    for col in cols:
        top[col] = top[col].map(safe_text)
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
