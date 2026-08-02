"""Apply frozen citation-only overlap decisions without reading targets."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT = HERE / "source_audit.csv"
REVIEW = HERE / "manual_overlap_review_v1.json"
OUTPUT = HERE / "source_audit_adjudicated.csv"


def main() -> None:
    frame = pd.read_csv(INPUT).fillna("")
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    if review.get("target_values_accessed") is not False:
        raise RuntimeError("manual review did not preserve target blinding")
    external = frame["parent"] == "deck464"
    citation = frame["source_citations"].astype(str)
    hicks = external & citation.str.contains(
        "Longitudinal Shear Resistance of Steel and Concrete Composite Beams",
        case=False,
        regex=False,
    )
    rambo = external & citation.str.contains(
        "Rambo-Roddenberry M, Easterling WS, Murray TM",
        case=False,
        regex=False,
    )
    frame["manual_overlap_decision"] = "retain"
    frame.loc[hicks, "manual_overlap_decision"] = (
        "quarantine_hicks_cambridge_thesis"
    )
    frame.loc[rambo, "manual_overlap_decision"] = (
        "quarantine_rambo_roddenberry_vt_2002"
    )
    frame["final_quarantine"] = (
        frame["quarantine_candidate"].astype(bool)
        | frame["manual_overlap_decision"].str.startswith("quarantine")
    )
    frame.to_csv(OUTPUT, index=False, encoding="utf-8")
    summaries = []
    for parent, group in frame.groupby("parent", sort=False):
        retained = group.loc[~group["final_quarantine"]]
        source_keys = {
            token
            for value in retained["exact_source_keys"]
            for token in str(value).split("|")
            if token
        }
        summaries.append(
            {
                "parent": parent,
                "n_rows": int(len(group)),
                "n_final_quarantine": int(group["final_quarantine"].sum()),
                "n_retained": int(len(retained)),
                "n_retained_exact_source_keys": len(source_keys),
                "source_recovery_rate": float(group["source_recovered"].mean()),
            }
        )
    payload = {
        "schema_version": "r30-headed-stud-overlap-adjudicated-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_accessed": False,
        "review_sha256": hashlib.sha256(REVIEW.read_bytes()).hexdigest(),
        "adjudicated_csv_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
        "summary": summaries,
    }
    (HERE / "overlap_adjudicated.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
