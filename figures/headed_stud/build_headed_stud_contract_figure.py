"""Path-neutral wrapper for the headed-stud contract-state figure.

Copy the original ``build_figure_5.py`` to ``_impl/build_figure_5.py`` beside
this wrapper.  Evidence is read only from the row-free public projection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


EXPECTED_STATUSES = {
    "C_source_claimcut": "compiled_exact",
    "C_source_groupkfold": "realized_plan_invalid_balance",
    "C_deck_claimcut": "certified_partition_infeasible",
    "C_lwc_claimcut": "certified_partition_infeasible",
    "C_rac_claimcut": "certified_partition_infeasible",
    "C_deck_datasail_lossy_plan": "plan_emitted_lossy",
    "C_deck_datasail_lossy_audited": "assignment_audited_lossy",
}


def load_impl(path: Path):
    spec = importlib.util.spec_from_file_location("headed_stud_figure5_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load figure implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence_loader(evidence: Path):
    adapter = json.loads((evidence / "receipts/adapter_receipt.json").read_text(encoding="utf-8"))
    contracts = json.loads((evidence / "results/contract_statuses.json").read_text(encoding="utf-8"))
    summaries = {item["parent"]: item for item in adapter["summaries"]}
    statuses = contracts["status_summary"]
    expected_counts = {
        "nwc242": (242, 242, 20),
        "deck464": (464, 413, 21),
        "lwc90": (90, 60, 5),
        "rac27": (27, 27, 1),
    }
    for parent, (raw, retained, groups) in expected_counts.items():
        item = summaries[parent]
        if (int(item["n_raw"]), int(item["n_retained"]), int(item["n_retained_exact_source_groups"])) != (raw, retained, groups):
            raise RuntimeError(f"unexpected adapter summary: {parent}")
    for key, expected in EXPECTED_STATUSES.items():
        if statuses[key]["status"] != expected:
            raise RuntimeError(f"unexpected contract status: {key}")
    if statuses["C_deck_datasail_lossy_audited"]["full_contract_satisfied"] is not False:
        raise RuntimeError("deck DataSAIL full-contract flag changed")
    return summaries, statuses


def multirelation_loader(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_cells = {
        ("solid", "normal_weight"): (242, 20),
        ("solid", "lightweight"): (60, 5),
        ("profiled_deck", "normal_weight"): (401, 21),
    }
    cells = {
        (row["slab_topology"], row["concrete_family"]): row
        for row in payload["cells"]
    }
    if set(cells) != set(expected_cells):
        raise RuntimeError("unexpected multi-relation support cells")
    for key, expected in expected_cells.items():
        observed = (int(cells[key]["n_records"]), int(cells[key]["n_source_groups"]))
        if observed != expected:
            raise RuntimeError(f"unexpected multi-relation cell: {key}")
    if payload["fold_sizes"] != [141, 141, 141, 140, 140]:
        raise RuntimeError("unexpected multi-relation fold sizes")
    if int(payload["min_lwc_per_fold"]) != 3:
        raise RuntimeError("unexpected minimum LWC cell count")
    if payload["claimcut_status"] != "compiled_exact":
        raise RuntimeError("unexpected multi-relation ClaimCut status")
    if payload["group_kfold_status"] != "realized_plan_invalid_balance":
        raise RuntimeError("unexpected multi-relation GroupKFold status")
    if payload["datasail_status"] != "plan_emitted_lossy":
        raise RuntimeError("unexpected multi-relation DataSAIL status")
    if payload["evidence_boundary"] != "metadata_only_observed_union":
        raise RuntimeError("unexpected multi-relation evidence boundary")
    return {
        "cells": cells,
        "fold_sizes": payload["fold_sizes"],
        "min_lwc_per_fold": int(payload["min_lwc_per_fold"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--multirelation-evidence",
        type=Path,
        default=Path(__file__).resolve().parent / "multirelation_evidence.json",
    )
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(__file__).resolve().parent / "_impl/build_figure_5.py",
    )
    args = parser.parse_args()
    implementation = load_impl(args.implementation)
    implementation.load_evidence = lambda: evidence_loader(args.evidence_root)
    implementation.load_multirelation_evidence = lambda: multirelation_loader(
        args.multirelation_evidence
    )
    implementation.SVG_PATH = args.output_dir / "svg/Figure_5.svg"
    implementation.PDF_PATH = args.output_dir / "pdf/Figure_5.pdf"
    implementation.PNG_PATH = args.output_dir / "png/Figure_5.png"
    implementation.main()


if __name__ == "__main__":
    main()
