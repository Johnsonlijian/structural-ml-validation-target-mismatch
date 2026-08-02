"""Path-neutral wrapper for the headed-stud consequence figure.

Copy the original ``build_figure_6.py`` to ``_impl/build_figure_6.py`` beside
this wrapper.  The wrapper verifies and concatenates only license-separated,
row-free aggregate tables from the public projection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_impl(path: Path):
    spec = importlib.util.spec_from_file_location("headed_stud_figure6_impl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load figure implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verified_csv(root: Path, receipt: dict, relative: str) -> pd.DataFrame:
    path = root / relative
    expected = receipt["files"][relative]
    if path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected["sha256"]:
        raise RuntimeError(f"aggregate hash mismatch: {relative}")
    return pd.read_csv(path)


def gather(root: Path, stem: str) -> pd.DataFrame:
    receipt = json.loads((root / "RELEASE_RECEIPT.json").read_text(encoding="utf-8"))
    relatives = sorted(
        relative
        for relative in receipt["files"]
        if Path(relative).name == stem + ".csv"
    )
    if not relatives:
        raise RuntimeError(f"aggregate not found: {stem}")
    return pd.concat([verified_csv(root, receipt, relative) for relative in relatives], ignore_index=True)


def evidence_loader(root: Path, model_order: list[str]) -> dict:
    support = gather(root, "control_multivariate_support")
    metrics = gather(root, "control_prediction_ratio_metrics")
    intervals = gather(root, "control_source_composition_intervals")
    positive = gather(root, "control_deck_positive_control")
    selection = gather(root, "evaluation_model_selection")
    deck_metrics = metrics.loc[
        (metrics["parent"] == "deck464") & metrics["model"].isin(model_order)
    ]
    deck_intervals = intervals.loc[
        (intervals["parent"] == "deck464") & intervals["model"].isin(model_order)
    ]
    positive_subset = positive.loc[
        positive["model"].isin(
            ["HistGradientBoosting", "DeckSpecific_Eq11_positive_control"]
        )
    ]
    deck_selection = selection.loc[selection["parent"] == "deck464"].copy()
    if set(support["parent"]) != {"deck464", "lwc90", "rac27"}:
        raise RuntimeError("support-parent set changed")
    if set(deck_metrics["model"]) != set(model_order):
        raise RuntimeError("deck metric model set changed")
    if set(deck_intervals["model"]) != set(model_order):
        raise RuntimeError("deck interval model set changed")
    if set(deck_intervals["status"]) != {"source_composition_resampling_interval"}:
        raise RuntimeError("deck interval status changed")
    if set(positive_subset["n"].astype(int)) != {401}:
        raise RuntimeError("positive-control subset changed")
    if len(deck_selection) != 4:
        raise RuntimeError("deck model-selection row count changed")
    return {
        "manifest": json.loads((root / "RELEASE_RECEIPT.json").read_text(encoding="utf-8")),
        "support": support,
        "metrics": deck_metrics,
        "intervals": deck_intervals,
        "positive": positive_subset,
        "selection": deck_selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path(__file__).resolve().parent / "_impl/build_figure_6.py",
    )
    args = parser.parse_args()
    implementation = load_impl(args.implementation)
    implementation.load_evidence = lambda: evidence_loader(
        args.evidence_root, list(implementation.MODEL_ORDER)
    )
    implementation.SVG_PATH = args.output_dir / "svg/Figure_6.svg"
    implementation.PDF_PATH = args.output_dir / "pdf/Figure_6.pdf"
    implementation.PNG_PATH = args.output_dir / "png/Figure_6.png"
    implementation.main()


if __name__ == "__main__":
    main()
