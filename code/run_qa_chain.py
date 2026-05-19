"""
One-shot QA chain for Cursor / local terminal:

  1) Run Mangalathu (06) and Mendeley (24) reproduction scripts.
  2) Normalize ``results.csv`` when NAS/sync renames outputs to ``results_*_Conflict.csv``.
  3) Run ``33_cace_nine_module_summary.py`` (Table S7 + Figure S5).
  4) Run ``34_graphical_abstract_cace_v0.py`` (graphical abstract PNG/PDF + editable SVG).

Use the same interpreter you use for other scripts, e.g.::

    python code/run_qa_chain.py

Optional flags: ``--skip-06``, ``--skip-24``, ``--skip-33``, ``--skip-34``, ``--only-sync-results``.
"""

from __future__ import annotations

import argparse
import runpy
import shutil
import sys
from pathlib import Path


CODE = Path(__file__).resolve().parent
REPRO = CODE / "outputs" / "reproductions"
MANGA = REPRO / "10-1016-j-engstruct-2019-110331"
MENDELEY = REPRO / "mendeley_beam_column_joint"


def _pick_latest(paths: list[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("no matching files")
    return max(paths, key=lambda p: p.stat().st_mtime)


def pick_repro_script(stem_prefix: str, canonical: str) -> Path:
    conflicts = list(CODE.glob(f"{stem_prefix}_*Conflict*.py"))
    if conflicts:
        return _pick_latest(conflicts)
    c = CODE / canonical
    if c.is_file():
        return c
    raise FileNotFoundError(f"Neither {stem_prefix}_*Conflict*.py nor {canonical} under {CODE}")


def ensure_results_csv(module_dir: Path, label: str) -> None:
    dst = module_dir / "results.csv"
    conflicts = list(module_dir.glob("results*Conflict*.csv"))
    if conflicts:
        src = _pick_latest(conflicts)
        shutil.copy2(src, dst)
        print(f"[sync] {label}: {src.name} -> results.csv", flush=True)
        return
    if dst.is_file():
        print(f"[sync] {label}: results.csv already present", flush=True)
        return
    raise FileNotFoundError(f"{label}: no results.csv and no results*Conflict*.csv in {module_dir}")


def run_path(label: str, script: Path) -> None:
    print(f"[run] {label}: {script.name}", flush=True)
    runpy.run_path(str(script), run_name="__main__")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run 06 + 24 + normalize results.csv + run 33 + run 34.")
    ap.add_argument("--skip-06", action="store_true", help="Skip Mangalathu reproduction.")
    ap.add_argument("--skip-24", action="store_true", help="Skip Mendeley reproduction.")
    ap.add_argument("--skip-33", action="store_true", help="Skip nine-module summary / Figure S5.")
    ap.add_argument("--skip-34", action="store_true", help="Skip graphical abstract export.")
    ap.add_argument(
        "--only-sync-results",
        action="store_true",
        help="Only copy results*Conflict*.csv -> results.csv for modules 06/24 depend on, then exit.",
    )
    args = ap.parse_args()

    if args.only_sync_results:
        ensure_results_csv(MANGA, "Mangalathu")
        ensure_results_csv(MENDELEY, "Mendeley")
        print("[OK] only-sync-results complete", flush=True)
        return

    if not args.skip_06:
        run_path("06", pick_repro_script("06_reproduce_mangalathu", "06_reproduce_mangalathu_2020_shear_wall.py"))
    if not args.skip_24:
        run_path("24", pick_repro_script("24_reproduce_mendeley", "24_reproduce_mendeley_beam_column_joint.py"))

    ensure_results_csv(MANGA, "Mangalathu")
    ensure_results_csv(MENDELEY, "Mendeley")

    if not args.skip_33:
        p33 = CODE / "33_cace_nine_module_summary.py"
        if not p33.is_file():
            raise FileNotFoundError(p33)
        run_path("33", p33)

    if not args.skip_34:
        p34 = CODE / "34_graphical_abstract_cace_v0.py"
        if p34.is_file():
            run_path("34", p34)
        else:
            print("[warn] skip 34: script not found", flush=True)

    print("[OK] run_qa_chain complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)
