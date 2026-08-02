"""Compatibility entry point for the canonical public Figure 4 builder.

The earlier core snapshot duplicated the primary/scaling implementation and
could drift from its submission-facing evidence gates.  This module delegates
to ``figures/contract_compiler/build_primary_scaling_figure.py`` so both public
entry points use the same Attempt-S retain-all, aggregate-hash and scalability
gates.
"""

from __future__ import annotations

import sys
from pathlib import Path


FIGURES_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_COMPILER_ROOT = FIGURES_ROOT / "contract_compiler"
PACKAGE_ROOT = FIGURES_ROOT.parent
if str(CONTRACT_COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRACT_COMPILER_ROOT))

import build_primary_scaling_figure as _canonical


Figure4GateError = _canonical.Figure4GateError
Figure4Data = _canonical.Figure4Data
Comparison = _canonical.Comparison
ScalingPoint = _canonical.ScalingPoint
build_figure = _canonical.build_figure
_load_primary_evidence_lock = _canonical._load_primary_evidence_lock
_load_primary_rows = _canonical._load_primary_rows
_load_scalability = _canonical._load_scalability
_write_receipt = _canonical._write_receipt

DEFAULT_AGGREGATE_ROOT = PACKAGE_ROOT / "derived" / "confirmatory_aggregates_v32"
DEFAULT_SCALABILITY_ROOT = PACKAGE_ROOT / "derived" / "scalability_v1"
DEFAULT_PRIMARY_EVIDENCE_LOCK = (
    CONTRACT_COMPILER_ROOT / "PRIMARY_EVIDENCE_LOCK.json"
)
DEFAULT_RETENTION_RECEIPT = (
    PACKAGE_ROOT / "provenance" / "ATTEMPT_S_RETENTION_AUDIT.json"
)


def load_verified_data(
    a32_root: Path = DEFAULT_AGGREGATE_ROOT,
    scalability_root: Path = DEFAULT_SCALABILITY_ROOT,
    primary_evidence_lock: Path = DEFAULT_PRIMARY_EVIDENCE_LOCK,
    retention_receipt: Path = DEFAULT_RETENTION_RECEIPT,
) -> Figure4Data:
    return _canonical.load_verified_data(
        a32_root,
        scalability_root,
        primary_evidence_lock,
        retention_receipt,
    )


def main() -> None:
    _canonical.main()


if __name__ == "__main__":
    main()
