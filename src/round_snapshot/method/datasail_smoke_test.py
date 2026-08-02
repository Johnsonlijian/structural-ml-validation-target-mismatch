"""Minimal DataSAIL 1.3.0 custom-distance smoke test."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np

from datasail.sail import datasail


def main() -> None:
    root = Path(__file__).resolve().parent
    ids = [f"record_{index:02d}" for index in range(20)]
    coordinates = np.asarray(
        [[index // 4, index % 4] for index in range(20)],
        dtype=float,
    )
    distance = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :],
        axis=2,
    )
    distance /= distance.max()
    started = time.perf_counter()
    e_splits, f_splits, inter_splits = datasail(
        techniques=["I1e", "C1e"],
        splits=[1, 1, 1, 1, 1],
        names=[f"fold_{fold}" for fold in range(5)],
        runs=1,
        epsilon=0.10,
        solver="SCIP",
        max_sec=60,
        verbose="W",
        e_type="O",
        e_data={name: coordinates[index] for index, name in enumerate(ids)},
        e_dist=(ids, distance),
        e_clusters=10,
    )
    elapsed = time.perf_counter() - started
    output = {
        "status": "PASS",
        "datasail_version": "1.3.0",
        "techniques": ["I1e", "C1e"],
        "solver": "SCIP",
        "installed_solvers": cp.installed_solvers(),
        "n_records": len(ids),
        "splits": [1, 1, 1, 1, 1],
        "epsilon": 0.10,
        "runtime_seconds": elapsed,
        "distance_sha256": hashlib.sha256(distance.tobytes()).hexdigest(),
        "e_splits_repr": repr(e_splits),
        "f_splits_repr": repr(f_splits),
        "inter_splits_repr": repr(inter_splits),
    }
    (root / "datasail_smoke_test_result.json").write_text(
        json.dumps(output, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
