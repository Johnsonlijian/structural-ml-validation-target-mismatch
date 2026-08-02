"""Path-neutral runtime helpers for the public headed-stud evaluator.

The four exported helpers preserve the frozen scientific settings without
importing the private confirmatory runner.  Fold assignments are returned only
in memory.  Public callers must persist aggregate audits or checksums, never the
assignment vector itself.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import random
import re
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


PINNED_DATASAIL_VERSION = "1.3.0"


def assignments_from_splitter(splitter: Any, X: Any, y: Any, groups: Any = None) -> np.ndarray:
    """Materialize one integer fold label per record from a scikit-learn splitter."""

    n_records = len(X)
    assignments = np.full(n_records, -1, dtype=int)
    for fold, (_, test) in enumerate(splitter.split(X, y, groups=groups)):
        test_indices = np.asarray(test, dtype=int)
        if test_indices.ndim != 1:
            raise RuntimeError("splitter returned a non-vector test index")
        if np.any((test_indices < 0) | (test_indices >= n_records)):
            raise RuntimeError("splitter returned an out-of-range test index")
        if np.any(assignments[test_indices] >= 0):
            raise RuntimeError("splitter assigned at least one record more than once")
        assignments[test_indices] = fold
    if np.any(assignments < 0):
        raise RuntimeError("splitter left unassigned records")
    return assignments


def build_models(seed: int) -> dict[str, object]:
    """Return the four frozen learners in their frozen insertion order."""

    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RBF_SVR": make_pipeline(
            StandardScaler(),
            SVR(C=10.0, epsilon=0.05, gamma="scale"),
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=160,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=int(seed),
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=int(seed),
        ),
    }


def atomic_write_json(path: Path | str, payload: dict[str, Any]) -> None:
    """Atomically replace a JSON receipt without exposing a partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    name_token = hashlib.sha256(destination.name.encode("utf-8")).hexdigest()[:6]
    temporary = destination.with_name(f".__tmp_{os.getpid()}_{name_token}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_datasail_solver_status(log: str, *, assignment_returned: bool) -> str:
    """Parse a conservative solver status without upgrading an ambiguous log."""

    lower = log.lower()
    if "solution may be inaccurate" in lower or "optimal_inaccurate" in lower:
        return "optimal_inaccurate"
    matches = re.findall(r"(?:solver\s+)?status\s*[:=]\s*([a-z_]+)", lower)
    if matches:
        return matches[-1]
    if "user_limit" in lower or "time limit" in lower:
        return "user_limit"
    if "optimal" in lower:
        return "optimal"
    return (
        "solution_returned_status_unparsed"
        if assignment_returned
        else "no_assignment_status_unparsed"
    )


def _require_package(distribution: str, expected: str | None = None) -> str:
    try:
        observed = version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(f"required package is not installed: {distribution}") from exc
    if expected is not None and observed != expected:
        raise RuntimeError(
            f"{distribution} version mismatch: expected {expected}, observed {observed}"
        )
    return observed


def datasail_scalar_assignment(
    provenance: pd.DataFrame,
    claim: Any,
    *,
    seed: int,
    max_sec: int,
    relation_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the frozen DataSAIL-C1e scalar proxy and return an in-memory split.

    The caller may publish the returned metadata and an assignment checksum, but
    must not serialize the returned assignment vector.
    """

    datasail_version = _require_package("datasail", PINNED_DATASAIL_VERSION)
    pyscipopt_version = _require_package("pyscipopt")
    from datasail.sail import datasail

    if int(max_sec) <= 0:
        raise ValueError("max_sec must be positive for the fail-closed public evaluator")
    np.random.seed(int(seed))
    random.seed(int(seed))
    frame = provenance.reset_index(drop=True)
    n_records = len(frame)
    if n_records < int(claim.n_splits):
        raise ValueError("fewer records than requested folds")
    ids = [f"record_{index:06d}" for index in range(n_records)]
    similarity = np.zeros((n_records, n_records), dtype=np.float64)
    total_weight = 0.0
    scalar_weights: dict[str, float] = {}
    compiled_weights = relation_weights or {
        target.name: float(target.weight * target.target)
        for target in claim.optimized_relations
    }
    for relation, raw_weight in compiled_weights.items():
        if relation not in frame.columns:
            raise KeyError(f"DataSAIL relation is absent: {relation}")
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight <= 0:
            continue
        values = frame[relation]
        observed = values.notna().to_numpy()
        string_values = values.astype(str).to_numpy()
        equal = string_values[:, None] == string_values[None, :]
        equal &= observed[:, None] & observed[None, :]
        similarity += weight * equal
        total_weight += weight
        scalar_weights[str(relation)] = weight
    if total_weight <= 0:
        raise ValueError("zero scalar similarity weight")
    similarity /= total_weight
    np.fill_diagonal(similarity, 1.0)
    n_clusters = min(50, max(10, n_records // 20))
    solver_capture = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(solver_capture), contextlib.redirect_stderr(
        solver_capture
    ):
        e_splits, _, _ = datasail(
            techniques=["C1e"],
            splits=[1] * int(claim.n_splits),
            names=[f"fold_{fold}" for fold in range(int(claim.n_splits))],
            runs=1,
            epsilon=0.05,
            solver="SCIP",
            max_sec=int(max_sec),
            verbose="W",
            e_type="O",
            e_data={name: np.asarray([index], dtype=float) for index, name in enumerate(ids)},
            e_sim=(ids, similarity),
            e_clusters=n_clusters,
        )
    runtime_seconds = float(time.perf_counter() - started)
    full_log = solver_capture.getvalue()
    try:
        mapping = e_splits["C1e"][0]
        assignments = np.asarray(
            [int(str(mapping[name]).split("_")[-1]) for name in ids],
            dtype=int,
        )
    except Exception as exc:
        raise RuntimeError("DataSAIL returned no complete C1e assignment") from exc
    if len(assignments) != n_records or np.any(
        (assignments < 0) | (assignments >= int(claim.n_splits))
    ):
        raise RuntimeError("DataSAIL returned an invalid fold vector")
    return assignments, {
        "version": datasail_version,
        "technique": "C1e",
        "solver": "SCIP",
        "pyscipopt_version": pyscipopt_version,
        "max_sec": int(max_sec),
        "epsilon": 0.05,
        "n_clusters": int(n_clusters),
        "scalar_relation_weights": scalar_weights,
        "similarity_sha256": hashlib.sha256(similarity.tobytes()).hexdigest(),
        "runtime_seconds": runtime_seconds,
        "solver_status": parse_datasail_solver_status(
            full_log, assignment_returned=True
        ),
        "assignment_returned": True,
        "solver_time_limit_likely": bool(
            runtime_seconds >= 0.98 * float(max_sec)
        ),
        "solver_log_sha256": hashlib.sha256(
            full_log.encode("utf-8", errors="replace")
        ).hexdigest(),
    }


__all__ = [
    "assignments_from_splitter",
    "atomic_write_json",
    "build_models",
    "datasail_scalar_assignment",
]
