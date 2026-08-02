"""Independent verifier for realized deployment-contract assignments.

This module intentionally does not import ``provenance_cut`` or call solver,
compiler, audit, objective, or hashing internals.  It consumes only a plain
contract dictionary, a provenance table, and an assignment vector.  The
duplication is deliberate: it supplies an implementation-diverse oracle for
property and mutation tests.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _key(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return f"{type(value).__name__}:{value}"


def _normalized(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        if column not in frame.columns:
            raise ValueError(f"missing provenance relation column: {column}")
        output[column] = frame[column].map(_key)
    return output.reset_index(drop=True)


def _provenance_sha256(normalized: pd.DataFrame) -> str:
    row_hash = pd.util.hash_pandas_object(
        normalized,
        index=True,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("\x1f".join(normalized.columns.astype(str)).encode("utf-8"))
    digest.update(row_hash.tobytes())
    return digest.hexdigest()


def _hard_components(
    normalized: pd.DataFrame,
    hard_relations: Iterable[str],
) -> tuple[list[list[int]], str]:
    parent = list(range(len(normalized)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a = find(left)
        b = find(right)
        if a != b:
            parent[b] = a

    for relation in hard_relations:
        groups = normalized.groupby(relation, dropna=True, sort=False).indices
        for indices in groups.values():
            indices = [int(index) for index in indices]
            for index in indices[1:]:
                union(indices[0], index)
    grouped: dict[int, list[int]] = {}
    for index in range(len(normalized)):
        grouped.setdefault(find(index), []).append(index)
    members = sorted(
        (sorted(values) for values in grouped.values()),
        key=lambda values: tuple(values),
    )
    canonical = ";".join(
        ",".join(str(index) for index in member)
        for member in members
    )
    membership_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return members, membership_hash


def verify_infeasibility_certificate(
    provenance: pd.DataFrame,
    contract: dict[str, Any],
    certificate: dict[str, Any],
) -> dict[str, Any]:
    """Recompute analytic facts without importing production code."""

    reasons: list[str] = []
    if certificate.get("schema_version") != "infeasibility-certificate-v1":
        reasons.append("schema_version")
    if certificate.get("verifier") != "analytic-certificate-verifier-v1":
        reasons.append("verifier")
    relation_names = [item["name"] for item in contract["relations"]]
    normalized = _normalized(provenance, relation_names)
    expected_contract_hash = _canonical_sha256(contract)
    expected_provenance_hash = _provenance_sha256(normalized)
    if certificate.get("contract_sha256") != expected_contract_hash:
        reasons.append("contract_sha256")
    if certificate.get("provenance_sha256") != expected_provenance_hash:
        reasons.append("provenance_sha256")
    unsigned = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    if certificate.get("certificate_sha256") != _canonical_sha256(unsigned):
        reasons.append("certificate_sha256")

    n = len(normalized)
    k = int(contract["n_splits"])
    hard_relations = [
        item["name"]
        for item in contract["relations"]
        if item["mode"] == "hard_unseen"
    ]
    members, membership_hash = _hard_components(normalized, hard_relations)
    sizes = [len(member) for member in members]
    min_fold_n = int(math.ceil(float(contract["min_fold_fraction"]) * n - 1e-12))
    max_fold_n = int(math.floor(float(contract["max_fold_fraction"]) * n + 1e-12))
    facts = certificate.get("facts", {})
    common_expected = {
        "n_records": n,
        "n_splits": k,
        "component_membership_sha256": membership_hash,
    }
    for key, value in common_expected.items():
        if facts.get(key) != value:
            reasons.append(f"facts.{key}")
    kind = certificate.get("kind")
    if kind == "hard_component_count_below_fold_count":
        if facts.get("n_components") != len(members):
            reasons.append("facts.n_components")
        if not len(members) < k:
            reasons.append("predicate")
    elif kind == "largest_hard_component_exceeds_fold_upper_bound":
        largest = max(sizes, default=0)
        if facts.get("largest_component_n") != largest:
            reasons.append("facts.largest_component_n")
        if facts.get("max_fold_n") != max_fold_n:
            reasons.append("facts.max_fold_n")
        if not largest > max_fold_n:
            reasons.append("predicate")
    elif kind == "global_integer_fold_bounds_infeasible":
        if facts.get("min_fold_n") != min_fold_n:
            reasons.append("facts.min_fold_n")
        if facts.get("max_fold_n") != max_fold_n:
            reasons.append("facts.max_fold_n")
        if not (k * min_fold_n > n or k * max_fold_n < n):
            reasons.append("predicate")
    else:
        reasons.append("kind")
    return {
        "valid": not reasons,
        "reasons": sorted(set(reasons)),
        "recomputed": {
            "n_records": n,
            "n_splits": k,
            "n_components": len(members),
            "largest_component_n": max(sizes, default=0),
            "min_fold_n": min_fold_n,
            "max_fold_n": max_fold_n,
            "component_membership_sha256": membership_hash,
        },
    }


def _relation_novelty_flags(
    values: pd.Series,
    assignments: np.ndarray,
    fold: int,
) -> np.ndarray:
    test = assignments == fold
    observed = test & values.notna().to_numpy()
    train_values = set(values[(assignments >= 0) & (assignments != fold)].dropna())
    flags = np.full(len(values), -1, dtype=int)
    flags[observed] = (~values[observed].isin(train_values)).astype(int)
    return flags


def _validate_assignments(
    assignments: np.ndarray,
    n_records: int,
    n_splits: int,
) -> tuple[np.ndarray | None, str | None]:
    raw = np.asarray(assignments)
    if raw.ndim != 1 or len(raw) != n_records:
        return None, "assignment_shape"
    if not np.issubdtype(raw.dtype, np.integer):
        return None, "assignment_dtype"
    values = raw.astype(int, copy=False)
    if np.any((values < 0) | (values >= n_splits)):
        return None, "assignment_label_range"
    if len(np.unique(values)) != n_splits:
        return None, "empty_fold"
    return values, None


def verify_assignment(
    provenance: pd.DataFrame,
    assignments: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the complete contract using an independent implementation."""

    relation_specs = list(contract["relations"])
    relation_names = [item["name"] for item in relation_specs]
    n_splits = int(contract["n_splits"])
    normalized = _normalized(provenance, relation_names)
    vector, invalid_reason = _validate_assignments(
        assignments,
        len(normalized),
        n_splits,
    )
    if vector is None:
        return {
            "assignment_valid": False,
            "invalid_reason": invalid_reason,
            "contract_satisfied": False,
        }

    counts = np.bincount(vector, minlength=n_splits)
    fractions = counts.astype(float) / float(len(vector))
    epsilon = 1e-12
    fold_bounds_satisfied = bool(
        fractions.min() >= float(contract["min_fold_fraction"]) - epsilon
        and fractions.max() <= float(contract["max_fold_fraction"]) + epsilon
    )

    hard_details: dict[str, Any] = {}
    hard_constraints_satisfied = True
    for spec in relation_specs:
        if spec["mode"] != "hard_unseen":
            continue
        relation = spec["name"]
        table = pd.DataFrame(
            {
                "entity": normalized[relation],
                "fold": vector,
            }
        ).dropna(subset=["entity"])
        spanning = (
            table.groupby("entity", sort=False)["fold"].nunique() > 1
        )
        n_spanning = int(spanning.sum())
        hard_details[relation] = {"n_spanning_entities": n_spanning}
        hard_constraints_satisfied &= n_spanning == 0

    novelty_flags: dict[str, dict[int, np.ndarray]] = {
        relation: {
            fold: _relation_novelty_flags(normalized[relation], vector, fold)
            for fold in range(n_splits)
        }
        for relation in relation_names
    }
    relation_details: dict[str, Any] = {}
    relation_targets_satisfied = True
    for spec in relation_specs:
        if spec["mode"] == "ignore" or float(spec["weight"]) <= 0:
            continue
        relation = spec["name"]
        values: list[float | None] = []
        observed_counts: list[int] = []
        minimum = int(spec.get("min_observations_per_fold", 1))
        passed = True
        for fold in range(n_splits):
            flags = novelty_flags[relation][fold]
            observed = flags >= 0
            n_observed = int(observed.sum())
            observed_counts.append(n_observed)
            novelty = None if not np.any(observed) else float(flags[observed].mean())
            values.append(novelty)
            passed &= bool(
                novelty is not None
                and n_observed >= minimum
                and abs(float(novelty) - float(spec["target"]))
                <= float(spec["tolerance"]) + epsilon
            )
        relation_details[relation] = {
            "fold_novelty": values,
            "n_observed_by_fold": observed_counts,
            "minimum_required": minimum,
            "satisfied": bool(passed),
        }
        relation_targets_satisfied &= bool(passed)

    joint_details: dict[str, Any] = {}
    joint_patterns_satisfied = True
    for target in contract.get("joint_patterns", []):
        relations = list(target["relations"])
        target_probabilities = {
            str(pattern): float(probability)
            for pattern, probability in target["probabilities"].items()
        }
        patterns = [
            "".join(bits)
            for bits in itertools.product("01", repeat=len(relations))
        ]
        minimum = int(target.get("min_observations_per_fold", 1))
        fold_rows: list[dict[str, Any]] = []
        passed = True
        for fold in range(n_splits):
            flags = [novelty_flags[name][fold] for name in relations]
            valid = vector == fold
            for relation_flags in flags:
                valid &= relation_flags >= 0
            indices = np.flatnonzero(valid)
            n_observed = int(len(indices))
            if n_observed < minimum:
                total_variation = None
                fold_passed = False
            else:
                observed_patterns = [
                    "".join(
                        str(int(relation_flags[index]))
                        for relation_flags in flags
                    )
                    for index in indices
                ]
                observed = {
                    pattern: observed_patterns.count(pattern) / n_observed
                    for pattern in patterns
                }
                total_variation = float(
                    0.5
                    * sum(
                        abs(
                            observed[pattern]
                            - target_probabilities.get(pattern, 0.0)
                        )
                        for pattern in patterns
                    )
                )
                fold_passed = bool(
                    total_variation <= float(target["tolerance"]) + epsilon
                )
            fold_rows.append(
                {
                    "n_observed": n_observed,
                    "total_variation": total_variation,
                    "satisfied": fold_passed,
                }
            )
            passed &= fold_passed
        joint_details[target["name"]] = {
            "folds": fold_rows,
            "satisfied": bool(passed),
        }
        joint_patterns_satisfied &= bool(passed)

    incidence_details: dict[str, Any] = {}
    incidence_targets_satisfied = True
    for target in contract.get("incidence_targets", []):
        left = target["left_relation"]
        right = target["right_relation"]
        minimum = int(target.get("min_left_entities_per_fold", 1))
        fold_rows: list[dict[str, Any]] = []
        passed = True
        for fold in range(n_splits):
            valid = (
                (vector == fold)
                & normalized[left].notna().to_numpy()
                & normalized[right].notna().to_numpy()
            )
            condition_relation = target.get("condition_relation")
            if condition_relation is not None:
                valid &= (
                    novelty_flags[condition_relation][fold]
                    == int(target["condition_novelty"])
                )
            pairs = pd.DataFrame(
                {
                    "left": normalized.loc[valid, left],
                    "right": normalized.loc[valid, right],
                }
            )
            if pairs.empty:
                n_left = 0
                mean_degree = None
                absolute_error = None
                fold_passed = False
            else:
                degree = pairs.groupby("left", sort=False)["right"].nunique()
                n_left = int(len(degree))
                mean_degree = float(degree.mean())
                absolute_error = float(
                    abs(mean_degree - float(target["target_mean_degree"]))
                )
                fold_passed = bool(
                    n_left >= minimum
                    and absolute_error <= float(target["tolerance"]) + epsilon
                )
            fold_rows.append(
                {
                    "n_left_entities": n_left,
                    "mean_degree": mean_degree,
                    "absolute_error": absolute_error,
                    "satisfied": fold_passed,
                }
            )
            passed &= fold_passed
        incidence_details[target["name"]] = {
            "folds": fold_rows,
            "satisfied": bool(passed),
        }
        incidence_targets_satisfied &= bool(passed)

    contract_satisfied = bool(
        hard_constraints_satisfied
        and fold_bounds_satisfied
        and relation_targets_satisfied
        and joint_patterns_satisfied
        and incidence_targets_satisfied
    )
    return {
        "assignment_valid": True,
        "invalid_reason": None,
        "fold_counts": counts.astype(int).tolist(),
        "fold_fractions": fractions.astype(float).tolist(),
        "fold_bounds_satisfied": fold_bounds_satisfied,
        "hard_constraints_satisfied": bool(hard_constraints_satisfied),
        "relation_targets_satisfied": bool(relation_targets_satisfied),
        "joint_patterns_satisfied": bool(joint_patterns_satisfied),
        "incidence_targets_satisfied": bool(incidence_targets_satisfied),
        "contract_satisfied": contract_satisfied,
        "hard_relations": hard_details,
        "relations": relation_details,
        "joint_patterns": joint_details,
        "incidence_targets": incidence_details,
    }


def brute_force_feasible(
    provenance: pd.DataFrame,
    contract: dict[str, Any],
    *,
    max_records: int = 10,
) -> dict[str, Any]:
    """Enumerate small assignments to obtain an implementation-independent truth."""

    n_records = len(provenance)
    n_splits = int(contract["n_splits"])
    if n_records > max_records:
        raise ValueError("brute-force oracle is restricted to small instances")
    attempted = 0
    # Fix the first label to zero to remove fold-label permutation symmetry.
    for tail in itertools.product(range(n_splits), repeat=max(n_records - 1, 0)):
        assignment = np.asarray((0, *tail), dtype=int)
        if len(np.unique(assignment)) != n_splits:
            continue
        attempted += 1
        audit = verify_assignment(provenance, assignment, contract)
        if audit["contract_satisfied"]:
            return {
                "feasible": True,
                "assignment": assignment.tolist(),
                "attempted_canonical_assignments": attempted,
            }
    return {
        "feasible": False,
        "assignment": None,
        "attempted_canonical_assignments": attempted,
    }
