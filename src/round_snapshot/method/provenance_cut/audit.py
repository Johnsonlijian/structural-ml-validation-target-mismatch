"""Audit metrics for provenance novelty and fold integrity."""

from __future__ import annotations

import hashlib
import itertools
from typing import Any

import numpy as np
import pandas as pd

from .claim import DeploymentClaim


def _entity_key(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return f"{type(value).__name__}:{value}"


def normalize_provenance(provenance: pd.DataFrame) -> pd.DataFrame:
    """Return type-stable entity keys while preserving missing values."""

    out = pd.DataFrame(index=provenance.index)
    for column in provenance.columns:
        out[column] = provenance[column].map(_entity_key)
    return out


def relation_novelty(
    provenance: pd.DataFrame,
    assignments: np.ndarray,
    n_splits: int,
) -> dict[str, dict[str, float | int | None]]:
    """Compute record-weighted novelty for every relation and validation fold."""

    normalized = normalize_provenance(provenance)
    result: dict[str, dict[str, float | int | None]] = {}
    for relation in normalized.columns:
        relation_result: dict[str, float | int | None] = {}
        values = normalized[relation]
        for fold in range(n_splits):
            test_mask = assignments == fold
            train_mask = (assignments >= 0) & (assignments != fold)
            test_values = values[test_mask].dropna()
            train_values = set(values[train_mask].dropna().tolist())
            if test_values.empty:
                novelty = None
                observed = 0
                unseen = 0
            else:
                flags = ~test_values.isin(train_values)
                novelty = float(flags.mean())
                observed = int(test_values.shape[0])
                unseen = int(flags.sum())
            relation_result[f"fold_{fold}"] = novelty
            relation_result[f"fold_{fold}_observed"] = observed
            relation_result[f"fold_{fold}_unseen"] = unseen
        valid = [
            relation_result[f"fold_{fold}"]
            for fold in range(n_splits)
            if relation_result[f"fold_{fold}"] is not None
        ]
        relation_result["mean"] = None if not valid else float(np.mean(valid))
        relation_result["missing_fraction"] = float(values.isna().mean())
        result[relation] = relation_result
    return result


def relation_entity_leakage(
    provenance: pd.DataFrame,
    assignments: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Report entity values that occur in more than one fold."""

    normalized = normalize_provenance(provenance)
    result: dict[str, dict[str, float | int]] = {}
    for relation in normalized.columns:
        frame = pd.DataFrame({"entity": normalized[relation], "fold": assignments})
        frame = frame[(frame["entity"].notna()) & (frame["fold"] >= 0)]
        if frame.empty:
            result[relation] = {
                "n_entities": 0,
                "n_spanning_entities": 0,
                "spanning_entity_fraction": 0.0,
                "record_fraction_in_spanning_entities": 0.0,
            }
            continue
        spans = frame.groupby("entity", sort=False)["fold"].nunique()
        spanning = set(spans[spans > 1].index)
        result[relation] = {
            "n_entities": int(spans.shape[0]),
            "n_spanning_entities": int(len(spanning)),
            "spanning_entity_fraction": float(len(spanning) / spans.shape[0]),
            "record_fraction_in_spanning_entities": float(frame["entity"].isin(spanning).mean()),
        }
    return result


def provenance_fingerprint(provenance: pd.DataFrame) -> str:
    """Stable fingerprint of relation names, order, values and missingness."""

    normalized = normalize_provenance(provenance)
    row_hash = pd.util.hash_pandas_object(normalized, index=True).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update("\x1f".join(normalized.columns.astype(str)).encode("utf-8"))
    digest.update(row_hash.tobytes())
    return digest.hexdigest()


def assignment_checksum(assignments: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(assignments, dtype=np.int64).tobytes()).hexdigest()


def _row_novelty_flags(
    normalized: pd.DataFrame,
    assignments: np.ndarray,
    relation: str,
    fold: int,
) -> np.ndarray:
    values = normalized[relation]
    train_values = set(values[(assignments >= 0) & (assignments != fold)].dropna())
    flags = np.full(len(values), -1, dtype=int)
    test = assignments == fold
    observed = test & values.notna().to_numpy()
    flags[observed] = (~values[observed].isin(train_values)).astype(int)
    return flags


def joint_pattern_audit(
    provenance: pd.DataFrame,
    assignments: np.ndarray,
    claim: DeploymentClaim,
) -> dict[str, Any]:
    normalized = normalize_provenance(provenance)
    output: dict[str, Any] = {}
    for target in claim.joint_patterns:
        target_map = target.probability_map
        all_patterns = [
            "".join(bits)
            for bits in itertools.product("01", repeat=len(target.relations))
        ]
        folds: dict[str, Any] = {}
        max_total_variation = 0.0
        n_evaluable_folds = 0
        for fold in range(claim.n_splits):
            flags = [
                _row_novelty_flags(normalized, assignments, relation, fold)
                for relation in target.relations
            ]
            test = assignments == fold
            valid = test.copy()
            for relation_flags in flags:
                valid &= relation_flags >= 0
            if not np.any(valid):
                observed = {pattern: None for pattern in all_patterns}
                total_variation = None
                n_observed = 0
            else:
                pattern_values = np.asarray(
                    [
                        "".join(str(int(relation_flags[index])) for relation_flags in flags)
                        for index in np.flatnonzero(valid)
                    ]
                )
                observed = {
                    pattern: float(np.mean(pattern_values == pattern))
                    for pattern in all_patterns
                }
                total_variation = float(
                    0.5
                    * sum(
                        abs(observed[pattern] - target_map.get(pattern, 0.0))
                        for pattern in all_patterns
                    )
                )
                max_total_variation = max(max_total_variation, total_variation)
                n_observed = int(valid.sum())
            evaluable = bool(n_observed >= target.min_observations_per_fold)
            n_evaluable_folds += int(evaluable)
            folds[f"fold_{fold}"] = {
                "observed_probabilities": observed,
                "target_probabilities": {
                    pattern: float(target_map.get(pattern, 0.0))
                    for pattern in all_patterns
                },
                "total_variation": total_variation,
                "n_observed": n_observed,
                "minimum_required": int(target.min_observations_per_fold),
                "evaluable": evaluable,
            }
        all_folds_evaluable = bool(n_evaluable_folds == claim.n_splits)
        output[target.name] = {
            "relations": list(target.relations),
            "folds": folds,
            "max_total_variation": float(max_total_variation),
            "tolerance": float(target.tolerance),
            "n_evaluable_folds": int(n_evaluable_folds),
            "all_folds_evaluable": all_folds_evaluable,
            "satisfied": bool(
                all_folds_evaluable
                and max_total_variation <= target.tolerance
            ),
        }
    return output


def incidence_target_audit(
    provenance: pd.DataFrame,
    assignments: np.ndarray,
    claim: DeploymentClaim,
) -> dict[str, Any]:
    normalized = normalize_provenance(provenance)
    output: dict[str, Any] = {}
    for target in claim.incidence_targets:
        folds: dict[str, Any] = {}
        max_error = 0.0
        n_evaluable_folds = 0
        for fold in range(claim.n_splits):
            test = assignments == fold
            valid = (
                test
                & normalized[target.left_relation].notna().to_numpy()
                & normalized[target.right_relation].notna().to_numpy()
            )
            if target.condition_relation is not None:
                condition = _row_novelty_flags(
                    normalized,
                    assignments,
                    target.condition_relation,
                    fold,
                )
                valid &= condition == int(target.condition_novelty)
            frame = pd.DataFrame(
                {
                    "left": normalized.loc[valid, target.left_relation],
                    "right": normalized.loc[valid, target.right_relation],
                }
            )
            if frame.empty:
                mean_degree = None
                error = None
                n_left = 0
            else:
                degrees = frame.groupby("left", sort=False)["right"].nunique()
                mean_degree = float(degrees.mean())
                error = float(abs(mean_degree - target.target_mean_degree))
                max_error = max(max_error, error)
                n_left = int(degrees.shape[0])
            evaluable = bool(n_left >= target.min_left_entities_per_fold)
            n_evaluable_folds += int(evaluable)
            folds[f"fold_{fold}"] = {
                "mean_degree": mean_degree,
                "target_mean_degree": float(target.target_mean_degree),
                "absolute_error": error,
                "n_left_entities": n_left,
                "minimum_required": int(target.min_left_entities_per_fold),
                "evaluable": evaluable,
            }
        all_folds_evaluable = bool(n_evaluable_folds == claim.n_splits)
        output[target.name] = {
            "left_relation": target.left_relation,
            "right_relation": target.right_relation,
            "condition_relation": target.condition_relation,
            "condition_novelty": target.condition_novelty,
            "folds": folds,
            "max_absolute_error": float(max_error),
            "tolerance": float(target.tolerance),
            "n_evaluable_folds": int(n_evaluable_folds),
            "all_folds_evaluable": all_folds_evaluable,
            "satisfied": bool(
                all_folds_evaluable and max_error <= target.tolerance
            ),
        }
    return output


def audit_assignment(
    provenance: pd.DataFrame,
    assignments: np.ndarray,
    claim: DeploymentClaim,
) -> dict[str, Any]:
    assignments = np.asarray(assignments, dtype=int)
    counts = np.bincount(assignments[assignments >= 0], minlength=claim.n_splits)
    n = int((assignments >= 0).sum())
    fractions = counts / max(n, 1)
    novelty = relation_novelty(provenance, assignments, claim.n_splits)
    leakage = relation_entity_leakage(provenance, assignments)
    hard_ok = all(leakage[name]["n_spanning_entities"] == 0 for name in claim.hard_relations)
    realized_error: dict[str, float | None] = {}
    realized_max_abs_error: dict[str, float | None] = {}
    relation_target_support: dict[str, Any] = {}
    normalized = normalize_provenance(provenance)
    for target in claim.optimized_relations:
        fold_observed = [
            int(
                (
                    (assignments == fold)
                    & normalized[target.name].notna().to_numpy()
                ).sum()
            )
            for fold in range(claim.n_splits)
        ]
        all_folds_evaluable = bool(
            all(
                count >= target.min_observations_per_fold
                for count in fold_observed
            )
        )
        relation_target_support[target.name] = {
            "n_observed_by_fold": fold_observed,
            "minimum_required": int(target.min_observations_per_fold),
            "all_folds_evaluable": all_folds_evaluable,
        }
        values = [
            novelty[target.name][f"fold_{fold}"]
            for fold in range(claim.n_splits)
            if novelty[target.name][f"fold_{fold}"] is not None
        ]
        realized_error[target.name] = (
            None
            if not values
            else float(np.mean([(float(value) - target.target) ** 2 for value in values]))
        )
        realized_max_abs_error[target.name] = (
            None
            if not values
            else float(max(abs(float(value) - target.target) for value in values))
        )
    joint_patterns = joint_pattern_audit(provenance, assignments, claim)
    incidence_targets = incidence_target_audit(provenance, assignments, claim)
    epsilon = 1e-12
    fold_bounds_satisfied = bool(
        fractions.size
        and fractions.min() >= claim.min_fold_fraction - epsilon
        and fractions.max() <= claim.max_fold_fraction + epsilon
    )
    relation_targets_satisfied = bool(
        all(
            realized_max_abs_error[target.name] is not None
            and relation_target_support[target.name]["all_folds_evaluable"]
            and float(realized_max_abs_error[target.name])
            <= float(target.tolerance) + epsilon
            for target in claim.optimized_relations
        )
    )
    joint_patterns_satisfied = bool(
        all(item["satisfied"] for item in joint_patterns.values())
    )
    incidence_targets_satisfied = bool(
        all(item["satisfied"] for item in incidence_targets.values())
    )
    contract_satisfied = bool(
        hard_ok
        and fold_bounds_satisfied
        and relation_targets_satisfied
        and joint_patterns_satisfied
        and incidence_targets_satisfied
    )
    return {
        "n_records": n,
        "fold_counts": counts.astype(int).tolist(),
        "fold_fractions": fractions.astype(float).tolist(),
        "min_fold_fraction_observed": float(fractions.min()) if fractions.size else 0.0,
        "max_fold_fraction_observed": float(fractions.max()) if fractions.size else 0.0,
        "hard_constraints_satisfied": bool(hard_ok),
        "fold_bounds_satisfied": fold_bounds_satisfied,
        "relation_targets_satisfied": relation_targets_satisfied,
        "joint_patterns_satisfied": joint_patterns_satisfied,
        "incidence_targets_satisfied": incidence_targets_satisfied,
        "contract_satisfied": contract_satisfied,
        "novelty": novelty,
        "relation_entity_leakage": leakage,
        "relation_target_mse": realized_error,
        "relation_target_max_abs_error": realized_max_abs_error,
        "relation_target_support": relation_target_support,
        "joint_patterns": joint_patterns,
        "incidence_targets": incidence_targets,
        "provenance_fingerprint": provenance_fingerprint(provenance),
        "assignment_checksum": assignment_checksum(assignments),
    }
