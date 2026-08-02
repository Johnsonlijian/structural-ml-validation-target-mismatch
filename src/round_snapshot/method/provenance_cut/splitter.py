"""Deterministic claim-conditioned provenance hypergraph partitioning."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .audit import audit_assignment, normalize_provenance
from .claim import CutResult, DeploymentClaim


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=int)

    def find(self, value: int) -> int:
        parent = self.parent
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


@dataclass(frozen=True)
class _Components:
    members: tuple[np.ndarray, ...]
    record_to_component: np.ndarray

    @property
    def sizes(self) -> np.ndarray:
        return np.asarray([len(item) for item in self.members], dtype=int)


@dataclass(frozen=True)
class _RelationCodes:
    codes: np.ndarray
    n_entities: int


@dataclass(frozen=True)
class _ObjectiveCache:
    relations: dict[str, _RelationCodes]


class ClaimConditionedProvenanceCut:
    """Scikit-learn-compatible splitter for a declared provenance signature."""

    def __init__(self, claim: DeploymentClaim) -> None:
        self.claim = claim
        self.result_: CutResult | None = None

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.claim.n_splits

    def split(self, X, y=None, groups=None) -> Iterable[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("groups must be a pandas DataFrame of provenance relations")
        result = self.assign(groups).require_feasible()
        all_indices = np.arange(len(result.assignments), dtype=int)
        for fold in range(self.claim.n_splits):
            test = all_indices[result.assignments == fold]
            train = all_indices[result.assignments != fold]
            yield train, test

    def assign(self, provenance: pd.DataFrame) -> CutResult:
        provenance = self._validate_provenance(provenance)
        if self.claim.ordered_relation is not None:
            result = CutResult(
                assignments=np.full(len(provenance), -1, dtype=int),
                feasible=False,
                objective=None,
                status="ordered_route_required",
                feasibility_status="route_required",
                candidate_hard_satisfied=None,
                candidate_balance_satisfied=None,
                candidate_targets_satisfied=None,
                diagnostics={
                    "reason": (
                        f"relation {self.claim.ordered_relation!r} is ordered; "
                        "use a predeclared forward/chronological outer split"
                    )
                },
            )
            self.result_ = result
            return result
        components = self._build_components(provenance)
        objective_cache = self._build_objective_cache(provenance)
        sizes = components.sizes
        n = len(provenance)
        n_components = len(components.members)
        diagnostics = {
            "n_records": int(n),
            "n_components": int(n_components),
            "component_sizes_desc": sorted(sizes.astype(int).tolist(), reverse=True),
            "largest_component_fraction": float(sizes.max() / n),
            "hard_relations": list(self.claim.hard_relations),
            "component_membership_sha256": self._component_membership_sha256(
                components
            ),
        }
        if n_components < self.claim.n_splits:
            result = CutResult(
                assignments=np.full(n, -1, dtype=int),
                feasible=False,
                objective=None,
                status="certified_partition_infeasible",
                feasibility_status="certified_infeasible",
                candidate_hard_satisfied=None,
                candidate_balance_satisfied=None,
                candidate_targets_satisfied=None,
                diagnostics={
                    **diagnostics,
                    "certificate": {
                        "kind": "hard_component_count_below_fold_count",
                        "n_components": int(n_components),
                        "n_splits": int(self.claim.n_splits),
                        "n_records": int(n),
                        "component_membership_sha256": diagnostics[
                            "component_membership_sha256"
                        ],
                    },
                    "reason": (
                        f"hard provenance graph has {n_components} connected components "
                        f"for {self.claim.n_splits} requested folds"
                    ),
                },
            )
            self.result_ = result
            return result

        largest_component_fraction = float(sizes.max() / n)
        min_fold_n = int(math.ceil(self.claim.min_fold_fraction * n - 1e-12))
        max_fold_n = int(math.floor(self.claim.max_fold_fraction * n + 1e-12))
        if (
            self.claim.n_splits * min_fold_n > n
            or self.claim.n_splits * max_fold_n < n
        ):
            result = CutResult(
                assignments=np.full(n, -1, dtype=int),
                feasible=False,
                objective=None,
                status="certified_balance_infeasible",
                feasibility_status="certified_infeasible",
                candidate_hard_satisfied=None,
                candidate_balance_satisfied=None,
                candidate_targets_satisfied=None,
                diagnostics={
                    **diagnostics,
                    "certificate": {
                        "kind": "global_integer_fold_bounds_infeasible",
                        "n_records": int(n),
                        "n_splits": int(self.claim.n_splits),
                        "min_fold_n": min_fold_n,
                        "max_fold_n": max_fold_n,
                        "component_membership_sha256": diagnostics[
                            "component_membership_sha256"
                        ],
                    },
                    "reason": "integer fold bounds cannot cover all records",
                },
            )
            self.result_ = result
            return result

        largest_component_n = int(sizes.max())
        if largest_component_n > max_fold_n:
            result = CutResult(
                assignments=np.full(n, -1, dtype=int),
                feasible=False,
                objective=None,
                status="certified_balance_infeasible",
                feasibility_status="certified_infeasible",
                candidate_hard_satisfied=None,
                candidate_balance_satisfied=None,
                candidate_targets_satisfied=None,
                diagnostics={
                    **diagnostics,
                    "certificate": {
                        "kind": "largest_hard_component_exceeds_fold_upper_bound",
                        "n_records": int(n),
                        "n_splits": int(self.claim.n_splits),
                        "largest_component_n": largest_component_n,
                        "max_fold_n": max_fold_n,
                        "largest_component_fraction": largest_component_fraction,
                        "max_fold_fraction": float(
                            self.claim.max_fold_fraction
                        ),
                        "component_membership_sha256": diagnostics[
                            "component_membership_sha256"
                        ],
                    },
                    "reason": (
                        "largest hard component exceeds the declared maximum "
                        "fold fraction"
                    ),
                },
            )
            self.result_ = result
            return result

        rng = np.random.default_rng(self.claim.random_state)
        best_assignment: np.ndarray | None = None
        best_score = np.inf
        for start in range(self.claim.n_starts):
            start_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            start_rng = np.random.default_rng(start_seed)
            component_folds = self._initial_assignment(sizes, start_rng)
            component_folds, score = self._local_search(
                objective_cache,
                components,
                component_folds,
                start_rng,
            )
            if score < best_score - 1e-12:
                best_score = score
                best_assignment = component_folds.copy()

        if best_assignment is None:
            raise RuntimeError("no split assignment was generated")
        assignments = best_assignment[components.record_to_component]
        audit = audit_assignment(provenance, assignments, self.claim)
        balance_ok = (
            audit["min_fold_fraction_observed"] >= self.claim.min_fold_fraction
            and audit["max_fold_fraction_observed"] <= self.claim.max_fold_fraction
        )
        diagnostics["balance_bounds_satisfied"] = bool(balance_ok)
        diagnostics["candidate_hard_satisfied_but_balance_invalid"] = bool(
            not balance_ok
        )
        relation_target_ok = True
        relation_residuals: dict[str, float | None] = {}
        for target in self.claim.optimized_relations:
            residual = audit["relation_target_max_abs_error"][target.name]
            relation_residuals[target.name] = residual
            if residual is None or float(residual) > target.tolerance:
                relation_target_ok = False
        joint_target_ok = all(
            item["satisfied"] for item in audit["joint_patterns"].values()
        )
        incidence_target_ok = all(
            item["satisfied"] for item in audit["incidence_targets"].values()
        )
        soft_target_ok = bool(
            relation_target_ok and joint_target_ok and incidence_target_ok
        )
        hard_ok = bool(audit["hard_constraints_satisfied"])
        diagnostics["relation_target_residuals"] = relation_residuals
        diagnostics["relation_targets_satisfied"] = relation_target_ok
        diagnostics["joint_patterns_satisfied"] = joint_target_ok
        diagnostics["incidence_targets_satisfied"] = incidence_target_ok
        if not hard_ok:
            status = "realized_plan_invalid_hard"
            diagnostics["candidate_status"] = "hard_violation"
        elif not balance_ok:
            status = "backend_search_exhausted"
            diagnostics["candidate_status"] = "realized_plan_invalid_balance"
        elif not soft_target_ok:
            status = "backend_search_exhausted"
            diagnostics["candidate_status"] = "realized_plan_invalid_targets"
        else:
            status = "feasible"
        if status != "feasible":
            diagnostics["reason"] = diagnostics.get("candidate_status", status)
        result = CutResult(
            assignments=assignments,
            feasible=bool(hard_ok and balance_ok and soft_target_ok),
            objective=float(best_score),
            status=status,
            feasibility_status=(
                "admissible"
                if status == "feasible"
                else (
                    "unresolved"
                    if status == "backend_search_exhausted"
                    else "invalid_candidate"
                )
            ),
            candidate_hard_satisfied=hard_ok,
            candidate_balance_satisfied=bool(balance_ok),
            candidate_targets_satisfied=soft_target_ok,
            audit=audit,
            diagnostics=diagnostics,
        )
        self.result_ = result
        return result

    def _validate_provenance(self, provenance: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(provenance, pd.DataFrame):
            raise TypeError("provenance must be a pandas DataFrame")
        if provenance.empty:
            raise ValueError("provenance must contain records")
        missing = [name for name in self.claim.relation_names if name not in provenance.columns]
        if missing:
            raise ValueError(f"missing provenance relation columns: {missing}")
        if len(provenance) < self.claim.n_splits:
            raise ValueError("number of records must be at least n_splits")
        return provenance.loc[:, list(self.claim.relation_names)].reset_index(drop=True)

    def _build_components(self, provenance: pd.DataFrame) -> _Components:
        n = len(provenance)
        union_find = _UnionFind(n)
        normalized = normalize_provenance(provenance)
        for relation in self.claim.hard_relations:
            groups = normalized.groupby(relation, dropna=True, sort=False).indices
            for indices in groups.values():
                indices = np.asarray(indices, dtype=int)
                if len(indices) < 2:
                    continue
                anchor = int(indices[0])
                for other in indices[1:]:
                    union_find.union(anchor, int(other))
        roots = np.asarray([union_find.find(i) for i in range(n)], dtype=int)
        unique_roots, record_to_component = np.unique(roots, return_inverse=True)
        members = tuple(
            np.flatnonzero(record_to_component == component)
            for component in range(len(unique_roots))
        )
        return _Components(members=members, record_to_component=record_to_component)

    @staticmethod
    def _component_membership_sha256(components: _Components) -> str:
        canonical = ";".join(
            ",".join(str(int(index)) for index in sorted(member.tolist()))
            for member in sorted(
                components.members,
                key=lambda values: tuple(sorted(values.tolist())),
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _build_objective_cache(self, provenance: pd.DataFrame) -> _ObjectiveCache:
        normalized = normalize_provenance(provenance)
        relations: dict[str, _RelationCodes] = {}
        for relation in self.claim.relation_names:
            codes, uniques = pd.factorize(normalized[relation], sort=False, use_na_sentinel=True)
            relations[relation] = _RelationCodes(
                codes=np.asarray(codes, dtype=int),
                n_entities=int(len(uniques)),
            )
        return _ObjectiveCache(relations=relations)

    def _initial_assignment(
        self,
        sizes: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        k = self.claim.n_splits
        jitter = rng.uniform(0.0, 0.01, size=len(sizes))
        order = np.lexsort((jitter, -sizes))
        assignment = np.full(len(sizes), -1, dtype=int)
        counts = np.zeros(k, dtype=int)
        first_folds = rng.permutation(k)
        for position, component in enumerate(order):
            if position < k:
                fold = int(first_folds[position])
            else:
                minimum = counts.min()
                candidates = np.flatnonzero(counts == minimum)
                fold = int(rng.choice(candidates))
            assignment[component] = fold
            counts[fold] += int(sizes[component])
        return assignment

    def _local_search(
        self,
        objective_cache: _ObjectiveCache,
        components: _Components,
        assignment: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        score = self._objective(objective_cache, components, assignment)
        for _ in range(self.claim.max_iter):
            improved = False
            component_counts = np.bincount(assignment, minlength=self.claim.n_splits)
            for component in rng.permutation(len(assignment)):
                source = int(assignment[component])
                if component_counts[source] <= 1:
                    continue
                best_fold = source
                best_score = score
                for target in range(self.claim.n_splits):
                    if target == source:
                        continue
                    candidate = assignment.copy()
                    candidate[component] = target
                    candidate_score = self._objective(
                        objective_cache,
                        components,
                        candidate,
                    )
                    if candidate_score < best_score - 1e-12:
                        best_fold = target
                        best_score = candidate_score
                if best_fold != source:
                    assignment[component] = best_fold
                    component_counts[source] -= 1
                    component_counts[best_fold] += 1
                    score = best_score
                    improved = True
            if improved:
                continue

            pairs = self._candidate_swaps(len(assignment), rng)
            for left, right in pairs:
                if assignment[left] == assignment[right]:
                    continue
                candidate = assignment.copy()
                candidate[left], candidate[right] = candidate[right], candidate[left]
                candidate_score = self._objective(
                    objective_cache,
                    components,
                    candidate,
                )
                if candidate_score < score - 1e-12:
                    assignment = candidate
                    score = candidate_score
                    improved = True
                    break
            if not improved:
                break
        return assignment, float(score)

    @staticmethod
    def _candidate_swaps(
        n_components: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        if n_components <= 60:
            pairs = [
                (left, right)
                for left in range(n_components)
                for right in range(left + 1, n_components)
            ]
            rng.shuffle(pairs)
            return pairs
        pairs: set[tuple[int, int]] = set()
        while len(pairs) < min(800, n_components * 10):
            left, right = sorted(rng.choice(n_components, size=2, replace=False).tolist())
            pairs.add((int(left), int(right)))
        return list(pairs)

    def _objective(
        self,
        objective_cache: _ObjectiveCache,
        components: _Components,
        component_assignment: np.ndarray,
    ) -> float:
        assignments = component_assignment[components.record_to_component]
        counts = np.bincount(assignments, minlength=self.claim.n_splits).astype(float)
        n = float(len(assignments))
        if np.any(counts == 0):
            return 1e12
        ideal = n / self.claim.n_splits
        size_penalty = float(np.mean((counts / ideal - 1.0) ** 2))
        fractions = counts / n
        lower = np.maximum(self.claim.min_fold_fraction - fractions, 0.0)
        upper = np.maximum(fractions - self.claim.max_fold_fraction, 0.0)
        bound_penalty = float(np.sum(lower**2 + upper**2))

        claim_penalty = 0.0
        total_weight = 0.0
        state_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def relation_state(name: str) -> tuple[np.ndarray, np.ndarray]:
            if name in state_cache:
                return state_cache[name]
            relation_codes = objective_cache.relations[name]
            valid = relation_codes.codes >= 0
            novelty_flags = np.full(len(assignments), -1, dtype=int)
            if not np.any(valid):
                novelty_values = np.full(self.claim.n_splits, np.nan, dtype=float)
                state_cache[name] = (novelty_values, novelty_flags)
                return state_cache[name]
            flat_index = (
                relation_codes.codes[valid] * self.claim.n_splits
                + assignments[valid]
            )
            entity_fold_counts = np.bincount(
                flat_index,
                minlength=relation_codes.n_entities * self.claim.n_splits,
            ).reshape(relation_codes.n_entities, self.claim.n_splits)
            entity_fold_presence = entity_fold_counts > 0
            unique_to_one_fold = entity_fold_presence.sum(axis=1) == 1
            unseen_counts = entity_fold_counts[unique_to_one_fold].sum(axis=0)
            observed_counts = entity_fold_counts.sum(axis=0)
            novelty_values = np.divide(
                unseen_counts,
                observed_counts,
                out=np.full(self.claim.n_splits, np.nan, dtype=float),
                where=observed_counts > 0,
            )
            novelty_flags[valid] = unique_to_one_fold[relation_codes.codes[valid]].astype(int)
            state_cache[name] = (novelty_values, novelty_flags)
            return state_cache[name]

        for relation in self.claim.optimized_relations:
            novelty_values, _ = relation_state(relation.name)
            if np.all(np.isnan(novelty_values)):
                claim_penalty += relation.weight
                total_weight += relation.weight
                continue
            for fold in range(self.claim.n_splits):
                value = novelty_values[fold]
                if np.isnan(value):
                    claim_penalty += relation.weight
                else:
                    fold_weight = counts[fold] / n
                    claim_penalty += (
                        relation.weight
                        * fold_weight
                        * (value - relation.target) ** 2
                    )
                total_weight += relation.weight / self.claim.n_splits

        for target in self.claim.joint_patterns:
            states = [relation_state(name)[1] for name in target.relations]
            n_patterns = 2 ** len(states)
            target_vector = np.zeros(n_patterns, dtype=float)
            for pattern, probability in target.probabilities:
                target_vector[int(pattern, 2)] = float(probability)
            for fold in range(self.claim.n_splits):
                valid = assignments == fold
                for flags in states:
                    valid &= flags >= 0
                if not np.any(valid):
                    claim_penalty += target.weight
                else:
                    pattern_codes = np.zeros(int(valid.sum()), dtype=int)
                    for position, flags in enumerate(states):
                        shift = len(states) - position - 1
                        pattern_codes += flags[valid] << shift
                    observed = np.bincount(
                        pattern_codes,
                        minlength=n_patterns,
                    ).astype(float)
                    observed /= observed.sum()
                    fold_weight = counts[fold] / n
                    claim_penalty += (
                        target.weight
                        * fold_weight
                        * float(np.sum((observed - target_vector) ** 2))
                    )
                total_weight += target.weight / self.claim.n_splits

        for target in self.claim.incidence_targets:
            left = objective_cache.relations[target.left_relation]
            right = objective_cache.relations[target.right_relation]
            condition_flags = None
            if target.condition_relation is not None:
                condition_flags = relation_state(target.condition_relation)[1]
            for fold in range(self.claim.n_splits):
                valid = (
                    (assignments == fold)
                    & (left.codes >= 0)
                    & (right.codes >= 0)
                )
                if condition_flags is not None:
                    valid &= condition_flags == int(target.condition_novelty)
                if not np.any(valid):
                    claim_penalty += target.weight
                else:
                    pairs = np.column_stack((left.codes[valid], right.codes[valid]))
                    unique_pairs = np.unique(pairs, axis=0)
                    degrees = np.bincount(
                        unique_pairs[:, 0],
                        minlength=left.n_entities,
                    )
                    degrees = degrees[degrees > 0]
                    mean_degree = float(np.mean(degrees))
                    relative_error = (
                        mean_degree - target.target_mean_degree
                    ) / max(target.target_mean_degree, 1.0)
                    fold_weight = counts[fold] / n
                    claim_penalty += (
                        target.weight * fold_weight * relative_error**2
                    )
                total_weight += target.weight / self.claim.n_splits
        if total_weight > 0:
            claim_penalty /= total_weight
        return float(
            self.claim.claim_weight * claim_penalty
            + self.claim.size_weight * size_penalty
            + self.claim.bound_weight * bound_penalty
        )
