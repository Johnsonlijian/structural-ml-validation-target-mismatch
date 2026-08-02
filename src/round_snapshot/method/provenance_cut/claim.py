"""Typed deployment claims and split results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


VALID_MODES = {"hard_unseen", "target", "support_seen", "ignore"}
VALID_TARGET_WEIGHTINGS = {
    "record_weighted",
    "equal_domain",
    "custom",
}
VALID_RISK_AGGREGATIONS = {"mean", "worst_domain"}
VALID_METADATA_SOURCES = {
    "legacy_defaulted",
    "prespecified",
    "author_declared",
    "imported_verified",
}


@dataclass(frozen=True)
class DeploymentEstimand:
    """Prediction target and decision quantity attached to a split contract.

    These fields are descriptive metadata used for hashing, validation cards,
    and downstream endpoint selection.  They are deliberately outcome-free:
    no observed target values are accepted by the compiler.
    """

    prediction_unit: str = "record"
    outcome: str = "continuous_response"
    loss: str = "squared_error"
    target_population: str = "declared_deployment_population"
    target_weighting: str = "record_weighted"
    risk_aggregation: str = "mean"
    decision_estimand: str = "expected_predictive_loss"
    domain_relation: str | None = None
    loss_parameters: tuple[tuple[str, str], ...] = ()
    custom_weighting_spec: tuple[tuple[str, str], ...] = ()
    metadata_source: str = "legacy_defaulted"

    def __post_init__(self) -> None:
        for name in (
            "prediction_unit",
            "outcome",
            "loss",
            "target_population",
            "decision_estimand",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"estimand field {name} must be non-empty")
        if self.target_weighting not in VALID_TARGET_WEIGHTINGS:
            raise ValueError(
                f"unknown target weighting: {self.target_weighting}"
            )
        if self.risk_aggregation not in VALID_RISK_AGGREGATIONS:
            raise ValueError(
                f"unknown risk aggregation: {self.risk_aggregation}"
            )
        if (
            self.target_weighting == "equal_domain"
            or self.risk_aggregation == "worst_domain"
        ) and not self.domain_relation:
            raise ValueError(
                "domain-aware weighting or aggregation requires domain_relation"
            )
        if self.target_weighting == "custom" and not self.custom_weighting_spec:
            raise ValueError("custom target weighting requires custom_weighting_spec")
        if self.metadata_source not in VALID_METADATA_SOURCES:
            raise ValueError(f"unknown metadata source: {self.metadata_source}")
        for field_name, items in (
            ("loss_parameters", self.loss_parameters),
            ("custom_weighting_spec", self.custom_weighting_spec),
        ):
            keys = [str(key) for key, _ in items]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{field_name} keys must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_unit": self.prediction_unit,
            "outcome": self.outcome,
            "loss": self.loss,
            "target_population": self.target_population,
            "target_weighting": self.target_weighting,
            "risk_aggregation": self.risk_aggregation,
            "decision_estimand": self.decision_estimand,
            "domain_relation": self.domain_relation,
            "loss_parameters": dict(self.loss_parameters),
            "custom_weighting_spec": dict(self.custom_weighting_spec),
            "metadata_source": self.metadata_source,
            "metadata_completeness": (
                "legacy_defaulted"
                if self.metadata_source == "legacy_defaulted"
                else "complete"
            ),
            "endpoint_binding_status": (
                "legacy_default"
                if self.metadata_source == "legacy_defaulted"
                else "bound"
            ),
        }


@dataclass(frozen=True)
class RelationTarget:
    """Desired train-validation novelty for one provenance relation."""

    name: str
    target: float
    weight: float = 1.0
    mode: str = "target"
    tolerance: float = 0.05
    min_observations_per_fold: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("relation name must be non-empty")
        if self.mode not in VALID_MODES:
            raise ValueError(f"unknown relation mode: {self.mode}")
        if not 0.0 <= float(self.target) <= 1.0:
            raise ValueError("relation target must be in [0, 1]")
        if float(self.weight) < 0.0:
            raise ValueError("relation weight must be non-negative")
        if not 0.0 <= float(self.tolerance) <= 1.0:
            raise ValueError("relation tolerance must be in [0, 1]")
        if self.mode == "hard_unseen" and not np.isclose(self.target, 1.0):
            raise ValueError("hard_unseen relations require target=1")
        if self.mode == "support_seen" and not np.isclose(self.target, 0.0):
            raise ValueError("support_seen relations require target=0")
        if int(self.min_observations_per_fold) < 1:
            raise ValueError(
                "relation min_observations_per_fold must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": float(self.target),
            "weight": float(self.weight),
            "mode": self.mode,
            "tolerance": float(self.tolerance),
            "min_observations_per_fold": int(self.min_observations_per_fold),
        }


@dataclass(frozen=True)
class JointPatternTarget:
    """Target distribution over joint per-record novelty bit patterns."""

    name: str
    relations: tuple[str, ...]
    probabilities: tuple[tuple[str, float], ...]
    weight: float = 1.0
    tolerance: float = 0.10
    min_observations_per_fold: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("joint-pattern name must be non-empty")
        if len(self.relations) < 2:
            raise ValueError("joint patterns require at least two relations")
        if len(self.relations) != len(set(self.relations)):
            raise ValueError("joint-pattern relations must be unique")
        probability_map = dict(self.probabilities)
        if len(probability_map) != len(self.probabilities):
            raise ValueError("joint-pattern keys must be unique")
        for pattern, probability in self.probabilities:
            if len(pattern) != len(self.relations) or set(pattern) - {"0", "1"}:
                raise ValueError(
                    f"pattern {pattern!r} must be a binary string of length "
                    f"{len(self.relations)}"
                )
            if float(probability) < 0.0:
                raise ValueError("joint-pattern probabilities must be non-negative")
        if not np.isclose(sum(probability_map.values()), 1.0):
            raise ValueError("joint-pattern probabilities must sum to 1")
        if float(self.weight) < 0.0:
            raise ValueError("joint-pattern weight must be non-negative")
        if not 0.0 <= float(self.tolerance) <= 1.0:
            raise ValueError("joint-pattern tolerance must be in [0, 1]")
        if int(self.min_observations_per_fold) < 1:
            raise ValueError(
                "joint-pattern min_observations_per_fold must be positive"
            )

    @property
    def probability_map(self) -> dict[str, float]:
        return {pattern: float(value) for pattern, value in self.probabilities}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "relations": list(self.relations),
            "probabilities": self.probability_map,
            "weight": float(self.weight),
            "tolerance": float(self.tolerance),
            "min_observations_per_fold": int(self.min_observations_per_fold),
        }


@dataclass(frozen=True)
class IncidenceTarget:
    """Target mean cross-relation entity degree inside each validation fold."""

    name: str
    left_relation: str
    right_relation: str
    target_mean_degree: float
    weight: float = 1.0
    tolerance: float = 0.25
    condition_relation: str | None = None
    condition_novelty: int | None = None
    min_left_entities_per_fold: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("incidence-target name must be non-empty")
        if self.left_relation == self.right_relation:
            raise ValueError("incidence target requires two different relations")
        if float(self.target_mean_degree) <= 0:
            raise ValueError("target_mean_degree must be positive")
        if float(self.weight) < 0:
            raise ValueError("incidence-target weight must be non-negative")
        if float(self.tolerance) < 0:
            raise ValueError("incidence-target tolerance must be non-negative")
        if self.condition_relation is None and self.condition_novelty is not None:
            raise ValueError("condition_novelty requires condition_relation")
        if self.condition_relation is not None and self.condition_novelty is None:
            raise ValueError("condition_relation requires condition_novelty")
        if self.condition_novelty not in (None, 0, 1):
            raise ValueError("condition_novelty must be 0, 1 or None")
        if int(self.min_left_entities_per_fold) < 1:
            raise ValueError(
                "incidence-target min_left_entities_per_fold must be positive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "left_relation": self.left_relation,
            "right_relation": self.right_relation,
            "target_mean_degree": float(self.target_mean_degree),
            "weight": float(self.weight),
            "tolerance": float(self.tolerance),
            "condition_relation": self.condition_relation,
            "condition_novelty": self.condition_novelty,
            "min_left_entities_per_fold": int(self.min_left_entities_per_fold),
        }


@dataclass(frozen=True)
class DeploymentClaim:
    """Fold-construction contract compiled from a deployment statement."""

    relations: tuple[RelationTarget, ...]
    estimand: DeploymentEstimand = field(default_factory=DeploymentEstimand)
    joint_patterns: tuple[JointPatternTarget, ...] = ()
    incidence_targets: tuple[IncidenceTarget, ...] = ()
    ordered_relation: str | None = None
    n_splits: int = 5
    random_state: int = 0
    n_starts: int = 16
    max_iter: int = 50
    claim_weight: float = 1.0
    size_weight: float = 0.35
    bound_weight: float = 25.0
    min_fold_fraction: float = 0.10
    max_fold_fraction: float = 0.35
    contract_schema_version: str = "deployment-contract-v3"
    strict_metadata: bool = False

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.contract_schema_version != "deployment-contract-v3":
            raise ValueError("unsupported contract_schema_version")
        if self.strict_metadata and self.estimand.metadata_source == "legacy_defaulted":
            raise ValueError(
                "strict_metadata requires an explicitly sourced estimand"
            )
        if self.n_starts < 1:
            raise ValueError("n_starts must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        names = [item.name for item in self.relations]
        if len(names) != len(set(names)):
            raise ValueError("relation names must be unique")
        if not self.relations:
            raise ValueError("at least one relation target is required")
        if not 0.0 <= self.min_fold_fraction < self.max_fold_fraction <= 1.0:
            raise ValueError("fold fraction bounds must satisfy 0 <= min < max <= 1")
        if min(self.claim_weight, self.size_weight, self.bound_weight) < 0:
            raise ValueError("objective weights must be non-negative")
        relation_set = set(names)
        joint_names = [target.name for target in self.joint_patterns]
        if len(joint_names) != len(set(joint_names)):
            raise ValueError("joint-pattern target names must be unique")
        incidence_names = [target.name for target in self.incidence_targets]
        if len(incidence_names) != len(set(incidence_names)):
            raise ValueError("incidence-target names must be unique")
        if (
            self.estimand.domain_relation is not None
            and self.estimand.domain_relation not in relation_set
        ):
            raise ValueError(
                "estimand domain_relation must be one of the declared relations"
            )
        for target in self.joint_patterns:
            unknown = set(target.relations) - relation_set
            if unknown:
                raise ValueError(f"joint pattern {target.name} uses unknown relations: {unknown}")
        for target in self.incidence_targets:
            used = {target.left_relation, target.right_relation}
            if target.condition_relation is not None:
                used.add(target.condition_relation)
            unknown = used - relation_set
            if unknown:
                raise ValueError(f"incidence target {target.name} uses unknown relations: {unknown}")
        if self.ordered_relation is not None and self.ordered_relation not in relation_set:
            raise ValueError("ordered_relation must be one of the declared relations")

    @property
    def hard_relations(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.relations if item.mode == "hard_unseen")

    @property
    def optimized_relations(self) -> tuple[RelationTarget, ...]:
        return tuple(item for item in self.relations if item.mode != "ignore" and item.weight > 0)

    @property
    def relation_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.relations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_schema_version": self.contract_schema_version,
            "strict_metadata": bool(self.strict_metadata),
            "estimand": self.estimand.to_dict(),
            "relations": [item.to_dict() for item in self.relations],
            "joint_patterns": [item.to_dict() for item in self.joint_patterns],
            "incidence_targets": [item.to_dict() for item in self.incidence_targets],
            "ordered_relation": self.ordered_relation,
            "n_splits": int(self.n_splits),
            "random_state": int(self.random_state),
            "n_starts": int(self.n_starts),
            "max_iter": int(self.max_iter),
            "claim_weight": float(self.claim_weight),
            "size_weight": float(self.size_weight),
            "bound_weight": float(self.bound_weight),
            "min_fold_fraction": float(self.min_fold_fraction),
            "max_fold_fraction": float(self.max_fold_fraction),
        }


@dataclass
class CutResult:
    """Fold assignment plus complete feasibility and audit diagnostics."""

    assignments: np.ndarray
    feasible: bool
    objective: float | None
    status: str = "feasible"
    feasibility_status: str = "admissible"
    candidate_hard_satisfied: bool | None = None
    candidate_balance_satisfied: bool | None = None
    candidate_targets_satisfied: bool | None = None
    audit: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def require_feasible(self) -> "CutResult":
        if not self.feasible:
            reason = self.diagnostics.get("reason", self.status)
            if self.feasibility_status == "certified_infeasible":
                message = f"deployment partition is certified infeasible: {reason}"
            elif self.feasibility_status == "unresolved":
                message = (
                    "no admissible assignment was found under the frozen "
                    f"search budget: {reason}"
                )
            elif self.feasibility_status == "route_required":
                message = f"a different validation route is required: {reason}"
            else:
                message = f"returned assignment is not admissible: {reason}"
            raise ValueError(message)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": self.assignments.astype(int).tolist(),
            "feasible": bool(self.feasible),
            "objective": None if self.objective is None else float(self.objective),
            "status": self.status,
            "feasibility_status": self.feasibility_status,
            "candidate_hard_satisfied": (
                None
                if self.candidate_hard_satisfied is None
                else bool(self.candidate_hard_satisfied)
            ),
            "candidate_balance_satisfied": (
                None
                if self.candidate_balance_satisfied is None
                else bool(self.candidate_balance_satisfied)
            ),
            "candidate_targets_satisfied": (
                None
                if self.candidate_targets_satisfied is None
                else bool(self.candidate_targets_satisfied)
            ),
            "audit": self.audit,
            "diagnostics": self.diagnostics,
        }
