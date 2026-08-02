"""Outcome-blind compiler from deployment contracts to validation backends."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .audit import audit_assignment, provenance_fingerprint
from .claim import CutResult, DeploymentClaim
from .splitter import ClaimConditionedProvenanceCut


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CompilationResult:
    """Typed compilation decision and an auditable backend plan."""

    status: str
    backend: str
    contract_sha256: str
    plan_sha256: str
    provenance_sha256: str
    semantic_gaps: tuple[str, ...] = ()
    assignments: np.ndarray | None = None
    cut_result: CutResult | None = None
    backend_specification: dict[str, Any] = field(default_factory=dict)
    validation_card: dict[str, Any] = field(default_factory=dict)

    @property
    def exact(self) -> bool:
        return self.status == "compiled_exact"

    @property
    def executable(self) -> bool:
        return bool(
            self.assignments is not None
            and self.cut_result is not None
            and self.cut_result.feasible
            and self.status in {"compiled_exact", "assignment_audited_lossy"}
        )

    @property
    def assignment_returned(self) -> bool:
        return self.assignments is not None

    @property
    def assignment_audited(self) -> bool:
        return bool(
            self.cut_result is not None
            and bool(self.cut_result.audit)
        )

    @property
    def plan_state(self) -> str:
        return "emitted" if self.plan_sha256 else "not_emitted"

    @property
    def fidelity(self) -> str:
        if self.semantic_gaps:
            return "lossy"
        if self.status in {"backend_semantic_gap", "assignment_returned_invalid"}:
            return "unknown"
        return "exact"

    @property
    def assignment_state(self) -> str:
        if self.assignments is None:
            return (
                "not_returned"
                if self.status == "plan_emitted_lossy"
                else "not_requested"
            )
        if not self.assignment_audited:
            return "returned_unaudited"
        if self.status == "compiled_exact":
            return "audited_exact"
        if self.status == "assignment_audited_lossy":
            return "audited_lossy"
        return "audit_failed"

    @property
    def full_contract_satisfied(self) -> bool | None:
        if not self.assignment_audited:
            return None
        return bool(self.cut_result.audit.get("contract_satisfied", False))

    @property
    def backend_projection_satisfied(self) -> bool | None:
        if not self.assignment_audited:
            return None
        return bool(self.cut_result.feasible)

    @property
    def contract_admissible(self) -> bool:
        return bool(self.status == "compiled_exact" and self.full_contract_satisfied)

    def require_exact(self) -> "CompilationResult":
        if not self.exact:
            detail = ", ".join(self.semantic_gaps) or self.status
            raise ValueError(f"validation contract was not compiled exactly: {detail}")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "contract_sha256": self.contract_sha256,
            "plan_sha256": self.plan_sha256,
            "provenance_sha256": self.provenance_sha256,
            "semantic_gaps": list(self.semantic_gaps),
            "plan_state": self.plan_state,
            "fidelity": self.fidelity,
            "assignment_state": self.assignment_state,
            "assignment_returned": self.assignment_returned,
            "assignment_audited": self.assignment_audited,
            "executable": self.executable,
            "full_contract_satisfied": self.full_contract_satisfied,
            "backend_projection_satisfied": self.backend_projection_satisfied,
            "contract_admissible": self.contract_admissible,
            "assignments": (
                None
                if self.assignments is None
                else self.assignments.astype(int).tolist()
            ),
            "cut_result": (
                None if self.cut_result is None else self.cut_result.to_dict()
            ),
            "backend_specification": self.backend_specification,
            "validation_card": self.validation_card,
        }


class ValidationContractCompiler:
    """Compile one typed contract without inspecting features or outcomes.

    ``claimcut`` is the native full-semantics backend. ``group_kfold`` is an
    exact specialization for one hard-unseen relation. ``datasail_c1e_scalar``
    emits a DataSAIL-ready scalar-similarity specification and explicitly
    records semantics that the scalar backend cannot represent.
    """

    VERSION = "deployment-contract-compiler-v3"
    BACKENDS = {"auto", "claimcut", "group_kfold", "datasail_c1e_scalar"}

    def __init__(self, claim: DeploymentClaim) -> None:
        self.claim = claim

    def compile(
        self,
        provenance: pd.DataFrame,
        *,
        backend: str = "auto",
        allow_lossy: bool = False,
    ) -> CompilationResult:
        provenance = self._validate_provenance(provenance)
        if backend not in self.BACKENDS:
            raise ValueError(f"unknown validation backend: {backend}")
        selected = self._select_backend() if backend == "auto" else backend
        contract_payload = self.claim.to_dict()
        contract_hash = _canonical_sha256(contract_payload)
        provenance_hash = provenance_fingerprint(provenance)
        gaps = tuple(self._semantic_gaps(selected))
        backend_spec = self._backend_specification(selected)
        plan_payload = {
            "compiler_version": self.VERSION,
            "backend": selected,
            "contract_sha256": contract_hash,
            "provenance_sha256": provenance_hash,
            "semantic_gaps": list(gaps),
            "backend_specification": backend_spec,
        }
        plan_hash = _canonical_sha256(plan_payload)

        if self.claim.ordered_relation is not None:
            return self._result(
                status="ordered_route_required",
                backend=selected,
                contract_hash=contract_hash,
                plan_hash=plan_hash,
                provenance_hash=provenance_hash,
                gaps=gaps,
                backend_spec=backend_spec,
                reason=(
                    f"relation {self.claim.ordered_relation!r} is ordered; "
                    "compile a predeclared forward/chronological outer split"
                ),
            )

        if gaps and not allow_lossy:
            return self._result(
                status="backend_semantic_gap",
                backend=selected,
                contract_hash=contract_hash,
                plan_hash=plan_hash,
                provenance_hash=provenance_hash,
                gaps=gaps,
                backend_spec=backend_spec,
                reason="selected backend cannot represent the complete contract",
            )

        assignments: np.ndarray | None = None
        cut_result: CutResult | None = None
        if selected == "claimcut":
            cut_result = ClaimConditionedProvenanceCut(self.claim).assign(provenance)
            self._bind_infeasibility_certificate(
                cut_result,
                contract_hash=contract_hash,
                provenance_hash=provenance_hash,
            )
            if not cut_result.feasible:
                return self._result(
                    status=cut_result.status,
                    backend=selected,
                    contract_hash=contract_hash,
                    plan_hash=plan_hash,
                    provenance_hash=provenance_hash,
                    gaps=gaps,
                    backend_spec=backend_spec,
                    assignments=None,
                    cut_result=cut_result,
                    reason=cut_result.diagnostics.get("reason", cut_result.status),
                )
            assignments = cut_result.assignments
        elif selected == "group_kfold":
            assignments = self._group_assignment(provenance)
            audit = audit_assignment(provenance, assignments, self.claim)
            if not audit["fold_bounds_satisfied"]:
                status = "realized_plan_invalid_balance"
                reason = (
                    "group_kfold assignment violates declared fold-size bounds: "
                    f"observed [{audit['min_fold_fraction_observed']:.6f}, "
                    f"{audit['max_fold_fraction_observed']:.6f}] versus "
                    f"[{self.claim.min_fold_fraction:.6f}, "
                    f"{self.claim.max_fold_fraction:.6f}]"
                )
                backend_plan_feasible = False
            elif gaps:
                status = "feasible_lossy"
                reason = None
                backend_plan_feasible = True
            elif not audit["hard_constraints_satisfied"]:
                status = "realized_plan_invalid_hard"
                reason = "group_kfold assignment violates a hard provenance constraint"
                backend_plan_feasible = False
            elif not (
                audit["relation_targets_satisfied"]
                and audit["joint_patterns_satisfied"]
                and audit["incidence_targets_satisfied"]
            ):
                status = "realized_plan_invalid_targets"
                reason = "group_kfold assignment violates a declared contract target"
                backend_plan_feasible = False
            else:
                status = "feasible"
                reason = None
                backend_plan_feasible = True
            cut_result = CutResult(
                assignments=assignments,
                feasible=backend_plan_feasible,
                objective=None,
                status=status,
                feasibility_status=(
                    "proxy_admissible"
                    if status == "feasible_lossy"
                    else (
                        "admissible"
                        if status == "feasible"
                        else "invalid_realized_plan"
                    )
                ),
                candidate_hard_satisfied=bool(
                    audit["hard_constraints_satisfied"]
                ),
                candidate_balance_satisfied=bool(
                    audit["fold_bounds_satisfied"]
                ),
                candidate_targets_satisfied=bool(
                    audit["relation_targets_satisfied"]
                    and audit["joint_patterns_satisfied"]
                    and audit["incidence_targets_satisfied"]
                ),
                audit=audit,
                diagnostics={"backend": "group_kfold"},
            )
            if not cut_result.feasible:
                return self._result(
                    status=status,
                    backend=selected,
                    contract_hash=contract_hash,
                    plan_hash=plan_hash,
                    provenance_hash=provenance_hash,
                    gaps=gaps,
                    backend_spec=backend_spec,
                    assignments=assignments,
                    cut_result=cut_result,
                    reason=reason,
                )
        elif selected == "datasail_c1e_scalar":
            # Execution remains in the optional DataSAIL adapter/environment.
            # The compiler emits only the frozen, outcome-blind specification.
            assignments = None
        else:  # pragma: no cover - protected by BACKENDS validation
            raise RuntimeError(f"unhandled backend: {selected}")

        if gaps and assignments is None:
            status = "plan_emitted_lossy"
        elif gaps:
            status = "assignment_audited_lossy"
        else:
            status = "compiled_exact"
        return self._result(
            status=status,
            backend=selected,
            contract_hash=contract_hash,
            plan_hash=plan_hash,
            provenance_hash=provenance_hash,
            gaps=gaps,
            backend_spec=backend_spec,
            assignments=assignments,
            cut_result=cut_result,
            reason=None,
        )

    def audit_emitted_assignment(
        self,
        provenance: pd.DataFrame,
        plan: CompilationResult,
        assignments: np.ndarray,
    ) -> CompilationResult:
        """Attach and independently audit an assignment returned by an adapter.

        A lossy plan is not executable merely because a backend specification
        was emitted.  This transition records that an assignment was returned,
        checks its shape/fold labels, audits it against the complete contract,
        and preserves every declared semantic gap.
        """

        provenance = self._validate_provenance(provenance)
        contract_hash = _canonical_sha256(self.claim.to_dict())
        provenance_hash = provenance_fingerprint(provenance)
        if plan.contract_sha256 != contract_hash:
            raise ValueError("plan contract hash does not match this compiler")
        if plan.provenance_sha256 != provenance_hash:
            raise ValueError("plan provenance hash does not match this dataset")
        self._verify_plan_integrity(
            plan,
            contract_hash=contract_hash,
            provenance_hash=provenance_hash,
        )
        vector = np.asarray(assignments)
        valid_vector = bool(
            vector.ndim == 1
            and len(vector) == len(provenance)
            and np.issubdtype(vector.dtype, np.integer)
        )
        if valid_vector:
            vector = vector.astype(int, copy=False)
            valid_vector = bool(
                np.all((vector >= 0) & (vector < self.claim.n_splits))
                and len(np.unique(vector)) == self.claim.n_splits
            )
        if not valid_vector:
            return self._result(
                status="assignment_returned_invalid",
                backend=plan.backend,
                contract_hash=contract_hash,
                plan_hash=plan.plan_sha256,
                provenance_hash=provenance_hash,
                gaps=plan.semantic_gaps,
                backend_spec=plan.backend_specification,
                assignments=None,
                cut_result=None,
                reason=(
                    "adapter assignment must contain one in-range integer fold "
                    "label per record and every requested fold"
                ),
            )

        audit = audit_assignment(provenance, vector, self.claim)
        proxy_feasible = bool(audit["fold_bounds_satisfied"])
        if plan.semantic_gaps:
            status = (
                "assignment_audited_lossy"
                if proxy_feasible
                else "realized_plan_invalid_balance"
            )
            cut_status = "feasible_lossy" if proxy_feasible else status
        elif audit["contract_satisfied"]:
            status = "compiled_exact"
            cut_status = "feasible"
            proxy_feasible = True
        elif not audit["fold_bounds_satisfied"]:
            status = "realized_plan_invalid_balance"
            cut_status = status
            proxy_feasible = False
        elif not audit["hard_constraints_satisfied"]:
            status = "realized_plan_invalid_hard"
            cut_status = status
            proxy_feasible = False
        else:
            status = "realized_plan_invalid_targets"
            cut_status = status
            proxy_feasible = False
        cut_result = CutResult(
            assignments=vector,
            feasible=proxy_feasible,
            objective=None,
            status=cut_status,
            feasibility_status=(
                "proxy_admissible"
                if status == "assignment_audited_lossy"
                else (
                    "admissible"
                    if status == "compiled_exact"
                    else "invalid_realized_plan"
                )
            ),
            candidate_hard_satisfied=bool(
                audit["hard_constraints_satisfied"]
            ),
            candidate_balance_satisfied=bool(audit["fold_bounds_satisfied"]),
            candidate_targets_satisfied=bool(
                audit["relation_targets_satisfied"]
                and audit["joint_patterns_satisfied"]
                and audit["incidence_targets_satisfied"]
            ),
            audit=audit,
            diagnostics={
                "backend": plan.backend,
                "semantic_gaps": list(plan.semantic_gaps),
                "full_contract_satisfied": bool(audit["contract_satisfied"]),
            },
        )
        return self._result(
            status=status,
            backend=plan.backend,
            contract_hash=contract_hash,
            plan_hash=plan.plan_sha256,
            provenance_hash=provenance_hash,
            gaps=plan.semantic_gaps,
            backend_spec=plan.backend_specification,
            assignments=vector,
            cut_result=cut_result,
            reason=None if proxy_feasible else status,
        )

    def _validate_provenance(self, provenance: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(provenance, pd.DataFrame):
            raise TypeError("provenance must be a pandas DataFrame")
        if provenance.empty:
            raise ValueError("provenance must contain records")
        missing = [
            name for name in self.claim.relation_names if name not in provenance.columns
        ]
        if missing:
            raise ValueError(f"missing provenance relation columns: {missing}")
        if len(provenance) < self.claim.n_splits:
            raise ValueError("number of records must be at least n_splits")
        return provenance.loc[:, list(self.claim.relation_names)].reset_index(drop=True)

    def _bind_infeasibility_certificate(
        self,
        cut_result: CutResult,
        *,
        contract_hash: str,
        provenance_hash: str,
    ) -> None:
        raw = cut_result.diagnostics.get("certificate")
        if not isinstance(raw, dict):
            return
        kind = str(raw.get("kind", "unknown"))
        facts = {key: value for key, value in raw.items() if key != "kind"}
        certificate = {
            "schema_version": "infeasibility-certificate-v1",
            "kind": kind,
            "contract_sha256": contract_hash,
            "provenance_sha256": provenance_hash,
            "n_records": int(facts.get("n_records", 0)),
            "n_splits": int(facts.get("n_splits", self.claim.n_splits)),
            "hard_relations": list(self.claim.hard_relations),
            "facts": facts,
            "verifier": "analytic-certificate-verifier-v1",
        }
        certificate["certificate_sha256"] = _canonical_sha256(certificate)
        cut_result.diagnostics["certificate"] = certificate

    def _verify_plan_integrity(
        self,
        plan: CompilationResult,
        *,
        contract_hash: str,
        provenance_hash: str,
    ) -> None:
        if plan.backend not in self.BACKENDS - {"auto"}:
            raise ValueError("plan backend is not a concrete compiler backend")
        expected_gaps = tuple(self._semantic_gaps(plan.backend))
        expected_spec = self._backend_specification(plan.backend)
        expected_payload = {
            "compiler_version": self.VERSION,
            "backend": plan.backend,
            "contract_sha256": contract_hash,
            "provenance_sha256": provenance_hash,
            "semantic_gaps": list(expected_gaps),
            "backend_specification": expected_spec,
        }
        if plan.semantic_gaps != expected_gaps:
            raise ValueError("plan semantic gaps failed integrity verification")
        if plan.backend_specification != expected_spec:
            raise ValueError("plan backend specification failed integrity verification")
        if plan.plan_sha256 != _canonical_sha256(expected_payload):
            raise ValueError("plan hash failed integrity verification")

    def _select_backend(self) -> str:
        active = self.claim.optimized_relations
        simple = (
            len(active) == 1
            and active[0].mode == "hard_unseen"
            and not self.claim.joint_patterns
            and not self.claim.incidence_targets
            and self.claim.ordered_relation is None
        )
        return "group_kfold" if simple else "claimcut"

    def _semantic_gaps(self, backend: str) -> list[str]:
        if backend == "claimcut":
            return []
        if backend == "group_kfold":
            gaps: list[str] = []
            active = self.claim.optimized_relations
            if len(active) != 1 or active[0].mode != "hard_unseen":
                gaps.append("group_kfold_requires_one_hard_unseen_relation")
            if self.claim.joint_patterns:
                gaps.append("joint_pattern_targets_not_representable")
            if self.claim.incidence_targets:
                gaps.append("incidence_topology_targets_not_representable")
            return gaps
        if backend == "datasail_c1e_scalar":
            gaps = []
            for target in self.claim.optimized_relations:
                if target.mode == "hard_unseen":
                    gaps.append(f"hard_constraint_not_guaranteed:{target.name}")
                elif target.mode in {"target", "support_seen"}:
                    gaps.append(f"direct_novelty_target_not_representable:{target.name}")
            if self.claim.joint_patterns:
                gaps.append("joint_pattern_targets_not_representable")
            if self.claim.incidence_targets:
                gaps.append("incidence_topology_targets_not_representable")
            return gaps
        return [f"unsupported_backend:{backend}"]

    def _backend_specification(self, backend: str) -> dict[str, Any]:
        if backend == "claimcut":
            return {
                "backend": backend,
                "n_starts": int(self.claim.n_starts),
                "max_iter": int(self.claim.max_iter),
                "random_state": int(self.claim.random_state),
            }
        if backend == "group_kfold":
            active = self.claim.optimized_relations
            return {
                "backend": backend,
                "relation": active[0].name if len(active) == 1 else None,
                "n_splits": int(self.claim.n_splits),
                "shuffle": True,
                "random_state": int(self.claim.random_state),
            }
        weights = {
            target.name: float(target.weight * target.target)
            for target in self.claim.optimized_relations
            if float(target.weight * target.target) > 0
        }
        return {
            "backend": "datasail_c1e_scalar",
            "datasail_version": "1.3.0",
            "technique": "C1e",
            "solver": "SCIP",
            "epsilon": 0.05,
            "runs": 1,
            "max_sec": 60,
            "relation_similarity_weights": weights,
        }

    def _group_assignment(self, provenance: pd.DataFrame) -> np.ndarray:
        relation = self.claim.optimized_relations[0].name
        groups = provenance[relation].map(
            lambda value: "__MISSING__" if pd.isna(value) else f"{type(value).__name__}:{value}"
        )
        if groups.nunique() < self.claim.n_splits:
            raise ValueError(
                f"relation {relation!r} has fewer than n_splits distinct groups"
            )
        splitter = GroupKFold(
            n_splits=self.claim.n_splits,
            shuffle=True,
            random_state=self.claim.random_state,
        )
        assignments = np.full(len(provenance), -1, dtype=int)
        dummy = np.zeros((len(provenance), 1), dtype=float)
        for fold, (_, test) in enumerate(splitter.split(dummy, groups=groups)):
            assignments[np.asarray(test, dtype=int)] = fold
        if np.any(assignments < 0):
            raise RuntimeError("group backend left records unassigned")
        return assignments

    def _result(
        self,
        *,
        status: str,
        backend: str,
        contract_hash: str,
        plan_hash: str,
        provenance_hash: str,
        gaps: tuple[str, ...],
        backend_spec: dict[str, Any],
        reason: str | None,
        assignments: np.ndarray | None = None,
        cut_result: CutResult | None = None,
    ) -> CompilationResult:
        card = {
            "compiler_version": self.VERSION,
            "status": status,
            "backend": backend,
            "estimand": self.claim.estimand.to_dict(),
            "contract_sha256": contract_hash,
            "plan_sha256": plan_hash,
            "provenance_sha256": provenance_hash,
            "outcome_blind_compilation": True,
            "plan_state": "emitted",
            "fidelity": "lossy" if gaps else "exact",
            "assignment_state": (
                "not_returned"
                if assignments is None and status == "plan_emitted_lossy"
                else (
                    "not_requested"
                    if assignments is None
                    else (
                        "audited_exact"
                        if status == "compiled_exact"
                        else (
                            "audited_lossy"
                            if status == "assignment_audited_lossy"
                            else "audit_failed"
                        )
                    )
                )
            ),
            "assignment_returned": assignments is not None,
            "assignment_audited": bool(
                cut_result is not None and bool(cut_result.audit)
            ),
            "executable": bool(
                assignments is not None
                and cut_result is not None
                and cut_result.feasible
                and status in {"compiled_exact", "assignment_audited_lossy"}
            ),
            "full_contract_satisfied": (
                None
                if cut_result is None or not cut_result.audit
                else bool(cut_result.audit.get("contract_satisfied", False))
            ),
            "backend_projection_satisfied": (
                None
                if cut_result is None or not cut_result.audit
                else bool(cut_result.feasible)
            ),
            "semantic_gaps": list(gaps),
            "infeasibility_certificate": (
                None
                if cut_result is None
                else cut_result.diagnostics.get("certificate")
            ),
            "reason": reason,
            "assignment_checksum": (
                None
                if cut_result is None
                else cut_result.audit.get("assignment_checksum")
            ),
        }
        return CompilationResult(
            status=status,
            backend=backend,
            contract_sha256=contract_hash,
            plan_sha256=plan_hash,
            provenance_sha256=provenance_hash,
            semantic_gaps=gaps,
            assignments=assignments,
            cut_result=cut_result,
            backend_specification=backend_spec,
            validation_card=card,
        )
