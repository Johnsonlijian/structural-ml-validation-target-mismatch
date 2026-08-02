"""Resumable confirmatory/pilot runner for typed provenance contracts."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROUND_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = ROUND_ROOT / "method"
if str(METHOD_ROOT) not in sys.path:
    sys.path.insert(0, str(METHOD_ROOT))

from provenance_cut import (  # noqa: E402
    ClaimConditionedProvenanceCut,
    DeploymentClaim,
    DeploymentEstimand,
    IncidenceTarget,
    JointPatternTarget,
    RelationTarget,
    ValidationContractCompiler,
)
from provenance_cut.audit import audit_assignment, normalize_provenance  # noqa: E402
from provenance_cut.synthetic import generate_contract_population_v2  # noqa: E402

MASTER_SEED = 2026071201
SCHEMA_VERSION = "cv1-replicate-v4"
PROVENANCE_COLUMNS = (
    "lab",
    "supplier",
    "source",
    "campaign",
    "material_family",
    "structural_family",
)


def stable_scenario_code(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def replicate_seeds(scenario: str, replicate: int) -> dict[str, int]:
    root = np.random.SeedSequence(
        [MASTER_SEED, stable_scenario_code(scenario), int(replicate)]
    )
    labels = ("generator", "split", "learner", "datasail")
    return {
        label: int(child.generate_state(1, dtype=np.uint32)[0])
        for label, child in zip(labels, root.spawn(len(labels)))
    }


def frozen_seed_table_sha256() -> str | None:
    path = Path(__file__).resolve().parent / "freeze" / "confirmatory_seed_table.csv"
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_seed_row(
    table: pd.DataFrame,
    scenario: str,
    replicate: int,
) -> None:
    """Verify one seed-table row without accepting an out-of-table run."""

    selected = table.loc[
        (table["scenario"] == scenario)
        & (table["replicate"] == int(replicate))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"scenario/replicate is absent or duplicated in frozen seed table: "
            f"{scenario}/{replicate}"
        )
    expected = replicate_seeds(scenario, replicate)
    row = selected.iloc[0]
    if int(row["master_seed"]) != MASTER_SEED or any(
        int(row[f"{name}_seed"]) != value
        for name, value in expected.items()
    ):
        raise RuntimeError(
            f"frozen seed mismatch for {scenario}/{replicate}"
        )


def verify_confirmatory_freeze(scenario: str, replicate: int) -> None:
    """Require a final v3 freeze and the exact prespecified seed row."""

    freeze_root = Path(__file__).resolve().parent / "freeze_v3"
    manifest_path = freeze_root / "freeze_manifest.json"
    seed_path = freeze_root / "confirmatory_seed_table.csv"
    if not manifest_path.is_file() or not seed_path.is_file():
        raise RuntimeError("confirmatory freeze manifest or seed table is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "final":
        raise RuntimeError("confirmatory execution requires a final freeze")
    if manifest.get("output_schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            "freeze schema does not match the current replicate schema"
        )
    verify_frozen_seed_row(pd.read_csv(seed_path), scenario, replicate)


def build_claim(
    *,
    split_seed: int,
    target_supplier_degree: float,
) -> DeploymentClaim:
    return DeploymentClaim(
        estimand=DeploymentEstimand(
            prediction_unit="synthetic structural record",
            outcome="continuous structural response",
            loss="squared_error",
            target_population="new laboratories under the frozen deployment generator",
            target_weighting="equal_domain",
            decision_estimand="mean normalized predictive error across target laboratories",
            domain_relation="lab",
            metadata_source="prespecified",
        ),
        relations=(
            RelationTarget("lab", 1.0, mode="hard_unseen", tolerance=0.0),
            RelationTarget(
                "supplier",
                0.5,
                weight=2.0,
                mode="target",
                tolerance=0.02,
            ),
            RelationTarget("source", 1.0, mode="target", tolerance=0.0),
            RelationTarget("campaign", 1.0, mode="target", tolerance=0.0),
            RelationTarget(
                "material_family",
                0.0,
                mode="support_seen",
                tolerance=0.02,
            ),
            RelationTarget(
                "structural_family",
                0.0,
                mode="support_seen",
                tolerance=0.02,
            ),
        ),
        joint_patterns=(
            JointPatternTarget(
                name="lab_supplier_joint",
                relations=("lab", "supplier"),
                probabilities=(("10", 0.5), ("11", 0.5)),
                tolerance=0.02,
            ),
        ),
        incidence_targets=(
            IncidenceTarget(
                name="unseen_supplier_lab_degree",
                left_relation="supplier",
                right_relation="lab",
                target_mean_degree=float(target_supplier_degree),
                condition_relation="supplier",
                condition_novelty=1,
                tolerance=0.05,
            ),
        ),
        n_splits=5,
        random_state=int(split_seed),
        n_starts=16,
        max_iter=50,
        min_fold_fraction=0.15,
        max_fold_fraction=0.25,
        claim_weight=1.0,
        size_weight=0.5,
        bound_weight=40.0,
        strict_metadata=True,
    )


def assignments_from_splitter(splitter, X, y, groups=None) -> np.ndarray:
    assignments = np.full(len(X), -1, dtype=int)
    for fold, (_, test) in enumerate(splitter.split(X, y, groups=groups)):
        assignments[np.asarray(test, dtype=int)] = fold
    if np.any(assignments < 0):
        raise RuntimeError("splitter left unassigned records")
    return assignments


def baseline_group_values(values: pd.Series) -> np.ndarray:
    """Type-stable groups with a distinct token for every unknown identity."""

    output = []
    for index, value in enumerate(values.reset_index(drop=True)):
        if pd.isna(value):
            output.append(f"__UNKNOWN_RECORD_{index:06d}")
        else:
            output.append(f"{type(value).__name__}:{value}")
    return np.asarray(output, dtype=str)


def parse_datasail_solver_status(log: str, *, assignment_returned: bool) -> str:
    """Return a structured status without upgrading an ambiguous solver log."""

    lower = log.lower()
    if "solution may be inaccurate" in lower or "optimal_inaccurate" in lower:
        return "optimal_inaccurate"
    matches = re.findall(
        r"(?:solver\s+)?status\s*[:=]\s*([a-z_]+)",
        lower,
    )
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


def external_novelty_by_fold(
    development: pd.DataFrame,
    external: pd.DataFrame,
    assignments: np.ndarray,
    n_splits: int,
) -> dict[str, dict[str, float | None]]:
    dev = normalize_provenance(development)
    ext = normalize_provenance(external)
    output: dict[str, dict[str, float | None]] = {}
    for relation in dev.columns:
        ext_values = ext[relation].dropna()
        folds: dict[str, float | None] = {}
        for fold in range(n_splits):
            train_values = set(dev.loc[assignments != fold, relation].dropna())
            folds[f"fold_{fold}"] = (
                None
                if ext_values.empty
                else float((~ext_values.isin(train_values)).mean())
            )
        output[relation] = folds
    return output


def signature_rmse(
    audit: dict[str, Any],
    external_novelty: dict[str, dict[str, float | None]],
    n_splits: int,
) -> float:
    squared = []
    for relation in PROVENANCE_COLUMNS:
        for fold in range(n_splits):
            internal = audit["novelty"][relation][f"fold_{fold}"]
            external = external_novelty[relation][f"fold_{fold}"]
            if internal is not None and external is not None:
                squared.append((float(internal) - float(external)) ** 2)
    return float(np.sqrt(np.mean(squared))) if squared else float("nan")


def datasail_scalar_assignment(
    provenance: pd.DataFrame,
    claim: DeploymentClaim,
    *,
    seed: int,
    max_sec: int,
    relation_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from datasail.sail import datasail

    np.random.seed(seed)
    random.seed(seed)
    n = len(provenance)
    ids = [f"record_{index:06d}" for index in range(n)]
    similarity = np.zeros((n, n), dtype=np.float64)
    total_weight = 0.0
    scalar_weights: dict[str, float] = {}
    compiled_weights = relation_weights or {
        target.name: float(target.weight * target.target)
        for target in claim.optimized_relations
    }
    for relation, raw_weight in compiled_weights.items():
        weight = float(raw_weight)
        if weight <= 0:
            continue
        values = provenance[relation]
        observed = values.notna().to_numpy()
        string_values = values.astype(str).to_numpy()
        equal = string_values[:, None] == string_values[None, :]
        equal &= observed[:, None] & observed[None, :]
        similarity += weight * equal
        total_weight += weight
        scalar_weights[relation] = weight
    if total_weight <= 0:
        raise ValueError("zero scalar similarity weight")
    similarity /= total_weight
    np.fill_diagonal(similarity, 1.0)
    n_clusters = min(50, max(10, n // 20))
    solver_capture = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(solver_capture), contextlib.redirect_stderr(
        solver_capture
    ):
        e_splits, _, _ = datasail(
            techniques=["C1e"],
            splits=[1] * claim.n_splits,
            names=[f"fold_{fold}" for fold in range(claim.n_splits)],
            runs=1,
            epsilon=0.05,
            solver="SCIP",
            max_sec=max_sec,
            verbose="W",
            e_type="O",
            e_data={
                name: np.asarray([index], dtype=float)
                for index, name in enumerate(ids)
            },
            e_sim=(ids, similarity),
            e_clusters=n_clusters,
        )
    runtime_seconds = float(time.perf_counter() - started)
    full_log = solver_capture.getvalue()
    mapping = e_splits["C1e"][0]
    assignments = np.asarray(
        [int(str(mapping[name]).split("_")[-1]) for name in ids],
        dtype=int,
    )
    return assignments, {
        "version": "1.3.0",
        "technique": "C1e",
        "solver": "SCIP",
        "max_sec": int(max_sec),
        "epsilon": 0.05,
        "n_clusters": int(n_clusters),
        "scalar_relation_weights": scalar_weights,
        "similarity_sha256": hashlib.sha256(similarity.tobytes()).hexdigest(),
        "runtime_seconds": runtime_seconds,
        "solver_status": parse_datasail_solver_status(
            full_log,
            assignment_returned=True,
        ),
        "assignment_returned": True,
        "solver_time_limit_likely": bool(
            max_sec > 0 and runtime_seconds >= 0.98 * float(max_sec)
        ),
        "solver_log_sha256": hashlib.sha256(
            full_log.encode("utf-8", errors="replace")
        ).hexdigest(),
        "solver_log_tail": full_log[-4000:],
    }


def build_models(seed: int) -> dict[str, object]:
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
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=seed,
        ),
    }


def scenario_parameters(scenario: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "n_development_labs": 20,
        "n_external_labs": 200,
        "external_supplier_group_size": 2,
        "sources_per_supplier": 2,
        "records_per_source": 8,
        "provenance_effect": 1.2,
        "covariate_shift": 0.9,
        "noise": 0.35,
        "target_effect_multiplier": 1.0,
        "target_covariate_multiplier": 1.0,
        "target_mechanism_shift": 0.0,
        "lab_latent_multiplier": 1.0,
        "supplier_latent_multiplier": 1.0,
        "source_latent_multiplier": 1.0,
    }
    if scenario == "S0":
        base.update(provenance_effect=0.0, covariate_shift=0.0)
    elif scenario == "S1":
        base["permute_provenance_rows"] = True
    elif scenario == "S2":
        base.update(
            supplier_latent_multiplier=0.0,
            source_latent_multiplier=0.0,
        )
    elif scenario == "S3":
        pass
    elif scenario == "S4":
        base["external_supplier_group_size"] = 1
    elif scenario == "S5":
        base["target_effect_multiplier"] = 2.0
    elif scenario == "S6":
        base.update(
            provenance_effect=0.0,
            target_covariate_multiplier=2.0,
        )
    elif scenario == "S7":
        base["target_mechanism_shift"] = 0.8
    elif scenario == "S9_missing10":
        base["missing_fraction"] = 0.10
    elif scenario == "S9_missing30":
        base["missing_fraction"] = 0.30
    elif scenario == "S9_corrupt5":
        base["corrupt_fraction"] = 0.05
    elif scenario.startswith("CORE_"):
        parts = scenario.split("_")
        if len(parts) != 4:
            raise ValueError(
                "CORE scenario must be CORE_<effect>_<covariate>_<noise>"
            )
        base.update(
            provenance_effect=float(parts[1]),
            covariate_shift=float(parts[2]),
            noise=float(parts[3]),
        )
    else:
        raise ValueError(f"unsupported fitted scenario: {scenario}")
    return base


def apply_provenance_stress(
    development: pd.DataFrame,
    parameters: dict[str, Any],
    rng: np.random.Generator,
) -> pd.DataFrame:
    result = development.copy()
    if parameters.get("permute_provenance_rows"):
        permutation = rng.permutation(len(result))
        result.loc[:, PROVENANCE_COLUMNS] = (
            result.loc[permutation, PROVENANCE_COLUMNS]
            .reset_index(drop=True)
            .to_numpy()
        )
    missing_fraction = float(parameters.get("missing_fraction", 0.0))
    if missing_fraction > 0:
        for relation in PROVENANCE_COLUMNS:
            mask = rng.random(len(result)) < missing_fraction
            result.loc[mask, relation] = np.nan
    corrupt_fraction = float(parameters.get("corrupt_fraction", 0.0))
    if corrupt_fraction > 0:
        for relation in PROVENANCE_COLUMNS:
            indices = np.flatnonzero(rng.random(len(result)) < corrupt_fraction)
            if len(indices) > 1:
                shuffled = indices.copy()
                rng.shuffle(shuffled)
                result.loc[indices, relation] = result.loc[
                    shuffled, relation
                ].to_numpy()
    return result


def typed_refusal_scenario(
    scenario: str,
    split_seed: int,
) -> dict[str, Any]:
    if scenario == "S8":
        provenance = pd.DataFrame(
            {
                "lab": ["dominant"] * 70
                + ["small_1"] * 10
                + ["small_2"] * 10
                + ["small_3"] * 10
            }
        )
        claim = DeploymentClaim(
            estimand=DeploymentEstimand(
                prediction_unit="synthetic structural record",
                outcome="continuous structural response",
                loss="squared_error",
                target_population="frozen dominant-component refusal case",
                target_weighting="record_weighted",
                decision_estimand="partition admissibility",
                metadata_source="prespecified",
            ),
            relations=(RelationTarget("lab", 1.0, mode="hard_unseen"),),
            n_splits=4,
            random_state=split_seed,
            min_fold_fraction=0.10,
            max_fold_fraction=0.40,
            strict_metadata=True,
        )
    elif scenario == "S10":
        provenance = pd.DataFrame({"time": np.arange(100)})
        claim = DeploymentClaim(
            estimand=DeploymentEstimand(
                prediction_unit="synthetic structural record",
                outcome="continuous structural response",
                loss="squared_error",
                target_population="future ordered records",
                target_weighting="record_weighted",
                decision_estimand="forward-deployment predictive loss",
                metadata_source="prespecified",
            ),
            relations=(RelationTarget("time", 1.0, mode="target"),),
            n_splits=5,
            random_state=split_seed,
            ordered_relation="time",
            strict_metadata=True,
        )
    else:
        raise ValueError(scenario)
    result = ClaimConditionedProvenanceCut(claim).assign(provenance)
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "claimcut_status": result.status,
        "claimcut_feasible": result.feasible,
        "diagnostics": result.diagnostics,
        "expected_status": (
            ["certified_balance_infeasible"]
            if scenario == "S8"
            else ["ordered_route_required"]
        ),
    }


def compact_audit(
    provenance: pd.DataFrame,
    external_provenance: pd.DataFrame,
    assignments: np.ndarray,
    claim: DeploymentClaim,
) -> dict[str, Any]:
    audit = audit_assignment(provenance, assignments, claim)
    external_novelty = external_novelty_by_fold(
        provenance,
        external_provenance,
        assignments,
        claim.n_splits,
    )
    return {
        "fold_counts": audit["fold_counts"],
        "fold_fractions": audit["fold_fractions"],
        "hard_constraints_satisfied": audit["hard_constraints_satisfied"],
        "fold_bounds_satisfied": audit["fold_bounds_satisfied"],
        "relation_targets_satisfied": audit["relation_targets_satisfied"],
        "joint_patterns_satisfied": audit["joint_patterns_satisfied"],
        "incidence_targets_satisfied": audit[
            "incidence_targets_satisfied"
        ],
        "contract_satisfied": audit["contract_satisfied"],
        "relation_target_max_abs_error": audit[
            "relation_target_max_abs_error"
        ],
        "joint_patterns": audit["joint_patterns"],
        "incidence_targets": audit["incidence_targets"],
        "signature_rmse": signature_rmse(
            audit,
            external_novelty,
            claim.n_splits,
        ),
        "assignment_checksum": audit["assignment_checksum"],
        "provenance_fingerprint": audit["provenance_fingerprint"],
    }


def evaluate_assignments(
    *,
    assignments: dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    X_external: np.ndarray,
    y_external: np.ndarray,
    external_domains: np.ndarray,
    models: dict[str, object],
    scale: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    fold_rows: list[dict[str, Any]] = []
    method_model_rows: list[dict[str, Any]] = []
    for method, fold_assignment in assignments.items():
        for model_name, estimator in models.items():
            model_fold_rows = []
            for fold in sorted(np.unique(fold_assignment)):
                train = fold_assignment != fold
                validation = fold_assignment == fold
                model = clone(estimator)
                model.fit(X[train], y[train])
                validation_prediction = model.predict(X[validation])
                external_prediction = model.predict(X_external)
                validation_nrmse = float(
                    mean_squared_error(y[validation], validation_prediction) ** 0.5
                    / scale
                )
                external_nrmse = float(
                    mean_squared_error(y_external, external_prediction) ** 0.5
                    / scale
                )
                domain_squared_errors = pd.DataFrame(
                    {
                        "domain": external_domains,
                        "squared_error": (y_external - external_prediction) ** 2,
                    }
                ).groupby("domain", sort=False)["squared_error"].mean()
                equal_domain_nrmse = float(
                    np.sqrt(domain_squared_errors.mean()) / scale
                )
                worst_domain_nrmse = float(
                    np.sqrt(domain_squared_errors.max()) / scale
                )
                row = {
                    "method": method,
                    "model": model_name,
                    "fold": int(fold),
                    "n_train": int(train.sum()),
                    "n_validation": int(validation.sum()),
                    "validation_nrmse": validation_nrmse,
                    "external_record_weighted_nrmse": external_nrmse,
                    "external_equal_domain_nrmse": equal_domain_nrmse,
                    "external_worst_domain_nrmse": worst_domain_nrmse,
                }
                fold_rows.append(row)
                model_fold_rows.append(row)
            model_frame = pd.DataFrame(model_fold_rows)
            weights = model_frame["n_validation"].to_numpy(dtype=float)
            weights /= weights.sum()
            validation = float(
                np.sum(weights * model_frame["validation_nrmse"].to_numpy())
            )
            external_record = float(
                np.sum(
                    weights
                    * model_frame["external_record_weighted_nrmse"].to_numpy()
                )
            )
            external_domain = float(
                np.sum(
                    weights
                    * model_frame["external_equal_domain_nrmse"].to_numpy()
                )
            )
            external_worst_domain = float(
                np.sum(
                    weights
                    * model_frame["external_worst_domain_nrmse"].to_numpy()
                )
            )
            signed_record = validation - external_record
            signed_domain = validation - external_domain
            signed_worst_domain = validation - external_worst_domain
            method_model_rows.append(
                {
                    "method": method,
                    "model": model_name,
                    "aggregate_validation_nrmse": validation,
                    "aggregate_external_record_weighted_nrmse": external_record,
                    "aggregate_external_equal_domain_nrmse": external_domain,
                    "aggregate_external_worst_domain_nrmse": (
                        external_worst_domain
                    ),
                    "signed_bias_record_weighted": signed_record,
                    "ere_record_weighted": abs(signed_record),
                    "squared_calibration_error_record_weighted": (
                        signed_record**2
                    ),
                    "signed_bias_equal_domain": signed_domain,
                    "ere_equal_domain": abs(signed_domain),
                    "squared_calibration_error_equal_domain": (
                        signed_domain**2
                    ),
                    "signed_bias_worst_domain": signed_worst_domain,
                    "ere_worst_domain": abs(signed_worst_domain),
                    "squared_calibration_error_worst_domain": (
                        signed_worst_domain**2
                    ),
                    "foldwise_absolute_error_secondary": float(
                        np.sum(
                            weights
                            * np.abs(
                                model_frame["validation_nrmse"].to_numpy()
                                - model_frame[
                                    "external_record_weighted_nrmse"
                                ].to_numpy()
                            )
                        )
                    ),
                }
            )

    full_refit_external: dict[str, float] = {}
    for model_name, estimator in models.items():
        model = clone(estimator)
        model.fit(X, y)
        full_refit_external[model_name] = float(
            mean_squared_error(y_external, model.predict(X_external)) ** 0.5
            / scale
        )
    summaries = []
    method_model_frame = pd.DataFrame(method_model_rows)
    for method, frame in method_model_frame.groupby("method", sort=False):
        selected = str(frame.loc[frame["aggregate_validation_nrmse"].idxmin(), "model"])
        oracle = min(full_refit_external, key=full_refit_external.get)
        rank = spearmanr(
            frame.set_index("model")["aggregate_validation_nrmse"],
            pd.Series(full_refit_external).reindex(frame["model"]),
        ).statistic
        summaries.append(
            {
                "method": method,
                "mean_ere_record_weighted": float(
                    frame["ere_record_weighted"].mean()
                ),
                "mean_ere_equal_domain": float(
                    frame["ere_equal_domain"].mean()
                ),
                "mean_ere_worst_domain": float(
                    frame["ere_worst_domain"].mean()
                ),
                "mean_squared_calibration_error_record_weighted": float(
                    frame[
                        "squared_calibration_error_record_weighted"
                    ].mean()
                ),
                "mean_squared_calibration_error_equal_domain": float(
                    frame["squared_calibration_error_equal_domain"].mean()
                ),
                "mean_squared_calibration_error_worst_domain": float(
                    frame["squared_calibration_error_worst_domain"].mean()
                ),
                "mean_external_worst_domain_nrmse": float(
                    frame["aggregate_external_worst_domain_nrmse"].mean()
                ),
                "mean_signed_bias_record_weighted": float(
                    frame["signed_bias_record_weighted"].mean()
                ),
                "model_rank_spearman": float(rank),
                "selected_model": selected,
                "external_oracle_model": oracle,
                "decision_regret": float(
                    full_refit_external[selected]
                    - full_refit_external[oracle]
                ),
            }
        )
    return fold_rows, method_model_rows, {
        **full_refit_external,
        "_method_summaries": summaries,
    }


def run_replicate(
    *,
    scenario: str,
    replicate: int,
    skip_datasail: bool,
    datasail_max_sec: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    seeds = replicate_seeds(scenario, replicate)
    seed_table_sha256 = frozen_seed_table_sha256()
    if scenario in {"S8", "S10"}:
        output = typed_refusal_scenario(scenario, seeds["split"])
        output.update(
            replicate=int(replicate),
            seeds=seeds,
            seed_table_sha256=seed_table_sha256,
            runtime_seconds=float(time.perf_counter() - started),
            status="complete",
        )
        return output

    parameters = scenario_parameters(scenario)
    generator_keys = {
        key: value
        for key, value in parameters.items()
        if key
        not in {
            "permute_provenance_rows",
            "missing_fraction",
            "corrupt_fraction",
        }
    }
    task = generate_contract_population_v2(
        seed=seeds["generator"],
        **generator_keys,
    )
    split_rng = np.random.default_rng(seeds["split"])
    development = apply_provenance_stress(
        task.development,
        parameters,
        split_rng,
    )
    external = task.external
    provenance = development.loc[:, PROVENANCE_COLUMNS]
    external_provenance = external.loc[:, PROVENANCE_COLUMNS]
    claim = build_claim(
        split_seed=seeds["split"],
        target_supplier_degree=float(
            parameters["external_supplier_group_size"]
        ),
    )
    X = development.loc[:, task.feature_columns].to_numpy(dtype=float)
    y = development[task.target_column].to_numpy(dtype=float)
    X_external = external.loc[:, task.feature_columns].to_numpy(dtype=float)
    y_external = external[task.target_column].to_numpy(dtype=float)
    scale = float(np.std(y, ddof=1))

    compiler = ValidationContractCompiler(claim)
    claim_started = time.perf_counter()
    claim_compilation = compiler.compile(provenance, backend="claimcut")
    claim_result = claim_compilation.cut_result
    if claim_result is None:
        raise RuntimeError(
            f"native compilation returned no cut result: {claim_compilation.status}"
        )
    claim_runtime = time.perf_counter() - claim_started
    union_claim = DeploymentClaim(
        estimand=claim.estimand,
        relations=(
            RelationTarget("lab", 1.0, mode="hard_unseen"),
            RelationTarget("supplier", 1.0, mode="hard_unseen"),
        ),
        n_splits=5,
        random_state=seeds["split"],
        strict_metadata=True,
    )
    union_result = ClaimConditionedProvenanceCut(union_claim).assign(
        provenance.loc[:, ["lab", "supplier"]]
    )

    if scenario == "S4":
        return {
            "schema_version": SCHEMA_VERSION,
            "scenario": scenario,
            "replicate": int(replicate),
            "seeds": seeds,
            "seed_table_sha256": seed_table_sha256,
            "generator_parameters": task.generator_parameters,
            "claim": claim.to_dict(),
            "claimcut_status": claim_result.status,
            "claimcut_feasible": claim_result.feasible,
            "claimcut_diagnostics": claim_result.diagnostics,
            "claimcut_runtime_seconds": float(claim_runtime),
            "claimcut_compilation": claim_compilation.to_dict(),
            "expected_status": ["backend_search_exhausted"],
            "status": "complete",
            "runtime_seconds": float(time.perf_counter() - started),
        }

    assignments: dict[str, np.ndarray] = {
        "RandomKFold": assignments_from_splitter(
            KFold(n_splits=5, shuffle=True, random_state=seeds["split"]),
            X,
            y,
        )
    }
    for relation, label in (
        ("lab", "GroupKFold_Lab"),
        ("source", "GroupKFold_Source"),
        ("supplier", "GroupKFold_Supplier"),
        ("lab_pair_or_external", "CompositeGroup_LabPair"),
    ):
        assignments[label] = assignments_from_splitter(
            GroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=seeds["split"],
            ),
            X,
            y,
            baseline_group_values(development[relation]),
        )
    if union_result.feasible and union_result.assignments is not None:
        assignments["HardUnion_LabSupplier"] = union_result.assignments
    if claim_result.feasible:
        assignments["ClaimCut"] = claim_result.assignments

    datasail_metadata: dict[str, Any] = {
        "attempted": False,
        "solver_status": "skipped",
        "assignment_returned": False,
        "solver_time_limit_likely": False,
        "solver_log_sha256": None,
        "solver_log_tail": "",
    }
    datasail_error = None
    datasail_execution = None
    datasail_compilation = compiler.compile(
        provenance,
        backend="datasail_c1e_scalar",
        allow_lossy=True,
    )
    if not skip_datasail:
        try:
            ds_assignment, datasail_metadata = datasail_scalar_assignment(
                provenance,
                claim,
                seed=seeds["datasail"],
                max_sec=datasail_max_sec,
                relation_weights=datasail_compilation.backend_specification[
                    "relation_similarity_weights"
                ],
            )
            datasail_metadata["compiler_status"] = datasail_compilation.status
            datasail_metadata["semantic_gaps"] = list(
                datasail_compilation.semantic_gaps
            )
            datasail_metadata["plan_sha256"] = datasail_compilation.plan_sha256
            datasail_metadata["attempted"] = True
            datasail_execution = compiler.audit_emitted_assignment(
                provenance,
                datasail_compilation,
                ds_assignment,
            )
            datasail_metadata["execution_status"] = datasail_execution.status
            datasail_metadata["assignment_audited"] = bool(
                datasail_execution.assignment_audited
            )
            datasail_metadata["executable"] = bool(datasail_execution.executable)
            assignments["DataSAIL_C1e_Scalar"] = ds_assignment
        except Exception as exc:
            datasail_error = f"{type(exc).__name__}: {exc}"
            datasail_metadata = {
                "attempted": True,
                "solver_status": "failed_exception",
                "assignment_returned": False,
                "solver_time_limit_likely": False,
                "solver_log_sha256": None,
                "solver_log_tail": "",
                "error": datasail_error,
                "compiler_status": datasail_compilation.status,
                "semantic_gaps": list(datasail_compilation.semantic_gaps),
                "plan_sha256": datasail_compilation.plan_sha256,
            }

    audits = {
        method: compact_audit(
            provenance,
            external_provenance,
            assignment,
            claim,
        )
        for method, assignment in assignments.items()
    }
    models = build_models(seeds["learner"])
    fold_rows, method_model_rows, full_payload = evaluate_assignments(
        assignments=assignments,
        X=X,
        y=y,
        X_external=X_external,
        y_external=y_external,
        external_domains=external["lab"].to_numpy(),
        models=models,
        scale=scale,
    )
    method_summaries = full_payload.pop("_method_summaries")
    method_summary_by_name = {
        row["method"]: row for row in method_summaries
    }
    for row in method_model_rows:
        method = str(row["method"])
        audit = audits[method]
        finite_relation_errors = [
            float(value)
            for value in audit["relation_target_max_abs_error"].values()
            if value is not None and np.isfinite(float(value))
        ]
        if method == "ClaimCut":
            semantic_loss_count: int | None = len(
                claim_compilation.semantic_gaps
            )
            backend_compile_exact: bool | None = bool(
                claim_compilation.exact
            )
        elif method == "DataSAIL_C1e_Scalar":
            semantic_loss_count = len(datasail_compilation.semantic_gaps)
            backend_compile_exact = bool(datasail_compilation.exact)
        else:
            semantic_loss_count = None
            backend_compile_exact = None
        mechanism = {
            "contract_signature_rmse": float(audit["signature_rmse"]),
            "max_relation_target_abs_error": (
                max(finite_relation_errors)
                if finite_relation_errors
                else None
            ),
            "semantic_loss_count": semantic_loss_count,
            "backend_compile_exact": backend_compile_exact,
        }
        row.update(mechanism)
        method_summary_by_name[method].update(mechanism)
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "replicate": int(replicate),
        "seeds": seeds,
        "seed_table_sha256": seed_table_sha256,
        "generator_parameters": task.generator_parameters,
        "n_development": int(len(development)),
        "n_external": int(len(external)),
        "n_external_domains": int(external["lab"].nunique()),
        "development_scale": scale,
        "claim": claim.to_dict(),
        "claimcut_status": claim_result.status,
        "claimcut_feasible": claim_result.feasible,
        "claimcut_diagnostics": claim_result.diagnostics,
        "claimcut_runtime_seconds": float(claim_runtime),
        "claimcut_compilation": claim_compilation.to_dict(),
        "union_component_status": union_result.status,
        "union_component_feasible": union_result.feasible,
        "datasail_metadata": datasail_metadata,
        "datasail_error": datasail_error,
        "datasail_compilation": datasail_compilation.to_dict(),
        "datasail_execution": (
            None if datasail_execution is None else datasail_execution.to_dict()
        ),
        "methods": list(assignments),
        "audits": audits,
        "fold_rows": fold_rows,
        "method_model_rows": method_model_rows,
        "method_summaries": method_summaries,
        "full_refit_external_nrmse": full_payload,
        "status": "complete",
        "runtime_seconds": float(time.perf_counter() - started),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Commit one result in-place without exposing a partial JSON file."""

    name_token = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:6]
    temporary = path.with_name(f".__tmp_{os.getpid()}_{name_token}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_confirmatory_claim(
    output_file: Path,
    *,
    scenario: str,
    replicate: int,
) -> Path:
    """Exclusively claim one immutable result path before any computation."""

    claim_path = output_file.with_suffix(".claim")
    payload = {
        "claim_schema": "confirmatory-exclusive-claim-v1",
        "scenario": scenario,
        "replicate": int(replicate),
        "process_id": int(os.getpid()),
        "claimed_utc": datetime.now(timezone.utc).isoformat(),
        "replicate_schema": SCHEMA_VERSION,
        "seed_table_sha256": frozen_seed_table_sha256(),
    }
    try:
        with claim_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(
            f"confirmatory replicate is already claimed; do not rerun in "
            f"this output root: {claim_path}"
        ) from exc
    return claim_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-datasail", action="store_true")
    parser.add_argument("--datasail-max-sec", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--confirmatory",
        action="store_true",
        help=(
            "require the frozen seed table and preserve every pre-existing "
            "result, including failures"
        ),
    )
    args = parser.parse_args()

    if args.confirmatory and args.overwrite:
        parser.error("--overwrite is forbidden in --confirmatory mode")
    output_dir = Path(args.output).resolve()
    if args.confirmatory:
        expected_output_dir = (
            Path(__file__).resolve().parent
            / "confirmatory_v3"
            / args.scenario
        ).resolve()
        if output_dir != expected_output_dir:
            parser.error(
                "--confirmatory output must be the frozen scenario path: "
                f"{expected_output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    for replicate in range(args.start, args.start + args.count):
        if args.confirmatory:
            verify_confirmatory_freeze(args.scenario, replicate)
        output_file = output_dir / f"rep_{replicate:04d}.json"
        if output_file.exists() and args.confirmatory:
            try:
                existing = json.loads(output_file.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"unreadable confirmatory result must not be replaced: "
                    f"{output_file}"
                ) from exc
            if existing.get("status") == "failed":
                raise RuntimeError(
                    f"preserved confirmatory failure requires a new protocol "
                    f"or output root: {output_file}"
                )
            print(f"SKIP_PRESERVED {output_file}")
            continue
        if output_file.exists() and not args.overwrite:
            try:
                existing = json.loads(output_file.read_text(encoding="utf-8"))
                if existing.get("status") == "complete":
                    print(f"SKIP {output_file}")
                    continue
            except Exception:
                pass
        if args.confirmatory:
            create_confirmatory_claim(
                output_file,
                scenario=args.scenario,
                replicate=replicate,
            )
        try:
            payload = run_replicate(
                scenario=args.scenario,
                replicate=replicate,
                skip_datasail=args.skip_datasail,
                datasail_max_sec=args.datasail_max_sec,
            )
        except Exception as exc:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "scenario": args.scenario,
                "replicate": int(replicate),
                "seeds": replicate_seeds(args.scenario, replicate),
                "seed_table_sha256": frozen_seed_table_sha256(),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        atomic_write_json(output_file, payload)
        print(
            f"{payload['status'].upper()} scenario={args.scenario} "
            f"replicate={replicate} output={output_file}"
        )
        if args.confirmatory and payload["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
