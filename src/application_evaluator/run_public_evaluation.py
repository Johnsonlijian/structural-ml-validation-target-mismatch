"""Recompute the headed-stud application evaluator from user-local adapters.

This release entry point is path-neutral and never reads or writes a cached
DataSAIL assignment.  It writes only aggregate metrics, typed contract/status
receipts, and hashes.  It does not reproduce the confirmatory synthetic
pipeline and must not be described as full end-to-end pipeline reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from public_runtime import (
    assignments_from_splitter,
    atomic_write_json,
    build_models,
    datasail_scalar_assignment,
)


SCHEMA_VERSION = "public-headed-stud-evaluator-v1"
ADAPTER_FREEZE_SHA256 = "42ac1a0bed31307884ce19bc38eb629cff198cd67c06b6ea1347af1f3d919515"
PLAN_SHA256 = "9dabb8178a8f1d96fe6278adb8ce4e38d9e1e0e8dc7699c12a4a001b00d09a76"
SPLIT_SEED = 2026071201
BOOTSTRAP_SEED = 2026071202
N_BOOTSTRAP = 10000
FEATURES = [
    "f_cm_MPa",
    "E_cm_MPa",
    "f_um_MPa",
    "d_m_mm",
    "d_dom_mm",
    "h_wm_mm",
    "h_scm_mm",
]
EXTERNALS = ("deck464", "lwc90", "rac27")
EXPECTED_ADAPTER_OUTPUTS = {
    "nwc242": (242, "0536b23c575c38cd576d7aca8b2e84e86ddf142bea08dd76015764b921b9085e"),
    "deck464": (413, "7dfc3c8b5cbf5a26db2ece4e34f3208e2e1fa3a51ffc70b3bd0030c4e819b3ff"),
    "lwc90": (60, "5ce01b736b1e3c58712edbb8b9018d7ad0ec10b3ac0dc8821856a1050b6bba9b"),
    "rac27": (27, "2db515e97ea5d6a8d081f3b025bb5d56e57f3d8f3d709b4c11ac717149249d64"),
}
REFERENCE_METRICS = (
    "external_rmse_kN",
    "external_mae_kN",
    "external_nrmse",
    "signed_mean_error_kN",
)
REFERENCE_PROJECTION_FILES = (
    "results/cc_by_4_0/deck464/evaluation_external_metrics.csv",
    "results/cc_by_4_0/lwc90/evaluation_external_metrics.csv",
    "results/cc_by_nc_sa_4_0/rac27/evaluation_external_metrics.csv",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment_checksum(assignment: np.ndarray) -> str:
    values = np.asarray(assignment, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def package_versions() -> dict[str, str]:
    distributions = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "datasail",
        "cvxpy",
        "pyscipopt",
    )
    observed: dict[str, str] = {}
    for distribution in distributions:
        try:
            observed[distribution] = version(distribution)
        except PackageNotFoundError:
            observed[distribution] = "not-installed"
    return observed


def _safe_relative(root: Path, value: str) -> Path:
    candidate = (root / value.replace("\\", "/")).resolve()
    resolved_root = root.resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError("adapter manifest path escapes adapter root") from exc
    return candidate


def verify_inputs(adapter_root: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest_path = adapter_root / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("adapter_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("adapter_freeze_sha256") != ADAPTER_FREEZE_SHA256:
        raise RuntimeError("adapter manifest does not match the frozen adapter")
    if manifest.get("evaluation_plan_sha256") != PLAN_SHA256:
        raise RuntimeError("adapter manifest does not match the frozen evaluation plan")
    output_records = manifest.get("outputs")
    if not isinstance(output_records, list):
        raise RuntimeError("adapter manifest has no output registry")
    tables: dict[str, pd.DataFrame] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for output in output_records:
        parent = str(output.get("parent"))
        if parent not in EXPECTED_ADAPTER_OUTPUTS or parent in tables:
            raise RuntimeError(f"unexpected or duplicate adapter parent: {parent}")
        path = _safe_relative(adapter_root, str(output.get("path")))
        expected_rows, expected_hash = EXPECTED_ADAPTER_OUTPUTS[parent]
        observed_hash = sha256(path)
        if observed_hash != str(output.get("sha256")) or observed_hash != expected_hash:
            raise RuntimeError(f"canonical table hash mismatch: {parent}")
        frame = pd.read_csv(path)
        if len(frame) != expected_rows or int(output.get("rows")) != expected_rows:
            raise RuntimeError(f"canonical table row-count mismatch: {parent}")
        required = set(FEATURES) | {
            "P_em_kN",
            "source_group",
            "slab_topology",
            "concrete_family",
            "parent",
        }
        if not required.issubset(frame.columns):
            raise RuntimeError(f"canonical table schema mismatch: {parent}")
        tables[parent] = frame
        inputs[parent] = {
            "rows": int(len(frame)),
            "sha256": observed_hash,
        }
    if set(tables) != set(EXPECTED_ADAPTER_OUTPUTS):
        raise RuntimeError("adapter parent set is incomplete")
    return tables, {
        "adapter_manifest_sha256": sha256(manifest_path),
        "adapter_freeze_sha256": ADAPTER_FREEZE_SHA256,
        "evaluation_plan_sha256": PLAN_SHA256,
        "canonical_tables": inputs,
    }


def import_contract_api(method_root: Path) -> dict[str, Any]:
    root = method_root.resolve()
    if not (root / "provenance_cut" / "__init__.py").is_file():
        raise FileNotFoundError("method root does not contain provenance_cut")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from provenance_cut import DeploymentClaim, RelationTarget, ValidationContractCompiler
    from provenance_cut.audit import audit_assignment

    return {
        "DeploymentClaim": DeploymentClaim,
        "RelationTarget": RelationTarget,
        "ValidationContractCompiler": ValidationContractCompiler,
        "audit_assignment": audit_assignment,
    }


def build_contracts(
    provenance: pd.DataFrame,
    api: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], Any]:
    DeploymentClaim = api["DeploymentClaim"]
    RelationTarget = api["RelationTarget"]
    ValidationContractCompiler = api["ValidationContractCompiler"]
    source_claim = DeploymentClaim(
        relations=(RelationTarget("source", 1.0, mode="hard_unseen"),),
        n_splits=5,
        random_state=SPLIT_SEED,
        n_starts=32,
        max_iter=100,
        min_fold_fraction=0.12,
        max_fold_fraction=0.28,
    )
    deck_claim = DeploymentClaim(
        relations=(
            RelationTarget("source", 1.0, mode="hard_unseen"),
            RelationTarget("slab_topology", 1.0, mode="hard_unseen"),
        ),
        n_splits=5,
        random_state=SPLIT_SEED,
        n_starts=32,
        max_iter=100,
        min_fold_fraction=0.12,
        max_fold_fraction=0.28,
    )
    material_claim = DeploymentClaim(
        relations=(
            RelationTarget("source", 1.0, mode="hard_unseen"),
            RelationTarget("concrete_family", 1.0, mode="hard_unseen"),
        ),
        n_splits=5,
        random_state=SPLIT_SEED,
        n_starts=32,
        max_iter=100,
        min_fold_fraction=0.12,
        max_fold_fraction=0.28,
    )
    source_compiler = ValidationContractCompiler(source_claim)
    deck_compiler = ValidationContractCompiler(deck_claim)
    material_compiler = ValidationContractCompiler(material_claim)
    source_frame = provenance[["source"]]
    deck_frame = provenance[["source", "slab_topology"]]
    material_frame = provenance[["source", "concrete_family"]]
    results = {
        "C_source_groupkfold": source_compiler.compile(source_frame, backend="group_kfold"),
        "C_source_claimcut": source_compiler.compile(source_frame, backend="claimcut"),
        "C_deck_claimcut": deck_compiler.compile(deck_frame, backend="claimcut"),
        "C_deck_groupkfold_lossy": deck_compiler.compile(
            deck_frame, backend="group_kfold", allow_lossy=True
        ),
        "C_deck_datasail_lossy": deck_compiler.compile(
            deck_frame, backend="datasail_c1e_scalar", allow_lossy=True
        ),
        "C_lwc_claimcut": material_compiler.compile(material_frame, backend="claimcut"),
        "C_rac_claimcut": material_compiler.compile(material_frame, backend="claimcut"),
        "C_material_groupkfold_lossy": material_compiler.compile(
            material_frame, backend="group_kfold", allow_lossy=True
        ),
    }
    assignments: dict[str, np.ndarray] = {}
    if results["C_source_groupkfold"].assignments is not None:
        assignments["GroupKFold_Source"] = results[
            "C_source_groupkfold"
        ].assignments
    if results["C_source_claimcut"].assignments is not None:
        assignments["ClaimCut_C_source"] = results["C_source_claimcut"].assignments
    return results, assignments, deck_claim


def validation_metrics(
    assignments: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scale = float(np.std(y, ddof=1))
    for method, folds in assignments.items():
        for model_name, estimator in build_models(SPLIT_SEED).items():
            estimator = clone(estimator)
            if model_name == "ExtraTrees":
                estimator.set_params(n_jobs=1)
            fold_rows: list[dict[str, float | int]] = []
            for fold in sorted(np.unique(folds)):
                train = folds != fold
                validation = folds == fold
                fitted = clone(estimator).fit(X[train], y[train])
                prediction = fitted.predict(X[validation])
                fold_rows.append(
                    {
                        "n_validation": int(validation.sum()),
                        "rmse_kN": float(mean_squared_error(y[validation], prediction) ** 0.5),
                        "mae_kN": float(mean_absolute_error(y[validation], prediction)),
                    }
                )
            frame = pd.DataFrame(fold_rows)
            weights = frame["n_validation"] / frame["n_validation"].sum()
            rmse = float(np.sum(weights * frame["rmse_kN"]))
            mae = float(np.sum(weights * frame["mae_kN"]))
            rows.append(
                {
                    "method": method,
                    "model": model_name,
                    "cv_rmse_kN": rmse,
                    "cv_mae_kN": mae,
                    "cv_nrmse": rmse / scale,
                    "cv_nmae": mae / scale,
                    "n_folds": int(len(frame)),
                }
            )
    return pd.DataFrame(rows)


def external_metrics(
    development: pd.DataFrame,
    external_tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    X = development[FEATURES].to_numpy(float)
    y = development["P_em_kN"].to_numpy(float)
    scale = float(np.std(y, ddof=1))
    rows: list[dict[str, Any]] = []
    residuals: dict[tuple[str, str], np.ndarray] = {}
    for model_name, estimator in build_models(SPLIT_SEED).items():
        estimator = clone(estimator)
        if model_name == "ExtraTrees":
            estimator.set_params(n_jobs=1)
        fitted = estimator.fit(X, y)
        for parent, external in external_tables.items():
            target = external["P_em_kN"].to_numpy(float)
            prediction = fitted.predict(external[FEATURES].to_numpy(float))
            error = prediction - target
            residuals[(parent, model_name)] = error
            rmse = float(np.sqrt(np.mean(error**2)))
            mae = float(np.mean(np.abs(error)))
            rows.append(
                {
                    "parent": parent,
                    "model": model_name,
                    "n_external": int(len(external)),
                    "n_source_clusters": int(external["source_group"].nunique()),
                    "external_rmse_kN": rmse,
                    "external_mae_kN": mae,
                    "external_nrmse": rmse / scale,
                    "external_nmae": mae / scale,
                    "signed_mean_error_kN": float(np.mean(error)),
                    "median_absolute_error_kN": float(np.median(np.abs(error))),
                }
            )
    return pd.DataFrame(rows), residuals


def cluster_bootstrap(
    external: pd.DataFrame, error: np.ndarray, *, seed: int
) -> dict[str, Any]:
    groups = external["source_group"].astype(str).to_numpy()
    unique = np.unique(groups)
    inferential = len(unique) > 1 and external["parent"].iloc[0] != "rac27"
    rng = np.random.default_rng(seed)
    if inferential:
        clusters = [np.flatnonzero(groups == group) for group in unique]
        rmse_samples = np.empty(N_BOOTSTRAP)
        mae_samples = np.empty(N_BOOTSTRAP)
        for index in range(N_BOOTSTRAP):
            selected = rng.integers(0, len(clusters), size=len(clusters))
            indices = np.concatenate([clusters[item] for item in selected])
            sample = error[indices]
            rmse_samples[index] = np.sqrt(np.mean(sample**2))
            mae_samples[index] = np.mean(np.abs(sample))
        label = "source_cluster_bootstrap"
    else:
        indices = rng.integers(0, len(error), size=(N_BOOTSTRAP, len(error)))
        samples = error[indices]
        rmse_samples = np.sqrt(np.mean(samples**2, axis=1))
        mae_samples = np.mean(np.abs(samples), axis=1)
        label = "specimen_bootstrap_sensitivity_not_population_inference"
    return {
        "bootstrap_type": label,
        "population_inference": inferential,
        "n_clusters": int(len(unique)),
        "rmse_ci_low_kN": float(np.quantile(rmse_samples, 0.025)),
        "rmse_ci_high_kN": float(np.quantile(rmse_samples, 0.975)),
        "mae_ci_low_kN": float(np.quantile(mae_samples, 0.025)),
        "mae_ci_high_kN": float(np.quantile(mae_samples, 0.975)),
        "rmse_samples": rmse_samples,
    }


def support_shift(development: pd.DataFrame, external: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        lower = float(development[feature].min())
        upper = float(development[feature].max())
        values = external[feature].to_numpy(float)
        rows.append(
            {
                "parent": str(external["parent"].iloc[0]),
                "feature": feature,
                "development_min": lower,
                "development_max": upper,
                "external_min": float(np.min(values)),
                "external_max": float(np.max(values)),
                "fraction_outside_development_range": float(
                    np.mean((values < lower) | (values > upper))
                ),
            }
        )
    return rows


def load_reference_metrics(reference_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load either one private aggregate file or the public license-separated projection."""

    if reference_path.is_file():
        return pd.read_csv(reference_path), {
            "kind": "single_aggregate_file",
            "sha256": sha256(reference_path),
            "files": [reference_path.name],
        }
    if not reference_path.is_dir():
        raise FileNotFoundError("external-metric reference is neither a file nor a projection root")
    frames: list[pd.DataFrame] = []
    bindings: list[dict[str, Any]] = []
    for logical in REFERENCE_PROJECTION_FILES:
        path = _safe_relative(reference_path, logical)
        if not path.is_file():
            raise FileNotFoundError(f"public projection reference is missing: {logical}")
        frames.append(pd.read_csv(path))
        bindings.append({"path": logical, "sha256": sha256(path)})
    canonical = json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return pd.concat(frames, ignore_index=True), {
        "kind": "license_separated_public_projection",
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "files": bindings,
    }


def check_frozen_48_cells(
    observed: pd.DataFrame, reference_path: Path, tolerance: float
) -> dict[str, Any]:
    reference, reference_binding = load_reference_metrics(reference_path)
    keys = ["parent", "model"]
    if len(observed) != 12 or len(reference) != 12:
        raise RuntimeError("the frozen external-metric control requires 12 keyed rows")
    if observed.duplicated(keys).any() or reference.duplicated(keys).any():
        raise RuntimeError("duplicate frozen external-metric keys")
    merged = observed.merge(reference, on=keys, how="outer", suffixes=("_observed", "_reference"), indicator=True)
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("frozen external-metric key mismatch")
    maximum = 0.0
    checked = 0
    per_metric: dict[str, float] = {}
    for metric in REFERENCE_METRICS:
        differences = np.abs(
            merged[f"{metric}_observed"].to_numpy(float)
            - merged[f"{metric}_reference"].to_numpy(float)
        )
        metric_maximum = float(np.max(differences))
        per_metric[metric] = metric_maximum
        maximum = max(maximum, metric_maximum)
        checked += int(len(differences))
    if checked != 48:
        raise RuntimeError("frozen metric-cell count is not 48")
    status = "pass" if maximum <= float(tolerance) else "fail"
    return {
        "status": status,
        "metric_cells_checked": checked,
        "maximum_absolute_difference": maximum,
        "per_metric_maximum_absolute_difference": per_metric,
        "tolerance": float(tolerance),
        "reference": reference_binding,
    }


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise RuntimeError("output root must be empty")
    try:
        tables, input_receipt = verify_inputs(args.adapter_root.resolve())
        api = import_contract_api(args.method_root.resolve())
        development = tables["nwc242"]
        external_tables = {name: tables[name] for name in EXTERNALS}
        provenance = development.rename(columns={"source_group": "source"})[
            ["source", "slab_topology", "concrete_family"]
        ]
        contract_results, assignments, deck_claim = build_contracts(provenance, api)
        X = development[FEATURES].to_numpy(float)
        y = development["P_em_kN"].to_numpy(float)
        assignments["RandomKFold"] = assignments_from_splitter(
            KFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED), X, y
        )
        datasail_status: dict[str, Any]
        try:
            datasail_compilation = contract_results["C_deck_datasail_lossy"]
            datasail_assignment, datasail_metadata = datasail_scalar_assignment(
                provenance[["source", "slab_topology"]],
                deck_claim,
                seed=SPLIT_SEED,
                max_sec=int(args.datasail_max_sec),
                relation_weights=datasail_compilation.backend_specification[
                    "relation_similarity_weights"
                ],
            )
            assignments["DataSAIL_C1e_Scalar_C_deck"] = datasail_assignment
            datasail_status = {
                "status": "assignment_returned",
                "assignment_checksum": assignment_checksum(datasail_assignment),
                "fold_counts": np.bincount(
                    datasail_assignment, minlength=int(deck_claim.n_splits)
                ).astype(int).tolist(),
                "metadata": datasail_metadata,
                "compiler_status": datasail_compilation.status,
                "semantic_gaps": list(datasail_compilation.semantic_gaps),
                "plan_sha256": datasail_compilation.plan_sha256,
                "assignment_vector_persisted": False,
            }
        except Exception as exc:
            datasail_status = {
                "status": "failed_exception",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "assignment_vector_persisted": False,
            }
            if not args.allow_core_without_datasail:
                atomic_write_json(output_root / "evaluation_status.json", {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_closed_datasail",
                    "datasail": datasail_status,
                    "pipeline_end_to_end_reproduction": False,
                })
                raise RuntimeError("DataSAIL failed; strict evaluator stopped") from exc

        audits: dict[str, Any] = {}
        DeploymentClaim = api["DeploymentClaim"]
        RelationTarget = api["RelationTarget"]
        source_claim = DeploymentClaim(
            relations=(RelationTarget("source", 1.0, mode="hard_unseen"),),
            n_splits=5,
            random_state=SPLIT_SEED,
            min_fold_fraction=0.12,
            max_fold_fraction=0.28,
        )
        for method, assignment in assignments.items():
            audit = api["audit_assignment"](
                provenance[["source"]], assignment, source_claim
            )
            audits[method] = {
                "fold_counts": audit["fold_counts"],
                "fold_fractions": audit["fold_fractions"],
                "hard_constraints_satisfied": audit["hard_constraints_satisfied"],
                "relation_target_max_abs_error": audit["relation_target_max_abs_error"],
                "assignment_checksum": audit["assignment_checksum"],
            }

        cv = validation_metrics(assignments, X, y)
        external, residuals = external_metrics(development, external_tables)
        combined = cv.merge(external, on="model", how="inner")
        combined["signed_risk_estimation_error"] = (
            combined["cv_nrmse"] - combined["external_nrmse"]
        )
        combined["absolute_risk_estimation_error"] = combined[
            "signed_risk_estimation_error"
        ].abs()

        bootstrap_rows: list[dict[str, Any]] = []
        for _, row in external.iterrows():
            parent = str(row["parent"])
            model = str(row["model"])
            boot = cluster_bootstrap(
                external_tables[parent],
                residuals[(parent, model)],
                seed=(
                    BOOTSTRAP_SEED
                    + int.from_bytes(
                        hashlib.sha256(f"{parent}|{model}".encode()).digest()[:4],
                        "little",
                    )
                )
                % (2**32),
            )
            rmse_samples = boot.pop("rmse_samples")
            for method_row in cv.loc[cv["model"] == model].to_dict(orient="records"):
                signed_samples = method_row["cv_nrmse"] - (
                    rmse_samples / float(np.std(y, ddof=1))
                )
                bootstrap_rows.append(
                    {
                        "parent": parent,
                        "model": model,
                        "method": method_row["method"],
                        **boot,
                        "signed_risk_error_ci_low": float(
                            np.quantile(signed_samples, 0.025)
                        ),
                        "signed_risk_error_ci_high": float(
                            np.quantile(signed_samples, 0.975)
                        ),
                    }
                )

        selection_rows: list[dict[str, Any]] = []
        for method, frame in cv.groupby("method", sort=False):
            selected = str(frame.loc[frame["cv_nrmse"].idxmin(), "model"])
            for parent, ext_frame in external.groupby("parent", sort=False):
                oracle_row = ext_frame.loc[ext_frame["external_nrmse"].idxmin()]
                selected_row = ext_frame.loc[ext_frame["model"] == selected].iloc[0]
                selection_rows.append(
                    {
                        "method": method,
                        "parent": parent,
                        "selected_model": selected,
                        "external_oracle_model": str(oracle_row["model"]),
                        "external_nrmse_regret": float(
                            selected_row["external_nrmse"]
                            - oracle_row["external_nrmse"]
                        ),
                    }
                )
        support_rows = [
            row
            for parent in EXTERNALS
            for row in support_shift(development, external_tables[parent])
        ]

        write_csv(cv, output_root / "cv_metrics.csv")
        write_csv(external, output_root / "external_metrics.csv")
        write_csv(combined, output_root / "risk_estimation.csv")
        write_csv(pd.DataFrame(bootstrap_rows), output_root / "bootstrap_summary.csv")
        write_csv(pd.DataFrame(selection_rows), output_root / "model_selection.csv")
        write_csv(pd.DataFrame(support_rows), output_root / "support_shift.csv")
        atomic_write_json(output_root / "contract_statuses.json", {
            name: {
                "status": result.status,
                "backend": result.backend,
                "semantic_gaps": list(result.semantic_gaps),
                "assignment_returned_in_memory": result.assignments is not None,
                "plan_sha256": result.plan_sha256,
            }
            for name, result in contract_results.items()
        })
        atomic_write_json(output_root / "split_audits.json", audits)
        atomic_write_json(output_root / "datasail_status.json", datasail_status)

        if args.reference_external_metrics is not None:
            frozen_check = check_frozen_48_cells(
                external,
                args.reference_external_metrics.resolve(),
                float(args.tolerance),
            )
            if frozen_check["status"] != "pass":
                atomic_write_json(output_root / "frozen_48_cell_check.json", frozen_check)
                raise RuntimeError("frozen 48-cell metric check failed")
        else:
            frozen_check = {
                "status": "not_requested",
                "metric_cells_checked": 0,
                "tolerance": float(args.tolerance),
            }
        atomic_write_json(output_root / "frozen_48_cell_check.json", frozen_check)

        public_outputs = {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output_root.iterdir())
            if path.is_file() and path.name != "evaluation_receipt.json"
        }
        full_application = (
            "DataSAIL_C1e_Scalar_C_deck" in assignments
            and len(cv) == 16
            and len(combined) == 48
        )
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete" if full_application else "core_only_datasail_unavailable",
            "scope": "headed-stud application evaluator only",
            "pipeline_end_to_end_reproduction": False,
            "assignment_vectors_persisted": False,
            "split_seed": SPLIT_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "methods": list(assignments),
            "cv_rows": int(len(cv)),
            "external_rows": int(len(external)),
            "risk_estimation_rows": int(len(combined)),
            "input_receipt": input_receipt,
            "package_versions": package_versions(),
            "frozen_48_cell_check": frozen_check,
            "runtime_seconds": float(time.perf_counter() - started),
            "outputs": public_outputs,
        }
        atomic_write_json(output_root / "evaluation_receipt.json", receipt)
        return receipt
    except Exception:
        if not (output_root / "evaluation_status.json").exists():
            atomic_write_json(output_root / "evaluation_status.json", {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "pipeline_end_to_end_reproduction": False,
            })
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-root",
        type=Path,
        required=True,
        help="Directory containing adapter_manifest.json and user-local canonical tables.",
    )
    parser.add_argument(
        "--method-root",
        type=Path,
        required=True,
        help="Directory containing the public provenance_cut package.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reference-external-metrics",
        type=Path,
        help=(
            "Optional aggregate external_metrics.csv or root of the public "
            "license-separated application projection used for the 48-cell check."
        ),
    )
    parser.add_argument("--datasail-max-sec", type=int, default=60)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--allow-core-without-datasail",
        action="store_true",
        help="Emit the three-method core if DataSAIL fails; output remains explicitly incomplete.",
    )
    return parser.parse_args()


def main() -> None:
    receipt = run(parse_args())
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
