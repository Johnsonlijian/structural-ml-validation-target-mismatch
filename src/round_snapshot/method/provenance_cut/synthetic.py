"""Controlled hierarchical engineering-regression generator."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticTask:
    development: pd.DataFrame
    external: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    provenance_columns: tuple[str, ...]
    external_signature: dict[str, float]
    generator_parameters: dict[str, float | int | str]


def generate_partial_supplier_deployment(
    *,
    seed: int = 0,
    n_development_labs: int = 20,
    n_external_labs: int = 5,
    external_supplier_group_size: int = 1,
    sources_per_supplier: int = 2,
    records_per_source: int = 16,
    provenance_effect: float = 1.2,
    covariate_shift: float = 0.9,
    noise: float = 0.35,
) -> SyntheticTask:
    """Generate a new-lab deployment with half known and half new suppliers.

    Development laboratories occur in pairs. Half of each lab's records use a
    supplier shared by all laboratories; the other half use a supplier shared
    only by the paired laboratories. External laboratories use the global
    supplier for half of their records and a previously unseen supplier for the
    remainder. The true supplier novelty is therefore approximately 0.5 while
    laboratory, source and campaign novelty are 1.
    """

    if n_development_labs < 6 or n_development_labs % 2:
        raise ValueError("n_development_labs must be an even integer of at least 6")
    if min(n_external_labs, sources_per_supplier, records_per_source) < 1:
        raise ValueError("external labs, sources and records must be positive")
    if external_supplier_group_size < 1:
        raise ValueError("external_supplier_group_size must be positive")
    rng = np.random.default_rng(seed)
    materials = np.asarray(["steel", "concrete", "timber", "frp"])
    structures = np.asarray(["beam", "column", "joint"])
    rows: list[dict[str, float | str | bool]] = []

    supplier_ids = ["SUP_GLOBAL"]
    supplier_ids.extend(
        f"SUP_PAIR_{pair:02d}" for pair in range(n_development_labs // 2)
    )
    supplier_ids.extend(
        f"SUP_NEW_{group:02d}"
        for group in range(math.ceil(n_external_labs / external_supplier_group_size))
    )
    supplier_shift = {
        supplier: rng.normal(0.0, covariate_shift, size=3)
        for supplier in supplier_ids
    }
    supplier_shift["SUP_GLOBAL"] = np.zeros(3)
    supplier_effect = {
        supplier: float(
            0.7 * shift[0]
            - 0.35 * shift[1]
            + rng.normal(0.0, provenance_effect * 0.45)
        )
        for supplier, shift in supplier_shift.items()
    }
    supplier_slope = {
        supplier: float(rng.normal(0.0, provenance_effect * 0.30))
        for supplier in supplier_ids
    }

    total_labs = n_development_labs + n_external_labs
    lab_shift = rng.normal(0.0, covariate_shift, size=(total_labs, 3))
    lab_effect = (
        0.6 * lab_shift[:, 0]
        + 0.25 * lab_shift[:, 2]
        + rng.normal(0.0, provenance_effect * 0.55, size=total_labs)
    )
    lab_slope = rng.normal(
        0.0,
        provenance_effect * 0.35,
        size=(total_labs, 2),
    )
    material_effect = dict(zip(materials, [-0.45, 0.10, 0.55, 0.95]))
    structural_effect = dict(zip(structures, [-0.30, 0.35, 0.80]))

    for lab_index in range(total_labs):
        external = lab_index >= n_development_labs
        external_index = lab_index - n_development_labs
        lab = f"{'E' if external else 'D'}LAB_{lab_index:03d}"
        pair = f"PAIR_{lab_index // 2:02d}" if not external else f"EXT_{external_index:02d}"
        local_supplier = (
            f"SUP_NEW_{external_index // external_supplier_group_size:02d}"
            if external
            else f"SUP_PAIR_{lab_index // 2:02d}"
        )
        for supplier_index, supplier in enumerate(("SUP_GLOBAL", local_supplier)):
            for source_index in range(sources_per_supplier):
                source = f"{lab}_SRC_{supplier_index}_{source_index}"
                campaign = f"{lab}_CAMPAIGN_{supplier_index}"
                source_effect = rng.normal(0.0, provenance_effect * 0.30)
                for record_index in range(records_per_source):
                    material = str(
                        materials[(lab_index + source_index + record_index) % len(materials)]
                    )
                    structure = str(
                        structures[(supplier_index + source_index + record_index) % len(structures)]
                    )
                    x = rng.normal(size=6)
                    x[:3] += lab_shift[lab_index] + supplier_shift[supplier]
                    x[3] += 0.30 * material_effect[material]
                    x[4] += 0.25 * structural_effect[structure]
                    x[5] += 0.20 * supplier_index
                    shared_response = (
                        1.35 * x[0]
                        - 0.85 * x[1]
                        + 0.50 * x[2] ** 2
                        + 0.75 * np.sin(x[3])
                        + 0.35 * x[0] * x[4]
                        - 0.20 * x[5] ** 2
                    )
                    target = (
                        shared_response
                        + float(lab_effect[lab_index])
                        + supplier_effect[supplier]
                        + source_effect
                        + float(lab_slope[lab_index, 0]) * x[0]
                        + float(lab_slope[lab_index, 1]) * np.tanh(x[2])
                        + supplier_slope[supplier] * x[1]
                        + material_effect[material]
                        + structural_effect[structure]
                        + rng.normal(0.0, noise)
                    )
                    rows.append(
                        {
                            **{f"x{feature + 1}": float(x[feature]) for feature in range(6)},
                            "target": float(target),
                            "lab": lab,
                            "supplier": supplier,
                            "source": source,
                            "campaign": campaign,
                            "material_family": material,
                            "structural_family": structure,
                            "lab_pair_or_external": pair,
                            "is_external": bool(external),
                        }
                    )

    frame = pd.DataFrame(rows)
    development = frame[~frame["is_external"]].drop(columns="is_external").reset_index(drop=True)
    external = frame[frame["is_external"]].drop(columns="is_external").reset_index(drop=True)
    signature = {
        "lab": 1.0,
        "supplier": 0.5,
        "source": 1.0,
        "campaign": 1.0,
        "material_family": 0.0,
        "structural_family": 0.0,
    }
    return SyntheticTask(
        development=development,
        external=external,
        feature_columns=("x1", "x2", "x3", "x4", "x5", "x6"),
        target_column="target",
        provenance_columns=tuple(signature),
        external_signature=signature,
        generator_parameters={
            "seed": seed,
            "n_development_labs": n_development_labs,
            "n_external_labs": n_external_labs,
            "external_supplier_group_size": external_supplier_group_size,
            "sources_per_supplier": sources_per_supplier,
            "records_per_source": records_per_source,
            "provenance_effect": provenance_effect,
            "covariate_shift": covariate_shift,
            "noise": noise,
        },
    )


def generate_contract_population_v2(
    *,
    seed: int,
    n_development_labs: int = 20,
    n_external_labs: int = 200,
    external_supplier_group_size: int = 2,
    sources_per_supplier: int = 2,
    records_per_source: int = 8,
    provenance_effect: float = 1.2,
    covariate_shift: float = 0.9,
    noise: float = 0.35,
    target_effect_multiplier: float = 1.0,
    target_covariate_multiplier: float = 1.0,
    target_mechanism_shift: float = 0.0,
    lab_latent_multiplier: float = 1.0,
    supplier_latent_multiplier: float = 1.0,
    source_latent_multiplier: float = 1.0,
) -> SyntheticTask:
    """Confirmatory generator with independent development and target streams."""

    if n_development_labs < 6 or n_development_labs % 2:
        raise ValueError("n_development_labs must be an even integer of at least 6")
    if min(n_external_labs, sources_per_supplier, records_per_source) < 1:
        raise ValueError("external labs, sources and records must be positive")
    if external_supplier_group_size < 1:
        raise ValueError("external_supplier_group_size must be positive")
    root_seed = np.random.SeedSequence(seed)
    shared_sequence, development_sequence, external_sequence = root_seed.spawn(3)
    rng_shared = np.random.default_rng(shared_sequence)
    rng_development = np.random.default_rng(development_sequence)
    rng_external = np.random.default_rng(external_sequence)
    materials = np.asarray(["steel", "concrete", "timber", "frp"])
    structures = np.asarray(["beam", "column", "joint"])
    material_effect = dict(zip(materials, [-0.45, 0.10, 0.55, 0.95]))
    structural_effect = dict(zip(structures, [-0.30, 0.35, 0.80]))

    supplier_shift: dict[str, np.ndarray] = {"SUP_GLOBAL": np.zeros(3)}
    supplier_effect: dict[str, float] = {
        "SUP_GLOBAL": float(
            supplier_latent_multiplier
            * rng_shared.normal(0.0, provenance_effect * 0.45)
        )
    }
    supplier_slope: dict[str, float] = {
        "SUP_GLOBAL": float(
            supplier_latent_multiplier
            * rng_shared.normal(0.0, provenance_effect * 0.30)
        )
    }
    for pair in range(n_development_labs // 2):
        supplier = f"SUP_PAIR_{pair:03d}"
        shift = rng_development.normal(0.0, covariate_shift, size=3)
        supplier_shift[supplier] = shift
        supplier_effect[supplier] = float(
            supplier_latent_multiplier
            * (
                0.7 * shift[0]
                - 0.35 * shift[1]
                + rng_development.normal(0.0, provenance_effect * 0.45)
            )
        )
        supplier_slope[supplier] = float(
            supplier_latent_multiplier
            * rng_development.normal(0.0, provenance_effect * 0.30)
        )
    n_external_supplier_groups = math.ceil(
        n_external_labs / external_supplier_group_size
    )
    for group in range(n_external_supplier_groups):
        supplier = f"SUP_NEW_{group:04d}"
        shift = rng_external.normal(
            0.0,
            covariate_shift * target_covariate_multiplier,
            size=3,
        )
        supplier_shift[supplier] = shift
        supplier_effect[supplier] = float(
            target_effect_multiplier
            * supplier_latent_multiplier
            * (
                0.7 * shift[0]
                - 0.35 * shift[1]
                + rng_external.normal(0.0, provenance_effect * 0.45)
            )
        )
        supplier_slope[supplier] = float(
            target_effect_multiplier
            * supplier_latent_multiplier
            * rng_external.normal(0.0, provenance_effect * 0.30)
        )

    development_lab_shift = rng_development.normal(
        0.0,
        covariate_shift,
        size=(n_development_labs, 3),
    )
    external_lab_shift = rng_external.normal(
        0.0,
        covariate_shift * target_covariate_multiplier,
        size=(n_external_labs, 3),
    )
    development_lab_effect = lab_latent_multiplier * (
        0.6 * development_lab_shift[:, 0]
        + 0.25 * development_lab_shift[:, 2]
        + rng_development.normal(
            0.0,
            provenance_effect * 0.55,
            size=n_development_labs,
        )
    )
    external_lab_effect = target_effect_multiplier * lab_latent_multiplier * (
        0.6 * external_lab_shift[:, 0]
        + 0.25 * external_lab_shift[:, 2]
        + rng_external.normal(
            0.0,
            provenance_effect * 0.55,
            size=n_external_labs,
        )
    )
    development_lab_slope = lab_latent_multiplier * rng_development.normal(
        0.0,
        provenance_effect * 0.35,
        size=(n_development_labs, 2),
    )
    external_lab_slope = (
        target_effect_multiplier
        * lab_latent_multiplier
        * rng_external.normal(
        0.0,
        provenance_effect * 0.35,
        size=(n_external_labs, 2),
        )
    )

    rows: list[dict[str, float | str | bool]] = []
    total_labs = n_development_labs + n_external_labs
    for combined_index in range(total_labs):
        external = combined_index >= n_development_labs
        if external:
            local_index = combined_index - n_development_labs
            rng = rng_external
            lab_shift = external_lab_shift[local_index]
            lab_effect = external_lab_effect[local_index]
            lab_slope = external_lab_slope[local_index]
            lab = f"ELAB_{local_index:04d}"
            local_supplier = (
                f"SUP_NEW_{local_index // external_supplier_group_size:04d}"
            )
            pair = f"EXT_GROUP_{local_index // external_supplier_group_size:04d}"
        else:
            local_index = combined_index
            rng = rng_development
            lab_shift = development_lab_shift[local_index]
            lab_effect = development_lab_effect[local_index]
            lab_slope = development_lab_slope[local_index]
            lab = f"DLAB_{local_index:03d}"
            local_supplier = f"SUP_PAIR_{local_index // 2:03d}"
            pair = f"PAIR_{local_index // 2:03d}"

        for supplier_index, supplier in enumerate(("SUP_GLOBAL", local_supplier)):
            for source_index in range(sources_per_supplier):
                source = f"{lab}_SRC_{supplier_index}_{source_index}"
                campaign = f"{lab}_CAMPAIGN_{supplier_index}"
                source_scale = provenance_effect * 0.30
                if external:
                    source_scale *= target_effect_multiplier
                source_scale *= source_latent_multiplier
                source_effect = rng.normal(0.0, source_scale)
                for record_index in range(records_per_source):
                    material = str(
                        materials[
                            (combined_index + source_index + record_index)
                            % len(materials)
                        ]
                    )
                    structure = str(
                        structures[
                            (supplier_index + source_index + record_index)
                            % len(structures)
                        ]
                    )
                    x = rng.normal(size=6)
                    x[:3] += lab_shift + supplier_shift[supplier]
                    x[3] += 0.30 * material_effect[material]
                    x[4] += 0.25 * structural_effect[structure]
                    x[5] += 0.20 * supplier_index
                    x0_coefficient = 1.35 + (
                        target_mechanism_shift if external else 0.0
                    )
                    shared_response = (
                        x0_coefficient * x[0]
                        - 0.85 * x[1]
                        + 0.50 * x[2] ** 2
                        + 0.75 * np.sin(x[3])
                        + 0.35 * x[0] * x[4]
                        - 0.20 * x[5] ** 2
                    )
                    target = (
                        shared_response
                        + float(lab_effect)
                        + supplier_effect[supplier]
                        + source_effect
                        + float(lab_slope[0]) * x[0]
                        + float(lab_slope[1]) * np.tanh(x[2])
                        + supplier_slope[supplier] * x[1]
                        + material_effect[material]
                        + structural_effect[structure]
                        + rng.normal(0.0, noise)
                    )
                    rows.append(
                        {
                            **{
                                f"x{feature + 1}": float(x[feature])
                                for feature in range(6)
                            },
                            "target": float(target),
                            "lab": lab,
                            "supplier": supplier,
                            "source": source,
                            "campaign": campaign,
                            "material_family": material,
                            "structural_family": structure,
                            "lab_pair_or_external": pair,
                            "is_external": bool(external),
                        }
                    )

    frame = pd.DataFrame(rows)
    development = (
        frame[~frame["is_external"]]
        .drop(columns="is_external")
        .reset_index(drop=True)
    )
    external = (
        frame[frame["is_external"]]
        .drop(columns="is_external")
        .reset_index(drop=True)
    )
    signature = {
        "lab": 1.0,
        "supplier": 0.5,
        "source": 1.0,
        "campaign": 1.0,
        "material_family": 0.0,
        "structural_family": 0.0,
    }
    return SyntheticTask(
        development=development,
        external=external,
        feature_columns=("x1", "x2", "x3", "x4", "x5", "x6"),
        target_column="target",
        provenance_columns=tuple(signature),
        external_signature=signature,
        generator_parameters={
            "generator_version": "v2_independent_streams",
            "seed": seed,
            "shared_spawn_key": str(shared_sequence.spawn_key),
            "development_spawn_key": str(development_sequence.spawn_key),
            "external_spawn_key": str(external_sequence.spawn_key),
            "n_development_labs": n_development_labs,
            "n_external_labs": n_external_labs,
            "external_supplier_group_size": external_supplier_group_size,
            "sources_per_supplier": sources_per_supplier,
            "records_per_source": records_per_source,
            "provenance_effect": provenance_effect,
            "covariate_shift": covariate_shift,
            "noise": noise,
            "target_effect_multiplier": target_effect_multiplier,
            "target_covariate_multiplier": target_covariate_multiplier,
            "target_mechanism_shift": target_mechanism_shift,
            "lab_latent_multiplier": lab_latent_multiplier,
            "supplier_latent_multiplier": supplier_latent_multiplier,
            "source_latent_multiplier": source_latent_multiplier,
        },
    )


def generate_hierarchical_engineering_regression(
    *,
    seed: int = 0,
    n_development_labs: int = 24,
    n_external_labs: int = 6,
    sources_per_lab: int = 3,
    records_per_source: int = 20,
    provenance_effect: float = 1.0,
    interaction_effect: float = 0.6,
    covariate_shift: float = 0.8,
    noise: float = 0.35,
) -> SyntheticTask:
    """Create development and untouched external labs with shared support families."""

    if min(n_development_labs, n_external_labs, sources_per_lab, records_per_source) < 1:
        raise ValueError("all entity and record counts must be positive")
    rng = np.random.default_rng(seed)
    n_labs = n_development_labs + n_external_labs
    material_levels = np.asarray(["steel", "concrete", "timber", "frp"])
    structural_levels = np.asarray(["beam", "column", "joint"])
    rows: list[dict[str, float | str | bool]] = []

    lab_effects = rng.normal(0.0, provenance_effect, size=n_labs)
    lab_feature_shift = rng.normal(0.0, covariate_shift, size=(n_labs, 3))
    material_effect = {name: value for name, value in zip(material_levels, [-0.5, 0.2, 0.6, 1.0])}
    structural_effect = {name: value for name, value in zip(structural_levels, [-0.3, 0.4, 0.9])}

    for lab_index in range(n_labs):
        is_external = lab_index >= n_development_labs
        lab_id = f"{'E' if is_external else 'D'}L{lab_index:03d}"
        for source_index in range(sources_per_lab):
            source_id = f"{lab_id}_S{source_index:02d}"
            campaign_id = f"{lab_id}_C{source_index // 2:02d}"
            source_effect = rng.normal(0.0, provenance_effect * 0.65)
            for record_index in range(records_per_source):
                material = str(material_levels[(source_index + record_index) % len(material_levels)])
                structural = str(
                    structural_levels[(lab_index + source_index + record_index) % len(structural_levels)]
                )
                x = rng.normal(size=5)
                x[:3] += lab_feature_shift[lab_index]
                x[3] += 0.35 * material_effect[material]
                x[4] += 0.25 * structural_effect[structural]
                nonlinear = (
                    1.4 * x[0]
                    - 0.9 * x[1]
                    + 0.55 * x[2] ** 2
                    + 0.8 * np.sin(x[3])
                    + 0.35 * x[0] * x[4]
                )
                interaction = interaction_effect * material_effect[material] * structural_effect[structural]
                target = (
                    nonlinear
                    + lab_effects[lab_index]
                    + source_effect
                    + material_effect[material]
                    + structural_effect[structural]
                    + interaction
                    + rng.normal(0.0, noise)
                )
                row: dict[str, float | str | bool] = {
                    **{f"x{j + 1}": float(x[j]) for j in range(5)},
                    "target": float(target),
                    "lab": lab_id,
                    "source": source_id,
                    "campaign": campaign_id,
                    "material_family": material,
                    "structural_family": structural,
                    "is_external": bool(is_external),
                }
                rows.append(row)

    frame = pd.DataFrame(rows)
    development = frame[~frame["is_external"]].drop(columns="is_external").reset_index(drop=True)
    external = frame[frame["is_external"]].drop(columns="is_external").reset_index(drop=True)
    signature = {
        "lab": 1.0,
        "source": 1.0,
        "campaign": 1.0,
        "material_family": 0.0,
        "structural_family": 0.0,
    }
    return SyntheticTask(
        development=development,
        external=external,
        feature_columns=("x1", "x2", "x3", "x4", "x5"),
        target_column="target",
        provenance_columns=tuple(signature),
        external_signature=signature,
        generator_parameters={
            "seed": seed,
            "n_development_labs": n_development_labs,
            "n_external_labs": n_external_labs,
            "sources_per_lab": sources_per_lab,
            "records_per_source": records_per_source,
            "provenance_effect": provenance_effect,
            "interaction_effect": interaction_effect,
            "covariate_shift": covariate_shift,
            "noise": noise,
        },
    )
