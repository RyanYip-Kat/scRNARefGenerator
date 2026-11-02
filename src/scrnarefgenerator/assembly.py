"""Assemble generated counts and metadata into the public AnnData contract."""

from __future__ import annotations

import uuid
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from .conditions import ConditionPlan
from .config import GeneratorConfig
from .errors import BarcodeSuffixCollisionError
from .simulate import CandidateBatch
from .validation import ReferenceData


def validate_barcode_policy(
    reference: ReferenceData,
    plan: ConditionPlan,
    config: GeneratorConfig,
) -> None:
    """Reject suffix collisions for any source that the plan could select."""

    if config.collision_policy != "error":
        return
    existing = set(reference.obs_names.astype(str))
    possible = {
        f"{reference.obs_names[int(index)]}{config.barcode_suffix}"
        for group in plan.groups
        for index in group.source_indices
    }
    collisions = sorted(possible & existing)
    if collisions:
        example = collisions[0]
        raise BarcodeSuffixCollisionError(
            f"generated barcode {example!r} already exists; "
            "choose another barcode_suffix or collision_policy='append_uuid'"
        )


def assemble_anndata(
    reference: ReferenceData,
    candidate: CandidateBatch,
    plan: ConditionPlan,
    config: GeneratorConfig,
    similarity_strength: float,
    manifest: dict[str, Any],
) -> ad.AnnData:
    """Build the final AnnData with stable barcodes and complete metadata."""

    source_barcodes = reference.obs_names[candidate.source_indices].astype(str)
    barcodes = _make_barcodes(source_barcodes.to_numpy(), config)
    if pd.Index(barcodes).duplicated().any():
        raise BarcodeSuffixCollisionError("generated barcode names are duplicated")
    if set(barcodes) & set(reference.obs_names.astype(str)):
        raise BarcodeSuffixCollisionError("generated barcode names collide with reference")

    obs = candidate.obs.copy()
    obs.index = pd.Index(barcodes, name=None)
    attempt_seed = int(
        np.random.SeedSequence([config.seed, candidate.attempt, 17]).generate_state(
            1, dtype=np.uint64
        )[0]
        % (2**63)
    )
    obs["source_barcode"] = source_barcodes.to_numpy()
    obs["sim_batch"] = f"sim_{config.seed}_{candidate.attempt}"
    obs["sim_seed"] = attempt_seed
    obs["similarity_strength"] = float(similarity_strength)
    obs["distribution_model"] = config.distribution_model
    obs["novelty_mode"] = config.novelty_mode
    obs["novelty_strength"] = config.effective_novelty_strength
    source_obs = reference.obs.iloc[candidate.source_indices].reset_index(drop=True)
    for column in plan.columns:
        if not obs[column].reset_index(drop=True).astype(str).equals(
            source_obs[column].astype(str)
        ):
            raise ValueError(
                f"generated condition column {column!r} does not match its source"
            )
    if config.source_metadata:
        for column in reference.obs.columns:
            obs[f"source_{column}"] = source_obs[column].to_numpy()
    inherited = config.inherit_obs_cols or []
    reserved = set(obs.columns) | {
        "source_barcode",
        "sim_batch",
        "sim_seed",
        "similarity_strength",
        "distribution_model",
        "novelty_mode",
        "novelty_strength",
    }
    for column in inherited:
        if column not in source_obs:
            raise ValueError(
                f"inherited obs column {column!r} is absent from reference"
            )
        if column in plan.columns:
            continue
        if column in reserved:
            raise ValueError(
                f"inherited obs column {column!r} conflicts with an output column"
            )
        obs[column] = source_obs[column].to_numpy()

    generated = ad.AnnData(
        X=candidate.counts,
        obs=obs,
        var=reference.var.copy(),
        layers={"counts": candidate.counts},
        uns={"scRNARefGenerator": manifest},
    )
    generated.var_names = reference.var_names.copy()
    return generated


def _make_barcodes(source_barcodes: np.ndarray, config: GeneratorConfig) -> list[str]:
    base = [
        f"{source_barcode}{config.barcode_suffix}"
        for source_barcode in source_barcodes.tolist()
    ]
    if config.collision_policy == "append_uuid":
        namespace = uuid.UUID("8f4f91fc-1d25-4dd4-b480-9f990f6f4887")
        base = [
            f"{barcode}_{uuid.uuid5(namespace, f'{config.seed}:{source}').hex[:8]}"
            for barcode, source in zip(base, source_barcodes.tolist())
        ]
    return base
