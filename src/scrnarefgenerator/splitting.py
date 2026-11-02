"""Deterministic condition-stratified reference splits."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import anndata as ad
import numpy as np

from .config import GeneratorConfig
from .errors import ConditionColumnNotFoundError
from .validation import ReferenceData, resolve_reference


@dataclass(frozen=True)
class ReferenceSplit:
    training: ad.AnnData
    holdout: ad.AnnData
    training_indices: np.ndarray
    holdout_indices: np.ndarray


def stratified_reference_split(
    reference: ReferenceData | ad.AnnData | str | PathLike[str],
    condition_col: str,
    fraction: float = 0.20,
    seed: int = 42,
) -> ReferenceSplit:
    """Split each condition category into disjoint training and holdout sets."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1")
    data = (
        reference
        if isinstance(reference, ReferenceData)
        else resolve_reference(reference, GeneratorConfig())
    )
    if condition_col not in data.obs:
        raise ConditionColumnNotFoundError(
            f"condition column {condition_col!r} is not present in reference.obs"
        )
    if data.obs[condition_col].isna().any():
        raise ValueError(f"condition column {condition_col!r} contains missing values")

    rng = np.random.default_rng(seed)
    training_parts: list[np.ndarray] = []
    holdout_parts: list[np.ndarray] = []
    categories = sorted(data.obs[condition_col].unique(), key=repr)
    for category in categories:
        indices = np.flatnonzero(data.obs[condition_col].eq(category).to_numpy())
        if indices.size < 4:
            raise ValueError(
                f"category {category!r} has {indices.size} cells; "
                "at least 4 are required for stratified splitting"
            )
        shuffled = rng.permutation(indices)
        n_holdout = max(2, int(round(indices.size * fraction)))
        n_holdout = min(n_holdout, indices.size - 2)
        holdout_parts.append(shuffled[:n_holdout])
        training_parts.append(shuffled[n_holdout:])

    training_indices = np.sort(np.concatenate(training_parts))
    holdout_indices = np.sort(np.concatenate(holdout_parts))
    return ReferenceSplit(
        training=_subset_anndata(data, training_indices),
        holdout=_subset_anndata(data, holdout_indices),
        training_indices=training_indices,
        holdout_indices=holdout_indices,
    )


def _subset_anndata(data: ReferenceData, indices: np.ndarray) -> ad.AnnData:
    counts = data.X[indices].copy()
    return ad.AnnData(
        X=counts,
        obs=data.obs.iloc[indices].copy(),
        var=data.var.copy(),
        layers={"counts": counts},
    )
