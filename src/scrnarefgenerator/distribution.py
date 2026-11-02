"""Parameter estimation for condition-aware count simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from .conditions import ConditionPlan
from .config import GeneratorConfig
from .geometry import GroupGeometry, fit_group_geometry
from .validation import ReferenceData


@dataclass(frozen=True)
class GroupParameters:
    values: tuple[Any, ...]
    probabilities: np.ndarray
    dispersion: np.ndarray
    zero_probability: np.ndarray
    n_reference: int
    geometry: GroupGeometry | None = None


@dataclass(frozen=True)
class FittedDistribution:
    groups: tuple[GroupParameters, ...]
    global_probabilities: np.ndarray
    latent_scores: np.ndarray | None = None
    latent_components: np.ndarray | None = None
    latent_covariance: np.ndarray | None = None


def fit_distribution(
    reference: ReferenceData,
    plan: ConditionPlan,
    config: GeneratorConfig,
) -> FittedDistribution:
    """Fit shrunk group probabilities, dispersions, and optional latent factors."""

    X = reference.X
    library = np.asarray(X.sum(axis=1), dtype=float).ravel()
    median_library = float(np.median(library[library > 0])) if np.any(library > 0) else 1.0
    gene_totals = np.asarray(X.sum(axis=0), dtype=float).ravel()
    denominator = float(library.sum())
    if denominator <= 0:
        raise ValueError("reference counts have zero total library size")
    global_prob = (gene_totals + 1e-8) / (denominator + 1e-8 * X.shape[1])
    global_prob /= global_prob.sum()
    global_zero = 1.0 - X.getnnz(axis=0) / X.shape[0]
    kappa = config.prior_strength * median_library

    groups: list[GroupParameters] = []
    for group_index, target in enumerate(plan.groups):
        indices = _fitting_indices(
            target.source_indices, config.max_cells_for_fitting, config.seed
        )
        group_X = X[indices]
        group_library = library[indices]
        group_total = float(group_library.sum())
        group_gene_totals = np.asarray(group_X.sum(axis=0), dtype=float).ravel()
        probabilities = (group_gene_totals + kappa * global_prob) / (
            group_total + kappa
        )
        probabilities /= probabilities.sum()

        dispersion = _estimate_dispersion(
            group_X,
            group_library,
            median_library,
            config.dispersion_n0,
        )
        zero_count = group_X.shape[0] - group_X.getnnz(axis=0)
        zero_probability = (zero_count + config.prior_strength * global_zero) / (
            group_X.shape[0] + config.prior_strength
        )
        groups.append(
            GroupParameters(
                values=target.values,
                probabilities=probabilities,
                dispersion=dispersion,
                zero_probability=zero_probability,
                n_reference=int(target.source_indices.size),
                geometry=(
                    fit_group_geometry(
                        X[target.source_indices],
                        target.source_indices,
                        config,
                        seed=config.seed + group_index,
                    )
                    if config.geometry_model == "neighbor_preserving"
                    else None
                ),
            )
        )

    scores = components = covariance = None
    if config.latent_k > 0 and config.latent_k < min(
        reference.n_obs, reference.n_vars
    ):
        scores, components, covariance = _fit_latent(reference, config)
    return FittedDistribution(
        groups=tuple(groups),
        global_probabilities=global_prob,
        latent_scores=scores,
        latent_components=components,
        latent_covariance=covariance,
    )


def _fitting_indices(
    source_indices: np.ndarray,
    maximum: int | None,
    seed: int,
) -> np.ndarray:
    if maximum is None or source_indices.size <= maximum:
        return source_indices
    rng = np.random.default_rng(np.random.SeedSequence([seed, 91]))
    selected = rng.choice(source_indices, size=maximum, replace=False)
    return np.sort(selected)


def _estimate_dispersion(
    X: sp.csr_matrix,
    library: np.ndarray,
    median_library: float,
    n0: float,
) -> np.ndarray:
    n_cells = X.shape[0]
    if n_cells < 2:
        return np.ones(X.shape[1], dtype=float)
    scale = np.divide(
        median_library,
        library,
        out=np.zeros_like(library, dtype=float),
        where=library > 0,
    )
    normalized = X.multiply(scale[:, None]).tocsr()
    mean = np.asarray(normalized.sum(axis=0), dtype=float).ravel() / n_cells
    square = normalized.copy()
    square.data = np.square(square.data, dtype=float)
    second_moment = np.asarray(square.sum(axis=0), dtype=float).ravel() / n_cells
    variance = np.maximum(second_moment - np.square(mean), 0.0)
    variance *= n_cells / max(n_cells - 1, 1)

    valid = variance > mean
    moment = np.full(mean.shape, 1e8, dtype=float)
    moment[valid] = np.square(mean[valid]) / np.maximum(
        variance[valid] - mean[valid], 1e-8
    )
    trend = _dispersion_trend(mean, moment)
    weight = n_cells / (n_cells + n0)
    log_theta = (1.0 - weight) * np.log(np.clip(moment, 1e-6, 1e8))
    log_theta += weight * np.log(np.clip(trend, 1e-6, 1e8))
    return np.clip(np.exp(log_theta), 1e-4, 1e8)


def _dispersion_trend(mean: np.ndarray, moment: np.ndarray) -> np.ndarray:
    usable = (mean > 0) & np.isfinite(moment) & (moment > 0)
    if usable.sum() < 2 or np.unique(np.log(mean[usable])).size < 2:
        fallback = float(np.median(moment[usable])) if usable.any() else 1.0
        return np.full(mean.shape, fallback, dtype=float)
    x = np.log(mean[usable])
    y = np.log(np.clip(moment[usable], 1e-6, 1e8))
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted = np.exp(np.clip(intercept + slope * np.log(np.clip(mean, 1e-12, None)), -14, 18))
    if usable.any():
        predicted[~usable] = np.median(predicted[usable])
    return np.clip(predicted, 1e-6, 1e8)


def _fit_latent(
    reference: ReferenceData,
    config: GeneratorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = reference.X
    library = np.asarray(X.sum(axis=1), dtype=float).ravel()
    median_library = float(np.median(library[library > 0]))
    scale = np.divide(
        median_library,
        library,
        out=np.zeros_like(library, dtype=float),
        where=library > 0,
    )
    normalized = X.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)

    fit_indices = _fitting_indices(
        np.arange(reference.n_obs), config.max_cells_for_fitting, config.seed
    )
    k = min(config.latent_k, fit_indices.size - 1, reference.n_vars - 1)
    svd = TruncatedSVD(n_components=k, random_state=config.seed)
    fit_scores = svd.fit_transform(normalized[fit_indices])
    mean = fit_scores.mean(axis=0)
    std = fit_scores.std(axis=0)
    std[std == 0] = 1.0
    fit_scores = (fit_scores - mean) / std
    scores = (normalized @ svd.components_.T - mean) / std
    covariance = np.cov(fit_scores, rowvar=False)
    covariance = np.atleast_2d(covariance)
    covariance += np.eye(k) * 1e-8
    return scores, svd.components_.T, covariance
