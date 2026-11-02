"""Condition-local latent neighborhoods for cell-level geometry preservation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

from .config import GeneratorConfig


@dataclass(frozen=True)
class GroupGeometry:
    source_indices: np.ndarray
    scores: np.ndarray
    fit_positions: np.ndarray
    neighbor_model: NearestNeighbors
    n_neighbors: int
    novelty_factors: np.ndarray


def fit_group_geometry(
    X: sp.csr_matrix,
    source_indices: np.ndarray,
    config: GeneratorConfig,
    seed: int,
) -> GroupGeometry | None:
    """Fit a within-condition latent space and nearest-neighbor index."""

    n_cells = X.shape[0]
    if n_cells < 3 or X.shape[1] < 2:
        return None
    library = np.asarray(X.sum(axis=1), dtype=float).ravel()
    positive = library[library > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    scale = np.divide(
        median,
        library,
        out=np.zeros_like(library),
        where=library > 0,
    )
    normalized = X.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    second = np.asarray(normalized.multiply(normalized).mean(axis=0)).ravel()
    variance = second - np.square(mean)
    n_hvg = min(config.geometry_hvg_count, X.shape[1])
    selected = np.argsort(variance)[-n_hvg:]
    selected.sort()

    rng = np.random.default_rng(np.random.SeedSequence([seed, 113]))
    fit_positions = np.arange(n_cells)
    if (
        config.max_cells_for_fitting is not None
        and n_cells > config.max_cells_for_fitting
    ):
        fit_positions = np.sort(
            rng.choice(n_cells, size=config.max_cells_for_fitting, replace=False)
        )
    n_components = min(
        config.geometry_latent_k,
        fit_positions.size - 1,
        selected.size - 1,
    )
    if n_components < 2:
        return None
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    fit_scores = svd.fit_transform(normalized[fit_positions][:, selected])
    score_mean = fit_scores.mean(axis=0)
    score_std = fit_scores.std(axis=0)
    score_std[score_std == 0] = 1.0
    fit_scores = (fit_scores - score_mean) / score_std
    scores = (normalized[:, selected] @ svd.components_.T - score_mean) / score_std
    fit_normalized = normalized[fit_positions]
    loadings = np.asarray(
        fit_normalized.T @ fit_scores, dtype=float
    ) / max(fit_positions.size - 1, 1)
    covariance = np.atleast_2d(np.cov(fit_scores, rowvar=False))
    covariance += np.eye(n_components) * 1e-8
    cholesky = np.linalg.cholesky(covariance)
    novelty_factors = loadings @ cholesky.T
    row_scale = np.sqrt(np.square(novelty_factors).sum(axis=1))
    nonzero = row_scale > 1e-12
    novelty_factors[nonzero] /= row_scale[nonzero, None]

    n_neighbors = min(
        config.geometry_neighbors,
        max(1, fit_positions.size - 1),
    )
    neighbor_model = NearestNeighbors(
        n_neighbors=min(fit_positions.size, n_neighbors + 1),
        metric="euclidean",
    ).fit(fit_scores)
    return GroupGeometry(
        source_indices=np.asarray(source_indices, dtype=np.int64),
        scores=scores,
        fit_positions=fit_positions,
        neighbor_model=neighbor_model,
        n_neighbors=n_neighbors,
        novelty_factors=novelty_factors,
    )


def query_neighbors(
    geometry: GroupGeometry,
    selected_positions: np.ndarray,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest neighbor pool positions and distances for selected sources."""

    del seed
    query_count = min(
        geometry.neighbor_model.n_neighbors,
        geometry.fit_positions.size,
    )
    distances, fit_neighbors = geometry.neighbor_model.kneighbors(
        geometry.scores[selected_positions],
        n_neighbors=query_count,
    )
    neighbor_positions = geometry.fit_positions[fit_neighbors]
    result_neighbors = np.empty(
        (selected_positions.size, geometry.n_neighbors), dtype=np.int64
    )
    result_distances = np.empty(
        (selected_positions.size, geometry.n_neighbors), dtype=float
    )
    for row, selected in enumerate(selected_positions):
        valid = neighbor_positions[row] != selected
        positions = neighbor_positions[row][valid][: geometry.n_neighbors]
        row_distances = distances[row][valid][: geometry.n_neighbors]
        if positions.size == 0:
            positions = np.array([selected], dtype=np.int64)
            row_distances = np.array([0.0])
        if positions.size < geometry.n_neighbors:
            repetitions = geometry.n_neighbors - positions.size
            positions = np.concatenate(
                [positions, np.repeat(positions[-1], repetitions)]
            )
            row_distances = np.concatenate(
                [row_distances, np.repeat(row_distances[-1], repetitions)]
            )
        result_neighbors[row] = positions
        result_distances[row] = row_distances
    return result_neighbors, result_distances


def perturbation_from_latent(
    geometry: GroupGeometry | None,
    n_cells: int,
    rng: np.random.Generator,
    strength: float,
) -> np.ndarray | None:
    """Sample gene-space perturbations from a condition's covariance decoder."""

    if geometry is None or strength == 0.0:
        return None
    latent = rng.standard_normal(
        (n_cells, geometry.novelty_factors.shape[1])
    )
    return strength * (latent @ geometry.novelty_factors.T)
