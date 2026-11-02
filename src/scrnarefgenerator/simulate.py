"""Sparse count sampling for concrete condition plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .conditions import ConditionPlan
from .config import GeneratorConfig
from .distribution import FittedDistribution, fit_distribution
from .geometry import query_neighbors
from .validation import ReferenceData


@dataclass(frozen=True)
class CandidateBatch:
    counts: sp.csr_matrix
    obs: pd.DataFrame
    target_library_size: np.ndarray
    source_indices: np.ndarray
    attempt: int


def sample_candidate(
    reference: ReferenceData,
    plan: ConditionPlan,
    config: GeneratorConfig,
    attempt: int = 0,
    fitted: FittedDistribution | None = None,
    similarity_strength: float | None = None,
) -> CandidateBatch:
    """Generate one exact-quota sparse candidate with an attempt-specific RNG."""

    if fitted is None:
        fitted = fit_distribution(reference, plan, config)
    strength = (
        config.similarity_strength
        if similarity_strength is None
        else float(similarity_strength)
    )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("similarity_strength must be between 0 and 1")

    rng = np.random.default_rng(
        np.random.SeedSequence([config.seed, int(attempt), 17])
    )
    source_library = np.asarray(reference.X.sum(axis=1), dtype=float).ravel()
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    all_sources: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    obs_parts: list[pd.DataFrame] = []
    offset = 0

    for group, parameters in zip(plan.groups, fitted.groups):
        pool = group.source_indices
        selected = rng.choice(pool, size=group.n_cells, replace=False)
        all_sources.append(selected)
        perturbation = 1.0 - strength
        noise = rng.normal(0.0, config.sigma_lib * perturbation, size=group.n_cells)
        targets = source_library[selected] * np.exp(noise)
        all_targets.append(targets)

        neighbor_positions = None
        neighbor_distances = None
        selected_positions = None
        group_X = None
        novelty_factors = None
        if parameters.geometry is not None:
            selected_positions = np.searchsorted(pool, selected).astype(np.int64)
            neighbor_positions, neighbor_distances = query_neighbors(
                parameters.geometry,
                selected_positions,
                seed=config.seed + attempt,
            )
            group_X = reference.X[pool]
            novelty_factors = parameters.geometry.novelty_factors

        z_source = None
        epsilon = None
        rho = 0.0
        if fitted.latent_scores is not None:
            if fitted.latent_covariance is None or fitted.latent_components is None:
                raise RuntimeError("incomplete latent-factor fit")
            z_source = fitted.latent_scores[selected]
            epsilon = rng.multivariate_normal(
                np.zeros(z_source.shape[1]),
                fitted.latent_covariance,
                size=group.n_cells,
                method="svd",
            )
            rho = np.clip(perturbation * config.latent_mix, 0.0, 1.0)

        if z_source is None:
            group_counts = _sample_group_vectorized(
                probabilities=parameters.probabilities,
                dispersion=parameters.dispersion,
                zero_probability=parameters.zero_probability,
                target_library=targets,
                source_rows=reference.X[selected],
                group_X=group_X,
                selected_positions=selected_positions,
                neighbor_positions=neighbor_positions,
                neighbor_distances=neighbor_distances,
                local_profile_mix=config.local_profile_mix,
                group_prior_mix=config.group_prior_mix,
                count_noise_model=config.count_noise_model,
                novelty_strength=config.effective_novelty_strength,
                novelty_factors=novelty_factors,
                use_zinb=config.distribution_model == "hierarchical_zinb",
                rng=rng,
            )
            local_rows, local_cols = group_counts.nonzero()
            row_parts.append(local_rows.astype(np.int64) + offset)
            col_parts.append(local_cols.astype(np.int64))
            data_parts.append(group_counts.data.astype(np.int64, copy=False))
        else:
            generated = [
                _sample_row(
                    probabilities=parameters.probabilities,
                    dispersion=parameters.dispersion,
                    zero_probability=parameters.zero_probability,
                    target_library=float(targets[i]),
                    source_row=reference.X[selected[i]],
                    latent_source=z_source[i],
                    latent_epsilon=epsilon[i],
                    latent_components=fitted.latent_components,
                    rho=rho,
                    beta=config.beta,
                    count_noise_model=config.count_noise_model,
                    use_zinb=config.distribution_model == "hierarchical_zinb",
                    rng=rng,
                )
                for i in range(group.n_cells)
            ]
            for local_row, (cols, values) in enumerate(generated):
                row_parts.append(
                    np.full(cols.size, offset + local_row, dtype=np.int64)
                )
                col_parts.append(cols)
                data_parts.append(values)
        obs = pd.DataFrame(
            {
                column: [value] * group.n_cells
                for column, value in zip(plan.columns, group.values)
            },
            index=range(offset, offset + group.n_cells),
        )
        obs_parts.append(obs)
        offset += group.n_cells

    rows = (
        np.concatenate(row_parts)
        if row_parts
        else np.empty(0, dtype=np.int64)
    )
    cols = (
        np.concatenate(col_parts)
        if col_parts
        else np.empty(0, dtype=np.int64)
    )
    values = (
        np.concatenate(data_parts)
        if data_parts
        else np.empty(0, dtype=np.int64)
    )
    counts = sp.coo_matrix(
        (values, (rows, cols)),
        shape=(plan.total_cells, reference.n_vars),
        dtype=np.int64,
    ).tocsr()
    obs = pd.concat(obs_parts, axis=0) if obs_parts else pd.DataFrame(
        index=range(plan.total_cells)
    )
    return CandidateBatch(
        counts=counts,
        obs=obs,
        target_library_size=np.concatenate(all_targets),
        source_indices=np.concatenate(all_sources),
        attempt=attempt,
    )


def _sample_group_vectorized(
    probabilities: np.ndarray,
    dispersion: np.ndarray,
    zero_probability: np.ndarray,
    target_library: np.ndarray,
    source_rows: sp.csr_matrix,
    group_X: sp.csr_matrix | None,
    selected_positions: np.ndarray | None,
    neighbor_positions: np.ndarray | None,
    neighbor_distances: np.ndarray | None,
    local_profile_mix: float,
    group_prior_mix: float,
    count_noise_model: str,
    novelty_strength: float,
    novelty_factors: np.ndarray | None,
    use_zinb: bool,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    """Sample a group in bounded dense blocks before returning sparse storage."""

    blocks: list[sp.csr_matrix] = []
    block_size = 128
    theta = dispersion[None, :]
    for start in range(0, target_library.size, block_size):
        stop = min(start + block_size, target_library.size)
        if group_X is None:
            block_probabilities = probabilities[None, :]
        else:
            if (
                selected_positions is None
                or neighbor_positions is None
                or neighbor_distances is None
            ):
                raise RuntimeError("incomplete neighbor geometry state")
            block_probabilities = _neighbor_probabilities(
                group_X=group_X,
                source_rows=source_rows[start:stop],
                selected_positions=selected_positions[start:stop],
                neighbor_positions=neighbor_positions[start:stop],
                neighbor_distances=neighbor_distances[start:stop],
                group_probabilities=probabilities,
                local_profile_mix=local_profile_mix,
                group_prior_mix=group_prior_mix,
                novelty_strength=novelty_strength,
                novelty_factors=novelty_factors,
                rng=rng,
            )
        mu = target_library[start:stop, None] * block_probabilities
        probability = np.clip(theta / (theta + mu), 1e-12, 1.0)
        if count_noise_model == "poisson":
            sampled = rng.poisson(mu).astype(np.int64, copy=False)
        else:
            sampled = rng.negative_binomial(
                np.broadcast_to(theta, probability.shape),
                probability,
            ).astype(np.int64, copy=False)
        if use_zinb:
            sampled[
                rng.random(sampled.shape) < zero_probability[None, :]
            ] = 0
        block = sp.csr_matrix(sampled)
        source_block = source_rows[start:stop]
        identical = np.flatnonzero(
            (block != source_block).getnnz(axis=1) == 0
        )
        if identical.size:
            replacement = block.tolil(copy=True)
            for local_row in identical:
                rows, values = _sample_row(
                    probabilities=probabilities,
                    dispersion=dispersion,
                    zero_probability=zero_probability,
                    target_library=float(target_library[start + local_row]),
                    source_row=source_rows[start + local_row],
                    latent_source=None,
                    latent_epsilon=None,
                    latent_components=None,
                    rho=0.0,
                    beta=0.0,
                    count_noise_model=count_noise_model,
                    use_zinb=use_zinb,
                    rng=rng,
                )
                replacement[local_row, :] = 0
                replacement[local_row, rows] = values
            block = replacement.tocsr()
        blocks.append(block)
    return sp.vstack(blocks, format="csr")


def _neighbor_probabilities(
    group_X: sp.csr_matrix,
    source_rows: sp.csr_matrix,
    selected_positions: np.ndarray,
    neighbor_positions: np.ndarray,
    neighbor_distances: np.ndarray,
    group_probabilities: np.ndarray,
    local_profile_mix: float,
    group_prior_mix: float,
    novelty_strength: float,
    novelty_factors: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    n_cells = source_rows.shape[0]
    group_library = np.asarray(group_X.sum(axis=1), dtype=float).ravel()
    group_scale = np.divide(
        1.0,
        group_library,
        out=np.zeros_like(group_library),
        where=group_library > 0,
    )
    context = group_X.multiply(group_scale[:, None]).tocsr()
    source_library = np.asarray(source_rows.sum(axis=1), dtype=float).ravel()
    source_scale = np.divide(
        1.0,
        source_library,
        out=np.zeros_like(source_library),
        where=source_library > 0,
    )
    source_context = source_rows.multiply(source_scale[:, None]).tocsr()

    bandwidth = np.median(neighbor_distances, axis=1)
    bandwidth[bandwidth <= 1e-12] = 1.0
    neighbor_weights = np.exp(
        -0.5 * np.square(neighbor_distances / bandwidth[:, None])
    )
    weight_sum = neighbor_weights.sum(axis=1, keepdims=True)
    neighbor_weights = np.divide(
        neighbor_weights,
        weight_sum,
        out=np.full_like(neighbor_weights, 1.0 / neighbor_weights.shape[1]),
        where=weight_sum > 0,
    )
    row_indices = np.repeat(np.arange(n_cells), neighbor_positions.shape[1])
    selector = sp.csr_matrix(
        (
            neighbor_weights.ravel(),
            (row_indices, neighbor_positions.ravel()),
        ),
        shape=(n_cells, group_X.shape[0]),
    )
    neighbor_context = selector @ context
    local_context = (
        (1.0 - local_profile_mix) * source_context
        + local_profile_mix * neighbor_context
    ).toarray()
    probabilities = (
        (1.0 - group_prior_mix) * local_context
        + group_prior_mix * group_probabilities[None, :]
    )
    if novelty_strength > 0 and novelty_factors is not None:
        latent = rng.standard_normal(
            (n_cells, novelty_factors.shape[1])
        )
        perturbation = np.clip(latent @ novelty_factors.T, -3.0, 3.0)
        novel_profile = group_probabilities[None, :] * np.exp(perturbation)
        novel_profile /= novel_profile.sum(axis=1, keepdims=True)
        probabilities = (
            (1.0 - novelty_strength) * probabilities
            + novelty_strength * novel_profile
        )
    totals = probabilities.sum(axis=1, keepdims=True)
    empty = totals[:, 0] <= 0
    probabilities = np.divide(
        probabilities,
        totals,
        out=np.broadcast_to(group_probabilities, probabilities.shape).copy(),
        where=totals > 0,
    )
    if empty.any():
        probabilities[empty] = group_probabilities
    return probabilities


def _sample_row(
    probabilities: np.ndarray,
    dispersion: np.ndarray,
    zero_probability: np.ndarray,
    target_library: float,
    source_row: sp.csr_matrix,
    latent_source: np.ndarray | None,
    latent_epsilon: np.ndarray | None,
    latent_components: np.ndarray | None,
    rho: float,
    beta: float,
    count_noise_model: str,
    use_zinb: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(10):
        weights = probabilities
        if latent_source is not None:
            if latent_components is None or latent_epsilon is None:
                raise RuntimeError("incomplete latent-factor state")
            z = (
                np.sqrt(max(1.0 - rho, 0.0)) * latent_source
                + np.sqrt(rho) * latent_epsilon
            )
            log_factor = np.clip(beta * (latent_components @ z), -8.0, 8.0)
            weights = probabilities * np.exp(log_factor)
        weights = weights / weights.sum()
        mu = target_library * weights
        nonzero_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        for start in range(0, probabilities.size, 2048):
            stop = min(start + 2048, probabilities.size)
            mu_chunk = mu[start:stop]
            theta = dispersion[start:stop]
            p = theta / (theta + mu_chunk)
            p = np.clip(p, 1e-12, 1.0)
            if count_noise_model == "poisson":
                sampled = rng.poisson(mu_chunk).astype(np.int64, copy=False)
            else:
                sampled = rng.negative_binomial(theta, p).astype(
                    np.int64, copy=False
                )
            if use_zinb:
                sampled[rng.random(mu_chunk.size) < zero_probability[start:stop]] = 0
            nonzero = np.flatnonzero(sampled)
            if nonzero.size:
                nonzero_parts.append(nonzero + start)
                value_parts.append(sampled[nonzero])
        if nonzero_parts:
            cols = np.concatenate(nonzero_parts)
            values = np.concatenate(value_parts)
        else:
            cols = np.empty(0, dtype=np.int64)
            values = np.empty(0, dtype=np.int64)
        candidate_row = sp.csr_matrix(
            (values, (np.zeros(cols.size), cols)), shape=(1, probabilities.size)
        )
        if (candidate_row != source_row).nnz > 0:
            return cols.astype(np.int64), values.astype(np.int64, copy=False)
    return cols.astype(np.int64), values.astype(np.int64, copy=False)
