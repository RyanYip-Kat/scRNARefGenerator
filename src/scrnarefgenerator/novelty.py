"""Source-dependence and exact-copy diagnostics for synthetic cells."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import anndata as ad
import numpy as np

from .config import GeneratorConfig
from .errors import ConditionColumnNotFoundError
from .validation import ReferenceData, resolve_reference


@dataclass(frozen=True)
class NoveltyReport:
    metrics: dict[str, float]
    n_evaluated: int

    def as_dict(self) -> dict[str, object]:
        return {"metrics": self.metrics, "n_evaluated": self.n_evaluated}


def evaluate_novelty(
    simulated: ad.AnnData | str | PathLike[str],
    source_reference: ReferenceData | ad.AnnData | str | PathLike[str],
    condition_col: str,
    sample_size: int = 2000,
    seed: int = 42,
) -> NoveltyReport:
    """Measure exact copying and source-profile dependence of simulations."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    simulation = resolve_reference(simulated, GeneratorConfig())
    reference = (
        source_reference
        if isinstance(source_reference, ReferenceData)
        else resolve_reference(source_reference, GeneratorConfig())
    )
    for label, data in (("simulated", simulation), ("reference", reference)):
        if condition_col not in data.obs:
            raise ConditionColumnNotFoundError(
                f"condition column {condition_col!r} is absent from {label}"
            )
    if "source_barcode" not in simulation.obs:
        raise ValueError("simulated.obs must contain source_barcode")

    names = reference.obs_names.astype(str)
    name_to_index = {name: index for index, name in enumerate(names)}
    categories = sorted(simulation.obs[condition_col].dropna().unique(), key=repr)
    per_category = max(1, sample_size // max(1, len(categories)))
    rng = np.random.default_rng(seed)
    exact_copies: list[bool] = []
    source_jaccard: list[float] = []
    random_jaccard: list[float] = []
    library_ratio: list[float] = []

    for category in categories:
        simulation_indices = np.flatnonzero(
            simulation.obs[condition_col].eq(category).to_numpy()
        )
        selected = rng.choice(
            simulation_indices,
            size=min(per_category, simulation_indices.size),
            replace=False,
        )
        reference_pool = np.flatnonzero(
            reference.obs[condition_col].eq(category).to_numpy()
        )
        if reference_pool.size < 2:
            continue
        for simulation_index in selected:
            source_name = str(
                simulation.obs["source_barcode"].iloc[simulation_index]
            )
            if source_name not in name_to_index:
                raise ValueError(
                    f"source barcode {source_name!r} is absent from source_reference"
                )
            source_index = name_to_index[source_name]
            source_row = reference.X[source_index]
            simulated_row = simulation.X[simulation_index]
            alternatives = reference_pool[reference_pool != source_index]
            random_index = alternatives[rng.integers(alternatives.size)]
            random_row = reference.X[random_index]
            exact_copies.append((simulated_row != source_row).nnz == 0)
            source_jaccard.append(_jaccard(simulated_row, source_row))
            random_jaccard.append(_jaccard(simulated_row, random_row))
            source_library = float(source_row.sum())
            simulated_library = float(simulated_row.sum())
            library_ratio.append(
                simulated_library / source_library if source_library > 0 else 0.0
            )

    if not exact_copies:
        raise ValueError("no simulated cells could be compared with source cells")
    source_median = float(np.median(source_jaccard))
    random_median = float(np.median(random_jaccard))
    gain = source_median / random_median if random_median > 0 else float("inf")
    return NoveltyReport(
        metrics={
            "exact_source_copy_fraction": float(np.mean(exact_copies)),
            "source_jaccard_median": source_median,
            "random_jaccard_median": random_median,
            "source_jaccard_gain": float(gain),
            "source_library_ratio_median": float(np.median(library_ratio)),
        },
        n_evaluated=len(exact_copies),
    )


def _jaccard(left, right) -> float:
    intersection = left.multiply(right > 0).nnz
    union = left.nnz + right.nnz - intersection
    return float(intersection / union) if union else 1.0
