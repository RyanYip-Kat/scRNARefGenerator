"""Validated public configuration objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Category = Any
ConditionKey = tuple[Category, ...]


@dataclass
class ConditionSpec:
    """Requested exact joint or marginal condition quotas."""

    columns: list[str]
    joint: dict[ConditionKey, int] | None = None
    marginal: dict[str, dict[Category, int]] | None = None

    def __post_init__(self) -> None:
        if not self.columns:
            raise ValueError("columns must contain at least one condition column")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("condition columns must be unique")
        if (self.joint is None) == (self.marginal is None):
            raise ValueError("provide exactly one of joint or marginal")

        if self.joint is not None:
            self.joint = dict(self.joint)
            for key, quota in self.joint.items():
                if not isinstance(key, tuple) or len(key) != len(self.columns):
                    raise ValueError("every joint key must match the condition columns")
                _validate_quota(quota)
        else:
            self.marginal = {
                column: dict(categories)
                for column, categories in (self.marginal or {}).items()
            }
            if set(self.marginal) != set(self.columns):
                raise ValueError("marginal keys must match the condition columns")
            for categories in self.marginal.values():
                if not categories:
                    raise ValueError("every marginal column needs a quota")
                for quota in categories.values():
                    _validate_quota(quota)


def _validate_quota(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("all condition quotas must be positive integers")


@dataclass
class EvaluationConfig:
    """Thresholds and sampling controls for quality evaluation."""

    mean_spearman_min: float = 0.80
    zero_spearman_min: float = 0.70
    library_ks_max: float = 0.20
    nn_ratio_range: tuple[float, float] = (0.50, 2.00)
    auc_range: tuple[float, float] = (0.55, 0.90)
    high_variable_min: float = 0.75
    matched_cells: int = 1000
    high_variable_genes: int = 200

    def __post_init__(self) -> None:
        _check_fraction(self.mean_spearman_min, "mean_spearman_min")
        _check_fraction(self.zero_spearman_min, "zero_spearman_min")
        _check_fraction(self.library_ks_max, "library_ks_max")
        _check_fraction(self.high_variable_min, "high_variable_min")
        _check_positive_range(self.nn_ratio_range, "nn_ratio_range")
        _check_unit_range(self.auc_range, "auc_range")
        if self.matched_cells < 10:
            raise ValueError("matched_cells must be at least 10")
        if self.high_variable_genes < 2:
            raise ValueError("high_variable_genes must be at least 2")


def _check_fraction(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def _check_positive_range(values: tuple[float, float], name: str) -> None:
    if len(values) != 2 or not 0.0 <= values[0] < values[1]:
        raise ValueError(f"{name} must be an increasing nonnegative range")


def _check_unit_range(values: tuple[float, float], name: str) -> None:
    if len(values) != 2 or not 0.0 <= values[0] < values[1] <= 1.0:
        raise ValueError(f"{name} must be an increasing range inside [0, 1]")


@dataclass
class GeneratorConfig:
    """End-to-end simulation configuration."""

    counts_layer: str = "counts"
    condition_cols: list[str] | None = None
    barcode_suffix: str = "_sim"
    seed: int = 42
    similarity_strength: float = 0.70
    distribution_model: Literal[
        "hierarchical_nb", "hierarchical_zinb"
    ] = "hierarchical_nb"
    geometry_model: Literal["none", "neighbor_preserving"] = (
        "neighbor_preserving"
    )
    geometry_neighbors: int = 30
    local_profile_mix: float = 0.50
    group_prior_mix: float = 0.10
    geometry_latent_k: int = 20
    geometry_hvg_count: int = 2000
    geometry_eval_holdout_fraction: float = 0.20
    count_noise_model: Literal["poisson", "nb"] = "poisson"
    novelty_mode: Literal["fidelity", "balanced", "novel"] = "balanced"
    novelty_strength: float | None = None
    latent_k: int = 0
    prior_strength: float = 10.0
    sigma_lib: float = 0.15
    latent_mix: float = 0.50
    beta: float = 0.20
    dispersion_n0: float = 50.0
    na_policy: Literal["error", "drop"] = "error"
    collision_policy: Literal["error", "append_uuid"] = "error"
    max_retries: int = 3
    max_cells_for_fitting: int | None = 10000
    strict_evaluation: bool = True
    overwrite: bool = False
    source_metadata: bool = True
    inherit_obs_cols: list[str] | None = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not 0.0 <= self.similarity_strength <= 1.0:
            raise ValueError("similarity_strength must be between 0 and 1")
        if self.latent_k < 0:
            raise ValueError("latent_k must be nonnegative")
        if self.geometry_model not in {"none", "neighbor_preserving"}:
            raise ValueError("unknown geometry_model")
        if self.geometry_neighbors < 1:
            raise ValueError("geometry_neighbors must be at least 1")
        if not 0.0 <= self.local_profile_mix <= 1.0:
            raise ValueError("local_profile_mix must be between 0 and 1")
        if not 0.0 <= self.group_prior_mix <= 1.0:
            raise ValueError("group_prior_mix must be between 0 and 1")
        if self.geometry_latent_k < 2:
            raise ValueError("geometry_latent_k must be at least 2")
        if self.geometry_hvg_count < 2:
            raise ValueError("geometry_hvg_count must be at least 2")
        if not 0.0 < self.geometry_eval_holdout_fraction < 1.0:
            raise ValueError(
                "geometry_eval_holdout_fraction must be between 0 and 1"
            )
        if self.count_noise_model not in {"poisson", "nb"}:
            raise ValueError("count_noise_model must be 'poisson' or 'nb'")
        if self.novelty_mode not in {"fidelity", "balanced", "novel"}:
            raise ValueError("novelty_mode must be fidelity, balanced, or novel")
        if self.novelty_strength is not None and not (
            0.0 <= self.novelty_strength <= 1.0
        ):
            raise ValueError("novelty_strength must be between 0 and 1")
        if self.prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        if self.sigma_lib < 0:
            raise ValueError("sigma_lib must be nonnegative")
        if not 0.0 <= self.latent_mix <= 1.0:
            raise ValueError("latent_mix must be between 0 and 1")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")
        if self.dispersion_n0 <= 0:
            raise ValueError("dispersion_n0 must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        if self.max_cells_for_fitting is not None and self.max_cells_for_fitting < 2:
            raise ValueError("max_cells_for_fitting must be at least 2")
        if not self.barcode_suffix:
            raise ValueError("barcode_suffix must not be empty")
        if self.inherit_obs_cols is not None:
            if not self.inherit_obs_cols:
                raise ValueError("inherit_obs_cols must not be empty when provided")
            if len(set(self.inherit_obs_cols)) != len(self.inherit_obs_cols):
                raise ValueError("inherit_obs_cols must be unique")

    @property
    def effective_novelty_strength(self) -> float:
        if self.novelty_strength is not None:
            return float(self.novelty_strength)
        return {
            "fidelity": 0.0,
            "balanced": 0.25,
            "novel": 0.50,
        }[self.novelty_mode]
