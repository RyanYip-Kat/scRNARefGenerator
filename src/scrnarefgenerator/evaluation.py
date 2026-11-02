"""Reference-versus-simulation quality metrics and acceptance gate."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import anndata as ad
import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors

from .config import EvaluationConfig, GeneratorConfig
from .errors import ConditionCategoryNotFoundError, ConditionColumnNotFoundError
from .simulate import CandidateBatch
from .validation import ReferenceData, resolve_reference


@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    metrics: dict[str, float]
    violations: list[str]


@dataclass(frozen=True)
class SimilarityReport:
    """Overall and per-condition similarity between reference and simulation."""

    condition_col: str | None
    overall: EvaluationResult
    groups: dict[str, EvaluationResult]

    @property
    def passed(self) -> bool:
        return self.overall.passed and all(
            result.passed for result in self.groups.values()
        )

    @property
    def violations(self) -> dict[str, list[str]]:
        violations = {"overall": list(self.overall.violations)}
        violations.update(
            {
                group: list(result.violations)
                for group, result in self.groups.items()
                if result.violations
            }
        )
        return violations

    def as_dict(self) -> dict[str, object]:
        return {
            "condition_col": self.condition_col,
            "passed": self.passed,
            "violations": self.violations,
            "overall": {
                "passed": self.overall.passed,
                "metrics": self.overall.metrics,
                "violations": self.overall.violations,
            },
            "groups": {
                group: {
                    "passed": result.passed,
                    "metrics": result.metrics,
                    "violations": result.violations,
                }
                for group, result in self.groups.items()
            },
        }


def evaluate_similarity(
    reference: ReferenceData | ad.AnnData | str | PathLike[str],
    simulated: ad.AnnData | str | PathLike[str],
    condition_col: str | None = None,
    config: EvaluationConfig | None = None,
    counts_layer: str = "counts",
) -> SimilarityReport:
    """Evaluate a generated AnnData against a reference, optionally by condition."""

    evaluation = config or EvaluationConfig()
    reference_data = (
        reference
        if isinstance(reference, ReferenceData)
        else resolve_reference(reference, GeneratorConfig(counts_layer=counts_layer))
    )
    simulated_data = resolve_reference(
        simulated, GeneratorConfig(counts_layer=counts_layer)
    )
    overall = _evaluate_matrices(
        reference_data.X,
        simulated_data.X,
        evaluation,
        seed=0,
    )
    groups: dict[str, EvaluationResult] = {}
    if condition_col is not None:
        for label, data in (
            ("reference", reference_data),
            ("simulated", simulated_data),
        ):
            if condition_col not in data.obs:
                raise ConditionColumnNotFoundError(
                    f"condition column {condition_col!r} is absent from {label}"
                )
        simulated_categories = sorted(
            simulated_data.obs[condition_col].dropna().unique(),
            key=repr,
        )
        for group_index, category in enumerate(simulated_categories):
            reference_group = reference_data.obs[condition_col].eq(category)
            simulation_group = simulated_data.obs[condition_col].eq(category)
            if not reference_group.any():
                raise ConditionCategoryNotFoundError(
                    f"simulated category {category!r} is absent from reference"
                )
            groups[str(category)] = _evaluate_matrices(
                reference_data.X[reference_group.to_numpy()],
                simulated_data.X[simulation_group.to_numpy()],
                evaluation,
                seed=101 * (group_index + 1),
            )
    return SimilarityReport(
        condition_col=condition_col,
        overall=overall,
        groups=groups,
    )


def evaluate_candidate(
    reference: ReferenceData,
    candidate: CandidateBatch,
    config: EvaluationConfig | GeneratorConfig,
) -> EvaluationResult:
    """Calculate matched-sample quality metrics and evaluate all thresholds."""

    evaluation = config.evaluation if isinstance(config, GeneratorConfig) else config
    return _evaluate_matrices(reference.X, candidate.counts, evaluation, seed=0)


def _evaluate_matrices(
    reference_X,
    simulation_X,
    evaluation: EvaluationConfig,
    seed: int,
) -> EvaluationResult:
    rng = np.random.default_rng(seed)
    n_matched = min(
        reference_X.shape[0],
        simulation_X.shape[0],
        evaluation.matched_cells,
    )
    reference_indices = rng.choice(
        reference_X.shape[0], size=n_matched, replace=False
    )
    simulation_indices = rng.choice(
        simulation_X.shape[0], size=n_matched, replace=False
    )
    reference_X = reference_X[reference_indices]
    simulation_X = simulation_X[simulation_indices]

    reference_normalized = _log_normalize(reference_X)
    simulation_normalized = _log_normalize(simulation_X)
    reference_mean = np.asarray(reference_normalized.mean(axis=0)).ravel()
    simulation_mean = np.asarray(simulation_normalized.mean(axis=0)).ravel()
    reference_zero = 1.0 - reference_X.getnnz(axis=0) / reference_X.shape[0]
    simulation_zero = 1.0 - simulation_X.getnnz(axis=0) / simulation_X.shape[0]
    reference_library = np.asarray(reference_X.sum(axis=1)).ravel()
    simulation_library = np.asarray(simulation_X.sum(axis=1)).ravel()

    reference_variance = np.asarray(
        reference_normalized.multiply(reference_normalized).mean(axis=0)
    ).ravel() - np.square(reference_mean)
    n_hvg = min(evaluation.high_variable_genes, reference_X.shape[1])
    high_variable = np.argsort(reference_variance)[-n_hvg:]

    embedding = _embedding(reference_normalized, simulation_normalized)
    reference_embedding = embedding[: reference_X.shape[0]]
    simulation_embedding = embedding[reference_X.shape[0] :]
    metrics = {
        "mean_spearman": _safe_spearman(reference_mean, simulation_mean),
        "zero_spearman": _safe_spearman(reference_zero, simulation_zero),
        "library_ks": float(ks_2samp(reference_library, simulation_library).statistic),
        "nn_ratio": _nn_ratio(reference_embedding, simulation_embedding),
        "classifier_auc": _classifier_auc(reference_embedding, simulation_embedding),
        "high_variable_spearman": _safe_spearman(
            reference_mean[high_variable], simulation_mean[high_variable]
        ),
    }

    violations: list[str] = []
    if metrics["mean_spearman"] < evaluation.mean_spearman_min:
        violations.append("mean_spearman")
    if metrics["zero_spearman"] < evaluation.zero_spearman_min:
        violations.append("zero_spearman")
    if metrics["library_ks"] > evaluation.library_ks_max:
        violations.append("library_ks")
    if not evaluation.nn_ratio_range[0] <= metrics["nn_ratio"] <= evaluation.nn_ratio_range[1]:
        violations.append("nn_ratio")
    if not evaluation.auc_range[0] <= metrics["classifier_auc"] <= evaluation.auc_range[1]:
        violations.append("classifier_auc")
    if metrics["high_variable_spearman"] < evaluation.high_variable_min:
        violations.append("high_variable_spearman")
    return EvaluationResult(
        passed=not violations,
        metrics=metrics,
        violations=violations,
    )


def _log_normalize(X) -> object:
    library = np.asarray(X.sum(axis=1), dtype=float).ravel()
    positive = library[library > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    scale = np.divide(
        median,
        library,
        out=np.zeros_like(library),
        where=library > 0,
    )
    result = X.multiply(scale[:, None]).tocsr()
    result.data = np.log1p(result.data)
    return result


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else 0.0


def _embedding(reference, simulation) -> np.ndarray:
    from scipy.sparse import vstack

    combined = vstack([reference, simulation], format="csr")
    n_components = min(20, combined.shape[0] - 1, combined.shape[1] - 1)
    if n_components < 1:
        return np.zeros((combined.shape[0], 1))
    return TruncatedSVD(n_components=n_components, random_state=0).fit_transform(
        combined
    )


def _nn_ratio(reference: np.ndarray, simulation: np.ndarray) -> float:
    if reference.shape[0] < 2:
        return float("inf")
    reference_neighbors = NearestNeighbors(n_neighbors=2).fit(reference)
    reference_distance = reference_neighbors.kneighbors(reference)[0][:, 1]
    simulation_distance = NearestNeighbors(n_neighbors=1).fit(reference).kneighbors(
        simulation
    )[0][:, 0]
    denominator = float(np.median(reference_distance))
    if denominator <= 0:
        return 0.0
    return float(np.median(simulation_distance) / denominator)


def _classifier_auc(reference: np.ndarray, simulation: np.ndarray) -> float:
    X = np.vstack([reference, simulation])
    y = np.concatenate(
        [np.zeros(reference.shape[0]), np.ones(simulation.shape[0])]
    )
    minimum_class = int(min(np.bincount(y.astype(int))))
    n_splits = min(5, minimum_class)
    if n_splits < 2:
        return 0.5
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    classifier = LogisticRegression(max_iter=2000, random_state=0)
    probabilities = cross_val_predict(
        classifier, X, y, cv=folds, method="predict_proba"
    )[:, 1]
    value = float(roc_auc_score(y, probabilities))
    return value if np.isfinite(value) else 0.5
