"""Conditional scRNA-seq reference simulation."""

from .api import scRNARefGenerator
from .conditions import (
    condition_spec_from_marginal_fraction,
    condition_spec_from_reference,
)
from .config import ConditionSpec, EvaluationConfig, GeneratorConfig
from .errors import (
    BarcodeSuffixCollisionError,
    ConditionCategoryNotFoundError,
    ConditionColumnNotFoundError,
    CountsNotFoundError,
    DuplicateObsNamesError,
    DuplicateVarNamesError,
    EvaluationGateError,
    GeneAlignmentError,
    InfeasibleConditionError,
    InvalidCountsError,
    OutputExistsError,
    ScRNARefGeneratorError,
)
from .evaluation import SimilarityReport, evaluate_similarity
from .embedding import EmbeddingResult, compute_pca_umap
from .export import ExportArtifacts, export_raw_counts
from .novelty import NoveltyReport, evaluate_novelty
from .splitting import ReferenceSplit, stratified_reference_split

__all__ = [
    "BarcodeSuffixCollisionError",
    "ConditionCategoryNotFoundError",
    "ConditionColumnNotFoundError",
    "ConditionSpec",
    "CountsNotFoundError",
    "DuplicateObsNamesError",
    "DuplicateVarNamesError",
    "EvaluationConfig",
    "EvaluationGateError",
    "EmbeddingResult",
    "ExportArtifacts",
    "GeneAlignmentError",
    "GeneratorConfig",
    "InfeasibleConditionError",
    "InvalidCountsError",
    "OutputExistsError",
    "NoveltyReport",
    "ReferenceSplit",
    "ScRNARefGeneratorError",
    "SimilarityReport",
    "condition_spec_from_reference",
    "condition_spec_from_marginal_fraction",
    "compute_pca_umap",
    "evaluate_similarity",
    "evaluate_novelty",
    "export_raw_counts",
    "scRNARefGenerator",
    "stratified_reference_split",
]

__version__ = "0.1.0"
