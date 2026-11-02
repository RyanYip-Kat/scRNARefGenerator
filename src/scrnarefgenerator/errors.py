"""Package-specific exceptions."""


class ScRNARefGeneratorError(Exception):
    """Base class for package errors."""


class CountsNotFoundError(ScRNARefGeneratorError):
    """No valid raw-count matrix could be found."""


class InvalidCountsError(ScRNARefGeneratorError):
    """A candidate count matrix violates the raw-count contract."""


class GeneAlignmentError(ScRNARefGeneratorError):
    """A matrix has genes that do not align with the reference AnnData."""


class DuplicateObsNamesError(ScRNARefGeneratorError):
    """Observation names are not unique."""


class DuplicateVarNamesError(ScRNARefGeneratorError):
    """Variable names are not unique."""


class ConditionColumnNotFoundError(ScRNARefGeneratorError):
    """A requested condition column is absent."""


class ConditionCategoryNotFoundError(ScRNARefGeneratorError):
    """A requested condition category is absent."""


class InfeasibleConditionError(ScRNARefGeneratorError):
    """A requested condition plan cannot be met exactly."""


class BarcodeSuffixCollisionError(ScRNARefGeneratorError):
    """Generated barcode names collide with existing names."""


class EvaluationGateError(ScRNARefGeneratorError):
    """No generated candidate satisfies the evaluation gate."""


class OutputExistsError(ScRNARefGeneratorError):
    """An output path already exists and overwrite is disabled."""
