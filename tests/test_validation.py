import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from scrnarefgenerator.config import GeneratorConfig
from scrnarefgenerator.errors import (
    CountsNotFoundError,
    GeneAlignmentError,
    InvalidCountsError,
)
from scrnarefgenerator.validation import resolve_reference


def _adata(matrix, obs_names=("a", "b"), var_names=("g1", "g2")):
    result = ad.AnnData(sp.csr_matrix(matrix))
    result.obs_names = list(obs_names)
    result.var_names = list(var_names)
    return result


def test_resolves_integral_counts_layer_without_densifying():
    adata = _adata([[0.0, 2.0], [3.0, 0.0]])
    adata.layers["counts"] = adata.X.copy()

    got = resolve_reference(adata, GeneratorConfig())

    assert sp.issparse(got.X)
    assert got.X.dtype.kind in "iu"
    assert got.X.toarray().tolist() == [[0, 2], [3, 0]]


def test_prefers_counts_layer_over_integer_x():
    adata = _adata([[9, 9], [9, 9]])
    adata.layers["counts"] = sp.csr_matrix([[0, 2], [3, 0]])

    got = resolve_reference(adata, GeneratorConfig())

    assert got.X.toarray().tolist() == [[0, 2], [3, 0]]


def test_rejects_nonintegral_counts():
    adata = _adata([[0.5, 1.0], [1.0, 0.0]])

    with pytest.raises(InvalidCountsError):
        resolve_reference(adata, GeneratorConfig())


def test_raises_when_no_count_matrix_can_be_found():
    adata = ad.AnnData(None, obs={"cell": ["a", "b"]}, var={"gene": ["g1", "g2"]})

    with pytest.raises(CountsNotFoundError):
        resolve_reference(adata, GeneratorConfig())


def test_rejects_misaligned_raw_genes():
    adata = _adata([[1, 2], [3, 4]])
    raw = ad.AnnData(sp.csr_matrix([[1, 2], [3, 4]]), var={"gene": ["g1", "other"]})
    raw.obs_names = adata.obs_names
    adata.raw = raw
    adata.X = np.asarray([[0.1, 1.0], [1.0, 0.0]])

    with pytest.raises(GeneAlignmentError):
        resolve_reference(adata, GeneratorConfig())


def test_rejects_nonfinite_sparse_values():
    adata = _adata([[0.0, 1.0], [1.0, 0.0]])
    adata.X.data[0] = np.nan

    with pytest.raises(InvalidCountsError):
        resolve_reference(adata, GeneratorConfig())
