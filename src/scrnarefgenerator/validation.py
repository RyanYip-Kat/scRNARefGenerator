"""Reference loading and raw-count validation."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import GeneratorConfig
from .errors import (
    CountsNotFoundError,
    DuplicateObsNamesError,
    DuplicateVarNamesError,
    GeneAlignmentError,
    InvalidCountsError,
)


@dataclass(frozen=True)
class ReferenceData:
    """Validated sparse-count reference used by all internal stages."""

    X: sp.csr_matrix
    obs: pd.DataFrame
    var: pd.DataFrame
    obs_names: pd.Index
    var_names: pd.Index
    path: str | None = None

    @property
    def n_obs(self) -> int:
        return self.X.shape[0]

    @property
    def n_vars(self) -> int:
        return self.X.shape[1]


def resolve_reference(
    ref: ad.AnnData | str | PathLike[str],
    config: GeneratorConfig | None = None,
) -> ReferenceData:
    """Read and validate a reference count matrix without densifying it."""

    config = config or GeneratorConfig()
    source_path: str | None = None
    if isinstance(ref, (str, PathLike)):
        source_path = str(Path(ref).expanduser().resolve())
        adata = ad.read_h5ad(source_path)
    elif isinstance(ref, ad.AnnData):
        adata = ref
    else:
        raise TypeError("ref must be an AnnData object or an .h5ad path")

    if not adata.obs_names.is_unique:
        raise DuplicateObsNamesError("reference obs_names must be unique")
    if not adata.var_names.is_unique:
        raise DuplicateVarNamesError("reference var_names must be unique")

    matrix: Any = None
    matrix_kind = ""
    if config.counts_layer in adata.layers:
        matrix = adata.layers[config.counts_layer]
        matrix_kind = config.counts_layer
    elif adata.raw is not None and _is_integral(matrix=adata.raw.X):
        if not adata.raw.var_names.equals(adata.var_names):
            raise GeneAlignmentError(
                "raw.var_names must exactly match adata.var_names and order"
            )
        matrix = adata.raw.X
        matrix_kind = "raw.X"
    elif adata.X is not None:
        matrix = adata.X
        matrix_kind = "X"
    else:
        raise CountsNotFoundError(
            "no nonnegative integer raw counts were found in layers, raw.X, or X"
        )

    if matrix.shape != adata.shape:
        raise GeneAlignmentError(
            f"{matrix_kind} shape {matrix.shape} does not match AnnData {adata.shape}"
        )
    sparse = _as_csr(matrix)
    sparse.sum_duplicates()
    sparse.eliminate_zeros()
    data = sparse.data
    if not np.all(np.isfinite(data)):
        raise InvalidCountsError(f"{matrix_kind} contains NaN or infinite values")
    if data.size and float(data.min()) < 0:
        raise InvalidCountsError(f"{matrix_kind} contains negative counts")
    rounded = np.rint(data)
    if data.size and not np.all(np.abs(data - rounded) < 1e-6):
        raise InvalidCountsError(
            f"{matrix_kind} is not an integer raw-count matrix within 1e-6"
        )
    sparse.data = rounded.astype(np.int64, copy=False)

    return ReferenceData(
        X=sparse,
        obs=adata.obs,
        var=adata.var,
        obs_names=adata.obs_names,
        var_names=adata.var_names,
        path=source_path,
    )


def _as_csr(matrix: Any) -> sp.csr_matrix:
    if sp.issparse(matrix):
        return matrix.tocsr(copy=True)
    array = np.asarray(matrix)
    if not np.all(np.isfinite(array)):
        raise InvalidCountsError("count matrix contains NaN or infinite values")
    if array.size and float(np.min(array)) < 0:
        raise InvalidCountsError("count matrix contains negative values")
    return sp.csr_matrix(array)


def _is_integral(matrix: Any) -> bool:
    if matrix is None:
        return False
    if sp.issparse(matrix):
        data = matrix.data
    else:
        data = np.asarray(matrix).ravel()
    if not np.all(np.isfinite(data)):
        return False
    if data.size and float(np.min(data)) < 0:
        return False
    return bool(np.all(np.abs(data - np.rint(data)) < 1e-6))
