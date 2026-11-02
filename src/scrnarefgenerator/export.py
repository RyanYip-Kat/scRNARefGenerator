"""Export validated raw counts and aligned AnnData metadata tables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
import tempfile

import anndata as ad
import scipy.io
import scipy.sparse as sp

from .config import GeneratorConfig
from .errors import OutputExistsError
from .validation import ReferenceData, resolve_reference


@dataclass(frozen=True)
class ExportArtifacts:
    matrix: Path
    obs: Path
    var: Path


def export_raw_counts(
    data: ReferenceData | ad.AnnData | str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    prefix: str = "counts",
    matrix_format: str = "mtx",
    overwrite: bool = False,
) -> ExportArtifacts:
    """Export integer raw counts with row-aligned cell and gene metadata.

    ``mtx`` stores genes-by-cells Matrix Market data, following common scRNA
    interchange conventions. ``npz`` stores cells-by-genes CSR arrays.
    """

    if matrix_format not in {"mtx", "npz"}:
        raise ValueError("matrix_format must be 'mtx' or 'npz'")
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a nonempty filename prefix")
    reference = (
        data
        if isinstance(data, ReferenceData)
        else resolve_reference(data, GeneratorConfig())
    )
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / f"{prefix}_matrix.{matrix_format}"
    obs_path = directory / f"{prefix}_obs.csv"
    var_path = directory / f"{prefix}_var.csv"
    paths = (matrix_path, obs_path, var_path)
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise OutputExistsError(
                f"export path already exists: {existing[0]}; set overwrite=True"
            )

    sparse = reference.X
    matrix = sparse.T.tocoo() if matrix_format == "mtx" else sparse
    _write_sparse_atomic(matrix, matrix_path, matrix_format)
    obs = reference.obs.copy()
    obs.index.name = "cell_id"
    obs.to_csv(obs_path, index=True)
    var = reference.var.copy()
    var.index.name = "gene_id"
    var.to_csv(var_path, index=True)
    return ExportArtifacts(matrix=matrix_path, obs=obs_path, var=var_path)


def _write_sparse_atomic(matrix, path: Path, matrix_format: str) -> None:
    handle, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=f".{matrix_format}",
    )
    os.close(handle)
    try:
        if matrix_format == "mtx":
            scipy.io.mmwrite(temporary, matrix, field="integer")
        else:
            sp.save_npz(temporary, matrix.tocsr())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
