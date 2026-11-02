import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

from scrnarefgenerator import export_raw_counts


def _annotated():
    counts = np.array(
        [[1, 0, 4], [0, 2, 0], [3, 1, 5], [0, 0, 2]],
        dtype=np.int64,
    )
    adata = ad.AnnData(
        sp.csr_matrix(np.zeros_like(counts)),
        obs=pd.DataFrame(
            {"celltype": ["T", "B", "T", "B"], "ident": [1, 2, 1, 2]},
            index=[f"cell{i}" for i in range(4)],
        ),
        var=pd.DataFrame(
            {"marker": ["a", "b", "c"]},
            index=["g1", "g2", "g3"],
        ),
    )
    adata.layers["counts"] = sp.csr_matrix(counts)
    return adata, counts


def test_mtx_export_preserves_raw_counts_and_metadata_order(tmp_path):
    adata, counts = _annotated()

    artifacts = export_raw_counts(
        adata,
        tmp_path,
        prefix="sim",
        matrix_format="mtx",
    )

    matrix = mmread(artifacts.matrix).tocsc()
    obs = pd.read_csv(artifacts.obs, index_col=0)
    var = pd.read_csv(artifacts.var, index_col=0)
    assert matrix.shape == (adata.n_vars, adata.n_obs)
    assert np.array_equal(matrix.toarray().T, counts)
    assert obs.index.tolist() == adata.obs_names.tolist()
    assert var.index.tolist() == adata.var_names.tolist()


def test_npz_export_uses_cells_by_genes_sparse_storage(tmp_path):
    adata, counts = _annotated()

    artifacts = export_raw_counts(
        adata,
        tmp_path,
        prefix="sim",
        matrix_format="npz",
    )

    matrix = sp.load_npz(artifacts.matrix)
    assert matrix.shape == counts.shape
    assert np.array_equal(matrix.toarray(), counts)
