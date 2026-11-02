import importlib.util
import os
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


ROOT = Path(__file__).parents[1]
os.environ["NUMBA_CACHE_DIR"] = str(
    Path(tempfile.gettempdir()) / "scrnarefgenerator-numba"
)
os.environ["MPLCONFIGDIR"] = str(
    Path(tempfile.gettempdir()) / "scrnarefgenerator-matplotlib"
)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_sim_biological",
    ROOT / "evaluate_sim_biological.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_integer_count_detection_does_not_depend_on_magnitude():
    small_counts = sp.csr_matrix([[0, 1], [2, 0]])
    log_expression = sp.csr_matrix([[0.0, 0.1], [1.2, 0.0]])

    assert MODULE.is_integer_count_matrix(small_counts)
    assert not MODULE.is_integer_count_matrix(log_expression)


def test_simulation_celltype_falls_back_to_source_celltype():
    simulated = ad.AnnData(
        sp.csr_matrix([[1, 0]]),
        obs=pd.DataFrame(
            {"source_celltype": ["T"]},
            index=["sim0"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    assert MODULE.infer_simulation_celltype_key(simulated, "celltype") == "source_celltype"


def test_holdout_reference_excludes_source_barcodes():
    real = ad.AnnData(
        sp.csr_matrix([[1, 0], [0, 1], [2, 0], [0, 2]]),
        obs=pd.DataFrame(
            {"celltype": ["T", "T", "B", "B"]},
            index=["a", "b", "c", "d"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    simulated = ad.AnnData(
        sp.csr_matrix([[1, 0], [0, 1]]),
        obs=pd.DataFrame(
            {"source_barcode": ["a", "b"]},
            index=["sim0", "sim1"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )

    holdout = MODULE.exclude_source_cells(real, simulated)

    assert holdout.obs_names.tolist() == ["c", "d"]


def test_stratified_subsample_balances_each_class():
    labels = np.array(["T"] * 5 + ["B"] * 3 + ["NK"] * 4)

    indices = MODULE.stratified_subsample_indices(
        labels,
        maximum_per_class=2,
        seed=7,
    )

    assert pd.Series(labels[indices]).value_counts().to_dict() == {
        "T": 2,
        "B": 2,
        "NK": 2,
    }


def test_hdf5_sparse_row_subset_matches_anndata_layer(tmp_path):
    counts = sp.csr_matrix(
        np.array(
            [[1, 0, 4], [0, 2, 0], [3, 1, 5], [0, 0, 2]],
            dtype=np.int64,
        )
    )
    adata = ad.AnnData(
        sp.csr_matrix(np.zeros_like(counts.toarray())),
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(4)]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    adata.layers["counts"] = counts
    path = tmp_path / "matrix.h5ad"
    adata.write_h5ad(path)

    subset = MODULE.read_sparse_row_subset(path, [0, 2], layer="counts")

    assert np.array_equal(subset.toarray(), counts[[0, 2]].toarray())
