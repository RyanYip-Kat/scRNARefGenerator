import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scrnarefgenerator.plotting import plot_similarity_umap


def test_shared_umap_and_library_figures_are_written(tmp_path):
    rng = np.random.default_rng(141)
    labels = np.repeat(["T", "B", "NK"], 10)
    reference_counts = rng.poisson(
        np.array([2.0, 5.0, 9.0, 3.0, 7.0, 1.0, 4.0, 6.0]),
        size=(labels.size, 8),
    ).astype(np.int64)
    simulated_counts = rng.poisson(
        np.array([2.2, 4.8, 8.5, 3.2, 6.7, 1.1, 4.1, 6.2]),
        size=(15, 8),
    ).astype(np.int64)
    simulated_labels = np.repeat(["T", "B", "NK"], 5)
    reference = ad.AnnData(
        sp.csr_matrix(reference_counts),
        obs=pd.DataFrame(
            {"celltype": labels},
            index=[f"ref{i}" for i in range(labels.size)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    simulated = ad.AnnData(
        sp.csr_matrix(simulated_counts),
        obs=pd.DataFrame(
            {"source_celltype": simulated_labels},
            index=[f"sim{i}" for i in range(15)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    simulated.layers["counts"] = simulated.X.copy()

    artifacts = plot_similarity_umap(
        reference,
        simulated,
        condition_col="celltype",
        simulated_condition_col="source_celltype",
        output_dir=tmp_path,
        formats=("png",),
        dpi=100,
        seed=9,
    )

    assert artifacts.files["umap_comparison"]["png"].exists()
    assert artifacts.files["library_size_ecdf"]["png"].exists()
    assert artifacts.embedding_rows == 30
