import numpy as np
import scipy.sparse as sp

from scrnarefgenerator import compute_pca_umap


def test_compute_pca_umap_returns_finite_coordinates():
    rng = np.random.default_rng(10)
    counts = sp.csr_matrix(
        rng.poisson(
            np.array([2.0, 5.0, 8.0, 3.0, 6.0, 1.0]),
            size=(36, 6),
        ).astype(np.int64)
    )

    result = compute_pca_umap(
        counts,
        n_pcs=4,
        n_neighbors=5,
        seed=3,
    )

    assert result.pca.shape == (36, 4)
    assert result.umap.shape == (36, 2)
    assert np.isfinite(result.pca).all()
    assert np.isfinite(result.umap).all()
