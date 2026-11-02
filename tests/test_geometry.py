import numpy as np
import scipy.sparse as sp

from scrnarefgenerator.config import GeneratorConfig
from scrnarefgenerator.geometry import (
    fit_group_geometry,
    perturbation_from_latent,
    query_neighbors,
)


def _group_counts():
    rng = np.random.default_rng(31)
    return sp.csr_matrix(
        rng.poisson(
            np.array([2.0, 5.0, 8.0, 3.0, 6.0, 1.0]),
            size=(18, 6),
        ).astype(np.int64)
    )


def test_neighbor_geometry_returns_other_cells_from_same_group():
    X = _group_counts()
    geometry = fit_group_geometry(
        X,
        np.arange(X.shape[0]),
        GeneratorConfig(
            geometry_neighbors=3,
            geometry_latent_k=3,
            geometry_hvg_count=5,
            max_cells_for_fitting=18,
        ),
        seed=4,
    )

    neighbors, distances = query_neighbors(
        geometry,
        selected_positions=np.array([0, 1]),
        seed=5,
    )

    assert neighbors.shape == (2, 3)
    assert distances.shape == (2, 3)
    assert np.all(neighbors[0] != 0)
    assert np.all(neighbors[1] != 1)
    assert np.all(distances[:, 0] >= 0)


def test_neighbor_lookup_is_reproducible():
    X = _group_counts()
    geometry = fit_group_geometry(
        X,
        np.arange(X.shape[0]),
        GeneratorConfig(geometry_neighbors=4, geometry_latent_k=3),
        seed=8,
    )

    first = query_neighbors(geometry, np.array([2, 7, 11]), seed=12)
    second = query_neighbors(geometry, np.array([2, 7, 11]), seed=12)

    assert np.array_equal(first[0], second[0])
    assert np.allclose(first[1], second[1])


def test_geometry_has_gene_space_novelty_factors():
    X = _group_counts()
    geometry = fit_group_geometry(
        X,
        np.arange(X.shape[0]),
        GeneratorConfig(
            geometry_neighbors=4,
            geometry_latent_k=3,
            geometry_hvg_count=5,
        ),
        seed=3,
    )

    perturbation = perturbation_from_latent(
        geometry,
        n_cells=1000,
        rng=np.random.default_rng(1),
        strength=0.2,
    )

    assert geometry is not None
    assert geometry.novelty_factors.shape[0] == X.shape[1]
    assert perturbation.shape == (1000, X.shape[1])
    assert np.isfinite(perturbation).all()
    assert 0.10 < float(np.median(np.std(perturbation, axis=0))) < 0.30
