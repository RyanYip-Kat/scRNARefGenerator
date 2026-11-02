import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scrnarefgenerator.conditions import build_condition_plan
from scrnarefgenerator.config import ConditionSpec, GeneratorConfig
from scrnarefgenerator.simulate import sample_candidate
from scrnarefgenerator.validation import resolve_reference


@pytest.fixture
def toy_reference():
    rng = np.random.default_rng(123)
    counts = rng.poisson(
        np.array([3.0, 8.0, 12.0, 1.0, 5.0, 15.0, 0.5, 9.0]), size=(12, 8)
    ).astype(np.int64)
    obs = pd.DataFrame(
        {
            "batch": ["A"] * 6 + ["B"] * 6,
            "group": ["x", "y"] * 6,
        },
        index=[f"cell{i}" for i in range(12)],
    )
    adata = ad.AnnData(
        sp.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    return resolve_reference(adata, GeneratorConfig())


@pytest.fixture
def toy_plan(toy_reference):
    return build_condition_plan(
        toy_reference,
        ConditionSpec(columns=["batch"], joint={("A",): 5}),
        None,
    )


def test_same_seed_generates_identical_sparse_counts(toy_reference, toy_plan):
    config = GeneratorConfig(
        seed=11,
        latent_k=0,
        sigma_lib=0.0,
        strict_evaluation=False,
    )

    first = sample_candidate(toy_reference, toy_plan, config, attempt=0)
    second = sample_candidate(toy_reference, toy_plan, config, attempt=0)

    assert (first.counts != second.counts).nnz == 0
    assert sp.isspmatrix_csr(first.counts)
    assert first.counts.dtype.kind in "iu"


def test_target_library_size_is_met_within_sampling_tolerance(
    toy_reference, toy_plan
):
    config = GeneratorConfig(
        seed=3,
        sigma_lib=0.0,
        latent_k=0,
        strict_evaluation=False,
    )

    candidate = sample_candidate(toy_reference, toy_plan, config, attempt=0)
    target = candidate.target_library_size
    observed = np.asarray(candidate.counts.sum(axis=1)).ravel()

    assert np.median(observed / target) > 0.75
    assert np.median(observed / target) < 1.25


def test_different_attempt_seeds_change_counts(toy_reference, toy_plan):
    config = GeneratorConfig(seed=5, sigma_lib=0.15, latent_k=0)

    first = sample_candidate(toy_reference, toy_plan, config, attempt=0)
    second = sample_candidate(toy_reference, toy_plan, config, attempt=1)

    assert (first.counts != second.counts).nnz > 0


def test_zinb_backend_is_sparse_integer_and_reproducible(toy_reference, toy_plan):
    config = GeneratorConfig(
        seed=23,
        latent_k=0,
        distribution_model="hierarchical_zinb",
    )

    first = sample_candidate(toy_reference, toy_plan, config, attempt=0)
    second = sample_candidate(toy_reference, toy_plan, config, attempt=0)

    assert (first.counts != second.counts).nnz == 0
    assert first.counts.dtype.kind in "iu"
    assert sp.isspmatrix_csr(first.counts)


def test_latent_backend_produces_reproducible_counts(toy_reference, toy_plan):
    config = GeneratorConfig(
        seed=29,
        latent_k=2,
        max_cells_for_fitting=12,
    )

    first = sample_candidate(toy_reference, toy_plan, config, attempt=0)
    second = sample_candidate(toy_reference, toy_plan, config, attempt=0)

    assert (first.counts != second.counts).nnz == 0
    assert first.counts.dtype.kind in "iu"
