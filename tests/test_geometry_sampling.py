import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scrnarefgenerator.conditions import build_condition_plan
from scrnarefgenerator.config import ConditionSpec, GeneratorConfig
from scrnarefgenerator.simulate import sample_candidate
from scrnarefgenerator.validation import resolve_reference


def _fixture():
    rng = np.random.default_rng(87)
    counts = rng.poisson(
        np.array([3.0, 8.0, 12.0, 1.0, 7.0, 15.0]),
        size=(16, 6),
    ).astype(np.int64)
    counts[8:, 0] += 6
    counts[:8, 5] += 5
    obs = pd.DataFrame(
        {"batch": ["A"] * 16},
        index=[f"cell{i}" for i in range(16)],
    )
    reference = resolve_reference(
        ad.AnnData(
            sp.csr_matrix(counts),
            obs=obs,
            var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
        ),
        GeneratorConfig(),
    )
    plan = build_condition_plan(
        reference,
        ConditionSpec(columns=["batch"], joint={("A",): 10}),
        None,
    )
    return reference, plan


def test_neighbor_geometry_changes_cell_specific_profiles():
    reference, plan = _fixture()
    baseline = sample_candidate(
        reference,
        plan,
        GeneratorConfig(
            seed=13,
            geometry_model="none",
            latent_k=0,
        ),
        attempt=0,
    )
    neighbor = sample_candidate(
        reference,
        plan,
        GeneratorConfig(
            seed=13,
            geometry_model="neighbor_preserving",
            geometry_neighbors=4,
            geometry_latent_k=3,
            local_profile_mix=0.6,
            group_prior_mix=0.1,
            latent_k=0,
        ),
        attempt=0,
    )

    assert (baseline.counts != neighbor.counts).nnz > 0
    assert neighbor.counts.dtype.kind in "iu"
    assert sp.isspmatrix_csr(neighbor.counts)


def test_neighbor_sampling_is_reproducible():
    reference, plan = _fixture()
    config = GeneratorConfig(
        seed=8,
        geometry_model="neighbor_preserving",
        geometry_neighbors=4,
        geometry_latent_k=3,
        latent_k=0,
    )

    first = sample_candidate(reference, plan, config, attempt=0)
    second = sample_candidate(reference, plan, config, attempt=0)

    assert (first.counts != second.counts).nnz == 0


def test_fidelity_mode_is_invariant_and_balanced_mode_changes_counts():
    reference, plan = _fixture()
    fidelity_config = GeneratorConfig(
        seed=17,
        geometry_model="neighbor_preserving",
        geometry_neighbors=4,
        geometry_latent_k=3,
        latent_k=0,
        novelty_mode="fidelity",
    )
    first = sample_candidate(reference, plan, fidelity_config, attempt=0)
    second = sample_candidate(reference, plan, fidelity_config, attempt=0)
    balanced = sample_candidate(
        reference,
        plan,
        GeneratorConfig(
            seed=17,
            geometry_model="neighbor_preserving",
            geometry_neighbors=4,
            geometry_latent_k=3,
            latent_k=0,
            novelty_mode="balanced",
        ),
        attempt=0,
    )

    assert (first.counts != second.counts).nnz == 0
    assert (first.counts != balanced.counts).nnz > 0
