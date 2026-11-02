import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scrnarefgenerator import (
    ConditionSpec,
    EvaluationConfig,
    GeneratorConfig,
    scRNARefGenerator,
)
from scrnarefgenerator.errors import (
    BarcodeSuffixCollisionError,
    EvaluationGateError,
    OutputExistsError,
)


@pytest.fixture
def toy_reference():
    rng = np.random.default_rng(88)
    counts = rng.poisson(
        np.array([2.0, 4.0, 8.0, 12.0, 3.0]), size=(16, 5)
    ).astype(np.int64)
    obs = pd.DataFrame(
        {
            "batch": ["A"] * 8 + ["B"] * 8,
            "group": ["x", "y"] * 8,
            "celltype": ["T"] * 8 + ["B"] * 8,
            "ident": [1, 2] * 8,
        },
        index=[f"cell{i}" for i in range(16)],
    )
    return ad.AnnData(
        sp.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame({"marker": list("abcde")}, index=[f"g{i}" for i in range(5)]),
    )


def test_simulate_has_exact_quota_and_reference_gene_order(toy_reference):
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(seed=5, latent_k=0, strict_evaluation=False),
    )

    generated = generator.simulate(
        ConditionSpec(columns=["batch"], joint={("A",): 2, ("B",): 1})
    )

    assert generated.n_obs == 3
    assert generated.var_names.tolist() == toy_reference.var_names.tolist()
    assert generated.obs["batch"].value_counts().to_dict() == {"A": 2, "B": 1}
    assert generated.obs_names.str.endswith("_sim").all()
    assert sp.isspmatrix_csr(generated.X)
    assert generated.X.dtype.kind in "iu"
    assert (generated.X != generated.layers["counts"]).nnz == 0
    assert {"source_barcode", "sim_batch", "sim_seed"} <= set(generated.obs)


def test_same_seed_reproduces_ann_data_counts(toy_reference):
    config = GeneratorConfig(seed=19, latent_k=0, strict_evaluation=False)
    spec = ConditionSpec(columns=["batch"], joint={("A",): 4})

    first = scRNARefGenerator(toy_reference, config).simulate(spec)
    second = scRNARefGenerator(toy_reference, config).simulate(spec)

    assert (first.X != second.X).nnz == 0
    assert first.obs_names.tolist() == second.obs_names.tolist()


def test_existing_barcode_suffix_raises_before_returning(toy_reference):
    names = list(toy_reference.obs_names.astype(str))
    names[1] = "cell0_sim"
    toy_reference.obs_names = names
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(seed=2, latent_k=0, strict_evaluation=False),
    )

    with pytest.raises(BarcodeSuffixCollisionError):
        generator.simulate(
            ConditionSpec(columns=["batch"], joint={("A",): 1})
        )


def test_simulate_writes_h5ad_and_json_report(toy_reference, tmp_path):
    out = tmp_path / "sim.h5ad"
    report = tmp_path / "sim_report.json"
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(seed=8, latent_k=0, strict_evaluation=False),
    )

    generated = generator.simulate(
        ConditionSpec(columns=["batch"], joint={("A",): 2}),
        output_path=out,
        report_path=report,
    )

    assert out.exists()
    assert report.exists()
    payload = json.loads(report.read_text())
    assert payload["actual_quota"][0]["n_cells"] == 2
    assert payload["evaluation"]["metrics"]
    assert generated.uns["scRNARefGenerator"]["seed"] == 8


def test_unrestricted_mode_generates_exact_requested_total(toy_reference):
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(seed=14, latent_k=0, strict_evaluation=False),
    )

    generated = generator.simulate(condition_spec=None, n_cells=4)

    assert generated.n_obs == 4
    assert generated.obs["source_barcode"].is_unique


def test_unrestricted_mode_can_be_written_to_h5ad(toy_reference, tmp_path):
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(seed=14, latent_k=0, strict_evaluation=False),
    )
    path = tmp_path / "unrestricted.h5ad"

    generator.simulate(condition_spec=None, n_cells=4, output_path=path)

    assert path.exists()


def test_marginal_condition_mode_in_api_has_exact_quota(toy_reference):
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(seed=4, latent_k=0, strict_evaluation=False),
    )

    generated = generator.simulate(
        ConditionSpec(
            columns=["batch", "group"],
            marginal={
                "batch": {"A": 2, "B": 2},
                "group": {"x": 2, "y": 2},
            },
        )
    )

    assert generated.obs["batch"].value_counts().to_dict() == {"A": 2, "B": 2}
    assert generated.obs["group"].value_counts().to_dict() == {"x": 2, "y": 2}


def test_append_uuid_policy_resolves_existing_suffix_collision(toy_reference):
    names = list(toy_reference.obs_names.astype(str))
    names[1] = "cell0_sim"
    toy_reference.obs_names = names
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(
            seed=6,
            latent_k=0,
            strict_evaluation=False,
            collision_policy="append_uuid",
        ),
    )

    generated = generator.simulate(
        ConditionSpec(columns=["batch"], joint={("A",): 3})
    )

    assert generated.obs_names.is_unique
    assert "cell0_sim" not in generated.obs_names


def test_existing_output_is_not_overwritten_by_default(toy_reference, tmp_path):
    out = tmp_path / "existing.h5ad"
    out.write_bytes(b"keep")
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(seed=3, latent_k=0, strict_evaluation=False),
    )

    with pytest.raises(OutputExistsError):
        generator.simulate(
            ConditionSpec(columns=["batch"], joint={("A",): 1}),
            output_path=out,
        )

    assert out.read_bytes() == b"keep"


def test_strict_evaluation_raises_when_gate_is_unreachable(toy_reference):
    config = GeneratorConfig(
        seed=3,
        latent_k=0,
        strict_evaluation=True,
        max_retries=0,
        evaluation=EvaluationConfig(mean_spearman_min=1.0),
    )
    generator = scRNARefGenerator(toy_reference, config)

    with pytest.raises(EvaluationGateError):
        generator.simulate(
            ConditionSpec(columns=["batch"], joint={("A",): 2})
        )


def test_inherited_ident_matches_source_condition(toy_reference):
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(
            seed=21,
            latent_k=0,
            strict_evaluation=False,
            inherit_obs_cols=["ident"],
        ),
    )

    generated = generator.simulate(
        ConditionSpec(
            columns=["celltype"],
            joint={("T",): 3, ("B",): 2},
        )
    )

    assert "source_ident" in generated.obs
    assert "ident" in generated.obs
    assert (generated.obs["celltype"].astype(str) == generated.obs["source_celltype"].astype(str)).all()
    assert (generated.obs["ident"] == generated.obs["source_ident"]).all()
    assert generated.obs["source_barcode"].is_unique


def test_celltype_ident_marginals_keep_exact_quotas(toy_reference):
    generator = scRNARefGenerator(
        toy_reference,
        GeneratorConfig(
            seed=22,
            latent_k=0,
            strict_evaluation=False,
            inherit_obs_cols=["ident"],
        ),
    )

    generated = generator.simulate(
        ConditionSpec(
            columns=["celltype", "ident"],
            marginal={
                "celltype": {"T": 4, "B": 4},
                "ident": {1: 4, 2: 4},
            },
        )
    )

    assert generated.obs["celltype"].value_counts().to_dict() == {"B": 4, "T": 4}
    assert generated.obs["ident"].value_counts().to_dict() == {1: 4, 2: 4}
    assert (
        generated.obs["celltype"].astype(str)
        == generated.obs["source_celltype"].astype(str)
    ).all()
    assert (generated.obs["ident"] == generated.obs["source_ident"]).all()
