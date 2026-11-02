import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scrnarefgenerator import (
    ConditionSpec,
    EvaluationConfig,
    GeneratorConfig,
    SimilarityReport,
    evaluate_similarity,
    scRNARefGenerator,
)
from scrnarefgenerator.errors import ConditionColumnNotFoundError


@pytest.fixture
def ref_and_simulated():
    rng = np.random.default_rng(203)
    labels = np.repeat(["T", "B", "NK"], 12)
    counts = rng.poisson(
        np.array([2.0, 4.0, 7.0, 10.0, 3.0, 6.0, 1.0, 5.0]),
        size=(labels.size, 8),
    ).astype(np.int64)
    adata = ad.AnnData(
        sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"celltype": labels},
            index=[f"cell{i}" for i in range(labels.size)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(8)]),
    )
    config = GeneratorConfig(
        seed=17,
        latent_k=0,
        max_cells_for_fitting=36,
        strict_evaluation=False,
    )
    simulated = scRNARefGenerator(adata, config).simulate(
        ConditionSpec(
            columns=["celltype"],
            joint={("T",): 3, ("B",): 3, ("NK",): 3},
        )
    )
    return adata, simulated


def test_public_similarity_api_reports_overall_and_each_condition(
    ref_and_simulated,
):
    reference, simulated = ref_and_simulated

    report = evaluate_similarity(
        reference,
        simulated,
        condition_col="celltype",
        config=EvaluationConfig(matched_cells=30, high_variable_genes=5),
    )

    assert isinstance(report, SimilarityReport)
    assert set(report.groups) == {"T", "B", "NK"}
    assert "mean_spearman" in report.overall.metrics
    assert all("library_ks" in result.metrics for result in report.groups.values())
    assert isinstance(report.passed, bool)
    payload = report.as_dict()
    assert payload["condition_col"] == "celltype"
    assert payload["groups"]["T"]["metrics"]["classifier_auc"] >= 0.0


def test_public_similarity_api_rejects_missing_condition_column(
    ref_and_simulated,
):
    reference, simulated = ref_and_simulated
    del simulated.obs["celltype"]

    with pytest.raises(ConditionColumnNotFoundError):
        evaluate_similarity(
            reference,
            simulated,
            condition_col="celltype",
        )
