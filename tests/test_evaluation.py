import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scrnarefgenerator.conditions import build_condition_plan
from scrnarefgenerator.config import ConditionSpec, EvaluationConfig, GeneratorConfig
from scrnarefgenerator.evaluation import EvaluationResult, evaluate_candidate
from scrnarefgenerator.simulate import sample_candidate
from scrnarefgenerator.validation import resolve_reference


@pytest.fixture
def reference_and_candidate():
    rng = np.random.default_rng(44)
    counts = rng.poisson(
        np.array([2.0, 4.0, 7.0, 10.0, 1.0, 5.0]), size=(20, 6)
    ).astype(np.int64)
    obs = pd.DataFrame(
        {"batch": ["A"] * 10 + ["B"] * 10},
        index=[f"cell{i}" for i in range(20)],
    )
    adata = ad.AnnData(
        sp.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
    )
    reference = resolve_reference(adata, GeneratorConfig())
    plan = build_condition_plan(
        reference,
        ConditionSpec(columns=["batch"], joint={("A",): 10}),
        None,
    )
    config = GeneratorConfig(
        seed=12, latent_k=0, sigma_lib=0.15, strict_evaluation=False
    )
    candidate = sample_candidate(reference, plan, config)
    return reference, candidate, config


def test_evaluation_reports_all_contract_metrics(reference_and_candidate):
    reference, candidate, config = reference_and_candidate
    evaluation = EvaluationConfig(
        matched_cells=20,
        high_variable_genes=4,
    )

    result = evaluate_candidate(reference, candidate, evaluation)

    assert set(result.metrics) == {
        "mean_spearman",
        "zero_spearman",
        "library_ks",
        "nn_ratio",
        "classifier_auc",
        "high_variable_spearman",
    }
    assert 0.0 <= result.metrics["classifier_auc"] <= 1.0
    assert isinstance(result.passed, bool)


def test_bad_similarity_report_preserves_violation_names():
    result = EvaluationResult(
        passed=False,
        metrics={"mean_spearman": 0.1, "library_ks": 0.8},
        violations=["mean_spearman", "library_ks"],
    )

    assert result.violations == ["mean_spearman", "library_ks"]
