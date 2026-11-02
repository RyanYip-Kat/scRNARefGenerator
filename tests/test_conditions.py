import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scrnarefgenerator.conditions import build_condition_plan
from scrnarefgenerator.config import ConditionSpec, GeneratorConfig
from scrnarefgenerator.errors import (
    ConditionCategoryNotFoundError,
    InfeasibleConditionError,
)
from scrnarefgenerator.validation import resolve_reference


@pytest.fixture
def toy_reference():
    obs = pd.DataFrame(
        {
            "batch": ["A", "A", "B", "B", "B", "C"],
            "group": ["x", "x", "x", "y", "y", "z"],
        },
        index=[f"cell{i}" for i in range(6)],
    )
    adata = ad.AnnData(
        sp.csr_matrix(np.arange(18, dtype=np.int64).reshape(6, 3) + 1),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    return resolve_reference(adata, GeneratorConfig())


def test_joint_quota_selects_only_matching_sources(toy_reference):
    spec = ConditionSpec(columns=["batch"], joint={("B",): 2})

    plan = build_condition_plan(toy_reference, spec, None)

    assert plan.total_cells == 2
    selected = toy_reference.obs.iloc[plan.groups[0].source_indices]
    assert selected["batch"].eq("B").all()


def test_marginal_plan_has_exact_rows_and_columns(toy_reference):
    spec = ConditionSpec(
        columns=["batch", "group"],
        marginal={
            "batch": {"A": 2, "B": 2},
            "group": {"x": 2, "y": 2},
        },
    )

    plan = build_condition_plan(toy_reference, spec, None)
    frame = pd.DataFrame(
        [
            {column: value for column, value in zip(plan.columns, group.values)}
            for group in plan.groups
            for _ in range(group.n_cells)
        ]
    )

    assert frame["batch"].value_counts().to_dict() == {"A": 2, "B": 2}
    assert frame["group"].value_counts().to_dict() == {"x": 2, "y": 2}


def test_impossible_marginals_raise_without_changing_quota(toy_reference):
    spec = ConditionSpec(
        columns=["batch", "group"],
        marginal={
            "batch": {"A": 2, "C": 2},
            "group": {"x": 2, "y": 2},
        },
    )
    # C only co-occurs with z, while the requested group totals exclude z.

    with pytest.raises(InfeasibleConditionError):
        build_condition_plan(toy_reference, spec, None)


def test_unknown_joint_category_is_rejected(toy_reference):
    spec = ConditionSpec(columns=["batch"], joint={("missing",): 1})

    with pytest.raises(ConditionCategoryNotFoundError):
        build_condition_plan(toy_reference, spec, None)


def test_marginal_plan_avoids_concentrating_joint_counts():
    combinations = [
        ("T", 1),
        ("T", 1),
        ("T", 2),
        ("T", 2),
        ("B", 1),
        ("B", 1),
        ("B", 2),
        ("B", 2),
    ]
    adata = ad.AnnData(
        sp.csr_matrix(np.arange(24, dtype=np.int64).reshape(8, 3) + 1),
        obs=pd.DataFrame(
            {
                "celltype": [item[0] for item in combinations],
                "ident": [item[1] for item in combinations],
            },
            index=[f"cell{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    reference = resolve_reference(adata, GeneratorConfig())
    spec = ConditionSpec(
        columns=["celltype", "ident"],
        marginal={
            "celltype": {"T": 4, "B": 4},
            "ident": {1: 4, 2: 4},
        },
    )

    plan = build_condition_plan(reference, spec, None)
    quota = {
        ("T", 1): 0,
        ("T", 2): 0,
        ("B", 1): 0,
        ("B", 2): 0,
    }
    quota.update(
        {
            group.values: group.n_cells
            for group in plan.groups
        }
    )

    assert sum(quota.values()) == 8
    assert max(quota.values()) - min(quota.values()) <= 1
