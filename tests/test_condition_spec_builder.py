import anndata as ad
import pandas as pd
import pytest
import scipy.sparse as sp

from scrnarefgenerator import (
    ConditionSpec,
    condition_spec_from_marginal_fraction,
    condition_spec_from_reference,
)


def _reference():
    labels = ["A"] * 5 + ["B"] * 3 + ["C"] * 10 + ["D"] * 4
    return ad.AnnData(
        sp.csr_matrix([[1] for _ in labels]),
        obs=pd.DataFrame(
            {"celltype": labels},
            index=[f"cell{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=["g1"]),
    )


def test_top_n_categories_are_ranked_by_reference_cell_count():
    spec = condition_spec_from_reference(
        _reference(),
        "celltype",
        top_n=3,
        fraction=0.6,
    )

    assert isinstance(spec, ConditionSpec)
    assert spec.joint == {("C",): 6, ("A",): 3, ("D",): 2}


def test_specified_categories_preserve_order_and_use_sixty_percent():
    spec = condition_spec_from_reference(
        _reference(),
        "celltype",
        top_n=3,
        categories=["D", "A", "B"],
        fraction=0.6,
    )

    assert list(spec.joint) == [("D",), ("A",), ("B",)]
    assert spec.joint == {("D",): 2, ("A",): 3, ("B",): 1}


def test_specified_category_count_cannot_exceed_top_n():
    with pytest.raises(ValueError):
        condition_spec_from_reference(
            _reference(),
            "celltype",
            top_n=2,
            categories=["A", "B", "C"],
            fraction=0.6,
        )


def test_fraction_must_be_between_zero_and_one():
    with pytest.raises(ValueError):
        condition_spec_from_reference(
            _reference(),
            "celltype",
            top_n=2,
            fraction=0,
        )


def test_multi_column_fraction_uses_equal_exact_totals():
    labels = ["T"] * 6 + ["B"] * 6
    idents = [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3
    adata = ad.AnnData(
        sp.csr_matrix([[1] for _ in labels]),
        obs=pd.DataFrame(
            {"celltype": labels, "ident": idents},
            index=[f"cell{i}" for i in range(12)],
        ),
        var=pd.DataFrame(index=["g1"]),
    )

    spec = condition_spec_from_marginal_fraction(
        adata,
        ["celltype", "ident"],
        fraction=0.6,
    )

    assert spec.columns == ["celltype", "ident"]
    assert sum(spec.marginal["celltype"].values()) == 7
    assert sum(spec.marginal["ident"].values()) == 7
    assert all(
        abs(quota - 6 * 0.6) <= 0.6 + 1e-9
        for quotas in spec.marginal.values()
        for quota in quotas.values()
    )
