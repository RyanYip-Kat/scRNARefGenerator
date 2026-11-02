import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scrnarefgenerator import stratified_reference_split


def _reference():
    labels = ["A"] * 10 + ["B"] * 6 + ["C"] * 4
    return ad.AnnData(
        sp.csr_matrix(
            np.arange(len(labels) * 3, dtype=np.int64).reshape(-1, 3) + 1
        ),
        obs=pd.DataFrame(
            {"celltype": labels},
            index=[f"cell{i}" for i in range(len(labels))],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )


def test_split_is_disjoint_and_condition_stratified():
    reference = _reference()

    split = stratified_reference_split(
        reference,
        "celltype",
        fraction=0.25,
        seed=7,
    )

    assert set(split.training_indices) & set(split.holdout_indices) == set()
    assert split.training.n_obs + split.holdout.n_obs == reference.n_obs
    assert set(split.training.obs["celltype"]) == {"A", "B", "C"}
    assert set(split.holdout.obs["celltype"]) == {"A", "B", "C"}
    assert split.holdout.obs["celltype"].value_counts().to_dict() == {
        "A": 2,
        "B": 2,
        "C": 2,
    }


def test_split_is_reproducible_for_the_same_seed():
    first = stratified_reference_split(_reference(), "celltype", seed=13)
    second = stratified_reference_split(_reference(), "celltype", seed=13)

    assert np.array_equal(first.training_indices, second.training_indices)
    assert np.array_equal(first.holdout_indices, second.holdout_indices)


def test_category_too_small_for_both_partitions_is_rejected():
    reference = _reference()
    labels = reference.obs["celltype"].tolist()
    labels[0] = "small"
    labels[1] = "tiny"
    reference.obs["celltype"] = labels

    with pytest.raises(ValueError):
        stratified_reference_split(reference, "celltype")
