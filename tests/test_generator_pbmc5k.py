import json

import anndata as ad
import scipy.sparse as sp

from scrnarefgenerator import ConditionSpec, GeneratorConfig, scRNARefGenerator


def test_generates_exact_small_quota_from_real_pbmc_reference(pbmc_path, tmp_path):
    reference = ad.read_h5ad(pbmc_path)
    config = GeneratorConfig(
        seed=31,
        similarity_strength=0.70,
        latent_k=0,
        max_cells_for_fitting=200,
        strict_evaluation=False,
        source_metadata=False,
    )
    report_path = tmp_path / "pbmc_report.json"
    generator = scRNARefGenerator(reference, config)

    generated = generator.simulate(
        ConditionSpec(
            columns=["celltype"],
            joint={("CD14",): 1, ("NK3",): 1},
        ),
        report_path=report_path,
    )

    assert generated.shape == (2, reference.n_vars)
    assert generated.var_names.equals(reference.var_names)
    assert generated.obs["celltype"].value_counts().to_dict() == {
        "CD14": 1,
        "NK3": 1,
    }
    assert sp.isspmatrix_csr(generated.X)
    assert generated.X.dtype.kind in "iu"
    payload = json.loads(report_path.read_text())
    assert payload["actual_quota"] == [
        {"conditions": {"celltype": "CD14"}, "n_cells": 1},
        {"conditions": {"celltype": "NK3"}, "n_cells": 1},
    ]
    assert "classifier_auc" in payload["evaluation"]["metrics"]
