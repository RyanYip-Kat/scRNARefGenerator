import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from scrnarefgenerator import NoveltyReport, evaluate_novelty


def _reference():
    labels = ["T"] * 6 + ["B"] * 6
    rng = np.random.default_rng(12)
    counts = rng.poisson(3.0, size=(12, 6)).astype(np.int64) + 1
    counts[rng.random(counts.shape) < 0.45] = 0
    counts[0, :3] = [1, 2, 3]
    counts[0, 3:] = 0
    counts[6, :3] = 0
    counts[6, 3:] = [4, 5, 6]
    return ad.AnnData(
        sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"celltype": labels},
            index=[f"ref{i}" for i in range(12)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(6)]),
    )


def test_exact_source_copy_is_detected():
    reference = _reference()
    copied = reference[[0, 6]].copy()
    copied.obs["source_barcode"] = copied.obs_names.astype(str)
    copied.obs["celltype"] = ["T", "B"]
    copied.obs_names = ["sim0", "sim1"]
    copied.layers["counts"] = copied.X.copy()

    report = evaluate_novelty(copied, reference, "celltype")

    assert isinstance(report, NoveltyReport)
    assert report.metrics["exact_source_copy_fraction"] == 1.0
    assert report.metrics["source_jaccard_median"] == 1.0
    assert report.metrics["source_jaccard_gain"] > 1.0
    assert report.metrics["source_library_ratio_median"] == 1.0
