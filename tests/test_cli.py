import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def tiny_h5ad(tmp_path):
    counts = np.array(
        [
            [1, 4, 2],
            [0, 3, 8],
            [2, 1, 5],
            [3, 2, 4],
            [1, 5, 2],
            [0, 2, 7],
        ],
        dtype=np.int64,
    )
    adata = ad.AnnData(
        sp.csr_matrix(counts),
        obs=pd.DataFrame(
            {"batch": ["A", "A", "A", "B", "B", "B"]},
            index=[f"cell{i}" for i in range(6)],
        ),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    path = tmp_path / "tiny.h5ad"
    adata.write_h5ad(path)
    return path


def test_cli_writes_h5ad_and_report(tmp_path, tiny_h5ad):
    out = tmp_path / "sim.h5ad"
    report = tmp_path / "report.json"

    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "scrnarefgenerator.cli",
            "simulate",
            "--ref",
            str(tiny_h5ad),
            "--out",
            str(out),
            "--condition-json",
            '{"A":2}',
            "--condition-col",
            "batch",
            "--strict-evaluation",
            "false",
            "--latent-k",
            "0",
            "--max-cells-for-fitting",
            "10",
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )

    assert process.returncode == 0, process.stderr
    assert out.exists()
    payload = json.loads(report.read_text())
    assert payload["actual_quota"] == [{"conditions": {"batch": "A"}, "n_cells": 2}]
