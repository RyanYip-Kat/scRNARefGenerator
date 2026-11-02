from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def pbmc_path() -> Path:
    path = Path(__file__).parents[1] / "testdata" / "pbmc5k_querydata.h5ad"
    if not path.exists():
        pytest.skip(f"missing integration fixture: {path}")
    return path
