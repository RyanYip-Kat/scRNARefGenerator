"""Reusable PCA and UMAP embeddings for sparse raw counts."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from sklearn.decomposition import TruncatedSVD


@dataclass(frozen=True)
class EmbeddingResult:
    pca: np.ndarray
    umap: np.ndarray


def compute_pca_umap(
    counts,
    *,
    n_hvg: int = 2000,
    n_pcs: int = 20,
    n_neighbors: int = 15,
    seed: int = 42,
) -> EmbeddingResult:
    """Compute PCA scores and UMAP coordinates from sparse count data."""

    X = counts.tocsr() if hasattr(counts, "tocsr") else counts
    library = np.asarray(X.sum(axis=1), dtype=float).ravel()
    positive = library[library > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    scale = np.divide(
        median,
        library,
        out=np.zeros_like(library),
        where=library > 0,
    )
    normalized = X.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    second = np.asarray(normalized.multiply(normalized).mean(axis=0)).ravel()
    variance = second - np.square(mean)
    selected = np.argsort(variance)[-min(n_hvg, X.shape[1]) :]
    selected.sort()
    n_components = min(n_pcs, X.shape[0] - 1, selected.size - 1)
    if n_components < 1:
        raise ValueError("at least two cells and two genes are required")
    pca = TruncatedSVD(
        n_components=n_components,
        random_state=seed,
    ).fit_transform(normalized[:, selected])
    cache_dir = Path(tempfile.gettempdir()) / "scrnarefgenerator-numba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    import umap

    coordinates = umap.UMAP(
        n_neighbors=min(n_neighbors, X.shape[0] - 1),
        min_dist=0.3,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    ).fit_transform(pca)
    return EmbeddingResult(pca=pca, umap=coordinates)
