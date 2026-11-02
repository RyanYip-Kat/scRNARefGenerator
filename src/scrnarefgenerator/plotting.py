"""Optional shared-embedding figures for reference-versus-simulation review."""

from __future__ import annotations

from dataclasses import dataclass
import os
from os import PathLike
from pathlib import Path
import tempfile
from typing import Iterable
import warnings

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from matplotlib.text import Text
from scipy.stats import ks_2samp
from sklearn.decomposition import TruncatedSVD

from .config import GeneratorConfig
from .errors import (
    ConditionCategoryNotFoundError,
    ConditionColumnNotFoundError,
)
from .validation import ReferenceData, resolve_reference


@dataclass(frozen=True)
class PlotArtifacts:
    files: dict[str, dict[str, Path]]
    embedding_path: Path
    embedding_rows: int


def plot_similarity_umap(
    reference: ReferenceData | ad.AnnData | str | PathLike[str],
    simulated: ad.AnnData | str | PathLike[str],
    condition_col: str,
    output_dir: str | PathLike[str],
    *,
    simulated_condition_col: str | None = None,
    formats: Iterable[str] = ("png", "pdf", "svg"),
    dpi: int = 300,
    seed: int = 42,
    max_reference_per_condition: int | None = None,
    n_hvg: int = 2000,
    n_pcs: int = 20,
    n_neighbors: int = 15,
) -> PlotArtifacts:
    """Create jointly fitted UMAP and library-size comparison figures."""

    reference_data = (
        reference
        if isinstance(reference, ReferenceData)
        else resolve_reference(reference, GeneratorConfig())
    )
    simulated_data = resolve_reference(simulated, GeneratorConfig())
    simulated_condition_col = simulated_condition_col or condition_col
    _validate_condition_columns(
        reference_data,
        simulated_data,
        condition_col,
        simulated_condition_col,
    )
    reference_sample, observed = _matched_reference_sample(
        reference_data,
        simulated_data,
        condition_col,
        simulated_condition_col,
        seed=seed,
        maximum=max_reference_per_condition,
    )
    simulated_sample = _matched_simulation_sample(
        simulated_data,
        simulated_condition_col,
        observed,
        seed=seed + 1,
    )
    combined = _combine(
        reference_sample,
        simulated_sample,
        condition_col,
        simulated_condition_col,
    )
    embedding, normalized_library = _shared_umap(
        combined["X"],
        seed=seed,
        n_hvg=n_hvg,
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
    )

    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    formats = tuple(dict.fromkeys(formats))
    invalid_formats = set(formats) - {"png", "pdf", "svg"}
    if invalid_formats:
        raise ValueError(f"unsupported figure formats: {sorted(invalid_formats)}")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")

    embedding_frame = pd.DataFrame(
        {
            "umap_1": embedding[:, 0],
            "umap_2": embedding[:, 1],
            "dataset": combined["dataset"],
            "celltype": combined["celltype"],
        }
    )
    embedding_path = output / "umap_embedding.csv"
    embedding_frame.to_csv(embedding_path, index=False)

    files: dict[str, dict[str, Path]] = {
        "umap_comparison": {},
        "library_size_ecdf": {},
    }
    figure = _plot_umap_comparison(embedding_frame)
    files["umap_comparison"] = _save_figure(
        figure, output, "umap_comparison", formats, dpi
    )
    plt.close(figure)

    figure = _plot_library_ecdf(
        normalized_library,
        combined["dataset"],
    )
    files["library_size_ecdf"] = _save_figure(
        figure, output, "library_size_ecdf", formats, dpi
    )
    plt.close(figure)
    return PlotArtifacts(
        files=files,
        embedding_path=embedding_path,
        embedding_rows=embedding_frame.shape[0],
    )


def _validate_condition_columns(
    reference: ReferenceData,
    simulated: ReferenceData,
    condition_col: str,
    simulated_condition_col: str,
) -> None:
    for label, data, column in (
        ("reference", reference, condition_col),
        ("simulated", simulated, simulated_condition_col),
    ):
        if column not in data.obs:
            raise ConditionColumnNotFoundError(
                f"condition column {column!r} is absent from {label}"
            )
    missing = set(simulated.obs[simulated_condition_col].dropna().unique()) - set(
        reference.obs[condition_col].dropna().unique()
    )
    if missing:
        raise ConditionCategoryNotFoundError(
            f"simulated categories absent from reference: {sorted(missing)}"
        )


def _matched_reference_sample(
    reference: ReferenceData,
    simulated: ReferenceData,
    condition_col: str,
    simulated_condition_col: str,
    seed: int,
    maximum: int | None,
) -> tuple[ReferenceData, dict[str, int]]:
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    observed: dict[str, int] = {}
    categories = sorted(
        simulated.obs[simulated_condition_col].dropna().unique(),
        key=repr,
    )
    for category in categories:
        simulation_count = int(
            simulated.obs[simulated_condition_col].eq(category).sum()
        )
        reference_indices = np.flatnonzero(
            reference.obs[condition_col].eq(category).to_numpy()
        )
        target = simulation_count if maximum is None else min(simulation_count, maximum)
        target = min(target, reference_indices.size)
        if target:
            sampled.append(
                rng.choice(reference_indices, size=target, replace=False)
            )
        observed[str(category)] = int(target)
    if not sampled:
        raise ValueError("no reference cells share conditions with the simulation")
    indices = np.sort(np.concatenate(sampled))
    sample = ReferenceData(
        X=reference.X[indices],
        obs=reference.obs.iloc[indices].copy(),
        var=reference.var.copy(),
        obs_names=reference.obs_names[indices].copy(),
        var_names=reference.var_names.copy(),
        path=reference.path,
    )
    return sample, observed


def _matched_simulation_sample(
    simulated: ReferenceData,
    condition_col: str,
    targets: dict[str, int],
    seed: int,
) -> ReferenceData:
    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    for category, target in targets.items():
        indices = np.flatnonzero(
            simulated.obs[condition_col].astype(str).eq(category).to_numpy()
        )
        if target and indices.size:
            sampled.append(rng.choice(indices, size=min(target, indices.size), replace=False))
    if not sampled:
        raise ValueError("no simulated cells share conditions with the reference")
    indices = np.sort(np.concatenate(sampled))
    return ReferenceData(
        X=simulated.X[indices],
        obs=simulated.obs.iloc[indices].copy(),
        var=simulated.var.copy(),
        obs_names=simulated.obs_names[indices].copy(),
        var_names=simulated.var_names.copy(),
        path=simulated.path,
    )


def _combine(
    reference: ReferenceData,
    simulated: ReferenceData,
    condition_col: str,
    simulated_condition_col: str,
) -> dict[str, object]:
    from scipy.sparse import vstack

    X = vstack([reference.X, simulated.X], format="csr")
    dataset = np.concatenate(
        [
            np.repeat("Reference", reference.n_obs),
            np.repeat("Simulated", simulated.n_obs),
        ]
    )
    return {
        "X": X,
        "dataset": dataset,
        "celltype": np.concatenate(
            [
                reference.obs[condition_col].astype(str).to_numpy(),
                simulated.obs[simulated_condition_col].astype(str).to_numpy(),
            ]
        ),
    }


def _shared_umap(
    X,
    seed: int,
    n_hvg: int,
    n_pcs: int,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    library = np.asarray(X.sum(axis=1), dtype=float).ravel()
    normalized_library = np.log10(np.clip(library, 1.0, None))
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
    n_hvg = min(n_hvg, X.shape[1])
    selected = np.argsort(variance)[-n_hvg:]
    selected.sort()
    n_components = min(n_pcs, X.shape[0] - 1, selected.size - 1)
    if n_components < 1:
        raise ValueError("at least two cells and two genes are required for UMAP")
    pca = TruncatedSVD(n_components=n_components, random_state=seed).fit_transform(
        normalized[:, selected]
    )

    cache_dir = Path(tempfile.gettempdir()) / "scrnarefgenerator-numba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    try:
        import umap
    except ImportError as error:
        raise RuntimeError(
            "UMAP plotting requires the optional dependency 'umap-learn'"
        ) from error
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, X.shape[0] - 1),
        min_dist=0.30,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    return reducer.fit_transform(pca), normalized_library


def _plot_umap_comparison(
    frame: pd.DataFrame,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.0, 4.5),
        constrained_layout=True,
    )
    reference = frame.loc[frame["dataset"].eq("Reference")]
    simulated = frame.loc[frame["dataset"].eq("Simulated")]
    reference_color = "#0072B2"
    simulation_color = "#E69F00"

    axes[0].scatter(
        reference["umap_1"],
        reference["umap_2"],
        s=7,
        color=reference_color,
        alpha=0.55,
        linewidths=0,
        label=f"Reference (n={len(reference)})",
    )
    axes[0].scatter(
        simulated["umap_1"],
        simulated["umap_2"],
        s=10,
        marker="^",
        color=simulation_color,
        alpha=0.60,
        linewidths=0,
        label=f"Simulated (n={len(simulated)})",
    )
    axes[0].set_title("Shared UMAP: dataset overlay")
    axes[0].legend(frameon=False, loc="best")

    categories = sorted(frame["celltype"].dropna().unique(), key=repr)
    palette = plt.get_cmap("tab20")
    colors = {
        category: palette(index % 20)
        for index, category in enumerate(categories)
    }
    for ax, subset, title in (
        (axes[1], reference, "Reference"),
        (axes[2], simulated, "Simulated"),
    ):
        ax.scatter(
            subset["umap_1"],
            subset["umap_2"],
            s=7,
            color="#B8B8B8",
            alpha=0.35,
            linewidths=0,
            zorder=1,
        )
        for category in categories:
            group = subset.loc[subset["celltype"].eq(category)]
            if group.empty:
                continue
            ax.scatter(
                group["umap_1"],
                group["umap_2"],
                s=9,
                color=colors[category],
                alpha=0.75,
                linewidths=0,
                zorder=2,
            )
        ax.set_title(f"{title}: celltype structure")

    for ax in axes:
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4,
            color=colors[category],
            label=str(category),
        )
        for category in categories
    ]
    figure.legend(
        handles=handles,
        title="Celltype",
        loc="outside lower center",
        ncol=7,
        frameon=False,
        fontsize=6.5,
        title_fontsize=6.8,
    )
    _audit_text_clipping(figure)
    return figure


def _plot_library_ecdf(
    log_library: np.ndarray,
    dataset: np.ndarray,
) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(5.2, 3.8), constrained_layout=True)
    colors = {"Reference": "#0072B2", "Simulated": "#E69F00"}
    for label in ("Reference", "Simulated"):
        values = np.sort(log_library[dataset == label])
        probability = np.arange(1, values.size + 1) / values.size
        ax.step(
            values,
            probability,
            where="post",
            color=colors[label],
            linewidth=1.5,
            label=f"{label} (n={values.size})",
        )
    statistic = ks_2samp(
        log_library[dataset == "Reference"],
        log_library[dataset == "Simulated"],
    ).statistic
    ax.set_xlabel("log10(library size)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Library-size distributions (KS={statistic:.3f})")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _audit_text_clipping(figure)
    return figure


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...],
    dpi: int,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        paths[extension] = path
    return paths


def _audit_text_clipping(figure: plt.Figure) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        figure.canvas.draw()
    missing_glyph = [
        str(item.message)
        for item in caught
        if "Glyph" in str(item.message) or "missing from current font" in str(item.message)
    ]
    if missing_glyph:
        raise RuntimeError(f"figure has missing glyphs: {missing_glyph[0]}")
    for text in figure.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        x, y = text.get_position()
        if not np.isfinite(x) or not np.isfinite(y):
            raise RuntimeError(f"figure text has invalid coordinates: {text.get_text()!r}")
