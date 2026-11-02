#!/usr/bin/env python3
"""Execute one reproducibility experiment configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scrnarefgenerator-matplotlib")
)

import anndata as ad
import numpy as np

from scrnarefgenerator import (
    EvaluationConfig,
    GeneratorConfig,
    compute_pca_umap,
    condition_spec_from_marginal_fraction,
    condition_spec_from_reference,
    evaluate_novelty,
    evaluate_similarity,
    scRNARefGenerator,
    stratified_reference_split,
)
from scrnarefgenerator.io import json_safe
from scrnarefgenerator.plotting import plot_similarity_umap


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--ref", type=Path, default=ROOT / "testdata/pbmc5k_querydata.h5ad")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--subset", default="all")
    parser.add_argument("--similarity-strength", type=float, default=0.7)
    parser.add_argument("--n-cells", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraction", type=float, default=0.6)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--max-cells-for-fitting", type=int, default=10000)
    parser.add_argument("--novelty-mode", default="balanced")
    parser.add_argument("--novelty-strength", type=float)
    parser.add_argument("--plot-cells-per-condition", type=int, default=200)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.experiment == 3 and args.n_cells is None:
        parser.error("--n-cells is required for experiment 3")
    return args


def _json_write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _plot_embedding(
    coordinates: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str,
) -> None:
    cache = Path(os.environ["MPLCONFIGDIR"])
    cache.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    categories = sorted(set(labels.astype(str)), key=repr)
    cmap = plt.get_cmap("tab20")
    colors = {category: cmap(i % 20) for i, category in enumerate(categories)}
    figure, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for category in categories:
        mask = labels.astype(str) == category
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=3,
            color=colors[category],
            alpha=0.55,
            linewidths=0,
            label=category,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(
        title="Celltype",
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=6,
        ncol=2,
    )
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _sample_balanced_embedding_cells(
    labels: np.ndarray,
    coordinates: np.ndarray,
    selected_categories: list[str],
    maximum_per_category: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if maximum_per_category <= 0:
        raise ValueError("maximum_per_category must be positive")
    labels = labels.astype(str)
    rng = np.random.default_rng(seed)
    sampled_indices: list[np.ndarray] = []
    for category in selected_categories:
        indices = np.flatnonzero(labels == category)
        if indices.size == 0:
            continue
        count = min(maximum_per_category, indices.size)
        sampled_indices.append(rng.choice(indices, size=count, replace=False))
    if not sampled_indices:
        raise ValueError("no cells match the simulated condition categories")
    selected = np.sort(np.concatenate(sampled_indices))
    return coordinates[selected], labels[selected]


def run(args: argparse.Namespace) -> dict[str, object]:
    outdir = args.outdir
    summary_path = outdir / "experiment_summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text())
    outdir.mkdir(parents=True, exist_ok=True)
    reference = ad.read_h5ad(args.ref)
    split = stratified_reference_split(
        reference,
        "celltype",
        fraction=args.holdout_fraction,
        seed=args.seed,
    )
    condition_columns: list[str] = []
    if args.experiment == 1:
        categories = None
        top_n = reference.obs["celltype"].nunique() if str(args.subset) == "all" else int(args.subset)
        spec = condition_spec_from_reference(
            reference,
            "celltype",
            top_n=top_n,
            categories=categories,
            fraction=args.fraction,
        )
        condition_columns = ["celltype"]
    elif args.experiment == 2:
        spec = condition_spec_from_marginal_fraction(
            reference,
            ["celltype", "ident"],
            fraction=args.fraction,
        )
        condition_columns = ["celltype", "ident"]
    else:
        spec = None

    config = GeneratorConfig(
        condition_cols=condition_columns or None,
        seed=args.seed,
        similarity_strength=args.similarity_strength,
        max_cells_for_fitting=args.max_cells_for_fitting,
        novelty_mode=args.novelty_mode,
        novelty_strength=args.novelty_strength,
        strict_evaluation=False,
        max_retries=0,
        overwrite=True,
        source_metadata=True,
        inherit_obs_cols=["ident"],
    )
    generator = scRNARefGenerator(split.training, config)
    simulated = generator.simulate(
        condition_spec=spec,
        n_cells=args.n_cells,
    )
    embedding = compute_pca_umap(
        simulated.layers["counts"],
        seed=args.seed,
        n_pcs=20,
        n_neighbors=15,
    )
    simulated.obsm["X_pca"] = embedding.pca
    simulated.obsm["X_umap"] = embedding.umap
    simulated.write_h5ad(outdir / "simulated.h5ad")

    if args.experiment == 3:
        evaluation = evaluate_similarity(split.holdout, simulated)
        baseline = evaluate_similarity(split.training, split.holdout)
        novelty = None
        shared_sim_col = "source_celltype"
    else:
        evaluation = evaluate_similarity(
            split.holdout,
            simulated,
            condition_col="celltype",
            config=EvaluationConfig(matched_cells=1000),
        )
        baseline = evaluate_similarity(
            split.training,
            split.holdout,
            condition_col="celltype",
            config=EvaluationConfig(matched_cells=1000),
        )
        novelty = evaluate_novelty(simulated, split.training, "celltype")
        shared_sim_col = "celltype"

    generated_labels = (
        simulated.obs["source_celltype"].astype(str).to_numpy()
        if args.experiment == 3
        else simulated.obs["celltype"].astype(str).to_numpy()
    )
    selected_categories = sorted(set(generated_labels), key=repr)
    reference_xy, reference_labels = _sample_balanced_embedding_cells(
        reference.obs["celltype"].astype(str).to_numpy(),
        reference.obsm["X_umap"],
        selected_categories,
        args.plot_cells_per_condition,
        seed=args.seed,
    )
    _plot_embedding(
        reference_xy,
        reference_labels,
        outdir / "reference_existing_umap.png",
        "Reference: existing UMAP (sampled selected conditions)",
    )
    generated_xy, sampled_generated_labels = _sample_balanced_embedding_cells(
        generated_labels,
        embedding.umap,
        selected_categories,
        args.plot_cells_per_condition,
        seed=args.seed + 1,
    )
    _plot_embedding(
        generated_xy,
        sampled_generated_labels,
        outdir / "generated_umap.png",
        "Simulated: recomputed UMAP (sampled selected conditions)",
    )
    shared = plot_similarity_umap(
        split.holdout,
        simulated,
        condition_col="celltype",
        simulated_condition_col=shared_sim_col,
        output_dir=outdir / "shared_umap",
        formats=tuple(args.formats),
        dpi=args.dpi,
        seed=args.seed,
    )
    payload = {
        "experiment": args.experiment,
        "subset": args.subset,
        "similarity_strength": args.similarity_strength,
        "n_cells_requested": args.n_cells,
        "n_cells_generated": int(simulated.n_obs),
        "condition_columns": condition_columns,
        "plot_cells_per_condition": args.plot_cells_per_condition,
        "actual_quota": (
            None
            if spec is None
            else [
                {"conditions": list(key), "n_cells": value}
                for key, value in spec.joint.items()
            ]
            if spec.joint is not None
            else spec.marginal
        ),
        "evaluation": evaluation.as_dict(),
        "baseline": baseline.as_dict(),
        "novelty": None if novelty is None else novelty.as_dict(),
        "files": {
            "simulated": str(outdir / "simulated.h5ad"),
            "reference_existing_umap": str(outdir / "reference_existing_umap.png"),
            "generated_umap": str(outdir / "generated_umap.png"),
            "shared_umap": {k: {ext: str(p) for ext, p in v.items()} for k, v in shared.files.items()},
            "embedding_csv": str(shared.embedding_path),
        },
    }
    _json_write(summary_path, payload)
    return payload


def main() -> int:
    args = _args()
    try:
        result = run(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "outdir": str(args.outdir),
                "n_cells": result["n_cells_generated"],
                "evaluation_passed": result["evaluation"]["passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
