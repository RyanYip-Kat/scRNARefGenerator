#!/usr/bin/env python3
"""Generate conditional PBMC data, evaluate it, and build comparison figures."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad

from scrnarefgenerator import (
    EvaluationConfig,
    GeneratorConfig,
    condition_spec_from_reference,
    evaluate_similarity,
    evaluate_novelty,
    scRNARefGenerator,
    stratified_reference_split,
)
from scrnarefgenerator.io import json_safe

matplotlib_cache = Path(tempfile.gettempdir()) / "scrnarefgenerator-matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

from scrnarefgenerator.plotting import plot_similarity_umap


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end scRNARefGenerator celltype pipeline."
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=ROOT / "testdata" / "pbmc5k_querydata.h5ad",
        help="reference .h5ad path",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "pipeline",
        help="directory for generated data, reports, and figures",
    )
    parser.add_argument("--condition-col", default="celltype")
    parser.add_argument(
        "--conditions",
        nargs="+",
        help="optional categories; defaults to the largest categories by cell count",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="number of largest reference categories to select",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.60,
        help="simulated fraction of each selected reference category",
    )
    parser.add_argument(
        "--max-reference-per-condition",
        type=int,
        help="maximum matched reference cells per category in UMAP",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--similarity-strength", type=float, default=0.70)
    parser.add_argument("--latent-k", type=int, default=0)
    parser.add_argument("--max-cells-for-fitting", type=int, default=10000)
    parser.add_argument(
        "--geometry-model",
        choices=["neighbor_preserving", "none"],
        default="neighbor_preserving",
    )
    parser.add_argument("--geometry-neighbors", type=int, default=30)
    parser.add_argument("--local-profile-mix", type=float, default=0.50)
    parser.add_argument("--group-prior-mix", type=float, default=0.10)
    parser.add_argument("--geometry-latent-k", type=int, default=20)
    parser.add_argument("--geometry-hvg-count", type=int, default=2000)
    parser.add_argument(
        "--count-noise-model",
        choices=["poisson", "nb"],
        default="poisson",
    )
    parser.add_argument(
        "--novelty-mode",
        choices=["fidelity", "balanced", "novel"],
        default="balanced",
    )
    parser.add_argument("--novelty-strength", type=float)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--strict-generation", type=_boolean, default=False)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png", "pdf", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if not args.ref.exists():
        raise FileNotFoundError(f"reference file does not exist: {args.ref}")
    args.outdir.mkdir(parents=True, exist_ok=True)

    reference = ad.read_h5ad(args.ref)
    if args.condition_col not in reference.obs:
        raise ValueError(
            f"condition column {args.condition_col!r} is absent from reference.obs"
        )
    condition_spec = condition_spec_from_reference(
        reference,
        args.condition_col,
        top_n=args.top_n,
        categories=args.conditions,
        fraction=args.fraction,
    )

    split = stratified_reference_split(
        reference,
        args.condition_col,
        fraction=args.holdout_fraction,
        seed=args.seed,
    )
    generator_config = GeneratorConfig(
        condition_cols=[args.condition_col],
        seed=args.seed,
        similarity_strength=args.similarity_strength,
        geometry_model=args.geometry_model,
        geometry_neighbors=args.geometry_neighbors,
        local_profile_mix=args.local_profile_mix,
        group_prior_mix=args.group_prior_mix,
        geometry_latent_k=args.geometry_latent_k,
        geometry_hvg_count=args.geometry_hvg_count,
        count_noise_model=args.count_noise_model,
        novelty_mode=args.novelty_mode,
        novelty_strength=args.novelty_strength,
        geometry_eval_holdout_fraction=args.holdout_fraction,
        latent_k=args.latent_k,
        max_cells_for_fitting=args.max_cells_for_fitting,
        strict_evaluation=args.strict_generation,
        max_retries=1 if args.strict_generation else 0,
        overwrite=args.overwrite,
        source_metadata=True,
        inherit_obs_cols=["ident"],
    )
    generator = scRNARefGenerator(split.training, generator_config)
    simulated_path = args.outdir / "simulated.h5ad"
    generation_report_path = args.outdir / "generation_report.json"
    simulated = generator.simulate(
        condition_spec=condition_spec,
        output_path=simulated_path,
        report_path=generation_report_path,
    )

    evaluation = evaluate_similarity(
        split.holdout,
        simulated,
        condition_col=args.condition_col,
        config=EvaluationConfig(matched_cells=1000),
    )
    baseline = evaluate_similarity(
        split.training,
        split.holdout,
        condition_col=args.condition_col,
        config=EvaluationConfig(matched_cells=1000),
    )
    novelty = evaluate_novelty(
        simulated,
        split.training,
        condition_col=args.condition_col,
    )
    evaluation_path = args.outdir / "similarity_evaluation.json"
    _json_write(
        evaluation_path,
        {
            "reference": str(args.ref),
            "simulated": str(simulated_path),
            "n_simulated": simulated.n_obs,
            "selection": {
                "top_n": args.top_n,
                "categories": args.conditions,
                "fraction": args.fraction,
                "rounding": "floor",
            },
            "geometry": {
                "model": args.geometry_model,
                "neighbors": args.geometry_neighbors,
                "local_profile_mix": args.local_profile_mix,
                "group_prior_mix": args.group_prior_mix,
                "latent_k": args.geometry_latent_k,
                "hvg_count": args.geometry_hvg_count,
                "count_noise_model": args.count_noise_model,
                "novelty_mode": args.novelty_mode,
                "effective_novelty_strength": generator_config.effective_novelty_strength,
            },
            "reference_split": {
                "holdout_fraction": args.holdout_fraction,
                "training_cells": split.training.n_obs,
                "holdout_cells": split.holdout.n_obs,
            },
            "quota": {
                key[0]: quota for key, quota in condition_spec.joint.items()
            },
            "baseline_training_vs_holdout": baseline.as_dict(),
            "novelty": novelty.as_dict(),
            **evaluation.as_dict(),
        },
    )

    figures_dir = args.outdir / "figures"
    artifacts = plot_similarity_umap(
        split.holdout,
        simulated,
        condition_col=args.condition_col,
        output_dir=figures_dir,
        formats=tuple(args.formats),
        dpi=args.dpi,
        seed=args.seed,
        max_reference_per_condition=args.max_reference_per_condition,
    )
    summary = {
        "simulated_path": simulated_path,
        "generation_report": generation_report_path,
        "similarity_evaluation": evaluation_path,
        "embedding": artifacts.embedding_path,
        "figures": artifacts.files,
        "evaluation_passed": evaluation.passed,
        "overall_metrics": evaluation.overall.metrics,
        "baseline_overall_metrics": baseline.overall.metrics,
        "novelty_metrics": novelty.metrics,
        "reference_split": {
            "training_cells": split.training.n_obs,
            "holdout_cells": split.holdout.n_obs,
        },
        "group_violations": {
            group: result.violations
            for group, result in evaluation.groups.items()
            if result.violations
        },
    }
    summary_path = args.outdir / "pipeline_summary.json"
    _json_write(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = run(args)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Generated data: {summary['simulated_path']}")
    print(f"Evaluation: {summary['similarity_evaluation']}")
    print(f"Embedding CSV: {summary['embedding']}")
    for name, formats in summary["figures"].items():
        for extension, path in formats.items():
            print(f"Figure {name} ({extension}): {path}")
    print(
        "Overall metrics: "
        + json.dumps(summary["overall_metrics"], indent=2, sort_keys=True)
    )
    print(
        "Novelty diagnostics: "
        + json.dumps(summary["novelty_metrics"], indent=2, sort_keys=True)
    )
    if summary["group_violations"]:
        print(f"Quality warnings: {summary['group_violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
