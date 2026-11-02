"""Retry and reporting orchestration for one simulation request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .assembly import assemble_anndata
from .conditions import ConditionPlan
from .config import ConditionSpec, GeneratorConfig
from .distribution import fit_distribution
from .errors import EvaluationGateError
from .evaluation import EvaluationResult, evaluate_candidate
from .io import json_safe
from .simulate import CandidateBatch, sample_candidate
from .validation import ReferenceData


@dataclass(frozen=True)
class PipelineResult:
    generated: Any
    report: dict[str, Any]


def run_pipeline(
    reference: ReferenceData,
    plan: ConditionPlan,
    spec: ConditionSpec | None,
    config: GeneratorConfig,
    output_path: str | None = None,
    report_path: str | None = None,
) -> PipelineResult:
    fitted = fit_distribution(reference, plan, config)
    strength = config.similarity_strength
    attempts: list[dict[str, Any]] = []
    final_candidate: CandidateBatch | None = None
    final_evaluation: EvaluationResult | None = None

    for attempt in range(config.max_retries + 1):
        candidate = sample_candidate(
            reference,
            plan,
            config,
            attempt=attempt,
            fitted=fitted,
            similarity_strength=strength,
        )
        evaluation = evaluate_candidate(reference, candidate, config)
        attempts.append(
            {
                "attempt": attempt,
                "similarity_strength": strength,
                **json_safe(
                    {
                        "metrics": evaluation.metrics,
                        "violations": evaluation.violations,
                        "passed": evaluation.passed,
                    }
                ),
            }
        )
        final_candidate = candidate
        final_evaluation = evaluation
        if evaluation.passed or not config.strict_evaluation:
            break
        if attempt < config.max_retries:
            strength = _next_strength(strength, evaluation)

    if final_candidate is None or final_evaluation is None:
        raise RuntimeError("simulation pipeline produced no candidate")
    if config.strict_evaluation and not final_evaluation.passed:
        raise EvaluationGateError(
            f"no candidate passed quality gates after {len(attempts)} attempt(s); "
            f"last violations={final_evaluation.violations}"
        )

    warnings = list(final_evaluation.violations)
    actual_quota = [
        {
            "conditions": {
                column: json_safe(value)
                for column, value in zip(plan.columns, group.values)
            },
            "n_cells": group.n_cells,
        }
        for group in plan.groups
    ]
    report = json_safe(
        {
            "version": "0.1.0",
            "ref_path": reference.path,
            "seed": config.seed,
            "condition_spec": None
            if spec is None
            else {
                "columns": spec.columns,
                "joint": None
                if spec.joint is None
                else [
                    {"conditions": list(key), "n_cells": value}
                    for key, value in spec.joint.items()
                ],
                "marginal": spec.marginal,
            },
            "actual_quota": actual_quota,
            "similarity_strength": strength,
            "requested_similarity_strength": config.similarity_strength,
            "distribution_model": config.distribution_model,
            "geometry_model": config.geometry_model,
            "geometry_neighbors": config.geometry_neighbors,
            "local_profile_mix": config.local_profile_mix,
            "group_prior_mix": config.group_prior_mix,
            "geometry_latent_k": config.geometry_latent_k,
            "geometry_hvg_count": config.geometry_hvg_count,
            "count_noise_model": config.count_noise_model,
            "novelty_mode": config.novelty_mode,
            "novelty_strength": config.effective_novelty_strength,
            "retries": len(attempts) - 1,
            "evaluation": {
                "passed": final_evaluation.passed,
                "metrics": final_evaluation.metrics,
                "violations": final_evaluation.violations,
                "attempts": attempts,
            },
            "source_pool_sizes": {
                str(group.values): int(group.source_indices.size)
                for group in plan.groups
            },
            "warnings": warnings,
            "output_path": output_path,
            "report_path": report_path,
        }
    )
    manifest = {
        "version": report["version"],
        "ref_path": report["ref_path"],
        "seed": report["seed"],
        "condition_columns": list(plan.columns),
        "actual_quota": {
            (
                " | ".join(
                    f"{column}={value}"
                    for column, value in zip(plan.columns, group.values)
                )
                or "unrestricted"
            ): group.n_cells
            for group in plan.groups
        },
        "similarity_strength": strength,
        "distribution_model": config.distribution_model,
        "geometry_model": config.geometry_model,
        "geometry_neighbors": config.geometry_neighbors,
        "local_profile_mix": config.local_profile_mix,
        "group_prior_mix": config.group_prior_mix,
        "geometry_latent_k": config.geometry_latent_k,
        "geometry_hvg_count": config.geometry_hvg_count,
        "count_noise_model": config.count_noise_model,
        "novelty_mode": config.novelty_mode,
        "novelty_strength": config.effective_novelty_strength,
        "retries": report["retries"],
        "evaluation": {
            "passed": report["evaluation"]["passed"],
            "metrics": report["evaluation"]["metrics"],
            "violations": report["evaluation"]["violations"],
        },
    }
    generated = assemble_anndata(
        reference,
        final_candidate,
        plan,
        config,
        similarity_strength=strength,
        manifest=json_safe(manifest),
    )
    return PipelineResult(generated=generated, report=report)


def _next_strength(current: float, evaluation: EvaluationResult) -> float:
    metrics = evaluation.metrics
    too_similar = (
        ("classifier_auc" in evaluation.violations and metrics["classifier_auc"] < 0.55)
        or ("nn_ratio" in evaluation.violations and metrics["nn_ratio"] < 0.50)
    )
    delta = 0.10 if too_similar else -0.10
    return float(np.clip(current + delta, 0.0, 1.0))
