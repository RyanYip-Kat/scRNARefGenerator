"""Condition quota validation and exact source-pool planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
import scipy.sparse as sp

from .config import ConditionSpec, GeneratorConfig
from .errors import (
    ConditionCategoryNotFoundError,
    ConditionColumnNotFoundError,
    InfeasibleConditionError,
)
from .validation import ReferenceData, resolve_reference


@dataclass(frozen=True)
class TargetGroup:
    values: tuple[Any, ...]
    n_cells: int
    source_indices: np.ndarray


@dataclass(frozen=True)
class ConditionPlan:
    columns: tuple[str, ...]
    groups: tuple[TargetGroup, ...]
    total_cells: int
    mode: Literal["unrestricted", "joint", "marginal"]

    @property
    def actual_quota(self) -> dict[tuple[Any, ...], int]:
        return {group.values: group.n_cells for group in self.groups}


def condition_spec_from_reference(
    reference: ReferenceData | ad.AnnData | str | PathLike[str],
    condition_col: str,
    *,
    top_n: int = 10,
    categories: list[Any] | None = None,
    fraction: float = 0.60,
) -> ConditionSpec:
    """Build per-category quotas as a fraction of reference cell counts.

    Without explicit categories, categories are ranked by descending reference
    frequency before selecting ``top_n``. Explicit categories preserve the
    caller-provided order. Integer quotas use floor rounding.
    """

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be greater than 0 and at most 1")
    data = (
        reference
        if isinstance(reference, ReferenceData)
        else resolve_reference(reference, GeneratorConfig())
    )
    if condition_col not in data.obs:
        raise ConditionColumnNotFoundError(
            f"condition column {condition_col!r} is not present in reference.obs"
        )
    counts = data.obs[condition_col].dropna().value_counts()
    ranked = [
        _python_scalar(category)
        for category in sorted(
            counts.index,
            key=lambda category: (-int(counts.loc[category]), repr(category)),
        )
    ]
    if categories is None:
        selected = ranked[:top_n]
    else:
        selected = [_python_scalar(category) for category in categories]
        if not selected:
            raise ValueError("categories must not be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("categories must be unique")
        if len(selected) > top_n:
            raise ValueError("number of categories cannot exceed top_n")
        unknown = [category for category in selected if category not in counts.index]
        if unknown:
            raise ConditionCategoryNotFoundError(
                f"categories are absent from reference: {unknown!r}"
            )

    quotas = {
        category: int(math.floor(int(counts.loc[category]) * fraction))
        for category in selected
    }
    zero_quota = [category for category, quota in quotas.items() if quota <= 0]
    if zero_quota:
        raise ValueError(
            f"fraction={fraction} produces zero cells for categories {zero_quota!r}"
        )
    return ConditionSpec(
        columns=[condition_col],
        joint={(category,): quota for category, quota in quotas.items()},
    )


def condition_spec_from_marginal_fraction(
    reference: ReferenceData | ad.AnnData | str | PathLike[str],
    columns: list[str],
    *,
    fraction: float = 0.60,
) -> ConditionSpec:
    """Build per-column category quotas with one exact shared total."""

    if not columns or len(set(columns)) != len(columns):
        raise ValueError("columns must be non-empty and unique")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be greater than 0 and at most 1")
    data = (
        reference
        if isinstance(reference, ReferenceData)
        else resolve_reference(reference, GeneratorConfig())
    )
    total_target = int(math.floor(data.n_obs * fraction))
    if total_target <= 0:
        raise ValueError("fraction produces zero simulated cells")
    marginal: dict[str, dict[Any, int]] = {}
    for column in columns:
        if column not in data.obs:
            raise ConditionColumnNotFoundError(
                f"condition column {column!r} is not present in reference.obs"
            )
        if data.obs[column].isna().any():
            raise ValueError(f"condition column {column!r} contains missing values")
        counts = data.obs[column].value_counts()
        marginal[column] = _fractional_quotas(counts, fraction, total_target)
    return ConditionSpec(columns=list(columns), marginal=marginal)


def _fractional_quotas(
    counts: pd.Series,
    fraction: float,
    total_target: int,
) -> dict[Any, int]:
    categories = sorted(counts.index, key=repr)
    raw = {
        category: float(counts.loc[category]) * fraction
        for category in categories
    }
    quotas = {
        category: int(math.floor(raw[category]))
        for category in categories
    }
    remaining = total_target - sum(quotas.values())
    if remaining < 0 or total_target > int(counts.sum()):
        raise InfeasibleConditionError(
            f"fraction {fraction} cannot allocate {total_target} cells"
        )
    ranked = sorted(
        categories,
        key=lambda category: (
            -(raw[category] - math.floor(raw[category])),
            repr(category),
        ),
    )
    for category in ranked[:remaining]:
        quotas[category] += 1
    return {
        _python_scalar(category): int(quota)
        for category, quota in quotas.items()
    }


def build_condition_plan(
    reference: ReferenceData,
    spec: ConditionSpec | None,
    n_cells: int | None,
    config: GeneratorConfig | None = None,
) -> ConditionPlan:
    """Resolve public quotas into exact, non-overlapping source-index pools."""

    config = config or GeneratorConfig()
    if n_cells is not None and (
        isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells <= 0
    ):
        raise ValueError("n_cells must be a positive integer")

    if spec is None:
        if n_cells is None:
            raise ValueError("n_cells is required when condition_spec is None")
        if n_cells > reference.n_obs:
            raise InfeasibleConditionError(
                f"requested {n_cells} cells but reference has {reference.n_obs}"
            )
        return ConditionPlan(
            columns=(),
            groups=(
                TargetGroup(
                    values=(),
                    n_cells=n_cells,
                    source_indices=np.arange(reference.n_obs, dtype=np.int64),
                ),
            ),
            total_cells=n_cells,
            mode="unrestricted",
        )

    _validate_columns(reference, spec.columns)
    eligible = np.ones(reference.n_obs, dtype=bool)
    for column in spec.columns:
        missing = reference.obs[column].isna().to_numpy()
        if missing.any():
            if config.na_policy == "error":
                raise ValueError(
                    f"condition column {column!r} contains missing values; "
                    "set na_policy='drop' to exclude them"
                )
            eligible &= ~missing

    if spec.joint is not None:
        plan = _joint_plan(reference, spec, eligible)
    else:
        plan = _marginal_plan(reference, spec, eligible)
    if n_cells is not None and n_cells != plan.total_cells:
        raise InfeasibleConditionError(
            f"n_cells={n_cells} conflicts with quota total={plan.total_cells}"
        )
    return plan


def _validate_columns(reference: ReferenceData, columns: list[str]) -> None:
    for column in columns:
        if column not in reference.obs.columns:
            raise ConditionColumnNotFoundError(
                f"condition column {column!r} is not present in reference.obs"
            )


def _joint_plan(
    reference: ReferenceData,
    spec: ConditionSpec,
    eligible: np.ndarray,
) -> ConditionPlan:
    quotas = {tuple(key): int(value) for key, value in (spec.joint or {}).items()}
    total = sum(quotas.values())
    groups: list[TargetGroup] = []
    used = np.zeros(reference.n_obs, dtype=bool)
    for values, n_group in quotas.items():
        sources = _matching_indices(reference, spec.columns, values, eligible)
        if sources.size == 0:
            raise ConditionCategoryNotFoundError(
                f"condition combination {values!r} does not occur in the reference"
            )
        if n_group > sources.size:
            raise InfeasibleConditionError(
                f"quota {n_group} exceeds the {sources.size} available cells for {values!r}"
            )
        if used[sources].any():
            raise InfeasibleConditionError("source pools overlap")
        used[sources] = True
        groups.append(TargetGroup(values=values, n_cells=n_group, source_indices=sources))
    if total > eligible.sum():
        raise InfeasibleConditionError(
            f"quota total {total} exceeds {int(eligible.sum())} eligible reference cells"
        )
    return ConditionPlan(
        columns=tuple(spec.columns),
        groups=tuple(groups),
        total_cells=total,
        mode="joint",
    )


def _marginal_plan(
    reference: ReferenceData,
    spec: ConditionSpec,
    eligible: np.ndarray,
) -> ConditionPlan:
    columns = list(spec.columns)
    margins = {
        column: {_python_scalar(k): int(v) for k, v in values.items()}
        for column, values in (spec.marginal or {}).items()
    }
    row_totals = [sum(margins[column].values()) for column in columns]
    if len(set(row_totals)) != 1:
        raise InfeasibleConditionError(
            f"marginal totals differ across columns: {dict(zip(columns, row_totals))}"
        )
    total = row_totals[0]
    if total <= 0:
        raise InfeasibleConditionError("marginal quota total must be positive")

    for column, categories in margins.items():
        available = set(reference.obs.loc[eligible, column].tolist())
        absent = [category for category in categories if category not in available]
        if absent:
            raise ConditionCategoryNotFoundError(
                f"categories {absent!r} do not occur in column {column!r}"
            )

    combinations, capacities, source_pools = _observed_combinations(
        reference, columns, eligible
    )
    n_vars = len(combinations)
    if total > int(capacities.sum()):
        raise InfeasibleConditionError(
            f"quota total {total} exceeds {int(capacities.sum())} eligible cells"
        )

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    target_counts = total * capacities / max(int(capacities.sum()), 1)
    row_id = 0

    # x[j]: integer joint counts; d[j]: absolute deviation from reference-
    # proportional allocation.
    rows.extend([row_id] * n_vars)
    cols.extend(range(n_vars))
    values.extend([1.0] * n_vars)
    lower.append(float(total))
    upper.append(float(total))
    row_id += 1

    for column_index, column in enumerate(columns):
        for category, quota in margins[column].items():
            for var_index, combination in enumerate(combinations):
                if combination[column_index] == category:
                    rows.append(row_id)
                    cols.append(var_index)
                    values.append(1.0)
            lower.append(float(quota))
            upper.append(float(quota))
            row_id += 1
    for var_index in range(n_vars):
        # d - x >= -target
        rows.extend([row_id, row_id])
        cols.extend([var_index, n_vars + var_index])
        values.extend([-1.0, 1.0])
        lower.append(-float(target_counts[var_index]))
        upper.append(float("inf"))
        row_id += 1
        # d + x >= target
        rows.extend([row_id, row_id])
        cols.extend([var_index, n_vars + var_index])
        values.extend([1.0, 1.0])
        lower.append(float(target_counts[var_index]))
        upper.append(float("inf"))
        row_id += 1

    constraint_matrix = sp.csr_matrix(
        (values, (rows, cols)), shape=(row_id, n_vars * 2)
    )
    objective = np.concatenate([np.zeros(n_vars), np.ones(n_vars)])
    variable_lower = np.zeros(n_vars * 2)
    variable_upper = np.concatenate(
        [capacities.astype(float), np.full(n_vars, np.inf)]
    )
    result = milp(
        c=objective,
        integrality=np.concatenate(
            [np.ones(n_vars, dtype=np.uint8), np.zeros(n_vars, dtype=np.uint8)]
        ),
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(constraint_matrix, np.array(lower), np.array(upper)),
        options={"presolve": True, "time_limit": 30.0},
    )
    if not result.success or result.x is None:
        support = {
            tuple(combination): int(capacity)
            for combination, capacity in zip(combinations, capacities)
        }
        raise InfeasibleConditionError(
            "marginal quotas cannot be met without changing quotas; "
            f"available joint support is {support!r}"
        )

    counts = np.rint(result.x[:n_vars]).astype(np.int64)
    groups: list[TargetGroup] = []
    for combination, quota, source_pool in zip(
        combinations, counts, source_pools
    ):
        if quota > 0:
            groups.append(
                TargetGroup(
                    values=tuple(_python_scalar(v) for v in combination),
                    n_cells=int(quota),
                    source_indices=source_pool,
                )
            )
    if sum(group.n_cells for group in groups) != total:
        raise InfeasibleConditionError("internal marginal solver returned a wrong total")
    return ConditionPlan(
        columns=tuple(columns),
        groups=tuple(groups),
        total_cells=total,
        mode="marginal",
    )


def _observed_combinations(
    reference: ReferenceData,
    columns: list[str],
    eligible: np.ndarray,
) -> tuple[list[tuple[Any, ...]], np.ndarray, list[np.ndarray]]:
    frame = reference.obs.loc[eligible, columns]
    keys = list(frame.itertuples(index=False, name=None))
    unique = sorted(set(keys), key=repr)
    combinations: list[tuple[Any, ...]] = []
    capacities: list[int] = []
    pools: list[np.ndarray] = []
    eligible_indices = np.flatnonzero(eligible)
    for combination in unique:
        sources = eligible_indices[
            np.array([key == combination for key in keys], dtype=bool)
        ]
        combinations.append(tuple(_python_scalar(v) for v in combination))
        capacities.append(int(sources.size))
        pools.append(sources)
    return combinations, np.asarray(capacities, dtype=np.int64), pools


def _matching_indices(
    reference: ReferenceData,
    columns: list[str],
    values: tuple[Any, ...],
    eligible: np.ndarray,
) -> np.ndarray:
    mask = eligible.copy()
    for column, value in zip(columns, values):
        mask &= reference.obs[column].eq(value).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
