"""Public simulation facade."""

from __future__ import annotations

from os import PathLike
from typing import Any

import anndata as ad

from .assembly import validate_barcode_policy
from .conditions import build_condition_plan
from .config import ConditionSpec, GeneratorConfig
from .io import validate_output_paths, write_outputs
from .pipeline import run_pipeline
from .validation import resolve_reference


class scRNARefGenerator:
    """Simulate conditional sparse count data from a validated reference."""

    def __init__(
        self,
        ref: ad.AnnData | str | PathLike[str],
        config: GeneratorConfig | None = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.reference = resolve_reference(ref, self.config)

    def simulate(
        self,
        condition_spec: ConditionSpec | None = None,
        n_cells: int | None = None,
        output_path: str | PathLike[str] | None = None,
        report_path: str | PathLike[str] | None = None,
    ) -> ad.AnnData:
        if (
            condition_spec is not None
            and self.config.condition_cols is not None
            and list(condition_spec.columns) != list(self.config.condition_cols)
        ):
            raise ValueError(
                "ConditionSpec.columns must match GeneratorConfig.condition_cols"
            )
        validate_output_paths(output_path, report_path, self.config.overwrite)
        plan = build_condition_plan(
            self.reference,
            condition_spec,
            n_cells,
            config=self.config,
        )
        validate_barcode_policy(self.reference, plan, self.config)
        result = run_pipeline(
            self.reference,
            plan,
            condition_spec,
            self.config,
            output_path=None if output_path is None else str(output_path),
            report_path=None if report_path is None else str(report_path),
        )
        write_outputs(
            result.generated,
            result.report,
            output_path,
            report_path,
            self.config.overwrite,
        )
        return result.generated
