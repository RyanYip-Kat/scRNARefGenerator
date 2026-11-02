"""Atomic output writing for AnnData and JSON reports."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np

from .errors import OutputExistsError


def validate_output_paths(
    output_path: str | os.PathLike[str] | None,
    report_path: str | os.PathLike[str] | None,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for path in (output_path, report_path):
        if path is not None and Path(path).expanduser().exists():
            raise OutputExistsError(
                f"output path {Path(path).expanduser()} already exists; "
                "set overwrite=True to replace it"
            )


def write_outputs(
    generated: ad.AnnData,
    report: Mapping[str, Any],
    output_path: str | os.PathLike[str] | None,
    report_path: str | os.PathLike[str] | None,
    overwrite: bool,
) -> None:
    validate_output_paths(output_path, report_path, overwrite)
    if output_path is not None:
        _write_h5ad_atomic(generated, Path(output_path).expanduser())
    if report_path is not None:
        _write_json_atomic(json_safe(report), Path(report_path).expanduser())


def _write_h5ad_atomic(generated: ad.AnnData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(handle)
    try:
        generated.write_h5ad(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [json_safe(item) for item in value]
    return str(value)
