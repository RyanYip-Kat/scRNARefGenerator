#!/usr/bin/env python3
"""Collect per-experiment biological summaries into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "biological_evaluation_passed",
    "celltype_js_distance",
    "gene_mean_spearman",
    "gene_log_var_spearman",
    "gene_zero_frac_spearman",
    "marker_auroc_macro_real",
    "marker_auroc_macro_sim",
    "marker_auroc_macro_sim_shuffled",
    "marker_direction_rate_macro_sim",
    "real_holdout_balanced_acc",
    "sim_transfer_balanced_acc",
    "sim_transfer_retention_vs_holdout",
    "nearest_real_same_celltype_fraction",
    "nn_distance_ratio_sim_over_real",
    "heterogeneity_ratio_median",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("experiment*/**/eval_bio_v2/summary.json")):
        result_dir = path.parent
        experiment_path = result_dir.parent
        experiment_summary = experiment_path / "experiment_summary.json"
        experiment = (
            json.loads(experiment_summary.read_text())
            if experiment_summary.exists()
            else {}
        )
        summary = json.loads(path.read_text())
        row = {
            "experiment": experiment.get("experiment"),
            "setting": experiment_path.name,
            "subset": experiment.get("subset"),
            "similarity_strength": experiment.get("similarity_strength"),
            "n_cells_generated": experiment.get("n_cells_generated"),
            "condition_columns": "|".join(
                experiment.get("condition_columns") or []
            ),
            "summary_path": str(path),
        }
        row.update({metric: summary.get(metric) for metric in METRICS})
        for check, passed in summary.get("checks", {}).items():
            row[f"check_{check}"] = passed
        rows.append(row)
    if not rows:
        raise SystemExit(f"no biological summaries found under {args.root}")
    frame = pd.DataFrame(rows).sort_values(
        ["experiment", "setting"], na_position="last"
    )
    prefix = args.output_prefix or (args.root / "biological_summary_all")
    frame.to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(frame)} rows to {prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
