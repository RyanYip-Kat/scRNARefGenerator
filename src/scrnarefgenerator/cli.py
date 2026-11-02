"""Command-line interface for scRNARefGenerator."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from .api import scRNARefGenerator
from .config import ConditionSpec, GeneratorConfig


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scrnarefgen")
    subparsers = parser.add_subparsers(dest="command", required=True)
    simulate = subparsers.add_parser("simulate", help="generate simulated counts")
    simulate.add_argument("--ref", required=True, help="reference .h5ad path")
    simulate.add_argument("--out", required=True, help="output .h5ad path")
    simulate.add_argument("--report", help="output JSON report path")
    simulate.add_argument("--condition-col", action="append", default=[])
    simulate.add_argument("--condition-json")
    simulate.add_argument("--n-cells", type=int)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--similarity-strength", type=float, default=0.70)
    simulate.add_argument("--barcode-suffix", default="_sim")
    simulate.add_argument(
        "--distribution-model",
        choices=["hierarchical_nb", "hierarchical_zinb"],
        default="hierarchical_nb",
    )
    simulate.add_argument("--latent-k", type=int, default=0)
    simulate.add_argument("--max-retries", type=int, default=3)
    simulate.add_argument("--max-cells-for-fitting", type=int, default=10000)
    simulate.add_argument("--strict-evaluation", type=_boolean, default=True)
    simulate.add_argument("--counts-layer", default="counts")
    simulate.add_argument(
        "--collision-policy",
        choices=["error", "append_uuid"],
        default="error",
    )
    simulate.add_argument("--overwrite", action="store_true")
    simulate.set_defaults(handler=_simulate)
    return parser


def _simulate(args: argparse.Namespace) -> int:
    spec = _condition_spec(args.condition_col, args.condition_json)
    config = GeneratorConfig(
        counts_layer=args.counts_layer,
        barcode_suffix=args.barcode_suffix,
        seed=args.seed,
        similarity_strength=args.similarity_strength,
        distribution_model=args.distribution_model,
        latent_k=args.latent_k,
        max_retries=args.max_retries,
        max_cells_for_fitting=args.max_cells_for_fitting,
        strict_evaluation=args.strict_evaluation,
        collision_policy=args.collision_policy,
        overwrite=args.overwrite,
    )
    generated = scRNARefGenerator(args.ref, config).simulate(
        condition_spec=spec,
        n_cells=args.n_cells,
        output_path=args.out,
        report_path=args.report,
    )
    print(f"wrote {generated.n_obs} simulated cells to {args.out}")
    return 0


def _condition_spec(columns: list[str], payload: str | None) -> ConditionSpec | None:
    if not columns and payload is None:
        return None
    if not columns or payload is None:
        raise ValueError("--condition-col and --condition-json must be used together")
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise ValueError("--condition-json must decode to an object")
    if len(columns) == 1:
        column = columns[0]
        categories = parsed.get(column, parsed)
        if not isinstance(categories, dict) or not categories:
            raise ValueError("single-column condition JSON must map categories to counts")
        return ConditionSpec(
            columns=columns,
            joint={(category,): int(count) for category, count in categories.items()},
        )
    if set(parsed) != set(columns):
        raise ValueError(
            "multi-column CLI conditions require marginal JSON keyed by condition columns"
        )
    return ConditionSpec(columns=columns, marginal=parsed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
