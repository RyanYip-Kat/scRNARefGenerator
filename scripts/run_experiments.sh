#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
REF="${REF:-${ROOT}/testdata/pbmc5k_querydata.h5ad}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/experiments}"
FORCE_ARGS=()
if [[ "${FORCE:-0}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

for subset in 5 8 10 12 15 all; do
  "${PYTHON_BIN}" "${ROOT}/scripts/run_experiment.py" \
    --experiment 1 \
    --ref "${REF}" \
    --outdir "${OUT_ROOT}/experiment1/subset_${subset}" \
    --subset "${subset}" \
    --similarity-strength 0.7 \
    "${FORCE_ARGS[@]}"
done

for strength in 0.5 0.6 0.7 0.8 0.9; do
  "${PYTHON_BIN}" "${ROOT}/scripts/run_experiment.py" \
    --experiment 2 \
    --ref "${REF}" \
    --outdir "${OUT_ROOT}/experiment2/strength_${strength}" \
    --similarity-strength "${strength}" \
    "${FORCE_ARGS[@]}"
done

for n_cells in 1000 3000 5000 10000 20000; do
  "${PYTHON_BIN}" "${ROOT}/scripts/run_experiment.py" \
    --experiment 3 \
    --ref "${REF}" \
    --outdir "${OUT_ROOT}/experiment3/n_cells_${n_cells}" \
    --n-cells "${n_cells}" \
    --similarity-strength 0.7 \
    "${FORCE_ARGS[@]}"
done
