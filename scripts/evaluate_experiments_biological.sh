#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REAL="${REAL:-${ROOT}/testdata/pbmc5k_querydata.h5ad}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${ROOT}/outputs/experiments}"
EXPERIMENT_FILTER="${EXPERIMENT_FILTER:-experiment*}"
MAX_CELLS_PER_CLASS="${MAX_CELLS_PER_CLASS:-500}"
N_TOP_MARKERS="${N_TOP_MARKERS:-30}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/scrnarefgenerator-numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/scrnarefgenerator-matplotlib}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

while IFS= read -r simulated; do
  result_dir="$(dirname "${simulated}")"
  outdir="${result_dir}/eval_bio_v2"
  if [[ -f "${outdir}/summary.json" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP ${result_dir}"
    continue
  fi
  echo "EVALUATE ${result_dir}"
  "${PYTHON_BIN}" "${ROOT}/evaluate_sim_biological.py" \
    --real "${REAL}" \
    --sim "${simulated}" \
    --outdir "${outdir}" \
    --max-cells-per-class "${MAX_CELLS_PER_CLASS}" \
    --n-top-markers "${N_TOP_MARKERS}" \
    --seed 42
done < <(
  find "${EXPERIMENT_ROOT}" \
    -mindepth 3 -maxdepth 3 \
    -path "*/${EXPERIMENT_FILTER}/*/simulated.h5ad" \
    | sort
)

"${PYTHON_BIN}" "${ROOT}/scripts/collect_biological_results.py" \
  "${EXPERIMENT_ROOT}" \
  --output-prefix "${EXPERIMENT_ROOT}/biological_summary_all"
