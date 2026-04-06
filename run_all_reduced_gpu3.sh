#!/bin/bash
set -euo pipefail

# Use only physical GPU 3.
# In each Python script, device=0 will map to this visible GPU.
export CUDA_VISIBLE_DEVICES=1
PYTHON_BIN_DEFAULT="$(command -v python3 || true)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: executable Python not found. Set PYTHON_BIN to an absolute path."
  exit 1
fi
export PYTHON_BIN

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

get_results_csv_path() {
  local dir="$1"
  echo "${ROOT_DIR}/reduced_train/${dir}_top3_subset_results.csv"
}

get_expected_success_count() {
  local dir="$1"
  case "${dir}" in
    v8|v10|v11)
      # baseline 1개 + top3 3개, 각 4개 ratio(75/50/25/10) => 16
      echo "16"
      ;;
    v8_diff|v10_diff|v11_diff)
      # top3 3개, 각 4개 ratio(75/50/25/10) => 12
      echo "12"
      ;;
    *)
      echo "0"
      ;;
  esac
}

should_skip_step() {
  local dir="$1"
  local results_csv
  results_csv="$(get_results_csv_path "${dir}")"
  local expected
  expected="$(get_expected_success_count "${dir}")"

  if [[ ! -f "${results_csv}" ]]; then
    return 1
  fi

  if [[ "${expected}" -le 0 ]]; then
    return 1
  fi

  local success_count
  success_count="$("${PYTHON_BIN}" - "${results_csv}" <<'PY'
import csv
import sys

path = sys.argv[1]
count = 0
with open(path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("status", "").strip().lower() == "success":
            count += 1
print(count)
PY
)"

  if [[ "${success_count}" -ge "${expected}" ]]; then
    echo "SKIP: ${dir} already completed (${success_count}/${expected} success rows)."
    return 0
  fi

  return 1
}

run_step() {
  local dir="$1"
  local script="$2"

  echo "=========================================="
  echo "Running: ${dir}/${script}"
  echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "=========================================="

  if should_skip_step "${dir}"; then
    return 0
  fi

  cd "${ROOT_DIR}/${dir}"
  bash "${script}"
}

# Sequential execution (6 targets)
run_step "v8" "run_grid_search_npp.sh"
run_step "v8_diff" "run_grid_search_diff.sh"
run_step "v10" "run_grid_search_npp.sh"
run_step "v10_diff" "run_grid_search_diff.sh"
run_step "v11" "run_grid_search_npp.sh"
run_step "v11_diff" "run_grid_search_diff.sh"

echo "=========================================="
echo "All 6 runs completed successfully."
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
