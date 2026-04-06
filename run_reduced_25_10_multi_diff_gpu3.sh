#!/bin/bash
# reduced train: 데이터 25%·10%만, epochs=400, patience=20 (early stop)
# baseline 없음. NPP 실험과 DIFF 실험은 **서로 독립** (곱하지 않음).
#
# 1) NPP — v8, v10, v11 만:
#    lambda(2d)×lambda(1d)×mask × 3버전 × 2비율 = 5×5×3×3×2 = 450
#    (half 스크립트와 동일 매핑: product 첫 인자 → npp_lambda_2d, 둘째 → npp_lambda_1d)
#
# 2) DIFF — v8_diff, v10_diff, v11_diff 만 (NPP 고정: l2d=0, l1d=0, mask=0.3, half diff 그리드와 동일 취지):
#    alpha×beta × 3버전 × 2비율 = 5×5×3×2 = 150
#
# 총 실행 수: 450 + 150 = 600

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PYTHON_BIN_DEFAULT="$(command -v python3 || true)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: executable Python not found. Set PYTHON_BIN to an absolute path."
  exit 1
fi
export PYTHON_BIN

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT_DIR

echo "=========================================="
echo "Reduced 25%/10% NPP grid + DIFF grid (separate, no baseline)"
echo "ROOT_DIR=${ROOT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

"${PYTHON_BIN}" - "${ROOT_DIR}" <<'PY'
import csv
import itertools
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT_DIR = ROOT / "reduced_train"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "reduced_25_10_fullgrid_results.csv"

# --- grid_search_npp_half.py (45-47) + 동일 product 매핑 ---
LAMBDA_FOR_2D = [0, 0.02, 0.04, 0.06, 0.08]  # half 파일의 lambda_1_values → train 2d
LAMBDA_FOR_1D = [0, 0.02, 0.04, 0.06, 0.08]  # half 파일의 lambda_2_values → train 1d
BBOX_MASK = [0.1, 0.2, 0.3]

# DIFF: alpha·beta 각 5단계 (5×5)
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]
BETA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]

# *_diff 전용: 차분만 스윕할 때 NPP 쪽 고정 (grid_search_diff_half baseline 분기와 동일)
DIFF_FIXED_L2D = 0.0
DIFF_FIXED_L1D = 0.0
DIFF_FIXED_MASK = 0.3

RATIOS = (25, 10)
EPOCHS = 400
PATIENCE = 20
DEVICE = 0
PYTHON_BIN = os.environ.get("PYTHON_BIN") or sys.executable

VERSIONS_NPP = {
    "v8": {"fpn": "15,18,21", "batch": 4},
    "v10": {"fpn": "16,19,22", "batch": 4},
    "v11": {"fpn": "16,19,22", "batch": 4},
}

NPP_COMBOS = list(itertools.product(LAMBDA_FOR_2D, LAMBDA_FOR_1D, BBOX_MASK))
DIFF_COMBOS = list(itertools.product(ALPHA_VALUES, BETA_VALUES))


def read_metrics(script_dir: Path, project_dir: str, run_name: str) -> dict:
    save_dir = script_dir / project_dir / run_name
    # NOTE:
    # torch.load(ckpt) 는 커스텀 hook/class 역직렬화 이슈로 실패할 수 있어
    # Ultralytics가 항상 남기는 results.csv 기반으로 메트릭을 읽는다.
    default = {
        "fitness": 0.0,
        "mAP50": 0.0,
        "mAP50-95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    results_csv = save_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"metrics file not found: {results_csv}")

    def _to_float(v, fallback=0.0):
        try:
            return float(v)
        except Exception:
            return fallback

    try:
        with open(results_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return default
        last = rows[-1]
        map50 = _to_float(last.get("metrics/mAP50(B)", 0.0))
        map50_95 = _to_float(last.get("metrics/mAP50-95(B)", 0.0))
        precision = _to_float(last.get("metrics/precision(B)", 0.0))
        recall = _to_float(last.get("metrics/recall(B)", 0.0))
        fitness = _to_float(last.get("fitness", map50_95), map50_95)
        return {
            "fitness": fitness,
            "mAP50": map50,
            "mAP50-95": map50_95,
            "precision": precision,
            "recall": recall,
        }
    except Exception as e:
        raise RuntimeError(f"failed to read metrics from {results_csv}: {e}") from e


def run_name_v8_diff(l2d: float, l1d: float, mask: float, fpn_csv: str, da: float, db: float) -> str:
    fpn = fpn_csv.replace(",", "_")
    if da > 0.0 or db > 0.0:
        if l2d > 0.0 or l1d > 0.0:
            return (
                f"train_combo_l2d{l2d}_l1d{l1d}_mask{mask}_fpn{fpn}_"
                f"dalpha{da}_dbeta{db}"
            )
        return f"train_diff_alpha{da}_beta{db}_mask{mask}_fpn{fpn}"
    return f"train_npp_l2d{l2d}_l1d{l1d}_mask{mask}_fpn{fpn}"


def run_name_v11_diff(l2d: float, l1d: float, mask: float, fpn_csv: str, alpha: float, beta: float) -> str:
    fpn = fpn_csv.replace(",", "_")
    _L2_DEF, _L1_DEF = 0.02, 0.06
    default_multi = abs(l2d - _L2_DEF) < 1e-9 and abs(l1d - _L1_DEF) < 1e-9
    if alpha > 0.0 or beta > 0.0:
        if default_multi or (l2d <= 0.0 and l1d <= 0.0):
            return f"train_diff_alpha{alpha}_beta{beta}_mask{mask}_fpn{fpn}"
        return (
            f"train_combo_l2d{l2d}_l1d{l1d}_mask{mask}_fpn{fpn}_alpha{alpha}_beta{beta}"
        )
    return f"train_npp_l2d{l2d}_l1d{l1d}_mask{mask}_fpn{fpn}"


def run_name_npp_only(l2d: float, l1d: float, mask: float, fpn_csv: str) -> str:
    fpn = fpn_csv.replace(",", "_")
    return f"train_npp_l2d{l2d}_l1d{l1d}_mask{mask}_fpn{fpn}"


def append_row(row: list) -> None:
    with open(RESULTS_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)


def main() -> None:
    npp_total = 3 * len(NPP_COMBOS) * len(RATIOS)
    diff_total = 3 * len(DIFF_COMBOS) * len(RATIOS)
    total = npp_total + diff_total
    print("결과 CSV:", RESULTS_CSV)
    print(
        f"NPP만: {npp_total} (버전당 {len(NPP_COMBOS) * len(RATIOS)}), "
        f"DIFF만: {diff_total} (버전당 {len(DIFF_COMBOS) * len(RATIOS)}), "
        f"합계: {total}"
    )
    print("시작:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    with open(RESULTS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            [
                "version",
                "mode",
                "dataset_ratio",
                "npp_lambda_2d",
                "npp_lambda_1d",
                "bbox_mask_weight",
                "diff_alpha",
                "diff_beta",
                "npp_fpn_sources",
                "data_yaml",
                "project_dir",
                "run_name",
                "fitness",
                "mAP50",
                "mAP50-95",
                "precision",
                "recall",
                "status",
                "save_dir",
            ]
        )

    step = 0

    for ver, cfg in VERSIONS_NPP.items():
        script_dir = ROOT / ver
        fpn = cfg["fpn"]
        batch = cfg["batch"]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
        for l2d, l1d, mask in NPP_COMBOS:
            l2d, l1d, mask = float(l2d), float(l1d), float(mask)
            run_name = run_name_npp_only(l2d, l1d, mask, fpn)
            for ratio in RATIOS:
                step += 1
                data_yaml = f"./data/data_{ratio}.yaml"
                project_dir = f"runs/detect_{ratio}"
                cmd = [
                    PYTHON_BIN,
                    "train.py",
                    f"--npp_lambda_2d={l2d}",
                    f"--npp_lambda_1d={l1d}",
                    f"--npp_bbox_mask_weight={mask}",
                    f"--npp_fpn_sources={fpn}",
                    f"--batch={batch}",
                    f"--device={DEVICE}",
                    f"--epochs={EPOCHS}",
                    f"--patience={PATIENCE}",
                    f"--data={data_yaml}",
                    f"--project={project_dir}",
                ]
                print(f"\n>>> [{step}/{total}] {ver} npp_only r={ratio} {run_name}")
                try:
                    subprocess.run(cmd, cwd=script_dir, env=env, check=True)
                    m = read_metrics(script_dir, project_dir, run_name)
                    status = "success"
                except subprocess.CalledProcessError as e:
                    print(f"  실패: {e}")
                    m = {
                        "fitness": 0.0,
                        "mAP50": 0.0,
                        "mAP50-95": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                    }
                    status = "failed"
                append_row(
                    [
                        ver,
                        "npp_only",
                        ratio,
                        l2d,
                        l1d,
                        mask,
                        "",
                        "",
                        fpn,
                        data_yaml,
                        project_dir,
                        run_name,
                        round(m["fitness"], 5),
                        round(m["mAP50"], 5),
                        round(m["mAP50-95"], 5),
                        round(m["precision"], 5),
                        round(m["recall"], 5),
                        status,
                        f"{project_dir}/{run_name}",
                    ]
                )

    l2d_f = float(DIFF_FIXED_L2D)
    l1d_f = float(DIFF_FIXED_L1D)
    mask_f = float(DIFF_FIXED_MASK)
    for ver, cfg in (
        ("v8_diff", VERSIONS_NPP["v8"]),
        ("v10_diff", VERSIONS_NPP["v10"]),
        ("v11_diff", VERSIONS_NPP["v11"]),
    ):
        script_dir = ROOT / ver
        fpn = cfg["fpn"]
        batch = cfg["batch"]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
        is_v11 = ver == "v11_diff"
        for da, db in DIFF_COMBOS:
            da, db = float(da), float(db)
            if is_v11:
                run_name = run_name_v11_diff(l2d_f, l1d_f, mask_f, fpn, da, db)
            else:
                run_name = run_name_v8_diff(l2d_f, l1d_f, mask_f, fpn, da, db)
            for ratio in RATIOS:
                step += 1
                data_yaml = f"./data/data_{ratio}.yaml"
                project_dir = f"runs/detect_{ratio}"
                cmd = [
                    PYTHON_BIN,
                    "train.py",
                    f"--npp_lambda_2d={l2d_f}",
                    f"--npp_lambda_1d={l1d_f}",
                    f"--npp_bbox_mask_weight={mask_f}",
                    f"--npp_fpn_sources={fpn}",
                    f"--npp_alpha={da}",
                    f"--npp_beta={db}",
                    f"--batch={batch}",
                    f"--device={DEVICE}",
                    f"--epochs={EPOCHS}",
                    f"--patience={PATIENCE}",
                    f"--data={data_yaml}",
                    f"--project={project_dir}",
                ]
                print(
                    f"\n>>> [{step}/{total}] {ver} diff_only r={ratio} "
                    f"a={da} b={db} (npp fixed l2d={l2d_f} l1d={l1d_f} mask={mask_f})"
                )
                try:
                    subprocess.run(cmd, cwd=script_dir, env=env, check=True)
                    m = read_metrics(script_dir, project_dir, run_name)
                    status = "success"
                except subprocess.CalledProcessError as e:
                    print(f"  실패: {e}")
                    m = {
                        "fitness": 0.0,
                        "mAP50": 0.0,
                        "mAP50-95": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                    }
                    status = "failed"
                append_row(
                    [
                        ver,
                        "diff_only",
                        ratio,
                        l2d_f,
                        l1d_f,
                        mask_f,
                        da,
                        db,
                        fpn,
                        data_yaml,
                        project_dir,
                        run_name,
                        round(m["fitness"], 5),
                        round(m["mAP50"], 5),
                        round(m["mAP50-95"], 5),
                        round(m["precision"], 5),
                        round(m["recall"], 5),
                        status,
                        f"{project_dir}/{run_name}",
                    ]
                )

    print("\n완료:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("결과:", RESULTS_CSV)


if __name__ == "__main__":
    main()
PY

echo "=========================================="
echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
