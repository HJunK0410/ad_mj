#!/bin/bash
# ============================================================
# crack-seg 데이터셋 Detection 실험 (YOLOv8 + 차분 Loss)
# baseline + best HP 순차 실행
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"

DEVICE="${1:-0}"

echo "=========================================="
echo " crack-seg 실험 시작 (v8_diff / 차분 Loss)"
echo " Device: ${DEVICE}"
echo "=========================================="

echo ""
echo "[1/2] Baseline 실험..."
python train_crack.py --experiment_type=baseline --device=${DEVICE}

echo ""
echo "[2/2] Best Diff HP 실험..."
python train_crack.py --experiment_type=best --device=${DEVICE}

echo ""
echo "=========================================="
echo " 모든 crack-seg 실험 완료 (v8_diff)"
echo "=========================================="
