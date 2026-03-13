#!/bin/bash

# Crack-seg 데이터셋 실험 스크립트 (v11: Baseline + NPP Best)
# 실험 1: Baseline (순수 YOLOv11m, NPP loss 비활성)
# 실험 2: NPP Best (l2d=0.02, l1d=0.08, mask=0.2)

# 환경 설정
export PYTHONPATH="/home/user/hyunjun/AD/v11:$PYTHONPATH"

# 스크립트 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Crack-seg 데이터셋 실험 (v11)"
echo "=========================================="
echo "시작 시간: $(date)"
echo "=========================================="
echo ""

# ---- 실험 1: Baseline ----
echo "=========================================="
echo "[실험 1/2] Baseline (순수 YOLOv11m)"
echo "  lambda_2d=0.0, lambda_1d=0.0, mask=0.0"
echo "=========================================="

python train_crack.py \
    --npp_lambda_2d=0.0 \
    --npp_lambda_1d=0.0 \
    --npp_bbox_mask_weight=0.0 \
    --npp_fpn_sources=16,19,22 \
    --batch=16 \
    --device=3 \
    --epochs=100 \
    --imgsz=640 \
    --workers=4

echo ""
echo "Baseline 학습 완료: $(date)"
echo ""

# ---- 실험 2: NPP Best ----
echo "=========================================="
echo "[실험 2/2] NPP Best"
echo "  lambda_2d=0.02, lambda_1d=0.08, mask=0.2"
echo "=========================================="

python train_crack.py \
    --npp_lambda_2d=0.02 \
    --npp_lambda_1d=0.08 \
    --npp_bbox_mask_weight=0.2 \
    --npp_fpn_sources=16,19,22 \
    --batch=16 \
    --device=3 \
    --epochs=100 \
    --imgsz=640 \
    --workers=4

echo ""
echo "NPP Best 학습 완료: $(date)"
echo ""

echo "=========================================="
echo "모든 학습 완료!"
echo "=========================================="
echo "학습 완료 시간: $(date)"
echo ""
echo "학습된 모델:"
echo "  Baseline:  runs/detect/crack_baseline/weights/best.pt"
echo "  NPP Best:  runs/detect/crack_npp_l2d0.02_l1d0.08_mask0.2_fpn16_19_22/weights/best.pt"
echo ""

# ---- Test 데이터셋 평가 ----
echo "=========================================="
echo "Test 데이터셋 평가 시작"
echo "=========================================="

python test_crack.py

echo ""
echo "=========================================="
echo "전체 파이프라인 완료! (학습 + 테스트)"
echo "=========================================="
echo "완료 시간: $(date)"
echo "결과 파일: crack_test_results.csv"
echo "=========================================="
