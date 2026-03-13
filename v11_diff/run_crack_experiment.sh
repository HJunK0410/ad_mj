#!/bin/bash

# Crack-seg 데이터셋 실험 스크립트 (v11_diff: Diff Best)
# 실험: Diff Best (alpha=0.1, beta=0.5)

# 환경 설정
export PYTHONPATH="/home/user/hyunjun/AD/v11_diff:$PYTHONPATH"

# 스크립트 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Crack-seg 데이터셋 실험 (v11_diff)"
echo "=========================================="
echo "시작 시간: $(date)"
echo "=========================================="
echo ""

# ---- 실험: Diff Best ----
echo "=========================================="
echo "[실험 1/1] Diff Best"
echo "  alpha=0.1, beta=0.5"
echo "=========================================="

python train_crack.py \
    --npp_alpha=0.1 \
    --npp_beta=0.5 \
    --npp_fpn_sources=16,19,22 \
    --batch=16 \
    --device=3 \
    --epochs=100 \
    --imgsz=640 \
    --workers=4

echo ""
echo "Diff Best 학습 완료: $(date)"
echo ""

echo "=========================================="
echo "모든 학습 완료!"
echo "=========================================="
echo "학습 완료 시간: $(date)"
echo ""
echo "학습된 모델:"
echo "  Diff Best: runs/detect/crack_diff_alpha0.1_beta0.5_fpn16_19_22/weights/best.pt"
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
