#!/bin/bash

# YOLOv12 NPP Loss 하이퍼파라미터 튜닝 스크립트
# Ultralytics Tuner를 사용하여 진화 알고리즘으로 하이퍼파라미터를 튜닝합니다.

# 환경 설정
export PYTHONPATH="/home/user/hyunjun/AD/v12:$PYTHONPATH"

# 스크립트 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 튜닝 반복 횟수 (기본값: 50)
ITERATIONS=${1:-50}

echo "=========================================="
echo "NPP Loss 하이퍼파라미터 튜닝 시작"
echo "=========================================="
echo "튜닝 반복 횟수: ${ITERATIONS}"
echo "시작 시간: $(date)"
echo "=========================================="
echo ""

# Python 튜닝 스크립트 실행
# conda 환경 활성화 (source activate 또는 conda run 사용)
if python tune_npp.py ${ITERATIONS}; then
    echo ""
    echo "=========================================="
    echo "튜닝 완료!"
    echo "=========================================="
    echo "완료 시간: $(date)"
    echo "결과는 runs/detect/npp_tune 디렉토리에 저장되었습니다."
    exit 0
else
    EXIT_CODE=$?
    echo ""
    echo "=========================================="
    echo "튜닝 실패! (Exit code: ${EXIT_CODE})"
    echo "=========================================="
    exit ${EXIT_CODE}
fi
