#!/bin/bash

# YOLOv8 차분 로스(Diff Loss) 하이퍼파라미터 Grid Search 스크립트
# 모든 조합을 체계적으로 테스트합니다.

# 환경 설정
export PYTHONPATH="/home/user/hyunjun/AD/v8_diff:$PYTHONPATH"

# 스크립트 디렉토리 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=========================================="
echo "차분 로스(Diff Loss) 하이퍼파라미터 Grid Search 시작"
echo "=========================================="
echo "시작 시간: $(date)"
echo "=========================================="
echo ""

PYTHON_BIN_DEFAULT="$(command -v python3 || true)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "Python 실행 파일을 찾지 못했습니다. PYTHON_BIN을 절대경로로 지정하세요."
    exit 1
fi

# Python Grid Search 스크립트 실행
if "${PYTHON_BIN}" grid_search_diff.py; then
    echo ""
    echo "=========================================="
    echo "Grid Search 완료!"
    echo "=========================================="
    echo "완료 시간: $(date)"
    echo "결과는 runs/detect/grid_search_diff/grid_search_results.csv에 저장되었습니다."
    exit 0
else
    EXIT_CODE=$?
    echo ""
    echo "=========================================="
    echo "Grid Search 실패! (Exit code: ${EXIT_CODE})"
    echo "=========================================="
    exit ${EXIT_CODE}
fi
