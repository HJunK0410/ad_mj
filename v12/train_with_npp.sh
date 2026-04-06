#!/bin/bash

# YOLOv12 학습 스크립트 (NPP Loss 사용)
# 이 스크립트는 NPP Loss 하이퍼파라미터를 튜닝할 수 있도록 구성되어 있습니다.

# 기본 하이퍼파라미터 설정
NPP_LAMBDA_2D=${NPP_LAMBDA_2D:-0.02}      # 2D NPP Loss 가중치 (기본값: 0.02)
NPP_LAMBDA_1D=${NPP_LAMBDA_1D:-0.06}      # 1D NPP Loss 가중치 (기본값: 0.06)
BBOX_MASK_WEIGHT=${BBOX_MASK_WEIGHT:-0.3} # Bbox 내부 마스크 가중치 (기본값: 0.3)
FPN_SOURCES=${FPN_SOURCES:-"14,17,20"}     # 사용할 FPN 소스 레이어 (기본값: 모두 사용)

# FPN 소스 옵션:
# "14"       -> P3만 사용
# "14,17"    -> P3, P4 사용
# "14,17,20" -> P3, P4, P5 모두 사용 (기본값)

# 프로젝트 이름 (하이퍼파라미터를 포함하여 구분)
FPN_STR=$(echo ${FPN_SOURCES} | tr ',' '_')
PROJECT_NAME="train_npp_l2d${NPP_LAMBDA_2D}_l1d${NPP_LAMBDA_1D}_mask${BBOX_MASK_WEIGHT}_fpn${FPN_STR}"
PROJECT_DIR="runs/${PROJECT_NAME}"

echo "=========================================="
echo "YOLOv12 Training with NPP Loss"
echo "=========================================="
echo "NPP Lambda 2D:     ${NPP_LAMBDA_2D}"
echo "NPP Lambda 1D:     ${NPP_LAMBDA_1D}"
echo "Bbox Mask Weight:  ${BBOX_MASK_WEIGHT}"
echo "FPN Sources:       ${FPN_SOURCES}"
echo "Project Directory: ${PROJECT_DIR}"
echo "=========================================="

# 환경 설정
export PYTHONPATH="/home/user/hyunjun/AD/v12:$PYTHONPATH"

# YOLO 학습 실행
if python train.py \
  --npp_lambda_2d=${NPP_LAMBDA_2D} \
  --npp_lambda_1d=${NPP_LAMBDA_1D} \
  --npp_bbox_mask_weight=${BBOX_MASK_WEIGHT} \
  --npp_fpn_sources=${FPN_SOURCES}; then
    echo ""
    echo "학습 완료. 결과는 runs/detect/${PROJECT_NAME} 디렉토리에 저장되었습니다."
    exit 0
else
    EXIT_CODE=$?
    echo ""
    echo "=========================================="
    echo "학습 실패! (Exit code: ${EXIT_CODE})"
    echo "=========================================="
    if [ ${EXIT_CODE} -eq 139 ]; then
        echo "Segmentation fault가 발생했습니다."
        echo "가능한 원인:"
        echo "  - GPU 메모리 부족"
        echo "  - 모델 forward pass 중 오류"
        echo "  - 환경 호환성 문제"
    fi
    echo "로그를 확인하여 문제를 해결하세요."
    exit ${EXIT_CODE}
fi
