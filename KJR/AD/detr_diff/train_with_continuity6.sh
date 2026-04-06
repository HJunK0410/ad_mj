#!/bin/bash

# DETR 학습 스크립트 (Continuity Loss 포함, YOLO 형식 지원)
# 사용법: bash train_with_continuity.sh
# 
# 주의: 아래 COCO_PATH를 실제 데이터셋 경로로 수정하세요!

# 기본 설정
export CUDA_VISIBLE_DEVICES=2  # GPU 1번 사용
TRAIN_IMAGE_DIR="/home/user/hyunjun/AD/data/save/train_75/images"
TRAIN_LABEL_DIR="/home/user/hyunjun/AD/data/save/train_75/labels"
VAL_IMAGE_DIR="/home/user/hyunjun/AD/data/images/val"
VAL_LABEL_DIR="/home/user/hyunjun/AD/data/labels/val"
OUTPUT_DIR="./outputs/detr_diff_75/"
BATCH_SIZE=4
EPOCHS=300
NUM_WORKERS=4

# 데이터 샘플링 설정
# main.py가 datasets/yaml/에서 자동으로 YAML 파일을 찾아서 사용합니다
# 파일이 없으면 랜덤 샘플링 후 YAML 파일을 생성합니다
MAX_SAMPLES=""  # 비우면 전체 데이터 사용
DATA_SEED=42

# 경로 존재 여부 체크
for path in "${TRAIN_IMAGE_DIR}" "${TRAIN_LABEL_DIR}" "${VAL_IMAGE_DIR}" "${VAL_LABEL_DIR}"; do
    if [ ! -d "${path}" ]; then
        echo "❌ 오류: 데이터셋 경로가 존재하지 않습니다: ${path}"
        exit 1
    fi
done

# main.py(yolo)가 읽을 수 있도록 런타임 YOLO 구조 생성
RUNTIME_YOLO_ROOT="/tmp/detr_diff_yolo_split_${USER}_$$"
mkdir -p "${RUNTIME_YOLO_ROOT}/images" "${RUNTIME_YOLO_ROOT}/labels"
ln -sfn "${TRAIN_IMAGE_DIR}" "${RUNTIME_YOLO_ROOT}/images/train"
ln -sfn "${VAL_IMAGE_DIR}" "${RUNTIME_YOLO_ROOT}/images/val"
ln -sfn "${TRAIN_LABEL_DIR}" "${RUNTIME_YOLO_ROOT}/labels/train"
ln -sfn "${VAL_LABEL_DIR}" "${RUNTIME_YOLO_ROOT}/labels/val"

# data.yaml 파일 체크 (YOLO 형식, 선택사항)
DATA_YAML="${RUNTIME_YOLO_ROOT}/data.yaml"
if [ ! -f "${DATA_YAML}" ]; then
    # 상위 디렉토리에서도 확인
    DATA_YAML="$(dirname ${RUNTIME_YOLO_ROOT})/data.yaml"
    if [ ! -f "${DATA_YAML}" ]; then
        echo "ℹ️  Info: data.yaml 파일을 찾을 수 없습니다 (선택사항)."
        echo "   yaml 없이도 작동하지만, 클래스 정보는 기본값(20)을 사용합니다."
        echo "   정확한 클래스 수를 사용하려면 data.yaml 파일을 추가하세요."
    else
        echo "✓ data.yaml 파일 발견: ${DATA_YAML}"
    fi
else
    echo "✓ data.yaml 파일 발견: ${DATA_YAML}"
fi

# Continuity Loss 설정
# detr_diff_100 기준 상위 10개 조합만 실행
TARGET_COMBINATIONS=(
    "0.02 0.06 0.20 0.50 1.00"
    "0.02 0.06 0.20 0.10 0.50"
    "0.02 0.06 0.20 0.50 0.50"
    "0.00 0.00 0.00 0.00 0.00"
)

# Curriculum Learning 및 Norm Mask 설정
CURRICULUM_EPOCHS=5  # Curriculum learning epoch 수 (lambda 점진적 증가)

# 학습 파라미터
LR=1e-4
LR_BACKBONE=1e-5
WEIGHT_DECAY=1e-4
LR_DROP=200
EARLY_STOP_PATIENCE=20

# 모델 파라미터
# 스크레치 탐지에 최적화된 설정
BACKBONE="resnet50"
NUM_QUERIES=200  # 스크레치가 많을 수 있으므로 증가 (기본값: 100)
HIDDEN_DIM=256
ENC_LAYERS=6
DEC_LAYERS=6

# GPU 설정
DEVICE="cuda"
NUM_GPUS=1

# 조합 실행 시작
TOTAL_COMBINATIONS=${#TARGET_COMBINATIONS[@]}
CURRENT_COMBINATION=0

echo "============================================================"
echo "DETR 선택 조합 실행 시작 (Continuity Loss + Norm Mask Bbox Coef + Alpha/Beta)"
echo "============================================================"
echo "테스크: 제품 스크래치 탐지 (Object Detection)"
echo "데이터셋 형식: YOLO"
echo "Train 이미지: ${TRAIN_IMAGE_DIR}"
echo "Train 라벨:   ${TRAIN_LABEL_DIR}"
echo "Val 이미지:   ${VAL_IMAGE_DIR}"
echo "Val 라벨:     ${VAL_LABEL_DIR}"
echo "런타임 경로:  ${RUNTIME_YOLO_ROOT}"
echo "총 조합 수: ${TOTAL_COMBINATIONS}"
echo "실행 조합:"
printf '  - %s\n' "${TARGET_COMBINATIONS[@]}"
echo "============================================================"
echo ""

# 선택 조합 실행 루프
for COMBINATION in "${TARGET_COMBINATIONS[@]}"; do
    read -r LAMBDA_CONT_2D LAMBDA_CONT_1D NORM_MASK_BBOX_COEF ALPHA BETA <<< "${COMBINATION}"
    CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))

    # 출력 디렉토리 설정 (각 조합마다 다른 디렉토리)
    LAMBDA_2D_STR=$(printf "%.2f" "${LAMBDA_CONT_2D}" | sed 's/\.//g' | sed 's/^0*//')
    LAMBDA_1D_STR=$(printf "%.2f" "${LAMBDA_CONT_1D}" | sed 's/\.//g' | sed 's/^0*//')
    NORM_MASK_BBOX_COEF_STR=$(printf "%.2f" "${NORM_MASK_BBOX_COEF}" | sed 's/\.//g' | sed 's/^0*//')
    ALPHA_STR=$(printf "%.2f" "${ALPHA}" | sed 's/\.//g' | sed 's/^0*//')
    BETA_STR=$(printf "%.2f" "${BETA}" | sed 's/\.//g' | sed 's/^0*//')
    [ -z "${LAMBDA_2D_STR}" ] && LAMBDA_2D_STR="000"
    [ -z "${LAMBDA_1D_STR}" ] && LAMBDA_1D_STR="000"
    [ -z "${NORM_MASK_BBOX_COEF_STR}" ] && NORM_MASK_BBOX_COEF_STR="00"
    [ -z "${ALPHA_STR}" ] && ALPHA_STR="03"
    [ -z "${BETA_STR}" ] && BETA_STR="10"
    CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_2d${LAMBDA_2D_STR}_1d${LAMBDA_1D_STR}_nm${NORM_MASK_BBOX_COEF_STR}_a${ALPHA_STR}_b${BETA_STR}"

    # 출력 디렉토리 생성
    mkdir -p "${CURRENT_OUTPUT_DIR}"

    # 학습 정보 출력
    echo "============================================================"
    echo "[${CURRENT_COMBINATION}/${TOTAL_COMBINATIONS}] 선택 조합 실행"
    echo "============================================================"
    echo "Lambda 2D: ${LAMBDA_CONT_2D}"
    echo "Lambda 1D: ${LAMBDA_CONT_1D}"
    echo "Curriculum Epochs: ${CURRICULUM_EPOCHS}"
    echo "Norm Mask Bbox Coef: ${NORM_MASK_BBOX_COEF}"
    echo "Alpha: ${ALPHA}"
    echo "Beta: ${BETA}"
    echo "출력 디렉토리: ${CURRENT_OUTPUT_DIR}"
    echo "최대 탐지 가능 객체 수: ${NUM_QUERIES}"
    if [ -n "${MAX_SAMPLES}" ]; then
        echo "최대 샘플 수: ${MAX_SAMPLES}개"
    fi
    echo "============================================================"
    echo ""

    # 학습 실행
    DATA_ARGS=""
    if [ -n "${MAX_SAMPLES}" ]; then
        DATA_ARGS="--max_samples ${MAX_SAMPLES}"
    fi

    python3 main.py \
        --dataset_file yolo \
        --coco_path "${RUNTIME_YOLO_ROOT}" \
        --output_dir "${CURRENT_OUTPUT_DIR}" \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --num_workers ${NUM_WORKERS} \
        --lr ${LR} \
        --lr_backbone ${LR_BACKBONE} \
        --weight_decay ${WEIGHT_DECAY} \
        --lr_drop ${LR_DROP} \
        --backbone ${BACKBONE} \
        --num_queries ${NUM_QUERIES} \
        --hidden_dim ${HIDDEN_DIM} \
        --enc_layers ${ENC_LAYERS} \
        --dec_layers ${DEC_LAYERS} \
        --device ${DEVICE} \
        --bbox_loss_coef 5 \
        --giou_loss_coef 2 \
        --eos_coef 0.1 \
        --lambda_cont_2d ${LAMBDA_CONT_2D} \
        --lambda_cont_1d ${LAMBDA_CONT_1D} \
        --curriculum_epochs ${CURRICULUM_EPOCHS} \
        --norm_mask_bbox_coef ${NORM_MASK_BBOX_COEF} \
        --alpha ${ALPHA} \
        --beta ${BETA} \
        --seed ${DATA_SEED} \
        --no_checkpoint \
        --early_stop_patience ${EARLY_STOP_PATIENCE} \
        ${DATA_ARGS}

    # 실행 결과 확인
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ [${CURRENT_COMBINATION}/${TOTAL_COMBINATIONS}] 학습 완료!"
        echo "   결과: ${CURRENT_OUTPUT_DIR}"
        echo "   Lambda 2D: ${LAMBDA_CONT_2D}, Lambda 1D: ${LAMBDA_CONT_1D}"
        echo "   Curriculum Epochs: ${CURRICULUM_EPOCHS}, Norm Mask Coef: ${NORM_MASK_BBOX_COEF}"
        echo "   Alpha: ${ALPHA}, Beta: ${BETA}"
        echo ""
    else
        echo ""
        echo "❌ [${CURRENT_COMBINATION}/${TOTAL_COMBINATIONS}] 학습 중 오류 발생!"
        echo "   Lambda 2D: ${LAMBDA_CONT_2D}, Lambda 1D: ${LAMBDA_CONT_1D}"
        echo "   Curriculum Epochs: ${CURRICULUM_EPOCHS}, Norm Mask Coef: ${NORM_MASK_BBOX_COEF}"
        echo "   Alpha: ${ALPHA}, Beta: ${BETA}"
        echo "   계속 진행합니다..."
        echo ""
    fi
done

echo "============================================================"
echo "선택 조합 실행 완료!"
echo "총 ${TOTAL_COMBINATIONS}개 조합 실행 완료"
echo "============================================================"
