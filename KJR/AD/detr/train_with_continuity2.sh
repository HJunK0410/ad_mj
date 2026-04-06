#!/bin/bash

# DETR 학습 스크립트 (Continuity Loss 포함, YOLO 형식 지원)
# 사용법: bash train_with_continuity.sh
# 
# 주의: 아래 COCO_PATH를 실제 데이터셋 경로로 수정하세요!

# 기본 설정
export CUDA_VISIBLE_DEVICES=0  # GPU 1번 사용
COCO_PATH="/home/user/DATA/sjpm/sjpm1"  # ⚠️ 여기를 실제 데이터셋 경로로 수정하세요!
#OUTPUT_DIR="./outputs_new_loss/100/detr_100"
OUTPUT_DIR="./outputs_new_loss/100/detr_100"
BATCH_SIZE=12
EPOCHS=400
NUM_WORKERS=4

# 데이터 샘플링 설정
# main.py가 datasets/yaml/에서 자동으로 YAML 파일을 찾아서 사용합니다
# 파일이 없으면 랜덤 샘플링 후 YAML 파일을 생성합니다
MAX_SAMPLES=100  # 사용할 샘플 수 (비워두면 전체 데이터 사용)
DATA_SEED=42

# 경로 존재 여부 체크
if [ ! -d "${COCO_PATH}" ]; then
    echo "❌ 오류: 데이터셋 경로가 존재하지 않습니다: ${COCO_PATH}"
    echo "📝 train_with_continuity.sh 파일의 10번째 줄에서 COCO_PATH를 수정하세요!"
    echo "   예시: COCO_PATH=\"/home/user/DATA/sjpm/sjpm1\""
    exit 1
fi

# data.yaml 파일 체크 (YOLO 형식, 선택사항)
DATA_YAML="${COCO_PATH}/data.yaml"
if [ ! -f "${DATA_YAML}" ]; then
    # 상위 디렉토리에서도 확인
    DATA_YAML="$(dirname ${COCO_PATH})/data.yaml"
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

# Continuity Loss 설정 (그리드 서치)
# 0.01 간격으로 그리드 서치 수행
LAMBDA_CONT_2D_VALUES=(0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)  # 2D continuity loss 가중치 범위
LAMBDA_CONT_1D_VALUES=(0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)  # 1D continuity loss 가중치 범위
CURRICULUM_EPOCHS=5  # Curriculum learning epochs

# 학습 파라미터
LR=1e-4
LR_BACKBONE=1e-5
WEIGHT_DECAY=1e-4
LR_DROP=200

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

# 그리드 서치 시작
TOTAL_COMBINATIONS=$((${#LAMBDA_CONT_2D_VALUES[@]} * ${#LAMBDA_CONT_1D_VALUES[@]}))
CURRENT_COMBINATION=0

echo "============================================================"
echo "DETR 그리드 서치 시작 (Continuity Loss)"
echo "============================================================"
echo "테스크: 제품 스크레치 탐지 (Object Detection)"
echo "데이터셋 형식: YOLO"
echo "데이터 경로: ${COCO_PATH}"
echo "총 조합 수: ${TOTAL_COMBINATIONS}"
echo "Lambda 2D 범위: ${LAMBDA_CONT_2D_VALUES[@]}"
echo "Lambda 1D 범위: ${LAMBDA_CONT_1D_VALUES[@]}"
echo "============================================================"
echo ""

# 그리드 서치 루프
for LAMBDA_CONT_2D in "${LAMBDA_CONT_2D_VALUES[@]}"; do
    for LAMBDA_CONT_1D in "${LAMBDA_CONT_1D_VALUES[@]}"; do
        CURRENT_COMBINATION=$((CURRENT_COMBINATION + 1))
        
        # 출력 디렉토리 설정 (각 조합마다 다른 디렉토리)
        LAMBDA_2D_STR=$(printf "%.2f" ${LAMBDA_CONT_2D} | sed 's/\.//g' | sed 's/^0*//')
        LAMBDA_1D_STR=$(printf "%.2f" ${LAMBDA_CONT_1D} | sed 's/\.//g' | sed 's/^0*//')
        if [ -z "${LAMBDA_2D_STR}" ]; then
            LAMBDA_2D_STR="000"
        fi
        if [ -z "${LAMBDA_1D_STR}" ]; then
            LAMBDA_1D_STR="000"
        fi
        CURRENT_OUTPUT_DIR="${OUTPUT_DIR}_2d${LAMBDA_2D_STR}_1d${LAMBDA_1D_STR}"
        
        # 출력 디렉토리 생성
        mkdir -p ${CURRENT_OUTPUT_DIR}
        
        # 학습 정보 출력
        echo "============================================================"
        echo "[${CURRENT_COMBINATION}/${TOTAL_COMBINATIONS}] 그리드 서치 실행"
        echo "============================================================"
        echo "Lambda 2D: ${LAMBDA_CONT_2D}"
        echo "Lambda 1D: ${LAMBDA_CONT_1D}"
        echo "출력 디렉토리: ${CURRENT_OUTPUT_DIR}"
        echo "최대 탐지 가능 객체 수: ${NUM_QUERIES}"
        echo "Curriculum Epochs: ${CURRICULUM_EPOCHS}"
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
        
        python main.py \
            --dataset_file yolo \
            --coco_path "${COCO_PATH}" \
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
            --seed ${DATA_SEED} \
            --no_checkpoint \
            ${DATA_ARGS}
        
        # 실행 결과 확인
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ [${CURRENT_COMBINATION}/${TOTAL_COMBINATIONS}] 학습 완료!"
            echo "   결과: ${CURRENT_OUTPUT_DIR}"
            echo "   Lambda 2D: ${LAMBDA_CONT_2D}, Lambda 1D: ${LAMBDA_CONT_1D}"
            echo ""
        else
            echo ""
            echo "❌ [${CURRENT_COMBINATION}/${TOTAL_COMBINATIONS}] 학습 중 오류 발생!"
            echo "   Lambda 2D: ${LAMBDA_CONT_2D}, Lambda 1D: ${LAMBDA_CONT_1D}"
            echo "   계속 진행합니다..."
            echo ""
        fi
    done
done

echo "============================================================"
echo "그리드 서치 완료!"
echo "총 ${TOTAL_COMBINATIONS}개 조합 실행 완료"
echo "============================================================"
