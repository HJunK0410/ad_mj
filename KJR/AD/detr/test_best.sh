#!/bin/bash

# 모델 테스트 스크립트
# 사용법: 
#   bash test_best.sh [디렉토리 경로]          # 디렉토리 내 모든 모델 테스트 (기본값: new_sjpm_top10)
#   bash test_best.sh [체크포인트.pt 경로]    # 단일 체크포인트 테스트
#   bash test_best.sh [best.csv 경로]         # CSV 파일 기반 테스트
#
# 하이퍼파라미터 지정 (환경변수):
#   LAMBDA_CONT_2D=0.02 LAMBDA_CONT_1D=0.10 NORM_MASK_BBOX_COEF=0.30 \
#   ALPHA=1.5 BETA=1.0 bash test_best.sh [경로]
#
# 예시:
#   bash test_best.sh /home/user/KJR/AD/detr/outputs/new_sjpm_top10  # 디렉토리 내 모든 모델 테스트
#   LAMBDA_CONT_2D=0.02 LAMBDA_CONT_1D=0.10 bash test_best.sh checkpoint.pt  # 단일 체크포인트 테스트

# 기본 설정
# BASE_DIR을 직접 지정하려면 아래 주석을 해제하고 경로를 설정하세요
# BASE_DIR="/home/user/KJR/AD/detr_diff"

export CUDA_VISIBLE_DEVICES=2
# 현재 스크립트가 있는 디렉토리를 기준으로 BASE_DIR 자동 감지
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-${SCRIPT_DIR}}"

# 프로젝트 이름 자동 감지 (detr_diff 또는 detr)
PROJECT_NAME=$(basename "${BASE_DIR}")

# 첫 번째 인자 확인 (기본값: new_sjpm_top10 디렉토리)
INPUT_ARG="${1:-/home/user/KJR/AD/detr/outputs/new_sjpm_10}"

COCO_PATH="${COCO_PATH:-/home/user/hyunjun/AD/data}"
TEST_IMAGE_DIR="${TEST_IMAGE_DIR:-/home/user/hyunjun/AD/data/images/test}"
TEST_LABEL_DIR="${TEST_LABEL_DIR:-/home/user/hyunjun/AD/data/labels/test}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
# 결과 저장 디렉토리 (직접 지정하려면 아래 주석을 해제하고 경로를 설정하세요)
# OUTPUT_BASE_DIR="/home/user/KJR/AD/detr/outputs/detr_05/results"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/home/user/KJR/AD/detr/outputs/new_sjpm_10/results}"

# 하이퍼파라미터 (환경변수로 지정 가능, 없으면 폴더명에서 추출)
USER_LAMBDA_2D="${LAMBDA_CONT_2D:-}"
USER_LAMBDA_1D="${LAMBDA_CONT_1D:-}"
USER_NORM_MASK="${NORM_MASK_BBOX_COEF:-}"
USER_ALPHA="${ALPHA:-}"
USER_BETA="${BETA:-}"

# 입력 인자 타입 확인
if [[ "${INPUT_ARG}" == *.pt ]]; then
    # 단일 체크포인트 모드
    MODE="checkpoint"
    CHECKPOINT_PATH="${INPUT_ARG}"
    
    # 체크포인트 파일 확인
    if [ ! -f "${CHECKPOINT_PATH}" ]; then
        echo "❌ 오류: 체크포인트 파일을 찾을 수 없습니다: ${CHECKPOINT_PATH}"
        exit 1
    fi
    
    # 체크포인트 경로에서 폴더명 추출
    # 예: /path/to/outputs/detr/folder_name/best.pt -> folder_name
    CHECKPOINT_DIR=$(dirname "${CHECKPOINT_PATH}")
    FOLDER_NAME=$(basename "${CHECKPOINT_DIR}")
    
    echo "============================================================"
    echo "단일 체크포인트 테스트 모드"
    echo "============================================================"
    echo "체크포인트: ${CHECKPOINT_PATH}"
    echo "폴더명: ${FOLDER_NAME}"
elif [ -d "${INPUT_ARG}" ]; then
    # 디렉토리 모드: 디렉토리 내의 모든 하위 디렉토리를 테스트
    MODE="directory"
    TARGET_DIR="${INPUT_ARG}"
    
    # 디렉토리 확인
    if [ ! -d "${TARGET_DIR}" ]; then
        echo "❌ 오류: 디렉토리가 존재하지 않습니다: ${TARGET_DIR}"
        exit 1
    fi
    
    echo "============================================================"
    echo "디렉토리 기반 자동 테스트 모드"
    echo "============================================================"
    echo "대상 디렉토리: ${TARGET_DIR}"
elif [[ "${INPUT_ARG}" == *.csv ]]; then
    # CSV 모드
    MODE="csv"
    BEST_CSV="${INPUT_ARG}"
    
    # CSV 파일 확인
    if [ ! -f "${BEST_CSV}" ]; then
        echo "❌ 오류: CSV 파일을 찾을 수 없습니다: ${BEST_CSV}"
        exit 1
    fi
else
    echo "❌ 오류: 입력 인자를 인식할 수 없습니다: ${INPUT_ARG}"
    echo "사용법:"
    echo "  bash test_best.sh [디렉토리 경로]  # 디렉토리 내 모든 모델 테스트"
    echo "  bash test_best.sh [체크포인트.pt 경로]  # 단일 체크포인트 테스트"
    echo "  bash test_best.sh [best.csv 경로]  # CSV 파일 기반 테스트"
    exit 1
fi

# 테스트 이미지/라벨 경로가 별도 지정된 경우 런타임 YOLO 구조 생성
if [ -n "${TEST_IMAGE_DIR}" ] || [ -n "${TEST_LABEL_DIR}" ]; then
    if [ -z "${TEST_IMAGE_DIR}" ] || [ -z "${TEST_LABEL_DIR}" ]; then
        echo "❌ 오류: TEST_IMAGE_DIR와 TEST_LABEL_DIR는 함께 지정해야 합니다."
        exit 1
    fi
    if [ ! -d "${TEST_IMAGE_DIR}" ]; then
        echo "❌ 오류: 테스트 이미지 경로가 존재하지 않습니다: ${TEST_IMAGE_DIR}"
        exit 1
    fi
    if [ ! -d "${TEST_LABEL_DIR}" ]; then
        echo "❌ 오류: 테스트 라벨 경로가 존재하지 않습니다: ${TEST_LABEL_DIR}"
        exit 1
    fi

    TEST_RUNTIME_ROOT="/tmp/detr_test_yolo_${USER}_$$"
    mkdir -p "${TEST_RUNTIME_ROOT}/images" "${TEST_RUNTIME_ROOT}/labels"
    ln -sfn "${TEST_IMAGE_DIR}" "${TEST_RUNTIME_ROOT}/images/val"
    ln -sfn "${TEST_LABEL_DIR}" "${TEST_RUNTIME_ROOT}/labels/val"
    # eval 시 main.py가 image_set='test'를 사용하므로 test 링크도 필요
    ln -sfn "${TEST_IMAGE_DIR}" "${TEST_RUNTIME_ROOT}/images/test"
    ln -sfn "${TEST_LABEL_DIR}" "${TEST_RUNTIME_ROOT}/labels/test"
    # 일부 코드 경로에서 train 참조가 있어 동일 링크를 함께 제공
    ln -sfn "${TEST_IMAGE_DIR}" "${TEST_RUNTIME_ROOT}/images/train"
    ln -sfn "${TEST_LABEL_DIR}" "${TEST_RUNTIME_ROOT}/labels/train"
    COCO_PATH="${TEST_RUNTIME_ROOT}"
fi

# 데이터셋 경로 확인
if [ ! -d "${COCO_PATH}" ]; then
    echo "❌ 오류: 데이터셋 경로가 존재하지 않습니다: ${COCO_PATH}"
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p "${OUTPUT_BASE_DIR}"

# 단일 체크포인트 모드 처리 함수
run_single_test() {
    local checkpoint_path=$1
    local folder_name=$2
    local epoch=$3  # epoch 정보 (선택적)
    
    # 하이퍼파라미터 추출 (폴더명에서 추출)
    extract_hyperparams "$folder_name"
    
    # 사용자가 환경변수로 지정한 하이퍼파라미터가 있으면 우선 사용, 없으면 폴더명에서 추출한 값 사용
    if [ -n "$USER_LAMBDA_2D" ]; then
        lambda_2d="$USER_LAMBDA_2D"
    fi
    if [ -n "$USER_LAMBDA_1D" ]; then
        lambda_1d="$USER_LAMBDA_1D"
    fi
    if [ -n "$USER_NORM_MASK" ]; then
        norm_mask="$USER_NORM_MASK"
    fi
    if [ -n "$USER_ALPHA" ]; then
        alpha="$USER_ALPHA"
    fi
    if [ -n "$USER_BETA" ]; then
        beta="$USER_BETA"
    fi
    
    # 출력 디렉토리 (epoch가 있으면 포함)
    if [ -n "$epoch" ]; then
        test_output_dir="${OUTPUT_BASE_DIR}/${folder_name}_epoch${epoch}"
    else
        test_output_dir="${OUTPUT_BASE_DIR}/${folder_name}"
    fi
    mkdir -p "${test_output_dir}"
    
    echo "============================================================"
    echo "테스트 실행"
    echo "============================================================"
    echo "폴더: ${folder_name}"
    echo "체크포인트: ${checkpoint_path}"
    echo "출력 디렉토리: ${test_output_dir}"
    
    # 하이퍼파라미터 출력
    hyperparams_args=""
    if [ -n "$lambda_2d" ]; then
        echo "  lambda_cont_2d: ${lambda_2d}"
        hyperparams_args="${hyperparams_args} --lambda_cont_2d ${lambda_2d}"
    fi
    if [ -n "$lambda_1d" ]; then
        echo "  lambda_cont_1d: ${lambda_1d}"
        hyperparams_args="${hyperparams_args} --lambda_cont_1d ${lambda_1d}"
    fi
    if [ -n "$norm_mask" ]; then
        echo "  norm_mask_bbox_coef: ${norm_mask}"
        hyperparams_args="${hyperparams_args} --norm_mask_bbox_coef ${norm_mask}"
    fi
    if [ -n "$alpha" ]; then
        echo "  alpha: ${alpha}"
        hyperparams_args="${hyperparams_args} --alpha ${alpha}"
    fi
    if [ -n "$beta" ]; then
        echo "  beta: ${beta}"
        hyperparams_args="${hyperparams_args} --beta ${beta}"
    fi
    echo "============================================================"
    
    # 테스트 실행
    cd "${BASE_DIR}" || exit 1
    
    python3 test.py \
        --resume "${checkpoint_path}" \
        --coco_path "${COCO_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --output_dir "${test_output_dir}" \
        ${hyperparams_args}
    
    if [ $? -eq 0 ]; then
        echo "✅ 테스트 완료: ${folder_name}"
    else
        echo "❌ 테스트 실패: ${folder_name}"
        return 1
    fi
}

# CSV 모드인 경우에만 필요한 변수들
if [ "$MODE" = "csv" ]; then
    echo "============================================================"
    echo "best.csv 기반 자동 테스트 시작"
    echo "============================================================"
    echo "CSV 파일: ${BEST_CSV}"
    echo "테스트 데이터셋: ${COCO_PATH}"
    echo "배치 크기: ${BATCH_SIZE}"
    echo "출력 디렉토리: ${OUTPUT_BASE_DIR}"
    echo "============================================================"
    echo ""
    
    # CSV 파일 읽기 (헤더 제외)
    total_lines=$(tail -n +2 "${BEST_CSV}" | wc -l)
    current_line=0
fi

# 폴더명에서 하이퍼파라미터 추출 함수
# 지원 형식:
# 1. test_2d2_1d10_nm30_a150_b100 (detr_diff 형식)
# 2. last_2d4_1d6_nm30 (detr 형식, alpha/beta 없음)
extract_hyperparams() {
    local folder_name=$1
    
    # 2d 값 추출 (2자리 또는 3자리 숫자 지원)
    # 예: 2d2 -> 0.02, 2d10 -> 0.10, 2d000 -> 0.00, 2d004 -> 0.04
    if [[ $folder_name =~ _2d([0-9]+) ]]; then
        lambda_2d_str="${BASH_REMATCH[1]}"
        # 앞의 0 제거
        lambda_2d_str=$(echo "$lambda_2d_str" | sed 's/^0*//')
        if [ -z "$lambda_2d_str" ]; then
            lambda_2d_str="0"
        fi
        # 1-2자리 숫자를 0.XX로 변환
        if [ ${#lambda_2d_str} -eq 1 ]; then
            lambda_2d="0.0${lambda_2d_str}"
        elif [ ${#lambda_2d_str} -eq 2 ]; then
            lambda_2d="0.${lambda_2d_str}"
        else
            # 3자리 이상이면 앞 2자리만 사용 (예: 200 -> 0.20, 004 -> 0.00)
            lambda_2d="0.${lambda_2d_str:0:2}"
        fi
    else
        lambda_2d=""
    fi
    
    # 1d 값 추출 (1-2자리 또는 3자리 숫자 지원)
    # 예: 1d6 -> 0.06, 1d10 -> 0.10, 1d000 -> 0.00, 1d006 -> 0.06
    if [[ $folder_name =~ _1d([0-9]+) ]]; then
        lambda_1d_str="${BASH_REMATCH[1]}"
        lambda_1d_str=$(echo "$lambda_1d_str" | sed 's/^0*//')
        if [ -z "$lambda_1d_str" ]; then
            lambda_1d_str="0"
        fi
        if [ ${#lambda_1d_str} -eq 1 ]; then
            lambda_1d="0.0${lambda_1d_str}"
        elif [ ${#lambda_1d_str} -eq 2 ]; then
            lambda_1d="0.${lambda_1d_str}"
        else
            # 3자리 이상이면 앞 2자리만 사용
            lambda_1d="0.${lambda_1d_str:0:2}"
        fi
    else
        lambda_1d=""
    fi
    
    # nm 값 추출
    if [[ $folder_name =~ _nm([0-9]+) ]]; then
        nm_str="${BASH_REMATCH[1]}"
        nm_str=$(echo "$nm_str" | sed 's/^0*//')
        if [ -z "$nm_str" ]; then
            nm_str="0"
        fi
        if [ ${#nm_str} -eq 1 ]; then
            norm_mask="0.0${nm_str}"
        elif [ ${#nm_str} -eq 2 ]; then
            norm_mask="0.${nm_str}"
        else
            norm_mask="0.${nm_str:0:2}"
        fi
    else
        norm_mask=""
    fi
    
    # a 값 추출 (detr_diff 형식에만 있음)
    if [[ $folder_name =~ _a([0-9]+) ]]; then
        alpha_str="${BASH_REMATCH[1]}"
        alpha_str=$(echo "$alpha_str" | sed 's/^0*//')
        if [ -z "$alpha_str" ]; then
            alpha_str="0"
        fi
        if [ ${#alpha_str} -eq 1 ]; then
            alpha="0.${alpha_str}"
        elif [ ${#alpha_str} -eq 2 ]; then
            alpha="0.${alpha_str}"
        else
            # 3자리 이상 (예: 150 -> 1.5)
            if [ ${#alpha_str} -eq 3 ]; then
                alpha="${alpha_str:0:1}.${alpha_str:1:2}"
            else
                alpha="${alpha_str:0:2}.${alpha_str:2:2}"
            fi
        fi
    else
        alpha=""
    fi
    
    # b 값 추출 (detr_diff 형식에만 있음)
    if [[ $folder_name =~ _b([0-9]+) ]]; then
        beta_str="${BASH_REMATCH[1]}"
        beta_str=$(echo "$beta_str" | sed 's/^0*//')
        if [ -z "$beta_str" ]; then
            beta_str="0"
        fi
        if [ ${#beta_str} -eq 1 ]; then
            beta="0.${beta_str}"
        elif [ ${#beta_str} -eq 2 ]; then
            beta="0.${beta_str}"
        else
            # 3자리 이상 (예: 150 -> 1.5)
            if [ ${#beta_str} -eq 3 ]; then
                beta="${beta_str:0:1}.${beta_str:1:2}"
            else
                beta="${beta_str:0:2}.${beta_str:2:2}"
            fi
        fi
    else
        beta=""
    fi
}

# 단일 체크포인트 모드 처리
if [ "$MODE" = "checkpoint" ]; then
    run_single_test "${CHECKPOINT_PATH}" "${FOLDER_NAME}" ""
    
    echo "============================================================"
    echo "테스트 완료!"
    echo "결과 저장 위치: ${OUTPUT_BASE_DIR}/${FOLDER_NAME}"
    echo "============================================================"
    exit $?
fi

# 디렉토리 모드 처리
if [ "$MODE" = "directory" ]; then
    echo "디렉토리 내 모든 모델 검색 중..."
    echo ""
    
    # 디렉토리 내의 모든 하위 디렉토리 찾기
    found_models=0
    tested_models=0
    failed_models=0
    
    # 하위 디렉토리들을 순회
    for subdir in "${TARGET_DIR}"/*; do
        if [ ! -d "${subdir}" ]; then
            continue
        fi
        
        folder_name=$(basename "${subdir}")
        checkpoint_path="${subdir}/best.pt"
        
        # best.pt 파일이 있는지 확인
        if [ ! -f "${checkpoint_path}" ]; then
            echo "⚠️  체크포인트 없음: ${folder_name}"
            continue
        fi
        
        found_models=$((found_models + 1))
        echo "[${found_models}] 테스트 시작: ${folder_name}"
        
        # 테스트 실행
        if run_single_test "${checkpoint_path}" "${folder_name}" ""; then
            tested_models=$((tested_models + 1))
        else
            failed_models=$((failed_models + 1))
        fi
        echo ""
    done
    
    echo "============================================================"
    echo "디렉토리 테스트 완료!"
    echo "  발견된 모델: ${found_models}개"
    echo "  성공: ${tested_models}개"
    echo "  실패: ${failed_models}개"
    echo "결과 저장 위치: ${OUTPUT_BASE_DIR}"
    echo "============================================================"
    exit 0
fi

# CSV 모드 처리
# CSV 파일 형식 감지 (헤더 확인)
CSV_HEADER=$(head -1 "${BEST_CSV}")
if [[ "$CSV_HEADER" == *"experiment"* ]]; then
    # detr 형식: experiment,best_epoch,best_mAP50,best_mAP50-95,...
    CSV_FORMAT="detr"
elif [[ "$CSV_HEADER" == *"folder"* ]]; then
    # detr_diff 형식: folder,epoch,time,...
    CSV_FORMAT="detr_diff"
else
    echo "⚠️  경고: CSV 형식을 자동 감지할 수 없습니다. 기본 형식(detr_diff)을 사용합니다."
    CSV_FORMAT="detr_diff"
fi

# CSV 파일 처리 (헤더 제외)
tail -n +2 "${BEST_CSV}" | while IFS=',' read -r line; do
    current_line=$((current_line + 1))
    
    if [ "$CSV_FORMAT" = "detr" ]; then
        # detr 형식: experiment,best_epoch,best_mAP50,best_mAP50-95,best_precision,best_recall,...
        folder=$(echo "$line" | cut -d',' -f1 | xargs)
        epoch=$(echo "$line" | cut -d',' -f2 | xargs)  # best_epoch
        map50_95=$(echo "$line" | cut -d',' -f4 | xargs)
    else
        # detr_diff 형식: folder,epoch,time,train/loss_bbox,...
        folder=$(echo "$line" | cut -d',' -f1 | xargs)
        epoch=$(echo "$line" | cut -d',' -f2 | xargs)  # epoch
        map50_95=$(echo "$line" | cut -d',' -f10 | xargs)  # metrics/mAP50-95(B)
    fi
    
    # 폴더명 정리 (공백 제거)
    folder=$(echo "$folder" | xargs)
    epoch=$(echo "$epoch" | xargs)
    
    # 체크포인트 경로 (detr_05에 있는 체크포인트 사용)
    checkpoint_path="${BASE_DIR}/outputs/detr_05/${folder}/best.pt"
    
    # 체크포인트 파일 확인
    if [ ! -f "${checkpoint_path}" ]; then
        echo "[${current_line}/${total_lines}] ⚠️  체크포인트 없음: ${checkpoint_path}"
        continue
    fi
    
    echo "[${current_line}/${total_lines}] mAP50-95: ${map50_95}, Epoch: ${epoch}"
    run_single_test "${checkpoint_path}" "${folder}" "${epoch}"
    echo ""
done

echo "============================================================"
echo "모든 테스트 완료!"
echo "결과 저장 위치: ${OUTPUT_BASE_DIR}"
echo "============================================================"