#!/bin/bash

# best.csv 또는 디렉토리를 기반으로 자동 테스트하는 스크립트
# 사용법:
#   bash test_best.sh [디렉토리 경로] [test_image_dir] [test_label_dir]
#   bash test_best.sh [best.csv 경로] [test_image_dir] [test_label_dir]
export CUDA_VISIBLE_DEVICES=2  # GPU 1번 사용

# 기본 설정
# 현재 스크립트가 있는 디렉토리를 기준으로 BASE_DIR 자동 감지
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-${SCRIPT_DIR}}"

# 프로젝트 이름 자동 감지 (detr_diff 또는 detr)
PROJECT_NAME=$(basename "${BASE_DIR}")

# 입력 인자 (디렉토리 또는 CSV)
INPUT_ARG="${1:-${BASE_DIR}/outputs/${PROJECT_NAME}}"

COCO_PATH="${COCO_PATH:-/home/user/KJR/AD/data/image/test}"
TEST_IMAGE_DIR="${TEST_IMAGE_DIR:-${2:-/home/user/hyunjun/AD/data/images/test}}"
TEST_LABEL_DIR="${TEST_LABEL_DIR:-${3:-/home/user/hyunjun/AD/data/labels/test}}"
BATCH_SIZE="${BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-}"

# 입력 인자 타입 확인
if [[ "${INPUT_ARG}" == *.csv ]]; then
    MODE="csv"
    BEST_CSV="${INPUT_ARG}"
    if [ ! -f "${BEST_CSV}" ]; then
        echo "❌ 오류: best.csv 파일을 찾을 수 없습니다: ${BEST_CSV}"
        exit 1
    fi
elif [ -d "${INPUT_ARG}" ]; then
    MODE="directory"
    TARGET_DIR="${INPUT_ARG}"
else
    echo "❌ 오류: 입력 인자를 인식할 수 없습니다: ${INPUT_ARG}"
    echo "사용법:"
    echo "  bash test_best.sh [디렉토리 경로] [test_image_dir] [test_label_dir]"
    echo "  bash test_best.sh [best.csv 경로] [test_image_dir] [test_label_dir]"
    exit 1
fi

# 출력 디렉토리 기본값 설정
# - directory 모드: 입력 폴더 안의 results/
# - csv 모드: CSV 파일이 있는 폴더 안의 results/
if [ -z "${OUTPUT_BASE_DIR}" ]; then
    if [ "${MODE}" = "directory" ]; then
        OUTPUT_BASE_DIR="${TARGET_DIR}/results"
    else
        OUTPUT_BASE_DIR="$(dirname "${BEST_CSV}")/results"
    fi
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

    TEST_RUNTIME_ROOT="/tmp/detr_diff_test_yolo_${USER}_$$"
    mkdir -p "${TEST_RUNTIME_ROOT}/images" "${TEST_RUNTIME_ROOT}/labels"
    ln -sfn "${TEST_IMAGE_DIR}" "${TEST_RUNTIME_ROOT}/images/val"
    ln -sfn "${TEST_LABEL_DIR}" "${TEST_RUNTIME_ROOT}/labels/val"
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

echo "============================================================"
if [ "${MODE}" = "directory" ]; then
    echo "디렉토리 기반 자동 테스트 시작"
    echo "============================================================"
    echo "대상 디렉토리: ${TARGET_DIR}"
else
    echo "best.csv 기반 자동 테스트 시작"
    echo "============================================================"
    echo "CSV 파일: ${BEST_CSV}"
fi
echo "============================================================"
echo "테스트 데이터셋: ${COCO_PATH}"
echo "배치 크기: ${BATCH_SIZE}"
echo "출력 디렉토리: ${OUTPUT_BASE_DIR}"
echo "============================================================"
echo ""

# 폴더명에서 하이퍼파라미터 추출 함수
extract_hyperparams() {
    local folder_name=$1
    
    # 폴더명 형식: test_2d2_1d10_nm30_a50_b50
    # 2d2 -> lambda_cont_2d = 0.02
    # 1d10 -> lambda_cont_1d = 0.10
    # nm30 -> norm_mask_bbox_coef = 0.30
    # a50 -> alpha = 0.50
    # b50 -> beta = 0.50
    
    # 2d 값 추출
    if [[ $folder_name =~ _2d([0-9]+) ]]; then
        lambda_2d_str="${BASH_REMATCH[1]}"
        # 앞에 0이 있으면 제거하고 소수점 추가
        lambda_2d_str=$(echo "$lambda_2d_str" | sed 's/^0*//')
        if [ -z "$lambda_2d_str" ]; then
            lambda_2d_str="0"
        fi
        # 2자리 숫자를 0.XX로 변환
        if [ ${#lambda_2d_str} -eq 1 ]; then
            lambda_2d="0.0${lambda_2d_str}"
        elif [ ${#lambda_2d_str} -eq 2 ]; then
            lambda_2d="0.${lambda_2d_str}"
        else
            # 3자리 이상이면 0.XX로 변환 (예: 200 -> 0.20)
            lambda_2d="0.${lambda_2d_str:0:2}"
        fi
    else
        lambda_2d=""
    fi
    
    # 1d 값 추출
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
    
    # a 값 추출
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
    
    # b 값 추출
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

run_single_test() {
    local folder=$1
    local checkpoint_path=$2
    local prefix=$3
    local map50_95=${4:-N/A}

    extract_hyperparams "$folder"

    test_output_dir="${OUTPUT_BASE_DIR}/${folder}"
    mkdir -p "${test_output_dir}"

    echo "============================================================"
    echo "${prefix} 테스트 실행"
    echo "============================================================"
    echo "폴더: ${folder}"
    echo "체크포인트: ${checkpoint_path}"
    echo "mAP50-95: ${map50_95}"
    echo "출력 디렉토리: ${test_output_dir}"

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

    cd "${BASE_DIR}" || exit 1
    python3 test.py \
        --resume "${checkpoint_path}" \
        --coco_path "${COCO_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --device "${DEVICE}" \
        --output_dir "${test_output_dir}" \
        ${hyperparams_args}
}

if [ "${MODE}" = "directory" ]; then
    found_models=0
    success_models=0
    failed_models=0
    for subdir in "${TARGET_DIR}"/*; do
        [ -d "${subdir}" ] || continue
        folder=$(basename "${subdir}")
        checkpoint_path="${subdir}/best.pt"
        if [ ! -f "${checkpoint_path}" ]; then
            continue
        fi
        found_models=$((found_models + 1))
        prefix="[${found_models}]"
        if run_single_test "${folder}" "${checkpoint_path}" "${prefix}" "N/A"; then
            echo "✅ ${prefix} 테스트 완료: ${folder}"
            success_models=$((success_models + 1))
        else
            echo "❌ ${prefix} 테스트 실패: ${folder}"
            failed_models=$((failed_models + 1))
        fi
        echo ""
    done
    echo "총 모델: ${found_models} / 성공: ${success_models} / 실패: ${failed_models}"
else
    total_lines=$(tail -n +2 "${BEST_CSV}" | wc -l)
    current_line=0
    tail -n +2 "${BEST_CSV}" | while IFS=',' read -r folder epoch time train_loss_bbox train_loss_ce train_loss_cont_1d train_loss_cont_2d train_loss_giou map50 map50_95 precision recall val_loss_bbox val_loss_ce val_loss_giou lr_pg0 lr_pg1; do
        current_line=$((current_line + 1))
        folder=$(echo "$folder" | xargs)
        checkpoint_path="${BASE_DIR}/outputs/${PROJECT_NAME}/${folder}/best.pt"
        if [ ! -f "${checkpoint_path}" ]; then
            echo "[${current_line}/${total_lines}] ⚠️  체크포인트 없음: ${checkpoint_path}"
            continue
        fi
        prefix="[${current_line}/${total_lines}]"
        if run_single_test "${folder}" "${checkpoint_path}" "${prefix}" "${map50_95}"; then
            echo "✅ ${prefix} 테스트 완료: ${folder}"
        else
            echo "❌ ${prefix} 테스트 실패: ${folder}"
        fi
        echo ""
    done
fi

echo "============================================================"
echo "모든 테스트 완료!"
echo "결과 저장 위치: ${OUTPUT_BASE_DIR}"
echo "============================================================"

