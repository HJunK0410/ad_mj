#!/usr/bin/env python3
"""
차분 로스(Diff Loss) 하이퍼파라미터 Grid Search - Half Training Data
학습 데이터의 절반만 사용하여 v8_diff, v10_diff, v12_diff에 대해 실험합니다.
모든 버전에서 동일한 학습 데이터 서브셋을 사용합니다. (seed=42)

기본 baseline (alpha=0, beta=0, 커스텀 로스 없음)도 함께 실행합니다.
"""

import itertools
import subprocess
import os
import sys
import random
from pathlib import Path
import csv
from datetime import datetime

# ============================================================
# 설정
# ============================================================
SEED = 42  # 모든 버전에서 동일한 데이터 서브셋을 사용하기 위한 고정 시드
AD_ROOT = Path("/home/user/hyunjun/AD")
DATA_ROOT = AD_ROOT / "data"
DATA_HALF_YAML = DATA_ROOT / "data_half.yaml"

# 버전별 설정
VERSION_CONFIGS = {
    "v8_diff": {
        "dir": AD_ROOT / "v8_diff",
        "fpn_sources": "15,18,21",
        "batch": 4,
    },
    "v10_diff": {
        "dir": AD_ROOT / "v10_diff",
        "fpn_sources": "16,19,22",
        "batch": 4,
    },
    "v12_diff": {
        "dir": AD_ROOT / "v12_diff",
        "fpn_sources": "14,17,20",
        "batch": 2,
    },
}

# Grid Search 파라미터 정의 (기존과 동일)
alpha_values = [0.0, 0.1, 0.3, 0.5]    # 1차 차분 가중치 (alpha)
beta_values = [0.0, 0.5, 1.0, 1.5, 2.0]  # 2차 차분 가중치 (beta)

# 학습 설정
EPOCHS = 400
DEVICE = "0"

# ============================================================
# Half Training Data 준비 함수
# ============================================================
def prepare_half_train_data():
    """
    고정 시드(seed=42)로 학습 데이터의 절반을 선택하여 심볼릭 링크로 구성합니다.
    이미 생성되어 있으면 기존 것을 사용합니다.
    """
    train_img_dir = DATA_ROOT / "images" / "train"
    train_lbl_dir = DATA_ROOT / "labels" / "train"
    half_img_dir = DATA_ROOT / "images" / "train_half"
    half_lbl_dir = DATA_ROOT / "labels" / "train_half"

    # 전체 학습 이미지 목록
    all_images = sorted([
        f.name for f in train_img_dir.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
    ])
    total_count = len(all_images)
    half_count = total_count // 2

    # 고정 시드로 절반 선택
    random.seed(SEED)
    half_images = sorted(random.sample(all_images, half_count))

    # 디렉토리 생성
    half_img_dir.mkdir(parents=True, exist_ok=True)
    half_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 기존 심볼릭 링크 확인
    existing_links = set(f.name for f in half_img_dir.iterdir()) if half_img_dir.exists() else set()
    expected_set = set(half_images)

    if existing_links == expected_set:
        print(f"Half training data가 이미 준비되어 있습니다: {half_count}/{total_count} 이미지")
        _save_file_list(half_images, half_count, total_count)
        return half_images

    # 기존 링크 제거 후 재생성
    print(f"Half training data 준비 중... (seed={SEED})")
    for f in half_img_dir.iterdir():
        f.unlink()
    for f in half_lbl_dir.iterdir():
        f.unlink()

    # 심볼릭 링크 생성
    created_img = 0
    created_lbl = 0
    for img_name in half_images:
        src_img = train_img_dir / img_name
        dst_img = half_img_dir / img_name
        if src_img.exists():
            os.symlink(src_img, dst_img)
            created_img += 1

        lbl_name = Path(img_name).stem + '.txt'
        src_lbl = train_lbl_dir / lbl_name
        dst_lbl = half_lbl_dir / lbl_name
        if src_lbl.exists():
            os.symlink(src_lbl, dst_lbl)
            created_lbl += 1

    print(f"  이미지 심볼릭 링크 생성: {created_img}/{half_count}")
    print(f"  라벨 심볼릭 링크 생성: {created_lbl}/{half_count}")
    print(f"  Half training data 준비 완료: {half_count}/{total_count} 이미지")

    _save_file_list(half_images, half_count, total_count)
    return half_images


def _save_file_list(half_images, half_count, total_count):
    """선택된 파일 목록을 기록용으로 저장"""
    list_file = DATA_ROOT / "train_half_filelist.txt"
    with open(list_file, 'w') as f:
        f.write(f"# Half Training Data File List\n")
        f.write(f"# Seed: {SEED}\n")
        f.write(f"# Selected: {half_count} / {total_count}\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"#\n")
        for img_name in half_images:
            f.write(f"{img_name}\n")
    print(f"  파일 목록 저장: {list_file}")


def create_half_data_yaml():
    """data_half.yaml 생성 (공용)"""
    content = (
        f"path: {DATA_ROOT}\n"
        f"train: images/train_half\n"
        f"val: images/val\n"
        f"# Classes\n"
        f"\n"
        f"names:  ['Stamp','Stamp_line']\n"
    )
    with open(DATA_HALF_YAML, 'w') as f:
        f.write(content)
    print(f"data_half.yaml 생성: {DATA_HALF_YAML}")


# ============================================================
# Grid Search 메인 로직
# ============================================================
def run_grid_search():
    """모든 버전에 대해 Diff Loss Grid Search 실행 (baseline 포함)"""

    # 1. Half training data 준비
    print("=" * 60)
    print("Half Training Data 차분 로스(Diff Loss) Grid Search")
    print("(baseline 포함)")
    print("=" * 60)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    half_images = prepare_half_train_data()
    create_half_data_yaml()
    print()

    # 2. 모든 하이퍼파라미터 조합 생성
    all_combinations = list(itertools.product(
        alpha_values,
        beta_values,
    ))

    # baseline (alpha=0, beta=0) 포함 - 기존에는 제외했지만 여기서는 포함
    # alpha=0, beta=0인 경우가 baseline (커스텀 로스 없는 순수 YOLO)
    filtered_combinations = list(all_combinations)  # 모든 조합 포함

    total_per_version = len(filtered_combinations)
    print(f"하이퍼파라미터 조합 수 (버전당): {total_per_version}")
    print(f"  (baseline alpha=0, beta=0 포함)")
    print(f"Alpha 값 (1차 차분 가중치): {alpha_values}")
    print(f"Beta 값 (2차 차분 가중치): {beta_values}")
    print(f"Epochs: {EPOCHS}, Device: {DEVICE}")
    print()

    # 3. 각 버전에 대해 Grid Search 실행
    for version_name, config in VERSION_CONFIGS.items():
        version_dir = config["dir"]
        fpn = config["fpn_sources"]
        batch = config["batch"]
        fpn_str = fpn.replace(',', '_')

        print()
        print("=" * 60)
        print(f"[{version_name.upper()}] 차분 로스 Grid Search (Half Training Data)")
        print(f"  디렉토리: {version_dir}")
        print(f"  FPN Sources: {fpn}")
        print(f"  Batch: {batch}")
        print("=" * 60)

        # 결과 저장 디렉토리
        results_dir = version_dir / "runs" / "detect_half" / "grid_search_diff"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_csv = results_dir / "grid_search_results.csv"

        # 이미 존재하는 run 확인을 위한 base 경로
        runs_base = version_dir / "runs" / "detect_half"

        # CSV 헤더 작성
        with open(results_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'iteration', 'diff_alpha', 'diff_beta',
                'diff_fpn_sources', 'fitness', 'mAP50', 'mAP50-95', 'precision', 'recall',
                'status', 'save_dir', 'is_baseline'
            ])

        # 각 조합 실행
        for idx, (alpha, beta) in enumerate(filtered_combinations, 1):
            is_baseline = (alpha == 0.0 and beta == 0.0)

            if is_baseline:
                # baseline: 차분로스 없음 → train.py에서 자동으로 npp 이름으로 생성됨
                save_dir_name = f"train_npp_l2d0.0_l1d0.0_mask0.3_fpn{fpn_str}"
                exp_label = "BASELINE (no custom loss)"
            else:
                save_dir_name = f"train_diff_alpha{alpha}_beta{beta}_fpn{fpn_str}"
                exp_label = f"Alpha: {alpha}, Beta: {beta}"

            save_dir = runs_base / save_dir_name

            # 이미 존재하는 run 건너뛰기
            if save_dir.exists():
                print(f"\n  [{idx}/{total_per_version}] 건너뜀 (이미 존재: {save_dir_name})")
                fitness, map50, map50_95, precision, recall = _read_metrics_from_checkpoint(save_dir)
                _write_csv_row(results_csv, idx, alpha, beta, fpn,
                               fitness, map50, map50_95, precision, recall,
                               "skipped", save_dir_name, is_baseline)
                continue

            print(f"\n  [{idx}/{total_per_version}] 실행 중...")
            print(f"    {exp_label}, FPN: {fpn}")

            # train.py 실행
            cmd = [
                "python", "train.py",
                f"--npp_alpha={alpha}",
                f"--npp_beta={beta}",
                f"--npp_fpn_sources={fpn}",
                f"--batch={batch}",
                f"--device={DEVICE}",
                f"--epochs={EPOCHS}",
                f"--data={DATA_HALF_YAML}",
                f"--project=runs/detect_half",
            ]

            env = os.environ.copy()
            env['PYTHONPATH'] = str(version_dir) + ':' + env.get('PYTHONPATH', '')

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(version_dir),
                    env=env,
                    check=True
                )

                fitness, map50, map50_95, precision, recall = _read_metrics_from_checkpoint(save_dir)
                status = "success"
                print(f"    ✓ 완료 - Fitness: {fitness:.5f}, mAP50: {map50:.5f}")

            except subprocess.CalledProcessError as e:
                print(f"    ✗ 실패: {e}")
                fitness, map50, map50_95, precision, recall = 0.0, 0.0, 0.0, 0.0, 0.0
                status = "failed"
                save_dir_name = ""

            _write_csv_row(results_csv, idx, alpha, beta, fpn,
                           fitness, map50, map50_95, precision, recall,
                           status, save_dir_name, is_baseline)

        # 버전별 최고 결과 출력
        _print_best_result_diff(results_csv, version_name)

    print(f"\n{'=' * 60}")
    print(f"모든 Grid Search 완료!")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")


# ============================================================
# 유틸리티 함수
# ============================================================
def _read_metrics_from_checkpoint(save_dir):
    """체크포인트에서 메트릭 읽기"""
    metrics = {}
    ckpt_file = save_dir / "weights" / "best.pt"
    if not ckpt_file.exists():
        ckpt_file = save_dir / "weights" / "last.pt"

    if ckpt_file.exists():
        try:
            import torch
            ckpt = torch.load(ckpt_file, map_location='cpu')
            metrics = ckpt.get("train_metrics", {})
        except Exception:
            pass

    fitness = metrics.get("fitness", 0.0)
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    precision = metrics.get("metrics/precision(B)", 0.0)
    recall = metrics.get("metrics/recall(B)", 0.0)
    return fitness, map50, map50_95, precision, recall


def _write_csv_row(results_csv, idx, alpha, beta, fpn,
                   fitness, map50, map50_95, precision, recall,
                   status, save_dir_name, is_baseline):
    """CSV에 결과 행 추가"""
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            idx, alpha, beta, fpn,
            round(fitness, 5), round(map50, 5), round(map50_95, 5),
            round(precision, 5), round(recall, 5),
            status, save_dir_name, is_baseline
        ])


def _print_best_result_diff(results_csv, version_name):
    """버전별 최고 결과 출력 (baseline과 비교)"""
    if results_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            if df.empty or df['fitness'].max() == 0:
                return

            # Baseline 결과
            baseline_df = df[df['is_baseline'] == True]
            if not baseline_df.empty:
                bl = baseline_df.iloc[0]
                print(f"\n  [{version_name.upper()}] Baseline 결과 (커스텀 로스 없음):")
                print(f"    Fitness: {bl['fitness']:.5f}")
                print(f"    mAP50: {bl['mAP50']:.5f}")
                print(f"    mAP50-95: {bl['mAP50-95']:.5f}")

            # 차분 로스가 있는 실험 중 최고
            diff_df = df[df['is_baseline'] == False]
            if not diff_df.empty and diff_df['fitness'].max() > 0:
                best_idx = diff_df['fitness'].idxmax()
                best_row = df.iloc[best_idx]
                print(f"\n  [{version_name.upper()}] 최고 차분 로스 결과:")
                print(f"    Alpha (1차 차분): {best_row['diff_alpha']}")
                print(f"    Beta (2차 차분): {best_row['diff_beta']}")
                print(f"    FPN Sources: {best_row['diff_fpn_sources']}")
                print(f"    Fitness: {best_row['fitness']:.5f}")
                print(f"    mAP50: {best_row['mAP50']:.5f}")
                print(f"    mAP50-95: {best_row['mAP50-95']:.5f}")

                # Baseline 대비 개선
                if not baseline_df.empty:
                    bl_fitness = baseline_df.iloc[0]['fitness']
                    improvement = best_row['fitness'] - bl_fitness
                    print(f"    Baseline 대비 Fitness 변화: {improvement:+.5f}")

        except ImportError:
            print("\n  최고 결과를 확인하려면 pandas를 설치하세요: pip install pandas")
        except Exception as e:
            print(f"\n  최고 결과 확인 중 오류: {e}")


if __name__ == "__main__":
    run_grid_search()
