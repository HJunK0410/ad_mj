#!/usr/bin/env python3
"""
NPP Loss 하이퍼파라미터 Grid Search 스크립트
모든 조합을 체계적으로 테스트합니다.
"""

import itertools
import subprocess
import os
import sys
from pathlib import Path
import csv
from datetime import datetime

# 환경 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# Grid Search 파라미터 정의
lambda_1_values = [0, 0.02, 0.04, 0.06, 0.08]  # npp_lambda_1d
lambda_2_values = [0, 0.02, 0.04, 0.06, 0.08]  # npp_lambda_2d
bbox_mask_weight = [0.1, 0.2, 0.3]  
fpn_sources = ["14,17,20"]  # FPN_SOURCES

# 결과 저장 디렉토리
results_dir = Path("runs/detect/grid_search_npp")
results_dir.mkdir(parents=True, exist_ok=True)
results_csv = results_dir / "grid_search_results.csv"

# 이미 존재하는 run을 건너뛸 때 사용할 기준 경로 (v12/runs)
runs_base = Path(script_dir) / "runs" / "detect"

# 모든 조합 생성 (bbox_mask_weight도 포함)
all_combinations = list(itertools.product(
    lambda_1_values,
    lambda_2_values,
    bbox_mask_weight,
    fpn_sources
))

# 필터링: lambda_1=lambda_2=0인 경우 mask=0.1만 포함
filtered_combinations = []
for l1, l2, mask, fpn in all_combinations:
    # lambda_1=0, lambda_2=0인 경우: mask=0.1만 포함 (mask가 의미 없으므로)
    if l1 == 0 and l2 == 0:
        if mask == 0.1:
            filtered_combinations.append((l1, l2, mask, fpn))
        # mask=0.2, 0.3인 경우는 제외
        continue
    # 나머지는 모두 포함 (lambda_1=0이거나 lambda_2=0이어도 포함)
    filtered_combinations.append((l1, l2, mask, fpn))

print(f"==========================================")
print(f"NPP Loss Grid Search 시작")
print(f"==========================================")
print(f"총 조합 수: {len(filtered_combinations)}")
print(f"Lambda 1 값: {lambda_1_values}")
print(f"Lambda 2 값: {lambda_2_values}")
print(f"Bbox Mask Weight: {bbox_mask_weight}")
print(f"FPN Sources: {fpn_sources}")
print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"==========================================")
print()

# CSV 헤더 작성
with open(results_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'iteration', 'npp_lambda_2d', 'npp_lambda_1d', 'npp_bbox_mask_weight', 
        'npp_fpn_sources', 'fitness', 'mAP50', 'mAP50-95', 'precision', 'recall',
        'status', 'save_dir'
    ])

# 각 조합 실행
for idx, (l1, l2, mask, fpn) in enumerate(filtered_combinations, 1):
    fpn_str = fpn.replace(',', '_')
    # train.py와 동일한 이름 형식 (0 -> "0.0", 그대로 매칭되도록 float 사용)
    save_dir_name = f"train_npp_l2d{float(l1)}_l1d{float(l2)}_mask{float(mask)}_fpn{fpn_str}"
    save_dir = runs_base / save_dir_name

    # 이미 존재하는 run은 건너뛰기
    if save_dir.exists():
        print(f"\n[{idx}/{len(filtered_combinations)}] 건너뜀 (이미 존재: {save_dir_name})")
        ckpt_file = save_dir / "weights" / "best.pt"
        if not ckpt_file.exists():
            ckpt_file = save_dir / "weights" / "last.pt"
        metrics = {}
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
        with open(results_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                idx, l1, l2, mask, fpn,
                round(fitness, 5), round(map50, 5), round(map50_95, 5),
                round(precision, 5), round(recall, 5),
                "skipped", save_dir_name
            ])
        continue

    print(f"\n[{idx}/{len(filtered_combinations)}] 실행 중...")
    print(f"  Lambda 2D: {l1}, Lambda 1D: {l2}, Mask: {mask}, FPN: {fpn}")
    
    # train.py 실행
    cmd = [
        "python", "train.py",
        f"--npp_lambda_2d={l1}",
        f"--npp_lambda_1d={l2}",
        f"--npp_bbox_mask_weight={mask}",
        f"--npp_fpn_sources={fpn}",
        "--batch=2",
        "--device=0",
        "--epochs=400"  # 빠른 테스트용, 필요시 변경
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            env=os.environ.copy(),
            # capture_output=False로 설정하여 학습 중 출력이 화면에 표시되도록 함
            check=True
        )
        
        # 결과 디렉토리 (save_dir_name, save_dir는 위에서 이미 정의됨)
        save_dir = runs_base / save_dir_name
        
        # 체크포인트에서 메트릭 읽기
        metrics = {}
        ckpt_file = save_dir / "weights" / "best.pt"
        if not ckpt_file.exists():
            ckpt_file = save_dir / "weights" / "last.pt"
        
        if ckpt_file.exists():
            import torch
            ckpt = torch.load(ckpt_file, map_location='cpu')
            metrics = ckpt.get("train_metrics", {})
        
        fitness = metrics.get("fitness", 0.0)
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)
        
        status = "success"
        print(f"  ✓ 완료 - Fitness: {fitness:.5f}, mAP50: {map50:.5f}")
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ 실패: {e}")
        fitness = 0.0
        map50 = 0.0
        map50_95 = 0.0
        precision = 0.0
        recall = 0.0
        status = "failed"
        save_dir_name = ""
    
    # 결과를 CSV에 저장
    with open(results_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            idx, l1, l2, mask, fpn,
            round(fitness, 5), round(map50, 5), round(map50_95, 5),
            round(precision, 5), round(recall, 5),
            status, save_dir_name
        ])

print(f"\n==========================================")
print(f"Grid Search 완료!")
print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"결과는 {results_csv}에 저장되었습니다.")
print(f"==========================================")

# 최고 결과 출력
if results_csv.exists():
    import pandas as pd
    try:
        df = pd.read_csv(results_csv)
        best_idx = df['fitness'].idxmax()
        best_row = df.iloc[best_idx]
        print(f"\n최고 결과:")
        print(f"  Lambda 2D: {best_row['npp_lambda_2d']}")
        print(f"  Lambda 1D: {best_row['npp_lambda_1d']}")
        print(f"  Bbox Mask Weight: {best_row['npp_bbox_mask_weight']}")
        print(f"  FPN Sources: {best_row['npp_fpn_sources']}")
        print(f"  Fitness: {best_row['fitness']:.5f}")
        print(f"  mAP50: {best_row['mAP50']:.5f}")
        print(f"  mAP50-95: {best_row['mAP50-95']:.5f}")
    except ImportError:
        print("\n최고 결과를 확인하려면 pandas를 설치하세요: pip install pandas")
    except Exception as e:
        print(f"\n최고 결과 확인 중 오류: {e}")
