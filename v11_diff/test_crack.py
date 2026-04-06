#!/usr/bin/env python3
"""
Crack-seg 데이터셋에 대해 학습된 모델의 test 데이터셋 평가 스크립트 (v11_diff)
- Diff Best: crack_diff_alpha0.1_beta0.5_fpn16_19_22
"""

import csv
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import os
import sys
import yaml


# 평가할 실험 목록
EXPERIMENTS = [
    {
        'name': 'crack_diff_alpha0.1_beta0.5_fpn16_19_22',
        'description': 'Diff Best (alpha=0.1, beta=0.5)'
    }
]


def find_weight_file(experiment_name):
    """실험 이름으로부터 best.pt 파일 경로 찾기"""
    weight_path = Path(f"runs/detect/{experiment_name}/weights/best.pt")
    
    if not weight_path.exists():
        weight_path = Path(f"runs/detect/{experiment_name}/weights/last.pt")
        if not weight_path.exists():
            print(f"오류: 가중치 파일을 찾을 수 없습니다: {experiment_name}")
            return None
        print(f"경고: best.pt가 없어서 last.pt를 사용합니다.")
    
    print(f"가중치 파일: {weight_path}")
    return weight_path


def evaluate_on_test(weight_path, experiment_name, data_yaml_path="data/crack_seg.yaml"):
    """crack-seg test 데이터셋에 대해 모델 평가 수행"""
    print(f"\n[{experiment_name}] Test 데이터셋 평가 시작")
    
    # 모델 로드
    print(f"모델 로드 중: {weight_path}")
    model = YOLO(str(weight_path))
    
    # data.yaml 확인
    data_yaml = Path(data_yaml_path)
    if not data_yaml.exists():
        print(f"오류: data.yaml 파일이 존재하지 않습니다: {data_yaml_path}")
        return None, None
    
    # 각 run마다 고유한 이름으로 결과 저장
    eval_name = f"crack_test_{experiment_name}"
    eval_name = eval_name.replace("/", "_").replace("\\", "_")
    
    print(f"Test 데이터셋에 대해 평가 수행 중...")
    results = model.val(
        data=data_yaml_path,
        split='test',
        project="runs/detect",
        name=eval_name,
        exist_ok=True,
        save_json=True,
        save_hybrid=False,
        plots=True
    )
    
    # 메트릭 추출
    metrics = {}
    
    # results 객체에서 직접 메트릭 추출
    if hasattr(results, 'results_dict') and results.results_dict:
        metrics = dict(results.results_dict)
        print(f"results.results_dict에서 메트릭 추출 완료")
    elif hasattr(results, 'box'):
        box_metrics = results.box
        metrics['metrics/precision(B)'] = getattr(box_metrics, 'mp', None) or getattr(box_metrics, 'p', [None])[0]
        metrics['metrics/recall(B)'] = getattr(box_metrics, 'mr', None) or getattr(box_metrics, 'r', [None])[0]
        metrics['metrics/mAP50(B)'] = getattr(box_metrics, 'map50', None)
        metrics['metrics/mAP50-95(B)'] = getattr(box_metrics, 'map', None)
        print(f"results.box에서 메트릭 추출 완료")
    
    # results.csv에서 메트릭 읽기 시도 (fallback)
    if not metrics:
        results_csv_path = Path(f"runs/detect/{eval_name}/results.csv")
        if results_csv_path.exists():
            df_results = pd.read_csv(results_csv_path)
            if not df_results.empty:
                last_row = df_results.iloc[-1]
                for col in df_results.columns:
                    col_lower = col.lower()
                    if 'precision' in col_lower and ('(b)' in col_lower or 'box' in col_lower):
                        metrics['metrics/precision(B)'] = last_row[col]
                    elif 'recall' in col_lower and ('(b)' in col_lower or 'box' in col_lower):
                        metrics['metrics/recall(B)'] = last_row[col]
                    elif 'map50' in col_lower and 'map50-95' not in col_lower and ('(b)' in col_lower or 'box' in col_lower):
                        metrics['metrics/mAP50(B)'] = last_row[col]
                    elif 'map50-95' in col_lower and ('(b)' in col_lower or 'box' in col_lower):
                        metrics['metrics/mAP50-95(B)'] = last_row[col]
                print(f"results.csv에서 메트릭 추출 완료")
    
    return results, metrics


def save_results_to_csv(metrics, experiment_name, description, output_csv="crack_test_results.csv"):
    """평가 결과를 CSV 파일로 저장"""
    output_path = Path(output_csv)
    file_exists = output_path.exists()
    
    with open(output_path, 'a', newline='') as f:
        fieldnames = [
            'experiment_name',
            'description',
            'metrics/precision(B)',
            'metrics/recall(B)',
            'metrics/mAP50(B)',
            'metrics/mAP50-95(B)'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        def find_metric_key(metrics_dict, possible_keys):
            for key in possible_keys:
                if key in metrics_dict:
                    return metrics_dict[key]
            return None
        
        precision = find_metric_key(metrics, ['metrics/precision(B)', 'precision(B)', 'precision'])
        recall = find_metric_key(metrics, ['metrics/recall(B)', 'recall(B)', 'recall'])
        map50 = find_metric_key(metrics, ['metrics/mAP50(B)', 'mAP50(B)', 'map50'])
        map50_95 = find_metric_key(metrics, ['metrics/mAP50-95(B)', 'mAP50-95(B)', 'map50-95', 'map'])
        
        row = {
            'experiment_name': experiment_name,
            'description': description,
            'metrics/precision(B)': round(safe_float(precision), 5),
            'metrics/recall(B)': round(safe_float(recall), 5),
            'metrics/mAP50(B)': round(safe_float(map50), 5),
            'metrics/mAP50-95(B)': round(safe_float(map50_95), 5)
        }
        
        writer.writerow(row)
    
    print(f"  결과 저장 완료 - Precision: {row['metrics/precision(B)']:.5f}, "
          f"Recall: {row['metrics/recall(B)']:.5f}, "
          f"mAP50: {row['metrics/mAP50(B)']:.5f}, "
          f"mAP50-95: {row['metrics/mAP50-95(B)']:.5f}")


def main():
    """메인 함수"""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    os.environ['PYTHONPATH'] = str(script_dir) + ':' + os.environ.get('PYTHONPATH', '')
    
    print("=" * 60)
    print("Crack-seg Test 데이터셋 평가 (v11_diff: Diff Best)")
    print("=" * 60)
    
    # 결과 CSV 초기화
    output_csv = "crack_test_results.csv"
    output_path = Path(output_csv)
    if output_path.exists():
        import shutil
        backup_path = Path(f"{output_csv}.backup")
        shutil.copy2(output_path, backup_path)
        print(f"기존 결과 파일을 백업했습니다: {backup_path}")
        output_path.unlink()
    
    total = len(EXPERIMENTS)
    success = 0
    fail = 0
    
    for idx, exp in enumerate(EXPERIMENTS):
        exp_name = exp['name']
        exp_desc = exp['description']
        
        print(f"\n[{idx + 1}/{total}] {exp_desc}")
        print(f"  실험명: {exp_name}")
        print("-" * 60)
        
        weight_path = find_weight_file(exp_name)
        if weight_path is None:
            print(f"  가중치 파일을 찾을 수 없어 건너뜁니다.")
            fail += 1
            continue
        
        try:
            results, metrics = evaluate_on_test(weight_path, exp_name)
            if results is None or not metrics:
                print(f"  평가 실패: 메트릭을 추출할 수 없습니다.")
                fail += 1
                continue
            
            save_results_to_csv(metrics, exp_name, exp_desc, output_csv)
            success += 1
            
        except Exception as e:
            print(f"  오류 발생: {str(e)}")
            fail += 1
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("평가 완료!")
    print("=" * 60)
    print(f"성공: {success}/{total}")
    print(f"실패: {fail}/{total}")
    print(f"결과 파일: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
