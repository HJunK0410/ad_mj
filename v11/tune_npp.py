#!/usr/bin/env python3
"""
NPP Loss 하이퍼파라미터 튜닝 스크립트
Ultralytics Tuner를 사용하여 NPP Loss 하이퍼파라미터를 진화 알고리즘으로 튜닝합니다.
"""

from ultralytics import YOLO
from ultralytics.engine.tuner import Tuner
import sys
import os

# 환경 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# 모델 로드
model = YOLO('yolo11m.pt')

# 튜닝 설정
tune_args = {
    'data': './data/data.yaml',
    'epochs': 1,
    'batch': 4,
    'imgsz': 1280,
    'device': "0",
    'workers': 0,
    'amp': False,
    'project': 'runs/detect',
    'name': 'npp_tune',
    # 커스텀 train.py 사용
    'use_custom_train': True,
    'custom_train_script': 'train.py',
    # NPP FPN sources 선택지
    'npp_fpn_sources_choices': ["16", "16,19", "16,19,22"],
    # 튜닝할 하이퍼파라미터 공간 (기본값에 NPP 파라미터가 포함됨)
    'space': {
        # NPP Loss 하이퍼파라미터만 튜닝
        "npp_lambda_2d": (0.0, 0.10),
        "npp_lambda_1d": (0.0, 0.10),
        "npp_bbox_mask_weight": (0.1, 0.3),
    }
}

# Tuner 생성 및 실행
from ultralytics.utils import callbacks
tuner = Tuner(args=tune_args, _callbacks=callbacks.get_default_callbacks())

# 튜닝 실행 (iterations: 튜닝 반복 횟수)
iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 30
print(f"NPP Loss 하이퍼파라미터 튜닝 시작 (iterations: {iterations})")
results = tuner(model=model, iterations=iterations, cleanup=True)

print(f"\n튜닝 완료! 결과는 {tuner.tune_dir}에 저장되었습니다.")
