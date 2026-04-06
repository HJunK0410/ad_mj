#!/usr/bin/env python3
"""
crack-seg 데이터셋에 대한 YOLOv10 Detection 학습 스크립트 (차분 로스)
- baseline: 순수 YOLO (차분 loss 없음)
- best: 기존 실험에서 가장 좋았던 차분 loss 하이퍼파라미터 적용
  (alpha=0.5, beta=0.0, fpn=16,19,22 / mAP50-95=0.41554)
"""

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import sys
import os

# 환경 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.environ['PYTHONPATH'] = script_dir + ':' + os.environ.get('PYTHONPATH', '')

# ============================================================
# Best 하이퍼파라미터 (test_results_summary.csv 기준 mAP50-95 최고)
# train_diff_alpha0.5_beta0.0_fpn16_19_22 -> mAP50-95=0.41554
# ============================================================
BEST_HP_DIFF = {
    'diff_alpha': 0.5,
    'diff_beta': 0.0,
    'npp_fpn_sources': '16,19,22',
}
BEST_HP_NPP = {
    'npp_lambda_2d': 0.0,
    'npp_lambda_1d': 0.0,
    'npp_bbox_mask_weight': 0.3,
    'npp_fpn_sources': '16,19,22',
}

BASELINE_HP_DIFF = {
    'diff_alpha': 0.0,
    'diff_beta': 0.0,
    'npp_fpn_sources': '16,19,22',
}
BASELINE_HP_NPP = {
    'npp_lambda_2d': 0.0,
    'npp_lambda_1d': 0.0,
    'npp_bbox_mask_weight': 0.0,
    'npp_fpn_sources': '16,19,22',
}


def parse_args():
    """명령줄 인자 파싱"""
    experiment_type = 'baseline'
    device = '0'
    batch = 2
    epochs = 100
    imgsz = 640
    workers = 0
    amp = False

    for arg in sys.argv[1:]:
        if arg.startswith('--experiment_type='):
            experiment_type = arg.split('=')[1]
        elif arg.startswith('--device='):
            device = arg.split('=')[1]
        elif arg.startswith('--batch='):
            batch = int(arg.split('=')[1])
        elif arg.startswith('--epochs='):
            epochs = int(arg.split('=')[1])
        elif arg.startswith('--imgsz='):
            imgsz = int(arg.split('=')[1])
        elif arg.startswith('--workers='):
            workers = int(arg.split('=')[1])
        elif arg.startswith('--amp='):
            amp = arg.split('=')[1].lower() == 'true'

    return {
        'experiment_type': experiment_type,
        'device': device,
        'batch': batch,
        'epochs': epochs,
        'imgsz': imgsz,
        'workers': workers,
        'amp': amp,
    }


args = parse_args()
experiment_type = args['experiment_type']

# 실험 유형에 따라 하이퍼파라미터 선택
if experiment_type == 'best':
    diff_params = dict(BEST_HP_DIFF)
    npp_params = dict(BEST_HP_NPP)
    PROJECT_NAME = 'crack_best_diff'
    print(f"[crack-seg] Best Diff HP 실험 시작: {diff_params}")
else:
    diff_params = dict(BASELINE_HP_DIFF)
    npp_params = dict(BASELINE_HP_NPP)
    PROJECT_NAME = 'crack_baseline'
    print(f"[crack-seg] Baseline 실험 시작 (차분 loss 비활성화)")

# 모델 로드
model = YOLO('yolov10m.pt')

# 학습 overrides
overrides = {
    'data': './data/crack_seg.yaml',
    'epochs': args['epochs'],
    'batch': args['batch'],
    'imgsz': args['imgsz'],
    'device': args['device'],
    'workers': args['workers'],
    'amp': args['amp'],
    'project': 'runs/detect',
    'name': PROJECT_NAME,
    'exist_ok': True,
}


# 커스텀 trainer 클래스 정의
class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        for key, value in diff_params.items():
            setattr(self.args, key, value)
        for key, value in npp_params.items():
            setattr(self.args, key, value)

    def get_validator(self):
        from copy import copy
        from ultralytics.models import yolo

        validator_args = copy(self.args)
        for name in ['diff_alpha', 'diff_beta', 'npp_lambda_2d', 'npp_lambda_1d', 'npp_bbox_mask_weight', 'npp_fpn_sources']:
            if hasattr(validator_args, name):
                delattr(validator_args, name)

        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=validator_args, _callbacks=self.callbacks
        )


# 학습 실행
print(f"[crack-seg] 학습 시작: {PROJECT_NAME}")
results = model.train(trainer=CustomDetectionTrainer, **overrides)

# test set 평가
print(f"\n[crack-seg] Test set 평가 시작...")
metrics = model.val(
    data='./data/crack_seg.yaml',
    split='test',
    project=overrides['project'],
    name=f'{PROJECT_NAME}_test',
    exist_ok=True,
)
print(f"[crack-seg] Test 결과: mAP50={metrics.box.map50:.5f}, mAP50-95={metrics.box.map:.5f}")
print(f"[crack-seg] 완료: {PROJECT_NAME}")
