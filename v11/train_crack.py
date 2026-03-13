from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import sys

# 명령줄 인자에서 하이퍼파라미터 파싱
def parse_args():
    """명령줄 인자에서 NPP loss 관련 하이퍼파라미터 및 표준 파라미터 파싱 (crack-seg용)"""
    npp_lambda_2d = 0.02
    npp_lambda_1d = 0.08
    npp_bbox_mask_weight = 0.2
    npp_fpn_sources = "16,19,22"  # 기본값: 모든 레벨 사용
    
    # 표준 파라미터 기본값 (crack-seg에 맞게 조정)
    batch = 16
    device = "3"
    epochs = 100
    imgsz = 640
    workers = 4
    amp = False
    
    for arg in sys.argv[1:]:
        if arg.startswith('--npp_lambda_2d='):
            npp_lambda_2d = float(arg.split('=')[1])
        elif arg.startswith('--npp_lambda_1d='):
            npp_lambda_1d = float(arg.split('=')[1])
        elif arg.startswith('--npp_bbox_mask_weight='):
            npp_bbox_mask_weight = float(arg.split('=')[1])
        elif arg.startswith('--npp_fpn_sources='):
            npp_fpn_sources = arg.split('=')[1]
        elif arg.startswith('--batch='):
            batch = int(arg.split('=')[1])
        elif arg.startswith('--device='):
            device = arg.split('=')[1]
        elif arg.startswith('--epochs='):
            epochs = int(arg.split('=')[1])
        elif arg.startswith('--imgsz='):
            imgsz = int(arg.split('=')[1])
        elif arg.startswith('--workers='):
            workers = int(arg.split('=')[1])
        elif arg.startswith('--amp='):
            amp = arg.lower() == 'true'
    
    return {
        'npp_lambda_2d': npp_lambda_2d,
        'npp_lambda_1d': npp_lambda_1d,
        'npp_bbox_mask_weight': npp_bbox_mask_weight,
        'npp_fpn_sources': npp_fpn_sources,
        'batch': batch,
        'device': device,
        'epochs': epochs,
        'imgsz': imgsz,
        'workers': workers,
        'amp': amp
    }

# 하이퍼파라미터 파싱
hyperparams = parse_args()

# 모델 로드
model = YOLO('yolo11m.pt')

# NPP Loss 하이퍼파라미터를 별도로 저장
npp_params = {
    'npp_lambda_2d': hyperparams['npp_lambda_2d'],
    'npp_lambda_1d': hyperparams['npp_lambda_1d'],
    'npp_bbox_mask_weight': hyperparams['npp_bbox_mask_weight'],
    'npp_fpn_sources': hyperparams['npp_fpn_sources']
}

# 하이퍼파라미터 기반 프로젝트 이름 생성
FPN_STR = hyperparams['npp_fpn_sources'].replace(',', '_')
PROJECT_NAME = f"crack_npp_l2d{hyperparams['npp_lambda_2d']}_l1d{hyperparams['npp_lambda_1d']}_mask{hyperparams['npp_bbox_mask_weight']}_fpn{FPN_STR}"

# 베이스라인 여부 확인
is_baseline = (hyperparams['npp_lambda_2d'] == 0.0 and hyperparams['npp_lambda_1d'] == 0.0)
if is_baseline:
    PROJECT_NAME = "crack_baseline"
    print("=" * 50)
    print("[Baseline] 순수 YOLOv11m 학습 (NPP loss 비활성)")
    print("=" * 50)
else:
    print("=" * 50)
    print(f"[NPP Loss] crack-seg 학습")
    print(f"  lambda_2d={hyperparams['npp_lambda_2d']}, lambda_1d={hyperparams['npp_lambda_1d']}, mask={hyperparams['npp_bbox_mask_weight']}")
    print("=" * 50)

# overrides에는 표준 인자만 포함
overrides = {
    'data': './data/crack_seg.yaml',
    'epochs': hyperparams.get('epochs', 100),
    'batch': hyperparams.get('batch', 16),
    'imgsz': hyperparams.get('imgsz', 640),
    'scale': 0.9,
    'mosaic': 0.6,
    'mixup': 0.15,
    'copy_paste': 0.4,
    'device': hyperparams.get('device', "3"),
    'workers': hyperparams.get('workers', 4),
    'amp': hyperparams.get('amp', False),
    'project': 'runs/detect',
    'name': PROJECT_NAME,
    'exist_ok': True
}

# 커스텀 trainer 클래스 정의
class CustomDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # NPP Loss 하이퍼파라미터를 args에 직접 추가 (check_dict_alignment 이후)
        for key, value in npp_params.items():
            setattr(self.args, key, value)
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation with NPP args filtered out."""
        from copy import copy
        from ultralytics.models import yolo
        
        # NPP 관련 인자를 제거한 args 복사본 생성
        validator_args = copy(self.args)
        npp_arg_names = ['npp_lambda_2d', 'npp_lambda_1d', 'npp_bbox_mask_weight', 'npp_fpn_sources']
        for npp_arg in npp_arg_names:
            if hasattr(validator_args, npp_arg):
                delattr(validator_args, npp_arg)
        
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=validator_args, _callbacks=self.callbacks
        )

# YOLO 모델의 train() 메서드를 사용하되, 커스텀 trainer 전달
results = model.train(trainer=CustomDetectionTrainer, **overrides)

# Evaluate model performance on the validation set
metrics = model.val(
    project=overrides['project'],
    name=PROJECT_NAME,
    exist_ok=True
)

print(f"\n학습 완료: {PROJECT_NAME}")
print(f"가중치 저장 위치: runs/detect/{PROJECT_NAME}/weights/best.pt")
