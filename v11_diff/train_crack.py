from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import sys

# 명령줄 인자에서 하이퍼파라미터 파싱
def parse_args():
    """명령줄 인자에서 차분 로스(diff loss) 관련 하이퍼파라미터 및 표준 파라미터 파싱 (crack-seg용)"""
    # 차분 로스 하이퍼파라미터 기본값
    npp_alpha = 0.1   # 1차 차분 가중치
    npp_beta = 0.5    # 2차 차분 가중치
    npp_fpn_sources = "16,19,22"  # 기본값: 모든 레벨 사용
    
    # 표준 파라미터 기본값 (crack-seg에 맞게 조정)
    batch = 16
    device = "3"
    epochs = 100
    imgsz = 640
    workers = 4
    amp = False
    
    for arg in sys.argv[1:]:
        if arg.startswith('--npp_alpha='):
            npp_alpha = float(arg.split('=')[1])
        elif arg.startswith('--npp_beta='):
            npp_beta = float(arg.split('=')[1])
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
        'npp_alpha': npp_alpha,
        'npp_beta': npp_beta,
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

# 차분 로스 하이퍼파라미터를 별도로 저장
npp_params = {
    'npp_alpha': hyperparams['npp_alpha'],
    'npp_beta': hyperparams['npp_beta'],
    'npp_fpn_sources': hyperparams['npp_fpn_sources']
}

# 하이퍼파라미터 기반 프로젝트 이름 생성
FPN_STR = hyperparams['npp_fpn_sources'].replace(',', '_')
PROJECT_NAME = f"crack_diff_alpha{hyperparams['npp_alpha']}_beta{hyperparams['npp_beta']}_fpn{FPN_STR}"

print("=" * 50)
print(f"[차분 Loss] crack-seg 학습")
print(f"  alpha={hyperparams['npp_alpha']}, beta={hyperparams['npp_beta']}")
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
        # 차분 로스 하이퍼파라미터를 args에 직접 추가 (check_dict_alignment 이후)
        for key, value in npp_params.items():
            setattr(self.args, key, value)
    
    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation with diff loss args filtered out."""
        from copy import copy
        from ultralytics.models import yolo
        
        # 차분 로스 관련 인자를 제거한 args 복사본 생성
        validator_args = copy(self.args)
        npp_arg_names = ['npp_alpha', 'npp_beta', 'npp_fpn_sources']
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
