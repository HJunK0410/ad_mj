from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import sys

# 명령줄 인자에서 하이퍼파라미터 파싱
def parse_args():
    """명령줄 인자에서 차분 로스(diff loss) 관련 하이퍼파라미터 및 표준 파라미터 파싱"""
    # 차분 로스 하이퍼파라미터 기본값
    npp_alpha = 0.3  # 1차 차분 가중치
    npp_beta = 1.0   # 2차 차분 가중치
    npp_fpn_sources = "16,19,22"  # 기본값: 모든 레벨 사용
    # multi(NPP) continuity 가중치 — 미지정 시 trainer 기본(0.02/0.06)과 동일
    npp_lambda_2d = 0.02
    npp_lambda_1d = 0.06
    npp_bbox_mask_weight = 1.0  # trainer 기본과 동일; grid에서는 0.1~0.3 전달 가능

    # 표준 파라미터 기본값
    batch = 2
    device = "3"
    epochs = 400
    patience = 20
    imgsz = 1280
    workers = 0
    amp = False
    data = './data/data.yaml'  # 기본 데이터 경로
    project = 'runs/detect'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--npp_alpha='):
            npp_alpha = float(arg.split('=')[1])
        elif arg.startswith('--npp_beta='):
            npp_beta = float(arg.split('=')[1])
        elif arg.startswith('--npp_fpn_sources='):
            npp_fpn_sources = arg.split('=')[1]
        elif arg.startswith('--npp_lambda_2d='):
            npp_lambda_2d = float(arg.split('=')[1])
        elif arg.startswith('--npp_lambda_1d='):
            npp_lambda_1d = float(arg.split('=')[1])
        elif arg.startswith('--npp_bbox_mask_weight='):
            npp_bbox_mask_weight = float(arg.split('=')[1])
        elif arg.startswith('--batch='):
            batch = int(arg.split('=')[1])
        elif arg.startswith('--device='):
            device = arg.split('=')[1]
        elif arg.startswith('--epochs='):
            epochs = int(arg.split('=')[1])
        elif arg.startswith('--patience='):
            patience = int(arg.split('=')[1])
        elif arg.startswith('--imgsz='):
            imgsz = int(arg.split('=')[1])
        elif arg.startswith('--workers='):
            workers = int(arg.split('=')[1])
        elif arg.startswith('--amp='):
            amp = arg.lower() == 'true'
        elif arg.startswith('--data='):
            data = arg.split('=')[1]
        elif arg.startswith('--project='):
            project = arg.split('=')[1]
    
    return {
        'npp_alpha': npp_alpha,
        'npp_beta': npp_beta,
        'npp_fpn_sources': npp_fpn_sources,
        'npp_lambda_2d': npp_lambda_2d,
        'npp_lambda_1d': npp_lambda_1d,
        'npp_bbox_mask_weight': npp_bbox_mask_weight,
        'batch': batch,
        'device': device,
        'epochs': epochs,
        'patience': patience,
        'imgsz': imgsz,
        'workers': workers,
        'amp': amp,
        'data': data,
        'project': project
    }

# 하이퍼파라미터 파싱
hyperparams = parse_args()

# 모델 로드
model = YOLO('yolo11m.pt')  # .pt 파일만 지정하면 자동으로 yaml과 가중치 로드

# 차분 로스 하이퍼파라미터를 별도로 저장 (trainer args에 주입)
npp_params = {
    'npp_alpha': hyperparams['npp_alpha'],
    'npp_beta': hyperparams['npp_beta'],
    'npp_fpn_sources': hyperparams['npp_fpn_sources'],
    'npp_lambda_2d': hyperparams['npp_lambda_2d'],
    'npp_lambda_1d': hyperparams['npp_lambda_1d'],
    'npp_bbox_mask_weight': hyperparams['npp_bbox_mask_weight'],
}

# 하이퍼파라미터 기반 프로젝트 이름 생성 (mask 포함 — grid 조합별 고유 경로)
FPN_STR = hyperparams['npp_fpn_sources'].replace(',', '_')
_m = hyperparams['npp_bbox_mask_weight']
_L2_DEF, _L1_DEF = 0.02, 0.06
_default_multi = (
    abs(hyperparams['npp_lambda_2d'] - _L2_DEF) < 1e-9
    and abs(hyperparams['npp_lambda_1d'] - _L1_DEF) < 1e-9
)
if hyperparams['npp_alpha'] > 0.0 or hyperparams['npp_beta'] > 0.0:
    if _default_multi or (
        hyperparams['npp_lambda_2d'] <= 0.0 and hyperparams['npp_lambda_1d'] <= 0.0
    ):
        PROJECT_NAME = (
            f"train_diff_alpha{hyperparams['npp_alpha']}_beta{hyperparams['npp_beta']}_"
            f"mask{_m}_fpn{FPN_STR}"
        )
    else:
        PROJECT_NAME = (
            f"train_combo_l2d{hyperparams['npp_lambda_2d']}_l1d{hyperparams['npp_lambda_1d']}_"
            f"mask{_m}_fpn{FPN_STR}_alpha{hyperparams['npp_alpha']}_beta{hyperparams['npp_beta']}"
        )
else:
    PROJECT_NAME = (
        f"train_npp_l2d{hyperparams['npp_lambda_2d']}_l1d{hyperparams['npp_lambda_1d']}_"
        f"mask{_m}_fpn{FPN_STR}"
    )

# overrides에는 표준 인자만 포함 (NPP 인자는 check_dict_alignment에서 에러 발생)
# 명령줄 인자에서 받은 값 사용 (없으면 기본값)
overrides = {
    'data': hyperparams.get('data', './data/data.yaml'),
    'epochs': hyperparams.get('epochs', 400),
    'patience': hyperparams.get('patience', 20),
    'batch': hyperparams.get('batch', 2),
    'imgsz': hyperparams.get('imgsz', 1280),
    'scale': 0.9,
    'mosaic': 0.6,
    'mixup': 0.15,
    'copy_paste': 0.4,
    'device': hyperparams.get('device', "3"),
    'workers': hyperparams.get('workers', 0),
    'amp': hyperparams.get('amp', False),  # AMP 비활성화 (Segmentation fault 방지)
    'project': hyperparams.get('project', 'runs/detect'),  # 프로젝트 디렉토리
    'name': PROJECT_NAME,  # 하이퍼파라미터 기반 이름
    'exist_ok': True  # 같은 이름의 디렉토리가 있으면 덮어쓰기 (추가 '2' 생성 방지)
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
        
        # 차분 로스 관련 인자를 제거한 args 복사본 생성
        validator_args = copy(self.args)
        npp_arg_names = [
            'npp_alpha',
            'npp_beta',
            'npp_fpn_sources',
            'npp_lambda_2d',
            'npp_lambda_1d',
            'npp_bbox_mask_weight',
        ]
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
# 학습 중 이미 검증이 수행되므로, 별도의 val() 호출은 선택사항입니다.
# 같은 디렉터리를 사용하도록 project와 name을 명시적으로 전달
metrics = model.val(
    project=overrides['project'],
    name=PROJECT_NAME,
    exist_ok=True
)

# Perform object detection on an image
#results = model("path/to/image.jpg")
#results[0].show()