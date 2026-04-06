# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
YOLO format dataset support for DETR.
Converts YOLO format to COCO format on-the-fly (v1과 동일한 방식).
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

import datasets.transforms as T
from util.misc import NestedTensor, nested_tensor_from_tensor_list

# pycocotools는 선택적 의존성 (평가용)
_has_pycocotools = False
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _has_pycocotools = True
except ImportError:
    pass


def _infer_label_folder_from_image_folder(img_folder: Path, default_root: Path = None, image_set: str = None) -> Path:
    """Infer YOLO label folder from an image folder path robustly.

    Supports:
    - .../images/train -> .../labels/train
    - .../images/val   -> .../labels/val
    - .../images       -> .../labels
    """
    img_folder = Path(img_folder)
    parts = list(img_folder.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        label_parts = parts.copy()
        label_parts[idx] = "labels"
        return Path(*label_parts)

    if default_root is not None and image_set is not None:
        candidate = Path(default_root) / "labels" / image_set
        if candidate.exists():
            return candidate
        return Path(default_root) / "labels"

    return img_folder.parent / "labels"


class YOLODataset(torch.utils.data.Dataset):
    """
    YOLO format dataset that converts to COCO format for DETR.
    
    YOLO format (v1과 동일):
    - Images: path/images/train/, path/images/val/ 또는 path/train, path/val
    - Labels: path/labels/train/, path/labels/val/ 또는 자동으로 images를 labels로 변환
    - Label format: class_id x_center y_center width height (normalized 0-1)
    """
    
    def __init__(self, img_folder, label_folder, transforms, return_masks=False):
        self.img_folder = Path(img_folder)
        self.label_folder = Path(label_folder) if label_folder else None
        self.transforms = transforms
        self.return_masks = return_masks
        
        # Get all image files
        self.im_files = sorted(list(self.img_folder.glob("*.jpg")) + 
                              list(self.img_folder.glob("*.png")) +
                              list(self.img_folder.glob("*.jpeg")))
        
        # Get corresponding label files (v1 방식: images를 labels로 변환)
        self.label_files = []
        for im_file in self.im_files:
            if self.label_folder:
                label_file = self.label_folder / (im_file.stem + ".txt")
            else:
                # v1 방식: images/ -> labels/ 자동 변환
                label_file = Path(str(im_file).replace('/images/', '/labels/').replace('\\images\\', '\\labels\\'))
                if not label_file.exists():
                    # Fallback: 같은 폴더에서 찾기
                    label_file = im_file.parent / (im_file.stem + ".txt")
            self.label_files.append(label_file)
        
        print(f"Found {len(self.im_files)} images in {img_folder}")
        
        # COCO API 객체를 위한 annotation 생성 (평가용)
        self._coco = None
        self._create_coco_api()
    
    def __len__(self):
        return len(self.im_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.im_files[idx]
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Load labels (YOLO format: class_id x_center y_center width height)
        label_path = self.label_files[idx]
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center = float(parts[1])  # normalized 0-1
                        y_center = float(parts[2])  # normalized 0-1
                        width = float(parts[3])      # normalized 0-1
                        height = float(parts[4])   # normalized 0-1
                        
                        # YOLO format: normalized cxcywh -> absolute xyxy (COCO transforms가 xyxy를 기대)
                        x1 = (x_center - width / 2) * w
                        y1 = (y_center - height / 2) * h
                        x2 = (x_center + width / 2) * w
                        y2 = (y_center + height / 2) * h
                        
                        # Clamp to image boundaries
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        boxes.append([x1, y1, x2, y2])  # xyxy format (absolute)
                        labels.append(cls_id)
        
        # Create target dict (COCO format)
        # image_id는 COCO 형식에 맞게 1부터 시작 (idx는 0부터 시작)
        target = {
            'image_id': torch.tensor([idx + 1]),  # COCO는 1부터 시작
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            'orig_size': torch.tensor([h, w]),
            'size': torch.tensor([h, w]),
        }
        
        # Add area and iscrowd for COCO compatibility
        if boxes:
            areas = torch.tensor([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=torch.float32)
            target['area'] = areas
            target['iscrowd'] = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            target['iscrowd'] = torch.zeros((0,), dtype=torch.int64)
        
        # Apply transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def _create_coco_api(self):
        """Create COCO API object from YOLO dataset for evaluation."""
        if not _has_pycocotools:
            print("⚠️  WARNING: pycocotools가 설치되지 않았습니다. COCO evaluator를 사용할 수 없습니다.")
            print("   설치 방법: pip install pycocotools")
            self._coco = None
            return
        
        # COCO 형식 annotation 생성
        coco_ann = {
            "info": {"description": "YOLO dataset converted to COCO format"},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Categories 생성 (클래스 ID는 0부터 시작, COCO는 1부터 시작하므로 +1)
        # 실제 클래스 수는 라벨 파일에서 추정
        max_class_id = -1
        for label_file in self.label_files:
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            max_class_id = max(max_class_id, cls_id)
        
        # Categories 생성 (기본값: 1개 클래스, 또는 max_class_id + 1)
        # 스크레치 탐지의 경우 보통 1개 클래스(스크레치)를 사용
        # DETR은 num_classes + 1 (background 포함)이므로, 실제 클래스는 0부터 num_classes-1까지
        # COCO는 1부터 시작하므로 1부터 num_classes까지
        # max_class_id가 -1이면 클래스가 없는 경우이므로 기본값 1 사용 (스크레치 1개 클래스)
        num_classes = max(max_class_id + 1, 1) if max_class_id >= 0 else 1
        for i in range(num_classes):
            coco_ann["categories"].append({
                "id": i + 1,  # COCO는 1부터 시작 (YOLO class 0 -> COCO category 1)
                "name": f"class_{i}",
                "supercategory": "object"
            })
        
        # Images와 Annotations 생성
        ann_id = 1
        for img_idx, (im_file, label_file) in enumerate(zip(self.im_files, self.label_files)):
            # Image 정보
            img = Image.open(im_file).convert('RGB')
            w, h = img.size
            
            image_id = img_idx + 1
            coco_ann["images"].append({
                "id": image_id,
                "width": w,
                "height": h,
                "file_name": im_file.name
            })
            
            # Annotation 정보
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # YOLO format -> COCO format (absolute coordinates)
                            x = (x_center - width / 2) * w
                            y = (y_center - height / 2) * h
                            bbox_w = width * w
                            bbox_h = height * h
                            
                            # Clamp to image boundaries
                            x = max(0, min(x, w))
                            y = max(0, min(y, h))
                            bbox_w = min(bbox_w, w - x)
                            bbox_h = min(bbox_h, h - y)
                            
                            if bbox_w > 0 and bbox_h > 0:
                                coco_ann["annotations"].append({
                                    "id": ann_id,
                                    "image_id": image_id,
                                    "category_id": cls_id + 1,  # COCO는 1부터 시작
                                    "bbox": [x, y, bbox_w, bbox_h],
                                    "area": bbox_w * bbox_h,
                                    "iscrowd": 0
                                })
                                ann_id += 1
        
        # COCO API 객체 생성
        if _has_pycocotools:
            # 임시 JSON 파일로 저장 후 로드
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(coco_ann, f)
                temp_file = f.name
            
            try:
                self._coco = COCO(temp_file)
                print(f"✓ YOLO dataset: Created COCO API with {len(coco_ann['images'])} images, {len(coco_ann['annotations'])} annotations, {len(coco_ann['categories'])} categories")
            except Exception as e:
                print(f"⚠️  WARNING: Failed to create COCO API: {e}")
                print(f"   이미지 수: {len(coco_ann['images'])}, 어노테이션 수: {len(coco_ann['annotations'])}, 카테고리 수: {len(coco_ann['categories'])}")
                import traceback
                print(f"   상세 오류:\n{traceback.format_exc()}")
                self._coco = None
            finally:
                # 임시 파일 삭제
                try:
                    os.unlink(temp_file)
                except:
                    pass
        else:
            print("Warning: pycocotools not available, COCO evaluator will not work")
    
    @property
    def coco(self):
        """Return COCO API object for evaluation."""
        return self._coco


def make_yolo_transforms(image_set):
    """Create transforms for YOLO dataset (same as COCO)."""
    # COCO transforms 재사용
    from .coco import make_coco_transforms
    return make_coco_transforms(image_set)


def build_yolo(image_set, args):
    """Build YOLO format dataset (v1과 동일한 방식, yaml 없이도 작동)."""
    root = Path(args.coco_path)  # Reuse coco_path for YOLO dataset root
    
    # v1 방식: data.yaml 파일이 있으면 읽기 (선택사항)
    data_yaml_path = root / "data.yaml"
    if not data_yaml_path.exists():
        # data.yaml이 없으면 상위 디렉토리에서 찾기
        data_yaml_path = root.parent / "data.yaml"
    
    data_yaml = None
    if data_yaml_path.exists():
        # yaml 파일이 있으면 사용
        import yaml
        with open(data_yaml_path, 'r') as f:
            data_yaml = yaml.safe_load(f)
    
    if data_yaml is not None:
        
        # v1 방식: path와 train/val 경로 처리
        yaml_path = data_yaml.get('path', '')
        if yaml_path:
            # yaml의 path가 절대 경로면 그대로 사용, 상대 경로면 yaml 파일 기준
            base_path = Path(yaml_path)
            if not base_path.is_absolute():
                # yaml 파일이 있는 디렉토리를 기준으로
                base_path = data_yaml_path.parent / base_path
            base_path = base_path.resolve()
        else:
            # path가 없으면 yaml 파일이 있는 디렉토리 사용
            base_path = data_yaml_path.parent.resolve()
        
        train_path = data_yaml.get('train', 'images/train')
        val_path = data_yaml.get('val', 'images/val')
        
        if image_set == 'train':
            img_folder = base_path / train_path
        else:  # val
            img_folder = base_path / val_path
        
        # v1 방식: images/ -> labels/ 자동 변환
        label_folder = _infer_label_folder_from_image_folder(img_folder, base_path, image_set)
    else:
        # yaml 파일이 없으면 v1처럼 자동으로 경로 찾기
        # v1 방식: images/train, images/val 또는 images/ 폴더 자동 탐색
        possible_paths = [
            root / "images" / image_set,  # images/train 또는 images/val
            root / "images",              # images/ (train/val 구분 없음)
            root / image_set,             # train/ 또는 val/ 직접
            root,                         # root 자체가 이미지 폴더
        ]
        
        img_folder = None
        for path in possible_paths:
            if path.exists() and path.is_dir():
                # 이미지 파일이 있는지 확인
                img_files = list(path.glob("*.jpg")) + list(path.glob("*.png")) + list(path.glob("*.jpeg"))
                if img_files:
                    img_folder = path
                    break
        
        if img_folder is None:
            raise FileNotFoundError(
                f"이미지 폴더를 찾을 수 없습니다: {root}\n"
                f"다음 경로 중 하나가 필요합니다:\n"
                f"  - {root}/images/{image_set}/\n"
                f"  - {root}/images/\n"
                f"  - {root}/{image_set}/\n"
                f"또는 data.yaml 파일을 사용하세요."
            )
        
        # v1 방식: images/ -> labels/ 자동 변환
        if img_folder == root:
            # root가 이미지 폴더면 labels/ 폴더 찾기
            label_folder = root / "labels"
        else:
            label_folder = _infer_label_folder_from_image_folder(img_folder, root, image_set)
    
    assert img_folder.exists(), f'Image folder does not exist: {img_folder}'
    # label_folder는 없어도 됨 (v1처럼 자동으로 찾음)
    if label_folder and not label_folder.exists():
        print(f'Info: Label folder does not exist: {label_folder}, labels will be searched automatically')
        label_folder = None
    
    dataset = YOLODataset(
        img_folder=str(img_folder),
        label_folder=str(label_folder) if label_folder else None,
        transforms=make_yolo_transforms(image_set),
        return_masks=args.masks
    )
    return dataset
