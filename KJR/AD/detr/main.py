# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # Continuity loss parameters
    parser.add_argument('--lambda_cont_2d', default=0.0, type=float,
                        help="Weight for 2D continuity loss")
    parser.add_argument('--lambda_cont_1d', default=0.0, type=float,
                        help="Weight for 1D continuity loss")
    parser.add_argument('--curriculum_epochs', default=5, type=int,
                        help="Number of epochs for curriculum learning (gradual increase of continuity loss weight)")
    parser.add_argument('--norm_mask_bbox_coef', default=0.3, type=float,
                        help="Normalization mask coefficient for bbox regions (default: 0.3)")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--data_ratio', type=float, default=1.0,
                        help='데이터셋 사용 비율 (0.0 ~ 1.0, 예: 0.5면 50%%만 사용, 기본값: 1.0 = 전체 사용)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='사용할 최대 샘플 수 (정확한 개수 지정, 예: 120이면 정확히 120개만 사용). data_subset_file보다 우선순위가 낮습니다.')
    parser.add_argument('--data_subset_file', type=str, default=None,
                        help='사용할 데이터 인덱스를 저장/로드할 파일 경로 (JSON 형식). 지정하면 data_ratio나 max_samples 대신 이 파일의 인덱스를 사용합니다.')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--no_checkpoint', action='store_true',
                        help='disable checkpoint saving during training')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--early_stop_patience', default=20, type=int,
                        help='Early stopping patience (epochs without mAP50-95 improvement). 0 = disabled.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # eval 모드일 때는 train 데이터셋을 빌드하지 않고, test 데이터셋만 사용
    if not args.eval:
        dataset_train = build_dataset(image_set='train', args=args)

    # eval 모드면 test, 아니면 val 사용
    dataset_val = build_dataset(image_set='test' if args.eval else 'val', args=args)
    
    # 데이터셋에서 파일명을 가져오는 헬퍼 함수
    def get_dataset_filenames(dataset):
        """데이터셋에서 파일명 리스트를 가져옵니다."""
        filenames = []
        # YOLODataset의 경우
        if hasattr(dataset, 'im_files'):
            filenames = [Path(f).name for f in dataset.im_files]
        # COCO dataset의 경우
        elif hasattr(dataset, 'coco'):
            if dataset.coco is not None:
                for img_id in dataset.coco.getImgIds():
                    img_info = dataset.coco.loadImgs(img_id)[0]
                    filenames.append(img_info['file_name'])
        # 다른 데이터셋의 경우 (fallback)
        else:
            # 데이터셋 크기만큼 인덱스로 파일명 생성 (최후의 수단)
            for i in range(len(dataset)):
                filenames.append(f"sample_{i:06d}.jpg")
        return filenames
    
    # 파일명으로 인덱스를 찾는 헬퍼 함수
    def find_indices_by_filenames(dataset, target_filenames):
        """파일명 리스트로부터 인덱스를 찾습니다."""
        all_filenames = get_dataset_filenames(dataset)
        filename_to_idx = {name: idx for idx, name in enumerate(all_filenames)}
        indices = []
        missing_files = []
        for filename in target_filenames:
            if filename in filename_to_idx:
                indices.append(filename_to_idx[filename])
            else:
                missing_files.append(filename)
        if missing_files:
            print(f"⚠️  경고: 다음 파일들을 데이터셋에서 찾을 수 없습니다: {missing_files[:5]}{'...' if len(missing_files) > 5 else ''}")
        return sorted(indices)
    
    # 데이터 샘플링 (일부만 사용) - eval 모드에서는 건너뜀
    if not args.eval and (args.data_ratio < 1.0 or args.max_samples is not None or args.data_subset_file is not None):
        from torch.utils.data import Subset
        
        # 데이터셋 크기
        train_size = len(dataset_train)
        val_size = len(dataset_val)
        
        # 샘플링할 인덱스 결정 (우선순위: data_subset_file > max_samples > data_ratio)
        if args.data_subset_file is not None:
            # 파일에서 파일명 또는 인덱스 로드
            subset_file = Path(args.data_subset_file)
            if subset_file.exists():
                print(f"📂 데이터 서브셋 파일 로드: {args.data_subset_file}")
                with open(subset_file, 'r') as f:
                    subset_data = json.load(f)
                
                # 파일명이 있으면 파일명 사용 (새 형식), 없으면 인덱스 사용 (구 형식 호환)
                train_filenames = subset_data.get('train_filenames', [])
                val_filenames = subset_data.get('val_filenames', [])
                
                if train_filenames or val_filenames:
                    # 새 형식: 파일명으로 인덱스 찾기
                    print("   파일명 기반 서브셋 사용 (재현성 보장)")
                    train_indices = find_indices_by_filenames(dataset_train, train_filenames)
                    val_indices = find_indices_by_filenames(dataset_val, val_filenames)
                else:
                    # 구 형식: 인덱스 직접 사용 (하위 호환성)
                    train_indices = subset_data.get('train_indices', [])
                    val_indices = subset_data.get('val_indices', [])
                
                print(f"   학습 데이터: {len(train_indices)}/{train_size} 샘플 사용")
                print(f"   검증 데이터: {len(val_indices)}/{val_size} 샘플 사용")
            else:
                # 파일이 없으면 새로 생성
                print(f"📝 데이터 서브셋 파일 생성: {args.data_subset_file}")
                # 샘플링 개수 결정
                if args.max_samples is not None:
                    num_train_samples = min(args.max_samples, train_size)
                    num_val_samples = min(args.max_samples, val_size)
                else:
                    num_train_samples = int(train_size * args.data_ratio)
                    num_val_samples = int(val_size * args.data_ratio)
                
                train_indices = sorted(random.sample(range(train_size), num_train_samples))
                val_indices = sorted(random.sample(range(val_size), num_val_samples))
                
                # 파일명 리스트 가져오기
                train_filenames = get_dataset_filenames(dataset_train)
                val_filenames = get_dataset_filenames(dataset_val)
                selected_train_filenames = [train_filenames[i] for i in train_indices]
                selected_val_filenames = [val_filenames[i] for i in val_indices]
                
                # 파일 저장 (파일명 기반)
                subset_file.parent.mkdir(parents=True, exist_ok=True)
                save_data = {
                    'train_filenames': selected_train_filenames,
                    'val_filenames': selected_val_filenames,
                    'train_indices': train_indices,  # 하위 호환성을 위해 유지
                    'val_indices': val_indices,      # 하위 호환성을 위해 유지
                    'train_size': train_size,
                    'val_size': val_size,
                    'total_train_files': len(train_filenames),
                    'total_val_files': len(val_filenames)
                }
                if args.max_samples is not None:
                    save_data['max_samples'] = args.max_samples
                else:
                    save_data['data_ratio'] = args.data_ratio
                with open(subset_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                print(f"   학습 데이터: {len(selected_train_filenames)}/{train_size} 샘플 저장 (파일명 기반)")
                print(f"   검증 데이터: {len(selected_val_filenames)}/{val_size} 샘플 저장 (파일명 기반)")
        elif args.max_samples is not None:
            # max_samples로 샘플링 - YAML 파일이 있으면 사용, 없으면 생성
            # YAML 파일 저장 경로: datasets/yaml/
            save_dir = Path('datasets/yaml')
            save_dir.mkdir(parents=True, exist_ok=True)
            
            yaml_file = save_dir / f"subset_data_{args.max_samples}.yaml"
            json_file = save_dir / f"subset_data_{args.max_samples}.json"
            
            if yaml_file.exists() and yaml is not None:
                # YAML 파일이 있으면 로드
                print(f"📂 데이터 서브셋 YAML 파일 로드: {yaml_file}")
                with open(yaml_file, 'r') as f:
                    subset_data = yaml.safe_load(f)
                
                train_filenames = subset_data.get('train_filenames', [])
                val_filenames = subset_data.get('val_filenames', [])
                
                if train_filenames or val_filenames:
                    print("   파일명 기반 서브셋 사용 (재현성 보장)")
                    train_indices = find_indices_by_filenames(dataset_train, train_filenames)
                    val_indices = find_indices_by_filenames(dataset_val, val_filenames)
                else:
                    train_indices = subset_data.get('train_indices', [])
                    val_indices = subset_data.get('val_indices', [])
                
                print(f"   학습 데이터: {len(train_indices)}/{train_size} 샘플 사용")
                print(f"   검증 데이터: {len(val_indices)}/{val_size} 샘플 사용")
            else:
                # YAML 파일이 없으면 랜덤 샘플링 후 생성
                print(f"📝 데이터 서브셋 YAML 파일 생성: {yaml_file}")
                num_train_samples = min(args.max_samples, train_size)
                num_val_samples = min(args.max_samples, val_size)
                train_indices = sorted(random.sample(range(train_size), num_train_samples))
                val_indices = sorted(random.sample(range(val_size), num_val_samples))
                
                # 파일명 리스트 가져오기
                train_filenames = get_dataset_filenames(dataset_train)
                val_filenames = get_dataset_filenames(dataset_val)
                selected_train_filenames = [train_filenames[i] for i in train_indices]
                selected_val_filenames = [val_filenames[i] for i in val_indices]
                
                # 저장할 데이터 준비
                save_data = {
                    'train_filenames': selected_train_filenames,
                    'val_filenames': selected_val_filenames,
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'train_size': train_size,
                    'val_size': val_size,
                    'total_train_files': len(train_filenames),
                    'total_val_files': len(val_filenames),
                    'max_samples': args.max_samples
                }
                
                # JSON 파일로도 저장
                with open(json_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                # YAML 파일로 저장
                if yaml is not None:
                    with open(yaml_file, 'w') as f:
                        yaml.dump(save_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                    print(f"   YAML 파일 저장: {yaml_file}")
                else:
                    print("   ⚠️  yaml 모듈이 없어 YAML 파일을 저장할 수 없습니다. JSON만 저장됩니다.")
                
                print(f"   JSON 파일 저장: {json_file}")
                print(f"   학습 데이터: {len(selected_train_filenames)}/{train_size} 샘플 저장")
                print(f"   검증 데이터: {len(selected_val_filenames)}/{val_size} 샘플 저장")
            
            print(f"📊 데이터 샘플링: 최대 {args.max_samples}개 사용")
            print(f"   학습 데이터: {len(train_indices)}/{train_size} 샘플")
            print(f"   검증 데이터: {len(val_indices)}/{val_size} 샘플")
        else:
            # data_ratio로 샘플링 - YAML 파일이 있으면 사용, 없으면 생성
            # YAML 파일 저장 경로: datasets/yaml/
            save_dir = Path('datasets/yaml')
            save_dir.mkdir(parents=True, exist_ok=True)
            
            ratio_str = f"{int(args.data_ratio * 100)}pct"
            yaml_file = save_dir / f"subset_data_{ratio_str}.yaml"
            json_file = save_dir / f"subset_data_{ratio_str}.json"
            
            if yaml_file.exists() and yaml is not None:
                # YAML 파일이 있으면 로드
                print(f"📂 데이터 서브셋 YAML 파일 로드: {yaml_file}")
                with open(yaml_file, 'r') as f:
                    subset_data = yaml.safe_load(f)
                
                train_filenames = subset_data.get('train_filenames', [])
                val_filenames = subset_data.get('val_filenames', [])
                
                if train_filenames or val_filenames:
                    print("   파일명 기반 서브셋 사용 (재현성 보장)")
                    train_indices = find_indices_by_filenames(dataset_train, train_filenames)
                    val_indices = find_indices_by_filenames(dataset_val, val_filenames)
                else:
                    train_indices = subset_data.get('train_indices', [])
                    val_indices = subset_data.get('val_indices', [])
                
                print(f"   학습 데이터: {len(train_indices)}/{train_size} 샘플 사용")
                print(f"   검증 데이터: {len(val_indices)}/{val_size} 샘플 사용")
            else:
                # YAML 파일이 없으면 랜덤 샘플링 후 생성
                print(f"📝 데이터 서브셋 YAML 파일 생성: {yaml_file}")
                num_train_samples = int(train_size * args.data_ratio)
                num_val_samples = int(val_size * args.data_ratio)
                train_indices = sorted(random.sample(range(train_size), num_train_samples))
                val_indices = sorted(random.sample(range(val_size), num_val_samples))
                
                # 파일명 리스트 가져오기
                train_filenames = get_dataset_filenames(dataset_train)
                val_filenames = get_dataset_filenames(dataset_val)
                selected_train_filenames = [train_filenames[i] for i in train_indices]
                selected_val_filenames = [val_filenames[i] for i in val_indices]
                
                # 저장할 데이터 준비
                save_data = {
                    'train_filenames': selected_train_filenames,
                    'val_filenames': selected_val_filenames,
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'train_size': train_size,
                    'val_size': val_size,
                    'total_train_files': len(train_filenames),
                    'total_val_files': len(val_filenames),
                    'data_ratio': args.data_ratio
                }
                
                # JSON 파일로도 저장
                with open(json_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                # YAML 파일로 저장
                if yaml is not None:
                    with open(yaml_file, 'w') as f:
                        yaml.dump(save_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                    print(f"   YAML 파일 저장: {yaml_file}")
                else:
                    print("   ⚠️  yaml 모듈이 없어 YAML 파일을 저장할 수 없습니다. JSON만 저장됩니다.")
                
                print(f"   JSON 파일 저장: {json_file}")
                print(f"   학습 데이터: {len(selected_train_filenames)}/{train_size} 샘플 저장")
                print(f"   검증 데이터: {len(selected_val_filenames)}/{val_size} 샘플 저장")
            
            print(f"📊 데이터 샘플링: {args.data_ratio*100:.1f}% 사용")
            print(f"   학습 데이터: {len(train_indices)}/{train_size} 샘플")
            print(f"   검증 데이터: {len(val_indices)}/{val_size} 샘플")
        
        # Subset으로 감싸기
        dataset_train = Subset(dataset_train, train_indices)
        dataset_val = Subset(dataset_val, val_indices)

    # DataLoader / Sampler 설정
    if not args.eval:
        # 학습 + 검증 모드
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )
    else:
        # eval 전용 모드: train 데이터로더 생성 안 함, val만 사용
        if args.distributed:
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    
    # base_ds가 None인지 확인 (YOLO 데이터셋에서 COCO API 생성 실패 시)
    if base_ds is None:
        print("=" * 60)
        print("⚠️  WARNING: COCO API 객체를 생성할 수 없습니다!")
        print("=" * 60)
        print("가능한 원인:")
        print("1. pycocotools가 설치되지 않았습니다: pip install pycocotools")
        print("2. YOLO 데이터셋의 COCO API 생성 중 오류가 발생했습니다")
        print("3. 데이터셋에 이미지나 라벨이 없습니다")
        if hasattr(dataset_val, 'coco'):
            print(f"   dataset_val.coco = {dataset_val.coco}")
        if hasattr(dataset_val, '_coco'):
            print(f"   dataset_val._coco = {dataset_val._coco}")
        print("=" * 60)
        print("평가 메트릭(mAP, precision, recall)이 0으로 표시될 수 있습니다.")
        print("=" * 60)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu', weights_only=False)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            
            # 테스트 결과를 CSV로 저장 (조합 이름으로)
            if utils.is_main_process():
                # output_dir의 마지막 폴더명을 조합 이름으로 사용
                combo_name = output_dir.name if output_dir.name else "test_result"
                csv_path = output_dir / f"{combo_name}.csv"
                
                metrics_dict = {}
                
                # Test metrics
                metrics_dict['test/loss_ce'] = test_stats.get('loss_ce', 0.0)
                metrics_dict['test/loss_bbox'] = test_stats.get('loss_bbox', 0.0)
                metrics_dict['test/loss_giou'] = test_stats.get('loss_giou', 0.0)
                
                # Metrics (mAP, precision, recall)
                if 'v1_precision' in test_stats and 'v1_recall' in test_stats:
                    metrics_dict['metrics/precision(B)'] = test_stats.get('v1_precision', 0.0)
                    metrics_dict['metrics/recall(B)'] = test_stats.get('v1_recall', 0.0)
                    if 'coco_eval_bbox' in test_stats:
                        coco_stats = test_stats['coco_eval_bbox']
                        if isinstance(coco_stats, list) and len(coco_stats) >= 12:
                            metrics_dict['metrics/mAP50(B)'] = coco_stats[1] if len(coco_stats) > 1 else 0.0
                            metrics_dict['metrics/mAP50-95(B)'] = coco_stats[0] if len(coco_stats) > 0 else 0.0
                        else:
                            metrics_dict['metrics/mAP50(B)'] = 0.0
                            metrics_dict['metrics/mAP50-95(B)'] = 0.0
                    else:
                        metrics_dict['metrics/mAP50(B)'] = 0.0
                        metrics_dict['metrics/mAP50-95(B)'] = 0.0
                elif 'v1_map50' in test_stats:
                    metrics_dict['metrics/mAP50(B)'] = test_stats.get('v1_map50', 0.0)
                    metrics_dict['metrics/mAP50-95(B)'] = test_stats.get('v1_map50_95', 0.0)
                    metrics_dict['metrics/precision(B)'] = test_stats.get('v1_precision', 0.0)
                    metrics_dict['metrics/recall(B)'] = test_stats.get('v1_recall', 0.0)
                elif 'coco_eval_bbox' in test_stats and coco_evaluator is not None:
                    coco_stats = test_stats['coco_eval_bbox']
                    if isinstance(coco_stats, list) and len(coco_stats) >= 12:
                        metrics_dict['metrics/mAP50(B)'] = coco_stats[1] if len(coco_stats) > 1 else 0.0
                        metrics_dict['metrics/mAP50-95(B)'] = coco_stats[0] if len(coco_stats) > 0 else 0.0
                        if 'coco_precision_bbox' in test_stats:
                            metrics_dict['metrics/precision(B)'] = test_stats['coco_precision_bbox']
                        else:
                            metrics_dict['metrics/precision(B)'] = 0.0
                        if 'coco_recall_bbox' in test_stats:
                            metrics_dict['metrics/recall(B)'] = test_stats['coco_recall_bbox']
                        else:
                            metrics_dict['metrics/recall(B)'] = 0.0
                    else:
                        metrics_dict['metrics/precision(B)'] = 0.0
                        metrics_dict['metrics/recall(B)'] = 0.0
                        metrics_dict['metrics/mAP50(B)'] = 0.0
                        metrics_dict['metrics/mAP50-95(B)'] = 0.0
                else:
                    metrics_dict['metrics/precision(B)'] = 0.0
                    metrics_dict['metrics/recall(B)'] = 0.0
                    metrics_dict['metrics/mAP50(B)'] = 0.0
                    metrics_dict['metrics/mAP50-95(B)'] = 0.0
                
                # CSV 헤더 작성
                test_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('test/')]
                metrics_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('metrics/')]
                keys = test_keys + metrics_keys
                
                # CSV 작성
                with csv_path.open("w") as f:
                    f.write(",".join(keys) + "\n")
                    values = [str(metrics_dict.get(k, 0.0)) for k in keys]
                    f.write(",".join(values) + "\n")
                
                print(f"✓ 테스트 결과 저장: {csv_path}")
        return

    print("Start training")
    start_time = time.time()
    best_map50_95 = 0.0  # Track best mAP50-95 for saving best model
    early_stop_counter = 0  # Early stopping counter
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            # Save to log.txt (JSON format)
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # Save to CSV (v1과 동일한 형식)
            csv_path = output_dir / "results.csv"
            metrics_dict = {}
            
            # Train metrics (v1 형식에 맞게 변환)
            # DETR loss: loss_ce, loss_bbox, loss_giou
            metrics_dict['train/loss_ce'] = train_stats.get('loss_ce', 0.0)
            metrics_dict['train/loss_bbox'] = train_stats.get('loss_bbox', 0.0)
            metrics_dict['train/loss_giou'] = train_stats.get('loss_giou', 0.0)
            if 'loss_continuity_2d' in train_stats:
                metrics_dict['train/loss_continuity_2d'] = train_stats.get('loss_continuity_2d', 0.0)
            if 'loss_continuity_1d' in train_stats:
                metrics_dict['train/loss_continuity_1d'] = train_stats.get('loss_continuity_1d', 0.0)
            
            # Test/Val metrics
            metrics_dict['val/loss_ce'] = test_stats.get('loss_ce', 0.0)
            metrics_dict['val/loss_bbox'] = test_stats.get('loss_bbox', 0.0)
            metrics_dict['val/loss_giou'] = test_stats.get('loss_giou', 0.0)
            
            # COCO eval metrics 또는 v1 방식 mAP50 (v1 형식: metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B))
            # 우선순위: v1 방식 > COCO evaluator
            if 'v1_precision' in test_stats and 'v1_recall' in test_stats:
                # v1 방식 precision/recall 사용
                metrics_dict['metrics/precision(B)'] = test_stats.get('v1_precision', 0.0)
                metrics_dict['metrics/recall(B)'] = test_stats.get('v1_recall', 0.0)
                # mAP50과 mAP50-95는 COCO evaluator에서 가져옴
                if 'coco_eval_bbox' in test_stats:
                    coco_stats = test_stats['coco_eval_bbox']
                    if isinstance(coco_stats, list) and len(coco_stats) >= 12:
                        metrics_dict['metrics/mAP50(B)'] = coco_stats[1] if len(coco_stats) > 1 else 0.0
                        metrics_dict['metrics/mAP50-95(B)'] = coco_stats[0] if len(coco_stats) > 0 else 0.0
                    else:
                        metrics_dict['metrics/mAP50(B)'] = 0.0
                        metrics_dict['metrics/mAP50-95(B)'] = 0.0
                else:
                    metrics_dict['metrics/mAP50(B)'] = 0.0
                    metrics_dict['metrics/mAP50-95(B)'] = 0.0
            elif 'v1_map50' in test_stats:
                # v1 방식 mAP50 사용
                metrics_dict['metrics/mAP50(B)'] = test_stats.get('v1_map50', 0.0)
                metrics_dict['metrics/mAP50-95(B)'] = test_stats.get('v1_map50_95', 0.0)
                metrics_dict['metrics/precision(B)'] = test_stats.get('v1_precision', 0.0)
                metrics_dict['metrics/recall(B)'] = test_stats.get('v1_recall', 0.0)
            elif 'coco_eval_bbox' in test_stats and coco_evaluator is not None:
                coco_stats = test_stats['coco_eval_bbox']
                if isinstance(coco_stats, list) and len(coco_stats) >= 12:
                    # COCO stats: [AP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
                    # AP = mAP50-95, AP50 = mAP50
                    metrics_dict['metrics/mAP50(B)'] = coco_stats[1] if len(coco_stats) > 1 else 0.0
                    metrics_dict['metrics/mAP50-95(B)'] = coco_stats[0] if len(coco_stats) > 0 else 0.0
                    # Precision과 Recall은 COCO evaluator에서 계산 (IoU 0.5에서)
                    # Precision: TP/(TP+FP) at IoU 0.5
                    if 'coco_precision_bbox' in test_stats:
                        metrics_dict['metrics/precision(B)'] = test_stats['coco_precision_bbox']
                    else:
                        # Fallback: 계산 실패 시 0.0
                        metrics_dict['metrics/precision(B)'] = 0.0
                    # Recall: TP/(TP+FN) at IoU 0.5 (AR100이 아닌 실제 recall)
                    if 'coco_recall_bbox' in test_stats:
                        metrics_dict['metrics/recall(B)'] = test_stats['coco_recall_bbox']
                    else:
                        # Fallback: 계산 실패 시 AR100 사용 (이전 방식과의 호환성)
                        metrics_dict['metrics/recall(B)'] = coco_stats[8] if len(coco_stats) > 8 else 0.0
                else:
                    metrics_dict['metrics/precision(B)'] = 0.0
                    metrics_dict['metrics/recall(B)'] = 0.0
                    metrics_dict['metrics/mAP50(B)'] = 0.0
                    metrics_dict['metrics/mAP50-95(B)'] = 0.0
            else:
                metrics_dict['metrics/precision(B)'] = 0.0
                metrics_dict['metrics/recall(B)'] = 0.0
                metrics_dict['metrics/mAP50(B)'] = 0.0
                metrics_dict['metrics/mAP50-95(B)'] = 0.0
            
            # Learning rate (v1 형식: lr/pg0, lr/pg1, lr/pg2)
            for i, param_group in enumerate(optimizer.param_groups):
                metrics_dict[f'lr/pg{i}'] = param_group['lr']
            
            # Time
            elapsed_time = time.time() - start_time
            
            # CSV 헤더 작성 (첫 epoch일 때만, 또는 헤더가 없는 경우)
            # v1 형식 순서: epoch, time, train/..., metrics/..., val/..., lr/...
            need_header = False
            if not csv_path.exists():
                need_header = True
            else:
                # 기존 파일에 헤더가 있는지 확인
                with csv_path.open("r") as f:
                    first_line = f.readline().strip()
                    # 헤더가 없거나 잘못된 형식인 경우 (숫자로 시작하면 데이터 행)
                    if not first_line or first_line.split(",")[0].isdigit():
                        need_header = True
            
            if need_header:
                # 순서대로 정렬: train -> metrics -> val -> lr
                train_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('train/')]
                metrics_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('metrics/')]
                val_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('val/')]
                lr_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('lr/')]
                keys = ['epoch', 'time'] + train_keys + metrics_keys + val_keys + lr_keys
                # 헤더가 없으면 기존 파일을 덮어쓰기
                with csv_path.open("w") as f:
                    f.write(",".join(keys) + "\n")
            
            # CSV 데이터 작성 (헤더 순서와 동일하게)
            # 헤더 읽기
            with csv_path.open("r") as f:
                header_line = f.readline().strip()
                if not header_line:
                    # 헤더가 없으면 다시 작성
                    train_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('train/')]
                    metrics_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('metrics/')]
                    val_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('val/')]
                    lr_keys = [k for k in sorted(metrics_dict.keys()) if k.startswith('lr/')]
                    keys = ['epoch', 'time'] + train_keys + metrics_keys + val_keys + lr_keys
                    header_line = ",".join(keys)
                header_keys = header_line.split(",")
            
            # epoch, time 제외한 키 순서
            data_keys = [k for k in header_keys if k not in ['epoch', 'time']]
            values = [epoch + 1, elapsed_time] + [metrics_dict.get(k, 0.0) for k in data_keys]
            with csv_path.open("a") as f:
                f.write(",".join([f"{v:.6g}" if isinstance(v, (int, float)) else str(v) for v in values]) + "\n")

            # Save best model based on mAP50-95
            if 'coco_eval_bbox' in test_stats:
                coco_stats = test_stats['coco_eval_bbox']
                if isinstance(coco_stats, list) and len(coco_stats) > 0:
                    current_map50_95 = coco_stats[0]  # mAP50-95
                    if current_map50_95 > best_map50_95:
                        best_map50_95 = current_map50_95
                        early_stop_counter = 0
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'map50_95': best_map50_95,
                        }, output_dir / 'best.pt')
                        print(f'Saved best model at epoch {epoch+1} with mAP50-95: {best_map50_95:.6f}')
                    else:
                        early_stop_counter += 1
                        print(f'EarlyStopping counter: {early_stop_counter}/{args.early_stop_patience} (best mAP50-95: {best_map50_95:.6f})')

        # Early stopping check (outside output_dir block, applies to all processes)
        if args.early_stop_patience > 0 and early_stop_counter >= args.early_stop_patience:
            print(f'Early stopping triggered at epoch {epoch+1}: no improvement for {args.early_stop_patience} epochs.')
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
