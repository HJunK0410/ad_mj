#!/usr/bin/env python3
"""
DETR 모델 테스트 스크립트
사용법: python test.py --resume <체크포인트 경로> --coco_path <데이터셋 경로>
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='DETR 모델 테스트')
    parser.add_argument('--resume', type=str, required=True,
                       help='테스트할 체크포인트 파일 경로 (예: outputs/detr_diff/test_2d2_1d10_nm30_a100_b100/best.pt)')
    parser.add_argument('--coco_path', type=str, default='/home/user/KJR/AD/data/image/test',
                       help='테스트 데이터셋 경로 (기본값: /home/user/KJR/AD/data/image/test)')
    parser.add_argument('--batch_size', type=int, default=12,
                       help='배치 크기 (기본값: 12)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='데이터 로더 워커 수 (기본값: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (기본값: cuda)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='결과 저장 디렉토리 (기본값: 체크포인트와 같은 디렉토리)')
    
    # Continuity Loss 하이퍼파라미터
    parser.add_argument('--lambda_cont_2d', type=float, default=None,
                       help='2D continuity loss 가중치')
    parser.add_argument('--lambda_cont_1d', type=float, default=None,
                       help='1D continuity loss 가중치')
    parser.add_argument('--norm_mask_bbox_coef', type=float, default=None,
                       help='Bbox 영역의 norm mask 계수')
    parser.add_argument('--alpha', type=float, default=None,
                       help='1차 차분 가중치')
    parser.add_argument('--beta', type=float, default=None,
                       help='2차 차분 가중치')
    parser.add_argument('--curriculum_epochs', type=int, default=None,
                       help='Curriculum learning epoch 수')
    
    # 모델 하이퍼파라미터
    parser.add_argument('--num_queries', type=int, default=None,
                       help='Query 개수')
    parser.add_argument('--hidden_dim', type=int, default=None,
                       help='Hidden dimension')
    parser.add_argument('--enc_layers', type=int, default=None,
                       help='Encoder layer 개수')
    parser.add_argument('--dec_layers', type=int, default=None,
                       help='Decoder layer 개수')
    parser.add_argument('--nheads', type=int, default=None,
                       help='Attention head 개수')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout 비율')
    parser.add_argument('--dim_feedforward', type=int, default=None,
                       help='Feedforward dimension')
    
    args = parser.parse_args()
    
    # 체크포인트 파일 확인
    checkpoint_path = Path(args.resume)
    if not checkpoint_path.exists():
        print(f"❌ 오류: 체크포인트 파일을 찾을 수 없습니다: {args.resume}")
        sys.exit(1)
    
    # 체크포인트에서 하이퍼파라미터 읽기 (num_queries 등)
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'args' in checkpoint:
            checkpoint_args = checkpoint['args']
            # 체크포인트의 args로 현재 args 업데이트 (명시적으로 전달되지 않은 경우만)
            if hasattr(checkpoint_args, 'num_queries') and args.num_queries is None:
                args.num_queries = checkpoint_args.num_queries
                print(f"ℹ️  체크포인트에서 num_queries 읽음: {args.num_queries}")
            if hasattr(checkpoint_args, 'hidden_dim') and args.hidden_dim is None:
                args.hidden_dim = checkpoint_args.hidden_dim
            if hasattr(checkpoint_args, 'enc_layers') and args.enc_layers is None:
                args.enc_layers = checkpoint_args.enc_layers
            if hasattr(checkpoint_args, 'dec_layers') and args.dec_layers is None:
                args.dec_layers = checkpoint_args.dec_layers
            if hasattr(checkpoint_args, 'nheads') and args.nheads is None:
                args.nheads = checkpoint_args.nheads
            if hasattr(checkpoint_args, 'dropout') and args.dropout is None:
                args.dropout = checkpoint_args.dropout
            if hasattr(checkpoint_args, 'dim_feedforward') and args.dim_feedforward is None:
                args.dim_feedforward = checkpoint_args.dim_feedforward
        del checkpoint  # 메모리 해제
    except Exception as e:
        print(f"⚠️  체크포인트에서 하이퍼파라미터 읽기 실패: {e}")
        print("   기본값 또는 명시적으로 전달된 하이퍼파라미터를 사용합니다.")
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        output_dir = checkpoint_path.parent
    else:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 경로 확인
    coco_path = Path(args.coco_path)
    if not coco_path.exists():
        print(f"❌ 오류: 테스트 데이터셋 경로가 존재하지 않습니다: {args.coco_path}")
        sys.exit(1)
    
    # 라벨 경로 자동 찾기 및 심볼릭 링크 생성
    # 이미지 경로가 /home/user/KJR/AD/data/image/test 라면
    # 라벨 경로는 /home/user/KJR/AD/data/label/test 일 가능성이 높음
    import os
    coco_path_str = str(coco_path)
    if '/image/' in coco_path_str:
        # image/ -> label/ 변환
        label_path = Path(coco_path_str.replace('/image/', '/label/'))
        if label_path.exists():
            # YOLO가 찾는 labels 경로에 심볼릭 링크 생성
            labels_link = coco_path / "labels"  # coco_path 아래에 labels 링크 생성
            if not labels_link.exists():
                try:
                    os.symlink(label_path, labels_link)
                    print(f"ℹ️  심볼릭 링크 생성: {labels_link} -> {label_path}")
                except (OSError, FileExistsError) as e:
                    print(f"⚠️  심볼릭 링크 생성 실패: {e}")
    
    print("=" * 60)
    print("DETR 모델 테스트 시작")
    print("=" * 60)
    print(f"체크포인트: {checkpoint_path}")
    print(f"테스트 데이터셋: {coco_path}")
    print(f"배치 크기: {args.batch_size}")
    print(f"출력 디렉토리: {output_dir}")
    
    # 하이퍼파라미터 출력
    hyperparams = []
    if args.lambda_cont_2d is not None:
        hyperparams.append(f"lambda_cont_2d={args.lambda_cont_2d}")
    if args.lambda_cont_1d is not None:
        hyperparams.append(f"lambda_cont_1d={args.lambda_cont_1d}")
    if args.norm_mask_bbox_coef is not None:
        hyperparams.append(f"norm_mask_bbox_coef={args.norm_mask_bbox_coef}")
    if args.alpha is not None:
        hyperparams.append(f"alpha={args.alpha}")
    if args.beta is not None:
        hyperparams.append(f"beta={args.beta}")
    if args.curriculum_epochs is not None:
        hyperparams.append(f"curriculum_epochs={args.curriculum_epochs}")
    if args.num_queries is not None:
        hyperparams.append(f"num_queries={args.num_queries}")
    if args.hidden_dim is not None:
        hyperparams.append(f"hidden_dim={args.hidden_dim}")
    if args.enc_layers is not None:
        hyperparams.append(f"enc_layers={args.enc_layers}")
    if args.dec_layers is not None:
        hyperparams.append(f"dec_layers={args.dec_layers}")
    if args.nheads is not None:
        hyperparams.append(f"nheads={args.nheads}")
    if args.dropout is not None:
        hyperparams.append(f"dropout={args.dropout}")
    if args.dim_feedforward is not None:
        hyperparams.append(f"dim_feedforward={args.dim_feedforward}")
    
    if hyperparams:
        print(f"하이퍼파라미터: {', '.join(hyperparams)}")
    print("=" * 60)
    
    # main.py를 호출하여 테스트 실행
    cmd = [
        sys.executable, 'main.py',
        '--eval',
        '--resume', str(checkpoint_path),
        '--coco_path', str(coco_path),
        '--dataset_file', 'yolo',
        '--batch_size', str(args.batch_size),
        '--num_workers', str(args.num_workers),
        '--device', args.device,
        '--output_dir', str(output_dir),
    ]
    
    # 하이퍼파라미터 추가 (값이 있는 경우만)
    if args.lambda_cont_2d is not None:
        cmd.extend(['--lambda_cont_2d', str(args.lambda_cont_2d)])
    if args.lambda_cont_1d is not None:
        cmd.extend(['--lambda_cont_1d', str(args.lambda_cont_1d)])
    if args.norm_mask_bbox_coef is not None:
        cmd.extend(['--norm_mask_bbox_coef', str(args.norm_mask_bbox_coef)])
    if args.alpha is not None:
        cmd.extend(['--alpha', str(args.alpha)])
    if args.beta is not None:
        cmd.extend(['--beta', str(args.beta)])
    if args.curriculum_epochs is not None:
        cmd.extend(['--curriculum_epochs', str(args.curriculum_epochs)])
    if args.num_queries is not None:
        cmd.extend(['--num_queries', str(args.num_queries)])
    if args.hidden_dim is not None:
        cmd.extend(['--hidden_dim', str(args.hidden_dim)])
    if args.enc_layers is not None:
        cmd.extend(['--enc_layers', str(args.enc_layers)])
    if args.dec_layers is not None:
        cmd.extend(['--dec_layers', str(args.dec_layers)])
    if args.nheads is not None:
        cmd.extend(['--nheads', str(args.nheads)])
    if args.dropout is not None:
        cmd.extend(['--dropout', str(args.dropout)])
    if args.dim_feedforward is not None:
        cmd.extend(['--dim_feedforward', str(args.dim_feedforward)])
    
    print(f"\n실행 명령어: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 60)
        print("✅ 테스트 완료!")
        print("=" * 60)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

