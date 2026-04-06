import pandas as pd
import os
from pathlib import Path

base_dir = Path("/home/user/KJR/AD/detr/outputs/detr")
target_folder = "last_2d000_1d000_nm100"

all_results = []

# 모든 폴더의 results.csv 파일 읽기
for folder in sorted(base_dir.iterdir()):
    if not folder.is_dir() or folder.name in ["best.csv", "best_no1.csv"]:
        continue
    
    results_file = folder / "results.csv"
    if not results_file.exists():
        continue
    
    try:
        df = pd.read_csv(results_file)
        
        # 컬럼 이름 매핑
        col_mapping = {}
        for col in df.columns:
            if "mAP50(B)" in col and "mAP50-95" not in col:
                col_mapping['mAP50'] = col
            if "mAP50-95(B)" in col:
                col_mapping['mAP50-95'] = col
            if "precision(B)" in col:
                col_mapping['precision'] = col
            if "recall(B)" in col:
                col_mapping['recall'] = col
            if "val/loss_bbox" in col:
                col_mapping['val_loss_bbox'] = col
            if "val/loss_ce" in col:
                col_mapping['val_loss_ce'] = col
            if "val/loss_giou" in col:
                col_mapping['val_loss_giou'] = col
        
        # mAP50-95 기준으로 최고 epoch 찾기
        best_row = {}
        best_row['experiment'] = folder.name
        
        if 'mAP50-95' in col_mapping:
            # mAP50-95가 최고인 epoch 찾기
            best_idx = df[col_mapping['mAP50-95']].idxmax()
            best_row['best_epoch'] = int(df.loc[best_idx, 'epoch']) if 'epoch' in df.columns else best_idx + 1
            
            # 해당 epoch의 모든 metric 저장
            best_row['best_mAP50'] = df.loc[best_idx, col_mapping['mAP50']] if 'mAP50' in col_mapping else None
            best_row['best_mAP50-95'] = df.loc[best_idx, col_mapping['mAP50-95']]
            best_row['best_precision'] = df.loc[best_idx, col_mapping['precision']] if 'precision' in col_mapping else None
            best_row['best_recall'] = df.loc[best_idx, col_mapping['recall']] if 'recall' in col_mapping else None
            best_row['val_loss_bbox'] = df.loc[best_idx, col_mapping['val_loss_bbox']] if 'val_loss_bbox' in col_mapping else None
            best_row['val_loss_ce'] = df.loc[best_idx, col_mapping['val_loss_ce']] if 'val_loss_ce' in col_mapping else None
            best_row['val_loss_giou'] = df.loc[best_idx, col_mapping['val_loss_giou']] if 'val_loss_giou' in col_mapping else None
        else:
            continue
        
        all_results.append(best_row)
        
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        continue

# DataFrame으로 변환
df_best = pd.DataFrame(all_results)

# mAP50-95 기준으로 정렬
df_best = df_best.sort_values('best_mAP50-95', ascending=False, na_position='last')

# 1. best.csv: mAP50-95 기준 top 10 + last_2d000_1d000_nm100의 모든 metric
top10 = df_best.head(10).copy()

# last_2d000_1d000_nm100의 mAP50-95 기준 최고값 1개 가져오기
target_folder_path = base_dir / target_folder
target_results_file = target_folder_path / "results.csv"
target_best_result = None

if target_results_file.exists():
    try:
        df_target = pd.read_csv(target_results_file)
        
        # 컬럼 이름 매핑
        col_mapping = {}
        for col in df_target.columns:
            if "mAP50(B)" in col and "mAP50-95" not in col:
                col_mapping['mAP50'] = col
            if "mAP50-95(B)" in col:
                col_mapping['mAP50-95'] = col
            if "precision(B)" in col:
                col_mapping['precision'] = col
            if "recall(B)" in col:
                col_mapping['recall'] = col
            if "val/loss_bbox" in col:
                col_mapping['val_loss_bbox'] = col
            if "val/loss_ce" in col:
                col_mapping['val_loss_ce'] = col
            if "val/loss_giou" in col:
                col_mapping['val_loss_giou'] = col
        
        # mAP50-95 기준 최고값 1개 찾기
        if 'mAP50-95' in col_mapping:
            best_idx = df_target[col_mapping['mAP50-95']].idxmax()
            target_best_result = {}
            target_best_result['experiment'] = target_folder
            target_best_result['best_epoch'] = int(df_target.loc[best_idx, 'epoch']) if 'epoch' in df_target.columns else best_idx + 1
            target_best_result['best_mAP50'] = df_target.loc[best_idx, col_mapping['mAP50']] if 'mAP50' in col_mapping else None
            target_best_result['best_mAP50-95'] = df_target.loc[best_idx, col_mapping['mAP50-95']]
            target_best_result['best_precision'] = df_target.loc[best_idx, col_mapping['precision']] if 'precision' in col_mapping else None
            target_best_result['best_recall'] = df_target.loc[best_idx, col_mapping['recall']] if 'recall' in col_mapping else None
            target_best_result['val_loss_bbox'] = df_target.loc[best_idx, col_mapping['val_loss_bbox']] if 'val_loss_bbox' in col_mapping else None
            target_best_result['val_loss_ce'] = df_target.loc[best_idx, col_mapping['val_loss_ce']] if 'val_loss_ce' in col_mapping else None
            target_best_result['val_loss_giou'] = df_target.loc[best_idx, col_mapping['val_loss_giou']] if 'val_loss_giou' in col_mapping else None
    except Exception as e:
        print(f"Error reading {target_results_file}: {e}")

# best.csv 생성: top 10 + target folder의 mAP50-95 기준 최고값 1개
df_best_csv_list = [top10]
if target_best_result is not None:
    df_target_best = pd.DataFrame([target_best_result])
    df_best_csv_list.append(df_target_best)

df_best_csv = pd.concat(df_best_csv_list).drop_duplicates(subset=['experiment'], keep='first')
df_best_csv = df_best_csv.sort_values('best_mAP50-95', ascending=False, na_position='last')

output_file = base_dir / "best.csv"
df_best_csv.to_csv(output_file, index=False)
print(f"best.csv에 {len(df_best_csv)}개의 결과를 저장했습니다 (Top 10 + {target_folder}의 mAP50-95 기준 최고값 1개).")

# 2. best_no1.csv: precision 1 제외한 mAP50-95 기준 top 10 + last_2d000_1d000_nm100의 precision 1 제외한 mAP50-95 기준 최고값 1개
df_no_precision1 = df_best[df_best['best_precision'] != 1.0].copy()
top10_no_precision1 = df_no_precision1.head(10).copy()

# target folder의 precision 1 제외한 mAP50-95 기준 최고값 1개
target_best_no_precision1 = None
if target_results_file.exists() and target_best_result is not None:
    if target_best_result['best_precision'] != 1.0:
        target_best_no_precision1 = target_best_result
    else:
        # precision 1이면 precision 1이 아닌 것 중 최고값 찾기
        try:
            df_target = pd.read_csv(target_results_file)
            col_mapping = {}
            for col in df_target.columns:
                if "mAP50(B)" in col and "mAP50-95" not in col:
                    col_mapping['mAP50'] = col
                if "mAP50-95(B)" in col:
                    col_mapping['mAP50-95'] = col
                if "precision(B)" in col:
                    col_mapping['precision'] = col
                if "recall(B)" in col:
                    col_mapping['recall'] = col
                if "val/loss_bbox" in col:
                    col_mapping['val_loss_bbox'] = col
                if "val/loss_ce" in col:
                    col_mapping['val_loss_ce'] = col
                if "val/loss_giou" in col:
                    col_mapping['val_loss_giou'] = col
            
            # precision 1이 아닌 것만 필터링
            df_target_no_prec1 = df_target[df_target[col_mapping['precision']] != 1.0].copy()
            if len(df_target_no_prec1) > 0:
                best_idx = df_target_no_prec1[col_mapping['mAP50-95']].idxmax()
                target_best_no_precision1 = {}
                target_best_no_precision1['experiment'] = target_folder
                target_best_no_precision1['best_epoch'] = int(df_target_no_prec1.loc[best_idx, 'epoch']) if 'epoch' in df_target_no_prec1.columns else best_idx + 1
                target_best_no_precision1['best_mAP50'] = df_target_no_prec1.loc[best_idx, col_mapping['mAP50']] if 'mAP50' in col_mapping else None
                target_best_no_precision1['best_mAP50-95'] = df_target_no_prec1.loc[best_idx, col_mapping['mAP50-95']]
                target_best_no_precision1['best_precision'] = df_target_no_prec1.loc[best_idx, col_mapping['precision']] if 'precision' in col_mapping else None
                target_best_no_precision1['best_recall'] = df_target_no_prec1.loc[best_idx, col_mapping['recall']] if 'recall' in col_mapping else None
                target_best_no_precision1['val_loss_bbox'] = df_target_no_prec1.loc[best_idx, col_mapping['val_loss_bbox']] if 'val_loss_bbox' in col_mapping else None
                target_best_no_precision1['val_loss_ce'] = df_target_no_prec1.loc[best_idx, col_mapping['val_loss_ce']] if 'val_loss_ce' in col_mapping else None
                target_best_no_precision1['val_loss_giou'] = df_target_no_prec1.loc[best_idx, col_mapping['val_loss_giou']] if 'val_loss_giou' in col_mapping else None
        except Exception as e:
            print(f"Error finding precision 1 excluded best for {target_folder}: {e}")

# best_no1.csv 생성
df_best_no1_csv_list = [top10_no_precision1]
if target_best_no_precision1 is not None:
    df_target_best_no_prec1 = pd.DataFrame([target_best_no_precision1])
    df_best_no1_csv_list.append(df_target_best_no_prec1)

df_best_no1_csv = pd.concat(df_best_no1_csv_list).drop_duplicates(subset=['experiment'], keep='first')
df_best_no1_csv = df_best_no1_csv.sort_values('best_mAP50-95', ascending=False, na_position='last')

output_file_no1 = base_dir / "best_no1.csv"
df_best_no1_csv.to_csv(output_file_no1, index=False)
print(f"best_no1.csv에 {len(df_best_no1_csv)}개의 결과를 저장했습니다 (Precision 1 제외 Top 10 + {target_folder}의 Precision 1 제외 mAP50-95 기준 최고값 1개).")

# 3. best_no1_no1.csv: precision 1과 recall 1 모두 제외한 조건에서 각 실험의 mAP50-95 최고값 찾기
all_results_no_precision1_recall1 = []

# 모든 폴더의 results.csv 파일을 다시 읽어서 precision 1과 recall 1 모두 제외한 조건에서 최고값 찾기
for folder in sorted(base_dir.iterdir()):
    if not folder.is_dir() or folder.name in ["best.csv", "best_no1.csv", "best_no1_no1.csv"]:
        continue
    
    results_file = folder / "results.csv"
    if not results_file.exists():
        continue
    
    try:
        df = pd.read_csv(results_file)
        
        # 컬럼 이름 매핑
        col_mapping = {}
        for col in df.columns:
            if "mAP50(B)" in col and "mAP50-95" not in col:
                col_mapping['mAP50'] = col
            if "mAP50-95(B)" in col:
                col_mapping['mAP50-95'] = col
            if "precision(B)" in col:
                col_mapping['precision'] = col
            if "recall(B)" in col:
                col_mapping['recall'] = col
            if "val/loss_bbox" in col:
                col_mapping['val_loss_bbox'] = col
            if "val/loss_ce" in col:
                col_mapping['val_loss_ce'] = col
            if "val/loss_giou" in col:
                col_mapping['val_loss_giou'] = col
        
        # precision 1과 recall 1 모두 제외한 데이터에서 mAP50-95 최고값 찾기
        if 'mAP50-95' in col_mapping and 'precision' in col_mapping and 'recall' in col_mapping:
            df_filtered = df[(df[col_mapping['precision']] != 1.0) & (df[col_mapping['recall']] != 1.0)].copy()
            
            if len(df_filtered) > 0:
                best_idx = df_filtered[col_mapping['mAP50-95']].idxmax()
                best_row = {}
                best_row['experiment'] = folder.name
                best_row['best_epoch'] = int(df_filtered.loc[best_idx, 'epoch']) if 'epoch' in df_filtered.columns else best_idx + 1
                best_row['best_mAP50'] = df_filtered.loc[best_idx, col_mapping['mAP50']] if 'mAP50' in col_mapping else None
                best_row['best_mAP50-95'] = df_filtered.loc[best_idx, col_mapping['mAP50-95']]
                best_row['best_precision'] = df_filtered.loc[best_idx, col_mapping['precision']]
                best_row['best_recall'] = df_filtered.loc[best_idx, col_mapping['recall']]
                best_row['val_loss_bbox'] = df_filtered.loc[best_idx, col_mapping['val_loss_bbox']] if 'val_loss_bbox' in col_mapping else None
                best_row['val_loss_ce'] = df_filtered.loc[best_idx, col_mapping['val_loss_ce']] if 'val_loss_ce' in col_mapping else None
                best_row['val_loss_giou'] = df_filtered.loc[best_idx, col_mapping['val_loss_giou']] if 'val_loss_giou' in col_mapping else None
                all_results_no_precision1_recall1.append(best_row)
    except Exception as e:
        print(f"Error reading {results_file} for best_no1_no1: {e}")
        continue

# DataFrame으로 변환
df_no_precision1_recall1 = pd.DataFrame(all_results_no_precision1_recall1)
df_no_precision1_recall1 = df_no_precision1_recall1.sort_values('best_mAP50-95', ascending=False, na_position='last')

# Top 10만 선택
top10_no_precision1_recall1 = df_no_precision1_recall1.head(10).copy()

# target folder의 precision 1과 recall 1 모두 제외한 mAP50-95 기준 최고값 1개
target_best_no_precision1_recall1 = None
if target_results_file.exists():
    try:
        df_target = pd.read_csv(target_results_file)
        col_mapping = {}
        for col in df_target.columns:
            if "mAP50(B)" in col and "mAP50-95" not in col:
                col_mapping['mAP50'] = col
            if "mAP50-95(B)" in col:
                col_mapping['mAP50-95'] = col
            if "precision(B)" in col:
                col_mapping['precision'] = col
            if "recall(B)" in col:
                col_mapping['recall'] = col
            if "val/loss_bbox" in col:
                col_mapping['val_loss_bbox'] = col
            if "val/loss_ce" in col:
                col_mapping['val_loss_ce'] = col
            if "val/loss_giou" in col:
                col_mapping['val_loss_giou'] = col
        
        # precision 1과 recall 1 모두 아닌 것만 필터링
        df_target_no_prec1_recall1 = df_target[(df_target[col_mapping['precision']] != 1.0) & (df_target[col_mapping['recall']] != 1.0)].copy()
        if len(df_target_no_prec1_recall1) > 0:
            best_idx = df_target_no_prec1_recall1[col_mapping['mAP50-95']].idxmax()
            target_best_no_precision1_recall1 = {}
            target_best_no_precision1_recall1['experiment'] = target_folder
            target_best_no_precision1_recall1['best_epoch'] = int(df_target_no_prec1_recall1.loc[best_idx, 'epoch']) if 'epoch' in df_target_no_prec1_recall1.columns else best_idx + 1
            target_best_no_precision1_recall1['best_mAP50'] = df_target_no_prec1_recall1.loc[best_idx, col_mapping['mAP50']] if 'mAP50' in col_mapping else None
            target_best_no_precision1_recall1['best_mAP50-95'] = df_target_no_prec1_recall1.loc[best_idx, col_mapping['mAP50-95']]
            target_best_no_precision1_recall1['best_precision'] = df_target_no_prec1_recall1.loc[best_idx, col_mapping['precision']] if 'precision' in col_mapping else None
            target_best_no_precision1_recall1['best_recall'] = df_target_no_prec1_recall1.loc[best_idx, col_mapping['recall']] if 'recall' in col_mapping else None
            target_best_no_precision1_recall1['val_loss_bbox'] = df_target_no_prec1_recall1.loc[best_idx, col_mapping['val_loss_bbox']] if 'val_loss_bbox' in col_mapping else None
            target_best_no_precision1_recall1['val_loss_ce'] = df_target_no_prec1_recall1.loc[best_idx, col_mapping['val_loss_ce']] if 'val_loss_ce' in col_mapping else None
            target_best_no_precision1_recall1['val_loss_giou'] = df_target_no_prec1_recall1.loc[best_idx, col_mapping['val_loss_giou']] if 'val_loss_giou' in col_mapping else None
    except Exception as e:
        print(f"Error finding precision 1 and recall 1 excluded best for {target_folder}: {e}")

# best_no1_no1.csv 생성: top 10 + target folder의 최고값 1개
df_best_no1_no1_csv_list = [top10_no_precision1_recall1]
if target_best_no_precision1_recall1 is not None:
    df_target_best_no_prec1_recall1 = pd.DataFrame([target_best_no_precision1_recall1])
    df_best_no1_no1_csv_list.append(df_target_best_no_prec1_recall1)

df_best_no1_no1_csv = pd.concat(df_best_no1_no1_csv_list).drop_duplicates(subset=['experiment'], keep='first')
df_best_no1_no1_csv = df_best_no1_no1_csv.sort_values('best_mAP50-95', ascending=False, na_position='last')

output_file_no1_no1 = base_dir / "best_no1_no1.csv"
df_best_no1_no1_csv.to_csv(output_file_no1_no1, index=False)
print(f"best_no1_no1.csv에 {len(df_best_no1_no1_csv)}개의 결과를 저장했습니다 (Precision 1과 Recall 1 모두 제외 Top 10 + {target_folder}의 Precision 1과 Recall 1 모두 제외 mAP50-95 기준 최고값 1개).")

print(f"\nbest.csv Top 5:")
print(df_best_csv[['experiment', 'best_epoch', 'best_mAP50', 'best_mAP50-95', 'best_precision', 'best_recall']].head().to_string(index=False))
print(f"\nbest_no1.csv Top 5:")
print(df_best_no1_csv[['experiment', 'best_epoch', 'best_mAP50', 'best_mAP50-95', 'best_precision', 'best_recall']].head().to_string(index=False))
print(f"\nbest_no1_no1.csv Top 5:")
print(df_best_no1_no1_csv[['experiment', 'best_epoch', 'best_mAP50', 'best_mAP50-95', 'best_precision', 'best_recall']].head().to_string(index=False))
