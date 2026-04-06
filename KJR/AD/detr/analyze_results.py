import pandas as pd
import os
from pathlib import Path

base_dir = Path("/home/user/KJR/AD/detr/outputs/detr")
target_folder = "last_2d000_1d000_nm100"

results = []

# 모든 폴더의 results.csv 파일 읽기
for folder in sorted(base_dir.iterdir()):
    if not folder.is_dir():
        continue
    
    results_file = folder / "results.csv"
    if not results_file.exists():
        continue
    
    try:
        df = pd.read_csv(results_file)
        
        # map50-95 컬럼 찾기
        map50_95_col = None
        precision_col = None
        epoch_col = None
        
        for col in df.columns:
            if "mAP50-95" in col or "map50-95" in col.lower():
                map50_95_col = col
            if "precision" in col.lower() and "(B)" in col:
                precision_col = col
            if col == "epoch":
                epoch_col = col
        
        if map50_95_col is None:
            continue
        
        # 각 epoch의 map50-95 값 확인
        for idx, row in df.iterrows():
            map50_95 = row[map50_95_col]
            precision = row[precision_col] if precision_col else None
            epoch = row[epoch_col] if epoch_col else idx + 1
            
            if pd.notna(map50_95):
                results.append({
                    'folder': folder.name,
                    'epoch': epoch,
                    'map50_95': float(map50_95),
                    'precision': float(precision) if precision is not None and pd.notna(precision) else None
                })
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        continue

# DataFrame으로 변환
df_results = pd.DataFrame(results)

if len(df_results) == 0:
    print("No results found!")
    exit(1)

# 1. 전체 map50-95 기준 best (precision 1 포함)
print("=" * 80)
print("전체 map50-95 기준 BEST (precision 1 포함)")
print("=" * 80)
best_all = df_results.loc[df_results['map50_95'].idxmax()]
print(f"폴더: {best_all['folder']}")
print(f"Epoch: {best_all['epoch']}")
print(f"mAP50-95: {best_all['map50_95']:.6f}")
if best_all['precision'] is not None:
    print(f"Precision: {best_all['precision']:.6f}")
print()

# 2. precision 1 제외한 map50-95 기준 best
print("=" * 80)
print("Precision 1 제외한 map50-95 기준 BEST")
print("=" * 80)
df_no_precision1 = df_results[df_results['precision'] != 1.0]
if len(df_no_precision1) > 0:
    best_no_precision1 = df_no_precision1.loc[df_no_precision1['map50_95'].idxmax()]
    print(f"폴더: {best_no_precision1['folder']}")
    print(f"Epoch: {best_no_precision1['epoch']}")
    print(f"mAP50-95: {best_no_precision1['map50_95']:.6f}")
    if best_no_precision1['precision'] is not None:
        print(f"Precision: {best_no_precision1['precision']:.6f}")
else:
    print("Precision 1이 아닌 결과가 없습니다.")
print()

# 3. last_2d000_1d000_nm100 폴더 결과 (무조건 표시)
print("=" * 80)
print(f"폴더: {target_folder} (무조건 표시)")
print("=" * 80)
target_results = df_results[df_results['folder'] == target_folder]
if len(target_results) > 0:
    # 최고 map50-95 값
    best_target = target_results.loc[target_results['map50_95'].idxmax()]
    print(f"최고 mAP50-95:")
    print(f"  Epoch: {best_target['epoch']}")
    print(f"  mAP50-95: {best_target['map50_95']:.6f}")
    if best_target['precision'] is not None:
        print(f"  Precision: {best_target['precision']:.6f}")
    print()
    
    # precision 1 제외한 최고 map50-95
    target_no_precision1 = target_results[target_results['precision'] != 1.0]
    if len(target_no_precision1) > 0:
        best_target_no_precision1 = target_no_precision1.loc[target_no_precision1['map50_95'].idxmax()]
        print(f"Precision 1 제외 최고 mAP50-95:")
        print(f"  Epoch: {best_target_no_precision1['epoch']}")
        print(f"  mAP50-95: {best_target_no_precision1['map50_95']:.6f}")
        if best_target_no_precision1['precision'] is not None:
            print(f"  Precision: {best_target_no_precision1['precision']:.6f}")
    else:
        print("Precision 1이 아닌 결과가 없습니다.")
else:
    print(f"{target_folder} 폴더의 결과를 찾을 수 없습니다.")
print()

# 4. Top 10 전체 (precision 1 포함)
print("=" * 80)
print("전체 Top 10 mAP50-95 (precision 1 포함)")
print("=" * 80)
top10_all = df_results.nlargest(10, 'map50_95')[['folder', 'epoch', 'map50_95', 'precision']]
for idx, row in top10_all.iterrows():
    precision_str = f"{row['precision']:.6f}" if row['precision'] is not None else "N/A"
    epoch_val = int(row['epoch']) if pd.notna(row['epoch']) else "N/A"
    print(f"{row['folder']:40s} Epoch {epoch_val:>4}  mAP50-95: {row['map50_95']:.6f}  Precision: {precision_str}")
print()

# 5. Top 10 precision 1 제외
print("=" * 80)
print("Precision 1 제외 Top 10 mAP50-95")
print("=" * 80)
if len(df_no_precision1) > 0:
    top10_no_precision1 = df_no_precision1.nlargest(10, 'map50_95')[['folder', 'epoch', 'map50_95', 'precision']]
    for idx, row in top10_no_precision1.iterrows():
        precision_str = f"{row['precision']:.6f}" if row['precision'] is not None else "N/A"
        epoch_val = int(row['epoch']) if pd.notna(row['epoch']) else "N/A"
        print(f"{row['folder']:40s} Epoch {epoch_val:>4}  mAP50-95: {row['map50_95']:.6f}  Precision: {precision_str}")
else:
    print("Precision 1이 아닌 결과가 없습니다.")

