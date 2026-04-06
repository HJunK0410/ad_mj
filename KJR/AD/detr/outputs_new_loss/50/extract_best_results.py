import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import numpy as np


def create_visualizations(results_df, best_10, output_dir):
    """결과 시각화 그래프 생성"""
    
    # 1. Best 10 mAP50-95 비교 막대 그래프
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Best 10 Experiments - Performance Metrics', fontsize=16, fontweight='bold')
    
    # 1-1. mAP50-95 막대 그래프
    ax1 = axes[0, 0]
    best_10_sorted = best_10.sort_values('metrics/mAP50-95(B)', ascending=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(best_10_sorted)))
    bars = ax1.barh(range(len(best_10_sorted)), best_10_sorted['metrics/mAP50-95(B)'], color=colors)
    ax1.set_yticks(range(len(best_10_sorted)))
    ax1.set_yticklabels([f.replace('detr_50_', '') for f in best_10_sorted['folder']], fontsize=8)
    ax1.set_xlabel('mAP50-95', fontsize=10)
    ax1.set_title('Top 10 mAP50-95 Scores', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    # 값 표시
    for i, (idx, row) in enumerate(best_10_sorted.iterrows()):
        ax1.text(row['metrics/mAP50-95(B)'] + 0.001, i, f"{row['metrics/mAP50-95(B)']:.4f}", 
                va='center', fontsize=8)
    
    # 1-2. mAP50 막대 그래프
    ax2 = axes[0, 1]
    bars2 = ax2.barh(range(len(best_10_sorted)), best_10_sorted['metrics/mAP50(B)'], color=colors)
    ax2.set_yticks(range(len(best_10_sorted)))
    ax2.set_yticklabels([f.replace('detr_50_', '') for f in best_10_sorted['folder']], fontsize=8)
    ax2.set_xlabel('mAP50', fontsize=10)
    ax2.set_title('Top 10 mAP50 Scores', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    ax2.set_xlim([0.98, 1.01])
    for i, (idx, row) in enumerate(best_10_sorted.iterrows()):
        ax2.text(row['metrics/mAP50(B)'] + 0.0005, i, f"{row['metrics/mAP50(B)']:.4f}", 
                va='center', fontsize=8)
    
    # 1-3. Precision vs Recall 산점도
    ax3 = axes[1, 0]
    scatter = ax3.scatter(best_10['metrics/precision(B)'], best_10['metrics/recall(B)'], 
                         s=100, c=best_10['metrics/mAP50-95(B)'], cmap='viridis', 
                         alpha=0.7, edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Precision', fontsize=10)
    ax3.set_ylabel('Recall', fontsize=10)
    ax3.set_title('Precision vs Recall (colored by mAP50-95)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='mAP50-95')
    # Best 3개 라벨
    top3 = best_10.head(3)
    for idx, row in top3.iterrows():
        ax3.annotate(row['folder'].replace('detr_50_', ''), 
                    (row['metrics/precision(B)'], row['metrics/recall(B)']),
                    fontsize=7, alpha=0.8)
    
    # 1-4. Loss 비교 (Train vs Val)
    ax4 = axes[1, 1]
    x_pos = np.arange(len(best_10_sorted))
    width = 0.35
    train_loss = best_10_sorted['train/loss_bbox'] + best_10_sorted['train/loss_ce'] + best_10_sorted['train/loss_giou']
    val_loss = best_10_sorted['val/loss_bbox'] + best_10_sorted['val/loss_ce'] + best_10_sorted['val/loss_giou']
    ax4.barh(x_pos - width/2, train_loss, width, label='Train Loss', alpha=0.7)
    ax4.barh(x_pos + width/2, val_loss, width, label='Val Loss', alpha=0.7)
    ax4.set_yticks(x_pos)
    ax4.set_yticklabels([f.replace('detr_50_', '') for f in best_10_sorted['folder']], fontsize=8)
    ax4.set_xlabel('Total Loss', fontsize=10)
    ax4.set_title('Train vs Validation Loss', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'best_10_performance_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 전체 실험 분포 히스토그램
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('All Experiments Distribution', fontsize=16, fontweight='bold')
    
    # 2-1. mAP50-95 분포
    ax1 = axes[0, 0]
    ax1.hist(results_df['metrics/mAP50-95(B)'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(results_df['metrics/mAP50-95(B)'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["metrics/mAP50-95(B)"].mean():.4f}')
    ax1.axvline(results_df['metrics/mAP50-95(B)'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {results_df["metrics/mAP50-95(B)"].median():.4f}')
    ax1.set_xlabel('mAP50-95', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('mAP50-95 Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2-2. mAP50 분포
    ax2 = axes[0, 1]
    ax2.hist(results_df['metrics/mAP50(B)'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    ax2.axvline(results_df['metrics/mAP50(B)'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {results_df["metrics/mAP50(B)"].mean():.4f}')
    ax2.axvline(results_df['metrics/mAP50(B)'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {results_df["metrics/mAP50(B)"].median():.4f}')
    ax2.set_xlabel('mAP50', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('mAP50 Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 2-3. Precision vs Recall (전체)
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['metrics/precision(B)'], results_df['metrics/recall(B)'], 
                         s=50, c=results_df['metrics/mAP50-95(B)'], cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    # Best 10 하이라이트
    top10_indices = best_10.index
    ax3.scatter(best_10['metrics/precision(B)'], best_10['metrics/recall(B)'], 
               s=150, c='red', marker='*', edgecolors='black', linewidth=1.5, 
               label='Top 10', zorder=5)
    ax3.set_xlabel('Precision', fontsize=10)
    ax3.set_ylabel('Recall', fontsize=10)
    ax3.set_title('Precision vs Recall (All Experiments)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='mAP50-95')
    
    # 2-4. Lambda 파라미터 분석 (2d, 1d 추출)
    ax4 = axes[1, 1]
    # 폴더명에서 lambda 값 추출
    lambda_2d = []
    lambda_1d = []
    map_scores = []
    for idx, row in results_df.iterrows():
        folder = row['folder']
        # detr_50_2dX_1dY 형식
        parts = folder.replace('detr_50_', '').split('_')
        if len(parts) >= 2:
            try:
                lambda_2d_val = float(parts[0].replace('2d', '').replace('000', '0'))
                lambda_1d_val = float(parts[1].replace('1d', '').replace('000', '0'))
                lambda_2d.append(lambda_2d_val)
                lambda_1d.append(lambda_1d_val)
                map_scores.append(row['metrics/mAP50-95(B)'])
            except:
                pass
    
    if lambda_2d and lambda_1d:
        scatter = ax4.scatter(lambda_2d, lambda_1d, s=100, c=map_scores, cmap='viridis', 
                             alpha=0.7, edgecolors='black', linewidth=1)
        ax4.set_xlabel('Lambda 2D', fontsize=10)
        ax4.set_ylabel('Lambda 1D', fontsize=10)
        ax4.set_title('Lambda Parameters vs mAP50-95', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='mAP50-95')
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'all_experiments_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 상관관계 히트맵
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_cols = ['metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/precision(B)', 
                       'metrics/recall(B)', 'train/loss_bbox', 'train/loss_ce', 'train/loss_giou',
                       'val/loss_bbox', 'val/loss_ce', 'val/loss_giou']
    available_corr_cols = [col for col in correlation_cols if col in results_df.columns]
    corr_matrix = results_df[available_corr_cols].corr()
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(available_corr_cols)))
    ax.set_yticks(range(len(available_corr_cols)))
    ax.set_xticklabels([col.replace('metrics/', '').replace('train/', '').replace('val/', '') 
                        for col in available_corr_cols], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([col.replace('metrics/', '').replace('train/', '').replace('val/', '') 
                        for col in available_corr_cols], fontsize=9)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    # 상관계수 값 표시
    for i in range(len(available_corr_cols)):
        for j in range(len(available_corr_cols)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - best_10_performance_metrics.png")
    print(f"  - all_experiments_distribution.png")
    print(f"  - correlation_matrix.png")


# 결과 디렉토리
base_dir = "/home/user/KJR/AD/detr/outputs_new_loss/50"

# 모든 results.csv 파일 찾기
results_files = glob.glob(os.path.join(base_dir, "detr_50_*/results.csv"))

all_results = []

print(f"총 {len(results_files)}개의 실험 결과 파일을 찾았습니다.")
print("결과를 수집하는 중...")

for file_path in results_files:
    try:
        df = pd.read_csv(file_path)
        if len(df) > 0:
            # Best epoch 찾기 (mAP50-95가 최대인 epoch)
            if 'metrics/mAP50-95(B)' in df.columns:
                best_idx = df['metrics/mAP50-95(B)'].idxmax()
                best_row = df.iloc[best_idx].copy()
                
                # 폴더명 추출
                folder_name = os.path.basename(os.path.dirname(file_path))
                best_row['folder'] = folder_name
                
                all_results.append(best_row)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# DataFrame으로 변환
if all_results:
    results_df = pd.DataFrame(all_results)
    
    # map50-95 기준으로 정렬 (내림차순)
    if 'metrics/mAP50-95(B)' in results_df.columns:
        results_df = results_df.sort_values('metrics/mAP50-95(B)', ascending=False)
        
        # Best 10개 선택
        best_10 = results_df.head(10).copy()
        
        # 중요한 컬럼만 선택
        important_cols = ['folder', 'epoch', 'metrics/mAP50-95(B)', 'metrics/mAP50(B)', 
                         'metrics/precision(B)', 'metrics/recall(B)',
                         'train/loss_bbox', 'train/loss_ce', 'train/loss_giou',
                         'val/loss_bbox', 'val/loss_ce', 'val/loss_giou']
        
        available_cols = [col for col in important_cols if col in best_10.columns]
        best_10_summary = best_10[available_cols]
        
        # 결과 저장
        output_file = os.path.join(base_dir, "best_10_results.csv")
        best_10_summary.to_csv(output_file, index=False)
        
        print("\n" + "=" * 100)
        print("Best 10 Results (sorted by mAP50-95)")
        print("=" * 100)
        print(best_10_summary.to_string(index=False))
        print(f"\n결과가 저장되었습니다: {output_file}")
        
        # 통계 정보
        print("\n" + "=" * 100)
        print("통계 정보")
        print("=" * 100)
        print(f"총 실험 수: {len(results_df)}")
        print(f"Best mAP50-95: {results_df['metrics/mAP50-95(B)'].max():.6f}")
        print(f"Average mAP50-95: {results_df['metrics/mAP50-95(B)'].mean():.6f}")
        print(f"Median mAP50-95: {results_df['metrics/mAP50-95(B)'].median():.6f}")
        print(f"Std mAP50-95: {results_df['metrics/mAP50-95(B)'].std():.6f}")
        
        # 전체 결과도 저장
        full_output_file = os.path.join(base_dir, "all_results_sorted.csv")
        results_df.to_csv(full_output_file, index=False)
        print(f"\n전체 결과 (정렬됨)가 저장되었습니다: {full_output_file}")
        
        # 그래프 생성
        print("\n그래프를 생성하는 중...")
        create_visualizations(results_df, best_10, base_dir)
        print("그래프 생성 완료!")
    else:
        print("mAP50-95 컬럼을 찾을 수 없습니다.")
        print("사용 가능한 컬럼:", results_df.columns.tolist())
else:
    print("결과를 찾을 수 없습니다.")

