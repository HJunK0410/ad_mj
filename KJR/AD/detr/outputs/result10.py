from pathlib import Path
import csv

base_dir = Path("/home/user/KJR/AD/detr/outputs/new_sjpm")
csv_files = sorted(base_dir.rglob("*.csv"))

# 컬럼명 후보 (환경마다 이름이 조금 다를 수 있어서)
candidates = [
    "metrics/mAP50-95(B)",
    "map50-95",
    "mAP50-95",
    "metrics/mAP50-95",
]

results = []

for csv_path in csv_files:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        continue

    # 타겟 컬럼 찾기
    target_col = next((c for c in candidates if c in rows[0]), None)
    if target_col is None:
        # 후보에 없으면 컬럼명에 map50-95 포함된 것 탐색
        target_col = next((k for k in rows[0].keys() if "map50-95" in k.lower()), None)

    if target_col is None:
        print(f"[스킵] {csv_path}: mAP50-95 컬럼 없음")
        continue

    # 숫자로 변환 가능한 행만 대상으로 최고값 선택
    valid_rows = []
    for r in rows:
        try:
            score = float(r[target_col])
            valid_rows.append((score, r))
        except (ValueError, TypeError):
            pass

    if not valid_rows:
        print(f"[스킵] {csv_path}: 유효한 수치 데이터 없음")
        continue

    best_score, best_row = max(valid_rows, key=lambda x: x[0])

    results.append({
        "file": str(csv_path.relative_to(base_dir)),
        "epoch": best_row.get("epoch", ""),
        "map50_95": best_score,
        "map50": best_row.get("metrics/mAP50(B)", best_row.get("map50", "")),
        "precision": best_row.get("metrics/precision(B)", ""),
        "recall": best_row.get("metrics/recall(B)", ""),
    })

# mAP50-95 기준 상위 10개만 출력
results.sort(key=lambda x: x["map50_95"], reverse=True)
top_n = 10
top_results = results[:top_n]

print(f"총 {len(results)}개 파일 중 상위 {len(top_results)}개")
print("-" * 120)
print(f"{'file':55} {'epoch':>6} {'mAP50-95':>10} {'mAP50':>10} {'precision':>10} {'recall':>10}")
print("-" * 120)
for r in top_results:
    print(
        f"{r['file'][:55]:55} "
        f"{str(r['epoch']):>6} "
        f"{r['map50_95']:>10.6f} "
        f"{str(r['map50']):>10} "
        f"{str(r['precision']):>10} "
        f"{str(r['recall']):>10}"
    )