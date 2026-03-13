#!/usr/bin/env python3
"""
v8_diff, v10_diff, v11_diff Diff Loss 실험 결과를 종합해
버전별 validation 상위 3개 조합을 test dataset으로 평가하는 스크립트.

- 입력: 각 버전의 runs/detect/grid_search_diff/grid_search_results.csv
- 선별: status=success 중 metric(기본 fitness) 기준 상위 3개
- 평가: 버전별 ultralytics 환경에서 subprocess로 model.val() 실행
- 출력: 루트의 diff_test_eval_results.csv (stage2_test_eval_results.csv와 동일 컬럼 형식)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff Loss top-3 x 3버전 test 평가")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v8_diff", "v10_diff", "v11_diff"],
        help="평가할 diff 버전 디렉토리 목록 (기본: v8_diff v10_diff v11_diff)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="버전별 validation 상위 개수 (기본: 3)",
    )
    parser.add_argument(
        "--rank-metric",
        choices=["fitness", "mAP50", "mAP50-95", "precision", "recall"],
        default="fitness",
        help="validation 순위 기준 컬럼 (기본: fitness)",
    )
    parser.add_argument(
        "--data-yaml",
        default="data/data_test.yaml",
        help="버전 디렉토리 기준 데이터 YAML 경로 (기본: data/data_test.yaml)",
    )
    parser.add_argument(
        "--split",
        choices=["auto", "test", "val"],
        default="auto",
        help="평가 split (auto면 test 우선, 없으면 val 사용)",
    )
    parser.add_argument("--device", default="0", help="평가 device (기본: 0)")
    parser.add_argument("--imgsz", type=int, default=1280, help="평가 이미지 크기 (기본: 1280)")
    parser.add_argument("--batch", type=int, default=4, help="평가 배치 크기 (기본: 4)")
    parser.add_argument(
        "--output-dir",
        default="runs/diff_test_eval",
        help="버전 내부 평가 결과 저장 디렉토리 (기본: runs/diff_test_eval)",
    )
    parser.add_argument(
        "--output-csv",
        default="diff_test_eval_results.csv",
        help="루트 기준 출력 CSV 파일명 (기본: diff_test_eval_results.csv)",
    )
    return parser.parse_args()


def has_key_in_yaml(yaml_path: Path, key: str) -> bool:
    if not yaml_path.exists():
        return False
    target = f"{key}:"
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(target):
            return True
    return False


def resolve_split(yaml_path: Path, split_arg: str) -> str:
    if split_arg in ("test", "val"):
        return split_arg
    if has_key_in_yaml(yaml_path, "test"):
        return "test"
    if has_key_in_yaml(yaml_path, "val"):
        return "val"
    raise ValueError(f"data yaml에 test/val 키가 없습니다: {yaml_path}")


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def read_top_diff_rows(csv_path: Path, top_k: int, rank_metric: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success":
                rows.append(row)

    # 높은 metric 우선, 동점이면 iteration 오름차순
    rows.sort(
        key=lambda r: (-to_float(r.get(rank_metric, "")), int(r.get("iteration", "999999"))),
    )
    return rows[:top_k]


def resolve_weight_path(version_dir: Path, save_dir_value: str) -> Tuple[Path, str]:
    save_value = (save_dir_value or "").strip()
    candidates: List[Path] = []
    if save_value:
        raw_path = Path(save_value)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(version_dir / raw_path)
            candidates.append(version_dir / "runs" / "detect" / raw_path)
            if save_value.startswith("runs/"):
                candidates.append(version_dir / save_value)

    # 기본 fallback: save_dir가 run name일 때
    if save_value:
        candidates.append(version_dir / "runs" / "detect" / save_value)

    checked = set()
    for run_dir in candidates:
        run_dir = run_dir.resolve()
        if run_dir in checked:
            continue
        checked.add(run_dir)
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            return best_pt, "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        if last_pt.exists():
            return last_pt, "last.pt"

    # 마지막 후보 기준 missing 경로 반환
    fallback = (version_dir / "runs" / "detect" / save_value / "weights" / "best.pt").resolve()
    return fallback, "missing"


def run_eval_in_subprocess(
    version_dir: Path,
    model_path: Path,
    data_yaml: Path,
    split: str,
    device: str,
    imgsz: int,
    batch: int,
    output_dir: str,
    run_name: str,
) -> Dict[str, float]:
    code = r"""
import json
import os
from ultralytics import YOLO

model_path = os.environ["MODEL_PATH"]
data_yaml = os.environ["DATA_YAML"]
split = os.environ["SPLIT"]
device = os.environ["DEVICE"]
imgsz = int(os.environ["IMGSZ"])
batch = int(os.environ["BATCH"])
project = os.environ["PROJECT"]
name = os.environ["RUN_NAME"]
result_json = os.environ["RESULT_JSON"]

model = YOLO(model_path)
metrics = model.val(
    data=data_yaml,
    split=split,
    device=device,
    imgsz=imgsz,
    batch=batch,
    project=project,
    name=name,
    exist_ok=True,
    verbose=False,
)

box = getattr(metrics, "box", None)
out = {
    "metrics/precision(B)": float(getattr(box, "mp", 0.0) if box else 0.0),
    "metrics/recall(B)": float(getattr(box, "mr", 0.0) if box else 0.0),
    "metrics/mAP50(B)": float(getattr(box, "map50", 0.0) if box else 0.0),
    "metrics/mAP50-95(B)": float(getattr(box, "map", 0.0) if box else 0.0),
}
with open(result_json, "w", encoding="utf-8") as f:
    json.dump(out, f)
"""
    with NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        tmp_json_path = Path(tmp.name)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{version_dir}:{env.get('PYTHONPATH', '')}"
    env["MODEL_PATH"] = str(model_path)
    env["DATA_YAML"] = str(data_yaml)
    env["SPLIT"] = split
    env["DEVICE"] = str(device)
    env["IMGSZ"] = str(imgsz)
    env["BATCH"] = str(batch)
    env["PROJECT"] = output_dir
    env["RUN_NAME"] = run_name
    env["RESULT_JSON"] = str(tmp_json_path)

    try:
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=version_dir,
            env=env,
            check=True,
        )
        return json.loads(tmp_json_path.read_text(encoding="utf-8"))
    finally:
        if tmp_json_path.exists():
            tmp_json_path.unlink()


def normalize_save_dir(save_dir_value: str) -> str:
    save_value = (save_dir_value or "").strip()
    if not save_value:
        return ""
    if save_value.startswith("runs/"):
        return save_value
    return f"runs/detect/{save_value}"


def main() -> None:
    args = parse_args()
    all_results: List[Dict[str, str]] = []

    for version in args.versions:
        version_dir = ROOT_DIR / version
        if not version_dir.exists():
            print(f"[WARN] 버전 디렉토리 없음: {version_dir}")
            continue

        grid_csv = version_dir / "runs" / "detect" / "grid_search_diff" / "grid_search_results.csv"
        if not grid_csv.exists():
            print(f"[WARN] grid search 결과 CSV 없음: {grid_csv}")
            continue

        data_yaml = version_dir / args.data_yaml
        if not data_yaml.exists():
            print(f"[WARN] data yaml 없음: {data_yaml}")
            continue

        try:
            split = resolve_split(data_yaml, args.split)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        top_rows = read_top_diff_rows(grid_csv, args.top_k, args.rank_metric)
        if not top_rows:
            print(f"[WARN] {version}: 성공 행이 없습니다.")
            continue

        print(f"\n========== {version} top-{len(top_rows)} 평가 시작 ==========")
        for rank, row in enumerate(top_rows, 1):
            iter_str = row.get("iteration", "")
            save_dir_raw = row.get("save_dir", "")
            save_dir = normalize_save_dir(save_dir_raw)
            weight_path, weight_kind = resolve_weight_path(version_dir, save_dir_raw)

            result_row = {
                "version": version,
                "iteration": iter_str,
                "save_dir": save_dir,
                "weight_path": str(weight_path),
                "weight_kind": weight_kind,
                "split": split,
                "status": "pending",
                "metrics/precision(B)": "",
                "metrics/recall(B)": "",
                "metrics/mAP50(B)": "",
                "metrics/mAP50-95(B)": "",
            }

            alpha = row.get("diff_alpha", "")
            beta = row.get("diff_beta", "")
            val_score = row.get(args.rank_metric, "")

            if weight_kind == "missing":
                print(
                    f"[SKIP] rank={rank} iter={iter_str} alpha={alpha} beta={beta} "
                    "weights 없음"
                )
                result_row["status"] = "missing_weights"
                all_results.append(result_row)
                continue

            run_name = f"diff_eval_rank{rank}_iter{iter_str}_{Path(save_dir_raw).name}"
            print(
                f"[RUN ] rank={rank} iter={iter_str} alpha={alpha} beta={beta} "
                f"val_{args.rank_metric}={val_score}"
            )

            try:
                metric = run_eval_in_subprocess(
                    version_dir=version_dir,
                    model_path=weight_path,
                    data_yaml=data_yaml,
                    split=split,
                    device=args.device,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    output_dir=args.output_dir,
                    run_name=run_name,
                )
                result_row["status"] = "success"
                result_row["metrics/precision(B)"] = f"{metric['metrics/precision(B)']:.6f}"
                result_row["metrics/recall(B)"] = f"{metric['metrics/recall(B)']:.6f}"
                result_row["metrics/mAP50(B)"] = f"{metric['metrics/mAP50(B)']:.6f}"
                result_row["metrics/mAP50-95(B)"] = f"{metric['metrics/mAP50-95(B)']:.6f}"
                print(
                    "[ OK ] "
                    f"mAP50={result_row['metrics/mAP50(B)']} "
                    f"mAP50-95={result_row['metrics/mAP50-95(B)']}"
                )
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] iter={iter_str} 평가 실패: {e}")
                result_row["status"] = "eval_failed"

            all_results.append(result_row)

    out_csv = ROOT_DIR / args.output_csv
    fieldnames = [
        "version",
        "iteration",
        "save_dir",
        "weight_path",
        "weight_kind",
        "split",
        "status",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n==========================================")
    print("Diff test 평가 완료")
    print(f"평가 기준 metric: {args.rank_metric}")
    print(f"결과 CSV: {out_csv}")
    print(f"총 결과 행 수: {len(all_results)} (기대값: 버전수 x top-k)")
    print("==========================================")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
v8_diff, v10_diff, v11_diff Diff Loss 실험 결과를 종합해
버전별 validation 상위 3개 조합을 test dataset으로 평가하는 스크립트.

- 입력: 각 버전의 runs/detect/grid_search_diff/grid_search_results.csv
- 선별: status=success 중 metric(기본 fitness) 기준 상위 3개
- 평가: 버전별 ultralytics 환경에서 subprocess로 model.val() 실행
- 출력: 루트의 diff_test_eval_results.csv (stage2_test_eval_results.csv와 동일 컬럼 형식)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff Loss top-3 x 3버전 test 평가")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v8_diff", "v10_diff", "v11_diff"],
        help="평가할 diff 버전 디렉토리 목록 (기본: v8_diff v10_diff v11_diff)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="버전별 validation 상위 개수 (기본: 3)",
    )
    parser.add_argument(
        "--rank-metric",
        choices=["fitness", "mAP50", "mAP50-95", "precision", "recall"],
        default="fitness",
        help="validation 순위 기준 컬럼 (기본: fitness)",
    )
    parser.add_argument(
        "--data-yaml",
        default="data/data_test.yaml",
        help="버전 디렉토리 기준 데이터 YAML 경로 (기본: data/data_test.yaml)",
    )
    parser.add_argument(
        "--split",
        choices=["auto", "test", "val"],
        default="auto",
        help="평가 split (auto면 test 우선, 없으면 val 사용)",
    )
    parser.add_argument("--device", default="0", help="평가 device (기본: 0)")
    parser.add_argument("--imgsz", type=int, default=1280, help="평가 이미지 크기 (기본: 1280)")
    parser.add_argument("--batch", type=int, default=4, help="평가 배치 크기 (기본: 4)")
    parser.add_argument(
        "--output-dir",
        default="runs/diff_test_eval",
        help="버전 내부 평가 결과 저장 디렉토리 (기본: runs/diff_test_eval)",
    )
    parser.add_argument(
        "--output-csv",
        default="diff_test_eval_results.csv",
        help="루트 기준 출력 CSV 파일명 (기본: diff_test_eval_results.csv)",
    )
    return parser.parse_args()


def has_key_in_yaml(yaml_path: Path, key: str) -> bool:
    if not yaml_path.exists():
        return False
    target = f"{key}:"
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(target):
            return True
    return False


def resolve_split(yaml_path: Path, split_arg: str) -> str:
    if split_arg in ("test", "val"):
        return split_arg
    if has_key_in_yaml(yaml_path, "test"):
        return "test"
    if has_key_in_yaml(yaml_path, "val"):
        return "val"
    raise ValueError(f"data yaml에 test/val 키가 없습니다: {yaml_path}")


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def read_top_diff_rows(csv_path: Path, top_k: int, rank_metric: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "success":
                rows.append(row)

    # 높은 metric 우선, 동점이면 iteration 오름차순
    rows.sort(
        key=lambda r: (-to_float(r.get(rank_metric, "")), int(r.get("iteration", "999999"))),
    )
    return rows[:top_k]


def resolve_weight_path(version_dir: Path, save_dir_value: str) -> Tuple[Path, str]:
    save_value = (save_dir_value or "").strip()
    candidates: List[Path] = []
    if save_value:
        raw_path = Path(save_value)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(version_dir / raw_path)
            candidates.append(version_dir / "runs" / "detect" / raw_path)
            if save_value.startswith("runs/"):
                candidates.append(version_dir / save_value)

    # 기본 fallback: save_dir가 run name일 때
    if save_value:
        candidates.append(version_dir / "runs" / "detect" / save_value)

    checked = set()
    for run_dir in candidates:
        run_dir = run_dir.resolve()
        if run_dir in checked:
            continue
        checked.add(run_dir)
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            return best_pt, "best.pt"
        last_pt = run_dir / "weights" / "last.pt"
        if last_pt.exists():
            return last_pt, "last.pt"

    # 마지막 후보 기준 missing 경로 반환
    fallback = (version_dir / "runs" / "detect" / save_value / "weights" / "best.pt").resolve()
    return fallback, "missing"


def run_eval_in_subprocess(
    version_dir: Path,
    model_path: Path,
    data_yaml: Path,
    split: str,
    device: str,
    imgsz: int,
    batch: int,
    output_dir: str,
    run_name: str,
) -> Dict[str, float]:
    code = r"""
import json
import os
from ultralytics import YOLO

model_path = os.environ["MODEL_PATH"]
data_yaml = os.environ["DATA_YAML"]
split = os.environ["SPLIT"]
device = os.environ["DEVICE"]
imgsz = int(os.environ["IMGSZ"])
batch = int(os.environ["BATCH"])
project = os.environ["PROJECT"]
name = os.environ["RUN_NAME"]
result_json = os.environ["RESULT_JSON"]

model = YOLO(model_path)
metrics = model.val(
    data=data_yaml,
    split=split,
    device=device,
    imgsz=imgsz,
    batch=batch,
    project=project,
    name=name,
    exist_ok=True,
    verbose=False,
)

box = getattr(metrics, "box", None)
out = {
    "metrics/precision(B)": float(getattr(box, "mp", 0.0) if box else 0.0),
    "metrics/recall(B)": float(getattr(box, "mr", 0.0) if box else 0.0),
    "metrics/mAP50(B)": float(getattr(box, "map50", 0.0) if box else 0.0),
    "metrics/mAP50-95(B)": float(getattr(box, "map", 0.0) if box else 0.0),
}
with open(result_json, "w", encoding="utf-8") as f:
    json.dump(out, f)
"""
    with NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        tmp_json_path = Path(tmp.name)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{version_dir}:{env.get('PYTHONPATH', '')}"
    env["MODEL_PATH"] = str(model_path)
    env["DATA_YAML"] = str(data_yaml)
    env["SPLIT"] = split
    env["DEVICE"] = str(device)
    env["IMGSZ"] = str(imgsz)
    env["BATCH"] = str(batch)
    env["PROJECT"] = output_dir
    env["RUN_NAME"] = run_name
    env["RESULT_JSON"] = str(tmp_json_path)

    try:
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=version_dir,
            env=env,
            check=True,
        )
        return json.loads(tmp_json_path.read_text(encoding="utf-8"))
    finally:
        if tmp_json_path.exists():
            tmp_json_path.unlink()


def normalize_save_dir(version_dir: Path, save_dir_value: str) -> str:
    save_value = (save_dir_value or "").strip()
    if not save_value:
        return ""
    if save_value.startswith("runs/"):
        return save_value
    return f"runs/detect/{save_value}"


def main() -> None:
    args = parse_args()
    all_results: List[Dict[str, str]] = []

    for version in args.versions:
        version_dir = ROOT_DIR / version
        if not version_dir.exists():
            print(f"[WARN] 버전 디렉토리 없음: {version_dir}")
            continue

        grid_csv = version_dir / "runs" / "detect" / "grid_search_diff" / "grid_search_results.csv"
        if not grid_csv.exists():
            print(f"[WARN] grid search 결과 CSV 없음: {grid_csv}")
            continue

        data_yaml = version_dir / args.data_yaml
        if not data_yaml.exists():
            print(f"[WARN] data yaml 없음: {data_yaml}")
            continue

        try:
            split = resolve_split(data_yaml, args.split)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        top_rows = read_top_diff_rows(grid_csv, args.top_k, args.rank_metric)
        if not top_rows:
            print(f"[WARN] {version}: 성공 행이 없습니다.")
            continue

        print(f"\n========== {version} top-{len(top_rows)} 평가 시작 ==========")
        for rank, row in enumerate(top_rows, 1):
            iter_str = row.get("iteration", "")
            save_dir_raw = row.get("save_dir", "")
            save_dir = normalize_save_dir(version_dir, save_dir_raw)
            weight_path, weight_kind = resolve_weight_path(version_dir, save_dir_raw)

            result_row = {
                "version": version,
                "iteration": iter_str,
                "save_dir": save_dir,
                "weight_path": str(weight_path),
                "weight_kind": weight_kind,
                "split": split,
                "status": "pending",
                "metrics/precision(B)": "",
                "metrics/recall(B)": "",
                "metrics/mAP50(B)": "",
                "metrics/mAP50-95(B)": "",
            }

            alpha = row.get("diff_alpha", "")
            beta = row.get("diff_beta", "")
            val_score = row.get(args.rank_metric, "")

            if weight_kind == "missing":
                print(
                    f"[SKIP] rank={rank} iter={iter_str} alpha={alpha} beta={beta} "
                    f"weights 없음"
                )
                result_row["status"] = "missing_weights"
                all_results.append(result_row)
                continue

            run_name = f"diff_eval_rank{rank}_iter{iter_str}_{Path(save_dir_raw).name}"
            print(
                f"[RUN ] rank={rank} iter={iter_str} alpha={alpha} beta={beta} "
                f"val_{args.rank_metric}={val_score}"
            )

            try:
                metric = run_eval_in_subprocess(
                    version_dir=version_dir,
                    model_path=weight_path,
                    data_yaml=data_yaml,
                    split=split,
                    device=args.device,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    output_dir=args.output_dir,
                    run_name=run_name,
                )
                result_row["status"] = "success"
                result_row["metrics/precision(B)"] = f"{metric['metrics/precision(B)']:.6f}"
                result_row["metrics/recall(B)"] = f"{metric['metrics/recall(B)']:.6f}"
                result_row["metrics/mAP50(B)"] = f"{metric['metrics/mAP50(B)']:.6f}"
                result_row["metrics/mAP50-95(B)"] = f"{metric['metrics/mAP50-95(B)']:.6f}"
                print(
                    "[ OK ] "
                    f"mAP50={result_row['metrics/mAP50(B)']} "
                    f"mAP50-95={result_row['metrics/mAP50-95(B)']}"
                )
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] iter={iter_str} 평가 실패: {e}")
                result_row["status"] = "eval_failed"

            all_results.append(result_row)

    out_csv = ROOT_DIR / args.output_csv
    fieldnames = [
        "version",
        "iteration",
        "save_dir",
        "weight_path",
        "weight_kind",
        "split",
        "status",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n==========================================")
    print("Diff test 평가 완료")
    print(f"평가 기준 metric: {args.rank_metric}")
    print(f"결과 CSV: {out_csv}")
    print(f"총 결과 행 수: {len(all_results)} (기대값: 버전수 x top-k)")
    print("==========================================")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
v8_diff, v10_diff, v11_diff의 Diff Loss 실험에서
validation 성능 상위 3개 하이퍼파라미터 조합을 선별해 test split 성능을 평가한다.

출력 형식은 stage2_test_eval_results.csv와 동일한 컬럼을 사용한다.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff 실험 top3 test dataset 일괄 평가")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v8_diff", "v10_diff", "v11_diff"],
        help="평가할 버전 디렉토리 목록",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="버전별 validation 상위 K개 조합 평가 (기본: 3)",
    )
    parser.add_argument(
        "--run-glob",
        default="train_diff_alpha*_beta*_fpn*",
        help="runs/detect 하위에서 Diff 실험 디렉토리를 찾기 위한 glob",
    )
    parser.add_argument(
        "--data-yaml",
        default="data/data_test.yaml",
        help="버전 디렉토리 기준 데이터 YAML 경로 (기본: data/data_test.yaml)",
    )
    parser.add_argument(
        "--split",
        choices=["auto", "test", "val"],
        default="auto",
        help="평가 split (auto면 test 우선, 없으면 val 사용)",
    )
    parser.add_argument("--device", default="0", help="평가에 사용할 device (기본: 0)")
    parser.add_argument("--imgsz", type=int, default=1280, help="평가 이미지 크기 (기본: 1280)")
    parser.add_argument("--batch", type=int, default=4, help="평가 배치 크기 (기본: 4)")
    parser.add_argument(
        "--output-dir",
        default="runs/diff_test_eval",
        help="버전별 val 결과 저장 디렉토리(버전 내부 상대경로, 기본: runs/diff_test_eval)",
    )
    parser.add_argument(
        "--out-csv",
        default="diff_test_eval_results.csv",
        help="루트 기준 출력 CSV 파일명 (기본: diff_test_eval_results.csv)",
    )
    return parser.parse_args()


def has_key_in_yaml(yaml_path: Path, key: str) -> bool:
    if not yaml_path.exists():
        return False
    target = f"{key}:"
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(target):
            return True
    return False


def resolve_split(yaml_path: Path, split_arg: str) -> str:
    if split_arg in ("test", "val"):
        return split_arg
    if has_key_in_yaml(yaml_path, "test"):
        return "test"
    if has_key_in_yaml(yaml_path, "val"):
        return "val"
    raise ValueError(f"data yaml에 test/val 키가 없습니다: {yaml_path}")


def to_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_val_metrics_from_results(results_csv: Path) -> Optional[Dict[str, float]]:
    rows: List[Dict[str, str]] = []
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    ranked: List[Tuple[float, float, float, float, Dict[str, str]]] = []
    for row in rows:
        map95 = to_float(row.get("metrics/mAP50-95(B)"))
        map50 = to_float(row.get("metrics/mAP50(B)"))
        precision = to_float(row.get("metrics/precision(B)"))
        recall = to_float(row.get("metrics/recall(B)"))
        if map95 is None or map50 is None or precision is None or recall is None:
            continue
        ranked.append((map95, map50, precision, recall, row))

    if not ranked:
        return None

    ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    best_row = ranked[0][4]
    return {
        "map95": float(best_row["metrics/mAP50-95(B)"]),
        "map50": float(best_row["metrics/mAP50(B)"]),
        "precision": float(best_row["metrics/precision(B)"]),
        "recall": float(best_row["metrics/recall(B)"]),
    }


def resolve_weight_path(run_dir: Path) -> Tuple[Path, str]:
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        return best_pt, "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if last_pt.exists():
        return last_pt, "last.pt"
    return best_pt, "missing"


def find_topk_diff_runs(version_dir: Path, run_glob: str, top_k: int) -> List[Dict[str, str]]:
    detect_dir = version_dir / "runs" / "detect"
    if not detect_dir.exists():
        return []

    candidates: List[Dict[str, str]] = []
    for run_dir in sorted(detect_dir.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        results_csv = run_dir / "results.csv"
        if not results_csv.exists():
            continue

        val_best = best_val_metrics_from_results(results_csv)
        if val_best is None:
            continue

        rel_save_dir = str(run_dir.relative_to(version_dir))
        weight_path, weight_kind = resolve_weight_path(run_dir)
        candidates.append(
            {
                "save_dir": rel_save_dir,
                "weight_path": str(weight_path),
                "weight_kind": weight_kind,
                "val_map95": val_best["map95"],
                "val_map50": val_best["map50"],
                "val_precision": val_best["precision"],
                "val_recall": val_best["recall"],
            }
        )

    candidates.sort(
        key=lambda x: (x["val_map95"], x["val_map50"], x["val_precision"], x["val_recall"]),
        reverse=True,
    )
    return candidates[:top_k]


def run_eval_in_subprocess(
    version_dir: Path,
    model_path: Path,
    data_yaml: Path,
    split: str,
    device: str,
    imgsz: int,
    batch: int,
    output_dir: str,
    run_name: str,
) -> Dict[str, float]:
    code = r"""
import json
import os
from ultralytics import YOLO

model_path = os.environ["MODEL_PATH"]
data_yaml = os.environ["DATA_YAML"]
split = os.environ["SPLIT"]
device = os.environ["DEVICE"]
imgsz = int(os.environ["IMGSZ"])
batch = int(os.environ["BATCH"])
project = os.environ["PROJECT"]
name = os.environ["RUN_NAME"]
result_json = os.environ["RESULT_JSON"]

model = YOLO(model_path)
metrics = model.val(
    data=data_yaml,
    split=split,
    device=device,
    imgsz=imgsz,
    batch=batch,
    project=project,
    name=name,
    exist_ok=True,
    verbose=False,
)

box = getattr(metrics, "box", None)
out = {
    "metrics/precision(B)": float(getattr(box, "mp", 0.0) if box else 0.0),
    "metrics/recall(B)": float(getattr(box, "mr", 0.0) if box else 0.0),
    "metrics/mAP50(B)": float(getattr(box, "map50", 0.0) if box else 0.0),
    "metrics/mAP50-95(B)": float(getattr(box, "map", 0.0) if box else 0.0),
}
with open(result_json, "w", encoding="utf-8") as f:
    json.dump(out, f)
"""
    with NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        tmp_json_path = Path(tmp.name)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{version_dir}:{env.get('PYTHONPATH', '')}"
    env["MODEL_PATH"] = str(model_path)
    env["DATA_YAML"] = str(data_yaml)
    env["SPLIT"] = split
    env["DEVICE"] = str(device)
    env["IMGSZ"] = str(imgsz)
    env["BATCH"] = str(batch)
    env["PROJECT"] = output_dir
    env["RUN_NAME"] = run_name
    env["RESULT_JSON"] = str(tmp_json_path)

    try:
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=version_dir,
            env=env,
            check=True,
        )
        result = json.loads(tmp_json_path.read_text(encoding="utf-8"))
        return result
    finally:
        if tmp_json_path.exists():
            tmp_json_path.unlink()


def main() -> None:
    args = parse_args()
    all_results: List[Dict[str, str]] = []

    for version in args.versions:
        version_dir = ROOT_DIR / version
        if not version_dir.exists():
            print(f"[WARN] 버전 디렉토리 없음: {version_dir}")
            continue

        data_yaml = version_dir / args.data_yaml
        if not data_yaml.exists():
            print(f"[WARN] data yaml 없음: {data_yaml}")
            continue

        try:
            split = resolve_split(data_yaml, args.split)
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        top_runs = find_topk_diff_runs(version_dir=version_dir, run_glob=args.run_glob, top_k=args.top_k)
        if not top_runs:
            print(f"[WARN] {version}: top{args.top_k} 선별 가능한 diff run이 없습니다.")
            continue

        print(f"\n========== {version} 평가 시작 (top {len(top_runs)}) ==========")
        for rank, run_info in enumerate(top_runs, start=1):
            save_dir = run_info["save_dir"]
            weight_path = Path(run_info["weight_path"])
            weight_kind = run_info["weight_kind"]

            result_row = {
                "version": version.replace("_diff", ""),
                "iteration": str(rank),
                "save_dir": save_dir,
                "weight_path": str(weight_path),
                "weight_kind": weight_kind,
                "split": split,
                "status": "pending",
                "metrics/precision(B)": "",
                "metrics/recall(B)": "",
                "metrics/mAP50(B)": "",
                "metrics/mAP50-95(B)": "",
            }

            if weight_kind == "missing":
                print(f"[SKIP] rank={rank} weights 없음: {weight_path.parent}")
                result_row["status"] = "missing_weights"
                all_results.append(result_row)
                continue

            run_name = f"diff_eval_rank{rank}_{Path(save_dir).name}"
            print(
                f"[RUN ] rank={rank} val_mAP50-95={run_info['val_map95']:.6f} "
                f"model={weight_kind} split={split}"
            )

            try:
                metric = run_eval_in_subprocess(
                    version_dir=version_dir,
                    model_path=weight_path,
                    data_yaml=data_yaml,
                    split=split,
                    device=args.device,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    output_dir=args.output_dir,
                    run_name=run_name,
                )
                result_row["status"] = "success"
                result_row["metrics/precision(B)"] = f"{metric['metrics/precision(B)']:.6f}"
                result_row["metrics/recall(B)"] = f"{metric['metrics/recall(B)']:.6f}"
                result_row["metrics/mAP50(B)"] = f"{metric['metrics/mAP50(B)']:.6f}"
                result_row["metrics/mAP50-95(B)"] = f"{metric['metrics/mAP50-95(B)']:.6f}"
                print(
                    "[ OK ] "
                    f"mAP50={result_row['metrics/mAP50(B)']} "
                    f"mAP50-95={result_row['metrics/mAP50-95(B)']}"
                )
            except subprocess.CalledProcessError as e:
                print(f"[FAIL] rank={rank} 평가 실패: {e}")
                result_row["status"] = "eval_failed"

            all_results.append(result_row)

    out_csv = ROOT_DIR / args.out_csv
    fieldnames = [
        "version",
        "iteration",
        "save_dir",
        "weight_path",
        "weight_kind",
        "split",
        "status",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n==========================================")
    print("Diff test 평가 완료")
    print(f"결과 CSV: {out_csv}")
    print("==========================================")


if __name__ == "__main__":
    main()
