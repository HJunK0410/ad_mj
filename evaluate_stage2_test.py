#!/usr/bin/env python3
"""
v8, v10, v11 Stage2 모델(test split) 일괄 평가 스크립트.

- 각 버전의 runs/detect/grid_search_npp/grid_search_results.csv에서 stage2 성공 행을 읽습니다.
- 기본적으로 상위 10개(phase=stage2, iteration 오름차순) 모델을 대상으로 평가합니다.
- 각 버전의 ultralytics를 분리 사용하기 위해 버전별 서브프로세스로 평가를 실행합니다.
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
    parser = argparse.ArgumentParser(description="Stage2 모델 test dataset 일괄 평가")
    parser.add_argument(
        "--versions",
        nargs="+",
        default=["v8", "v10", "v11"],
        help="평가할 버전 디렉토리 목록 (기본: v8 v10 v11)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="버전별 평가할 stage2 모델 개수 (기본: 10)",
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
        default="runs/stage2_test_eval",
        help="버전별 결과 저장 디렉토리(버전 내부 상대경로, 기본: runs/stage2_test_eval)",
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


def read_stage2_rows(csv_path: Path, top_k: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("phase") == "stage2" and row.get("status") == "success":
                rows.append(row)
    rows.sort(key=lambda r: int(r.get("iteration", "999999")))
    return rows[:top_k]


def resolve_weight_path(version_dir: Path, save_dir_value: str) -> Tuple[Path, str]:
    run_dir = version_dir / Path(save_dir_value)
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        return best_pt, "best.pt"
    last_pt = run_dir / "weights" / "last.pt"
    if last_pt.exists():
        return last_pt, "last.pt"
    return best_pt, "missing"


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

        grid_csv = version_dir / "runs" / "detect" / "grid_search_npp" / "grid_search_results.csv"
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

        stage2_rows = read_stage2_rows(grid_csv, args.top_k)
        if not stage2_rows:
            print(f"[WARN] {version}: stage2 성공 행이 없습니다.")
            continue

        print(f"\n========== {version} 평가 시작 ({len(stage2_rows)}개) ==========")
        for row in stage2_rows:
            iter_str = row.get("iteration", "")
            save_dir = row.get("save_dir", "")
            weight_path, weight_kind = resolve_weight_path(version_dir, save_dir)
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

            if weight_kind == "missing":
                print(f"[SKIP] iter={iter_str} weights 없음: {weight_path.parent}")
                result_row["status"] = "missing_weights"
                all_results.append(result_row)
                continue

            run_name = f"stage2_eval_iter{iter_str}_{Path(save_dir).name}"
            print(f"[RUN ] iter={iter_str} model={weight_kind} split={split}")
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

    out_csv = ROOT_DIR / "stage2_test_eval_results.csv"
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
    print("Stage2 test 평가 완료")
    print(f"결과 CSV: {out_csv}")
    print("==========================================")


if __name__ == "__main__":
    main()
