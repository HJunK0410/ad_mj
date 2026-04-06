#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="baseline 모델 test 성능 평가")
    parser.add_argument(
        "--summary-csv",
        default="baseline_train_results.csv",
        help="학습 요약 CSV 파일명 (기본: baseline_train_results.csv)",
    )
    parser.add_argument(
        "--output-csv",
        default="baseline_test_eval_results.csv",
        help="test 평가 결과 CSV 파일명 (기본: baseline_test_eval_results.csv)",
    )
    parser.add_argument(
        "--data-yaml",
        default="data/data_test.yaml",
        help="버전 폴더 기준 데이터 YAML 경로 (기본: data/data_test.yaml)",
    )
    parser.add_argument(
        "--split",
        choices=["auto", "test", "val"],
        default="auto",
        help="평가 split (auto면 test 우선, 없으면 val)",
    )
    parser.add_argument("--device", default="0", help="평가 device (기본: 0)")
    parser.add_argument("--imgsz", type=int, default=1280, help="평가 이미지 크기")
    parser.add_argument("--batch", type=int, default=4, help="평가 배치 크기")
    parser.add_argument(
        "--project",
        default="runs/baseline_test_eval",
        help="버전 폴더 기준 평가 결과 저장 project",
    )
    return parser.parse_args()


def has_key_in_yaml(path, key):
    if not path.exists():
        return False
    key_prefix = f"{key}:"
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(key_prefix):
            return True
    return False


def resolve_split(yaml_path, split_arg):
    if split_arg in ("test", "val"):
        return split_arg
    if has_key_in_yaml(yaml_path, "test"):
        return "test"
    if has_key_in_yaml(yaml_path, "val"):
        return "val"
    raise ValueError(f"data yaml에 test/val 키가 없습니다: {yaml_path}")


def run_eval_in_subprocess(version_dir, model_path, data_yaml, split, device, imgsz, batch, project, run_name, workers=0):
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
    workers=int(os.environ["WORKERS"]),
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
        tmp_json = Path(tmp.name)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{version_dir}:{env.get('PYTHONPATH', '')}"
    env["MODEL_PATH"] = str(model_path)
    env["DATA_YAML"] = str(data_yaml)
    env["SPLIT"] = split
    env["DEVICE"] = str(device)
    env["IMGSZ"] = str(imgsz)
    env["BATCH"] = str(batch)
    env["PROJECT"] = project
    env["RUN_NAME"] = run_name
    env["RESULT_JSON"] = str(tmp_json)
    env["WORKERS"] = str(workers)

    try:
        subprocess.run([sys.executable, "-c", code], cwd=version_dir, env=env, check=True)
        return json.loads(tmp_json.read_text(encoding="utf-8"))
    finally:
        if tmp_json.exists():
            tmp_json.unlink()


def main():
    args = parse_args()
    summary_csv = ROOT_DIR / args.summary_csv
    if not summary_csv.exists():
        raise FileNotFoundError(f"{summary_csv} 파일이 없습니다. 먼저 baseline 학습을 실행하세요.")

    rows = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for row in rows:
        version = row.get("version", "")
        version_dir = ROOT_DIR / version
        weight_path = Path(row.get("weight_path", ""))
        run_name = row.get("run_name", "")

        out_row = {
            "version": version,
            "run_name": run_name,
            "save_dir": row.get("save_dir", ""),
            "weight_path": str(weight_path),
            "weight_kind": row.get("weight_kind", ""),
            "split": "",
            "status": "pending",
            "metrics/precision(B)": "",
            "metrics/recall(B)": "",
            "metrics/mAP50(B)": "",
            "metrics/mAP50-95(B)": "",
        }

        if not weight_path.exists():
            out_row["status"] = "missing_weights"
            results.append(out_row)
            continue

        data_yaml = version_dir / args.data_yaml
        if not data_yaml.exists():
            out_row["status"] = "missing_data_yaml"
            results.append(out_row)
            continue

        try:
            split = resolve_split(data_yaml, args.split)
        except ValueError:
            out_row["status"] = "invalid_data_yaml"
            results.append(out_row)
            continue

        out_row["split"] = split
        eval_name = f"baseline_test_{run_name}"
        try:
            metric = run_eval_in_subprocess(
                version_dir=version_dir,
                model_path=weight_path,
                data_yaml=data_yaml,
                split=split,
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
                project=args.project,
                run_name=eval_name,
                workers=0,
            )
            out_row["status"] = "success"
            out_row["metrics/precision(B)"] = f"{metric['metrics/precision(B)']:.6f}"
            out_row["metrics/recall(B)"] = f"{metric['metrics/recall(B)']:.6f}"
            out_row["metrics/mAP50(B)"] = f"{metric['metrics/mAP50(B)']:.6f}"
            out_row["metrics/mAP50-95(B)"] = f"{metric['metrics/mAP50-95(B)']:.6f}"
        except subprocess.CalledProcessError:
            # 일부 환경에서 GPU 평가가 세그폴트(-11) 나는 경우가 있어 CPU로 1회 재시도
            try:
                metric = run_eval_in_subprocess(
                    version_dir=version_dir,
                    model_path=weight_path,
                    data_yaml=data_yaml,
                    split=split,
                    device="cpu",
                    imgsz=args.imgsz,
                    batch=1,
                    project=args.project,
                    run_name=f"{eval_name}_cpu_fallback",
                    workers=0,
                )
                out_row["status"] = "success_cpu_fallback"
                out_row["metrics/precision(B)"] = f"{metric['metrics/precision(B)']:.6f}"
                out_row["metrics/recall(B)"] = f"{metric['metrics/recall(B)']:.6f}"
                out_row["metrics/mAP50(B)"] = f"{metric['metrics/mAP50(B)']:.6f}"
                out_row["metrics/mAP50-95(B)"] = f"{metric['metrics/mAP50-95(B)']:.6f}"
            except subprocess.CalledProcessError:
                out_row["status"] = "eval_failed"

        results.append(out_row)

    out_csv = ROOT_DIR / args.output_csv
    fieldnames = [
        "version",
        "run_name",
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
        writer.writerows(results)

    print(f"baseline test 평가 완료: {out_csv}")


if __name__ == "__main__":
    main()
