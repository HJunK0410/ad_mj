#!/usr/bin/env python3
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VERSIONS = ["v8", "v10", "v11"]


def parse_args():
    parser = argparse.ArgumentParser(description="v8/v10/v11 baseline stage2 학습 실행")
    parser.add_argument("--versions", nargs="+", default=VERSIONS, help="실행할 버전 목록")
    parser.add_argument("--epochs", type=int, default=400, help="학습 epoch (기본: 400)")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience (기본: 20)")
    parser.add_argument("--batch", type=int, default=4, help="batch size (기본: 4)")
    parser.add_argument("--imgsz", type=int, default=1280, help="image size (기본: 1280)")
    parser.add_argument("--workers", type=int, default=0, help="dataloader workers (기본: 0)")
    parser.add_argument("--device", default="0", help="device id (기본: 0)")
    parser.add_argument("--amp", default="false", choices=["true", "false"], help="AMP 사용 여부")
    parser.add_argument("--data", default="./data/data.yaml", help="버전 폴더 기준 data yaml 경로")
    parser.add_argument(
        "--project",
        default="runs/detect_baseline",
        help="버전 폴더 기준 저장 project 경로 (기본: runs/detect_baseline)",
    )
    parser.add_argument(
        "--name-prefix",
        default="baseline_stage2",
        help="실험 이름 prefix (버전명이 뒤에 자동 추가됨)",
    )
    parser.add_argument(
        "--summary-csv",
        default="baseline_train_results.csv",
        help="학습 요약 CSV 파일명 (루트 기준, 기본: baseline_train_results.csv)",
    )
    return parser.parse_args()


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize_run(version, version_dir, project, run_name):
    run_dir = version_dir / project / run_name
    results_csv = run_dir / "results.csv"
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    summary = {
        "version": version,
        "run_name": run_name,
        "save_dir": str(Path(project) / run_name),
        "results_csv": str(results_csv),
        "epochs_trained": "",
        "best_epoch": "",
        "best_metrics/precision(B)": "",
        "best_metrics/recall(B)": "",
        "best_metrics/mAP50(B)": "",
        "best_metrics/mAP50-95(B)": "",
        "weight_path": "",
        "weight_kind": "missing",
        "status": "failed",
    }

    if best_pt.exists():
        summary["weight_path"] = str(best_pt)
        summary["weight_kind"] = "best.pt"
    elif last_pt.exists():
        summary["weight_path"] = str(last_pt)
        summary["weight_kind"] = "last.pt"

    if not results_csv.exists():
        return summary

    rows = []
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return summary

    summary["status"] = "success"
    last_row = rows[-1]
    summary["epochs_trained"] = str(int(_to_float(last_row.get("epoch", 0), 0.0)) + 1)

    best_row = max(rows, key=lambda r: _to_float(r.get("metrics/mAP50-95(B)", 0.0), 0.0))
    summary["best_epoch"] = str(int(_to_float(best_row.get("epoch", 0), 0.0)) + 1)
    summary["best_metrics/precision(B)"] = f"{_to_float(best_row.get('metrics/precision(B)', 0.0)):.6f}"
    summary["best_metrics/recall(B)"] = f"{_to_float(best_row.get('metrics/recall(B)', 0.0)):.6f}"
    summary["best_metrics/mAP50(B)"] = f"{_to_float(best_row.get('metrics/mAP50(B)', 0.0)):.6f}"
    summary["best_metrics/mAP50-95(B)"] = f"{_to_float(best_row.get('metrics/mAP50-95(B)', 0.0)):.6f}"
    return summary


def write_summary_csv(rows, out_csv):
    fieldnames = [
        "version",
        "run_name",
        "save_dir",
        "results_csv",
        "epochs_trained",
        "best_epoch",
        "best_metrics/precision(B)",
        "best_metrics/recall(B)",
        "best_metrics/mAP50(B)",
        "best_metrics/mAP50-95(B)",
        "weight_path",
        "weight_kind",
        "status",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one_version(version, args):
    version_dir = ROOT_DIR / version
    script_path = version_dir / "train_baseline.py"
    if not script_path.exists():
        raise FileNotFoundError(f"{script_path} 파일이 없습니다.")

    run_name = f"{args.name_prefix}_{version}"
    cmd = [
        sys.executable,
        str(script_path),
        f"--epochs={args.epochs}",
        f"--patience={args.patience}",
        f"--batch={args.batch}",
        f"--imgsz={args.imgsz}",
        f"--workers={args.workers}",
        f"--device={args.device}",
        f"--amp={args.amp}",
        f"--data={args.data}",
        f"--project={args.project}",
        f"--name={run_name}",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{version_dir}:{env.get('PYTHONPATH', '')}"

    print(f"\n========== {version} baseline 학습 시작 ==========")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=version_dir, env=env, check=True)
    print(f"========== {version} baseline 학습 완료 ==========")
    return summarize_run(version, version_dir, args.project, run_name)


def main():
    args = parse_args()
    summaries = []
    for version in args.versions:
        summaries.append(run_one_version(version, args))

    out_csv = ROOT_DIR / args.summary_csv
    write_summary_csv(summaries, out_csv)

    print("\n모든 baseline 학습이 완료되었습니다.")
    print(f"학습 요약 CSV: {out_csv}")


if __name__ == "__main__":
    main()
