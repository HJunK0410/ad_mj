#!/usr/bin/env python3
"""
stage2_test_eval_results.csv의 모델별 상위3 + baseline을
10/25/50/75% 데이터셋으로 재학습/평가하는 스크립트.
"""

import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
os.environ["PYTHONPATH"] = str(script_dir) + ":" + os.environ.get("PYTHONPATH", "")

FOLDER_VERSION = script_dir.name  # v8, v10, v11
STAGE2_CSV = Path("/home/user/hyunjun/AD/stage2_test_eval_results.csv")
REDUCED_RESULTS_DIR = Path("/home/user/hyunjun/AD/reduced_train")
REDUCED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = REDUCED_RESULTS_DIR / f"{FOLDER_VERSION}_top3_subset_results.csv"

DATASET_MAP = {
    75: "./data/data_75.yaml",
    50: "./data/data_50.yaml",
    25: "./data/data_25.yaml",
    10: "./data/data_10.yaml",
}

EPOCHS = 400
PATIENCE = 20
BATCH = 4
DEVICE = 0
PYTHON_BIN = os.environ.get("PYTHON_BIN") or sys.executable


def parse_npp_from_save_dir(save_dir: str):
    train_name = Path(save_dir).name
    pattern = re.compile(
        r"train_npp_l2d(?P<l2d>[^_]+)_l1d(?P<l1d>[^_]+)_mask(?P<mask>[^_]+)_fpn(?P<fpn>.+)$"
    )
    match = pattern.match(train_name)
    if not match:
        raise ValueError(f"NPP save_dir 파싱 실패: {save_dir}")
    return {
        "type": "npp",
        "name": train_name,
        "npp_lambda_2d": match.group("l2d"),
        "npp_lambda_1d": match.group("l1d"),
        "npp_bbox_mask_weight": match.group("mask"),
        "npp_fpn_sources": match.group("fpn").replace("_", ","),
    }


def load_experiments():
    experiments = []
    with open(STAGE2_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["version"] != FOLDER_VERSION:
                continue
            iteration = row["iteration"].strip()
            if iteration in {"1", "2", "3"}:
                cfg = parse_npp_from_save_dir(row["save_dir"])
                cfg["source_iteration"] = iteration
                cfg["source_file"] = str(STAGE2_CSV)
                experiments.append(cfg)
            elif iteration.startswith("baseline_stage2"):
                experiments.append(
                    {
                        "type": "baseline",
                        "name": iteration,
                        "source_iteration": iteration,
                        "source_file": str(STAGE2_CSV),
                    }
                )
    return experiments


def read_metrics(project_dir: str, run_name: str):
    save_dir = Path(project_dir) / run_name
    ckpt_file = save_dir / "weights" / "best.pt"
    if not ckpt_file.exists():
        ckpt_file = save_dir / "weights" / "last.pt"
    if not ckpt_file.exists():
        return {
            "fitness": 0.0,
            "mAP50": 0.0,
            "mAP50-95": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }

    ckpt = torch.load(ckpt_file, map_location="cpu")
    metrics = ckpt.get("train_metrics", {})
    return {
        "fitness": metrics.get("fitness", 0.0),
        "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
        "precision": metrics.get("metrics/precision(B)", 0.0),
        "recall": metrics.get("metrics/recall(B)", 0.0),
    }


def build_command(exp: dict, ratio: int, data_yaml: str, project_dir: str, run_name: str):
    if exp["type"] == "baseline":
        return [
            PYTHON_BIN,
            "train_baseline.py",
            f"--batch={BATCH}",
            f"--device={DEVICE}",
            f"--epochs={EPOCHS}",
            f"--patience={PATIENCE}",
            f"--data={data_yaml}",
            f"--project={project_dir}",
            f"--name={run_name}",
        ]

    return [
        PYTHON_BIN,
        "train.py",
        f"--npp_lambda_2d={exp['npp_lambda_2d']}",
        f"--npp_lambda_1d={exp['npp_lambda_1d']}",
        f"--npp_bbox_mask_weight={exp['npp_bbox_mask_weight']}",
        f"--npp_fpn_sources={exp['npp_fpn_sources']}",
        f"--batch={BATCH}",
        f"--device={DEVICE}",
        f"--epochs={EPOCHS}",
        f"--patience={PATIENCE}",
        f"--data={data_yaml}",
        f"--project={project_dir}",
    ]


def main():
    experiments = load_experiments()
    if not experiments:
        print(f"[오류] {FOLDER_VERSION} 실험 조합을 찾지 못했습니다.")
        return

    print("==========================================")
    print(f"{FOLDER_VERSION} reduced dataset 재학습 시작")
    print(f"실험 조합 수: {len(experiments)}")
    print(f"데이터셋 비율: {list(DATASET_MAP.keys())}")
    print(f"결과 CSV: {RESULTS_CSV}")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==========================================")

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "version",
                "experiment_type",
                "source_file",
                "source_iteration",
                "dataset_ratio",
                "data_yaml",
                "project_dir",
                "run_name",
                "fitness",
                "mAP50",
                "mAP50-95",
                "precision",
                "recall",
                "status",
                "save_dir",
            ]
        )

    total = len(experiments) * len(DATASET_MAP)
    step = 0
    for exp in experiments:
        for ratio, data_yaml in DATASET_MAP.items():
            step += 1
            project_dir = f"runs/detect_{ratio}"
            run_name = (
                f"{exp['name']}_r{ratio}" if exp["type"] == "baseline" else exp["name"]
            )
            print(f"\n[{step}/{total}] {exp['type']} ratio={ratio} 실행")
            cmd = build_command(exp, ratio, data_yaml, project_dir, run_name)

            try:
                subprocess.run(cmd, cwd=script_dir, env=os.environ.copy(), check=True)
                metrics = read_metrics(project_dir, run_name)
                status = "success"
            except subprocess.CalledProcessError as e:
                print(f"  ✗ 실패: {e}")
                metrics = {
                    "fitness": 0.0,
                    "mAP50": 0.0,
                    "mAP50-95": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }
                status = "failed"

            with open(RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        FOLDER_VERSION,
                        exp["type"],
                        exp["source_file"],
                        exp["source_iteration"],
                        ratio,
                        data_yaml,
                        project_dir,
                        run_name,
                        round(metrics["fitness"], 5),
                        round(metrics["mAP50"], 5),
                        round(metrics["mAP50-95"], 5),
                        round(metrics["precision"], 5),
                        round(metrics["recall"], 5),
                        status,
                        f"{project_dir}/{run_name}",
                    ]
                )

    print("\n==========================================")
    print("reduced dataset 재학습 완료")
    print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"결과 저장: {RESULTS_CSV}")
    print("==========================================")


if __name__ == "__main__":
    main()
