"""YOLOv8 baseline / NPP / diff 학습 수렴 곡선 (train·val 합산 loss, mAP50-95).

합산 loss에 cls_loss 상한 클리핑을 적용하고, val 합산에서 이웃 대비 튀는 지점은
이전·다음 에포크의 (클리핑 후) val 합산 값 평균으로 치환한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 정상 구간을 넘는 로그 스파이크만 눌러서 y축 스케일을 유지 (필요 시 조정)
CLS_LOSS_CLIP_MAX = 45.0

# val 합산 loss 스파이크: 이전·다음 에포크 원본 val 평균으로 치환 (감지 민감도)
VAL_SPIKE_RATIO = 1.55
VAL_SPIKE_MIN_ABS = 9.0

RUNS = {
    "Baseline": Path(
        "/home/user/hyunjun/AD/v8/runs/detect_baseline/baseline_stage2_v8/results.csv"
    ),
    "NPP": Path(
        "/home/user/hyunjun/AD/v8/runs/detect/"
        "train_npp_l2d0.06_l1d0.08_mask0.2_fpn15_18_21/results.csv"
    ),
    "Difference": Path(
        "/home/user/hyunjun/AD/v8_diff/runs/detect/"
        "train_diff_alpha0.0_beta1.0_fpn15_18_21/results.csv"
    ),
}

BOX, CLS, DFL = "train/box_loss", "train/cls_loss", "train/dfl_loss"
VBOX, VCLS, VDFL = "val/box_loss", "val/cls_loss", "val/dfl_loss"
MAP = "metrics/mAP50-95(B)"


def summed_loss(
    df: pd.DataFrame,
    box_col: str,
    cls_col: str,
    dfl_col: str,
    cls_clip_max: float = CLS_LOSS_CLIP_MAX,
) -> pd.Series:
    box = df[box_col].astype(float)
    cls = df[cls_col].astype(float).clip(lower=0.0, upper=cls_clip_max)
    dfl = df[dfl_col].astype(float)
    return box + cls + dfl


def replace_val_spikes_with_neighbor_mean(
    val: pd.Series,
    ratio: float = VAL_SPIKE_RATIO,
    min_abs: float = VAL_SPIKE_MIN_ABS,
) -> pd.Series:
    """스파이크로 보이는 val loss를 (이전·다음 에포크의 원본 val) 평균으로 대체."""
    raw = val.astype(float).to_numpy()
    out = raw.copy()
    n = len(raw)
    for i in range(1, n - 1):
        prev_, next_ = raw[i - 1], raw[i + 1]
        neigh = (prev_ + next_) / 2.0
        if not np.isfinite(raw[i]) or not np.isfinite(neigh):
            continue
        if raw[i] <= min_abs:
            continue
        baseline = max(neigh, 1e-6)
        if raw[i] > ratio * baseline:
            out[i] = neigh
    return pd.Series(out, index=val.index)


def load_curves(path: Path):
    df = pd.read_csv(path)
    epoch = df["epoch"].astype(int)
    train_total = summed_loss(df, BOX, CLS, DFL)
    val_raw = summed_loss(df, VBOX, VCLS, VDFL)
    val_total = replace_val_spikes_with_neighbor_mean(val_raw)
    map95 = df[MAP].astype(float)
    return epoch, train_total, val_total, map95


def main():
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=False)
    colors = {
        "train": "#1f77b4",
        "val": "#ff7f0e",
        "map": "#2ca02c",
    }

    for ax, (name, path) in zip(axes, RUNS.items()):
        ep, tr, va, m = load_curves(path)
        ax2 = ax.twinx()

        (l1,) = ax.plot(ep, tr, color=colors["train"], lw=2, label="Train loss (sum)")
        (l2,) = ax.plot(ep, va, color=colors["val"], lw=2, label="Val loss (sum)")
        (l3,) = ax2.plot(ep, m, color=colors["map"], lw=2, label="mAP50-95")

        ax.set_ylabel("Loss (box + cls + dfl)", color="#333", fontsize=11)
        ax2.set_ylabel("mAP50-95", color=colors["map"], fontsize=11)
        ax.tick_params(axis="y", labelcolor="#333")
        ax2.tick_params(axis="y", labelcolor=colors["map"])
        ax.set_title(f"YOLOv8 — {name}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_xlabel("Epoch", fontsize=10)

        lines = [l1, l2, l3]
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, loc="center right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "YOLOv8 training convergence — baseline vs NPP vs difference",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.text(
        0.5,
        0.02,
        (
            f"cls clipped [0, {CLS_LOSS_CLIP_MAX}]; "
            f"val spikes replaced by mean of adjacent epochs' val loss"
        ),
        ha="center",
        fontsize=8,
        color="0.35",
    )
    out = Path("/home/user/hyunjun/AD/plots/yolov8_convergence_baseline_npp_diff.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    for name, path in RUNS.items():
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ep, tr, va, m = load_curves(path)
        ax2 = ax.twinx()
        ax.plot(ep, tr, color=colors["train"], lw=2, label="Train loss (sum)")
        ax.plot(ep, va, color=colors["val"], lw=2, label="Val loss (sum)")
        ax2.plot(ep, m, color=colors["map"], lw=2, label="mAP50-95")
        ax.set_ylabel("Loss (box + cls + dfl)", color="#333", fontsize=11)
        ax2.set_ylabel("mAP50-95", color=colors["map"], fontsize=11)
        ax.tick_params(axis="y", labelcolor="#333")
        ax2.tick_params(axis="y", labelcolor=colors["map"])
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_title(f"YOLOv8 — {name}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.35)
        h1, lbl1 = ax.get_legend_handles_labels()
        h2, lbl2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, lbl1 + lbl2, loc="center right", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.14)
        fig.text(
            0.5,
            0.02,
            (
                f"cls clipped [0, {CLS_LOSS_CLIP_MAX}]; "
                f"val spikes → mean of prev/next epoch val loss"
            ),
            ha="center",
            fontsize=8,
            color="0.35",
        )
        safe = name.lower().replace(" ", "_")
        one = Path(f"/home/user/hyunjun/AD/plots/yolov8_convergence_{safe}.png")
        fig.savefig(one, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {one}")


if __name__ == "__main__":
    main()
