import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datasets = {
    "100": {
        "file": "exp_ad_100.csv",
        "mAP50-95": {
            "baseline": [0.660209, 0.6613, 0.659429, 0.69484, 0.658914, 0.6039451066],
            "NPP":       [0.701691, 0.662773, 0.66916, 0.7141, 0.68263, 0.644945002],
            "difference": [0.699784, 0.68774, 0.698032, 0.70113, 0.720862, 0.6404478645],
        },
    },
    "75": {
        "file": "exp_ad_75.csv",
        "mAP50-95": {
            "baseline": [0.66577, 0.64458, 0.62275, 0.6124, 0.621448, 0.5643151125],
            "NPP":       [0.71802, 0.64194, 0.67539, 0.65724, 0.637824, 0.6026832064],
            "difference": [0.6979, 0.69125, 0.68112, 0.67516, 0.627397, 0.6324918029],
        },
    },
    "50": {
        "file": "exp_ad_50.csv",
        "mAP50-95": {
            "baseline": [0.52717, 0.56215, 0.60315, 0.5194, 0.574572, 0.5377957793],
            "NPP":       [0.63465, 0.56036, 0.56693, 0.59378, 0.586726, 0.6086990566],
            "difference": [0.69039, 0.65772, 0.59297, 0.64184, 0.595388, 0.574133773],
        },
    },
}

models = ["YOLO8", "YOLO10", "YOLO11", "YOLO12", "MBYOLO", "DeTR"]
loss_types = ["baseline", "NPP", "difference"]
colors = {"baseline": "#4C72B0", "NPP": "#DD8452", "difference": "#55A868"}

for pct, info in datasets.items():
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, model in enumerate(models):
        ax = axes[idx]
        for i, loss in enumerate(loss_types):
            val = info["mAP50-95"][loss][idx]
            ax.bar(i, val, color=colors[loss], width=0.6, label=loss)
            ax.text(i, val + 0.003, f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_xticks(range(len(loss_types)))
        ax.set_xticklabels(loss_types, fontsize=10)
        ax.set_ylabel("mAP50-95", fontsize=10)
        ax.set_ylim(0.45, 0.78)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[l]) for l in loss_types]
    fig.legend(handles, loss_types, loc="upper center", ncol=3, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(f"mAP50-95 Performance by Model and Loss Type ({pct}% Training Data)",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = f"exp_ad_visualization_{pct}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
