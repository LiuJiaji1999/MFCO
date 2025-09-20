import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pwd = os.getcwd()

# 设置Nature风格
sns.set_theme(style="whitegrid") # , font="Times New Roman"
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "lines.linewidth": 2,
})

exp_labels = {
    'exp2': 'YOLO11',
    'exp4': 'MFCONet',
}

palette = sns.color_palette("tab10", len(exp_labels))

# ---------------- Loss 和 Metric 分开定义 ----------------
loss_keys = {
    'train/box_loss': 'Train Box Loss',
    'train/dfl_loss': 'Train DFL Loss',
    'train/cls_loss': 'Train Cls Loss',
    'train/gob_loss': 'Train Glo Loss',
    'train/loc_loss': 'Train Loc Loss',
    'val/box_loss': 'Val Box Loss',
    'val/dfl_loss': 'Val DFL Loss',
    'val/cls_loss': 'Val Cls Loss',
}

metric_keys = {
    'metrics/precision(B)': 'Precision',
    'metrics/recall(B)': 'Recall',
    'metrics/mAP50(B)': 'mAP@0.5',
    'metrics/mAP50-95(B)': 'mAP@0.5:0.95'
}

# ---------------- 画布：2行6列 ----------------
fig, axes = plt.subplots(2, 6, figsize=(28, 10))
axes = axes.flatten()

# ---- 前4列（8个子图）画 Loss ----
for ax, (key, title) in zip(axes[:len(loss_keys)], loss_keys.items()):
    for (exp, label), color in zip(exp_labels.items(), palette):
        data = pd.read_csv(f"runs/train/{exp}/results.csv")
        if key not in data.columns:
            continue
        series = data[key].astype(np.float32).replace(np.inf, np.nan)
        series = series.fillna(series.interpolate())
        smooth = series.rolling(window=3, min_periods=1).mean()
        ax.plot(smooth, label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ---- 后2列（4个子图）画 Metrics ----
for ax, (key, title) in zip(axes[len(loss_keys):], metric_keys.items()):
    for (exp, label), color in zip(exp_labels.items(), palette):
        data = pd.read_csv(f"runs/train/{exp}/results.csv")
        if key not in data.columns:
            continue
        series = data[key].astype(np.float32).replace(np.inf, np.nan)
        series = series.fillna(series.interpolate())
        smooth = series.rolling(window=3, min_periods=1).mean()
        ax.plot(smooth, label=label, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ---- 统一图例 ----
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("overall_curve.png", dpi=300, bbox_inches="tight")
print(f"保存成功: {pwd}/overall_curve.png")
