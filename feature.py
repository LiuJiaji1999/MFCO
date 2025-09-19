from ultralytics import YOLO
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# -------------------
# Step1: 提取特征
# -------------------
def extract_features(weight_path, img_dir, local_layers, global_layers):
    """提取局部和全局特征"""
    model = YOLO(weight_path)

    # 局部特征
    fea_local = model.predict(
        source=img_dir,
        imgsz=640,
        conf=0.3,
        embed=local_layers,
        save=False,
        verbose=False
    )

    # 全局特征
    fea_global = model.predict(
        source=img_dir,
        imgsz=640,
        conf=0.3,
        embed=global_layers,
        save=False,
        verbose=False
    )

    return fea_local, fea_global


# -------------------
# Step2: t-SNE 可视化（局部 / 全局分开）
# -------------------
def tsne_compare(features_dict, title, save_path):
    """
    features_dict: { "modelA": feats, "modelB": feats, ... }
    """
    all_features, all_labels = [], []

    for label, feats in features_dict.items():
        if isinstance(feats, (list, tuple)):
            feats = torch.cat(feats, dim=-1)
        feats = feats.detach().cpu().numpy()
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)

        all_features.append(feats)
        all_labels.extend([label] * feats.shape[0])

    all_features = np.vstack(all_features)

    n_samples = all_features.shape[0]

    if n_samples < 2:
        print("[警告] 样本数不足以执行 t-SNE，直接返回原始特征")
        tsne_result = all_features
    else:
        perplexity = min(30, n_samples - 1)
        tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
        tsne_result = tsne.fit_transform(all_features)

    # tsne = TSNE(n_components=2, random_state=42, init="pca",perplexity=min(30, all_features.shape[0] - 1)  )
    # tsne_result = tsne.fit_transform(all_features)

    plt.figure(figsize=(8, 6))
    for label in set(all_labels):
        idx = [i for i, l in enumerate(all_labels) if l == label]
        plt.scatter(tsne_result[idx, 0], tsne_result[idx, 1], label=label, alpha=0.6)

    plt.legend()
    plt.title(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[保存成功] {save_path}")


# -------------------
# 使用示例
# -------------------
if __name__ == "__main__":
    # img_dir = "/home/lenovo/data/liujiaji/yolov8/powerdata/images/test/"
    img_dir = "/home/lenovo/data/liujiaji/powerGit/yolov8/testImg"
    local_layers = [2, 4, 6, 8, 9]
    global_layers = [10]

    # 多个不同权重
    weight_paths = [
        "runs/train/exp2/weights/best.pt",
        "runs/train/exp4/weights/best.pt"
    ]

    features_local = {}
    features_global = {}

    for w in weight_paths:
        fea_local, fea_global = extract_features(w, img_dir, local_layers, global_layers)
        print(len(fea_local))
        print(len(fea_global))
        model_name = os.path.basename(w).replace(".pt", "")
        features_local[model_name] = fea_local
        features_global[model_name] = fea_global

    # # 局部特征对比
    # tsne_compare(features_local, "t-SNE Local Features Across Models", "/home/lenovo/data/liujiaji/powerGit/mvod/features/tsne_local.png")

    # # 全局特征对比
    # tsne_compare(features_global, "t-SNE Global Features Across Models", "/home/lenovo/data/liujiaji/powerGit/mvod/features/tsne_global.png")
