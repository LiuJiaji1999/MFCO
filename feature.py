from ultralytics import YOLO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

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

def tsne_compare(features_dict, title, ax, pca_dim=100): # save_path, 
    """
    features_dict: { "modelA": feats, "modelB": feats, ... }
    """
    all_features_list, all_labels = [], []
    for label, feats in features_dict.items():
        if isinstance(feats, (list, tuple)):
            # 每个元素应是单张图的 1D tensor，使用 stack 得到 (N, D)
            if len(feats) == 0:
                print(f"[警告] 模型 {label} 的特征列表为空，跳过")
                continue
            # detach & cpu for each sample
            sample_tensors = []
            for f in feats:
                t = f.detach().cpu()
                # 若某元素为多维（理论上不应），拉平成向量
                if t.ndim > 1:
                    t = t.reshape(-1)
                sample_tensors.append(t)
            # stack -> (N, D)
            feats_arr = torch.stack(sample_tensors, dim=0).numpy()
        elif isinstance(feats, torch.Tensor):
            t = feats.detach().cpu().numpy()
            if t.ndim == 1:
                feats_arr = t.reshape(1, -1)
            elif t.ndim == 2:
                feats_arr = t
            else:
                feats_arr = t.reshape(t.shape[0], -1)
        else:
            # 假如已经是 numpy array
            feats_arr = np.array(feats)
            if feats_arr.ndim == 1:
                feats_arr = feats_arr.reshape(1, -1)
        n_s, d = feats_arr.shape
        print(f"Model '{label}': samples={n_s}, feat_dim={d}")
        all_features_list.append(feats_arr)
        all_labels.extend([label] * n_s)

    if len(all_features_list) == 0:
        print("[错误] 无任何特征可绘制")
        return

    all_features = np.vstack(all_features_list)  # (total_samples, feat_dim)
    n_samples, n_features = all_features.shape
    print("Combined features shape:", all_features.shape)

    # PCA 先降维到 pca_dim（不能超过 n_features）
    target_pca = min(pca_dim, n_features)
    if target_pca < n_features:
        pca = PCA(n_components=target_pca, random_state=42)
        all_reduced = pca.fit_transform(all_features)
        print(f"PCA: {n_features} -> {target_pca}")
    else:
        all_reduced = all_features

    # 如果样本很少，直接用前两维可视化；否则用 t-SNE
    if n_samples < 2:
        print("[警告] 样本数 < 2，无法执行 t-SNE，使用 PCA 前两维可视化")
        if all_reduced.shape[1] >= 2:
            vis = all_reduced[:, :2]
        else:
            # pad zero列（极少情况）
            vis = np.hstack([all_reduced, np.zeros((n_samples, 2 - all_reduced.shape[1]))])
    else:
        perplexity = min(30, max(1, n_samples - 1))
        tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=perplexity)
        vis = tsne.fit_transform(all_reduced)
        print("t-SNE done, shape:", vis.shape)

    # # 绘图：每个模型一种颜色
    # plt.figure(figsize=(8, 6))
    # labels_unique = list(dict.fromkeys(all_labels))  # 保持顺序
    # cmap = cm.get_cmap('tab10', len(labels_unique))
    # for i, lab in enumerate(labels_unique):
    #     idx = [j for j, l in enumerate(all_labels) if l == lab]
    #     plt.scatter(vis[idx, 0], vis[idx, 1], label=lab, alpha=0.7, s=20, color=cmap(i))
    # plt.legend()
    # plt.title(title)
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path, dpi=300)
    # # plt.close()
    # print(f"[保存成功] {save_path}")

    # 绘制到指定子图 ax
    labels_unique = list(dict.fromkeys(all_labels))
    cmap = cm.get_cmap('tab10', len(labels_unique))
    for i, lab in enumerate(labels_unique):
        idx = [j for j, l in enumerate(all_labels) if l == lab]
        ax.scatter(vis[idx, 0], vis[idx, 1], label=lab, alpha=0.7, s=20, color=cmap(i))
    ax.set_title(title)

   

# -------------------
# 使用示例
# -------------------
if __name__ == "__main__":
    img_dir = "/home/lenovo/data/liujiaji/yolov8/powerdata/images/test/"
    # img_dir = "/home/lenovo/data/liujiaji/powerGit/yolov8/testImg"
    local_layers = [2, 4, 6, 8, 9]
    global_layers = [10]

    # 两个不同权重
    weight_paths = {
        "YOLO11": "runs/train/exp2/weights/best.pt",
        "MV-YOLO11": "runs/train/exp3/weights/best.pt",
        "MFCONet": "runs/train/exp4/weights/best.pt"
    }

    features_local = {}
    features_global = {}

    for name, w in weight_paths.items():
        fea_local, fea_global = extract_features(w, img_dir, local_layers, global_layers)
        features_local[name] = fea_local
        features_global[name] = fea_global


    # # # 局部特征对比
    # tsne_compare(features_local, "Local Features Across Models", "/home/lenovo/data/liujiaji/powerGit/mvod/features/tsne_local.png")
    # # # 全局特征对比
    # tsne_compare(features_global, "Global Features Across Models", "/home/lenovo/data/liujiaji/powerGit/mvod/features/tsne_global.png")

    # 创建一张大图，左右子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    tsne_compare(features_local, "Local Features", axes[1])
    tsne_compare(features_global, "Global Features", axes[0])

    axes[0].legend()
    axes[1].legend()

    save_path = "/home/lenovo/data/liujiaji/powerGit/mvod/features/tsne_local_global.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300,bbox_inches="tight")
    plt.close()
    print(f"[保存成功] {save_path}")


