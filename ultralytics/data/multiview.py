import albumentations as A
import cv2
import numpy as np
import torch
import os

# 定义多视角增强流水线
# 全局视角-原始图像 original、远距视图 far、近距视图 near
# 局部视角-倾斜视图 shear transformation、投影视图 affine transformation、色彩扰动 color jittering
multi_view_transforms = {
    "far": A.Compose([
        A.Resize(320, 320, always_apply=True, interpolation=cv2.INTER_LANCZOS4),  # 模拟远距（缩小目标）
        A.Resize(640, 640, always_apply=True)  # 保证统一大小
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

    "near": A.Compose([
        A.RandomResizedCrop(640, 640, scale=(0.8, 1.2), ratio=(0.75, 1.33), always_apply=True),
        A.Resize(640, 640, always_apply=True)  # 保证统一大小
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

    "shear": A.Compose([
        A.Affine(shear={"x": (-15, 15), "y": (-10, 10)}, p=1.0),
        A.Resize(640, 640, always_apply=True)  # 保证统一大小
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

    "affine": A.Compose([
        A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), translate_percent=(0.1, 0.1), p=1.0),
        A.Resize(640, 640, always_apply=True)  # 保证统一大小
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),

    "color": A.Compose([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2, p=1.0),
        A.Resize(640, 640, always_apply=True)  # 保证统一大小
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])),
}


EPS = 1e-6

def sanitize_yolo_bboxes(bboxes: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    将 YOLO 格式 (cx, cy, w, h) 的 bbox 转为安全范围，保证转换前后
    x_min/x_max/y_min/y_max 在 [eps, 1-eps]，并返回新的 (cx,cy,w,h)。
    bboxes: shape (N,4)，若 N==0 返回 shape (0,4)
    """
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x_min, y_min = cx - w / 2, cy - h / 2
    x_max, y_max = cx + w / 2, cy + h / 2

    # clip
    x_min = np.clip(x_min, eps, 1.0 - eps)
    y_min = np.clip(y_min, eps, 1.0 - eps)
    x_max = np.clip(x_max, eps, 1.0 - eps)
    y_max = np.clip(y_max, eps, 1.0 - eps)

    new_w = np.maximum(x_max - x_min, eps)
    new_h = np.maximum(y_max - y_min, eps)
    new_cx = (x_min + x_max) / 2
    new_cy = (y_min + y_max) / 2

    return np.stack([new_cx, new_cy, new_w, new_h], axis=1)


def generate_multiview_batch(batch, visualize=False, save_dir="/home/lenovo/data/liujiaji/powerGit/mvod/image/debug_multiview"):
    """
    输入: YOLOv8 batch (B=1 的 dict)
    输出: 多视角增强后的 batch，保持原始格式一致
           img:[B*V, C, H, W], cls:[N_total,1], bboxes:[N_total,4], batch_idx:[N_total]
    """
   

    device = batch["img"].device
    aug_views = []
    imgs = []
    bboxes_all = []
    cls_all = []
    batch_idx_all = []

    im_files = []
    ori_shapes = []
    resized_shapes = []

    # -------------------
    # 遍历 batch 内每张图像
    # -------------------
    B = batch["img"].shape[0]
    for b in range(B):
        img = (batch["img"][b].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        bboxes = batch["bboxes"][batch["batch_idx"] == b].cpu().numpy()
        labels = batch["cls"][batch["batch_idx"] == b].cpu().numpy().astype(int).reshape(-1)

        bboxes = sanitize_yolo_bboxes(bboxes)

        # 原始图也作为一个视角
        views = [("original", img, bboxes, labels)]

        # 额外增强视角
        for vname, aug in multi_view_transforms.items():
            transformed = aug(image=img,
                              bboxes=bboxes.tolist(),
                              class_labels=labels.tolist())
            t_img = transformed["image"]
            t_bboxes = np.array(transformed.get("bboxes", []), dtype=np.float32).reshape(-1, 4)
            t_bboxes = sanitize_yolo_bboxes(t_bboxes)
            t_labels = np.array(transformed.get("class_labels", []), dtype=np.int64).reshape(-1)

            views.append((vname, t_img, t_bboxes, t_labels))

        # 将每个视角加入 batch
        for vi, (vname, vimg, vbboxes, vlabels) in enumerate(views):
            # image
            vimg_t = torch.from_numpy(vimg).permute(2, 0, 1).float() / 255.0
            vimg_t = vimg_t.to(device)
            imgs.append(vimg_t)

            # bboxes / cls
            if vbboxes.shape[0] > 0:
                bboxes_all.append(torch.tensor(vbboxes, dtype=torch.float32))
                cls_all.append(torch.tensor(vlabels, dtype=torch.int64).unsqueeze(1))  # [N,1]
                batch_idx_all.append(torch.full((vbboxes.shape[0],), len(imgs)-1, dtype=torch.int64))
            else:
                bboxes_all.append(torch.zeros((0,4), dtype=torch.float32))
                cls_all.append(torch.zeros((0,1), dtype=torch.int64))
                batch_idx_all.append(torch.zeros((0,), dtype=torch.int64))

            im_files.append(f"{batch['im_file'][b]}_{vname}")
            ori_shapes.append(batch["ori_shape"][b])
            resized_shapes.append((vimg.shape[0], vimg.shape[1]))

            # 可视化
            if visualize:
                os.makedirs(save_dir, exist_ok=True)
                vis = vimg.copy()
                H, W = vis.shape[:2]
                for box, cls in zip(vbboxes, vlabels):
                    cx, cy, bw, bh = box
                    x1, y1 = int((cx - bw/2) * W), int((cy - bh/2) * H)
                    x2, y2 = int((cx + bw/2) * W), int((cy + bh/2) * H)
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 1)
                    cv2.putText(vis, str(int(cls)), (x1,max(4,y1-3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)
                save_path = os.path.join(save_dir, f"{os.path.basename(batch['im_file'][b])}_{vname}.jpg")
                cv2.imwrite(save_path, vis[..., ::-1])
                print(f"Saved {save_path}")
    
     # 添加首次调用标志
    if not hasattr(generate_multiview_batch, "_first_call"):
        print("\n")
        print("***首次调用 generate_multiview_batch 函数")
        for i in range(len(views)):
            aug_views.append(views[i][0])
        print('***aug view have:',aug_views)
        generate_multiview_batch._first_call = True

    # 拼接
    batch["img"] = torch.stack(imgs, dim=0)
    batch["bboxes"] = torch.cat(bboxes_all, dim=0) if len(bboxes_all) else torch.zeros((0,4), dtype=torch.float32)
    batch["cls"] = torch.cat(cls_all, dim=0) if len(cls_all) else torch.zeros((0,1), dtype=torch.int64)
    batch["batch_idx"] = torch.cat(batch_idx_all, dim=0) if len(batch_idx_all) else torch.zeros((0,), dtype=torch.int64)

    batch["im_file"] = im_files
    batch["ori_shape"] = ori_shapes
    batch["resized_shape"] = resized_shapes

    return batch
