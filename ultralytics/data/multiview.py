import albumentations as A
import cv2
import numpy as np
import torch

# 定义多视角增强流水线
multi_view_transforms = {
    "far": A.Compose([
        A.Resize(320, 320, always_apply=True),  # 模拟远距（缩小目标）
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


def generate_multiview_batch(batch):
    """
    输入: batch (YOLOv8的batch字典，含 img/bboxes/cls 等)
    输出: 扩展后的 batch，包含多视角版本
    """
    # 原图 img 已经在 cuda 上，保留 device  
    device = batch["img"].device 

    imgs, bboxes, labels, im_files = [], [], [], []
    # 取原始数据
    img = (batch["img"][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # CHW -> HWC
    h, w = img.shape[:2]
    orig_bboxes = batch["bboxes"].cpu().numpy()  # yolo 格式 [x, y, w, h]
    orig_labels = batch["cls"].cpu().numpy().astype(int).tolist()

    # 原始图
    imgs.append(batch["img"][0])
    bboxes.append(orig_bboxes)
    labels.append(orig_labels)
    im_files.append(batch["im_file"][0])

    # 多视角增强
    for view_name, aug in multi_view_transforms.items():
        transformed = aug(image=img, bboxes=orig_bboxes, class_labels=orig_labels)

        t_img = transformed["image"]
        t_bboxes = transformed["bboxes"]
        t_labels = transformed["class_labels"]

        # 转回 torch (HWC -> CHW)
        t_img = torch.from_numpy(t_img).permute(2, 0, 1).float() / 255.0
        t_img = t_img.to(device)   # ⚡只把 image 放到和原图相同的 device

        imgs.append(t_img)
        bboxes.append(np.array(t_bboxes)) # ⚡保持 CPU numpy
        labels.append(t_labels)
        im_files.append(f"{batch['im_file'][0]}_{view_name}")

    # 拼接成新的 batch
    batch["img"] = torch.stack(imgs, dim=0)  # 所有 img 在 cuda
    batch["bboxes"] = [torch.tensor(b, dtype=torch.float32) for b in bboxes] # 保持在 CPU
    batch["cls"] = [torch.tensor(l, dtype=torch.int64) for l in labels] # 保持在 CPU
    batch["im_file"] = im_files

    return batch
