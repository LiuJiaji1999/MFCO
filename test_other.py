# 1.anaconda 环境变量问题，加载的包 老是 提示到 其他环境下

# import os
# import sys

# print("当前解释器路径:", sys.executable)
# print("环境变量PYTHONPATH:", os.environ.get("PYTHONPATH"))
# print("模块搜索路径:")
# for p in sys.path:
#     print("  ", p)

# try:
#     import einops
#     print("einops 路径:", einops.__file__)
# except ImportError:
#     print("没有安装 einops")


# import tkinter as tk
# tk.Tk().mainloop()


# 2.想看下数据增强后的结果

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from types import SimpleNamespace
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 创建具有属性的hyp对象而不是字典
hyp_dict = {"mosaic": 1.0, "copy_paste": 0.5, "degrees": 10.0, "translate": 0.2, "scale": 0.9,"shear":0.1,"perspective":0.2}
hyp = SimpleNamespace(**hyp_dict)

dataset = YOLODataset(img_path="/home/lenovo/data/liujiaji/Datasets-Video/VisDrone2019-DET/images/val", imgsz=640)

transforms = v8_transforms(dataset, imgsz=640, hyp=hyp)
augmented_data = transforms(dataset[0])
print(augmented_data)