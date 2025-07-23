import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# yaml是做轻量化的话可以用get_all_yaml_param_and_flops.py脚本

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml') # YOLO11
    model.load('yolo11n.pt') # loading pretrain weights
    model.train(data='/home/lenovo/data/liujiaji/ultralytics-yolo11-main/dataset/powerdata.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=1,
                close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练
                optimizer='SGD', # using SGD
                patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )