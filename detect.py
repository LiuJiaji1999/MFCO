import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

### old
if __name__ == '__main__':
    model = YOLO('runs/train/exp3/weights/best.pt') # select your model.pt path
    model.predict(
                  # source='/home/lenovo/data/liujiaji/Datasets-Power/privatepower/pin/rust/img',
                  source='/home/lenovo/data/liujiaji/powerGit/yolov8/testImg',
                  imgsz=640,
                  project='runs/detect',
                  name='11mvb-prp',
                  save=True,
                  conf=0.3,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # embed=[2],
                  # visualize=True, # visualize model features maps
                  line_width=5, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )