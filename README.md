### Multi-view Object Detection

```shell
export WANDB_MODE=disabled
conda activate DA #debug
conda activate ObjectDetection #train
```

###### input
```bash
/ultralytics/models/yolo/model.py
/ultralytics/models/yolo/detect/__init__.py 
/ultralytics/models/yolo/detect/uda_train.py
/ultralytics/data/uda_build.py  # load dataset  def uda_build_dataloader
/ultralytics/nn/uda_tasks.py  # update model structure
/ultralytics/engine/uda_trainer.py # update trainer
/ultralytics/utils/daca.py # compute loss 
/ultralytics/engine/validator.py  # loss
/ultralytics/cfg/default.yaml # add weight value
/ultralytics/nn/modules/head.py # head pseudo
/ultralytics/utils/plotting.py # output_to_target

```
print('一. trainer.py/get_dataset 先从yaml文件获取 train')
print('二. trainer.py/get_dataloader 开始加载训练数据')
print('三. detect/train.py/build_dataset 开始真正构建数据集')
print('四. bulid.py/build_yolo_dataset 构建YOLO数据集')
print('五. dataset.py/build_transforms 开始数据增强')
