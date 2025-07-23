### Multi-view Object Detection



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