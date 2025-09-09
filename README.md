
## Multi-View Object Detection

## Introduction
This is our PyTorch implementation of the paper "[`A Multi-view Feature Enhancement Detection Method Integrating Consistency and Complementarity`].

<div align="center">
    <img src="mfenet.png" width="1000" alt="MFENet">
</div>

## Dataset
```bash
/dataset/powerdata.yaml: Private power data 
    # Due to the signing of a confidentiality agreement, this dataset is not publicly available at this time.
/dataset/publicallpower.yaml: Public power data
    # CPLID: https://github.com/InsulatorData/InsulatorDataSet
    # IDID: https://ieee-dataport.org/competitions/insulator-defect-detection
    # VPMBGI: https://github.com/phd-benel/VPMBGI
/dataset/VisDrone.yaml:VisDrone2019 
    # https://github.com/VisDrone/VisDrone-Dataset
other...
```

## Quick Start Examples

<details open>
<summary>Install</summary>

```bash
export WANDB_MODE=disabled
conda activate DA #debug
conda activate ObjectDetection #train

# clone the project and configure the environment.
git clone https://github.com/LiuJiaji1999/HFDNet.git
# the version of ultralytics is '8.3.9'           
# GPU-NVIDIA GeForce RTX 3090 
# CPU-12th Gen Intel(R) Core(TM) i9-12900
python: 3.8.18
torch:  1.12.0+cu113
torchvision: 0.13.0+cu113 
numpy: 1.22.3
```

</details>

<details open>
<summary>Train</summary>

```shell
python train.py 
# save outputlog
nohup python train.py > /log/XXX.log 2>&1 & tail -f /log/XXX.log
```
</details>


<details open>
<summary>Test</summary>

```bash
python val.py # test dataset 
python detect.py # visualize
```
</details>


#### Explanation of the file
```bash
1. main_profile.py ：model.info
2. test_yaml.py：test all yaml is run 
3. heatmap.py ：heatmap
4. get_FPS.py ：compute model param、inference-time、FPS
5. plot_result.py：visualize compare
6. get_model_erf.py ： erf
7. test_other.py: debug
```

<details >
<summary>Personal Debug</summary>

```bash
print('一. trainer.py/get_dataset 先从yaml文件获取 train')
print('二. trainer.py/get_dataloader 开始加载训练数据')
print('三. detect/train.py/build_dataset 开始真正构建数据集')
print('四. bulid.py/build_yolo_dataset 构建YOLO数据集')
print('五. dataset.py/build_transforms 开始数据增强')
print('六. augment.py/v8_transforms 开始执行数据增强函数，') #随机增强方式直接替换原图送进模型    
print('七.ultralytics/data/base.py/get_image_and_label，数据增强后的图片-标签对应'）
```
</details>



