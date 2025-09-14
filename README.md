
## Multi-View Object Detection

## 💡Introduction
This is our PyTorch implementation of the paper "[`Multi-view Feature Collaborative Optimization Network for Object Detection`].

<div align="center">
    <img src="MFCONet.png" width="1000" alt="MFCONet">
</div>

## 🚩Dataset
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

## 🛠️Quick Start Examples

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
Since github cannot push big file, we put '.pt' into the Google Drive,so you can directly test :
# https://drive.google.com/drive/folders/1SSlZEZvlthQqWaZFEPYCWwkEe-40fqsX
python val.py # test dataset 
python detect.py # visualize
```
</details>

#### 📝RTDETR result

| Method | pin-un | pin-ru | pin-de | Insulator-bu | Insulator-de | Insulator-di | mAP(%) |
|---|---|---|---|---|---|---|---|
| RT-DETR(640,640) | |  |  |  |  |  | 51.3 |
| RT-DETR(640,640)+ours | -| - |-  |-  |- | - | - |
| RT-DETR(384,384) | 6.04 | 48.4 | 28.0 | 54.7 | 45.8 | 23.4 | 34.4 |
| RT-DETR(384,384)+ours | 10.6 | 57.4 | 33.1 | 66.0 | 66.6 | 35.2 | 44.8 |

| Method | CPLID-defect | VPMBGI-defect | IDID-flashover | IDID-broken | mAP(%) |
|---|---|---|---|---|---|
| RT-DETR(640,640) |  |  | |  | 94.5 |
| RT-DETR(640,640)+ours |-| - | - | - | -|
| RT-DETR(384,384) | 97.9 | 80.1 | 81.0 | 82.5 | 85.4 |
| RT-DETR(384,384)+ours | 99.5 | 99.7 | 88.4 | 94.5 | 92.8 |


#### 📢Plan
```bash
We will modify network to the Camouflaged/Salient Object Detection,enabling fair comparison with other methods.
Thanks again to the sprirt of the open-source:https://github.com/stars/LiuJiaji1999/lists/mvod
```

#### 🌟Explanation of the file
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



