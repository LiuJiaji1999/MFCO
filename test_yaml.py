import warnings
warnings.filterwarnings('ignore')
import os, tqdm
from ultralytics import YOLO

## test simgle
if __name__ == '__main__':
    # 直接指定你要测试的YAML文件
    yaml_file = 'yolo11-MVAFHAFB.yaml'  # 替换为你要测试的具体文件名
    yaml_path = f'ultralytics/cfg/models/11/{yaml_file}'
    
    try:
        print(f"Testing: {yaml_file}")
        model = YOLO(yaml_path)
        model.info(detailed=True)
        model.profile([640, 640])
        model.fuse()
        print("✓ All tests passed!")
    except Exception as e:
        print(f"❌ Error: {e}")

# ## test all 
# if __name__ == '__main__':
#     error_result = []
#     for yaml_path in tqdm.tqdm(os.listdir('ultralytics/cfg/models/11')):
#         if 'rtdetr' not in yaml_path and 'cls' not in yaml_path and 'world' not in yaml_path:
#             try:
#                 model = YOLO(f'ultralytics/cfg/models/11/{yaml_path}')
#                 model.info(detailed=True)
#                 model.profile([640, 640])
#                 model.fuse()
#             except Exception as e:
#                 error_result.append(f'{yaml_path} {e}')
    
#     for i in error_result:
#         print(i)