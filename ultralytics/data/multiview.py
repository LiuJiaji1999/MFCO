import cv2
import os
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from PIL import Image


class MultiViewAugmentor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_img = cv2.imread(image_path)
        self.original_img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.views = {}  # key: name, value: image
        self.results = {}

        # LPIPS 模型
        self.lpips_model = lpips.LPIPS(net='alex').eval()

        # PIL transform
        self.to_tensor = transforms.ToTensor()

    def generate_views(self):
        img = self.original_img_rgb
        h, w = img.shape[:2]

        # 水平翻转
        self.views['horizontal_flip'] = cv2.flip(img, 1)

        # 对角翻转 = 水平 + 垂直
        self.views['diagonal_flip'] = cv2.flip(img, -1)

        # 缩放（0.5 倍 + 中心裁剪）
        small = cv2.resize(img, (int(w * 0.5), int(h * 0.5)))
        small_padded = cv2.copyMakeBorder(small,
                                          (h - small.shape[0]) // 2,
                                          (h - small.shape[0] + 1) // 2,
                                          (w - small.shape[1]) // 2,
                                          (w - small.shape[1] + 1) // 2,
                                          cv2.BORDER_CONSTANT,
                                          value=0)
        self.views['scale_0.5'] = small_padded

        # 放大（1.5 倍 + 中心裁剪）
        large = cv2.resize(img, (int(w * 1.5), int(h * 1.5)))
        center = (large.shape[0] // 2, large.shape[1] // 2)
        large_cropped = large[center[0] - h // 2:center[0] + h // 2,
                              center[1] - w // 2:center[1] + w // 2]
        self.views['scale_1.5'] = large_cropped

    def evaluate_views(self):
        ref = self.original_img_rgb
        ref_tensor = self.to_tensor(Image.fromarray(ref)).unsqueeze(0)

        for name, img in self.views.items():
            # 计算指标
            psnr_val = psnr(ref, img, data_range=255)
            ssim_val = ssim(ref, img, multichannel=True, data_range=255)
            img_tensor = self.to_tensor(Image.fromarray(img)).unsqueeze(0)
            with torch.no_grad():
                lpips_val = self.lpips_model(ref_tensor, img_tensor).item()

            self.results[name] = {
                'PSNR': round(psnr_val, 2),
                'SSIM': round(ssim_val, 4),
                'LPIPS': round(lpips_val, 4)
            }

    def save_views(self, output_dir='multi_views'):
        os.makedirs(output_dir, exist_ok=True)
        for name, img in self.views.items():
            save_path = os.path.join(output_dir, f'{name}.jpg')
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def print_results(self):
        print(f"\n📊 多视角质量评估（与原图对比）:")
        for name, metrics in self.results.items():
            print(f"[{name}] → PSNR: {metrics['PSNR']}, SSIM: {metrics['SSIM']}, LPIPS: {metrics['LPIPS']}")


if __name__ == '__main__':
    # 示例用法
    image_path = 'example.jpg'  # 替换成你的图片路径
    augmenter = MultiViewAugmentor(image_path)
    augmenter.generate_views()
    augmenter.evaluate_views()
    augmenter.save_views()
    augmenter.print_results()
