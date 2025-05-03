# generate_cifar10_samples.py

import os
import zipfile
from torchvision.datasets import CIFAR10
from PIL import Image

# ———— 1. CIFAR-10 测试集加载 ————
dataset = CIFAR10(root='./data', train=False, download=True)

# ———— 2. 中文类别列表 ——（务必与模型使用的 CLASS_NAMES 顺序一致） ————
CLASS_NAMES = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]
assert len(CLASS_NAMES) == 10, "CLASS_NAMES 列表长度应为 10"

# ———— 3. 建立保存目录 ————
output_dir = 'samples'
os.makedirs(output_dir, exist_ok=True)

# ———— 4. 从测试集中各取一张样本保存 ————
saved = set()
for img, label in dataset:
    if label not in saved:
        class_name = CLASS_NAMES[label]
        # dataset 返回的是 PIL.Image.Image，直接保存
        img.save(os.path.join(output_dir, f"{class_name}.png"))
        saved.add(label)
    if len(saved) == 10:
        break

print("已保存 10 张不同类别的样本图像到 samples/ 目录。")

# ———— 5. 打包为 ZIP ————
zip_path = 'cifar10_samples.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(output_dir):
        zf.write(os.path.join(output_dir, fname), arcname=f"samples/{fname}")

print(f"已将 samples/ 打包为 {zip_path}。")
