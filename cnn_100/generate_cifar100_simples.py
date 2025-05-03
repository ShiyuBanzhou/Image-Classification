# generate_cifar100_samples.py

import os
import zipfile
from torchvision.datasets import CIFAR100
from PIL import Image

# ———— 1. CIFAR-100 测试集加载 ————
dataset = CIFAR100(root='./data', train=False, download=True)

# ———— 2. 中文类别列表 ————
CLASS_NAMES = [
    '苹果', '水族鱼', '婴儿', '熊', '海狸', '床', '蜜蜂', '甲虫',
    '自行车', '瓶子', '碗', '男孩', '桥', '公共汽车', '蝴蝶', '骆驼',
    '罐头', '城堡', '毛毛虫', '牛', '椅子', '黑猩猩', '时钟',
    '云', '蟑螂', '沙发', '螃蟹', '鳄鱼', '杯子', '恐龙',
    '海豚', '大象', '比目鱼', '森林', '狐狸', '女孩', '仓鼠',
    '房子', '袋鼠', '键盘', '灯', '割草机', '豹', '狮子',
    '蜥蜴', '龙虾', '男人', '枫树', '摩托车', '山', '老鼠',
    '蘑菇', '橡树', '橙子', '兰花', '水獭', '棕榈树', '梨',
    '皮卡车', '松树', '平原', '盘子', '罂粟', '豪猪',
    '负鼠', '兔子', '浣熊', '射线', '路', '火箭', '玫瑰',
    '海', '海豹', '鲨鱼', '鼩鼱', '臭鼬', '摩天大楼', '蜗牛', '蛇',
    '蜘蛛', '松鼠', '有轨电车', '向日葵', '甜椒', '桌子',
    '坦克', '电话', '电视', '老虎', '拖拉机', '火车', '鳟鱼',
    '郁金香', '乌龟', '衣柜', '鲸鱼', '柳树', '狼', '女人', '蠕虫'
]
assert len(CLASS_NAMES) == 100, "CLASS_NAMES 列表长度应为 100"

# ———— 3. 建立保存目录 ————
output_dir = 'samples'
os.makedirs(output_dir, exist_ok=True)

# ———— 4. 从测试集中各取一张样本保存 ————
saved = set()
for img, label in dataset:
    if label not in saved:
        class_name = CLASS_NAMES[label]
        img.save(os.path.join(output_dir, f"{class_name}.png"))
        saved.add(label)
    if len(saved) == 100:
        break

print("已保存 100 张不同类别的样本图像到 samples/ 目录。")

# ———— 5. 打包为 ZIP ————
zip_path = './cifar100_samples.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(output_dir):
        zf.write(os.path.join(output_dir, fname), arcname=f"samples/{fname}")

print(f"已将 samples/ 打包为 {zip_path}。")
