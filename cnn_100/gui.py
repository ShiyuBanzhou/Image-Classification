import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import torch
# import torch.nn as nn # 此文件中没有直接使用 nn
import torchvision.transforms as transforms

# —————— 1. CIFAR-100 中文类名 ——————
# *** 修改: 更新为 CIFAR-100 的 100 个类别名称 ***
# (来源: generate_cifar100_simples.py)
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
# 确保类别数量正确
if len(CLASS_NAMES) != 100:
    messagebox.showerror("配置错误", f"CLASS_NAMES 列表应包含 100 个类别，但当前包含 {len(CLASS_NAMES)} 个。")
    exit()

# —————— 2. 导入所有模型定义 ——————
# 确保这些 .py 文件与 gui.py 在同一目录下
# 假设这些文件中的模型定义适用于 CIFAR-100 (例如，接受 num_classes=100)
from SimpleCNN import SimpleCNN
from VGG16 import VGG16CIFAR10 # 类名未改，但假设内部实现适配100类
from VGG19 import VGG19CIFAR10 # 类名未改，但假设内部实现适配100类
from ResNet50 import ResNet50 # 假设 ResNet50 函数接受 num_classes
from PreResNet18 import PreActResNet18 # 假设 PreActResNet18 函数接受 num_classes
from PreResNet50 import PreActResNet50 # 假设 PreActResNet50 函数接受 num_classes
from PreResNet50_small import PreActResNet50 as PreActResNet50Small # 假设 PreActResNet50 函数接受 num_classes
from Vit_b import create_vit_b_16_model
from Vit_t import create_vit_l_16_model # 假设 Vit_t.py 包含 Vit_L 模型

# —————— 3. 模型配置 ——————
# *** 修改: 更新为指向 CIFAR-100 训练好的模型文件和配置 ***
MODEL_CONFIG = {
    # 键是显示在下拉列表中的名称
    # 值是元组: (模型类/创建函数, 'checkpoints/cifar100模型文件名.pth', 'transform_type')
    'SimpleCNN_CIFAR100':       (SimpleCNN,             'checkpoints/simplecnn_cifar100.pth'),              # 假设使用'cifar'类型转换
    'VGG16_CIFAR100':           (VGG16CIFAR10,          'checkpoints/vgg16_cifar100.pth'),               # 假设使用'cifar'类型转换
    'VGG19_CIFAR100':           (VGG19CIFAR10,          'checkpoints/vgg19_cifar100.pth'),               # 假设使用'cifar'类型转换
    'ResNet50_CIFAR100':        (ResNet50,              'checkpoints/resnet50_cifar100.pth'),            # 假设使用'cifar'类型转换
    'PreActResNet18_CIFAR100':  (PreActResNet18,        'checkpoints/preactresnet18_cifar100.pth'),      # 假设使用'cifar'类型转换
    'PreActResNet50_CIFAR100':  (PreActResNet50,        'checkpoints/preactresnet50_cifar100.pth'),      # 假设使用'cifar'类型转换
    'PreActResNet50Small_CIFAR100': (PreActResNet50Small, 'checkpoints/preactresnet50small_cifar100.pth'),# 假设使用'cifar'类型转换
    'ViT_B_16_CIFAR100':        (create_vit_b_16_model, 'checkpoints/vit_b_16_224_cifar100.pth'),      # 使用'vit'类型转换
    'ViT_L_16_CIFAR100':        (create_vit_l_16_model, 'checkpoints/vit_l_16_224_cifar100.pth'),      # 使用'vit'类型转换
}

# *** 自动添加转换类型（如果缺少） ***
# 假设非ViT模型都使用 'cifar' 转换
updated_config = {}
for name, config_tuple in MODEL_CONFIG.items():
    if len(config_tuple) == 2: # 如果只提供了类/函数和路径
        model_class_or_func, path = config_tuple
        # 根据模型名称猜测转换类型
        transform_type = 'vit' if 'vit' in name.lower() else 'cifar'
        updated_config[name] = (model_class_or_func, path, transform_type)
        print(f"为模型 '{name}' 自动分配转换类型: '{transform_type}'")
    elif len(config_tuple) == 3:
        updated_config[name] = config_tuple
    else:
        print(f"警告：模型 '{name}' 的配置格式不正确，已跳过。")
MODEL_CONFIG = updated_config


# 检查 checkpoints 文件夹是否存在
if not os.path.isdir('checkpoints'):
    messagebox.showerror("错误", "未找到 'checkpoints' 文件夹。\n请确保包含模型文件的 'checkpoints' 文件夹与此脚本位于同一目录。")
    exit()

# 检查 MODEL_CONFIG 是否为空
if not MODEL_CONFIG:
    messagebox.showerror("错误", "模型配置 'MODEL_CONFIG' 为空。\n请在 gui.py 文件中配置至少一个模型。")
    exit()


# —————— 4. 图像预处理流水线 ——————
# *** 修改: 更新为 CIFAR-100 的均值和标准差 ***
cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)), # 确保输入尺寸为 32x32
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408], # CIFAR-100 常用均值
        std=[0.2675, 0.2565, 0.2761],  # CIFAR-100 常用标准差
    )
])

# ViT 转换保持不变 (ImageNet 标准)
vit_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # ImageNet 均值
        std=[0.229, 0.224, 0.225],  # ImageNet 标准差
    )
])

# 将转换存储在字典中以便查找
TRANSFORMS = {
    'cifar': cifar_transform,
    'vit': vit_transform
}

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# —————— 5. GUI 主类 ——————
class ClassifierGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        # *** 修改: 更新窗口标题 ***
        self.title("图像分类器 GUI (CIFAR-100)") # <--- 修改
        self.geometry("520x650")
        self.model = None
        self.current_model_name = None
        self.current_transform = None
        self.img_path = None
        self.photo = None

        # --- 界面布局 ---
        ttk.Label(self, text="选择模型：").pack(anchor='nw', padx=10, pady=(10, 0))
        model_list = list(MODEL_CONFIG.keys())
        self.model_var = tk.StringVar(value=model_list[0])
        self.model_combo = ttk.Combobox(
            self,
            textvariable=self.model_var,
            values=model_list,
            state='readonly',
            width=40 # 可能需要调整宽度
        )
        self.model_combo.pack(anchor='nw', padx=10, pady=(5, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)

        frm_buttons = ttk.Frame(self)
        frm_buttons.pack(anchor='nw', pady=5, padx=10)
        ttk.Button(frm_buttons, text="选择图片", command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Button(frm_buttons, text="开始分类", command=self.classify).grid(row=0, column=1, padx=5)

        frm_image = ttk.Frame(self, width=310, height=310, relief=tk.SUNKEN, borderwidth=1)
        frm_image.pack(pady=10, padx=10)
        frm_image.pack_propagate(False)
        self.image_label = ttk.Label(frm_image, text="请先选择图片")
        self.image_label.pack(expand=True)

        self.result_label = ttk.Label(self, text="分类结果：[尚未分类]", font=("Arial", 14), anchor='center')
        self.result_label.pack(pady=20, fill=tk.X, padx=10)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

        self.load_model()

    def set_status(self, text):
        self.status_var.set(text)
        self.update_idletasks()

    def on_model_select(self, event=None):
        selected_model_name = self.model_var.get()
        if selected_model_name != self.current_model_name:
            print(f"模型选择改变: {selected_model_name}")
            self.load_model()
        else:
            print("选择了当前已加载的模型，无需重新加载。")

    def load_model(self):
        name = self.model_var.get()
        self.set_status(f"正在加载模型: {name}...")
        try:
            model_class_or_func, checkpoint_path, transform_type = MODEL_CONFIG[name]

            if not os.path.exists(checkpoint_path):
                messagebox.showerror("错误：模型文件丢失", f"找不到模型文件：\n{checkpoint_path}")
                self.set_status(f"错误：找不到 {name} 的模型文件")
                self.model = None
                self.current_model_name = None
                self.current_transform = None
                return

            # *** 修改: 实例化模型时传递 num_classes=100 ***
            print(f"正在为 CIFAR-100 (100 类) 实例化模型: {name}")
            try:
                # 尝试传递 num_classes=100
                self.model = model_class_or_func(num_classes=100).to(device) # <--- 关键修改
                print(f"模型 {name} 使用 num_classes=100 成功实例化。")
            except TypeError as te:
                # 如果模型类不接受 num_classes 参数 (不太可能，但作为后备)
                print(f"警告：模型 {name} 实例化时可能不接受 num_classes 参数。尝试不带参数实例化... ({te})")
                self.model = model_class_or_func().to(device)
            except Exception as instantiate_e:
                 raise RuntimeError(f"实例化模型 '{name}' 失败: {instantiate_e}")

            state_dict = torch.load(checkpoint_path, map_location=device)

            if all(key.startswith('module.') for key in state_dict.keys()):
                print("检测到 'module.' 前缀，正在移除...")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name_new = k[7:]
                    new_state_dict[name_new] = v
                state_dict = new_state_dict

            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.current_model_name = name
            self.current_transform = TRANSFORMS.get(transform_type, cifar_transform) # 默认为 cifar 转换
            print(f"成功加载模型: {name}")
            print(f"使用 '{transform_type}' 类型的转换。")
            self.set_status(f"模型已加载: {name}")

        except Exception as e:
            messagebox.showerror("加载模型时出错", f"加载模型 '{name}' 失败：\n{e}")
            print(f"加载模型 {name} 时出错: {e}")
            self.model = None
            self.current_model_name = None
            self.current_transform = None
            self.set_status(f"错误：加载 {name} 失败")


    def load_image(self):
        path = filedialog.askopenfilename(
            title="选择一个图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if not path:
            return

        self.img_path = path
        self.set_status(f"已选择图片: {os.path.basename(path)}")
        try:
            img = Image.open(path).convert('RGB')
            img_copy = img.copy()
            img_copy.thumbnail((300, 300))
            self.photo = ImageTk.PhotoImage(img_copy)
            self.image_label.config(image=self.photo, text="")
            self.image_label.image = self.photo
            self.result_label.config(text="分类结果：[图片已更改，请重新分类]")
        except FileNotFoundError:
            messagebox.showerror("错误", f"无法找到文件：\n{path}")
            self.img_path = None
            self.set_status("错误：无法找到图片文件")
        except Exception as e:
            messagebox.showerror("打开图片时出错", f"无法加载或处理图片：\n{path}\n错误：{e}")
            self.img_path = None
            self.photo = None
            self.image_label.config(image=None, text="无法加载图片")
            self.set_status("错误：加载图片失败")

    def classify(self):
        if not self.img_path:
            messagebox.showwarning("操作无效", "请先点击“选择图片”按钮选择一张图片。")
            return

        if not self.model or not self.current_transform:
            self.load_model()
            if not self.model or not self.current_transform:
                 messagebox.showwarning("操作无效", f"模型 '{self.model_var.get()}' 或其转换未能成功加载。")
                 return

        if self.model_var.get() != self.current_model_name:
             print(f"模型选择已更改为 '{self.model_var.get()}' 但未加载，正在加载...")
             self.load_model()
             if not self.model or not self.current_transform: return

        self.set_status(f"正在使用 {self.current_model_name} 对图片进行分类...")
        try:
            img = Image.open(self.img_path).convert('RGB')
            input_tensor = self.current_transform(img)
            input_batch = input_tensor.unsqueeze(0)
            input_batch = input_batch.to(device)

            with torch.no_grad():
                outputs = self.model(input_batch)

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class_index = predicted_idx.item()

            # *** 修改: 确保使用包含 100 个类的 CLASS_NAMES ***
            if 0 <= predicted_class_index < len(CLASS_NAMES):
                 predicted_class_name = CLASS_NAMES[predicted_class_index]
            else:
                 predicted_class_name = f"未知类别 (索引: {predicted_class_index})"

            confidence_score = confidence.item() * 100

            result_text = f"分类结果：{predicted_class_name}\n置信度：{confidence_score:.2f}%"
            self.result_label.config(text=result_text)
            print(f"分类完成: {predicted_class_name} ({confidence_score:.2f}%)")
            self.set_status("分类完成")

        except Exception as e:
            messagebox.showerror("分类时出错", f"对图片进行分类时发生错误：\n{e}")
            print(f"分类时出错: {e}")
            self.result_label.config(text="分类结果：[分类失败]")
            self.set_status("错误：分类失败")


if __name__ == '__main__':
    if not MODEL_CONFIG:
        print("错误：没有在 MODEL_CONFIG 中配置任何模型。请编辑 gui.py 文件。")
    else:
        app = ClassifierGUI()
        app.mainloop()