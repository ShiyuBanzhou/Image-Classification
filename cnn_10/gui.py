import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import torch
# import torch.nn as nn # nn is not directly used in this file, can be removed if not needed elsewhere indirectly
import torchvision.transforms as transforms

# —————— 1. CIFAR-10 中文类名 ——————
# (确保这与你的模型训练时使用的类别顺序一致)
CLASS_NAMES = [
    '飞机', '汽车', '鸟', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]

# —————— 2. 导入所有模型定义 ——————
# 确保这些 .py 文件与 gui.py 在同一目录下，或者在 Python 的搜索路径中
# 取消注释你已经训练好并希望在 GUI 中使用的模型
from SimpleCNN import SimpleCNN
from VGG16 import VGG16CIFAR10
from VGG19 import VGG19CIFAR10
from ResNet50 import ResNet50
from PreResNet18 import PreActResNet18
from PreResNet50 import PreActResNet50
from PreResNet50_small import PreActResNet50 as PreActResNet50Small
from Vit_b import create_vit_b_16_model # [cite: 15]
from Vit_t import create_vit_l_16_model # [cite: 17]

# —————— 3. 模型配置 ——————
# 将你训练好的模型和对应的 .pth 文件路径添加到这里
# 键是显示在下拉列表中的名称
# 值是一个元组：(模型类名, 'checkpoints/模型文件名.pth', 'transform_type')
# transform_type 可以是 'cifar' 或 'vit'
MODEL_CONFIG = {
    'SimpleCNN':        (SimpleCNN,             'checkpoints/simplecnn.pth', 'cifar'),
    'VGG16':            (VGG16CIFAR10,          'checkpoints/vgg16.pth', 'cifar'),
    'VGG19':            (VGG19CIFAR10,          'checkpoints/vgg19.pth', 'cifar'),
    'ResNet50':         (ResNet50,              'checkpoints/resnet50.pth', 'cifar'),
    'PreActResNet18':   (PreActResNet18,        'checkpoints/preactresnet18.pth', 'cifar'),
    'PreActResNet50':   (PreActResNet50,        'checkpoints/preactresnet50.pth', 'cifar'),
    'PreActResNet50small': (PreActResNet50Small, 'checkpoints/preactresnet50_small.pth', 'cifar'),
    'ViT_B_16_224_CIFAR10': (create_vit_b_16_model, 'checkpoints/vit_b_16_224_cifar10.pth', 'vit'), # [cite: 1, 15]
    'ViT_L_16_224_CIFAR10': (create_vit_l_16_model, 'checkpoints/vit_l_16_224_cifar10.pth', 'vit'), # [cite: 1, 17]
}

# 检查 checkpoints 文件夹是否存在
if not os.path.isdir('checkpoints'):
    messagebox.showerror("错误", "未找到 'checkpoints' 文件夹。\n请确保包含模型文件的 'checkpoints' 文件夹与此脚本位于同一目录。")
    exit() # 如果文件夹不存在，则退出

# 检查 MODEL_CONFIG 是否为空
if not MODEL_CONFIG:
    messagebox.showerror("错误", "模型配置 'MODEL_CONFIG' 为空。\n请在 gui.py 文件中配置至少一个模型。")
    exit() # 如果没有配置模型，则退出


# —————— 4. 图像预处理流水线 ——————
# 定义不同类型的预处理
cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)), # CIFAR-10 默认是 32x32
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], # CIFAR-10 常用均值 [cite: 1]
        std=[0.2470, 0.2435, 0.2616],  # CIFAR-10 常用标准差 [cite: 1]
    )
])

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)), # ViT 需要 224x224 [cite: 15, 17]
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # ImageNet 均值 [cite: 15, 17]
        std=[0.229, 0.224, 0.225],  # ImageNet 标准差 [cite: 15, 17]
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
        self.title("图像分类器 GUI")
        self.geometry("520x650") # 可以根据需要调整窗口大小
        self.model = None
        self.current_model_name = None
        self.current_transform = None # 用于存储当前模型的转换
        self.img_path = None
        self.photo = None # 保持对 PhotoImage 的引用

        # --- 界面布局 ---
        # 模型选择
        ttk.Label(self, text="选择模型：").pack(anchor='nw', padx=10, pady=(10, 0))
        self.model_var = tk.StringVar(value=list(MODEL_CONFIG.keys())[0]) # 默认选中第一个
        self.model_combo = ttk.Combobox(
            self,
            textvariable=self.model_var,
            values=list(MODEL_CONFIG.keys()),
            state='readonly', # 防止用户输入自定义值
            width=30 # 设置下拉框宽度
        )
        self.model_combo.pack(anchor='nw', padx=10, pady=(5, 10))
        # *** 关键修改：绑定下拉框选择事件到 on_model_select ***
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select) # [cite: 1]

        # 按钮区域
        frm_buttons = ttk.Frame(self)
        frm_buttons.pack(anchor='nw', pady=5, padx=10)
        ttk.Button(frm_buttons, text="选择图片", command=self.load_image).grid(row=0, column=0, padx=5) # [cite: 1]
        ttk.Button(frm_buttons, text="开始分类", command=self.classify).grid(row=0, column=1, padx=5) # [cite: 1]

        # 图片预览区域 (使用 Frame 控制大小)
        frm_image = ttk.Frame(self, width=310, height=310, relief=tk.SUNKEN, borderwidth=1) # [cite: 1]
        frm_image.pack(pady=10, padx=10)
        frm_image.pack_propagate(False) # 防止 Label 撑大 Frame [cite: 1]
        self.image_label = ttk.Label(frm_image, text="请先选择图片") # [cite: 1]
        self.image_label.pack(expand=True) # 让 Label 在 Frame 中居中 [cite: 1]

        # 分类结果展示
        self.result_label = ttk.Label(self, text="分类结果：[尚未分类]", font=("Arial", 14), anchor='center') # [cite: 1]
        self.result_label.pack(pady=20, fill=tk.X, padx=10)

        # 状态栏 (可选，用于显示加载信息)
        self.status_var = tk.StringVar(value="就绪") # [cite: 1]
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X) # [cite: 1]

        # --- 初始化：加载默认选中的模型 ---
        self.load_model() # [cite: 1]

    def set_status(self, text):
        """更新状态栏文本"""
        self.status_var.set(text)
        self.update_idletasks() # 立即更新界面

    def on_model_select(self, event=None): # [cite: 1]
        """当用户从下拉框选择不同模型时调用"""
        selected_model_name = self.model_var.get()
        if selected_model_name != self.current_model_name:
            print(f"模型选择改变: {selected_model_name}")
            self.load_model() # 重新加载模型（并更新转换）
        else:
            print("选择了当前已加载的模型，无需重新加载。")

    def load_model(self): # [cite: 1]
        """加载下拉框中选定的模型"""
        name = self.model_var.get()
        self.set_status(f"正在加载模型: {name}...")
        try:
            # 从配置中获取模型类、路径和转换类型
            model_class, checkpoint_path, transform_type = MODEL_CONFIG[name]

            # 检查模型文件是否存在
            if not os.path.exists(checkpoint_path):
                messagebox.showerror("错误：模型文件丢失", f"找不到模型文件：\n{checkpoint_path}\n\n请检查 'checkpoints' 文件夹和 MODEL_CONFIG 配置。")
                self.set_status(f"错误：找不到 {name} 的模型文件")
                self.model = None # 确保模型置空
                self.current_model_name = None
                self.current_transform = None # 清空转换
                return # 提前退出

            # 实例化模型
            # **注意:** 确保模型类接受 num_classes=10 或不接受参数
            # 如果模型类（如 create_vit_b_16_model）需要参数，请在此处传递
            # 假设所有模型类都接受 num_classes=10 或无参数
            # 对于 ViT, create_model 函数处理 num_classes
            if transform_type == 'vit':
                self.model = model_class(num_classes=10).to(device) # [cite: 15, 17] ViT 创建函数需要 num_classes
            else:
                 # 假设其他模型类接受 num_classes 或无参数
                 try:
                      self.model = model_class(num_classes=10).to(device)
                 except TypeError:
                      self.model = model_class().to(device) # 尝试无参数实例化

            # 加载权重
            # 使用 map_location=device 确保权重加载到正确的设备上
            state_dict = torch.load(checkpoint_path, map_location=device) # [cite: 1]

            # --- 处理可能的键名不匹配 (例如 DataParallel 保存的模型) ---
            if all(key.startswith('module.') for key in state_dict.keys()): # [cite: 1]
                print("检测到 'module.' 前缀，正在移除...")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name_new = k[7:] # remove `module.`
                    new_state_dict[name_new] = v
                state_dict = new_state_dict
            # --- 处理结束 ---

            self.model.load_state_dict(state_dict)
            self.model.eval()  # 设置为评估模式（非常重要！） # [cite: 1]
            self.current_model_name = name
            # **关键修改：设置当前模型的转换**
            self.current_transform = TRANSFORMS.get(transform_type, cifar_transform) # 默认为 cifar 转换
            print(f"成功加载模型: {name}")
            print(f"使用 '{transform_type}' 类型的转换。")
            self.set_status(f"模型已加载: {name}")

        except Exception as e:
            messagebox.showerror("加载模型时出错", f"加载模型 '{name}' 失败：\n{e}")
            print(f"加载模型 {name} 时出错: {e}")
            self.model = None
            self.current_model_name = None
            self.current_transform = None # 清空转换
            self.set_status(f"错误：加载 {name} 失败")


    def load_image(self): # [cite: 1]
        """打开文件对话框让用户选择图片，并显示预览"""
        path = filedialog.askopenfilename(
            title="选择一个图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.gif"), ("所有文件", "*.*")]
        )
        if not path: # 用户取消选择
            return

        self.img_path = path
        self.set_status(f"已选择图片: {os.path.basename(path)}")
        try:
            img = Image.open(path).convert('RGB') # 确保是 RGB 格式 [cite: 1]

            # 创建预览图 (保持比例缩放)
            img_copy = img.copy()
            img_copy.thumbnail((300, 300)) # 限制预览图最大尺寸为 300x300 [cite: 1]
            self.photo = ImageTk.PhotoImage(img_copy)

            # 更新图片标签
            self.image_label.config(image=self.photo, text="") # 显示图片，清空文字 [cite: 1]
            self.image_label.image = self.photo # 保持引用，防止被垃圾回收 [cite: 1]

            # 重置分类结果
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

    def classify(self): # [cite: 1]
        """使用当前加载的模型对选定的图片进行分类"""
        if not self.img_path:
            messagebox.showwarning("操作无效", "请先点击“选择图片”按钮选择一张图片。")
            return

        # 检查模型和转换是否已加载
        if not self.model or not self.current_transform:
            # 尝试重新加载当前选中的模型（这也会设置转换）
            self.load_model()
            if not self.model or not self.current_transform: # 如果加载仍然失败
                 messagebox.showwarning("操作无效", f"模型 '{self.model_var.get()}' 或其转换未能成功加载。\n请检查配置或选择其他模型。")
                 return

        # 确保选择的模型和当前加载的模型一致
        if self.model_var.get() != self.current_model_name:
             messagebox.showwarning("模型不匹配", f"当前选择的模型是 '{self.model_var.get()}'，\n但加载的模型是 '{self.current_model_name}'。\n\n模型未自动切换，请重新点击“开始分类”或重新选择模型。")
             # 加载选定的模型以确保一致性
             print("检测到模型不匹配，正在加载选定的模型...")
             self.load_model()
             if not self.model or not self.current_transform: return # 如果加载失败则退出
             # return # 或者直接返回，让用户再次点击分类

        self.set_status(f"正在使用 {self.current_model_name} 对图片进行分类...")
        try:
            # 1. 加载并预处理图像
            img = Image.open(self.img_path).convert('RGB')
            # **关键修改：使用当前模型对应的转换**
            input_tensor = self.current_transform(img) # 应用正确的预处理 [cite: 1]
            input_batch = input_tensor.unsqueeze(0) # 创建一个 batch (添加 batch 维度) [cite: 1]
            input_batch = input_batch.to(device) # 将数据移动到正确的设备 [cite: 1]

            # 2. 模型推理
            with torch.no_grad(): # 关闭梯度计算，节省内存和计算 [cite: 1]
                outputs = self.model(input_batch)
                # outputs 的形状通常是 [batch_size, num_classes]

            # 3. 获取预测结果
            probabilities = torch.softmax(outputs, dim=1) # 转换为概率 [cite: 1]
            confidence, predicted_idx = torch.max(probabilities, 1) # 获取最高概率和对应的索引 [cite: 1]

            predicted_class_index = predicted_idx.item() # 从 tensor 中提取索引值 [cite: 1]
            predicted_class_name = CLASS_NAMES[predicted_class_index] # 获取类别名称 [cite: 1]
            confidence_score = confidence.item() * 100 # 获取置信度百分比 [cite: 1]

            # 4. 更新结果标签
            result_text = f"分类结果：{predicted_class_name}\n置信度：{confidence_score:.2f}%" # [cite: 1]
            self.result_label.config(text=result_text)
            print(f"分类完成: {predicted_class_name} ({confidence_score:.2f}%)")
            self.set_status("分类完成")

        except Exception as e:
            messagebox.showerror("分类时出错", f"对图片进行分类时发生错误：\n{e}")
            print(f"分类时出错: {e}")
            self.result_label.config(text="分类结果：[分类失败]")
            self.set_status("错误：分类失败")


if __name__ == '__main__':
    # 确保 MODEL_CONFIG 不为空再启动 GUI
    if not MODEL_CONFIG:
        print("错误：没有在 MODEL_CONFIG 中配置任何模型。请编辑 gui.py 文件。")
    else:
        app = ClassifierGUI()
        app.mainloop() # [cite: 1]