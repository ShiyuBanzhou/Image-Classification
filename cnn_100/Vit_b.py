# train_vit_b_16_224.py
# ---------------------
# 用于在 CIFAR-10 数据集上训练 Vision Transformer (ViT) Base 模型 (ViT-B_16) 的脚本。
# 使用 torchvision 加载预训练的 ViT 模型，并对其进行微调。
# ---------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models # 导入 models 以访问预训练模型
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
import time

# --- 1. 可复现性设置 ---
def set_seed(seed=42):
    """设置随机种子以保证结果的可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 为了完全可复现，有时需要牺牲性能，可以取消注释下面两行
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为 {seed}")

# --- 2. 模型定义 ---
def create_vit_b_16_model(num_classes=10, pretrained=True):
    """
    创建并加载 ViT-B_16 模型。
    - 使用 torchvision.models 加载模型。
    - 可选地加载 ImageNet 预训练权重 (推荐)。
    - 修改最终的分类头以适应指定的类别数量 (num_classes)。

    Args:
        num_classes (int): 数据集中的类别数量 (例如 CIFAR-10 为 10)。
        pretrained (bool): 是否加载 ImageNet 预训练权重。

    Returns:
        torch.nn.Module: 配置好的 ViT-B_16 模型。
    """
    print(f"正在加载 ViT-B_16 模型 (预训练={pretrained})...")
    # 使用推荐的权重 API (torchvision >= 0.13)
    weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_b_16(weights=weights)

    # ViT 的分类头通常在 'heads.head'
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        # 获取原始分类头的输入特征数
        num_ftrs = model.heads.head.in_features
        # 替换为新的线性层
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        print(f"已将分类头替换为输出 {num_classes} 个类别。输入特征数: {num_ftrs}")
    else:
        # 如果结构不同 (例如旧版 torchvision)，可能需要调整
        print("警告：无法自动找到 'heads.head'。可能需要手动调整模型结构以替换分类头。")
        # 尝试查找常见的 'fc' 或 'classifier'
        if hasattr(model, 'fc'):
             num_ftrs = model.fc.in_features
             model.fc = nn.Linear(num_ftrs, num_classes)
             print(f"已将 'fc' 层替换为输出 {num_classes} 个类别。")
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
             num_ftrs = model.classifier.in_features
             model.classifier = nn.Linear(num_ftrs, num_classes)
             print(f"已将 'classifier' 层替换为输出 {num_classes} 个类别。")
        else:
             print("错误：无法找到要替换的分类头。")
             return None # 或者抛出错误

    return model

# --- 3. 训练与评估函数 ---
def train_one_epoch(model, trainloader, criterion, optimizer, device):
    """训练模型一个 epoch。"""
    model.train() # 设置模型为训练模式
    running_loss = 0.0
    start_time = time.time()
    num_batches = len(trainloader)

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 可选：梯度裁剪（防止梯度爆炸，对某些 Transformer 训练有帮助）
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 更新权重
        optimizer.step()

        running_loss += loss.item()

        # 打印批次进度 (可选)
        if (i + 1) % 100 == 0 or (i + 1) == num_batches:
             print(f'  Batch {i+1}/{num_batches}, 当前批次损失: {loss.item():.4f}')

    epoch_loss = running_loss / num_batches
    epoch_time = time.time() - start_time
    print(f"  Epoch 训练损失: {epoch_loss:.4f}, 耗时: {epoch_time:.2f}s")
    return epoch_loss

def evaluate(model, testloader, criterion, device):
    """在测试集上评估模型。"""
    model.eval() # 设置模型为评估模式
    all_labels = []
    all_preds = []
    running_loss = 0.0
    num_batches = len(testloader)

    with torch.no_grad(): # 评估时不需要计算梯度
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    avg_loss = running_loss / num_batches
    print(f"  验证损失: {avg_loss:.4f}, 准确率: {accuracy:.2f}%")
    return avg_loss, accuracy

# --- 4. 绘图函数 ---
def plot_history(history, model_name, save_dir='plots'):
    """绘制训练和验证损失及准确率曲线。"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir); print(f"创建目录: {save_dir}")

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='训练损失 (Training loss)')
    plt.plot(epochs, history['val_loss'], 'ro-', label='验证损失 (Validation loss)')
    plt.title(f'{model_name} - 训练和验证损失')
    plt.xlabel('Epochs (轮次)')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], 'go-', label='验证准确率 (Validation accuracy)')
    plt.title(f'{model_name} - 验证准确率')
    plt.xlabel('Epochs (轮次)')
    plt.ylabel('准确率 (Accuracy %)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # 调整布局防止重叠
    plot_filename = os.path.join(save_dir, f'{model_name.lower()}_training_curves.png')
    plt.savefig(plot_filename)
    print(f"训练曲线图已保存至: {plot_filename}")
    plt.close() # 关闭图形，防止在某些环境中自动显示

# --- 5. 检查点函数 ---
def save_checkpoint(model, optimizer, epoch, history, filename):
    """
    保存模型检查点。
    当前仅保存模型的状态字典 (model.state_dict()) 以节省空间。
    如果需要恢复训练，需要保存 optimizer 状态和 history。
    """
    # 仅保存模型权重
    torch.save(model.state_dict(), filename)
    print(f"检查点 (仅模型权重) 已保存至: {filename}")

    # # 如果需要保存完整状态以供恢复训练:
    # state = {
    #     'epoch': epoch,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'history': history, # 注意 history 可能会变得很大
    # }
    # torch.save(state, filename)
    # print(f"完整检查点已保存至: {filename}")


# --- 6. 主执行代码块 ---
if __name__ == '__main__':

    print("="*30)
    print("开始执行 ViT-B_16_224 训练脚本...")
    print("="*30)
    set_seed(42)

    # --- 配置参数 ---
    MODEL_NAME = "ViT_B_16_224_CIFAR10" # 用于日志、绘图和模型文件的名称
    NUM_CLASSES = 10                   # CIFAR-10 有 10 个类别
    NUM_EPOCHS = 50                    # 训练轮数 (ViT 微调通常不需要太多轮次)
    BATCH_SIZE = 64                    # **重要**: 根据 GPU 显存调整此值！ViT 需要较大显存。
    LEARNING_RATE = 3e-5               # 微调 ViT 常用的较小学习率 (AdamW)
    WEIGHT_DECAY = 0.01                # AdamW 优化器的权重衰减
    USE_PRETRAINED = True              # 是否使用 ImageNet 预训练权重 (强烈推荐)
    SAVE_INTERVAL = 5                  # 每隔多少轮保存一次检查点
    CHECKPOINT_DIR = 'checkpoints'     # 保存检查点和最终模型的目录
    LOG_DIR = 'logs'                   # 保存训练日志 (CSV) 的目录
    PLOT_DIR = 'plots'                 # 保存训练曲线图的目录

    # 自动生成文件路径
    FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}.pth')
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}_best.pth')
    CSV_LOG_FILE = os.path.join(LOG_DIR, f'{MODEL_NAME.lower()}_training_log.csv')

    # --- 创建所需目录 ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    print(f"检查点目录: '{CHECKPOINT_DIR}'")
    print(f"日志目录: '{LOG_DIR}'")
    print(f"绘图目录: '{PLOT_DIR}'")

    # --- 数据预处理与加载 ---
    # ViT 通常在 224x224 图像上预训练，因此需要调整 CIFAR-10 图像大小
    # 归一化使用 ImageNet 的统计数据，因为模型是在 ImageNet 上预训练的
    IMG_SIZE = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    print(f"图像将调整至 {IMG_SIZE}x{IMG_SIZE} 并使用 ImageNet 归一化。")

    # 训练集数据增强
    transform_train = transforms.Compose([
        transforms.Resize(IMG_SIZE), # **调整大小是 ViT 的关键步骤**
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)), # 轻微的随机裁剪和缩放
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        # 可选：添加更强的增强，如 TrivialAugment 或 RandAugment
        # transforms.TrivialAugmentWide(),
        transforms.ToTensor(), # 转换为 Tensor
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std) # 归一化
    ])

    # 测试集转换 (通常只做 Resize 和归一化)
    transform_test = transforms.Compose([
        transforms.Resize(IMG_SIZE), # **调整大小**
        # transforms.CenterCrop(IMG_SIZE), # 如果需要严格的 224x224，可以添加 CenterCrop
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # 检查数据目录是否存在
    data_dir = './data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"创建数据目录: {data_dir}")

    # 加载 CIFAR-10 数据集
    try:
        print(f"正在从 '{data_dir}' 加载或下载 CIFAR-10 数据集...")
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        print("数据集加载成功。")
    except Exception as e:
        print(f"错误：下载或加载 CIFAR-10 数据集失败: {e}")
        print("请检查网络连接或 '{data_dir}' 目录的权限。")
        exit()

    # 创建 DataLoader
    # num_workers 可以根据你的 CPU 核心数调整，pin_memory=True 在 GPU 训练时通常能加速
    num_workers = min(os.cpu_count(), 4) # 使用最多 4 个 worker
    print(f"使用 {num_workers} 个 Dataloader workers。")
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"数据加载器已创建。训练集大小: {len(trainset)}, 测试集大小: {len(testset)}")
    print(f"训练批次数/Epoch: {len(trainloader)}, 测试批次数/Epoch: {len(testloader)}")

    # --- 模型、设备、损失函数、优化器、调度器 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if not torch.cuda.is_available():
        print("警告：未检测到 CUDA 设备，将在 CPU 上运行。ViT 训练在 CPU 上会非常慢！")

    # 创建模型实例
    model = create_vit_b_16_model(num_classes=NUM_CLASSES, pretrained=USE_PRETRAINED)
    if model is None:
        print("错误：模型创建失败。")
        exit()
    model = model.to(device) # 将模型移动到设备
    print(f"模型 '{MODEL_NAME}' 已成功初始化并移至 {device}。")

    # 统计模型参数量 (可选)
    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型可训练参数总量: {total_params / 1e6:.2f} M (百万)")
    except Exception as e:
        print(f"无法计算模型参数量: {e}")


    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    print("损失函数: CrossEntropyLoss")

    # 定义优化器 (AdamW 通常用于 ViT)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"优化器: AdamW (学习率={LEARNING_RATE}, 权重衰减={WEIGHT_DECAY})")

    # 定义学习率调度器 (可选，但推荐)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6) # 备选：余弦退火
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # 每 10 个 epoch 学习率乘以 0.5
    print(f"学习率调度器: StepLR (step_size=10, gamma=0.5)")

    # --- 训练循环 ---
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0

    print(f"\n{'='*15} 开始训练 {'='*15}")
    start_train_time = time.time()

    # 检查并准备日志文件
    log_mode = 'a' if os.path.exists(CSV_LOG_FILE) else 'w'
    print(f"将以 '{log_mode}' 模式打开日志文件: {CSV_LOG_FILE}")

    try:
        with open(CSV_LOG_FILE, mode=log_mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 如果是新文件，写入标题行
            if log_mode == 'w':
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy', 'Learning Rate'])

            for epoch in range(1, NUM_EPOCHS + 1):
                current_lr = optimizer.param_groups[0]['lr'] # 获取当前学习率
                print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
                print(f"当前学习率: {current_lr:.8f}")

                # 训练一个 Epoch
                train_loss = train_one_epoch(model, trainloader, criterion, optimizer, device)

                # 评估模型
                val_loss, val_accuracy = evaluate(model, testloader, criterion, device)

                # 记录历史数据
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                # 写入 CSV 日志
                writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_accuracy:.2f}", f"{current_lr:.8f}"])
                file.flush() # 确保实时写入

                # 更新学习率调度器
                if scheduler:
                     scheduler.step()

                # 定期保存检查点
                if epoch % SAVE_INTERVAL == 0:
                     intermediate_ckpt_path = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}_epoch_{epoch}.pth')
                     save_checkpoint(model, optimizer, epoch, history, intermediate_ckpt_path)

                # 保存最佳模型 (基于验证准确率)
                if val_accuracy > best_val_accuracy:
                    print(f"  *** 新的最佳验证准确率: {val_accuracy:.2f}% (优于之前的 {best_val_accuracy:.2f}%) ***")
                    best_val_accuracy = val_accuracy
                    # 保存最佳模型的状态字典
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"  最佳模型权重已保存至: {BEST_MODEL_PATH}")

    except Exception as e:
        print(f"\n！！！训练过程中发生错误: {e}！！！")
        # 可以在这里添加更详细的错误处理或日志记录
    finally:
        total_train_time = time.time() - start_train_time
        print(f"\n{'='*15} 训练结束 {'='*15}")
        print(f"总训练时间: {total_train_time // 60:.0f} 分 {total_train_time % 60:.0f} 秒 ({total_train_time:.2f} 秒)")
        print(f"记录的最佳验证准确率: {best_val_accuracy:.2f}%")

        # --- 保存最终模型并绘制历史曲线 ---
        # 保存最后一个 epoch 的模型状态
        try:
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print(f"最终模型权重已保存至: {FINAL_MODEL_PATH}")
        except Exception as e:
            print(f"错误：保存最终模型失败: {e}")

        # 绘制训练历史曲线 (如果 history 不为空)
        if history['train_loss']:
             try:
                 plot_history(history, MODEL_NAME, save_dir=PLOT_DIR)
             except Exception as e:
                 print(f"错误：绘制训练曲线失败: {e}")
        else:
            print("未记录训练历史，无法绘制曲线。")

        print("\n脚本执行完毕。")
