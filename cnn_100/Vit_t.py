# train_vit_l_16_224.py
# ---------------------
# 用于在 CIFAR-10 数据集上训练 Vision Transformer (ViT) Large 模型 (ViT-L_16) 的脚本。
# 使用 torchvision 加载预训练的 ViT 模型，并对其进行微调。
# **警告**: ViT-Large 模型非常大，需要大量的 GPU 显存和计算资源。
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
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为 {seed}")

# --- 2. 模型定义 ---
def create_vit_l_16_model(num_classes=10, pretrained=True):
    """
    创建并加载 ViT-L_16 模型。
    - 使用 torchvision.models 加载模型。
    - 可选地加载 ImageNet 预训练权重 (强烈推荐)。
    - 修改最终的分类头以适应指定的类别数量 (num_classes)。

    Args:
        num_classes (int): 数据集中的类别数量 (例如 CIFAR-10 为 10)。
        pretrained (bool): 是否加载 ImageNet 预训练权重。

    Returns:
        torch.nn.Module: 配置好的 ViT-L_16 模型。
    """
    print(f"正在加载 ViT-L_16 模型 (预训练={pretrained})... (这可能需要一些时间)")
    # 使用推荐的权重 API
    weights = models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vit_l_16(weights=weights)

    # ViT 的分类头通常在 'heads.head'
    if hasattr(model, 'heads') and hasattr(model.heads, 'head'):
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        print(f"已将分类头替换为输出 {num_classes} 个类别。输入特征数: {num_ftrs}")
    else:
        # 备用逻辑，以防模型结构变化
        print("警告：无法自动找到 'heads.head'。可能需要手动调整模型结构。")
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
             return None
    return model

# --- 3. 训练与评估函数 ---
def train_one_epoch(model, trainloader, criterion, optimizer, device):
    """训练模型一个 epoch。"""
    model.train()
    running_loss = 0.0
    start_time = time.time()
    num_batches = len(trainloader)

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 可选梯度裁剪
        optimizer.step()
        running_loss += loss.item()

        # 调整打印频率，因为 ViT-L 的批次可能更少
        if (i + 1) % 20 == 0 or (i + 1) == num_batches:
             print(f'  Batch {i+1}/{num_batches}, 当前批次损失: {loss.item():.4f}')

    epoch_loss = running_loss / num_batches
    epoch_time = time.time() - start_time
    print(f"  Epoch 训练损失: {epoch_loss:.4f}, 耗时: {epoch_time:.2f}s")
    return epoch_loss

def evaluate(model, testloader, criterion, device):
    """在测试集上评估模型。"""
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
    num_batches = len(testloader)

    with torch.no_grad():
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
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='训练损失 (Training loss)')
    plt.plot(epochs, history['val_loss'], 'ro-', label='验证损失 (Validation loss)')
    plt.title(f'{model_name} - 训练和验证损失')
    plt.xlabel('Epochs (轮次)'); plt.ylabel('损失 (Loss)'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], 'go-', label='验证准确率 (Validation accuracy)')
    plt.title(f'{model_name} - 验证准确率')
    plt.xlabel('Epochs (轮次)'); plt.ylabel('准确率 (Accuracy %)'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f'{model_name.lower()}_training_curves.png')
    plt.savefig(plot_filename); print(f"训练曲线图已保存至: {plot_filename}")
    plt.close()

# --- 5. 检查点函数 ---
def save_checkpoint(model, optimizer, epoch, history, filename):
    """保存模型检查点（仅模型权重）。"""
    torch.save(model.state_dict(), filename)
    print(f"检查点 (仅模型权重) 已保存至: {filename}")

# --- 6. 主执行代码块 ---
if __name__ == '__main__':

    print("="*30)
    print("开始执行 ViT-L_16_224 训练脚本...")
    print("**警告**: ViT-Large 模型需要大量 GPU 显存和计算时间。")
    print("="*30)
    set_seed(42)

    # --- 配置参数 ---
    MODEL_NAME = "ViT_L_16_224_CIFAR10" # 修改模型名称
    NUM_CLASSES = 10
    NUM_EPOCHS = 30                    # ViT-L 微调可能需要更少轮次，或更长时间
    BATCH_SIZE = 16                    # **极重要**: ViT-L 需要巨大显存，Batch Size 可能需要非常小 (例如 8, 16, 32)
    LEARNING_RATE = 1e-5               # Large 模型微调通常需要更小的学习率
    WEIGHT_DECAY = 0.01
    USE_PRETRAINED = True
    SAVE_INTERVAL = 5
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    PLOT_DIR = 'plots'

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
    IMG_SIZE = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    print(f"图像将调整至 {IMG_SIZE}x{IMG_SIZE} 并使用 ImageNet 归一化。")

    transform_train = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        # transforms.TrivialAugmentWide(), # 可选增强
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    transform_test = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    data_dir = './data'
    if not os.path.exists(data_dir): os.makedirs(data_dir); print(f"创建数据目录: {data_dir}")
    try:
        print(f"正在从 '{data_dir}' 加载或下载 CIFAR-10 数据集...")
        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        print("数据集加载成功。")
    except Exception as e:
        print(f"错误：下载或加载 CIFAR-10 数据集失败: {e}"); exit()

    # **注意 num_workers 和 batch_size**
    num_workers = min(os.cpu_count(), 2) # ViT-L 训练时减少 workers 数量可能有助于减少 CPU/内存瓶颈
    print(f"使用 {num_workers} 个 Dataloader workers。")
    print(f"**注意**: Batch Size 设置为 {BATCH_SIZE}。如果遇到显存不足 (OOM) 错误，请减小此值。")
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"数据加载器已创建。训练集大小: {len(trainset)}, 测试集大小: {len(testset)}")
    print(f"训练批次数/Epoch: {len(trainloader)}, 测试批次数/Epoch: {len(testloader)}")

    # --- 模型、设备、损失函数、优化器、调度器 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    if not torch.cuda.is_available():
        print("错误：未检测到 CUDA 设备。ViT-Large 无法在 CPU 上有效训练。")
        exit() # 在 CPU 上运行 ViT-L 几乎不可行

    # **加载 ViT-Large 模型**
    model = create_vit_l_16_model(num_classes=NUM_CLASSES, pretrained=USE_PRETRAINED)
    if model is None: print("错误：模型创建失败。"); exit()
    model = model.to(device)
    print(f"模型 '{MODEL_NAME}' 已成功初始化并移至 {device}。")

    try:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型可训练参数总量: {total_params / 1e6:.2f} M (百万)")
    except Exception as e:
        print(f"无法计算模型参数量: {e}")

    criterion = nn.CrossEntropyLoss()
    print("损失函数: CrossEntropyLoss")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"优化器: AdamW (学习率={LEARNING_RATE}, 权重衰减={WEIGHT_DECAY})")

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7) # 可选，尝试更低的 eta_min
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5) # 调整 StepLR 参数
    print(f"学习率调度器: StepLR (step_size=8, gamma=0.5)")

    # --- 训练循环 ---
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0

    print(f"\n{'='*15} 开始训练 {'='*15}")
    start_train_time = time.time()

    log_mode = 'a' if os.path.exists(CSV_LOG_FILE) else 'w'
    print(f"将以 '{log_mode}' 模式打开日志文件: {CSV_LOG_FILE}")

    try:
        with open(CSV_LOG_FILE, mode=log_mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if log_mode == 'w':
                writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy', 'Learning Rate'])

            for epoch in range(1, NUM_EPOCHS + 1):
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
                print(f"当前学习率: {current_lr:.8f}")

                train_loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
                val_loss, val_accuracy = evaluate(model, testloader, criterion, device)

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)

                writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_accuracy:.2f}", f"{current_lr:.8f}"])
                file.flush()

                if scheduler:
                     scheduler.step()

                if epoch % SAVE_INTERVAL == 0:
                     intermediate_ckpt_path = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}_epoch_{epoch}.pth')
                     save_checkpoint(model, optimizer, epoch, history, intermediate_ckpt_path)

                if val_accuracy > best_val_accuracy:
                    print(f"  *** 新的最佳验证准确率: {val_accuracy:.2f}% (优于之前的 {best_val_accuracy:.2f}%) ***")
                    best_val_accuracy = val_accuracy
                    torch.save(model.state_dict(), BEST_MODEL_PATH)
                    print(f"  最佳模型权重已保存至: {BEST_MODEL_PATH}")

    except torch.cuda.OutOfMemoryError:
         print("\n！！！错误：GPU 显存不足 (CUDA Out of Memory)！！！")
         print(f"当前的 Batch Size 为 {BATCH_SIZE}。请尝试减小 Batch Size 并重新运行脚本。")
         # 可以在这里添加清理 GPU 缓存的代码（但不一定总能解决根本问题）
         # torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n！！！训练过程中发生错误: {e}！！！")
    finally:
        total_train_time = time.time() - start_train_time
        print(f"\n{'='*15} 训练结束 {'='*15}")
        print(f"总训练时间: {total_train_time // 60:.0f} 分 {total_train_time % 60:.0f} 秒 ({total_train_time:.2f} 秒)")
        print(f"记录的最佳验证准确率: {best_val_accuracy:.2f}%")

        # --- 保存最终模型并绘制历史曲线 ---
        try:
            torch.save(model.state_dict(), FINAL_MODEL_PATH)
            print(f"最终模型权重已保存至: {FINAL_MODEL_PATH}")
        except Exception as e:
            print(f"错误：保存最终模型失败: {e}")

        if history['train_loss']:
             try:
                 plot_history(history, MODEL_NAME, save_dir=PLOT_DIR)
             except Exception as e:
                 print(f"错误：绘制训练曲线失败: {e}")
        else:
            print("未记录训练历史，无法绘制曲线。")

        print("\n脚本执行完毕。")