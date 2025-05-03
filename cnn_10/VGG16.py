import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
import os
import random
import numpy as np
import time
from torchvision import models # 导入 models

# --- 可复现性 ---
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

# --- 模型定义 ---
class VGG16CIFAR10(nn.Module):
    """为 CIFAR-10 调整的 VGG16 模型。"""
    def __init__(self, num_classes=10, dropout_prob=0.5, pretrained=False): # 添加 pretrained 选项
        super(VGG16CIFAR10, self).__init__()
        # 加载 VGG16 特征提取器（可选预训练权重）
        vgg16_features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features

        self.features = vgg16_features
        # CIFAR-10 图像是 32x32。经过 VGG16 特征层（5个最大池化）后，尺寸变为 1x1
        # VGG16 特征层输出 512 个通道。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 使用 AdaptiveAvgPool2d
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096), # 调整了输入尺寸
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(4096, num_classes),
        )
        if not pretrained:
            self._initialize_weights() # 如果不使用预训练权重则进行初始化

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平输出
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        print("Initializing weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # 为完整性添加（如果使用了BN）
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# --- 训练与评估函数 (与 SimpleCNN 版本相同) ---
def train_one_epoch(model, trainloader, criterion, optimizer, device):
    """训练模型一个 epoch。"""
    model.train()
    running_loss = 0.0
    start_time = time.time()
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(trainloader)
    epoch_time = time.time() - start_time
    print(f"  Train Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
    return epoch_loss

def evaluate(model, testloader, criterion, device):
    """在测试集上评估模型。"""
    model.eval()
    all_labels = []
    all_preds = []
    running_loss = 0.0
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
    avg_loss = running_loss / len(testloader)
    print(f"  Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# --- 绘图函数 (与 SimpleCNN 版本相同) ---
def plot_history(history, model_name, save_dir='plots'):
    """绘制训练和验证损失及准确率曲线。"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir); print(f"Created directory: {save_dir}")
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], 'go-', label='Validation accuracy')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f'{model_name.lower()}_training_curves.png')
    plt.savefig(plot_filename); print(f"Training curves saved to {plot_filename}")

# --- 检查点函数 (与 SimpleCNN 版本相同) ---
def save_checkpoint(model, optimizer, epoch, history, filename):
    """保存模型检查点。"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }
    torch.save(state, filename); print(f"Checkpoint saved to {filename}")


# ========== 主执行代码块 ==========
if __name__ == '__main__':

    print("正在运行 VGG16.py 进行训练...") # User-facing message in Chinese
    set_seed(42)

    # --- 配置参数 ---
    MODEL_NAME = "VGG16"
    NUM_EPOCHS = 200     # VGG 通常需要更多 epochs，可调整
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01 # SGD 常用学习率，或使用 Adam
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9       # SGD 需要
    USE_PRETRAINED = False # 设置为 True 使用 ImageNet 权重 (可能需要调整学习率)
    SAVE_INTERVAL = 10
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

    # --- 数据预处理与加载 ---
    # 标准 CIFAR-10 转换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    if not os.path.exists('./data'): os.makedirs('./data')
    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    except Exception as e:
        print(f"下载或加载 CIFAR-10 数据集时出错: {e}"); exit() # Error message in Chinese

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"数据已加载: {len(trainset)} 训练图像, {len(testset)} 测试图像.") 

    # --- 模型、设备、损失函数、优化器、调度器 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}') 

    model = VGG16CIFAR10(num_classes=10, pretrained=USE_PRETRAINED).to(device)
    print(f"模型 '{MODEL_NAME}' 已初始化 (预训练={USE_PRETRAINED}).") 

    criterion = nn.CrossEntropyLoss()
    # VGG 通常使用 SGD
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # 使用余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # --- 训练循环 ---
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0

    print(f"开始训练 {NUM_EPOCHS} 个 epochs...") 
    start_train_time = time.time()

    with open(CSV_LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy']) # Header

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---") 
            print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}") 

            train_loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, testloader, criterion, device)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            writer.writerow([epoch, train_loss, val_loss, val_accuracy])
            file.flush()

            # 更新调度器
            scheduler.step()

            if epoch % SAVE_INTERVAL == 0:
                 intermediate_ckpt_path = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}_epoch_{epoch}.pth')
                 save_checkpoint(model, optimizer, epoch, history, intermediate_ckpt_path)

            if val_accuracy > best_val_accuracy:
                print(f"  新的最佳验证准确率: {val_accuracy:.2f}% (之前: {best_val_accuracy:.2f}%)") 
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  最佳模型已保存至 {BEST_MODEL_PATH}") 

    total_train_time = time.time() - start_train_time
    print(f"\n--- 训练完成 ---") 
    print(f"总训练时间: {total_train_time // 60:.0f}分 {total_train_time % 60:.0f}秒") 
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%") 

    # --- 保存最终模型并绘制历史曲线 ---
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"最终模型状态已保存至 {FINAL_MODEL_PATH}") 
    plot_history(history, MODEL_NAME, save_dir=PLOT_DIR)
    print("\n脚本执行完毕.") 
