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
import time # 用于计时

# --- 可复现性 ---
def set_seed(seed=42):
    """设置随机种子以保证结果的可重复性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 确保确定性行为（可能会影响性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# --- 模型定义 ---
class SimpleCNN(nn.Module):
    """一个用于 CIFAR-10 的简单 CNN 模型。"""
    def __init__(self, num_classes=10): # 添加 num_classes 参数
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # 添加了 BatchNorm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) # 添加了 BatchNorm
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) # 添加了 BatchNorm
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True) # 使用 inplace ReLU
        # 动态计算展平后的大小（假设输入为 32x32）
        # Input -> Conv1 -> BN1 -> ReLU -> Pool -> (32 / 2) = 16x16
        # -> Conv2 -> BN2 -> ReLU -> Pool -> (16 / 2) = 8x8
        # -> Conv3 -> BN3 -> ReLU -> Pool -> (8 / 2) = 4x4
        self._fc_input_size = 128 * 4 * 4
        self.fc1 = nn.Linear(self._fc_input_size, 512)
        self.dropout = nn.Dropout(0.5) # 添加了 Dropout
        self.fc2 = nn.Linear(512, num_classes)

        # 可选：初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self._fc_input_size) # 展平
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # 应用 dropout
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# --- 训练与评估函数 ---
def train_one_epoch(model, trainloader, criterion, optimizer, device):
    """训练模型一个 epoch。"""
    model.train() # 设置模型为训练模式
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
    # 使用英文打印日志，保持一致性
    print(f"  Train Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s")
    return epoch_loss

def evaluate(model, testloader, criterion, device):
    """在测试集上评估模型。"""
    model.eval() # 设置模型为评估模式
    all_labels = []
    all_preds = []
    running_loss = 0.0
    with torch.no_grad(): # 评估时无需计算梯度
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
    # 使用英文打印日志
    print(f"  Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

# --- 绘图函数 ---
def plot_history(history, model_name, save_dir='plots'):
    """绘制训练和验证损失及准确率曲线。"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # print(f"已创建目录: {save_dir}") # 改为英文或移除
        print(f"Created directory: {save_dir}")


    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation loss')
    plt.title(f'{model_name} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], 'go-', label='Validation accuracy')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_filename = os.path.join(save_dir, f'{model_name.lower()}_training_curves.png')
    plt.savefig(plot_filename)
    # print(f"训练曲线已保存至 {plot_filename}") # 改为英文
    print(f"Training curves saved to {plot_filename}")

    # plt.show() # 可选：显示图像

# --- 检查点函数 ---
def save_checkpoint(model, optimizer, epoch, history, filename):
    """保存模型检查点。"""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }
    torch.save(state, filename)
    # print(f"检查点已保存至 {filename}") # 改为英文
    print(f"Checkpoint saved to {filename}")


# ========== 主执行代码块 ==========
if __name__ == '__main__':

    print("正在运行 SimpleCNN.py 进行训练...") # User-facing message in Chinese
    set_seed(42) # 设置随机种子

    # --- 配置参数 ---
    MODEL_NAME = "SimpleCNN"
    NUM_EPOCHS = 50      # 可根据需要调整 (例如 50-100)
    BATCH_SIZE = 128     # 常用批次大小
    LEARNING_RATE = 0.001 # Adam 优化器的初始学习率
    WEIGHT_DECAY = 1e-4  # L2 正则化
    SAVE_INTERVAL = 10   # 每 N 个 epoch 保存一次检查点
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    PLOT_DIR = 'plots'
    FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}.pth') # 最终模型保存路径
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}_best.pth') # 最佳模型保存路径
    CSV_LOG_FILE = os.path.join(LOG_DIR, f'{MODEL_NAME.lower()}_training_log.csv') # CSV 日志文件路径

    # --- 创建所需目录 ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 数据预处理与加载 ---
    # 使用标准的 CIFAR-10 归一化值
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

    # 确保 ./data 目录存在
    if not os.path.exists('./data'):
        os.makedirs('./data')

    # 加载 CIFAR-10 数据集
    try:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    except Exception as e:
        print(f"下载或加载 CIFAR-10 数据集时出错: {e}") # Error message in Chinese
        exit()

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"数据已加载: {len(trainset)} 训练图像, {len(testset)} 测试图像.") 

    # --- 模型、设备、损失函数、优化器、调度器 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}') 

    model = SimpleCNN(num_classes=10).to(device)
    print(f"模型 '{MODEL_NAME}' 已初始化.") 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 可选：添加学习率调度器
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) # 监控验证损失

    # --- 训练循环 ---
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    best_val_accuracy = 0.0

    print(f"开始训练 {NUM_EPOCHS} 个 epochs...") 
    start_train_time = time.time()

    with open(CSV_LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Accuracy']) # CSV Header in English

        for epoch in range(1, NUM_EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---") 
            print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}") 

            train_loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, testloader, criterion, device)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            # 记录到 CSV
            writer.writerow([epoch, train_loss, val_loss, val_accuracy])
            file.flush() # 确保数据写入

            # 更新学习率调度器 (如果使用 ReduceLROnPlateau)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(val_loss)
            # elif scheduler is not None: # 对于其他调度器，如 CosineAnnealingLR
            #      scheduler.step()

            # 定期保存检查点
            if epoch % SAVE_INTERVAL == 0:
                 intermediate_ckpt_path = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}_epoch_{epoch}.pth')
                 save_checkpoint(model, optimizer, epoch, history, intermediate_ckpt_path)

            # 根据验证准确率保存最佳模型
            if val_accuracy > best_val_accuracy:
                print(f"  新的最佳验证准确率: {val_accuracy:.2f}% (之前: {best_val_accuracy:.2f}%)") 
                best_val_accuracy = val_accuracy
                # 立即保存最佳模型状态
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  最佳模型已保存至 {BEST_MODEL_PATH}") 


    total_train_time = time.time() - start_train_time
    print(f"\n--- 训练完成 ---") 
    print(f"总训练时间: {total_train_time // 60:.0f}分 {total_train_time % 60:.0f}秒") 
    print(f"最佳验证准确率: {best_val_accuracy:.2f}%") 

    # --- 保存最终模型并绘制历史曲线 ---
    # 保存最后一个 epoch 的模型状态
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"最终模型状态已保存至 {FINAL_MODEL_PATH}") 

    # 绘制训练历史曲线
    plot_history(history, MODEL_NAME, save_dir=PLOT_DIR)

    print("\n脚本执行完毕.") 
