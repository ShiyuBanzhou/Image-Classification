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
import torch.nn.functional as F # PreActResNet 需要

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

# --- 模型定义 (PreActResNet Bottleneck Block 和 Model - 减少通道数) ---
class PreActBottleneckSmall(nn.Module): # 重命名类以清晰区分
    '''具有减少通道数的 Bottleneck 模块的预激活版本。'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckSmall, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
                 ,nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)
        out = self.relu3(self.bn3(out))
        out = self.conv3(out)
        out += shortcut
        return out


class PreActResNetSmall(nn.Module): # 重命名类以清晰区分
    """具有减少通道数的预激活 ResNet 模型。"""
    def __init__(self, block, num_blocks, num_classes=10, base_planes=16): # 添加 base_planes
        super(PreActResNetSmall, self).__init__()
        self.in_planes = base_planes # 以较少的 planes 开始

        # 具有减少输出通道数的初始卷积层
        self.conv1 = nn.Conv2d(3, base_planes, kernel_size=3, stride=1, padding=1, bias=False)

        # 根据 base_planes 调整每层的 planes
        self.layer1 = self._make_layer(block, base_planes,     num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_planes*2,   num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_planes*4,   num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_planes*8,   num_blocks[3], stride=2)

        # 最终层 - 输入通道数取决于最后一层的输出
        final_planes = base_planes * 8 * block.expansion
        self.bn_final = nn.BatchNorm2d(final_planes)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(final_planes, num_classes)

        self._initialize_weights() # 初始化权重

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu_final(self.bn_final(out)) # 最终的 BN 和 ReLU
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

    def _initialize_weights(self):
        print("Initializing weights...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 nn.init.normal_(m.weight, 0, 0.01)
                 nn.init.constant_(m.bias, 0)

# 保留原始名称以便与 GUI 配置兼容（如果需要），
# 但内部使用更具描述性的类名。
# 注意：这个函数现在创建的是通道数减少的版本
def PreActResNet50(num_classes=10):
    """构建一个用于 CIFAR-10 的具有减少通道数的 PreActResNet-50 模型。"""
    # base_planes=16 使得 layer1 输出 16*4=64, layer2 输出 32*4=128, 等等。
    # 原始 PreResNet50 以 64 planes 开始。
    return PreActResNetSmall(PreActBottleneckSmall, [3, 4, 6, 3], num_classes=num_classes, base_planes=16)

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

    print("正在运行 PreActResNet50_small.py 进行训练...")
    set_seed(42)

    # --- 配置参数 ---
    # 使用一个独特的名称用于日志/绘图/检查点
    MODEL_NAME = "PreActResNet50Small"
    NUM_EPOCHS = 150     # 可根据需要调整
    BATCH_SIZE = 128
    LEARNING_RATE = 0.1
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    SAVE_INTERVAL = 10
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    PLOT_DIR = 'plots'
    # 使用独特的名称保存，但如果 GUI 需要，可以链接 PreActResNet50_少通道数.pth 到它
    FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}.pth')
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'{MODEL_NAME.lower()}_best.pth')
    # 确保 GUI 配置指向正确的文件名（如果直接使用此文件）
    # 示例: 'PreActResNet50_少通道数': (PreActResNet50, 'checkpoints/preactresnet50small.pth'),
    GUI_COMPATIBLE_NAME = 'preactresnet50_small.pth' # 原始 GUI 配置期望的名称
    GUI_COMPATIBLE_PATH = os.path.join(CHECKPOINT_DIR, GUI_COMPATIBLE_NAME)

    CSV_LOG_FILE = os.path.join(LOG_DIR, f'{MODEL_NAME.lower()}_training_log.csv')

    # --- 创建所需目录 ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # --- 数据预处理与加载 ---
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
        print(f"下载或加载 CIFAR-10 数据集时出错: {e}"); exit() 

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print(f"数据已加载: {len(trainset)} 训练图像, {len(testset)} 测试图像.") 

    # --- 模型、设备、损失函数、优化器、调度器 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}') 

    # 调用工厂函数，该函数创建的是小型版本
    model = PreActResNet50(num_classes=10).to(device)
    print(f"模型 '{MODEL_NAME}' 已初始化.") 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

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

            scheduler.step() # 每个 epoch 后更新调度器

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
    # 使用独特的名称保存
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f"最终模型状态已保存至 {FINAL_MODEL_PATH}") 
    # 同时使用 GUI 配置期望的名称保存，以方便使用
    torch.save(model.state_dict(), GUI_COMPATIBLE_PATH)
    print(f"最终模型状态也已保存至 {GUI_COMPATIBLE_PATH} 以兼容 GUI") 


    plot_history(history, MODEL_NAME, save_dir=PLOT_DIR)
    print("\n脚本执行完毕.") 
