import tarfile
import pickle
import numpy as np
# import matplotlib.pyplot as plt # 注释掉，此处不用于绘制曲线
from sklearn.preprocessing import StandardScaler, normalize
# from sklearn.svm import SVC # 使用 LinearSVC 以提高速度
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from tqdm import tqdm # 用于进度条
import random
from sklearn.svm import LinearSVC
import os
import time

# *** 修改: CIFAR-100 类别名称列表 (可选, 训练本身不需要，但最好有) ***
# 这里使用英文细粒度名称作为占位符
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# --- 数据加载 ---
# *** 修改: 重写加载函数以适配 CIFAR-100 ***
def load_cifar100(file_path):
    """从 tar.gz 文件加载 CIFAR-100 数据。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未在以下位置找到 CIFAR-100 数据文件: {file_path}\n"
                              "请从 https://www.cs.toronto.edu/~kriz/cifar.html 下载 "
                              "并将 cifar-100-python.tar.gz 放置在脚本目录下。")

    print(f"正在从以下位置加载 CIFAR-100 数据: {file_path}")
    with tarfile.open(file_path, 'r:gz') as tar:
        # 加载训练数据
        train_member_name = 'cifar-100-python/train'
        print(f"  正在提取 {train_member_name}...")
        train_file = tar.extractfile(train_member_name)
        if train_file is None:
            raise FileNotFoundError(f"在 tar 归档文件中未找到训练文件 {train_member_name}。")
        train_batch = pickle.load(train_file, encoding='bytes')
        # 使用 'data' 和 'fine_labels' 键 (bytes 类型)
        train_data = train_batch[b'data']
        train_labels = np.array(train_batch[b'fine_labels']) # 使用细粒度标签
        train_file.close()

        # 加载测试数据
        test_member_name = 'cifar-100-python/test'
        print(f"  正在提取 {test_member_name}...")
        test_file = tar.extractfile(test_member_name)
        if test_file is None:
             raise FileNotFoundError(f"在 tar 归档文件中未找到测试文件 {test_member_name}。")
        test_batch = pickle.load(test_file, encoding='bytes')
        # 使用 'data' 和 'fine_labels' 键 (bytes 类型)
        test_data = np.array(test_batch[b'data'])
        test_labels = np.array(test_batch[b'fine_labels']) # 使用细粒度标签
        test_file.close()

        # (可选) 加载元数据以获取类别名称
        # meta_member_name = 'cifar-100-python/meta'
        # meta_file = tar.extractfile(meta_member_name)
        # meta_data = pickle.load(meta_file, encoding='bytes')
        # fine_label_names = [t.decode('utf8') for t in meta_data[b'fine_label_names']]
        # meta_file.close()

    print("CIFAR-100 数据加载成功。")
    # 返回训练数据、训练标签、测试数据、测试标签
    return train_data, train_labels, test_data, test_labels #, fine_label_names (如果需要)

# --- 特征提取函数 (保持不变) ---
def extract_patches(images, patch_size, num_patches_per_image, max_total_patches=None):
    """从图像中提取随机补丁。"""
    patches = []
    total_extracted = 0
    # 计算要提取的总补丁数，如果设置了 max_total_patches 则遵守限制
    total_target_patches = len(images) * num_patches_per_image
    if max_total_patches is not None:
        total_target_patches = min(total_target_patches, max_total_patches)
        # 如果受 max_total_patches 限制，则调整 num_patches_per_image
        num_patches_per_image = max(1, total_target_patches // len(images))


    print(f"每张图像提取约 {num_patches_per_image} 个补丁 (最大总数: {total_target_patches})...")
    for img in tqdm(images, desc='提取补丁'): # Progress bar description in Chinese
        img = img.reshape(3, 32, 32).transpose(1, 2, 0) # (32, 32, 3)
        img_patches_extracted = 0
        # 尝试提取补丁，直到达到每张图像的限制或总限制
        while img_patches_extracted < num_patches_per_image and (max_total_patches is None or total_extracted < max_total_patches):
            if 32 - patch_size < 0: # 确保补丁大小有效
                 raise ValueError("补丁大小大于图像尺寸。")
            x = random.randint(0, 32 - patch_size)
            y = random.randint(0, 32 - patch_size)
            patch = img[y:y + patch_size, x:x + patch_size, :].flatten()
            patches.append(patch)
            img_patches_extracted += 1
            total_extracted += 1
        if max_total_patches is not None and total_extracted >= max_total_patches:
            break # 如果达到总限制则停止

    print(f"总共提取了 {len(patches)} 个补丁。")
    return np.array(patches)

def create_visual_dictionary(patches, num_features, random_state=42):
    """使用 K-means 创建视觉词典。"""
    print(f"\n正在预处理 {len(patches)} 个补丁 (StandardScaler + PCA)...")
    start_time = time.time()
    scaler = StandardScaler()
    patches_scaled = scaler.fit_transform(patches)

    # PCA 用于降维和白化
    # n_components=0.95 保留 95% 的方差，可根据需要调整
    pca = PCA(whiten=True, n_components=0.95, random_state=random_state)
    patches_pca = pca.fit_transform(patches_scaled)
    print(f"  PCA 将维度降至: {patches_pca.shape[1]}")
    print(f"  预处理时间: {time.time() - start_time:.2f}秒")


    print(f"\n正在运行 K-means，聚类数量为 {num_features}...")
    start_time = time.time()
    # 对于大型数据集，使用 MiniBatchKMeans 可能更快
    # from sklearn.cluster import MiniBatchKMeans
    # kmeans = MiniBatchKMeans(n_clusters=num_features, random_state=random_state, n_init='auto', batch_size=1024*4)
    kmeans = KMeans(n_clusters=num_features, random_state=random_state, n_init='auto', verbose=0) # n_init='auto' 在较新 sklearn 中是默认值
    kmeans.fit(patches_pca) # 在 PCA 转换后的数据上拟合
    print(f"  K-means 聚类时间: {time.time() - start_time:.2f}秒")

    return kmeans, scaler, pca

def extract_bovw_features(images, patch_size, kmeans, scaler, pca):
    """为一组图像提取视觉词袋 (BoVW) 特征。"""
    num_features = kmeans.n_clusters
    features = []
    print(f"\n正在提取 BoVW 特征 (词典大小: {num_features})...")
    for img in tqdm(images, desc='提取 BoVW'): # Progress bar description in Chinese
        img = img.reshape(3, 32, 32).transpose(1, 2, 0) # (32, 32, 3)
        img_patches = []
        # 提取密集补丁（滑动窗口）
        for y in range(0, 32 - patch_size + 1):
            for x in range(0, 32 - patch_size + 1):
                patch = img[y:y + patch_size, x:x + patch_size, :].flatten()
                img_patches.append(patch)

        if not img_patches: # 处理未提取到补丁的边缘情况
            features.append(np.zeros(num_features))
            continue

        img_patches = np.array(img_patches)
        # 应用与创建词典时相同的预处理
        img_patches_scaled = scaler.transform(img_patches)
        img_patches_pca = pca.transform(img_patches_scaled)

        # 将补丁分配给最近的聚类中心（视觉词）
        words = kmeans.predict(img_patches_pca)

        # 创建视觉词的直方图
        histogram, _ = np.histogram(words, bins=np.arange(num_features + 1))

        # 归一化直方图（L1 或 L2 范数）- L2 通常效果不错
        # histogram = normalize(histogram.reshape(1, -1), norm='l1')[0]
        histogram = normalize(histogram.reshape(1, -1), norm='l2')[0]
        features.append(histogram)

    return np.array(features)

# --- SVM 训练与评估 (保持不变) ---
def train_and_evaluate_svm(train_features, train_labels, test_features, test_labels, C=1.0, max_iter=5000):
    """训练 LinearSVC 并进行评估。"""
    print(f"\n正在训练 LinearSVC (C={C}, max_iter={max_iter})...")
    start_time = time.time()
    # 对于大型数据集，LinearSVC 通常比 SVC(kernel='linear') 更快
    svm = LinearSVC(C=C, max_iter=max_iter, random_state=42, tol=1e-4, dual='auto') # dual='auto' 在新版本中推荐
    # 如果特征数量非常大，可以考虑 dual=False

    svm.fit(train_features, train_labels)
    print(f"  SVM 训练时间: {time.time() - start_time:.2f}秒")

    print("\n正在测试集上评估 SVM...")
    start_time = time.time()
    predictions = svm.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"  SVM 评估时间: {time.time() - start_time:.2f}秒")
    print(f"\n最终测试集准确率: {accuracy * 100:.2f}%")
    return accuracy


# ========== 主执行代码块 ==========
if __name__ == '__main__':

    print("正在运行 SVM.py (BoVW + SVM) for CIFAR-100...") # <-- 修改
    start_overall_time = time.time()

    # --- 配置参数 ---
    # *** 修改: 更新 CIFAR 文件名 ***
    CIFAR_FILE = 'cifar-100-python.tar.gz' # <--- 修改
    # 以下参数可能需要为 CIFAR-100 调整
    PATCH_SIZE = 6           # 补丁大小 (例如 6x6 或 8x8)
    NUM_PATCHES_PER_IMAGE = 20 # 用于词典的每张图像的随机补丁数
    MAX_TOTAL_PATCHES = 200000 # 限制词典的总补丁数 (加速 KMeans)
    NUM_FEATURES = 1000      # 视觉词数量 (K-means 聚类数), 例如 500-4000
    SVM_C = 1.0              # SVM 正则化参数
    SVM_MAX_ITER = 5000      # SVM 求解器的最大迭代次数

    # --- 加载数据 ---
    # *** 修改: 调用新的加载函数 ***
    try:
        train_data, train_labels, test_data, test_labels = load_cifar100(CIFAR_FILE) # <--- 修改
        # 可选：使用子集进行更快的测试/调试
        # print("正在使用数据子集以提高速度...")
        # subset_size = 5000
        # train_data, train_labels = train_data[:subset_size], train_labels[:subset_size]
        # test_data, test_labels = test_data[:subset_size//5], test_labels[:subset_size//5]
    except FileNotFoundError as e:
        print(e)
        exit()
    except Exception as e:
        print(f"数据加载过程中发生错误: {e}") # Error message in Chinese
        exit()

    # --- 创建视觉词典 (保持不变) ---
    # 1. 从训练数据中提取随机补丁
    train_patches = extract_patches(train_data, PATCH_SIZE, NUM_PATCHES_PER_IMAGE, MAX_TOTAL_PATCHES)

    # 2. 预处理补丁并运行 K-means
    if len(train_patches) > 0:
        kmeans, scaler, pca = create_visual_dictionary(train_patches, NUM_FEATURES)
    else:
        print("错误：未提取到补丁。无法创建视觉词典。") # Error message in Chinese
        exit()

    # --- 提取 BoVW 特征 (保持不变) ---
    # 3. 为训练集提取特征
    train_features = extract_bovw_features(train_data, PATCH_SIZE, kmeans, scaler, pca)

    # 4. 为测试集提取特征
    test_features = extract_bovw_features(test_data, PATCH_SIZE, kmeans, scaler, pca)

    # --- 训练和评估分类器 (保持不变) ---
    # 5. 训练 SVM 并评估
    final_accuracy = train_and_evaluate_svm(train_features, train_labels, test_features, test_labels, C=SVM_C, max_iter=SVM_MAX_ITER)

    end_overall_time = time.time()
    print(f"\n总执行时间: {(end_overall_time - start_overall_time) / 60:.2f} 分钟")
    print("脚本执行完毕.")
