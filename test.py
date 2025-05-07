import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import weibull_min
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import random
import torch.nn.functional as F
SEED = 42  # 设置随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)  # 固定 PyTorch CPU 计算的随机数种子
# 添加混淆矩阵和AUC-ROC相关函数
def plot_confusion_matrix(y_true, y_pred, classes, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    plt.show()

def plot_roc_curve(y_true, y_scores, num_classes, title="ROC Curve"):
    """绘制 ROC 曲线"""
    # 按类别计算 ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_scores[:, i])
        roc_auc[i] = roc_auc_score(y_true_binary, y_scores[:, i])

    # 绘制每个类别的 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

# 小型CNN模型（含CBAM注意力机制 + OpenMax支持）
class SmallCNNWithCAAndOpenMax(nn.Module):
    def __init__(self, num_classes=4, alpha=2):
        super(SmallCNNWithCAAndOpenMax, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.mean_vecs = None
        self.weibull_models = {}
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 32x32
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 16x16
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 8x8
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 4x4
        )
        # 注意力模块放在conv4之后
        self.ca = ChannelAttention(128)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2x2
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
    def activations_hook(self, grad):
        self.gradients = grad
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.ca(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # 注册hook获取梯度（用于Grad-CAM）
        x.requires_grad_()
        x.register_hook(self.activations_hook)
        self.activations = x
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.fc(x)
        return out

    def extract_features(self, x):
        """提取模型的特征向量"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.ca(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.normalize(x, dim=1)  # 标准化
        return x

    def fit_weibull(self, features, labels, tailsize=5):
        """拟合Weibull分布"""
        """拟合Weibull分布"""
        self.mean_vecs = []
        for cls in range(self.num_classes):
            cls_features = features[labels == cls]
            mean_vec = np.mean(cls_features, axis=0)
            self.mean_vecs.append(mean_vec)
            # 计算到类中心的距离
            distances = [euclidean(f, mean_vec) for f in cls_features]
            # 选取尾部数据
            tail = sorted(distances, reverse=True)[:tailsize]
            # 拟合 Weibull 分布
            weibull = weibull_min.fit(tail, floc=0)
            self.weibull_models[cls] = weibull

    def openmax(self, logits, features):
        """使用OpenMax进行推理"""
        """改进后的OpenMax推理"""
        recalibrated_logits = np.copy(logits)
        w_scores = []
        for cls in range(self.num_classes):
            distance = euclidean(features, self.mean_vecs[cls])
            weibull = self.weibull_models[cls]
            w_score = weibull_min.cdf(distance, *weibull)
            w_scores.append(w_score)
            recalibrated_logits[cls] *= (1 - w_score)

        # 关键修改：增强未知类激活值
        alpha = 5.0  # 增大alpha值（需通过实验调整）
        unknown_activation = alpha * np.sum(np.array(logits) * np.array(w_scores))
        recalibrated_logits = np.append(recalibrated_logits, unknown_activation)
        from scipy.special import softmax
        # 强制归一化
        recalibrated_probs = softmax(recalibrated_logits)
        return recalibrated_probs
    def get_activations_gradient(self):
        return self.gradients
    def get_activations(self):
        return self.activations
def visualize_gradcam(model, image_tensor, class_idx=None, alpha=0.5):
    """
    改进版可视化函数，将热力图叠加到原图
    参数：
        alpha: 热力图透明度 (1-4-1-4)
    """
    model.eval()
    # 前向传播获取梯度
    output = model(image_tensor.unsqueeze(0))
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    # 反向传播获取梯度
    model.zero_grad()
    output[0, class_idx].backward()
    # 获取激活和梯度
    gradients = model.get_activations_gradient()
    activations = model.get_activations()
    # 计算权重
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # 生成热力图
    heatmap = torch.zeros(activations.shape[2:])
    for i in range(activations.shape[1]):
        heatmap += pooled_gradients[i] * activations[0, i, :, :]
    # 后处理热力图
    # heatmap = np.maximum(heatmap.cpu().numpy(), 0)
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap = cv2.resize(heatmap, (image_tensor.shape[2], image_tensor.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    # 转换原始图像
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # 反归一化
    image = np.clip(image, 0, 1)
    image = np.uint8(255 * image)
    # 转换颜色空间
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 叠加热力图
    superimposed_img = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # 创建可视化
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='turbo')
    plt.title("Attention Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img_rgb)
    plt.title(f"Overlay (alpha={alpha})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
# # 模型训练函数
# def train_one_epoch(model, device, train_loader, criterion, optimizer):
#     model.train()
#     total_loss, correct, total = 0, 0, 0
#     for inputs, targets in train_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * inputs.size(0)
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#     return total_loss / total, correct / total
#
# # Weibull拟合
# def fit_weibull_and_evaluate(model, device, train_loader):
#     model.eval()
#     features, labels = [], []
#     with torch.no_grad():
#         for inputs, targets in train_loader:
#             inputs = inputs.to(device)
#             features.append(model.extract_features(inputs).cpu().numpy())
#             labels.append(targets.cpu().numpy())
#     features = np.vstack(features)
#     labels = np.hstack(labels)
#     model.fit_weibull(features, labels)


# EarlyStopping 类
class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc, model, path):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0

    def save_checkpoint(self, model, path):
        """保存模型到指定路径"""
        torch.save(model.state_dict(), path)
        if self.verbose:
            print(f"Validation accuracy improved. Saving model to {path}")

# 训练一个epoch
def train_one_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # **梯度裁剪，防止梯度爆炸**
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total
# 验证函数
def validate(model, device, val_loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total

# Weibull拟合
def fit_weibull_and_evaluate(model, device, train_loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            features.append(model.extract_features(inputs).cpu().numpy())
            labels.append(targets.cpu().numpy())
    features = np.vstack(features)
    labels = np.hstack(labels)
    model.fit_weibull(features, labels)

# 测试函数
def test(model, device, test_loader, test_dataset, num_classes, unknown_class_label=999):
    print(f"Total samples in test dataset: {len(test_dataset)}")
    model.eval()

    correct_softmax = 0
    correct_openmax = 0
    total_known = 0  # 用于softmax准确率（只对已知类评估）
    total_all = 0  # 用于openmax准确率（包括未知类）

    all_targets = []
    all_softmax_preds = []
    all_openmax_preds = []
    all_softmax_probs = []
    all_openmax_probs = []

    known_class_indices = list(range(num_classes))  # 假设已知类为0 ~ num_classes-1
    # 在数据加载前，将所有未知类别标签映射为统一值（假设测试集中未知类原始标签为4,5）
    test_dataset.targets = [999 if label >= num_classes else label for label in test_dataset.targets]
    with torch.no_grad():
        for inputs, targets in test_loader:
            test_dataset.targets = [999 if label >= num_classes else label for label in test_dataset.targets]
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            softmax_probs = logits.softmax(dim=1).cpu().numpy()
            # 提取特征用于OpenMax
            features = model.extract_features(inputs).cpu().numpy()
            recalibrated_logits = [
                model.openmax(logit.cpu().numpy(), feature)
                for logit, feature in zip(logits, features)
            ]
            recalibrated_probs = np.array(recalibrated_logits)
            # Softmax预测
            _, predicted_softmax = logits.softmax(dim=1).max(1)
            predicted_softmax = predicted_softmax.cpu().numpy()
            # OpenMax预测
            predicted_openmax = recalibrated_probs.argmax(axis=1)
            targets_np = targets.cpu().numpy()
            for i in range(len(targets_np)):
                target_label = targets_np[i]
                total_all += 1
                if target_label in known_class_indices:
                    total_known += 1
                    if predicted_softmax[i] == target_label:
                        correct_softmax += 1
                if predicted_openmax[i] == target_label:
                    correct_openmax += 1
            # 保存所有预测和目标
            all_targets.extend(targets_np)
            all_softmax_preds.extend(predicted_softmax)
            all_openmax_preds.extend(predicted_openmax)
            all_softmax_probs.extend(softmax_probs)
            all_openmax_probs.extend(recalibrated_probs)
    # 准确率计算
    softmax_acc = correct_softmax / total_known if total_known > 0 else 0.0
    openmax_acc = correct_openmax / total_all if total_all > 0 else 0.0

    print(f"Softmax Closed-Set Accuracy (known classes): {softmax_acc:.4f}")
    print(f"OpenMax Open-Set Accuracy (all classes): {openmax_acc:.4f}")

    # 转换为 NumPy 数组
    all_targets = np.array(all_targets)
    all_softmax_probs = np.array(all_softmax_probs)
    all_openmax_probs = np.array(all_openmax_probs)
    # 绘制混淆矩阵
    print("Softmax Confusion Matrix:")
    plot_confusion_matrix(all_targets, all_softmax_preds, classes=[f"Class {i}" for i in range(num_classes)])
    print("OpenMax Confusion Matrix:")
    plot_confusion_matrix(all_targets, all_openmax_preds, classes=[f"Class {i}" for i in range(num_classes)] + ["Unknown"])
    return softmax_acc, openmax_acc
# 主函数
def main():
    # 数据准备
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder('./ddpm_train_data2', transform=transform)
    val_dataset = datasets.ImageFolder('./ddpm_val_data2', transform=transform)
    test_dataset = datasets.ImageFolder('./img_test_data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 模型定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNWithCAAndOpenMax(num_classes=4).to(device)

    # 优化器和损失函数
    # optimizer = optim.Adam(model.parameters(), lr=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    # # 可选：学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # EarlyStopping 初始化
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_model_path = './best_model.pth'
    # 训练和验证
    for epoch in range(50):  # 最大训练50个epoch
        train_loss, train_acc = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, device, val_loader, criterion)
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f},Val Acc={val_acc:.4f}")
        scheduler.step()
        # 检查早停条件
        early_stopping(val_acc, model, best_model_path)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
def test_main():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallCNNWithCAAndOpenMax(num_classes=4).to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    train_dataset = datasets.ImageFolder('./ddpm_train_data2', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = datasets.ImageFolder('./img_test_data', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    # Weibull拟合
    fit_weibull_and_evaluate(model, device, train_loader)
    # 测试模型
    softmax_acc, openmax_acc = test(model, device, test_loader, test_dataset, num_classes = 4)
    print(f"Test Softmax Accuracy: {softmax_acc:.4f}")
    print(f"Test OpenMax Accuracy: {openmax_acc:.4f}")
if __name__ == '__main__':
    # main()
    test_main()

