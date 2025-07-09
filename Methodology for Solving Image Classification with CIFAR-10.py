# %% 修正后的导入语句
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets  # 修正导入方式
import torchvision.transforms as transforms  # 修正导入方式
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import warnings

# 忽略警告（可选）
warnings.filterwarnings('ignore')

# %% 检查PyTorch和CUDA可用性
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA设备数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前设备: {torch.cuda.get_device_name(0)}")

# %% 加载数据并定义预处理
# 定义预处理转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1,1]
])

# 加载数据集
train_dataset = datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True,
    transform=transform
)

# %% 数据可视化
classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

# 创建未归一化的数据集用于可视化
raw_train_dataset = datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True
)

# 可视化样本图像
plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i+1)
    plt.imshow(raw_train_dataset.data[i])
    plt.title(classes[raw_train_dataset.targets[i]])
    plt.axis('off')
plt.suptitle('CIFAR-10 数据集样本', fontsize=16)
plt.tight_layout()
plt.savefig('cifar10_samples.png', dpi=100)
plt.show()

# %% 创建数据加载器
# 分割训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_data, val_data = random_split(train_dataset, [train_size, val_size])

BATCH_SIZE = 64
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# %% 创建CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # 输入: 3通道(RGB), 输出: 16个特征图, 卷积核3x3, 填充1保持尺寸
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 尺寸减半: 32x32 -> 16x16
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),  # 64个4x4特征图
            nn.ReLU(),
            nn.Dropout(0.5),  # 添加Dropout减少过拟合
            nn.Linear(512, 10)  # 输出10个类别
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        return self.fc_layers(x)

# 实例化模型
model = SimpleCNN()
print("模型结构:")
print(model)

# %% 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练参数
EPOCHS = 15

# 记录训练过程
train_losses, val_losses = [], []
train_acc, val_acc = [], []

# %% 训练循环
print("\n开始训练...")
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # 计算训练集指标
    epoch_train_loss = running_loss / len(train_loader)
    epoch_train_acc = correct / total
    train_losses.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 计算验证集指标
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = correct / total
    val_losses.append(epoch_val_loss)
    val_acc.append(epoch_val_acc)
    
    # 打印进度
    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"训练损失: {epoch_train_loss:.4f}, 准确率: {epoch_train_acc:.4f} | "
          f"验证损失: {epoch_val_loss:.4f}, 准确率: {epoch_val_acc:.4f}")

# %% 可视化训练过程
plt.figure(figsize=(14, 6))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失', marker='o')
plt.plot(val_losses, label='验证损失', marker='o')
plt.title('损失曲线')
plt.xlabel('训练轮次')
plt.ylabel('损失')
plt.legend()
plt.grid(True)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='训练准确率', marker='o')
plt.plot(val_acc, label='验证准确率', marker='o')
plt.title('准确率曲线')
plt.xlabel('训练轮次')
plt.ylabel('准确率')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=100)
plt.show()

# %% 测试集评估
model.eval()
all_preds = []
all_labels = []

print("\n开始测试集评估...")
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 收集所有预测结果
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算总体准确率
test_accuracy = correct / total
print(f"测试集准确率: {test_accuracy:.4f} ({correct}/{total})")

# %% 详细评估报告
print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=classes))

# 混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.savefig('confusion_matrix.png', dpi=100)
plt.show()

# %% 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'classes': classes,
    'test_accuracy': test_accuracy
}, 'cifar10_cnn_model.pth')
print("模型已保存为 'cifar10_cnn_model.pth'")

# %% 可视化一些测试样本及其预测
def denormalize(img):
    """将归一化后的图像反归一化以便显示"""
    img = img * 0.5 + 0.5  # 反归一化
    return np.clip(img, 0, 1)

# 获取测试集样本
sample_images, sample_labels = next(iter(test_loader))
sample_images = sample_images[:12]  # 取前12个样本
sample_labels = sample_labels[:12]

# 预测
model.eval()
with torch.no_grad():
    sample_images_gpu = sample_images.to(device)
    outputs = model(sample_images_gpu)
    _, predictions = torch.max(outputs, 1)
    predictions = predictions.cpu().numpy()

# 可视化预测结果
plt.figure(figsize=(15, 10))
for i in range(12):
    plt.subplot(3, 4, i+1)
    img = sample_images[i].permute(1, 2, 0).numpy()  # C, H, W -> H, W, C
    img = denormalize(img)
    
    # 显示图像
    plt.imshow(img)
    
    # 显示预测结果（绿色为正确，红色为错误）
    true_label = classes[sample_labels[i]]
    pred_label = classes[predictions[i]]
    color = 'green' if true_label == pred_label else 'red'
    title = f"真: {true_label}\n预: {pred_label}"
    plt.title(title, color=color)
    plt.axis('off')

plt.suptitle('测试集样本预测结果', fontsize=16)
plt.tight_layout()
plt.savefig('predictions.png', dpi=100)
plt.show()