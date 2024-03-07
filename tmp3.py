
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from preprocess import get_imgData  # 自定义的图像处理函数

from tmp1 import FeatureSeparator
# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=4, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # 二分类任务，输出两个类别的分数

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # print("x",x.size())
        x = torch.relu(self.conv2(x))
        # print("x",x.size())
        x = x.view(x.size(0), -1)  # 展平特征
        # print("x",x.size())
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据预处理
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 数据集路径
data_root = './data/office_caltech_10/tmp_data/'

# 加载数据集并进行预处理
dataset = ImageFolder(root=data_root, transform=data_transforms)

# 划分数据集为训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 初始化模型、损失函数和优化器
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

# 测试模型

# 测试数据集类，接受256维特征向量作为输入
class TestDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


# 初始化模型、损失函数和优化器
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 测试模型
model.eval()
correct = 0
total = 0

feature_dim = 512
separator = FeatureSeparator(feature_dim)

with torch.no_grad():
    for images, labels in test_loader:
        image_features = get_imgData(images).to(torch.float32)  # 调用预处理函数处理图像数据
        subject_features, background_features = separator(image_features)
        outputs = model(subject_features)  # 输入特征向量而不是图像
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {(100 * correct / total):.2f}%")
