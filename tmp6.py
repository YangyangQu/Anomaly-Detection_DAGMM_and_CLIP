import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset

# 自定义的图像处理函数
from preprocess import get_imgData
from scipy.stats import entropy

import numpy as np
from torch.nn.functional import softmax


# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 224 * 224, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # 二分类任务，输出两个类别的分数

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # 展平特征
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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


# # 训练模型
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
#
# # 保存模型
# torch.save(model.state_dict(), 'model_tmp6.pth')

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

feature_dim = 512


class PartialModel(nn.Module):
    def __init__(self):
        super(PartialModel, self).__init__()
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建部分模型并加载原始模型的部分参数
partial_model = PartialModel()
# 如果您有CUDA可用
if torch.cuda.is_available():
    partial_model = partial_model.cuda()  # 将模型移动到GPU
    # subject_features = subject_features.cuda()  # 将输入数据也移动到GPU
partial_model.eval()

# 加载原始模型的部分参数
state_dict = torch.load('model_tmp6.pth')  # 路径根据实际情况修改

# # 查看状态字典的键名
# print(state_dict.keys())

if isinstance(state_dict, dict):
    # If 'state_dict' is a dictionary, then load it correctly
    partial_model.fc2.load_state_dict({'weight': state_dict['fc2.weight'], 'bias': state_dict['fc2.bias']})
    partial_model.fc3.load_state_dict({'weight': state_dict['fc3.weight'], 'bias': state_dict['fc3.bias']})

else:
    # If 'state_dict' is not a dictionary, this is where you need to correct your code
    print("state_dict is not a dictionary. It is:", type(state_dict))

# 使用部分模型进行测试
correct = 0
total = 0
import torch
from scipy.stats import entropy
from torch.nn.functional import softmax

# 初始化准确性计数器
correct = 0
total = 0

# 不计算梯度，用于评估/测试模式

with torch.no_grad():
    for features, labels in test_loader:
        # 获取图像特征并转换为浮点类型
        image_features = get_imgData(features).to(torch.float32)

        # 归一化特征以形成概率分布
        image_features_normalized = softmax(image_features, dim=1)

        # 初始化特性向量
        subject_features = torch.zeros_like(image_features[:, :256])

        # 对每个样本单独计算熵并分割特征向量
        for i in range(image_features.shape[0]):
            # 使用 PyTorch 的 clamp 方法将特征向量的值转换为非负值
            non_negative_vector = image_features[i].clamp(min=0)

            # 归一化以形成概率分布
            probability_distribution = non_negative_vector / non_negative_vector.sum()

            # 计算整个概率分布的熵
            vector_entropy = entropy(probability_distribution.cpu().numpy())

            # 对原始特征向量的值进行排序并选择最大的256个值
            top_values = torch.topk(image_features[i], 256).values
            subject_features[i] = top_values

        # 如果模型在 CUDA 上，确保张量也在 CUDA 上
        subject_features = subject_features.to('cuda') if torch.cuda.is_available() else subject_features

        outputs = partial_model(subject_features)
        _, predicted = torch.max(outputs.data, 1)

        # 更新准确性计数器
        total += labels.size(0)
        correct += (predicted == labels.to(predicted.device)).sum().item()

# 计算总准确率
accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")
