import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from preprocess import get_imgData

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from preprocess import get_imgData  # 自定义的图像处理函数


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionLayer, self).__init__()
        self.feature_dim = feature_dim
        # 定义注意力权重张量为可学习参数
        self.attention_weights = nn.Parameter(torch.randn(feature_dim, dtype=torch.float32))

    def forward(self, features):
        # 将注意力权重移动到与特征相同的设备上
        attention_weights = self.attention_weights.to(features.device)
        # 应用注意力权重
        attention_scores = torch.matmul(features, attention_weights)
        attention_scores = F.softmax(attention_scores, dim=0)
        # 加权特征
        weighted_features = features * attention_scores.unsqueeze(-1)
        return weighted_features, attention_scores


class FeatureSeparator(nn.Module):
    def __init__(self, feature_dim):
        super(FeatureSeparator, self).__init__()
        self.attention_layer = AttentionLayer(feature_dim)

    def forward(self, features):
        # 获取加权特征和注意力分数
        weighted_features, attention_scores = self.attention_layer(features)
        # 分离主体和背景
        subject_features = weighted_features * attention_scores.unsqueeze(-1)
        background_features = weighted_features * (1 - attention_scores).unsqueeze(-1)
        return subject_features[:, :256], background_features[:, :256]


# 定义损失函数
def contrastive_loss(subject_features, background_features, margin=1.0):
    # 计算主体和背景特征之间的欧几里得距离
    distance = F.pairwise_distance(subject_features, background_features)
    # 计算对比损失
    loss = torch.mean(torch.max(distance - margin, torch.zeros_like(distance)))
    return loss


# 定义数据预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 图像数据所在的文件夹路径
data_folder = './data/office_caltech_10'

# 创建 ImageFolder 数据集实例
image_dataset = datasets.ImageFolder(root=data_folder, transform=preprocess)

# 创建 DataLoader 实例
batch_size = 16
data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)

# 创建特征分离器模型
feature_dim = 512
separator = FeatureSeparator(feature_dim)

# 定义优化器
optimizer = torch.optim.Adam(separator.parameters(), lr=0.001)

num_epochs = 100
margin = 1.0

# 使用 DataLoader 进行迭代训练
for epoch in range(num_epochs):
    total_loss = 0.0  # 初始化总损失

    for images, labels in data_loader:
        optimizer.zero_grad()

        # images = preprocess(images).unsqueeze(0)
        image_features = get_imgData(images).to(torch.float32)  # 调用预处理函数处理图像数据
        subject_features, background_features = separator(image_features)
        loss = contrastive_loss(subject_features, background_features, margin)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  # 累加损失

    # 打印训练进度
    if (epoch + 1) % 1 == 0:
        avg_loss = total_loss / len(data_loader)  # 计算平均损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Avg Loss: {avg_loss}")

model = FeatureSeparator(feature_dim)

# 加载已训练的模型参数
model.load_state_dict(torch.load('your_trained_model.pth'))

# 将模型设置为评估模式
model.eval()

# 测试数据预处理
img_path = "frame_0002.jpg"
img = Image.open(img_path)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img = preprocess(img).unsqueeze(0)
image_features = get_imgData(img).to(torch.float32)  # 将测试数据转换为与模型训练时相同的数据类型

# 使用训练后的模型进行特征分离
with torch.no_grad():
    subject_features, background_features = model(image_features)

# 打印结果或进行后续处理
print("主体特征:", subject_features)
print("背景特征:", background_features)

subject_features = subject_features.cpu().detach().numpy().flatten()
background_features = background_features.cpu().detach().numpy().flatten()

# 合并特征
all_features = np.concatenate((subject_features, background_features), axis=0)

# 标签，1代表主体，0代表背景
labels = np.concatenate((np.ones(subject_features.shape[0]), np.zeros(background_features.shape[0])))

# 使用KMeans聚类算法
kmeans = KMeans(n_clusters=2)  # 假设有两个类别
# 将特征重塑为二维数组
all_features = all_features.reshape(-1, 1)
kmeans.fit(all_features)

# 获取聚类中心
centroids = kmeans.cluster_centers_

# 绘制聚类结果
plt.scatter(all_features[:, 0], np.zeros_like(all_features), c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(centroids[:, 0], np.zeros_like(centroids), marker='*', s=200, c='red', label='Centroids')
plt.xlabel('Feature')
plt.title('Clustering of Features')
plt.legend()
plt.show()
