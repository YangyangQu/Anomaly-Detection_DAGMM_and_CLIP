import torch
import torch.nn as nn
import torch.nn.functional as F

class DAGMM(nn.Module):
    def __init__(self, n_gmm=2, z_dim=512):
        super(DAGMM, self).__init__()
        # 编码器网络
        self.fc1 = nn.Linear(512, 128)  # 假设输入大小为[32, 512]
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 输出维度修改为[32, 1]

    def encode(self, x):
        print("11",x.size())
        x = F.relu(self.fc1(x))
        print("22",x.size())
        x = F.relu(self.fc2(x))
        print("33",x.size())
        x = self.fc3(x)
        print("44",x.size())

        return x

# 示例用法
model = DAGMM()
x = torch.randn(32, 512)  # 示例输入数据
output = model.encode(x)
print(output.shape)  # 检查输出形状