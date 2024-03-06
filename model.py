import torch
import torch.nn as nn
import torch.nn.functional as F
from forward_step import is_positive_definite
from preprocess import get_imgData

# class DAGMM(nn.Module):
#     def __init__(self, n_gmm=2, z_dim=1, input_size=512, output_size=256):
#         """Network for DAGMM (KDDCup99)"""
#         super(DAGMM, self).__init__()
#         self.fc1 = nn.Linear(input_size, 256)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(256, output_size)
#
#         # #Estimation network
#         self.fc9 = nn.Linear(z_dim, 10)
#         self.fc10 = nn.Linear(10, n_gmm)
#
#     def estimate(self, z):
#         h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
#         return F.softmax(self.fc10(h), dim=1)
#
#     def forward(self, x):
#         x = x.float()
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         # print("x", x.shape)
#         z_front = x[:, :, :128]
#         z_back = x[:, :, 128:]
#         z_back = z_back.squeeze(0).permute(1,0)
#         # print("z_back", z_back)
#
#         gamma = self.estimate(z_back)
#         # print("gamma", gamma)
#         return z_front, z_back, gamma
#
#


import torch
import torch.nn as nn
import torch.nn.functional as F


class DAGMM(nn.Module):
    def __init__(self, n_gmm=2, z_dim=1):
        super(DAGMM, self).__init__()
        # Encoder network
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.fc1 = nn.Linear(32 * 224 * 224, 1024)  # Assuming the image size is 224x224x3 and after convolution
        # self.fc2 = nn.Linear(1024, z_dim)

        self.fc1 = nn.Linear(512, 128)  # 假设输入大小为[32, 512]
        self.fc2 = nn.Linear(128, 64)
        self.fc7 = nn.Linear(64, z_dim)  # 输出维度修改为[32, 1]

        # Decoder network
        self.fc3 = nn.Linear(z_dim, 1024)
        self.fc4 = nn.Linear(1024, 32 * 224 * 224)  # Assuming the image size is 224x224x3 and before deconvolution
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Estimation network
        self.fc5 = nn.Linear(z_dim + 2, 10)
        self.fc6 = nn.Linear(10, n_gmm)

    # def encode(self, x):
    #     x = get_imgData(x)
    #     print("x11",x.size())
    #     return x
    def encode(self, x):
        x = get_imgData(x)
        x = x.float()
        # print("11",x.size())
        x = F.relu(self.fc1(x))
        # print("22",x.size())
        x = F.relu(self.fc2(x))
        # print("33",x.size())
        x = self.fc7(x)
        # print("44",x.size())

        return x

    def decode(self, z):
        z = z.to(torch.float32)
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        h = h.view(-1, 32, 224, 224)  # Reshape to the original image size
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))  # Apply the final convolutional layer
        return h  # Permute dimensions to match the expected output size

    def estimate(self, z):
        h = F.dropout(F.relu(self.fc5(z)), 0.5)
        return F.softmax(self.fc6(h), dim=1)

    def compute_reconstruction(self, x, x_hat):
        # 计算相对欧氏距离
        euclidean_distance = torch.norm((x - x_hat), p=2, dim=(1, 2, 3))  # 在所有维度上计算欧氏距离
        norm_x = torch.norm(x, p=2, dim=(1, 2, 3))  # 计算输入数据 x 的范数
        relative_euclidean_distance = euclidean_distance / norm_x

        # 确保 relative_euclidean_distance 的维度为 [32]
        relative_euclidean_distance = relative_euclidean_distance.squeeze()

        # 计算余弦相似度
        x_flat = x.view(x.size(0), -1)  # 将输入数据 x 展平为二维张量
        x_hat_flat = x_hat.view(x_hat.size(0), -1)  # 将重构数据 x_hat 展平为二维张量
        cosine_similarity = F.cosine_similarity(x_flat, x_hat_flat, dim=1)

        return relative_euclidean_distance, cosine_similarity


    def forward(self, x):
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        # print("x_hat", x_hat.size())
        # print("x", x.size())
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        # print("rec_1", z_c.size(), rec_1.size(), rec_2.size())
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        # print("z", z.size())
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma

    #
    # def forward(self, x):
    #     x = x.float()
    #     x = self.fc1(x)
    #     x = self.relu(x)
    #     x = self.fc2(x)
    #     # print("x", x.shape)
    #     z_front = x[:, :, :128]
    #     z_back = x[:, :, 128:]
    #     z_back = z_back.squeeze(0).permute(1,0)
    #     # print("z_back", z_back)
    #
    #     gamma = self.estimate(z_back)
    #     # print("gamma", gamma)
    #     return z_front, z_back, gamma
