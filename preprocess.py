import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
import clip

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.backgrounds = []  # 存储背景名称
        self.classes = []  # 存储类别名称
        self.data = []  # 存储图像路径
        self.labels = []  # 存储标签

        # 自动遍历文件夹获取背景和类别名称
        for background in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, background)):
                self.backgrounds.append(background)

        # 循环遍历所有背景中的类别，选取至少一张图像
        for background in self.backgrounds:
            class_dirs = os.listdir(os.path.join(root_dir, background))
            for class_name in class_dirs:
                if os.path.isdir(os.path.join(root_dir, background, class_name)):
                    self.classes.append(class_name)
                    class_dir = os.path.join(root_dir, background, class_name)
                    images = [f for f in os.listdir(class_dir) if f.endswith('.jpg') or f.endswith('.png')]
                    if len(images) >= 1:  # 至少需要一张图
                        self.data.append([os.path.join(class_dir, img) for img in images])
                        self.labels.append((background, class_name))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 从一个背景中随机选择一个类别
        background = random.choice(self.backgrounds)
        class_name = random.choice(self.classes)

        # 从当前背景和类别中随机选择一张图像
        class_dir = os.path.join(self.root_dir, background, class_name)
        image_path = random.choice([os.path.join(class_dir, img) for img in os.listdir(class_dir)])
        image = Image.open(image_path).convert("RGB")

        # 从另一个随机背景中选择相同类别的图像
        other_background = random.choice([bg for bg in self.backgrounds if bg != background])
        other_class_dir = os.path.join(self.root_dir, other_background, class_name)
        other_image_path = random.choice([os.path.join(other_class_dir, img) for img in os.listdir(other_class_dir)])
        other_image = Image.open(other_image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            other_image = self.transform(other_image)
        # image = get_imgData(image)
        # other_image = get_imgData(other_image)
        return image, other_image



def get_imgData(img):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # # 将输入的图像转换为张量对象
    # image_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # print(img.shape)
        # image_features = model.encode_image(img.to(device))
        image_features = model.encode_image(img.to(device))
        # print("feature",image_features.size())
    return image_features
