import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from torchvision.models import ResNet18_Weights  # 导入特定的权重

# 创建一个处理图像对的数据集类
class ImagePairDataset(Dataset):
    def __init__(self, folder1, folder2, transform=None):
        self.folder1 = folder1  # 第一个文件夹路径
        self.folder2 = folder2  # 第二个文件夹路径
        self.transform = transform  # 图像转换操作
        self.images1 = sorted(os.listdir(folder1))  # 确保图像顺序
        self.images2 = sorted(os.listdir(folder2))  # 确保图像顺序

    def __len__(self):
        return min(len(self.images1), len(self.images2))  # 数据集大小

    def __getitem__(self, idx):
        img_name1 = os.path.join(self.folder1, self.images1[idx])
        img_name2 = os.path.join(self.folder2, self.images2[idx])
        image1 = Image.open(img_name1).convert('RGB')  # 以RGB格式打开图像1
        image2 = Image.open(img_name2).convert('RGB')  # 以RGB格式打开图像2
        label = int(self.images2[idx].startswith('1'))  # 从文件名提取标签
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, torch.tensor([label], dtype=torch.float32)  # 返回图像对和标签

# 创建一个孪生网络模型类
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn_base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用预训练权重
        self.cnn_base.fc = nn.Linear(self.cnn_base.fc.in_features, 128)  # 修改全连接层
        self.fc = nn.Linear(256, 1)  # 定义一个新的全连接层来输出相似度

    def forward(self, input1, input2):
        output1 = self.cnn_base(input1)
        output2 = self.cnn_base(input2)
        combined_features = torch.cat((output1, output2), 1)  # 合并特征
        similarity = self.fc(combined_features)  # 计算相似度
        return torch.sigmoid(similarity).squeeze()  # 输出单一维度的结果

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小至224x224
    transforms.ToTensor()  # 将图像转换为Tensor
])

# 创建数据集和数据加载器
dataset = ImagePairDataset("G:\\ass\\shixi\\DAY2\\DAY2\\Image_398_result", "G:\\ass\\shixi\\DAY2\\DAY2\\Image_398", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # 批量大小为4，打乱顺序

# 初始化模型、损失函数和优化器
model = SiameseNetwork()
model_save_path = "G:/suanfa/毕业实习/siamese_model.pth"
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
    print("已加载预训练模型。")
else:
    print("未找到预训练模型，将从头开始训练。")
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images1, images2, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images1, images2).squeeze()
        loss = criterion(outputs, labels.squeeze())  # 确保标签维度与输出匹配
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}')

print("训练完成。")
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存至 {model_save_path}")
