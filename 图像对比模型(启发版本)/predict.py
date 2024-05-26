import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision.models import ResNet18_Weights
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os
from torchvision.models import ResNet18_Weights
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os
from torchvision.models import ResNet18_Weights

# Dataset Definition
class ImagePairDataset(Dataset):
    def __init__(self, folder1, folder2, transform=None):
        self.folder1 = folder1
        self.folder2 = folder2
        self.transform = transform
        self.images1 = sorted(os.listdir(folder1))
        self.images2 = sorted(os.listdir(folder2))

    def __len__(self):
        return min(len(self.images1), len(self.images2))

    def __getitem__(self, idx):
        img_name1 = os.path.join(self.folder1, self.images1[idx])
        img_name2 = os.path.join(self.folder2, self.images2[idx])
        image1 = Image.open(img_name1).convert('RGB')
        image2 = Image.open(img_name2).convert('RGB')
        label = int(self.images2[idx].startswith('1'))  # Extract label from file name
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2, label, img_name2  # Return the image path for output

# Model Definition
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn_base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.cnn_base.fc = nn.Linear(self.cnn_base.fc.in_features, 128)
        self.fc = nn.Linear(256, 1)

    def forward(self, input1, input2):
        output1 = self.cnn_base(input1)
        output2 = self.cnn_base(input2)
        combined_features = torch.cat((output1, output2), 1)
        similarity = self.fc(combined_features)
        return torch.sigmoid(similarity).squeeze()

# Load the model
model = SiameseNetwork()
model_save_path = "G:/suanfa/毕业实习/siamese_model.pth"
model.load_state_dict(torch.load(model_save_path))
model.eval()  # Set the model to evaluation mode

# Set up data loading
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = ImagePairDataset("G:\\ass\\shixi\\DAY45\\all\\Image_164_result",
                           "G:\\ass\\shixi\\DAY45\\all\\Image_164", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Prediction and comparison
num_mismatches = 0
for i, (image1, image2, true_label, img_path) in enumerate(dataloader):
    with torch.no_grad():
        output = model(image1, image2).item()
        predicted_label = int(output >= 0.3)  # Threshold set at 0.5
        if predicted_label != true_label:
            num_mismatches += 1
            print(f"Mismatch: {os.path.basename(img_path[0])}")


print(f"Total mismatches: {num_mismatches}")
