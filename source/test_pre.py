import torch
import timm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
# import random

from vit_prefix import vitPrefix

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_data_dir = '/data/ILSVRC2012'

dataset = ImageFolder(root=os.path.join(imagenet_data_dir, 'val'), transform=transform)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = vitPrefix(model_name='vit_base_patch16_224', pretrained=True, num_classes=1000, n=4)
model.eval()  # Set the model to evaluation mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images, token_type='pretrained_reg')  # Adjust token_type as needed
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy from ImageNet IK dataset: {accuracy:.2%}')
