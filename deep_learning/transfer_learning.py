import torch
import torch.nn as nn
from torchvision import models

num_classes = 5
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, num_classes)


# last block

# Step 1: Freeze everything
for param in model.parameters():
    param.requires_grad = False

# Step 2: Unfreeze only layer4 and fc
for param in model.layer4.parameters():
    param.requires_grad = True

for param in model.fc.parameters():
    param.requires_grad = True
    
num_classes = 10  # your custom class count
model.fc = nn.Linear(512, num_classes)
