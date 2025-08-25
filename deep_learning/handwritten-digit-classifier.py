import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()
        )
    
    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        x = images.view(images.shape[0], -1).float()
        out = self.net(x)
        return torch.round(out, decimals=4)
