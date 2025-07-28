import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class MinimalCNNSequential(nn.Module):
    def __init__(self, num_classes=10):
        super(MinimalCNNSequential, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),  # [B, 16, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # [B, 32, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=3)
        )
        self.classsifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*7*7, out_features=num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classsifier(x)
        return x

model = MinimalCNNSequential()
sample = torch.randn(1,1,28,28)
output = model(sample)
print(output.shape)

activations = []

def get_activation(name):
    def hook(model, input, output):
        activations.append((name, output.detach()))
    return hook

# Register hooks to feature layers
for idx, layer in enumerate(model.features):
    if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.MaxPool2d)):
        layer.register_forward_hook(get_activation(f"{layer.__class__.__name__}_{idx}"))

sample_input = torch.randn(1, 1, 28, 28)
model(sample_input)

# Plot function
def plot_feature_maps(name, fmap):
    fmap = fmap[0]  # Take the first image in batch
    num_channels = fmap.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    fig.suptitle(name, fontsize=14)

    for i in range(grid_size * grid_size):
        ax = axes[i // grid_size, i % grid_size]
        if i < num_channels:
            ax.imshow(fmap[i].cpu(), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

for name, fmap in activations:
    plot_feature_maps(name, fmap)