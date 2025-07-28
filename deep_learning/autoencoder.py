import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Load MNIST data (flattened to 784-dim vectors)
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)


# 2. Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder: 784 → 128 → 32 (compression)
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),  # W1*x + b1
            nn.ReLU(),  # σ(W1*x + b1)
            nn.Linear(128, 32),  # compressed z
            nn.ReLU()
        )

        # Decoder: 32 → 128 → 784 (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),  # W2*z + b2
            nn.ReLU(),
            nn.Linear(128, 784),  # output reconstruction
            nn.Sigmoid()  # final output: x̂ ≈ x (0 to 1)
        )

    def forward(self, x):
        z = self.encoder(x)  # z = σ(W1*x + b1)
        x_hat = self.decoder(z)  # x̂ = σ(W2*z + b2)
        return x_hat


# 3. Initialize model, loss, optimizer
model = Autoencoder()
criterion = nn.MSELoss()  # Loss: ||x - x̂||²
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training loop
for epoch in range(5):
    for data, _ in train_loader:
        optimizer.zero_grad()

        output = model(data)  # Forward pass
        loss = criterion(output, data)  # Compute reconstruction loss

        loss.backward()  # Backpropagation: compute gradients ∇loss
        optimizer.step()  # Update weights (W, b)

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
