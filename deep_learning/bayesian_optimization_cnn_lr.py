import torch
import torch.nn as nn
import torch.optim as optim 
from torchvision import datasets, transforms
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from torch.utils.data import DataLoader, Subset

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    
    def forward(self, x):
        return self.net(x)

transform = transforms.ToTensor()
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
valset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_subset = Subset(trainset, range(512))
val_subset = Subset(valset, range(256))

trainloader = DataLoader(train_subset, batch_size=64, shuffle=True)
valloader = DataLoader(val_subset, batch_size=64, shuffle=False)

def evaluate_learning_rate(lr):
    model = SimpleCNN()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train on one batch for speed
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break

    # Validation loss on one batch
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            break
    return val_loss

# Expected Improvement (Acquisition Function)
def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.min(Y_sample)

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

# Bayesian Optimization Loop
search_space = np.atleast_2d(np.linspace(0.0001, 0.1, 30)).T
X_sample = []
Y_sample = []

for step in range(10):
    print(f"\nStep {step + 1}/10")
    
    if step < 3:
        lr = float(search_space[np.random.randint(len(search_space))])
    else:
        kernel = Matern()
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(np.array(X_sample), np.array(Y_sample))
        ei = expected_improvement(search_space, np.array(X_sample), np.array(Y_sample), gpr)
        lr = float(search_space[np.argmax(ei)])
    
    print(f"Trying learning rate: {lr:.5f}")
    loss = evaluate_learning_rate(lr)
    print(f"Validation loss: {loss:.4f}")
    
    X_sample.append([lr])
    Y_sample.append(loss) 

best_idx = int(np.argmin(Y_sample))
best_lr = X_sample[best_idx][0]
best_loss = Y_sample[best_idx]

print("\nBest Learning Rate Found:")
print(f"Learning Rate: {best_lr:.5f}  →  Validation Loss: {best_loss:.4f}")

# Plot Results
X_sample = np.array(X_sample).flatten()
Y_sample = np.array(Y_sample)

plt.figure(figsize=(8, 4))
plt.plot(X_sample, Y_sample, 'o-')
plt.xlabel("Learning Rate")
plt.ylabel("Validation Loss")
plt.title("Bayesian Optimization: Learning Rate Tuning")
plt.grid(True)
plt.show()
