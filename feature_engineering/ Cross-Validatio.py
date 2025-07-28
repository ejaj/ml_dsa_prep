import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import numpy as np

# ---------------------
# 1. Define the model
# ---------------------
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

# ---------------------
# 2. Set device and hyperparameters
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5
batch_size = 64
learning_rate = 0.001
k_folds = 5

# ---------------------
# 3. Load dataset
# ---------------------
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# ---------------------
# 4. K-Fold CV Setup
# ---------------------
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\nFold {fold+1}/{k_folds}")

    # Create subsets for this fold
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ---------------------
    # 5. Train loop
    # ---------------------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # ---------------------
    # 6. Validate loop
    # ---------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    fold_accuracies.append(accuracy)
    print(f"✅ Validation Accuracy for Fold {fold+1}: {accuracy:.4f}")

# ---------------------
# 7. Report Final Results
# ---------------------
print("\n📊 K-Fold Cross-Validation Results:")
for i, acc in enumerate(fold_accuracies):
    print(f"Fold {i+1}: {acc:.4f}")
print(f"\n📈 Average Accuracy: {np.mean(fold_accuracies):.4f}")
