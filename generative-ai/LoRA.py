import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy

if torch.cuda.is_available():
    # NVIDIA CUDA Deep Neural Network (cuDNN) is a GPU-accelerated library of primitives for deep neural networks
    torch.backends.cudnn.deterministic=True

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev=1/torch.sqrt(torch.tensor(rank).float())
        self.A=nn.Parameter(torch.randn(in_dim, rank)*std_dev)
        self.B=nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha=alpha
        
    def forward(self, x):
        x=self.alpha*(x@self.A@self.B)
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear=linear
        self.lora=LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
    
    def forward(self, x):
        return self.linear(x)+self.lora(x)
    
# Hyperparameters
random_seed=123

torch.manual_seed(random_seed)
layer=nn.Linear(10,2)
x=torch.randn((1, 10))

print(x)
print(layer)
print('Original output:', layer(x))

layer_lora_1=LinearWithLoRA(layer, rank=2, alpha=4)
print(layer_lora_1(x))

class LinearWithLoRAMerged(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora=LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )
    def forward(self, x):
        lora=self.lora.A @ self.lora.B # combine LoRA metrices
        # then combine LoRA original weights
        combined_weight=self.linear.weight+self.lora.alpha*lora.T
        return F.linear(x, combined_weight, self.linear.bias)

layer_lora_2=LinearWithLoRAMerged(layer, rank=2, alpha=4)
print(layer_lora_2(x))

class MultilayerPerceptron(nn.Module):
    def __init__(self, num_features, num_hidden_1, num_hidden_2, num_classes):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(num_features, num_hidden_1),
            nn.ReLU(),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.ReLU(),
            nn.Linear(num_hidden_2, num_classes)
        )
    
    def forward(self, x):
        x=self.layers(x)
        return x

# Architecture
num_features=784
num_hidden_1=128
num_hidden_2=256
num_classes=10

# Settings
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate=0.005
num_epochs=10

model=MultilayerPerceptron(
    num_features=num_features,
    num_hidden_1=num_hidden_1,
    num_hidden_2=num_hidden_2,
    num_classes=num_classes
)

model.to(DEVICE)
optimizer_pretrained=torch.optim.Adam(model.parameters(), lr=learning_rate)
print(DEVICE)
print(model)
print(optimizer_pretrained)

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

BATCH_SIZE=64

# Note: transforms.ToTensor() scales input images to 0-1 range
train_dataset=datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset=datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader=DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader=DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

for images, labels in train_loader:
    print('Image batch dimensions:', images.shape)
    print('Image label dimensions:', labels.shape)
    break

def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples=0,0
    with torch.no_grad():
        for features, targets in data_loader:
            features=features.view(-1, 28*28).to(device)
            targets=targets.to(device)
            logits=model(features)
            _, predicted_labels=torch.max(logits,1)
            num_examples+=targets.size(0)
            correct_pred+=(predicted_labels==targets).sum()
        return correct_pred.float()/num_examples*100

def train(num_epochs, model, optimizer, train_loader, device):
    start_time=time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features=features.view(-1, 28*28).to(device)
            targets=targets.to(device)
            
            # forward and back propagation
            logits=model(features)
            loss=F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # logging
            if not batch_idx %400:
                print('Epoch: %03d/%03d|Batch %03d/%03d| Loss: %.4f' % (epoch+1, num_epochs, batch_idx, len(train_loader), loss))
        
        with torch.set_grad_enabled(False):
            print('Epoch: %03d/%03d training accuracy: %.2f%%' % (epoch+1, num_epochs, compute_accuracy(model, train_loader, device)))
        
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

train(num_epochs, model, optimizer_pretrained, train_loader, DEVICE)
print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')

model_lora = copy.deepcopy(model)

model_lora.layers[0]=LinearWithLoRAMerged(model_lora.layers[0], rank=4, alpha=8)
model_lora.layers[2]=LinearWithLoRAMerged(model_lora.layers[2], rank=4, alpha=8)
model_lora.layers[4]=LinearWithLoRAMerged(model_lora.layers[4], rank=4, alpha=8)
model_lora.to(DEVICE)
optimizer_lora=torch.optim.Adam(model_lora.parameters(), lr=learning_rate)
print(model_lora)

print(f'Test accuracy orig model:{compute_accuracy(model, test_loader, DEVICE):.2f}%')
print(f'Test accuracy LoRA model:{compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')

def freeze_linear_layers(model):
    for child in model.children():
        if isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad=False
        else:
            # recursively freeze linear layers in children modules
            freeze_linear_layers(child)

freeze_linear_layers(model_lora)
for name, param in model_lora.named_parameters():
    print(f'{name}:{param.requires_grad}')
        
optimizer_lora=torch.optim.Adam(model_lora.parameters(), lr=learning_rate)
train(num_epochs, model_lora, optimizer_lora, train_loader, DEVICE)
print(f'Test accuracy LoRA finetune: {compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')

print(f'Test accuracy orig model:{compute_accuracy(model, test_loader, DEVICE):.2f}%')
print(f'Test accuracy LoRA model:{compute_accuracy(model_lora, test_loader, DEVICE):.2f}%')