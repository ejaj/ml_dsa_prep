# from pyexpat import model
#
# import torch
# import torch.nn.functional as F
# from torch._C.cpp import nn
# from torchtyping import TensorType
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader, Dataset
#
#
# class Solution:
#     def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
#         """
#         Reshape an MxN tensor into a ((M*N)//2)x2 tensor.
#         """
#         M, N = to_reshape.size()
#         reshaped = to_reshape.reshape((M * N) // 2, 2)
#         return torch.round(reshaped, decimals=4)
#
#     def average(self, to_avg: TensorType[float]) -> TensorType[float]:
#         """
#         Compute the column-wise mean of a tensor.
#         """
#         avg_values = torch.mean(to_avg, dim=0)
#         return torch.round(avg_values, decimals=4)
#
#     def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
#         """
#         Concatenate an MxN tensor and an MxM tensor along axis 1.
#         """
#         concatenated = torch.cat((cat_one, cat_two), dim=1)
#         return torch.round(concatenated, decimals=4)
#
#     def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
#         """
#         Compute the mean squared error (MSE) loss.
#         """
#         loss = F.mse_loss(prediction, target)
#         return torch.round(loss, decimals=4)
#
#
# solution = Solution()
#
# # Example 1: Reshape
# to_reshape = torch.tensor([
#     [1.0, 1.0, 1.0, 1.0],
#     [1.0, 1.0, 1.0, 1.0],
#     [1.0, 1.0, 1.0, 1.0]
# ])
# print(solution.reshape(to_reshape))
#
# # Example 2: Column-wise Average
# to_avg = torch.tensor([
#     [0.8088, 1.2614, -1.4371],
#     [-0.0056, -0.2050, -0.7201]
# ])
# print(solution.average(to_avg))
#
# # Example 3: Concatenate
# cat_one = torch.tensor([
#     [1.0, 1.0, 1.0],
#     [1.0, 1.0, 1.0]
# ])
# cat_two = torch.tensor([
#     [1.0, 1.0],
#     [1.0, 1.0]
# ])
# print(solution.concatenate(cat_one, cat_two))
#
# # Example 4: Compute Loss
# prediction = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
# target = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
# print(solution.get_loss(prediction, target))
#
# print(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
# print(torch.cuda.is_available())
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tensor = torch.tensor([1.0, 2.0])
# tensor.to(device)
# print(tensor, tensor.device)
#
# tensor = torch.zeros(3, 3)
# print(tensor)
#
# tensor = torch.ones(3, 3)
# tensor = torch.rand(4, 4)
# tensor = torch.arange(1, 11)
# tensor = torch.tensor([1, 2, 3])
# numpy_arra = tensor.numpy()
#
# x = torch.tensor(2.0, requires_grad=True)
# y = x ** 2
# y.backward()  # Compute gradient
# print(x.grad)
#
#
# class SimpleNN:
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(10, 5)  # Input 10, Hidden 5
#         self.fc2 = nn.Linear(5, 1)  # Output 1
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# sim = SimpleNN()
#
# optimizer = optim.AdamW(sim.parameters(), lr=0.001)
# pred = torch.tensor([0.5, 0.6, 0.1])
# target = torch.tensor([0.0, 1.0, 0.0])
# loss = F.mse_loss(pred, target)
# print(loss)
#
# torch.save(model.state_dict(), "model.pth")
# model.load_state_dict(torch.load("model.pth"))
#
# datasets = TensorDataset(torch.rand(100, 10), torch.rand(100, 1))
# dataloader = DataLoader(datasets, batch_size=100, shuffle=True)
# model = model.half()
#
#
# class CustomDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx], self.labels[idx]
#
#
# data = torch.rand(100, 10)
# labels = torch.randint(0, 2, (100,))
# dataset = CustomDataset(data, labels)
#
# from torch.cuda.amp import autocast, GradScaler
#
# scaler = GradScaler()
# with autocast():
#     loss = model(input)
# scaler.scale(loss).backward()
# # scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()
#
# accumulation_steps = 4
# for i in range(accumulation_steps):
#     loss = model(input) / accumulation_steps
#     loss.backward()
#     if i % accumulation_steps == 0:
#         optimizer.step()
#         optimizer.zero_grad()
# torch.cuda.empty_cache()


import torch
import torch.fx


class MyModel(torch.nn.Module):
    def forward(self, x):
        return x + 2  # Simple operation


# Instantiate model
model = MyModel()

# Trace the model
traced_model = torch.fx.symbolic_trace(model)

# Print the computation graph
print(traced_model.graph)

import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


model = nn.Sequential(
    nn.Linear(10, 20),
    Swish(),  # Custom activation
    nn.Linear(20, 1)
)


class EarlyStopping:
    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")
    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter =  0
        else:
            self.counter += 1
        return self.counter > self.patience
early_stopping = EarlyStopping(10)
for epoch in range(100):
    loss = compute_loss()  # Dummy function
    if early_stopping.step(loss):
        print("Stopping early!")
        break


















