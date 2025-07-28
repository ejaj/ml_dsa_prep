import torch

# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
# -- forward pass: compute prediction
# -- backward pass: gradients
# -- update weights

# f = w * x
# f = 2 ** x

import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
x_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape
print(n_samples, n_features)


# model = nn.Linear(n_features, n_features)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


model = LinearRegressionModel(n_features, n_features)
print(f"Prediction before training f(5) = {model(x_test).item():.3f}")

learning_rate = 0.01
n_epochs = 100

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_epochs):
    y_pred = model(X)
    # loss
    l = loss_fn(y_pred, Y)
    # gradients = backward pass
    l.backward()  # dl/dw
    optimizer.step()
    # zero gradient
    optimizer.zero_grad()
    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"Epoch {epoch + 1}: w={w[0].item():.3f}, loss = {l:.3f}")
print(f"Prediction after training f(5) = {model(x_test).item():.3f}")
