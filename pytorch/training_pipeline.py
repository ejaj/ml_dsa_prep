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

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model prediction
def forward(x):
    return w * x


print(f"Prediction before training f(5) = {forward(5):.3f}")
learning_rate = 0.01
n_epochs = 100

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_epochs):
    y_pred = forward(X)
    # loss
    l = loss_fn(y_pred, Y)
    # gradients = backward pass
    l.backward()  # dl/dw
    optimizer.step()
    # zero gradient
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}: w={w:.3f}, loss = {l:.3f}")
print(f"Prediction after training f(5) = {forward(5):.3f}")
