# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
# -- forward pass: compute prediction
# -- backward pass: gradients
# -- update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
print(y.shape)
y = y.view(y.shape[0], 1)
print(y.shape)
n_samples, n_features = x.shape

model = nn.Linear(n_features, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    # forward pass
    y_pred = model(x)
    # loss
    loss = criterion(y_pred, y)
    # backward pass
    loss.backward()
    # update
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

# plot
pred = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'b.')
plt.plot(x_numpy, pred, 'r.')
plt.show()
