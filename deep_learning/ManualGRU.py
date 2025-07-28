import torch
import torch.nn.functional as F

# Sample sizes
input_size = 4
hidden_size = 3
# Inputs
x_t = torch.tensor([0.5, -0.1, 0.8, 0.2])  # input at time t (size: input_size)
h_prev = torch.tensor([0.1, 0.2, -0.3])  # previous hidden state (size: hidden_size)

# Weight matrcis (random values)
W_z = torch.rand(hidden_size, input_size)
U_z = torch.rand(hidden_size, input_size)
W_r = torch.rand(hidden_size, input_size)
U_r = torch.rand(hidden_size, input_size)

W_h = torch.rand(hidden_size, input_size)
U_h = torch.rand(hidden_size, input_size)

# Bias terms
b_z = torch.randn(hidden_size)
b_r = torch.randn(hidden_size)
b_h = torch.randn(hidden_size)

# Update gate
z_t = torch.sigmoid(W_z @ x_t + U_z @ h_prev + b_z)

# Reset gate
r_t = torch.sigmoid(W_r @ x_t + U_r @ h_prev + b_r)

# Candidate hidden state
h_tilde = torch.tanh(W_h @ x_t + U_h @ (r_t * h_prev) + b_h)

# Final hidden state
h_t = (1 - z_t) * h_prev + z_t * h_tilde

print("Update gate (z_t):", z_t)
print("Reset gate (r_t):", r_t)
print("Candidate hidden state (h~):", h_tilde)
print("New hidden state (h_t):", h_t)
