import torch
import torch.nn.functional as F

# Input size and hidden size
input_size = 3
hidden_size = 3

x_t = torch.tensor([0.5, -0.1, 0.3])  # current input x_t
h_prev = torch.tensor([0.2, 0.4, -0.3])  # previous hidden state h_{t-1}
C_prev = torch.tensor([0.8, -0.3, 0.5])  # previous cell state C_{t-1}

# Weight matrices and biases for gates (random or fixed for testing)
# [x_t ; h_prev] → combined input of size 6
Wx = torch.randn((4 * hidden_size, input_size))  # weights for x_t
Wh = torch.randn((4 * hidden_size, hidden_size)) # weights for h_{t-1}
b = torch.randn((4 * hidden_size,))              # biases

# combine input and hidden state
combined = torch.cat((x_t, h_prev))  # shape: (6,)

# Linear transformations for gates
gates = torch.matmul(Wx, x_t) + torch.matmul(Wh, h_prev) + b
f_t, i_t, o_t, g_t = torch.chunk(gates, 4, dim=0)

# Activations
f_t = torch.sigmoid(f_t)           # forget gate
i_t = torch.sigmoid(i_t)           # input gate
o_t = torch.sigmoid(o_t)           # output gate
g_t = torch.tanh(g_t)              # candidate cell state ~C_t

# Cell state update
C_t = f_t * C_prev + i_t * C_prev * g_t
# Hidden state update
h_t = o_t * torch.tanh(C_t)

# Print outputs
print("Forget Gate f_t       :", f_t)
print("Input Gate i_t        :", i_t)
print("Candidate Cell ~C_t   :", g_t)
print("New Cell State C_t    :", C_t)
print("Output Gate o_t       :", o_t)
print("New Hidden State h_t  :", h_t)












