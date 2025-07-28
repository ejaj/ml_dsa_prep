import torch
import torch.nn as nn
import torch.nn.functional as F


class MyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Learnable parameters
        self.W_x = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_h = nn.Parameter(torch.randn(hidden_size, input_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        h_t = torch.tanh(self.W_x @ x_t + self.W_h @ h_prev + self.b)
        return h_t


# Initialize
input_size = 4
hidden_size = 3
rnn_cell = MyRNNCell(input_size, hidden_size)

# Dummy input for one time step
x_t = torch.randn(input_size)  # e.g., input at time t
h_prev = torch.zeros(hidden_size)  # previous hidden state

# Forward pass
h_t = rnn_cell(x_t, h_prev)

# Simulated sequence (5 time steps)
sequence = [torch.randn(input_size) for _ in range(5)]
h_t = torch.zeros(hidden_size)

for t, x_t in enumerate(sequence):
    h_t = rnn_cell(x_t, h_t)
    print(f"Step {t + 1}, h_t: {h_t}")


class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, hn = self.rnn(x)
        final_hidden = hn.squeeze(0)  # [batch_size, hidden_size]
        logits = self.fc(final_hidden)  # [batch_size, output_size]
        return logits


model = RNNClassifier(input_size=4, hidden_size=6, output_size=3)

# Dummy input: batch=2, time=5, features=4
x = torch.randn(2, 5, 4)

logits = model(x)
print("Logits:", logits)

print("Input x_t       :", x_t)
print("Previous h_prev :", h_prev)
print("New hidden h_t  :", h_t)
