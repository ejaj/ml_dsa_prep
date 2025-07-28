import torch
import torch.nn as nn
import torch.nn.functional as F


# Custom GRU Cell class
class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Weight parameters
        self.W_z = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_z = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))

        self.W_r = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_r = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_r = nn.Parameter(torch.zeros(hidden_size))

        self.W_h = nn.Parameter(torch.randn(hidden_size, input_size))
        self.U_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x_t, h_prev):
        z_t = torch.sigmoid(self.W_z @ x_t + self.U_z @ h_prev + self.b_z)
        r_t = torch.sigmoid(self.W_r @ x_t + self.U_r @ h_prev + self.b_r)
        h_tilde = torch.tanh(self.W_h @ x_t + self.U_h @ (r_t * h_prev) + self.b_h)
        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        print("Input x_t       :", x_t)
        print("Previous h_prev :", h_prev)
        print("Update gate z_t :", z_t)
        print("Reset gate r_t  :", r_t)
        print("h_tilde         :", h_tilde)
        print("New hidden h_t  :", h_t)
        return h_t


# Test
input_size = 4
hidden_size = 3

gru_cell = MyGRUCell(input_size, hidden_size)

x_t = torch.randn(input_size)  # e.g., tensor([ 0.5, -1.2, 0.8, 0.3])
h_prev = torch.randn(hidden_size)  # e.g., tensor([ 0.1,  0.2, -0.3])

# Forward pass
h_t = gru_cell(x_t, h_prev)


class MyGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyGRUClassifier, self).__init__()
        self.gru_cell = MyGRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)  # Prediction layer

    def forward(self, x_sequence):
        h_t = torch.zeros(self.gru_cell.hidden_size)  # Init hidden state

        for x_t in x_sequence:  # Process each time step
            h_t = self.gru_cell(x_t, h_t)

        out = self.fc(h_t)  # Final prediction from last hidden state
        return out


# Simulated sequence input (5 time steps)
x_sequence = [torch.randn(4) for _ in range(5)]  # 5 vectors of size 4

model = MyGRUClassifier(input_size=4, hidden_size=3, output_size=2)  # binary output
prediction = model(x_sequence)

print("Final prediction (logits):", prediction)


class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, hn = self.gru(x)  # hn: [1, batch, hidden_size]
        out = self.fc(hn.squeeze(0))  # remove layer dim → [batch, output_size]
        return out


model = GRUClassifier(input_size=4, hidden_size=8, output_size=2)
x = torch.randn(2, 5, 4)  # batch=2, sequence=5, features=4
logits = model(x)
print("Prediction logits:", logits)
