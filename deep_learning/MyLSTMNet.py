import torch
import torch.nn as nn

input_size = 10  # Size of each input vector
hidden_size = 20  # LSTM hidden units
num_layers = 1  # One LSTM layer
seq_len = 5  # Sequence length (timesteps)
batch_size = 3  # Batch size


class MyLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)  # x: [batch, seq_len, input_size]
        out = self.fc(out[:, -1, :])  # Take only last time step
        return out


model = MyLSTMNet(input_size=10, hidden_size=20, output_size=5)
dummy_input = torch.randn(batch_size, seq_len, 10)  # [batch, seq_len, input_size]
output = model(dummy_input)
print("Model output:", output.shape)  # [batch_size, output_size]
print("Predictions (logits):", output)