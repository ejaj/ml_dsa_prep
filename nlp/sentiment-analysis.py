import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        assert vocabulary_size > 0, "vocabulary_size must be > 0"
        self.embdeding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16)
        self.fc = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long)
        else:
            x = x.long()
        # B, T, 16
        emb = self.embdeding(x)
        # Average over time dimension -> B, 16
        avg = emb.mean(dim=1)
        # Liner -> Sigmoid -> B, 1
        logits = self.fc(avg)
        probs = self.sigmoid(logits)

        probs = torch.round(probs * 10000) / 10000
        return probs

vocabulary_size = 170_000
model = Solution(vocabulary_size)

x = [
  [2, 7, 14, 8, 0, 0, 0, 0, 0, 0, 0, 0],
  [1, 4, 12, 3, 10, 5, 15, 11, 6, 9, 13, 7]
]

with torch.no_grad():
    y = model(x)   # shape: [2, 1], values in [0, 1], e.g. [[0.5],[0.1]]
