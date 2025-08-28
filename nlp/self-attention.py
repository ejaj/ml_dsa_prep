import torch
import torch.nn as nn
from torchtyping import TensorType

# 0. Instantiate the linear layers in the following order: Key, Query, Value.
# 1. Biases are not used in Attention, so for all 3 nn.Linear() instances, pass in bias=False.
# 2. torch.transpose(tensor, 1, 2) returns a B x T x A tensor as a B x A x T tensor.
# 3. This function is useful:
#    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
# 4. Apply the masking to the TxT scores BEFORE calling softmax() so that the future
#    tokens don't get factored in at all.
#    To do this, set the "future" indices to float('-inf') since e^(-infinity) is 0.
# 5. To implement masking, note that in PyTorch, tensor == 0 returns a same-shape tensor 
#    of booleans. Also look into utilizing torch.ones(), torch.tril(), and tensor.masked_fill(),
#    in that order.
class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create Q, K, V layers
        self.key = nn.Linear(embedding_dim, attention_dim, bias=True)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=True)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=True)
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # Compute Q, K, V
        Q = self.query(embedded)
        K = self.key(embedded)
        V = self.value(embedded)        
        # Compute attention scores
        scores = Q @ K.transpose(1, 2)  # (B, T, T)
        scores = scores/(K.shape(-1) ** 0.5)
        # scores = torch.sqrt(torch.tensor(K.shape[-1], dtype=torch.float32))
        # Mask future tokens (causal attention)
        T = embedded.shape[1]
        mask = torch.tril(torch.ones(T, T)).to(embedded.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attnetion weights
        attn_weights = torch.softmax(scores, dim=-1)
        # Weighted sum of V
        out = attn_weights @ V  # (B, T, A)
        return torch.round(out, decimals=4)

