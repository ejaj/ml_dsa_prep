import numpy as np

def softmax(x):
    # softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # for numerical stability
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Implements:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """

    # Q: (batch_size, seq_len, d_k)
    # K: (batch_size, seq_len, d_k)
    # V: (batch_size, seq_len, d_k)

    d_k = Q.shape[-1]  # dimensionality of keys/queries

    # 1. Compute dot product between Q and K^T
    # QK^T: shape (batch_size, seq_len, seq_len)
    scores = np.matmul(Q, K.transpose(0, 2, 1))

    # 2. Scale by sqrt(d_k)
    # scores_scaled = QK^T / sqrt(d_k)
    scores_scaled = scores / np.sqrt(d_k)

    # 3. Apply softmax to get attention weights
    # A = softmax(scores_scaled)
    attention_weights = softmax(scores_scaled)

    # 4. Multiply attention weights by V
    # output = A * V (matrix multiplication)
    output = np.matmul(attention_weights, V)

    return output, attention_weights

# 1 sentence, 3 tokens, embedding size 4
np.random.seed(42)
Q = np.random.rand(1, 3, 4)  # Query matrix
K = np.random.rand(1, 3, 4)  # Key matrix
V = np.random.rand(1, 3, 4)  # Value matrix

# Compute attention
output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:\n", attn_weights)
print("\nOutput (weighted sum of values):\n", output)
