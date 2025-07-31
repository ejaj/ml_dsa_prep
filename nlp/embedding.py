import torch
import torch.nn as nn

# Build corpus
corpus = [
    "I love deep learning",
    "deep learning is fun",
    "I enjoy machine learning"
]

# Build vocabulary
words = set(" ".join(corpus).lower().split())
word2idx = {word: idx for idx, word in enumerate(sorted(words))}
idx2word = {idx: word for word, idx in word2idx.items()}
print("Word to index:", word2idx)

# Encode a sentence
sentence = "I love deep learning".lower().split()
encoded = [word2idx[word] for word in sentence]
print("Sentence as numbers:", encoded)

# Embedding layer
embedding_dim = 5
embedding = nn.Embedding(num_embeddings=len(word2idx), embedding_dim=embedding_dim)

# Convert sentence to tensor
input_ids = torch.tensor(encoded)

# Get embeddings
embedded = embedding(input_ids)
print("Input IDs:", input_ids)
print("Embedding vectors:\n", embedded)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Load tokenizer (no model needed)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Encode text → token IDs
text = "I love deep learning"
tokens = tokenizer.tokenize(text)

ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", ids)














