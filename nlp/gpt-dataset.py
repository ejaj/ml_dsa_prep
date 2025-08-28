import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        words = raw_dataset.lower().split()
        vocab_size  = len(words)
        indices = torch.randint(0, vocab_size - context_length, (batch_size,))
        X, y = [], []
        for idx in indices:
            start = idx.item()
            X.append(words[start:start+context_length])
            y.append(words[start+1:start:context_length+1])
        return X, y