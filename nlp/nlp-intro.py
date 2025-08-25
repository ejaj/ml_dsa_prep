import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List
from torch.nn.utils.rnn import pad_sequence

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        all_sentences = positive + negative
        # print(all_sentences)
        words = []
        for sentence in all_sentences:
            words.extend(sentence.split())
        # print(words)
        # Build vocab (sorted lexicographically)
        vocab = {word: i+1 for i, word in enumerate(sorted(set(words)))}
        
        # Encode each sentence into numbers
        encoded_sentences = []
        for sentence in all_sentences:
            encoded_sentences.append(torch.tensor(
                [vocab[word] for word in sentence.split()]
            ))
        # print(encoded_sentences)
        # Pad sequences to equal length
        # T = max(len(sentenc) for sentenc in encoded_sentences)
        # padded_sentences = []
        # for sentence in encoded_sentences:
        #     paddin_needed = T - len(sentence)
        #     padded_sentence = sentence + [0] * paddin_needed
        #     padded_sentences.append(pad_sequence)
        padded = pad_sequence(encoded_sentences, batch_first=True, padding_value=0)
        return padded.float()



        
       


positive = ["Dogecoin to the moon"]
negative = ["I will short Tesla today"]

solution = Solution()
print(solution.get_dataset(positive, negative))

