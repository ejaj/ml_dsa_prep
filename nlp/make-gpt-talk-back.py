import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Use torch.multinomial() to choose the next token.
        #    This function simulates a weighted draw from a given list of probabilities
        #    It's similar to picking marbles out of a bag.
        # 2. the given model's output is BEFORE softmax is applied,
        #    and the forward() output has shape batch X time X vocab_size
        # 3. Do not alter the code below, only add to it. This is for maintaining reproducibility in testing.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        model.eval()
        device = next(model.parameters()).device if hasattr(model, "parameters") else context.device
        context = context.to(device)
        out_chars = []

        for i in range(new_chars):

            # YOUR CODE (arbitrary number of lines)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            ctx = context[:, -context_length:]
            with torch.no_grad():
                logits = model(ctx)
            # Next-token logits (B, V)
            next_logits = logits[:, -1, :]

            # Convert to probabilities
            probs = torch.softmax(next_logits, dim=-1)

            # Sample next token id (B, 1) — pass the provided generator
            next_token = torch.multinomial(probs, num_samples=1, generator=generator)

            generator.set_state(initial_state)
            # MORE OF YOUR CODE (arbitrary number of lines)
            context = torch.cat([context, next_token], dim=1)

            # Decode (assumes batch size 1 for returning a single string)
            token_id = int(next_token[0, 0].item())
            out_chars.append(int_to_char.get(token_id, ""))


        # Once your code passes the test, check out the Colab link and hit Run to see your code generate new Drake lyrics!
        # Your code's output, ran in this sandbox will be boring because of the computational limits in this sandbox
        return "".join(out_chars)
