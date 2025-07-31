import numpy as np 
vocab = ["I", "like", "love", "pizza", "<end>"]
def get_next_token_probs(sequence):
    """
    Simulated language model: returns next-token probabilities.
    """
    if sequence == []:
        return {"I": 0.6, "You": 0.4}
    elif sequence[-1] == "I":
        return {"like": 0.4, "love": 0.6}
    elif sequence[-1] in {"like", "love"}:
        return {"pizza": 0.8, "<end>": 0.2}
    else:
        return {"<end>": 1.0}
def greedy_search():
    """
    Greedy Search:
    At each step t, pick:
        w_t = argmax_w P(w | w_1, ..., w_{t-1})
    Only one path is followed based on local best choice.
    """
    sequence = []
    total_prob = 1.0
    while True:
        probs = get_next_token_probs(sequence)
        # Select the word with the highest conditional probability
        # Equivalent to: w_t = argmax P(w_t | context)
        next_token = max(probs, key=probs.get)

        # Multiply probabilities: P(sequence) = Π P(w_t | w_{<t})
        total_prob *= probs[next_token]
        sequence.append(next_token)
        if next_token == "<end>":
            break
    print(f"Greedy log-prob = log({total_prob:.4f}) = {np.log(total_prob):.4f}")
    return sequence

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_text = "The weather today"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Greedy decoding
output_ids = model.generate(input_ids, max_length=20)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))