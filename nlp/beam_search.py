import numpy as np

def get_next_token_probs(sequence):
    if sequence == []:
        return {"I": 0.6, "You": 0.4}
    elif sequence[-1] == "I":
        return {"love": 0.6, "like": 0.4}
    elif sequence[-1] in {"love", "like"}:
        return {"pizza": 0.8, "<end>": 0.2}
    else:
        return {"<end>": 1.0}

def beam_search(beam_width=2, max_len=4):
    """
    Beam Search: keeps top-k sequences at each step based on log-probability.
    """
    beams = [([], 0.0)] # sequence, log-prob
    for _ in range(max_len):
        candidates = []
        for seq, log_prob in beams:
            # If already ended, don't expand
            if seq and seq[-1] == "<end>":
                candidates.append((seq, log_prob))
                continue
            probs = get_next_token_probs(seq)

            for word, prob in probs.items():
                new_seq = seq + [word]
                new_seq_log = log_prob + np.log(prob)
                candidates.append((new_seq, new_seq_log))
            # Keep top-k candidates
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            # Stop early if all sequences ended
            if all(seq[-1] == "<end>" for seq, _ in beams):
                break
    best_seq, best_log_prob = beams[0]
    print(f"Best sequence: {best_seq}")
    print(f"Log-probability: {best_log_prob:.4f}")
    return best_seq

beam_search(beam_width=2)

from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_text = "The weather today"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output_ids = model.generate(
    input_ids,
    max_length=20,
    num_beams=5,
    early_stopping=True
)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
