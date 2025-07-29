import re
import collections

# Build initial vocabulary (character-level with </w>)
def get_vocab(corpus):
    vocab = collections.defaultdict(int)
    for word in corpus:
        tokens = list(word) + ['</w>']  # Append end-of-word token
        vocab[' '.join(tokens)] += 1
    return vocab

# Count symbol pairs (bigrams)
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

# Merge the most frequent pair
def merge_vocab(pair, vocab):
    new_vocab = {}
    pattern = re.escape(' '.join(pair))
    pattern = re.compile(r'(?<!\S)' + pattern + r'(?!\S)')
    for word in vocab:
        new_word = pattern.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

# BPE merge loop
def learn_bpe(corpus, num_merges):
    vocab = get_vocab(corpus)
    merges = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_vocab(best, vocab)
    return merges

# Encoding function
def encode_bpe(word, merges):
    word = list(word) + ['</w>']
    while True:
        pairs = [(word[i], word[i+1]) for i in range(len(word) - 1)]
        merge_candidate = None
        for merge in merges:
            if merge in pairs:
                merge_candidate = merge
                break
        if merge_candidate is None:
            break
        i = pairs.index(merge_candidate)
        word = word[:i] + [''.join(merge_candidate)] + word[i+2:]
    return word

#  Decoding function
def decode_bpe(tokens):
    return ''.join([t.replace('</w>', '') for t in tokens])

# Mini training corpus
corpus = ["low", "lower", "newest", "widest"]
num_merges = 10

# Train BPE merges
merges = learn_bpe(corpus, num_merges)
print("\nLearned merges:")
print(merges)

# Encode a word using BPE
word = "lowest"
encoded = encode_bpe(word, merges)
print("\nEncoded:", encoded)

# Decode the tokens
decoded = decode_bpe(encoded)
print("Decoded:", decoded)

from transformers import AutoTokenizer

# Load a BPE-based tokenizer (e.g., RoBERTa or GPT-2)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Your input sentence
text = "the highest mountain"

# -------- ENCODING --------
# Convert sentence into token IDs (subword tokens)
encoded = tokenizer(text)
print("Encoded token IDs:", encoded['input_ids'])
print("Encoded tokens:", tokenizer.convert_ids_to_tokens(encoded['input_ids']))

# -------- DECODING --------
# Convert token IDs back to original text
decoded = tokenizer.decode(encoded['input_ids'])
print("Decoded text:", decoded)
