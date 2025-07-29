def wordpiece_tokenize(word, vocab):
    tokens = []
    start = 0
    while start < len(word):
        end = len(word)
        match = None
        while start < end:
            substr = word[start:end]
            if start > 0:
                substr = "##" + substr
            if substr in vocab:
                match = substr
                break
            end -= 1
        if match in None:
            return ["[UNK]"]
        tokens.append(match)
        start = end if match.startswith("##") else len(match)
    return tokens
vocab = {"un", "##happy", "##happi", "##ness", "happy", "##y", "[UNK]"}

word = "unhappiness"
tokens = wordpiece_tokenize(word, vocab)
print(tokens)

# Function to initialize the vocabulary (predefined)
def initialize_vocab():
    """
    Predefined vocabulary with subwords and their frequencies.
    """
    return {
        "I": 5,
        "love": 4,
        "mach": 3,
        "in": 2,
        "e": 2,
        "learning": 4,
        "[UNK]": 1  # Unknown token
    }

# Function to find the longest matching subword from the vocabulary
def find_longest_match(token, vocab):
    """
    Find the longest subword in the vocabulary that matches the given token.
    """
    for i in range(len(token), 0, -1):
        sub = token[:i]
        if sub in vocab:
            return sub
    return None

# Function to tokenize a single word into subwords
def tokenize_word(word, vocab, unk_token="[UNK]", subword_prefix="##"):
    """
    Tokenizes a single word into subwords using the predefined vocabulary.
    """
    token = list(word)  # Start by treating the word as a list of characters
    subwords = []
    
    # Try to find longest subwords
    while token:
        match = find_longest_match("".join(token), vocab)
        if match:
            # If it's not the first subword, add "##" to indicate it's a continuation
            if subwords:
                subwords.append(subword_prefix + match)
            else:
                subwords.append(match)
            token = token[len(match):]  # Remove matched subword from token
        else:
            subwords.append(unk_token)  # Unknown token if no match
            break  # If no match, use [UNK] and break
        
    return subwords

# Function to tokenize a sentence
def tokenize_sentence(sentence, vocab, unk_token="[UNK]", subword_prefix="##"):
    """
    Tokenizes a sentence into subwords using the predefined vocabulary.
    """
    words = sentence.split()
    all_subwords = []
    
    for word in words:
        subwords = tokenize_word(word, vocab, unk_token, subword_prefix)
        all_subwords.extend(subwords)
    
    return all_subwords

# Example usage
vocab = initialize_vocab()

# Test sentence
sentence = "I love machine learning"

# Tokenize the sentence
tokens = tokenize_sentence(sentence, vocab)

# Output the result
print("Tokens:", tokens)

from transformers import BertTokenizer

# Step 2: Load a pre-trained BERT tokenizer (which uses WordPiece)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 3: Tokenize a sentence
sentence = "I love using Hugging Face for NLP tasks!"

# Tokenizing the sentence using WordPiece
tokens = tokenizer.tokenize(sentence)

# Print the tokenized output (WordPiece subwords)
print("Tokenized Sentence:", tokens)

# Convert tokens back to IDs for model input (if needed)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print("Token IDs:", token_ids)

# Optionally, convert token IDs back to a sentence
decoded_sentence = tokenizer.decode(token_ids)
print("Decoded Sentence:", decoded_sentence)
