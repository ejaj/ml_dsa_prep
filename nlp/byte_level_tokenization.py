text = "café"

# Convert the text to its byte representation using UTF-8 encoding
byte_tokens = list(text.encode('utf-8'))

# Print the byte tokens
print("Text:", text)
print("Byte Tokens:", byte_tokens)

# Convert the byte tokens back to the original text
decoded_text = bytes(byte_tokens).decode('utf-8')
print("Decoded Text:", decoded_text)

from transformers import RobertaTokenizer

# Load pre-trained Roberta tokenizer (uses Byte-Level BPE Tokenization)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Sample text
text = "café"

# Tokenizing the text at the byte-level
encoded_text = tokenizer.encode(text, add_special_tokens=False)

# Decoding the byte-level tokens back to text
decoded_text = tokenizer.decode(encoded_text)

# Display results
print(f"Original Text: {text}")
print(f"Encoded Byte-Level Tokens: {encoded_text}")
print(f"Decoded Text: {decoded_text}")
