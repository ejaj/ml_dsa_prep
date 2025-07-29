import re
from transformers import AutoTokenizer

text = "July 2025"
tokens = list(text)
print(tokens)

cleaned = re.sub(r'[^A-Za-z0-9]','', text)
tokens = list(cleaned)
print(tokens)
tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
tokens = tokenizer("Hello!", return_tensors="pt")
print(tokens.input_ids)

import spacy
nlp = spacy.load("en_core_web_sm")
text = "NLP is cool!"
doc = nlp(text)

normalized_text = doc.text
char_tokens = list(normalized_text)
print(char_tokens)

text = "Deep learning rocks!"
doc = nlp(text)

for token in doc:
    print(f"{token.text} → {list(token.text)}")
chars_only = [char for char in list(doc.text) if char.isalnum()]
print(chars_only)