text = "Tokenization is the first step in NLP."
tokens = text.split()
print(tokens)

import nltk
from nltk.tokenize import WhitespaceTokenizer
text = "Tokenization is the first step in NLP."

tokenizer = WhitespaceTokenizer()
tokens = tokenizer.tokenize(text)
print(tokens)

import re
text = "Tokenization is the first step in NLP."
tokens = re.split(r'\s+', text)
print(tokens)
tokens_find = re.findall(r'\b\w\w', text)
print(tokens_find)

# Search engine keyword match
review = "This camera has excellent battery life and great image quality."
query = "battery camera"

review_token = review.lower().split()
query_token = query.lower().split()

matches = [word for word in query_token if word in review_token]
print("Matched terms:", matches)















