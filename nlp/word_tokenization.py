import nlk
from nltk.tokenize import word_tokenize, RegexpTokenizer

text = "Hello, world! NLP is fun."
tokens = word_tokenize(text)
print(tokens)

import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp(text)
tokens = [token.text for token in doc]
print(tokens)
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokens = tokenizer.tokenize(text)
print(tokens)

# Resume parser / Named Entity Recognition

resume = "John Smith is a data scientist at OpenAI. He lives in Copenhagen and speaks Danish fluently."
doc = nlp(resume)
print("Named Entities:")
for ent in doc.ents:
    print(ent.text, ent.label_)

from transformers import BertTokenizer, BertForSequenceClassification
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
text = "The laptop is incredibly fast and lightweight!"

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
predicted_class = torch.argmax(outputs.logits, dim=1).item()
print("Predicted rating:", predicted_class + 1)