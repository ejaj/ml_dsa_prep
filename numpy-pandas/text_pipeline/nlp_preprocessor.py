import pandas as pd
import re
from langdetect import detect
import spacy


class TextPreprocessor:
    def __init__(self, language='en', lemmatize=True):
        self.language = language
        self.lemmatize = lemmatize
        self.nlp = spacy.load('en_core_web_sm') if lemmatize else None

    def clean_text(self, text):
        text = str(text).lower().strip()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        text = re.sub(r"\d+", "", text)  # Remove number
        return text

    def remove_emojis(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"
                                   u"\U0001F300-\U0001F5FF"
                                   u"\U0001F680-\U0001F6FF"
                                   u"\U0001F1E0-\U0001F1FF"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "unknown"
    def tokenize(self, text):
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    def transform(self, df, text_col='message'):
        df['clean_msg'] = df[text_col].apply(self.remove_emojis).apply(self.clean_text)
        df['language'] = df['clean_msg'].apply(self.detect_language)
        df = df[df['language'] == self.language]

        df['tokens'] = df['clean_msg'].apply(self.tokenize)
        df['msg_length'] = df['clean_msg'].str.len()
        df['word_count'] = df['tokens'].apply(len)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour

        return df[['user_id', 'tokens', 'msg_length', 'word_count', 'hour', 'category']]














