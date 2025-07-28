import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df = pd.read_parquet('preprocessed.parquet')
X = df['clean_msg']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# TF-IDF + Logistic Regression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression(max_iter=500))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))
