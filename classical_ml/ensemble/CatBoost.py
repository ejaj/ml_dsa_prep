from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from catboost import CatBoostClassifier

# Initialize and train
catboost_model = CatBoostClassifier(verbose=0)  # silent output
catboost_model.fit(X_train, y_train)

# Predict and evaluate
catboost_preds = catboost_model.predict(X_test)
catboost_acc = accuracy_score(y_test, catboost_preds)

print("CatBoost Accuracy:", catboost_acc)
