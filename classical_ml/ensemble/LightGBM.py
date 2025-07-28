from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train, y_train)

# Predict and evaluate
lgbm_preds = lgbm_model.predict(X_test)
lgbm_acc = accuracy_score(y_test, lgbm_preds)

print("LightGBM Accuracy:", lgbm_acc)
