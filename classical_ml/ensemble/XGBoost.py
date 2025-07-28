import numpy as np

class DecisionStump:
    """
    A very simple decision tree with one split.
    It finds the best feature and threshold to minimize squared error.
    """
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_value = None
        self.right_value = None

    def fit(self, X, residuals):
        """
        Fit the stump to predict the residuals.
        """
        n_samples, n_features = X.shape
        min_error = float('inf')

        # Try all features and thresholds
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_mask = X[:, feature] <= thresh
                right_mask = ~left_mask

                # Predict the mean residual on each side
                left_pred = np.mean(residuals[left_mask]) if np.any(left_mask) else 0
                right_pred = np.mean(residuals[right_mask]) if np.any(right_mask) else 0

                # Compute squared error
                error = np.sum((residuals[left_mask] - left_pred)**2) + \
                        np.sum((residuals[right_mask] - right_pred)**2)

                if error < min_error:
                    min_error = error
                    self.feature_index = feature
                    self.threshold = thresh
                    self.left_value = left_pred
                    self.right_value = right_pred

    def predict(self, X):
        """
        Predict using the stump.
        """
        predictions = np.where(
            X[:, self.feature_index] <= self.threshold,
            self.left_value,
            self.right_value
        )
        return predictions


class MiniXGBoost:
    """
    A simplified XGBoost-like model using gradient boosting with decision stumps.
    Only first-order gradients (residuals) and squared error loss are used.
    """

    def __init__(self, n_estimators=10, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        """
        Fit the model by adding trees one at a time.
        Each new tree learns to predict the residuals (errors) of the current model.
        """
        y_pred = np.zeros_like(y, dtype=np.float64)

        for i in range(self.n_estimators):
            # Compute residuals (gradients of squared loss)
            residuals = y - y_pred

            # Train a decision stump on residuals
            stump = DecisionStump()
            stump.fit(X, residuals)

            # Predict residuals and update prediction
            update = stump.predict(X)
            y_pred += self.learning_rate * update

            # Save the model
            self.models.append(stump)

    def predict(self, X):
        """
        Predict the output by summing the outputs of all stumps.
        """
        y_pred = np.zeros(X.shape[0])
        for stump in self.models:
            y_pred += self.learning_rate * stump.predict(X)
        return y_pred

# Create some simple data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.2, 1.9, 3.0, 3.9, 5.1])

# Train the model
model = MiniXGBoost(n_estimators=5, learning_rate=0.5)
model.fit(X, y)

# Predict
preds = model.predict(X)
print("Predictions:", preds)


import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate sample regression data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([1.2, 1.9, 3.0, 3.9, 5.1, 6.1])

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Convert to DMatrix (XGBoost's internal data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for regression
params = {
    'objective': 'reg:squarederror',  # Squared loss (L2)
    'max_depth': 2,                   # Depth of each tree
    'eta': 0.1,                       # Learning rate
    'verbosity': 0                   # No training logs
}

# Train the model with 20 boosting rounds
model = xgb.train(params, dtrain, num_boost_round=20)

# Predict
y_pred = model.predict(dtest)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("Predictions:", y_pred)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from xgboost import XGBClassifier

# Initialize and train
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Predict and evaluate
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

print("XGBoost Accuracy:", xgb_acc)
