import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mathematically: F₀(x) = argmin_c Σ L(yᵢ, c)
# For MSE, it's simply mean(y)

y_pred_train = np.full_like(y_train, fill_value=np.mean(y_train), dtype=np.float64)
y_pred_test = np.full_like(y_test, fill_value=np.mean(y_train), dtype=np.float64)

train_preds = [y_pred_train.copy()]
test_preds = [y_pred_test.copy()]

n_estimators = 5         # Number of boosting rounds (trees)
learning_rate = 0.1      # α: learning rate
trees = []

for m in range(n_estimators):
    """
    Compute pseudo-residuals
    For MSE: L = ½(y - F(x))²
    Gradient: ∂L/∂F(x) = F(x) - y ⇒ Residual = y - F(x)
    """
    residuals = y_train - y_pred_train
    """
    Fit regression tree hₘ(x) to residuals
    """
    tree  = DecisionTreeRegressor(max_depth=3)
    tree.fit(X_train, residuals)
    trees.append(tree)

    """
    Update model
    Fₘ(x) = Fₘ₋₁(x) + α * hₘ(x)
    """
    update_train = learning_rate * tree.predict(X_train)
    update_test = learning_rate * tree.predict(X_test)

    y_pred_train += update_train
    y_pred_test += update_test
    train_preds.append(y_pred_train.copy())
    test_preds.append(y_pred_test.copy())


train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(y_test, label="True values", color="black", marker='o', linestyle='None', alpha=0.5)
plt.plot(y_pred_test, label=f"Final prediction (MSE: {test_mse:.2f})", color="blue", marker='x', linestyle='None')
plt.title("Gradient Boosted Trees – Final Predictions")
plt.xlabel("Test sample index")
plt.ylabel("Target")
plt.legend()
plt.grid(True)
plt.show()

from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import load_breast_cancer
from scipy.spatial import expit

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize F0(x) — log odds from mean

prob = np.mean(y_train)  # Prior probability of class 1
F_train = np.full_like(y_train, fill_value=np.log(prob / (1 - prob)), dtype=np.float64)
F_test = np.full_like(y_test, fill_value=np.log(prob / (1 - prob)), dtype=np.float64)

n_estimators = 10
learning_rate = 0.1
trees = []

for m in range(n_estimators):
    # Compute predicted probabilities
    p_train = expit(F_train)  # sigmoid(F)
    
    # Compute pseudo-residuals: r_i = y_i - p_i
    residuals = y_train - p_train
    
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(X_train, residuals)
    trees.append(tree)

    # Update F(x)
    F_train += learning_rate * tree.predict(X_train)
    F_test += learning_rate * tree.predict(X_test)

# Final predictions: convert F(x) → probabilities using sigmoid
p_test = expit(F_test)
y_pred = (p_test >= 0.5).astype(int)

# Evaluation
acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, p_test)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test Log Loss: {loss:.4f}")