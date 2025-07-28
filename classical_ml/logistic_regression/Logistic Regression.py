import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification


def sigmoid(z):
    """
    Compute the Sigmoid function.

    Mathematically:
        σ(z) = 1 / (1 + exp(-z))

    This function maps any real number to the range (0,1), making it useful for probability estimation.

    Parameters:
        z (numpy array): Linear combination of weights and input features.

    Returns:
        numpy array: Sigmoid activation applied to input z.
    """
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, weights):
    """
    Compute the Binary Cross-Entropy (Log Loss) cost function.

    Mathematically:
        J(w) = - (1/m) * Σ [ y log(h) + (1 - y) log(1 - h) ]

    where:
        - h = σ(Xw) (Predicted probabilities)
        - y is the actual class (0 or 1)

    Parameters:
        X (numpy array): Feature matrix of shape (m, n).
        y (numpy array): Actual labels of shape (m,).
        weights (numpy array): Weight vector of shape (n,).

    Returns:
        float: Computed cost function value.
    """
    m = len(y)
    h = sigmoid(X @ weights)
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def gradient_descent(X, y, weights, learning_rate, iterations):
    """
    Perform Gradient Descent optimization.

    Updates weights using the formula:
        w := w - α * (1/m) * X^T (h - y)

    where:
        - h = σ(Xw) (Predicted probabilities)
        - α is the learning rate

    Parameters:
        X (numpy array): Feature matrix of shape (m, n).
        y (numpy array): Actual labels of shape (m,).
        weights (numpy array): Initial weight vector of shape (n,).
        learning_rate (float): Step size for gradient descent.
        iterations (int): Number of iterations.

    Returns:
        tuple: (Updated weights, Cost history)
    """
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X @ weights)  # Compute predictions
        gradient = (1 / m) * X.T @ (h - y)  # Compute gradient
        weights -= learning_rate * gradient  # Update weights
        cost = compute_cost(X, y, weights)  # Compute cost
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost:.4f}")

    return weights, cost_history


def predict(X, weights):
    """
    Make predictions using the learned weights.

    Decision rule:
        ŷ = 1  if σ(Xw) ≥ 0.5
        ŷ = 0  otherwise

    Parameters:
        X (numpy array): Feature matrix of shape (m, n).
        weights (numpy array): Trained weight vector of shape (n,).

    Returns:
        numpy array: Predicted binary labels (0 or 1).
    """
    probabilities = sigmoid(X @ weights)
    return (probabilities >= 0.5).astype(int)


# Generate synthetic binary classification dataset
X, y = make_classification(n_samples=500, n_features=2, random_state=42)

# Normalize features for better convergence
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Add bias term (intercept)
X = np.c_[np.ones(X.shape[0]), X]  # Shape becomes (500, 3)

# Initialize weights
weights = np.zeros(X.shape[1])

# Set hyperparameters
learning_rate = 0.1
iterations = 1000

# Train logistic regression model
weights, cost_history = gradient_descent(X, y, weights, learning_rate, iterations)

# Evaluate the model
y_pred = predict(X, weights)
accuracy = np.mean(y_pred == y) * 100
print(f"Accuracy: {accuracy:.2f}%")

# ---------------------- Plot Decision Boundary ----------------------

# Create mesh grid for decision boundary
x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict on grid points
Z = predict(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], weights)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 1], X[:, 2], c=y, edgecolors="k")
plt.title("Decision Boundary (Logistic Regression)")
plt.show()
