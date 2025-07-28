import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # Or 'Agg', 'Qt5Agg', depending on your system

X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
Y = np.array([1, 3, 5, 7], dtype=np.float64)

w = np.array([0.5, 0.5], dtype=np.float64)  # Initial weight values

# Learning rate α (controls step size in gradient descent)
alpha = 0.01

# Number of iterations (how many times to update weights)
num_iterations = 100

# Store loss at each iteration for visualization
loss_history = []


def predict(X, w):
    """
    Computes the predicted values (ŷ) using the linear model:

    ŷ = X * w

    where:
    - X is the feature matrix (Nx2)
    - w is the weight vector (2x1)
    - ŷ is the predicted output (Nx1)
    """
    return np.matmul(X, w)


def compute_error(Y, y_pred):
    """
    Computes the error (difference between actual and predicted values):

        error = Y - ŷ

    where:
    - Y is the actual ground truth values (Nx1)
    - ŷ is the predicted output (Nx1)
    """
    return Y - y_pred


def compute_gradient(X, error):
    """
    Computes the gradient of the cost function w.r.t. each weight:

        ∂J/∂w_j = - (2/N) * Σ (y_i - ŷ_i) * x_ij

    where:
    - J(w) is the Mean Squared Error (MSE)
    - N is the number of training samples
    - y_i and ŷ_i are actual and predicted values
    - x_ij is the feature value for weight w_j

    The gradient tells us how much each weight should be updated.
    """
    N = len(X)
    return (-2 / N) * np.dot(X.T, error)  # Computes the sum for all training samples


def update_weights(w, gradient, alpha):
    """
    Updates each weight w_j using the gradient descent rule:

        w_j = w_j - α * (∂J/∂w_j)

    where:
    - α (alpha) is the learning rate
    - ∂J/∂w_j is the computed gradient for weight w_j
    """
    return w - alpha * gradient


for _ in range(num_iterations):
    y_pred = predict(X, w)  # Compute predictions
    error = compute_error(Y, y_pred)  # Compute error
    gradient = compute_gradient(X, error)  # Compute gradient
    w = update_weights(w, gradient, alpha)  # Update weights

    # Compute loss (Mean Squared Error)
    loss = np.mean(np.square(error))
    loss_history.append(loss)

print("Final Weights:", np.round(w, 5))

plt.plot(range(num_iterations), loss_history, label="Loss Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Loss Reduction Over Training")
plt.legend()
plt.show()

X_test = np.array([[5, 6], [6, 7]], dtype=np.float64)  # New test samples

# Compute predictions for new samples
y_test_pred = predict(X_test, w)

print("Predictions for New Data:", y_test_pred)
