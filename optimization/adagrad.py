import numpy as np


def adagrad(gradient_fn, X, learning_rate=0.01, iterations=100, epsilon=1e-8):
    """
    Implements the Adagrad Optimization Algorithm.

    **Mathematical Formula:**
    -------------------------
    Accumulate squared gradients:
        G_t = G_{t-1} + (∇f(x_t))^2
    Update rule:
        x_new = x_old - (α / sqrt(G_t + ε)) * ∇f(x_t)

    Where:
    - α (learning_rate) is the step size.
    - G_t is the sum of squared past gradients.
    - ε (epsilon) is a small constant to prevent division by zero.

    **Parameters:**
    - gradient_fn (function): Function that computes the gradient of f(x).
    - X (numpy array): Dataset containing multiple samples.
    - learning_rate (float): Step size.
    - iterations (int): Number of iterations.
    - epsilon (float): Small value to avoid division by zero.

    **Returns:**
    - float: Optimized value of x.

    **Example Usage:**
    # gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
    # X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
    # adagrad(gradient_fn, X, learning_rate=0.1, iterations=100)
    0.0
    """
    x = np.mean(X)  # Initialize x as the mean of the dataset
    G_t = 0  # Initialize accumulated squared gradients

    print(f"Initial x: {x}")

    for i in range(iterations):
        gradient = np.mean(gradient_fn(X))  # Compute mean gradient over dataset
        G_t += gradient ** 2  # Accumulate squared gradients
        adjusted_learning_rate = learning_rate / (np.sqrt(G_t) + epsilon)  # Compute adaptive learning rate
        x -= adjusted_learning_rate * gradient  # Update x

        # Print x at each iteration for tracking
        print(f"Iteration {i + 1}: x = {x}, Gradient = {gradient}, G_t = {G_t}, Adjusted LR = {adjusted_learning_rate}")

    return round(x, 5)


gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
print(adagrad(gradient_fn, X, learning_rate=0.1, iterations=10))
