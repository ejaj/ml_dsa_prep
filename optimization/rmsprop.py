import numpy as np


def rmsprop(gradient_fn, X, learning_rate=0.01, iterations=100, beta=0.9, epsilon=1e-8):
    """
    Implements the RMSprop Optimization Algorithm.

    **Mathematical Formula:**
    -------------------------
    Update rule:
        E[g^2]_t = β * E[g^2]_{t-1} + (1 - β) * (∇f(x_t))^2
        x_new = x_old - (α / sqrt(E[g^2]_t + ε)) * ∇f(x_t)

    Where:
    - α (learning_rate) is the step size.
    - β (decay rate) controls how much past gradients influence updates.
    - E[g^2]_t is the exponentially decaying average of past squared gradients.
    - ε (epsilon) prevents division by zero.

    **Parameters:**
    - gradient_fn (function): Function that computes the gradient of f(x).
    - X (numpy array): Dataset containing multiple samples.
    - learning_rate (float): Step size.
    - iterations (int): Number of iterations.
    - beta (float): Decay rate for squared gradient moving average.
    - epsilon (float): Small value to avoid division by zero.

    **Returns:**
    - float: Optimized value of x.

    **Example Usage:**
    # gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
    # X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
    # rmsprop(gradient_fn, X, learning_rate=0.1, iterations=100)
    0.0
    """
    x = np.mean(X)  # Start with the mean of the dataset
    E_g2 = 0  # Initialize moving average of squared gradients

    print(f"Initial x: {x}")

    for i in range(iterations):
        gradient = np.mean(gradient_fn(X))  # Compute mean gradient over dataset
        E_g2 = beta * E_g2 + (1 - beta) * (gradient ** 2)  # Compute moving average of squared gradients
        adjusted_learning_rate = learning_rate / (np.sqrt(E_g2) + epsilon)  # Compute adaptive learning rate
        x -= adjusted_learning_rate * gradient  # Update x

        # Print x at each iteration for tracking
        print(
            f"Iteration {i + 1}: x = {x}, Gradient = {gradient}, E_g2 = {E_g2}, Adjusted LR = {adjusted_learning_rate}")

    return round(x, 5)


gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
print(rmsprop(gradient_fn, X, learning_rate=0.1, iterations=10))
