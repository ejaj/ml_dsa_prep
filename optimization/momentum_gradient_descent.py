import numpy as np


def momentum_gradient_descent(gradient_fn, X, learning_rate=0.01, iterations=100, beta=0.9):
    """
    Implements Momentum-Based Gradient Descent.

    **Mathematical Formula:**
    -------------------------
    Velocity update:
        v_t = β * v_{t-1} - α * ∇f(x_t)
    Parameter update:
        x_new = x_old + v_t

    Where:
    - β (momentum) controls how much past gradients influence the update.
    - α (learning_rate) is the step size.

    **Parameters:**
    - gradient_fn (function): Function that computes the gradient of f(x).
    - X (numpy array): Dataset containing multiple samples.
    - learning_rate (float): Step size.
    - iterations (int): Number of iterations.
    - beta (float): Momentum factor.

    **Returns:**
    - float: Optimized value of x.

    **Example Usage:**
    ## gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
    ## X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
    ## momentum_gradient_descent(gradient_fn, X, learning_rate=0.1, iterations=100, beta=0.9)
    0.0
    """
    x = np.mean(X)  # Start with the mean of the dataset
    velocity = 0  # Initialize velocity

    print(f"Initial x: {x}")

    for i in range(iterations):
        gradient = np.mean(gradient_fn(X))  # Compute mean gradient over dataset
        velocity = beta * velocity - learning_rate * gradient  # Update velocity
        x += velocity  # Update x

        # Print x at each iteration for tracking
        print(f"Iteration {i + 1}: x = {x}, Gradient = {gradient}, Velocity = {velocity}")

    return round(x, 5)


gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
print(momentum_gradient_descent(gradient_fn, X, learning_rate=0.1, iterations=10, beta=0.9))
