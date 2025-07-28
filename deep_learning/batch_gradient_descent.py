import numpy as np


def batch_gradient_descent(gradient_fn, X, learning_rate=0.01, num_iterations=100):
    x = np.mean(X)  # Initialize x as the mean of the batch
    print(f"Initial x: {x}")  # Track initial value of x

    for i in range(num_iterations):
        gradient = np.mean(gradient_fn(X))  # Compute batch gradient
        x -= learning_rate * gradient
        # Print x at each iteration for tracking
        print(f"Iteration {i + 1}: x = {x}, Gradient = {gradient}")
    return round(x, 5)


gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
X = np.array([5, 3, -2, 7])  # Batch of data
print(batch_gradient_descent(gradient_fn, X, learning_rate=0.1, iterations=10))
