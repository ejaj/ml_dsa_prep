import numpy as np


def stochastic_gradient_descent(gradient_fn, X, learning_rate=0.01, iterations=100):
    """
    Implements Stochastic Gradient Descent (SGD).

    **Mathematical Formula:**
    -------------------------
    Update rule:
    x_new = x_old - α * ∇f(x_i)

    Where:
    - α (learning_rate) is the step size.
    - ∇f(x_i) is the gradient computed for **a single randomly chosen** sample.

    **Parameters:**
    - gradient_fn (function): Function that computes the gradient of f(X).
    - X (numpy array): Dataset containing multiple samples.
    - learning_rate (float): Step size.
    - iterations (int): Number of iterations.

    **Returns:**
    - float: Optimized value of x.

    **Example Usage:**
    # gradient_fn = lambda x: 2 * x  # f(x) = x^2 → f'(x) = 2x
    # X = np.array([5, 3, -2, 7])  # Dataset
    # stochastic_gradient_descent(gradient_fn, X, learning_rate=0.1, iterations=100)
    0.0
    """
    x = np.mean(X)  # Start with the mean of the dataset
    print(f"Initial x: {x}")
    for i in range(iterations):
        random_sample = np.random.choice(X)  # Randomly pick one sample
        gradient = gradient_fn(random_sample)  # Compute gradient on this sample
        x -= learning_rate * gradient  # Update x

        print(f"Iteration {i + 1}: x = {x}, Sample = {random_sample}, Gradient = {gradient}")
    return round(x, 5)


gradient_fn = lambda x: 2 * x  # f(x) = x^2 → f'(x) = 2x
X = np.array([5, 3, -2, 7])  # Dataset
print(stochastic_gradient_descent(gradient_fn, X, learning_rate=0.1, iterations=10))
