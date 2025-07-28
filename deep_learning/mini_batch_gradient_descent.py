import numpy as np


def mini_batch_gradient_descent(gradient_fn, X, learning_rate=0.01, iterations=100, batch_size=10):
    """
    Implements Mini-Batch Gradient Descent (MBGD).

    **Mathematical Formula:**
    -------------------------
    Update rule:
        x_new = x_old - α * (1/b) * Σ (∇f(x_i))

    Where:
    - α (learning_rate) is the step size.
    - b (batch_size) is the number of samples in a batch.
    - ∇f(x_i) is the gradient of f(x) computed over a mini-batch.

    **Parameters:**
    - gradient_fn (function): Function that computes the gradient of f(x).
    - X (numpy array): Dataset containing multiple samples.
    - learning_rate (float): Step size.
    - iterations (int): Number of iterations.
    - batch_size (int): Number of samples per batch.

    **Returns:**
    - float: Optimized value of x.

    **Example Usage:**
    # gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
    # X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
    # mini_batch_gradient_descent(gradient_fn, X, learning_rate=0.1, iterations=100, batch_size=3)
    0.0
    """
    x = np.mean(X)  # Start with the mean of the dataset
    print(f"Initial x: {x}")

    for i in range(iterations):
        mini_batch = np.random.choice(X, batch_size, replace=False)  # Select a random mini-batch
        batch_gradient = np.mean(gradient_fn(mini_batch))  # Compute mean gradient over mini-batch
        x -= learning_rate * batch_gradient  # Update x

        # Print x at each iteration for tracking
        print(f"Iteration {i + 1}: x = {x}, Mini-batch = {mini_batch}, Gradient = {batch_gradient}")

    return round(x, 5)


# gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
# X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
# print(mini_batch_gradient_descent(gradient_fn, X, learning_rate=0.1, iterations=10, batch_size=3))


def mini_batch_gradient_descent_1(X, Y, num_iterations, initial_weights, batch_size=2, learning_rate=0.01):
    """
    Implements Mini-Batch Gradient Descent (MBGD).
    Updates weights based on a subset of training examples (batch).
    """
    weights = np.array(initial_weights, dtype=np.float64)
    N = len(X)

    # Ensure num_iterations is an integer
    num_iterations = int(num_iterations)

    for _ in range(num_iterations):  # This line was causing the issue
        indices = np.random.permutation(N)  # Shuffle dataset indices
        X_shuffled, Y_shuffled = X[indices], Y[indices]

        for i in range(0, N, batch_size):  # Process each batch
            X_batch = X_shuffled[i:i + batch_size]
            Y_batch = Y_shuffled[i:i + batch_size]

            predictions = np.matmul(X_batch, weights)  # Compute predictions for batch
            gradient = (-2 / len(X_batch)) * np.dot(X_batch.T, (Y_batch - predictions))  # Compute mini-batch gradient
            weights -= learning_rate * gradient  # Update weights

    return np.round(weights, 5)


X = np.array([
    [1, 2, 3],
    [1, 1, 1],
    [2, 3, 4],
    [3, 2, 1]
], dtype=np.float64)

Y = np.array([6, 3, 9, 8], dtype=np.float64)  # Ground truth values

initial_weights = np.array([0.2, 0.1, 0.6], dtype=np.float64)
num_iterations = 10
batch_size = 2

final_weights = mini_batch_gradient_descent_1(X, Y, num_iterations, initial_weights, batch_size)
print("Final Weights:", final_weights)
