import numpy as np


def adam(gradient_fn, X, learning_rate=0.01, iterations=100, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Implements the Adam Optimization Algorithm.

    **Mathematical Formula:**
    -------------------------
    First moment estimate (Momentum-like):
        m_t = β1 * m_{t-1} + (1 - β1) * ∇f(x_t)
    Second moment estimate (RMSprop-like):
        v_t = β2 * v_{t-1} + (1 - β2) * (∇f(x_t))^2
    Bias correction:
        m_hat_t = m_t / (1 - β1^t)
        v_hat_t = v_t / (1 - β2^t)
    Parameter update:
        x_new = x_old - (α / sqrt(v_hat_t) + ε) * m_hat_t

    **Parameters:**
    - gradient_fn (function): Function that computes the gradient of f(x).
    - X (numpy array): Dataset containing multiple samples.
    - learning_rate (float): Step size.
    - iterations (int): Number of iterations.
    - beta1 (float): Momentum decay factor.
    - beta2 (float): RMSprop decay factor.
    - epsilon (float): Small value to avoid division by zero.

    **Returns:**
    - float: Optimized value of x.

    **Example Usage:**
    ## gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
    ## X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
    ## adam(gradient_fn, X, learning_rate=0.1, iterations=100)
    0.0
    """
    x = np.mean(X)  # Start with the mean of the dataset
    m_t, v_t = 0, 0  # Initialize first and second moment estimates

    print(f"Initial x: {x}")

    for t in range(1, iterations + 1):
        gradient = np.mean(gradient_fn(X))  # Compute mean gradient over dataset

        # Compute biased first and second moment estimates
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * (gradient ** 2)

        # Compute bias-corrected first and second moments
        m_hat_t = m_t / (1 - beta1 ** t)
        v_hat_t = v_t / (1 - beta2 ** t)

        # Compute adaptive learning rate
        x -= learning_rate * m_hat_t / (np.sqrt(v_hat_t) + epsilon)

        # Print x at each iteration for tracking
        print(f"Iteration {t}: x = {x}, Gradient = {gradient}, m_t = {m_t}, v_t = {v_t}")

    return round(x, 5)


gradient_fn = lambda X: 2 * X  # f(X) = X^2 → f'(X) = 2X
X = np.array([5, 3, -2, 7, 1, -1, 4, 6, -3, 2])  # Dataset
print(adam(gradient_fn, X, learning_rate=0.1, iterations=10))
