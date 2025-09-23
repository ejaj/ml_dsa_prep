import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Convert input to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Normal equation: θ = (X^T X)^(-1) X^T y
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # use pinv for safety

    # Round to 4 decimal places
    theta = np.round(theta, 4)

    # Return as list
    return theta.tolist()
