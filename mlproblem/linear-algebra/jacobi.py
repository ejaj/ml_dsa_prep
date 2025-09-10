import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    m = len(b)
    x = np.zeros(m)

    for _ in range(n):
        x_new = np.zeros(m)
        for i in range(m):
            # Sum of A[i][j] * x[j] for all j except i
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            # Jacobi formula
            x_new[i] = (b[i] - s) / A[i, i]
        x = np.round(x_new, 4)
    return x