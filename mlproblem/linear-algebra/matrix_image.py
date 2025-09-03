import numpy as np

def matrix_image(A, tol=1e-12):
    A = np.asarray(A)
    m, n = A.shape

    R = A.astype(float).copy() 
    pivots = []
    row = 0

    for col in range(n):
        if row >= m:
            break

        # Find pivot row (partial pivoting)
        pivot = row + np.argmax(np.abs(R[row:, col]))
        if np.abs(R[pivot, col]) <= tol:
            continue  # no pivot in this column

        # Swap pivot row into position
        if pivot != row:
            R[[row, pivot]] = R[[pivot, row]]

        # Normalize pivot row
        R[row] = R[row] / R[row, col]

        # Eliminate below pivot
        for r in range(row + 1, m):
            R[r] = R[r] - R[r, col] * R[row]

        pivots.append(col)
        row += 1

    # Return the original columns that correspond to pivot columns
    return A[:, pivots]