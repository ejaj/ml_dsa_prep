import numpy as np 
def cramers_rule(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    det_A = np.linalg.det(A)

    if np.isclose(det_A, 0):
        return -1
    n = A.shape[0]
    x = np.zeros(n)
    for i in range(n):
        A_i = A.copy()
        A_i[:, i] = b
        det_A_i = np.linalg.det(A_i)
        x[i] = det_A_i/det_A
    return np.round(x, 4)

