import numpy as np
import torch


def transform_matrix(A, T, S):
    # Convert to numpy arrays
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)

    # Get shape of A
    n, m = A.shape

    # Check dimensions
    if T.shape != (n, n) or S.shape != (m, m):
        return -1

    # Check if T and S are invertible
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        return -1

    # Calculate transformation
    T_inv = np.linalg.inv(T)
    result = T_inv @ A @ S  # '@' means matrix multiplication

    #  Return as list of lists
    return result.tolist()


def transform_matrix_pytorch(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform Tâ»Â¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2Ã2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # --- shape checks ---
    if A_t.dim() != 2 or T_t.dim() != 2 or S_t.dim() != 2:
        return torch.tensor(-1.0)

    n, m = A_t.shape
    if T_t.shape != (n, n) or S_t.shape != (m, m):
        return torch.tensor(-1.0)

    # --- invertibility checks (full rank ⇒ invertible for square matrices) ---
    if torch.linalg.matrix_rank(T_t) < n or torch.linalg.matrix_rank(S_t) < m:
        return torch.tensor(-1.0)

    # --- compute T^{-1} A via solve (safer than forming T^{-1}) ---
    X = torch.linalg.solve(T_t, A_t)   # solves T * X = A  ⇒ X = T^{-1} A

    # --- multiply by S ---
    out = X @ S_t

    return out

