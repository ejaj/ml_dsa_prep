import math
import numpy as np 
import torch
def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # Extract elements
    a, b = matrix[0]
    c, d = matrix[1]

    # trace and determinant
    trace = a + d
    determinant = a * d - b * c

    # discriminant
    discriminant = trace**2 - 4 * determinant

    # eigenvalues using quadratic formula
    eig1 = (trace + math.sqrt(discriminant)) / 2
    eig2 = (trace - math.sqrt(discriminant)) / 2

    # sort from highest to lowest
    return sorted([eig1, eig2], reverse=True)


matrix = [[2, 1], [1, 2]]
print(calculate_eigenvalues(matrix))  # [3.0, 1.0]

def calculate_eigenvalues_numpy(matrix: list[list[float | int]]) -> list[float]:
    arr = np.array(matrix, dtype=float)
    eigenvalues, _ = np.linalg.eig(arr)   # eig() returns (eigenvalues, eigenvectors)
    return sorted(eigenvalues.tolist(), reverse=True)

def calculate_eigenvalues_torch(matrix: list[list[float | int]]) -> list[float]:
    tensor = torch.tensor(matrix, dtype=torch.float32)
    eigenvalues, _ = torch.linalg.eig(tensor)  # Returns complex eigenvalues
    # Convert to real numbers (imag part = 0 for symmetric real matrices)
    eigenvalues = eigenvalues.real.tolist()
    return sorted(eigenvalues, reverse=True)
