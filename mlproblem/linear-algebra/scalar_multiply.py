import numpy as np
import torch

def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    result = [[elm * scalar for elm in row] for row in matrix]
    return result

matrix = [[1, 2], [3, 4]]
scalar = 2
print(scalar_multiply(matrix, scalar))  # Output: [[2, 4], [6, 8]]

def scalar_multiply_numpy(matrix: list[list[int | float]], scalar: int | float) -> np.ndarray:
    arr = np.array(matrix)        # Convert list to NumPy array
    return arr * scalar           # Broadcasting does element-wise multiplication
def scalar_multiply_torch(matrix: list[list[int | float]], scalar: int | float) -> torch.Tensor:
    tensor = torch.tensor(matrix, dtype=torch.float32)
    return tensor * scalar

