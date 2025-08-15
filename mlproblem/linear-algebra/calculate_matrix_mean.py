import numpy as np 
import torch

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        return []
    if mode == "row":
        return[sum(row) / len(row) for row in matrix]
    elif mode == "column":
        num_cols = len(matrix[0])
        return [sum(matrix[i][j] for i in range(len(matrix))) / len(matrix) for j in range(num_cols)]
    else:
        raise ValueError("Mode must be either 'row' or 'column'")

def calculate_matrix_mean_np(matrix: list[list[float]], mode: str) -> list[float]:
    arr = np.array(matrix, dtype=float)
    
    if mode == "row":
        return arr.mean(axis=1).tolist()
    elif mode == "column":
        return arr.mean(axis=0).tolist()
    else:
        raise ValueError("Mode must be either 'row' or 'column'")

def calculate_matrix_mean_torch(matrix: list[list[float]], mode: str) -> list[float]:
    t = torch.tensor(matrix, dtype=torch.float32)
    
    if mode == "row":
        return t.mean(dim=1).tolist()
    elif mode == "column":
        return t.mean(dim=0).tolist()
    else:
        raise ValueError("Mode must be either 'row' or 'column'")



m = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
]
print(calculate_matrix_mean(m, "row"))     # [2.0, 5.0]
print(calculate_matrix_mean(m, "column"))  # [2.5, 3.5, 4.5]