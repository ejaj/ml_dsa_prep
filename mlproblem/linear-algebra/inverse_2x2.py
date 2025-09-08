import torch

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    
   # Extract values from the matrix
    a, b = matrix[0]
    c, d = matrix[1]
    
    # Calculate determinant
    det = a * d - b * c
    
    # If determinant is zero, matrix is not invertible
    if det == 0:
        return None
    
    # Apply formula for inverse
    inv_matrix = [[d / det, -b / det],
                  [-c / det, a / det]]
    
    return inv_matrix


def inverse_2x2_pytroch(matrix: list[list[float]]) -> list[list[float]]:
    m =  torch.tensor(matrix, dtype=torch.float32)
     # Calculate determinant
    
    det = torch.det(m)
    # If determinant is zero, matrix is not invertible
    if det.item() == 0:
        return None
    inverse = torch.linalg.inv(m)
    return inverse.tolist()
    
