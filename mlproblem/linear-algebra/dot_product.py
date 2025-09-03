import numpy as np
import torch

def calculate_dot_product(vec1, vec2) -> float: 
    """ 
    Calculate the dot product of two vectors. Args: vec1 (numpy.ndarray): 1D array representing the first vector. vec2 (numpy.ndarray): 1D array representing the second vector. 
    """ 
    if vec1.shape != vec2.shape: 
        raise ValueError("Vectors must have the same dimensions")
    return np.dot(vec1, vec2)


def dot_product_torch(vec1, vec2):
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must have the same dimensions")
    
    return torch.dot(vec1, vec2)

vec1 = torch.tensor([1, 2, 3], dtype=torch.float32)
vec2 = torch.tensor([4, 5, 6], dtype=torch.float32)
result = dot_product_torch(vec1, vec2)

print("Dot Product:", result.item())  # Use .item() to convert tensor to Python number