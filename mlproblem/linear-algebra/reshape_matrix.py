import numpy as np
import torch
def reshape_matrix(a, new_shape):
    # Flatten the matrix
    flat = [elem for row in a for elem in row]
    rows, cols  = new_shape
    if len(flat) != rows * cols:
        return []
    
    #reshaped = [flat[i*cols:(i+1)*cols] for i in range(rows)]

    reshaped = []
    for i in range(rows):
        start = i * cols
        end = (i+1) * cols
        reshaped.append(flat[start:end])

def reshape_matrix_np(a, new_shape):
    arr = np.array(a)
    try:
        return arr.reshape(new_shape).tolist()
    except ValueError:
        return []
def reshape_matrix_torch(a, new_shape):
    tensor = torch.tensor(a)
    if tensor.numel() != new_shape[0] * new_shape[1]:
        return []
    return tensor.reshape(new_shape).tolist()