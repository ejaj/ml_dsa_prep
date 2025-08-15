import numpy as np 
import torch
def matrix_vector_dot(a, b):
    if not a or not a[0]:
        return -1
    
    num_cols = len(a[0])
    if num_cols != len(b):
        return -1
    result = []
    for row in a:
        dot_sum = sum(row[i] * b[i] for i in range(num_cols))
        result.append(dot_sum)
    return result

    return [sum(row[i] * b[i] for i in range(num_cols)) for row in a]

# numpy
def matrix_dot_vector_numpy(a, b):
    a = np.array(a)
    b = np.array(b)

    if a.shape[1] != b.shape[0]:
        return -1
    return a.dot(b).tolist()
def matrix_dot_vector_torch(a, b):
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    if a.shape[1] != b.shape[0]:
        return -1
    return torch.mv(a,b).tolist()


a = [[1, 2], [2, 4]]
b = [1, 2]
print(matrix_vector_dot(a, b))  # Output: [5, 10]
