import numpy as np 
import torch
def transpose_matrix(a):
    return [list(row) for row in zip(*a)]

a = [[1, 2, 3], [4, 5, 6]]
print(transpose_matrix(a))  

def transpose_matrix_numpy(a):
    return np.array(a).T.tolist()
def transpose_matrix(a):
    return torch.tensor(a).T.tolist()
