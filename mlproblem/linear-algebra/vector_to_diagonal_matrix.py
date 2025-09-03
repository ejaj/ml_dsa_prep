import numpy as np
import torch
from typing import Union

# def make_diagonal(x):
#     identity_matrix = np.identity(np.size(x))
#     return (identity_matrix*x)

def make_diagonal(x):
    if x.ndim != 1:
        raise ValueError("Input must be a 1D numpy array")
    return np.diag(x)
def make_diagonal_pytorch(x: Union[torch.Tensor, list, np.ndarray]) -> torch.Tensor:
    """Return a diagonal matrix whose diagonal elements are the 1-D values in `x`.
    If `x` is not a torch tensor it will be converted automatically.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.diag_embed(x)

# x = np.array([1, 2, 3])
# output = make_diagonal(x)
# print(output)

x = [1, 2, 3]
output = make_diagonal_pytorch(x)
print(output)
