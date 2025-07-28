import torch

# Create a 2x3 tensor on the CPU without initializing the values
# This means the tensor will contain whatever values were already in that memory location (random garbage)
# x = torch.empty(2, 3)
# Common Use Case:
# x.uniform_(0, 1)  # Now it's filled with random values between 0 and 1

# x = torch.rand(2,3) # [0, 1)	Uniform random floats
# print(x)
# # Normal (Gaussian) random floats with mean=0, std=1
# x2 = torch.randn(2, 3)
# print(x2)
# # Random integers from 0 to 9
# x3 = torch.randint(0, 10, (2, 3))
# print(x3)

# x1 = torch.zeros(2,3)	# 0.0	All elements are 0	torch.zeros(2, 3)
# x2 = torch.ones(2,3)	# 1.0	All elements are 1	torch.ones(2, 3)

# print(x1.size())
# print(x2)
# x3 = torch.full((2,3), 7) # Custom value, All elements are the same value
# print(x3)
# xi = torch.eye(3)	# Identity matrix	1s on diagonal, 0 elsewhere	torch.eye(3)
# print(xi)

# x = torch.arange(0, 10, 2) # 0, 2, 4, 6, 8
# print(x)
# x1 = torch.linspace(0, 1, 5) # 0.00, 0.25, ..., 1.00
# print(x1)

## for static data
# x = torch.tensor([1, 2, 3])
# print(x)  # tensor([1, 2, 3])
# # From nested lists (2D
# x1 = torch.tensor([[1, 2], [3, 4]])
# print(x1)
# # Specify data type
# x3 = torch.tensor([1.0, 2.0], dtype=torch.float64)
# print(x3.dtype)  # torch.float64

######### Arithmetic Operations
# v1 = torch.tensor([1.0, 2.0, 3.0])
# v2 = torch.tensor([4.0, 5.0, 6.0])
#
# print(v1 + v2)  # or print(v1.add(v2)) Element-wise Add
# print(v1 - v2)  # or print(v1.sub(v2)) Element-wise sub
# print(v1 / v2)  # or print(v1.div(v2)) Element-wise div
# print(v1 * v2)  # or print(v1.mul(v2)) Element-wise mul
#
# # In-place Operations (modifies v1)
# v1_copy = v1.clone()
# v1_copy.add_(v2)
# print(v1_copy)
#
# # 2D Tensors (matrices)
# A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
#
# # Element-wise Operations
# print("\nA + B:\n", A + B)
# print("A - B:\n", A - B)
# print("A * B:\n", A * B)
# print("A / B:\n", A / B)
# print("A ** 2:\n", A ** 2)
#
# # Reduction
# print("Mean of A:", A.mean())
#
# # Matrix Multiplication
# print("Matrix Multiply:\n", torch.matmul(A, B))
#
# # In-place example on matrix
# A_copy = A.clone()
# A_copy.mul_(2)  # Multiply all elements by 2 in-place
# print("In-place Mul A * 2:\n", A_copy)


# slice tensors
# x = torch.tensor([[11, 12, 13],
#                   [21, 22, 23],
#                   [31, 32, 33],
#                   [41, 42, 43]])  # shape 4 rows and 3 columns
# print("Tensor:\n", x)
#
# # Rows
# print("First row:", x[0])  # [11, 12, 13]
# print("Last row:", x[-1])  # [41, 42, 43]
# print("First two rows:\n", x[:2])  # [[11, 12, 13], [21, 22, 23]]
#
# # Columns
# print("First column:", x[:, 0])  # [11, 21, 31, 41]
# print("Last column:", x[:, -1])  # [13, 23, 33, 43]
# print("Middle columns:\n", x[:, 1:3])  # [[12, 13], [22, 23], ...]
#
# # Sub-matrix
# print("Middle 2x2 block:\n", x[1:3, 1:3])  # [[22, 23], [32, 33]]
#
# # Fancy indexing
# print("Rows 0 & 2:\n", x[[0, 2], :])
# print("Cols 1 & 3:\n", x[:, [1, 3]])
# print("x[0,1] and x[1,2]:", x[[0, 1], [1, 2]])  # -> [12, 23]

## Get actually value
# x = torch.tensor([3.14])
# py_value = x.item()
#
# print("Tensor:", x)  # tensor([3.1400])
# print("Python float:", py_value)  # 3.14
# print("Type:", type(py_value))  # <class 'float'>
# # Create a tensor with multiple values
# x = torch.tensor([1.0, 2.0, 3.0])
#
# # Use .tolist() to convert to a Python list
# py_list = x.tolist()
#
# print("Tensor:", x)  # tensor([1., 2., 3.])
# print("Python list:", py_list)  # [1.0, 2.0, 3.0]
# print("Type:", type(py_list))  # <class 'list'>

# Reshaping
x = torch.randn(2, 3, 4)  # Shape: [2, 3, 4]

# 1. view()
x1 = x.view(6, 4)
print("view:", x1.shape)  # [6, 4]

# 2. reshape()
x2 = x.reshape(6, 4)
print("reshape:", x2.shape)  # [6, 4]

# 3. permute() - rearrange dimensions
x3 = x.permute(0, 2, 1)  # [2, 4, 3]
print("permute:", x3.shape)

# 4. transpose() - swap 2 dims
x4 = x.transpose(1, 2)  # swap dims 1 and 2 → [2, 4, 3]
print("transpose:", x4.shape)

# 5. squeeze() - remove size-1 dims
x5 = torch.randn(1, 3, 1, 5)
print("squeeze:", x5.squeeze().shape)  # [3, 5]

# 6. unsqueeze() - add dim of size 1
x6 = torch.randn(3, 5)
print("unsqueeze:", x6.unsqueeze(0).shape)  # [1, 3, 5]

# 7. flatten() - flatten dimensions
x7 = torch.randn(2, 3, 4)
print("flatten:", x7.flatten().shape)  # [24]
print("flatten from dim 1:", x7.flatten(1).shape)  # [2, 12]
