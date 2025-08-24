import torch
import torch.nn
import torch.nn.functional as F
from torchtyping import TensorType

# Round your answers to 4 decimal places using torch.round(input_tensor, decimals = 4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # torch.reshape() will be useful - check out the documentation
        out = torch.reshape(to_reshape, (-1,))
        return torch.round(out, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # torch.mean() will be useful - check out the documentation
        out = torch.mean(to_avg)
        return torch.round(out, decimals=4)


    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # torch.cat() will be useful - check out the documentation
        out = torch.cat((cat_one, cat_two), dim=0)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # torch.nn.functional.mse_loss() will be useful - check out the documentation
        out = F.mse_loss(prediction, target)
        return torch.round(out, decimals=4)
    
sol = Solution()

# -------------------------------
# 1. Example for reshape()
# -------------------------------
tensor_2d = torch.tensor([[1.12345, 2.67891], [3.98765, 4.55555]])
reshaped = sol.reshape(tensor_2d)
print("Reshaped Tensor:", reshaped)
# Output: tensor([1.1235, 2.6789, 3.9877, 4.5556])

# -------------------------------
# 2. Example for average()
# -------------------------------
tensor_avg = torch.tensor([1.12345, 2.67891, 3.98765, 4.55555])
avg = sol.average(tensor_avg)
print("Average:", avg)
# Output: tensor(3.0869)

# -------------------------------
# 3. Example for concatenate()
# -------------------------------
tensor_a = torch.tensor([1.11111, 2.22222])
tensor_b = torch.tensor([3.33333, 4.44444])
concatenated = sol.concatenate(tensor_a, tensor_b)
print("Concatenated Tensor:", concatenated)
# Output: tensor([1.1111, 2.2222, 3.3333, 4.4444])

# -------------------------------
# 4. Example for get_loss()
# -------------------------------
pred = torch.tensor([2.5, 3.5, 4.5])
target = torch.tensor([3.0, 3.0, 5.0])
loss = sol.get_loss(pred, target)
print("MSE Loss:", loss)
# Output: tensor(0.3333)

