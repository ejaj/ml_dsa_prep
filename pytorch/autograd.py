import torch

x = torch.randn(3, requires_grad=True)
print(x)
# y = x + 2
# print(y)
# z = y * y * 2
# # print(z)
# # z = z.mean()
# # z.backward()
# # print(z)
# v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
# z.backward(v)
# print(x.grad)


# x.requires_grad_(False)
# y = x.detach()
# print(y)
# with torch.no_grad():
#     y = x * 2
#     print(y)

weight = torch.randn(4, requires_grad=True)
for epoch in range(3):
    model_output = (weight * 3).sum()
    model_output.backward()
    print(weight.grad)
    weight.grad.zero_()

optimizer = torch.optim.SGD(weight, lr=0.1)
optimizer.step()
optimizer.zero_grad()
