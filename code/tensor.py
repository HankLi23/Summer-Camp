import torch
import numpy as np

# 张量的创建
x1 = torch.empty(5, 3)
print(x1)

x2 = torch.rand(5, 3)
print(x2)

data = [1.5, 2.5, 3.5]
x3 = torch.tensor(data)
print(x3)

x4 = torch.rand_like(x3)
print(x4)

print(x2.shape)
print(x2.size())

# 张量的基本操作
A, B = torch.ones(2, 3)
print(A, B)

print(A + B)
print(torch.add(A, B))

A.add_(B)  # 慎用，会覆盖原来的值，autagrad需要保存中间变量的值
print(A)

matrix = torch.rand(4, 4)
print(matrix)
matrix = matrix.view(2, -1)
print(matrix)

loss = torch.tensor([0.7632])
print(loss)
py_loss = loss.item()
print(py_loss)

# numpy与 tensor 相互转化
np_arr = np.ones(3)
torch_tensor = torch.from_numpy(np_arr)
np_arr[0] = 99.0
print(torch_tensor)

# cuda加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
cpu_tensor = torch.rand(4, 4)
gpu_tensor = cpu_tensor.to(device, dtype = torch.double)
print(gpu_tensor.device)