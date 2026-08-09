import torch

# 求导属性
x = torch.ones(2, 2)
print(x.requires_grad)
x.requires_grad_(True)
print(x.requires_grad)

y = x + 2
print(x.grad_fn, y.grad_fn) # x 输出None

# 标量反向传播
z = y * y * 3
out = z.mean() # 将z变为标量以直接反向传播
out.backward()
print(x.grad) # 输出4.5

z = y * y * 3
out = z.mean()
out.backward()
print(x.grad) # 输出9.0， 梯度会自动累加，
# 每次代入数据需 optimizer.zero_grad() 清空历史数据

# 非标量反向传播
x = torch.randn(3, requires_grad=True)
y = x * 2
v = torch.randn(3, dtype=torch.float)
y.backward(v) # 引入形状相同的矩阵作为参数即可进行非标量的反向传播
print(x.grad)

# 计算阻断
with torch.no_grad():
    infr_y = x ** 2
print(infr_y.requires_grad) # False

detached_x = x.detach()
print(detached_x.requires_grad) #False

detached_x.zero_()
print(x) # 原生张量可被修改