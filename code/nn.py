import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3)
        self.fc1 = nn.Linear(12*26*26, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # 动态展平：保留Batch维度，将后面的Channel、H、W全部乘起来展平
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
# single_image = torch.rand(3, 28, 28)
# model = SimpleNet()
# batch_ready_image = single_image.unsqueeze(0)

# output = model(batch_ready_image) 
# print(output.shape) # 升维后才能跑

# 训练骨架
net = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.05)

batch_input = torch.randn(4, 3, 28, 28)
target = torch.randn(4, 2)

optimizer.zero_grad()             # 步骤一：手动将所有参数的梯度缓冲区清零
output = net(batch_input)               # 步骤二：前向传播，通过模型获取输出
loss = criterion(output, target)  # 步骤三：计算当前损失值
loss.backward()                   # 步骤四：反向传播，积攒当前批次的梯度
optimizer.step()  

print(loss.item())