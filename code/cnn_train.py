import torch
import torch.nn as nn

# 输出矩阵维度计算
class trainmodel(nn.Module):
    def __init__(self):
        super(trainmodel, self)._init_()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.fc1 = nn.Linear(14400, 10)
    pass

# 输入张量
X = torch.randn(10, 3, 64, 64)

# 请在脑海中或草稿纸上推导：
# 1. 经过 conv1 + pool1 后，Shape 是多少？
# 2. 经过 conv2 + pool2 后，Shape 是多少？
# 3. 展平为 2D 后的 Shape (10, ?) 是多少？

# 练习二：训练阶段

# ----------------- 训练循环片段 -----------------
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 重要：梯度清零
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# ----------------- 前向传播展平片段 -----------------
class BadLeNet(nn.Module):
    # ... 初始化省略 ...
    def forward(self, x):
        x = x.view(-1, 32 * 5 * 5)  # 使用 -1 动态适配 Batch 维度
        x = self.fc1(x)
        return x

# ----------------- 单张图片推理片段 -----------------
def predict_single_image(model, img_tensor, device):
    # img_tensor 形状为 (3, 32, 32)
    model.eval() # 切换为评估模式
    img_tensor = img_tensor.unsqueeze(0).to(device) # (1, 3, 32, 32)
    
    # 关闭梯度追踪
    with torch.no_grad():
        output = model(img_tensor)
        pred_class = torch.argmax(output, dim=1).item()
    return pred_class

# 训练三：加入正则化
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModernLeNet(nn.Module):
    def __init__(self):
        super(ModernLeNet, self).__init__()
        # 1. 第一层卷积组件
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 2. 第二层卷积组件
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # 3. 防止全连接层过拟合的 Dropout
        self.dropout = nn.Dropout(p = 0.)
        
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        # 卷积层 1: Conv2d -> BatchNorm -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积层 2: Conv2d -> BatchNorm -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 展平
        x = x.view(-1, 32 * 5 * 5)
        
        # TODO: 在过全连接层前应用 dropout
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练四：自己搭建一个cnn
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 第一层
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 第二层
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 第三层
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 全连接层
        self.dropout = nn.Dropout(p = 0.3)
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 10)

        pass

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        # 展平
        x = x.view(-1, 8192)

        x = self.dropout(x)
        x = self.fc2(F.relu(self.fc1(x)))

        return x


# ==================== 自动化测试校验代码 ====================
if __name__ == "__main__":
    # 实例化模型
    model = CustomCNN()
    
    # 模拟一个 Batch 大小为 8 的 RGB 图像输入，尺寸为 64x64
    dummy_input = torch.randn(8, 3, 64, 64)
    
    try:
        output = model(dummy_input)
        print("🎉 恭喜！模型成功跑通前向传播！")
        print(f"输入形状: {dummy_input.shape}")
        print(f"输出形状: {output.shape}")
        
        # 校验断言
        assert output.shape == (8, 10), f"❌ 输出维度不对，期望 (8, 10)，实际得到 {output.shape}"
        print("✅ 维度校验完全正确！代码通过验证！")
    except Exception as e:
        print(f"❌ 运行出错，报错信息如下:\n{e}")