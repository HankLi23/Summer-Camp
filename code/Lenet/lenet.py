import torch.nn as nn
import torch.nn.functional as F

# 模型搭建
class LeNet(nn.Module): 					# 继承于nn.Module这个父类
    def __init__(self):						# 初始化网络结构
        super(LeNet, self).__init__()    	# 多继承需用到super函数
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):			 # 正向传播过程
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x

# 模型训练

# 数据导入
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import torchvision

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 动态获取当前运行的 .py 文件所在的绝对文件夹路径
current_dir = os.path.dirname(__file__)

# 导入50000张训练图片
train_set = torchvision.datasets.CIFAR10(root = current_dir, 	 # 数据集存放目录
										 train = True,		 # 表示是数据集中的训练集
                                        download = False,  	 # 第一次运行时为True，下载数据集，下载完成后改为False
                                        transform = transform) # 预处理过程

# 加载训练集，实际过程需要分批次（batch）训练                                        
train_loader = torch.utils.data.DataLoader(train_set, 	  # 导入的训练集
										   batch_size=50, # 每批训练的样本数
                                          shuffle=True,  # 是否打乱训练集
                                          num_workers=0)  # 使用线程数，在windows下设置为0

# 测试
# 导入10000张测试图片
test_set = torchvision.datasets.CIFAR10(root = current_dir, 
										train = False,	# 表示是数据集中的测试集
                                        download = False, transform = transform)
# 加载测试集
test_loader = torch.utils.data.DataLoader(test_set, 
										  batch_size = 10000, # 每批用于验证的样本数
										  shuffle = False, num_workers=0)
# 获取测试集中的图像和标签，用于accuracy计算
test_data_iter = iter(test_loader)
test_image, test_label = next(test_data_iter)

# 训练 --直接使用 GPU 训练
net = LeNet()
device = torch.device("cuda")
net.to(device) # 将网络分配到指定的device中
loss_function = nn.CrossEntropyLoss() 
optimizer = optim.Adam(net.parameters(), lr=0.001) 

for epoch in range(5): 

    running_loss = 0.0
    time_start = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.to(device))				  # 将inputs分配到指定的device中
        loss = loss_function(outputs, labels.to(device))  # 将labels分配到指定的device中
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if step % 1000 == 999:    # 每1000步打印一次
            with torch.no_grad():  # 以下不占用内存计算梯度
                outputs = net(test_image.to(device)) # 将test_image分配到指定的device中
                predict_y = torch.max(outputs, dim=1)[1]
                accuracy = (predict_y == test_label.to(device)).sum().item() / test_label.size(0) # 将test_label分配到指定的device中

                print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                      (epoch + 1, step + 1, running_loss / 1000, accuracy))

                print('%f s' % (time.perf_counter() - time_start))
                running_loss = 0.0

print('Finished Training')

save_path = os.path.join(current_dir, 'Lenet.pth')
torch.save(net.state_dict(), save_path)
print(f"Model saved successfully at: {save_path}")

# ==================== 3. 批量模型预测 ====================
from PIL import Image

# 预测用数据预处理
transform_predict = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 锁定本地同级目录下的 horse 文件夹路径
horse_dir = os.path.join(current_dir, 'horse')

# 工业级防灾：检查用户是否切实创建了该文件夹
if not os.path.exists(horse_dir):
    print(f"提示：找不到文件夹 {horse_dir}，请先在代码同级目录下创建名为 'horse' 的文件夹，并放入照片。")
else:
    # 实例化网络并动态加载同级目录下的模型权重
    predict_net = LeNet()
    predict_net.load_state_dict(torch.load(save_path))
    predict_net.eval()
    
    # 顺手将推理模型也搬运到 GPU 上，加速批量推理
    predict_net.to(device)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 允许扫描的合法图片后缀白名单
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
    
    print("\n运行核心提示：开始依次读取文件夹内的图片进行批量预测...")
    print("-" * 50)
    
    # 动态遍历文件夹下的所有文件
    for file_name in os.listdir(horse_dir):
        # 过滤掉隐藏文件或非图片文件
        if file_name.endswith(valid_extensions):
            image_path = os.path.join(horse_dir, file_name)
            
            # 读取图像并流转预处理
            im = Image.open(image_path)
            im = transform_predict(im)
            im = torch.unsqueeze(im, dim=0) # 升维伪装成批次 [1, 3, 32, 32]
            
            with torch.no_grad():
                # 注意：数据搬运到 GPU 必须与模型硬件对齐
                outputs = predict_net(im.to(device))
                # 核心修正：使用 .item() 直接提取标量，完美消灭旧版本 NumPy 的强转警告
                predict_idx = torch.max(outputs, dim=1)[1].item()
                
            print(f"图片文件名: {file_name}  ===>  模型预测结果: {classes[predict_idx]}")
            
    print("-" * 50)
    print("批量预测流程全部顺利结束！")