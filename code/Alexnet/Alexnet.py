import torch.nn as nn
import torch

# 模型构建
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 用nn.Sequential()将网络打包成一个模块，精简代码
        self.features = nn.Sequential(   # 卷积层提取图像特征
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True), 									# 直接修改覆盖原值，节省运算内存
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(   # 全连接层对图像分类
            nn.Dropout(p=0.5),			   # Dropout 随机失活神经元，默认比例为0.5
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()
            
	# 前向传播过程
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)	# 展平后再传入全连接层
        x = self.classifier(x)
        return x
        
	# 网络权重初始化，实际上 pytorch 在构建网络时会自动初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):                            # 若是卷积层
                nn.init.kaiming_normal_(m.weight, mode='fan_out',   # 用（何）kaiming_normal_法初始化权重
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)                    # 初始化偏重为0
            elif isinstance(m, nn.Linear):            # 若是全连接层
                nn.init.normal_(m.weight, 0, 0.01)    # 正态分布初始化
                nn.init.constant_(m.bias, 0)          # 初始化偏重为0

# 模型训练
# 导入包
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os
import json
import time

# 使用GPU训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),       # 随机裁剪，再缩放成 224×224
                                 transforms.RandomHorizontalFlip(p=0.5),  # 水平方向随机翻转，概率为 0.5, 即一半的概率翻转, 一半的概率不翻转
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# 获取当前 Alexnet.py 文件所在的绝对文件夹路径
current_dir = os.path.dirname(__file__)

# 2. 因为 flower_data 就在当前目录下，直接进行标准拼接
image_path = os.path.join(current_dir, "flower_data")


# ==================== 下面的数据导入跟着对应修改 ====================

# 导入训练集并进行预处理（修正为 os.path.join 拼接）
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),		
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# 按batch_size分批次加载训练集
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=32, 
                                           shuffle=True,
                                           num_workers=0)

# 导入验证集并进行预处理（修正为 os.path.join 拼接）
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)

# 加载验证集
validate_loader = torch.utils.data.DataLoader(validate_dataset,	# 导入的验证集
                                              batch_size=32, 
                                              shuffle=True,
                                              num_workers=0)

# 字典，类别：索引 {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
# 将 flower_list 中的 key 和 val 调换位置
cla_dict = dict((val, key) for key, val in flower_list.items())

# 将 cla_dict 写入 json 文件中
json_str = json.dumps(cla_dict, indent=4)
# 强行使用 current_dir 拼接绝对路径
json_save_path = os.path.join(current_dir, 'class_indices.json')
with open(json_save_path, 'w') as json_file:
    json_file.write(json_str)

net = AlexNet(num_classes=5, init_weights=True)  	  # 实例化网络（输出类型为5，初始化权重）
net.to(device)									 	  # 分配网络到指定的设备（GPU/CPU）训练
loss_function = nn.CrossEntropyLoss()			 	  # 交叉熵损失
optimizer = optim.Adam(net.parameters(), lr=0.0002)	  # 优化器（训练参数，学习率）
save_path = os.path.join(current_dir, 'AlexNet.pth')
best_acc = 0.0

for epoch in range(10):
    ######################## train ########################
    net.train()     					# 训练过程中开启 Dropout
    running_loss = 0.0					# 每个 epoch 都会对 running_loss  清零
    time_start = time.perf_counter()	# 对训练一个 epoch 计时
    
    for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
        images, labels = data   # 获取训练集的图像和标签
        optimizer.zero_grad()	# 清除历史梯度
        
        outputs = net(images.to(device))				 # 正向传播
        loss = loss_function(outputs, labels.to(device)) # 计算损失
        loss.backward()								     # 反向传播
        optimizer.step()								 # 优化器更新参数
        running_loss += loss.item()
        
        # 打印训练进度（使训练过程可视化）
        rate = (step + 1) / len(train_loader)           # 当前进度 = 当前step / 训练一轮epoch所需总step
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print('%f s' % (time.perf_counter()-time_start))

    ######################## validate ########################
    net.eval()    # 验证过程中关闭 Dropout
    acc = 0.0  
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
            acc += (predict_y == val_labels.to(device)).sum().item()    
        val_accurate = acc / val_num
        
        # 保存准确率最高的那次网络参数
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')

import os
import json
import torch
from PIL import Image
from torchvision import transforms

# ==================== 1. 环境与路径初始化 ====================
# 动态获取当前运行的 .py 文件所在的绝对文件夹路径
current_dir = os.path.dirname(__file__)

# 自动检测硬件，优先选用 GPU 加速推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的推理设备: {device}")

# 锁定本地同级目录下的 predict 文件夹
predict_dir = os.path.join(current_dir, 'predict')

# ==================== 2. 预处理与核心配置加载 ====================
# AlexNet 标准输入的图像预处理流
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 安全读取同级目录下的类别映射 JSON 文件
json_path = os.path.join(current_dir, 'class_indices.json')
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        class_indict = json.load(f)
except Exception as e:
    print(f"错误：读取配置文件 {json_path} 失败！原因为: {e}")
    exit(-1)

# 创建网络模型并部署到指定硬件
model = AlexNet(num_classes=5)
model_weight_path = os.path.join(current_dir, "AlexNet.pth")

try:
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.to(device)
    model.eval()  # 生产防灾：显式关闭 Dropout 随机失活机制
except Exception as e:
    print(f"错误：权重文件 {model_weight_path} 加载失败！请确认模型是否已训练完成。")
    exit(-1)

# ==================== 3. 动态扫描并执行批量推理 ====================
# 允许扫描的合法图片后缀白名单
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')

if not os.path.exists(predict_dir):
    print(f"提示：找不到文件夹 {predict_dir}，请先在代码同级目录下创建名为 'predict' 的文件夹并放入测试照片。")
else:
    print("\n" + "="*60)
    print(f"{'图片文件名':<25}{'模型预测类别':<18}{'置信度(Probability)':<15}")
    print("="*60)
    
    # 动态遍历文件夹下的所有文件
    for file_name in os.listdir(predict_dir):
        if file_name.endswith(valid_extensions):
            image_path = os.path.join(predict_dir, file_name)
            
            # 读取图像并转化为RGB格式（防止有RGBA四通道图引发报错）
            img = Image.open(image_path).convert('RGB')
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)  # 升维伪装成批次 [1, 3, 224, 224]
            
            with torch.no_grad():
                # 数据硬件对齐并前向传播
                outputs = model(img.to(device))
                # 在特征维度（dim=1）上进行归一化概率映射
                predict_probs = torch.softmax(outputs, dim=1)
                # 提取概率最大的类别索引
                predict_cla = torch.argmax(predict_probs, dim=1).item()
                # 提取对应的置信度数值
                confidence = predict_probs[0][predict_cla].item()
                
            # 格式化美化输出
            class_name = class_indict[str(predict_cla)]
            print(f"{file_name:<25}{class_name:<18}{confidence:.2%}")
            
    print("="*60)
    print("批量推理任务顺利结束！")