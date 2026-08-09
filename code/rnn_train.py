import torch
from torch import nn
import torch.nn.functional as F

# 训练一：数据的预处理
def preprocess_input(X, vocab_size):
    # 1. X.T 转置 -> (20, 16)
    # 2. F.one_hot 编码 -> (20, 16, 100)
    # 3. .float() 转类型
    out = F.one_hot(X.T.long(), vocab_size).float()
    return out

# # 测试代码
# X_test = torch.randint(0, 100, (16, 20))
# out = preprocess_input(X_test, 100)
# print(out.shape)

# 训练二：向前传播函数
def forward(self, inputs, state):

    X = F.one_hot(inputs.T.long(), self.vocab_size).float()
    Y, state = self.rnn(X, state)

    # 把 3D 展平为 2D (1120, num_hiddens)
    output = self.linear(Y.reshape((-1, Y.shape[-1])))
    
    return output, state

# 训练三：搭建双向多层模型
class GRUMultiLayer(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.num_directions = 2  # 双向即为 2
        
        # 创建双向 2 层 nn.GRU (bidirectional参数为true)
        self.gru = nn.GRU(input_size=vocab_size,
            hidden_size=num_hiddens,
            num_layers=num_layers,
            bidirectional=True)
        
        # 定义全连接层 Linear(双向输出特征数要乘二)
        self.linear = nn.Linear(num_hiddens * num_layers, vocab_size)

    def begin_state(self, batch_size, device):
        total_layers = self.num_layers * self.num_directions
        return torch.zeros((total_layers, batch_size, self.num_hiddens), device=device)

model = GRUMultiLayer(vocab_size=28, num_hiddens=256)
state = model.begin_state(batch_size=32, device='cpu')
print(state.shape)

