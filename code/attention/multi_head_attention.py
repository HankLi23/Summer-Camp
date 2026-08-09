import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        # 确保总维度能被头数整除
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 将投影拼接在一起
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask = None):
        '''
        T 序列长度
        head_dim 每个头被分配到的特征维度
        '''
        batch_size = Q.size(0)

        q = self.W_q(Q)
        k = self.W_k(K)
        v = self.W_v(V)
        
        # (B, T, d_model) -> (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算QK^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, 1)) / math.sqrt(self.d_k)

        # 如果传入了 mask（解码器不能看未来的词），把掩码位置填充为负无穷 -1e9
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weight = torch.softmax(scores, dim = -1)
        weight = self.dropout(weight)

        # weight与 v 矩阵乘法
        context = torch.matmul(weight, v)

        # 拼接所有头的输出
        # (B, num_heads, T, d_k) -> (B, T, num_heads, d_k) -> (B, T, d_model)
        context= context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(context)

        return output, weight