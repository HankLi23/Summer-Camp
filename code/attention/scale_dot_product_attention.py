import torch
import torch.nn as nn
import math

class attention(nn.modules):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask = None):
        d_k = Q.size(-1)

        # 计算QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, 1)) / math.sqrt(d_k)

        # 如果传入了 mask（解码器不能看未来的词），把掩码位置填充为负无穷 -1e9
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weight = torch.softmax(scores, dim = -1)
        weight = self.dropout(weight)

        # weight与 V 矩阵乘法
        output = torch.matmul(weight, V)

        return output, weight