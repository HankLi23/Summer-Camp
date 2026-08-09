import torch
import torch.nn as nn
import math

# 多头注意力
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
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

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

# 前缀神经网络
class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(self.dropout(x))
        return x

# 位置编码
class PositionalEncoding(nn.Module):
    """正弦/余弦位置编码 (Positional Encoding)"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x Shape: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# encoder层
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.multi = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)

        # norm 层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # dropout 
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, mask = None):
        out, weight = self.multi(X, X, X, mask)
        X = self.norm1(X + self.dropout1(out))

        ffn_out = self.ffn(X)
        X = self.norm2(X + self.dropout2(ffn_out))

        return X, weight

# decoder 层
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # 带掩码的自注意力
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 交叉注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)

        # norm 层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # dropout 
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, X, memory, X_mask = None, memory_mask = None):
        # Masked Self-Attention (Q=X, K=X, V=X)
        self_out, _ = self.self_attn(X, X, X, X_mask)
        X = self.norm1(X + self.dropout1(self_out))

        # Cross-Attention: Q 来自 Decoder(x)，K 和 V 来自 Encoder(memory)
        cross_out, _ = self.cross_attn(X, memory, memory, memory_mask)
        X = self.norm2(X + self.dropout2(cross_out))

        ffn_out = self.ffn(X)
        X = self.norm3(X + self.dropout3(ffn_out))

        return X

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        '''
        src_vocab_size 源语言词表大小   tgt_vocab_size  目标语言词表大小
        d_model  模型隐藏层特征维度
        '''
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # N * Encoder Layers
        self.encoder_layer = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # N * Encoder Layers
        self.decoder_layer = nn.ModuleList([
            TransformerDecoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # linear 层
        self.fc = nn.Linear(d_model, tgt_vocab_size)

        self.d_model = d_model

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        # 生成 Decoder 的因果掩码 (Causal Mask)
        if tgt_mask is None:
            tgt_mask = self.generate_causal_mask(tgt.size(1), tgt.device)

        # encoder
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        memory = src_emb
        for layer in self.encoder_layer:
            memory, _ = layer(memory, src_mask)

        # decoder
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        out = tgt_emb
        for layer in self.decoder_layer:
            out = layer(out, memory, tgt_mask, src_mask)

        out = self.fc(out)
        return out

    # 生成掩码
    def generate_causal_mask(self, sz, device):
        mask = torch.tril(torch.ones(sz, sz, device=device))
        return mask # Shape: (sz, sz)

# ==================== 自动化测试 ====================
if __name__ == "__main__":
    # 词表大小设为 1000，模型基础维度 512，堆叠 6 层
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model=512, num_encoder_layers=6, num_decoder_layers=6)
    
    # 模拟 Batch = 2，源句子长度为 10，目标句子长度为 8
    dummy_src = torch.randint(0, src_vocab_size, (2, 10)) # 源输入 (Inputs)
    dummy_tgt = torch.randint(0, tgt_vocab_size, (2, 8))  # 目标输入 (Outputs shifted right)
    
    output_logits = model(dummy_src, dummy_tgt)
    
    print("🎉 完整 Transformer 前向传播成功跑通！")
    print(f"源输入 Shape (Inputs):                  {dummy_src.shape}")
    print(f"目标输入 Shape (Outputs shifted right):  {dummy_tgt.shape}")
    print(f"最终输出 Logits Shape:                   {output_logits.shape}")
    
    assert output_logits.shape == (2, 8, 1000), "输出维度不匹配！"
    print("✅ 维度校验完全匹配图 1 中的架构设计！")