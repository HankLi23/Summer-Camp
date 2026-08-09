import torch
import torch.nn as nn
import math

# bert embedding层
class BERTEmbedding(nn.Module):
    """
    BERT Embedding: Token + Segment + Positional
    input_ids 输入序列  segment_ids 段落标记
    """
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(vocab_size, d_model)
        self.seg_embed = nn.Embedding(vocab_size, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)

        # 自动生成 0 ~ seq_len-1 的位置索引用作向量查表
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # 三个Embedding 相加
        embedding = self.tok_embed(input_ids) + self.pos_embed(pos_ids) + self.seg_embed(segment_ids)
        return embedding

# 其余部分与tranformer一致（少decoder）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weight = torch.softmax(scores, dim=-1)
        weight = self.dropout(weight)

        context = torch.matmul(weight, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context), weight


class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()  # BERT 原论文使用的是 GELU 激活函数
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.gelu(self.fc1(x))))


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.multi = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        out, weight = self.multi(X, X, X, mask)
        X = self.norm1(X + self.dropout1(out))
        ffn_out = self.ffn(X)
        X = self.norm2(X + self.dropout2(ffn_out))
        return X, weight

# BERT 搭建
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_encoder_layers=6, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout)

        # 堆叠 N 层 Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # 掩码模型预测头
        self.mlm_head = nn.Linear(d_model, vocab_size)

        # NSP 预测头  做二分类 (IsNext / NotNext)
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, input_ids, segment_ids, mask = None):
        x = self.embedding(input_ids, segment_ids)

        for layer in self.encoder_layers:
            x, _ = layer(x, mask)

        # 计算 MLM 输出 logits
        mlm_logits = self.mlm_head(x)

        # 提取 [CLS] token 的输出 (全句全局特征) 预测 NSP -> (Batch, 2)
        cls_output = x[:, 0, :]
        nsp_logits = self.nsp_head(cls_output)

        return mlm_logits, nsp_logits

if __name__ == "__main__":
    # 超参数设定
    vocab_size = 30000  # BERT 标准词表大小约为 3 万
    d_model = 512
    batch_size = 2
    seq_len = 10

    # 1. 实例化手撕的 BERT 模型
    bert_model = BERT(vocab_size=vocab_size, d_model=d_model, num_encoder_layers=6)

    # 2. 构造模拟输入
    # 假设输入为: [CLS] I love deep [MASK] [SEP] It is fun [SEP]
    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # segment_ids: 前半句(包含 [CLS] 和 [SEP]) 设为 0，后半句设为 1
    dummy_segment_ids = torch.tensor([
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ])

    # 3. 前向传播
    mlm_logits, nsp_logits = bert_model(dummy_input_ids, dummy_segment_ids)

    print("🎉 手撕 BERT 前向传播顺利通关！")
    print(f"输入序列 Shape (input_ids):            {dummy_input_ids.shape}")
    print(f"段落标记 Shape (segment_ids):          {dummy_segment_ids.shape}")
    print(f"掩码预测 Logits Shape (MLM):           {mlm_logits.shape}  <-- 预测每个 [MASK] 填什么词")
    print(f"下一句预测 Logits Shape (NSP):          {nsp_logits.shape}       <-- 判断句 B 是不是句 A 的下一句")