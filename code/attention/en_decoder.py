from torch import nn

# 基础架构
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# rnn的编码器实现
class RNNEncoder(nn.modules):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(RNNEncoder, self).__init__(**kwargs)
        # 输入: (batch_size, num_steps) -> 输出: (batch_size, num_steps, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout = dropout)

    def forward(self, X, *args):
        embs = self.embedding(X)
        # 转置时间轴：(batch_size, num_steps, embed_size) -> (num_steps, batch_size, embed_size)
        embs = embs.permute(1, 0, 2)

        # 计算
        outputs, state = self.rnn(embs)
        return outputs, state

# rnn解码器的实现
class RNNDecoder(nn.modules):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0, **kwargs):
        super(RNNDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(vocab_size, num_hiddens, num_layers, dropout = dropout)

        # 全连接层  把隐藏特征映射到目标词表维度
        self.fc = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # enc_outputs 是 Encoder 返回的 (outputs, state)
        # 这里提取 Encoder 最后一个时间步的隐状态，直接作为 Decoder 的初始隐状态
        return enc_outputs[1]

    def forward(self, X, *args):
        embs = self.embedding(X).permute(1, 0, 2)
        outputs, state = self.rnn(embs, state)

        # 将输出向量转化为预测概率空间
        outputs = self.fc(outputs)

        # 转换回 Batch 优先形状 -> (batch_size, num_steps, vocab_size)
        outputs = outputs.permute(1, 0, 2)
        
        return outputs, state