import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
from datasets import load_dataset

# ==================== 0. 全局特殊 Token 标记定义 ====================
PAD_TOKEN = "<PAD>"  # 0: 填充词
UNK_TOKEN = "<UNK>"  # 1: 未知词
BOS_TOKEN = "<BOS>"  # 2: 句子开头
EOS_TOKEN = "<EOS>"  # 3: 句子结尾

# ==================== 1. 加载 Multi30k 数据集 ====================
print("正在下载/加载 Multi30k 数据集...")
raw_dataset = load_dataset("bentrevett/multi30k")

print(raw_dataset)
sample = raw_dataset['train'][0]
print("样本示例:", sample)

# ==================== 2. 补全 SimpleVocab 类 (增加 decode 方法) ====================
class SimpleVocab:
    def __init__(self, texts, max_size=10000):
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN
        
        self.token2idx = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3}
        self.idx2token = {0: self.pad_token, 1: self.unk_token, 2: self.bos_token, 3: self.eos_token}
        
        # 统计词频
        word_counts = {}
        for text in texts:
            for word in text.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
                
        # 按词频排序取前 max_size 个词
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:max_size]
        for word, _ in sorted_words:
            if word not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[word] = idx
                self.idx2token[idx] = word

    def encode(self, text, add_bos_eos=True):
        tokens = text.lower().split()
        ids = [self.token2idx.get(token, self.token2idx[self.unk_token]) for token in tokens]
        if add_bos_eos:
            ids = [self.token2idx[self.bos_token]] + ids + [self.token2idx[self.eos_token]]
        return ids

    def decode(self, ids):
        tokens = []
        for idx in ids:
            if idx in (self.token2idx[self.pad_token], self.token2idx[self.bos_token]):
                continue
            if idx == self.token2idx[self.eos_token]:
                break
            tokens.append(self.idx2token.get(idx, self.unk_token))
        return " ".join(tokens)

    def __len__(self):
        return len(self.token2idx)


# 构建英文 (Src) 和 德文 (Tgt) 词表
en_texts = [item['en'] for item in raw_dataset['train']]
de_texts = [item['de'] for item in raw_dataset['train']]

src_vocab = SimpleVocab(en_texts, max_size=8000)
tgt_vocab = SimpleVocab(de_texts, max_size=8000)

print(f"英文词表大小: {len(src_vocab)} | 德文词表大小: {len(tgt_vocab)}")


# ==================== 3. 桥接 HuggingFace 数据集与 PyTorch DataLoader ====================

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    
    for item in batch:
        src_ids = src_vocab.encode(item['en'], add_bos_eos=False)
        tgt_ids = tgt_vocab.encode(item['de'], add_bos_eos=True)
        src_batch.append(torch.tensor(src_ids, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long))
        
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_padded, tgt_padded


train_loader = DataLoader(
    raw_dataset['train'], 
    batch_size=32, 
    shuffle=True, 
    collate_fn=collate_fn
)


# ==================== 4. 补全位置编码 PositionalEncoding ====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4, 
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_padding_mask=None, tgt_padding_mask=None):
        # 叠加词向量与位置编码
        src_emb = self.pos_encoder(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt) * math.sqrt(self.d_model))

        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        return self.fc_out(out)


# ==================== 5. 训练模型 ====================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Seq2SeqTransformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=128,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2
).to(device)

pad_idx = tgt_vocab.token2idx[PAD_TOKEN]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10
print(f"\n🚀 开始训练 (设备: {device}) ...")
model.train()

for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for src_batch, tgt_batch in train_loader:
        src_batch, tgt_batch = src_batch.to(device), tgt_batch.to(device)

        tgt_input = tgt_batch[:, :-1]
        tgt_label = tgt_batch[:, 1:]

        src_padding_mask = (src_batch == pad_idx)
        tgt_padding_mask = (tgt_input == pad_idx)

        optimizer.zero_grad()
        logits = model(src_batch, tgt_input, src_padding_mask, tgt_padding_mask)

        loss = criterion(logits.reshape(-1, len(tgt_vocab)), tgt_label.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch:02d}/{EPOCHS}] ---> Loss: {avg_loss:.4f}")


# ==================== 6. 实际翻译测试 (Greedy Search 解码) ====================

def translate(model, src_text, src_vocab, tgt_vocab, max_len=15):
    model.eval()
    with torch.no_grad():
        src_ids = src_vocab.encode(src_text, add_bos_eos=False)
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

        tgt_ids = [tgt_vocab.token2idx[BOS_TOKEN]]
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(device)
            logits = model(src_tensor, tgt_tensor)
            
            next_token_id = logits[0, -1, :].argmax().item()
            
            if next_token_id == tgt_vocab.token2idx[EOS_TOKEN]:
                break
                
            tgt_ids.append(next_token_id)

    return tgt_vocab.decode(tgt_ids)


print("\n🎉 ===== 训练完成，开始推理翻译验证 (英 -> 德) =====")
test_sentences = [
    "Two young males are outside near many bushes .",
    "A man in a black shirt is dancing .",
    "A dog is running on the grass ."
]

for sentence in test_sentences:
    translated_text = translate(model, sentence, src_vocab, tgt_vocab)
    print(f"英文原句: {sentence}")
    print(f"德文翻译: {translated_text}")
    print("-" * 40)