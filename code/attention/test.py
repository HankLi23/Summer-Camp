import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# ==================== 1. 一行代码加载开源 Multi30k 数据集 ====================
# 该数据集包含 29,000 条训练集，1,014 条验证集，1,000 条测试集
print("正在下载/加载 Multi30k 数据集...")
raw_dataset = load_dataset("bentrevett/multi30k")

print(raw_dataset)
# 查看一条样本结构
sample = raw_dataset['train'][0]
print("样本示例:", sample)
# 输出: {'en': 'Two young, white males are outside near many bushes.', 
#        'de': 'Zwei junge weiße Männer sind draußen in der Nähe von vielen Büschen.'}


# ==================== 2. 构建分词器 (Tokenizer) 与 词表 (Vocab) ====================

class SimpleVocab:
    def __init__(self, texts, max_size=10000):
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        
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
    """把原始文本转换为 Batch Tensor 并进行 Padding"""
    src_batch, tgt_batch = [], []
    
    for item in batch:
        src_ids = src_vocab.encode(item['en'], add_bos_eos=False)
        tgt_ids = tgt_vocab.encode(item['de'], add_bos_eos=True)
        src_batch.append(torch.tensor(src_ids, dtype=torch.long))
        tgt_batch.append(torch.tensor(tgt_ids, dtype=torch.long))
        
    # 动态填充到当前 Batch 内的最长长度
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_padded, tgt_padded


# 构建 PyTorch 标准 DataLoader
train_loader = DataLoader(
    raw_dataset['train'], 
    batch_size=32, 
    shuffle=True, 
    collate_fn=collate_fn
)


# ==================== 4. 模拟训练 Batch 迭代 ====================

print("\n🚀 成功构建 30,000 条样本的 PyTorch 数据加载流水线！")

for batch_idx, (src_tensor, tgt_tensor) in enumerate(train_loader):
    print(f"\n--- Batch {batch_idx + 1} ---")
    print("英文输入 Tensor Shape (Src):", src_tensor.shape)  # 如: torch.Size([32, 18])
    print("德文目标 Tensor Shape (Tgt):", tgt_tensor.shape)  # 如: torch.Size([32, 21])
    
    # 拿到这两个 Tensor 之后，直接送入你前面写的 Transformer 模型里就可以开始真正的大规模训练了！
    # logits = model(src_tensor.to(device), tgt_tensor[:, :-1].to(device))
    break