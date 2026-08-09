import os
# 1. 解决网络与警告问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments
)

# ==================== Step 1: 加载数据集 (已修复命名空间) ====================
# 使用完整的仓库名 "nyu-mll/glue" 替代原来的 "glue"
dataset = load_dataset("nyu-mll/glue", "sst2")

# ==================== Step 2: 加载分词器与模型 ====================
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


# ==================== Step 3: 数据预处理 (Tokenize) ====================
def preprocess_function(examples):
    # 截断与填充到统一长度 128
    return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

# 批量应用 Tokenizer
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 截取一部分数据快速体验微调（真实训练时可全量训练）
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_datasets["validation"].select(range(200))


# ==================== Step 4: 设置微调超参数 ====================
training_args = TrainingArguments(
    output_dir="./bert_sentiment_model", # 模型与 Checkpoint 保存路径
    learning_rate=2e-5,                  # ⚠️ 微调关键：极其轻柔的学习率！
    per_device_train_batch_size=16,      # 训练 Batch Size
    per_device_eval_batch_size=16,       # 评估 Batch Size
    num_train_epochs=3,                  # 微调通常只需要 3~5 个 Epoch
    weight_decay=0.01,                   # 权重衰减
    eval_strategy="epoch",               # 每个 Epoch 结束后评估一次
    save_strategy="epoch",               # 每个 Epoch 保存一次模型
    logging_steps=20,                    # 每 20 步打印一次日志
    fp16=torch.cuda.is_available(),      # 若有 GPU 则自动开启半精度 (FP16) 加速
)


# ==================== Step 5: 实例化 Trainer 并启动微调 ====================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,  
)

print("🚀 开始微调 BERT 模型 ...")
trainer.train()


# ==================== Step 6: 拿微调好的模型做实测推理 ====================
print("\n🎉 ===== 微调完成，测试推理 =====")
test_texts = [
    "I absolutely loved this movie, the plot was amazing!",
    "This was the worst experience I have ever had, terrible service."
]

model.eval()
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # 取 Logits 最大的概率类别
    pred_label = logits.argmax(dim=-1).item()
    sentiment = "Positive (正面) 😊" if pred_label == 1 else "Negative (负面) 😡"
    
    print(f"输入文本: {text}")
    print(f"预测结果: {sentiment}\n")