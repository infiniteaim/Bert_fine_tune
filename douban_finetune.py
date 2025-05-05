import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import wandb
# 定义豆瓣电影短评数据集类
class DoubanDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 加载豆瓣电影短评数据
def load_douban_data(file_path):
    df = pd.read_csv(file_path)
    texts = df['Comment'].values  # 假设评论内容在 'comment' 列
    labels = df['Star'].values  # 假设情感标签在 'sentiment' 列
    labels = labels - 1  # 将标签转换为从0开始
    return texts, labels


# 训练函数
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss.mean()
        wandb.log({"train_loss": loss.item()})
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(dataloader)


# 评估函数
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples


# 主函数
def main():
    # 配置参数
    max_length = 128
    batch_size = 512
    num_epochs = 5
    learning_rate = 2e-2
    wandb.init(project="amazon-bert-finetuning", 
               entity="anonusycw-hong-kong-university-of-science-and-technology",
                name="douban-bert-lp"
               )  # 替换为你的wandb项目和实体名称
    # 加载数据
    texts, labels = load_douban_data('all_movies_with_id.csv')

    # 划分训练集和验证集
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 加载分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=5)

    # 创建数据集和数据加载器
    train_dataset = DoubanDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = DoubanDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for param in model.bert.parameters():
        param.requires_grad = False
    model = nn.DataParallel(model)  # 使用多GPU训练
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, optimizer, device)
        val_accuracy = evaluate(model, val_dataloader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


if __name__ == "__main__":
    main()
    