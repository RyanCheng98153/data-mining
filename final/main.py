import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. Config & Hyperparameters
# ==========================================
Config = {
    'MAX_LEN': 128,          # 文本最大長度
    'BATCH_SIZE': 32,        # V100 可以開大一點
    'EPOCHS': 10,
    'LR': 2e-5,              # BERT 微調需要小 Learning Rate
    'HIDDEN_DIM': 256,       # LSTM/GRU 隱藏層
    'MODEL_NAME': 'bert-base-uncased', # 或是 'roberta-base'
    'NUM_LAYERS': 2,         # LSTM 層數
    'DROPOUT': 0.3
}

# ==========================================
# 2. Dataset Preparation
# ==========================================
class KTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, mode='train'):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode
        
        # 將資料按學生分組，並按時間排序
        self.group = df.groupby('student_id')
        self.student_ids = list(self.group.groups.keys())
        
    def __len__(self):
        return len(self.student_ids)
    
    def __getitem__(self, idx):
        student_id = self.student_ids[idx]
        student_df = self.group.get_group(student_id).sort_values('interaction_id')
        
        # 提取特徵
        problems = student_df['Problem'].fillna("").values
        concepts = student_df['concepts_of_the_problem'].fillna("").values
        
        # 文本特徵：將題目和概念結合 (增加上下文)
        text_inputs = [f"{p} [SEP] {c}" for p, c in zip(problems, concepts)]
        
        # Tokenization
        encoding = self.tokenizer(
            text_inputs,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 歷史作答 (Response)
        if self.mode == 'train':
            responses = student_df['response'].values
            return {
                'input_ids': encoding['input_ids'], # (Seq_Len, Max_Len)
                'attention_mask': encoding['attention_mask'],
                'responses': torch.tensor(responses, dtype=torch.float),
                'interaction_ids': student_df['interaction_id'].values
            }
        else:
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'interaction_ids': student_df['interaction_id'].values
            }

# ==========================================
# 3. Model Architecture (BERT + LSTM)
# ==========================================
class BertKTModel(nn.Module):
    def __init__(self):
        super(BertKTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(Config['MODEL_NAME'])
        
        # 凍結部分 BERT 層以節省顯存並防止 Overfitting (因為數據少)
        # 如果顯存夠 (V100)，可以嘗試解凍最後 2 層
        for param in self.bert.parameters():
            param.requires_grad = False
        
        bert_hidden_size = self.bert.config.hidden_size # 768 for base
        
        # 序列模型 (處理學生做題的順序)
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size + 1, # BERT Embedding + 上一題對錯(0/1)
            hidden_size=Config['HIDDEN_DIM'],
            num_layers=Config['NUM_LAYERS'],
            batch_first=True,
            dropout=Config['DROPOUT']
        )
        
        self.fc = nn.Linear(Config['HIDDEN_DIM'], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, prev_responses=None):
        # input_ids shape: (Batch, Seq_Len, Max_Text_Len)
        # 我們需要將 Batch 和 Seq_Len 合併來過 BERT，然後再拆開
        batch_size, seq_len, max_len = input_ids.size()
        
        flat_input_ids = input_ids.view(-1, max_len) # (Batch*Seq, Max_Len)
        flat_mask = attention_mask.view(-1, max_len)
        
        # Extract Text Embeddings
        with torch.no_grad(): # 如果不微調 BERT，這裡用 no_grad
            bert_out = self.bert(flat_input_ids, attention_mask=flat_mask)
            # 取 [CLS] token 作為句向量
            embeddings = bert_out.last_hidden_state[:, 0, :] # (Batch*Seq, 768)
            
        embeddings = embeddings.view(batch_size, seq_len, -1) # Restore (Batch, Seq, 768)
        
        # 準備 LSTM 輸入
        # 我們將 "當前題目 Embedding" 和 "上一題的作答結果" 拼接
        # 第一題的 "上一題結果" 補 0
        if prev_responses is not None:
            # prev_responses: (Batch, Seq)
            # Shift right: [r1, r2, r3] -> [0, r1, r2]
            prev_input = torch.cat([torch.zeros(batch_size, 1).to(device), prev_responses[:, :-1]], dim=1)
            prev_input = prev_input.unsqueeze(-1) # (Batch, Seq, 1)
            
            lstm_input = torch.cat([embeddings, prev_input], dim=2) # (Batch, Seq, 768+1)
        else:
            # Inference mode logic (needs handling state iteratively if real-time, 
            # but for Kaggle bulk predict, we usually assume predicted history or just 0s if cold start)
            # For simplicity in training code:
            zeros = torch.zeros(batch_size, seq_len, 1).to(device)
            lstm_input = torch.cat([embeddings, zeros], dim=2)

        # LSTM Forward
        lstm_out, _ = self.lstm(lstm_input) # (Batch, Seq, Hidden)
        
        # Prediction
        logits = self.fc(lstm_out)
        probs = self.sigmoid(logits)
        
        return probs.squeeze(-1)

# ==========================================
# 4. Main Execution Block
# ==========================================
def run_training():
    # Load Data
    train_df = pd.read_csv('./dataset/train.csv') # 修改路徑
    test_df = pd.read_csv('./dataset/test.csv')   # 修改路徑
    
    tokenizer = AutoTokenizer.from_pretrained(Config['MODEL_NAME'])
    
    # 由於每個學生的序列長度不同，這裡需要自定義 collate_fn 做 Padding
    def collate_fn(batch):
        # 找出這一個 batch 中最長的序列
        max_seq_len = max([item['input_ids'].shape[0] for item in batch])
        text_len = batch[0]['input_ids'].shape[1]
        
        batch_input_ids = []
        batch_masks = []
        batch_responses = []
        batch_masks_pad = [] # 用來標記哪些是 padding 的題目，不算 loss
        
        for item in batch:
            seq_len = item['input_ids'].shape[0]
            # Pad Input IDs (text)
            pad_text = torch.zeros((max_seq_len - seq_len, text_len), dtype=torch.long)
            batch_input_ids.append(torch.cat([item['input_ids'], pad_text], dim=0))
            
            # Pad Attention Mask
            pad_att = torch.zeros((max_seq_len - seq_len, text_len), dtype=torch.long)
            batch_masks.append(torch.cat([item['attention_mask'], pad_att], dim=0))
            
            # Pad Responses (用 -1 標記 padding)
            if 'responses' in item:
                pad_res = torch.full((max_seq_len - seq_len,), -1, dtype=torch.float)
                batch_responses.append(torch.cat([item['responses'], pad_res], dim=0))
            
            # Sequence Mask (1 for real data, 0 for padding)
            seq_mask = torch.cat([torch.ones(seq_len), torch.zeros(max_seq_len - seq_len)], dim=0)
            batch_masks_pad.append(seq_mask)
            
        return {
            'input_ids': torch.stack(batch_input_ids).to(device),
            'attention_mask': torch.stack(batch_masks).to(device),
            'responses': torch.stack(batch_responses).to(device) if batch_responses else None,
            'seq_mask': torch.stack(batch_masks_pad).to(device)
        }

    dataset = KTDataset(train_df, tokenizer, Config['MAX_LEN'])
    dataloader = DataLoader(dataset, batch_size=Config['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
    
    model = BertKTModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config['LR'])
    criterion = nn.BCELoss(reduction='none') # 使用 Mask 自行計算 Loss
    
    print("Start Training...")
    for epoch in range(Config['EPOCHS']):
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch in tqdm(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            mask = batch['attention_mask']
            targets = batch['responses']
            seq_mask = batch['seq_mask'] # (Batch, Seq)
            
            preds = model(input_ids, mask, targets) # 這裡用真實的 history (Teacher Forcing)
            
            # 只計算非 Padding 部分的 Loss
            loss = criterion(preds, targets)
            masked_loss = (loss * seq_mask).sum() / seq_mask.sum()
            
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
            
            # 收集數據計算 AUC (排除 padding)
            active_elements = seq_mask == 1
            all_preds.extend(preds[active_elements].detach().cpu().numpy())
            all_targets.extend(targets[active_elements].detach().cpu().numpy())
            
        auc = roc_auc_score(all_targets, all_preds)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}, AUC: {auc:.4f}")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("Generating Submission...")
    # 注意：測試時需按 student 分組並保留原始順序
    test_dataset = KTDataset(test_df, tokenizer, Config['MAX_LEN'], mode='test')
    
    # 這裡的 Inference 比較 tricky，因為沒有 response column
    # 我們可以假設學生之前的答對率，或是用模型自身的預測作為下一題的輸入 (Autoregressive)
    # 為了簡化，這裡演示如何生成提交文件
    # 實際操作建議：將 train 和 test 合併，按 student_id 和 interaction_id 排序
    # 跑一次完整的 forward pass，然後只取 test 部分的 rows
    
    # ... (省略合併代碼，邏輯如上) ...
    # 最終輸出需要包含 interaction_id 和 response (probability)
    
    # save to csv
    # submission = pd.DataFrame({'interaction_id': ids, 'response': probs})
    # submission.to_csv('final_group_xx.csv', index=False)

if __name__ == "__main__":
    run_training()