import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import os

# ==========================================
# 0. 設定與環境
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.device_count() > 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")

Config = {
    'MAX_LEN': 128,           # BERT 文本最大長度
    'MAX_HISTORY': 256,       # [新增] 限制最大歷史題數，避免 OOM
    'BATCH_SIZE': 16,         # [修改] 降低 Batch Size (64 -> 16)
    'EPOCHS': 15,
    'LR': 2e-5,
    'HIDDEN_DIM': 256,
    'MODEL_NAME': 'bert-base-uncased',
    'NUM_LAYERS': 2,
    'DROPOUT': 0.3,
    'n_folds': 5
}

# ==========================================
# 1. 資料集定義 (加入截斷邏輯)
# ==========================================
class KTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, max_history, mode='train'):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_history = max_history # 新增
        self.mode = mode
        
        self.df = self.df.sort_values(['student_id', 'interaction_id'])
        self.group = self.df.groupby('student_id')
        self.student_ids = list(self.group.groups.keys())
        
    def __len__(self):
        return len(self.student_ids)
    
    def __getitem__(self, idx):
        student_id = self.student_ids[idx]
        student_df = self.group.get_group(student_id)
        
        # === [關鍵修改] 截斷過長的歷史 ===
        # 如果學生做題超過 MAX_HISTORY (例如 256 題)，只取最後 256 題
        if len(student_df) > self.max_history:
            student_df = student_df.iloc[-self.max_history:]
        
        # 提取特徵
        problems = student_df['Problem'].fillna("").values
        concepts = student_df['concepts_of_the_problem'].fillna("").values
        
        text_inputs = [f"{p} [SEP] {c}" for p, c in zip(problems, concepts)]
        
        encoding = self.tokenizer(
            text_inputs,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'interaction_ids': torch.tensor(student_df['interaction_id'].values, dtype=torch.long)
        }
        
        if self.mode == 'train':
            responses = student_df['response'].values
            item['responses'] = torch.tensor(responses, dtype=torch.float)
            
        return item

def collate_fn(batch):
    # Pad 到該 Batch 的最大長度 (但已被 MAX_HISTORY 限制住，不會無限大)
    max_seq_len = max([item['input_ids'].shape[0] for item in batch])
    text_len = batch[0]['input_ids'].shape[1]
    
    batch_input_ids = []
    batch_masks = []
    batch_responses = []
    batch_interaction_ids = []
    batch_seq_masks = []
    
    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_seq_len - seq_len
        
        # Pad Input IDs
        pad_text = torch.zeros((pad_len, text_len), dtype=torch.long)
        batch_input_ids.append(torch.cat([item['input_ids'], pad_text], dim=0))
        
        # Pad Attention Mask
        pad_att = torch.zeros((pad_len, text_len), dtype=torch.long)
        batch_masks.append(torch.cat([item['attention_mask'], pad_att], dim=0))
        
        # Pad Interaction IDs
        pad_ids = torch.full((pad_len,), -1, dtype=torch.long)
        batch_interaction_ids.append(torch.cat([item['interaction_ids'], pad_ids], dim=0))
        
        # Sequence Mask
        seq_mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)], dim=0)
        batch_seq_masks.append(seq_mask)
        
        if 'responses' in item:
            # Pad Response with 0
            pad_res = torch.zeros((pad_len,), dtype=torch.float)
            batch_responses.append(torch.cat([item['responses'], pad_res], dim=0))
            
    out = {
        'input_ids': torch.stack(batch_input_ids).to(device),
        'attention_mask': torch.stack(batch_masks).to(device),
        'seq_mask': torch.stack(batch_seq_masks).to(device),
        'interaction_ids': torch.stack(batch_interaction_ids).to(device)
    }
    if batch_responses:
        out['responses'] = torch.stack(batch_responses).to(device)
        
    return out

# ==========================================
# 2. 模型架構 (BERT + LSTM) - 無變更
# ==========================================
class BertKTModel(nn.Module):
    def __init__(self):
        super(BertKTModel, self).__init__()
        self.bert = AutoModel.from_pretrained(Config['MODEL_NAME'])
        
        # 如果仍然 OOM，可以取消註解下面這行來凍結 BERT 參數
        # for param in self.bert.parameters(): param.requires_grad = False
        
        embedding_dim = self.bert.config.hidden_size
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim + 1,
            hidden_size=Config['HIDDEN_DIM'],
            num_layers=Config['NUM_LAYERS'],
            batch_first=True,
            dropout=Config['DROPOUT']
        )
        
        self.fc = nn.Linear(Config['HIDDEN_DIM'], 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask, prev_responses=None):
        batch_size, seq_len, max_len = input_ids.size()
        
        flat_input = input_ids.view(-1, max_len)
        flat_mask = attention_mask.view(-1, max_len)
        
        outputs = self.bert(flat_input, attention_mask=flat_mask)
        text_emb = outputs.last_hidden_state[:, 0, :]
        text_emb = text_emb.view(batch_size, seq_len, -1)
        
        if prev_responses is None:
            feature_concat = torch.zeros(batch_size, seq_len, 1).to(input_ids.device)
        else:
            prev_input = torch.cat([torch.zeros(batch_size, 1).to(input_ids.device), prev_responses[:, :-1]], dim=1)
            feature_concat = prev_input.unsqueeze(-1)
            
        lstm_input = torch.cat([text_emb, feature_concat], dim=2)
        lstm_out, _ = self.lstm(lstm_input)
        
        logits = self.fc(lstm_out)
        probs = self.sigmoid(logits)
        return probs.squeeze(-1)

# ==========================================
# 3. 訓練與推論流程
# ==========================================
def train_and_predict():
    print("Loading Data...")
    train_df = pd.read_csv('./dataset/train.csv')
    test_df = pd.read_csv('./dataset/test.csv')
    
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    test_df['response'] = 0
    
    tokenizer = AutoTokenizer.from_pretrained(Config['MODEL_NAME'])
    
    print("Preparing Datasets...")
    # 傳入 Config['MAX_HISTORY']
    full_train_dataset = KTDataset(train_df, tokenizer, Config['MAX_LEN'], Config['MAX_HISTORY'], mode='train')
    train_loader = DataLoader(full_train_dataset, batch_size=Config['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
    
    test_dataset = KTDataset(test_df, tokenizer, Config['MAX_LEN'], Config['MAX_HISTORY'], mode='test')
    test_loader = DataLoader(test_dataset, batch_size=Config['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn)
    
    model = BertKTModel()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config['LR'])
    criterion = nn.BCELoss(reduction='none')
    
    print(f"Start Training for {Config['EPOCHS']} Epochs...")
    for epoch in range(Config['EPOCHS']):
        model.train()
        total_loss = 0
        
        # 進度條
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids']
            mask = batch['attention_mask']
            seq_mask = batch['seq_mask']
            targets = batch['responses']
            
            preds = model(input_ids, mask, targets)
            loss = criterion(preds, targets)
            masked_loss = (loss * seq_mask).sum() / seq_mask.sum()
            
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
            pbar.set_postfix({'loss': masked_loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")
        
    print("Predicting on Test Set...")
    model.eval()
    submission_data = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids']
            mask = batch['attention_mask']
            seq_mask = batch['seq_mask']
            interaction_ids = batch['interaction_ids']
            
            preds = model(input_ids, mask, prev_responses=None) 
            
            batch_size, seq_len = input_ids.shape
            for i in range(batch_size):
                valid_len = int(seq_mask[i].sum().item())
                s_ids = interaction_ids[i, :valid_len].cpu().numpy()
                s_preds = preds[i, :valid_len].cpu().numpy()
                
                for i_id, prob in zip(s_ids, s_preds):
                    submission_data.append({'interaction_id': i_id, 'response': prob})
    
    submit_df = pd.DataFrame(submission_data)
    submit_df = submit_df.sort_values('interaction_id')
    submit_df.to_csv('submission.csv', index=False)
    print("Done! Submission saved to 'submission.csv'.")

if __name__ == "__main__":
    train_and_predict()