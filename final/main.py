import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import os

# ==========================================
# 1. 設定與參數 (Configuration)
# ==========================================
CONFIG = {
    'MAX_SEQ_LEN': 100,       # 序列最大長度
    'EMBED_DIM': 384,         # MiniLM 的輸出維度
    'HIDDEN_DIM': 128,        # LSTM 隱藏層維度
    'BATCH_SIZE': 32,
    'EPOCHS': 5,              # 訓練輪數 (可視情況增加)
    'LR': 1e-3,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'TRAIN_PATH': './dataset/train.csv',
    'TEST_PATH': './dataset/test.csv',
    'SUBMISSION_PATH': 'submission.csv'
}

print(f"Using device: {CONFIG['DEVICE']}")

# ==========================================
# 2. 資料處理 (Data Preprocessing)
# ==========================================
def load_and_preprocess():
    print("正在讀取資料...")
    # 讀取資料
    train_df = pd.read_csv(CONFIG['TRAIN_PATH'])
    test_df = pd.read_csv(CONFIG['TEST_PATH'])
    
    # 標記來源並初始化 Test 的 response
    train_df['is_test'] = False
    test_df['is_test'] = True
    test_df['response'] = 0 # Placeholder，不會被用到
    
    # 確保欄位一致，只取共有的重要欄位
    common_cols = ['interaction_id', 'question_id', 'Problem', 'student_id', 'response', 'is_test']
    # 注意：我們不使用 student_process，因為 Test set 裡沒有，且無法作為預測當下的輸入
    
    full_df = pd.concat([train_df[common_cols], test_df[common_cols]], axis=0)
    
    # 按照學生與時間排序 (假設 interaction_id 代表時間順序)
    full_df = full_df.sort_values(['student_id', 'interaction_id'])
    
    # 填補缺失值
    full_df['Problem'] = full_df['Problem'].fillna('')
    
    # Label Encoding for Question ID
    le = LabelEncoder()
    full_df['q_encoded'] = le.fit_transform(full_df['question_id'])
    NUM_QUESTIONS = len(le.classes_)
    
    print(f"總資料筆數: {len(full_df)}, 題目總數: {NUM_QUESTIONS}")
    
    return full_df, NUM_QUESTIONS

# ==========================================
# 3. NLP 特徵提取 (Feature Extraction)
# ==========================================
def extract_text_embeddings(df):
    print("正在提取文本向量 (使用 sentence-transformers)...")
    # 使用輕量級但強大的模型
    model_name = 'all-MiniLM-L6-v2' 
    try:
        model = SentenceTransformer(model_name, device=CONFIG['DEVICE'])
    except Exception as e:
        print(f"無法載入 {model_name}，嘗試使用 CPU 或檢查網路。錯誤: {e}")
        model = SentenceTransformer(model_name, device='cpu')
        
    # 為了加速，我們只對「唯一」的題目文字做 Embedding
    unique_problems = df[['q_encoded', 'Problem']].drop_duplicates('q_encoded')
    problem_texts = unique_problems['Problem'].tolist()
    
    embeddings = model.encode(problem_texts, show_progress_bar=True, convert_to_numpy=True)
    
    # 建立 ID 到 Embedding 的對照表
    id_to_emb = {qid: emb for qid, emb in zip(unique_problems['q_encoded'], embeddings)}
    
    # 將 Embedding 映射回主 Dataframe
    # 這裡我們先存成一個大矩陣，方便 Dataset 索引
    emb_matrix = np.zeros((df['q_encoded'].max() + 1, CONFIG['EMBED_DIM']))
    for qid, emb in id_to_emb.items():
        emb_matrix[qid] = emb
        
    return torch.tensor(emb_matrix, dtype=torch.float32)

# ==========================================
# 4. Dataset 定義
# ==========================================
class KTDataset(Dataset):
    def __init__(self, df, problem_emb_matrix, max_len):
        self.data = []
        self.problem_emb_matrix = problem_emb_matrix
        self.max_len = max_len
        
        # 將資料按學生分組
        for student_id, group in df.groupby('student_id'):
            # 獲取序列特徵
            q_ids = group['q_encoded'].values
            responses = group['response'].values
            interaction_ids = group['interaction_id'].values
            is_test_mask = group['is_test'].values
            
            # 切割過長的序列 (或者使用 Sliding Window，這裡採簡單截斷)
            # 為了保留最新的 Test 資訊，我們取最後 max_len
            if len(q_ids) > max_len:
                start_idx = len(q_ids) - max_len
            else:
                start_idx = 0
                
            self.data.append({
                'q_seq': q_ids[start_idx:],
                'r_seq': responses[start_idx:],
                'int_id_seq': interaction_ids[start_idx:],
                'mask_seq': is_test_mask[start_idx:] # 用來標記哪些是需要預測的 Test Data
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        seq_len = len(row['q_seq'])
        
        # Padding
        pad_len = self.max_len - seq_len
        
        # Question IDs (Padding with 0, though usually safe if masked)
        q = np.pad(row['q_seq'], (pad_len, 0), 'constant', constant_values=0)
        
        # Responses (Padding with 0)
        r = np.pad(row['r_seq'], (pad_len, 0), 'constant', constant_values=0)
        
        # Mask (True for Test data)
        mask = np.pad(row['mask_seq'], (pad_len, 0), 'constant', constant_values=False)
        
        # Interaction IDs (for output)
        int_ids = np.pad(row['int_id_seq'], (pad_len, 0), 'constant', constant_values=-1)
        
        return {
            'q': torch.tensor(q, dtype=torch.long),
            'r': torch.tensor(r, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'int_ids': torch.tensor(int_ids, dtype=torch.long)
        }

# ==========================================
# 5. 模型定義 (LSTM-based KT)
# ==========================================
class TextKTModel(nn.Module):
    def __init__(self, num_questions, problem_emb_matrix, hidden_dim, embed_dim):
        super(TextKTModel, self).__init__()
        self.problem_emb_matrix = nn.Parameter(problem_emb_matrix, requires_grad=False) # 固定預訓練向量
        
        # ID Embedding
        self.q_embedding = nn.Embedding(num_questions, 64)
        
        # Input Projection: ID(64) + Text(384) + Prev_Response(1)
        input_dim = 64 + embed_dim + 1
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, q_seq, r_seq):
        # q_seq: [batch, len]
        # r_seq: [batch, len] (這是 Ground Truth，輸入時要 shift)
        
        batch_size, seq_len = q_seq.shape
        
        # 1. 取得 Question Features
        q_emb = self.q_embedding(q_seq) # [batch, len, 64]
        text_emb = self.problem_emb_matrix[q_seq] # [batch, len, 384]
        
        # 2. 準備 Response Input (Shifted)
        # 輸入給 t 的是 r_{t-1}。第一個時間點輸入 0。
        r_input = torch.cat([torch.zeros(batch_size, 1).to(q_seq.device), r_seq[:, :-1]], dim=1)
        r_input = r_input.unsqueeze(-1) # [batch, len, 1]
        
        # 3. 結合所有特徵
        x = torch.cat([q_emb, text_emb, r_input], dim=-1)
        
        # 4. 模型前向傳播
        x = torch.relu(self.input_proj(x))
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        probs = self.sigmoid(logits)
        
        return probs.squeeze(-1)

# ==========================================
# 6. 主程式 (Main)
# ==========================================
if __name__ == "__main__":
    # A. 載入資料
    full_df, num_questions = load_and_preprocess()
    
    # B. 提取特徵
    problem_emb_matrix = extract_text_embeddings(full_df)
    
    # C. 建立 Dataset
    dataset = KTDataset(full_df, problem_emb_matrix, CONFIG['MAX_SEQ_LEN'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    
    # D. 初始化模型
    model = TextKTModel(num_questions, problem_emb_matrix, CONFIG['HIDDEN_DIM'], CONFIG['EMBED_DIM'])
    model = model.to(CONFIG['DEVICE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
    criterion = nn.BCELoss()
    
    # E. 訓練迴圈
    print("開始訓練...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        for batch in dataloader:
            q = batch['q'].to(CONFIG['DEVICE'])
            r = batch['r'].to(CONFIG['DEVICE'])
            mask = batch['mask'].to(CONFIG['DEVICE']) # mask 為 True 代表是 Test Data
            
            optimizer.zero_grad()
            
            preds = model(q, r)
            
            # 計算 Loss 時，只看 Train Data (mask == False) 且不是 Padding 的部分
            # 注意：這裡簡單處理，實際 Padding 位置的 q 應為 0，可以額外加 padding mask
            train_mask = (~mask) & (batch['int_ids'].to(CONFIG['DEVICE']) != -1)
            
            if train_mask.sum() > 0:
                loss = criterion(preds[train_mask], r[train_mask])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}, Loss: {total_loss/len(dataloader):.4f}")

    # F. 預測與產生 Submission
    print("開始預測並產生 Submission...")
    model.eval()
    submission_results = []
    
    # 重新建立一個不 Shuffle 的 Loader 確保順序 (或直接用 dataset loop)
    # 這裡直接遍歷 dataset 比較簡單，因為我們需要每一筆 Test 資料
    with torch.no_grad():
        for i in range(len(dataset)):
            item = dataset[i]
            # Add batch dimension
            q = item['q'].unsqueeze(0).to(CONFIG['DEVICE'])
            r = item['r'].unsqueeze(0).to(CONFIG['DEVICE'])
            mask = item['mask'].unsqueeze(0).to(CONFIG['DEVICE'])
            int_ids = item['int_ids'].unsqueeze(0).to(CONFIG['DEVICE'])
            
            preds = model(q, r) # [1, seq_len]
            
            # 取出標記為 Test 的預測結果
            test_mask = mask[0] # [seq_len]
            test_preds = preds[0][test_mask]
            test_ids = int_ids[0][test_mask]
            
            for tid, tpred in zip(test_ids.cpu().numpy(), test_preds.cpu().numpy()):
                if tid != -1: # 確保不是 padding
                    submission_results.append({'interaction_id': tid, 'response': float(tpred)})

    # G. 儲存結果
    sub_df = pd.DataFrame(submission_results)
    
    # 再次確保格式正確 (Kaggle 有時對順序敏感，雖然這裡只有 interaction_id)
    # 讀取 sample 來確認 ID 順序 (Optional，但推薦)
    try:
        sample_sub = pd.read_csv('sample_submission.csv') # 如果你有的話
        # 只保留 sample 裡有的 ID (確保沒預測到多餘的)
        required_ids = sample_sub['interaction_id'].unique()
        sub_df = sub_df[sub_df['interaction_id'].isin(required_ids)]
        # 補回缺失的 ID (如果有的話，填 0.5)
        # ... (略，假設都預測到了)
    except:
        pass

    sub_df.to_csv(CONFIG['SUBMISSION_PATH'], index=False)
    print(f"檔案已儲存至 {CONFIG['SUBMISSION_PATH']}")
    print(sub_df.head())