import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from collections import Counter
import re
import os
import random
import sys
from tqdm import tqdm

# --- 1. 設定參數 (針對 8x V100 優化) ---
MAX_TITLE_LEN = 30
MAX_HISTORY_LEN = 50

# [優化] 單卡 Batch Size 加大，V100 32GB 絕對吃得下 512 甚至 1024
# 總 Batch Size = 512 * 8 = 4096
BATCH_SIZE = 512  

NUM_EPOCHS = 30
# [優化] 因為總 Batch Size 變大，Learning Rate 建議稍微調大
LEARNING_RATE = 0.001  
NUM_NEGATIVES = 4

# Embedding 設定
WORD_EMB_DIM = 100
CAT_EMB_DIM = 50
NEWS_HEADS = 20
NEWS_DIM = 250
TOTAL_NEWS_DIM = NEWS_DIM + CAT_EMB_DIM
USER_HEADS = 20

# 路徑設定
PATH_TRAIN_NEWS = 'dataset/train/train_news.tsv'
PATH_TRAIN_BEHAVIORS = 'dataset/train/train_behaviors.tsv'
PATH_GLOVE = 'dataset/glove_6B_100d.txt'
PATH_TEST_NEWS = 'dataset/test/test_news.tsv'
PATH_TEST_BEHAVIORS = 'dataset/test/test_behaviors.tsv'

# --- 2. DDP 工具函數 ---

def setup_ddp():
    # torchrun 會自動設定環境變數
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank, rank, world_size
    else:
        # 單機 fallback
        print("Not running in DDP mode. Using standard single GPU.")
        return 0, 0, 1

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

# --- 3. 模型定義 (保持不變，但在 Main 中會被 DDP 包裝) ---

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        N, seq_length, _ = x.shape
        Q = self.query(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.d_model ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous().view(N, seq_length, self.d_model)
        return self.fc_out(out)

class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, query_dim=200):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim, 1))
        
    def forward(self, x):
        attn_scores = torch.matmul(torch.tanh(self.linear(x)), self.query)
        attn_weights = F.softmax(attn_scores, dim=1)
        output = torch.bmm(attn_weights.permute(0, 2, 1), x).squeeze(1)
        return output

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, cat_size, pretrained_emb=None):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, WORD_EMB_DIM, padding_idx=0)
        if pretrained_emb is not None:
            self.word_embedding.weight.data.copy_(pretrained_emb)
            self.word_embedding.weight.requires_grad = True
        self.cat_embedding = nn.Embedding(cat_size, CAT_EMB_DIM, padding_idx=0)
        self.self_attn = MultiHeadSelfAttention(WORD_EMB_DIM, NEWS_HEADS)
        self.dropout = nn.Dropout(0.2)
        self.attn_pool = AdditiveAttention(WORD_EMB_DIM, 200)
        self.final_proj = nn.Linear(WORD_EMB_DIM + CAT_EMB_DIM, TOTAL_NEWS_DIM)
        
    def forward(self, title_inputs, cat_inputs):
        w_emb = self.dropout(self.word_embedding(title_inputs))
        title_rep = self.self_attn(w_emb)
        title_vec = self.attn_pool(title_rep)
        c_emb = self.cat_embedding(cat_inputs)
        combined = torch.cat([title_vec, c_emb], dim=1)
        return F.relu(self.final_proj(combined))

class UserEncoder(nn.Module):
    def __init__(self, news_dim):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(news_dim, USER_HEADS)
        self.dropout = nn.Dropout(0.2)
        self.attn_pool = AdditiveAttention(news_dim, 200)
        
    def forward(self, history_vecs):
        h_rep = self.dropout(self.self_attn(history_vecs))
        return self.attn_pool(h_rep)

class AdvancedNRMS(nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        # 註冊 Buffer 讓 DDP 可以管理這些常量
        self.register_buffer('news_title_matrix', torch.LongTensor(preprocessor.news_title_matrix))
        self.register_buffer('news_cat_matrix', torch.LongTensor(preprocessor.news_cat_matrix))
        
        vocab_size = len(preprocessor.word2int)
        cat_size = len(preprocessor.cat2int)
        
        self.news_encoder = NewsEncoder(vocab_size, cat_size, preprocessor.glove_matrix)
        self.user_encoder = UserEncoder(news_dim=TOTAL_NEWS_DIM)
        
    def forward(self, history_ids, candidate_ids):
        hist_title = self.news_title_matrix[history_ids]
        hist_cat = self.news_cat_matrix[history_ids]
        cand_title = self.news_title_matrix[candidate_ids]
        cand_cat = self.news_cat_matrix[candidate_ids]
        
        batch_size, hist_len, seq_len = hist_title.shape
        hist_vecs = self.news_encoder(hist_title.view(-1, seq_len), hist_cat.view(-1)).view(batch_size, hist_len, -1)
        user_vec = self.user_encoder(hist_vecs)
        
        cand_len = cand_title.shape[1]
        cand_vecs = self.news_encoder(cand_title.view(-1, seq_len), cand_cat.view(-1)).view(batch_size, cand_len, -1)
        
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(2)).squeeze(2)
        return scores

# --- 4. 資料處理 (Preprocessor & Dataset) ---

class NewsPreprocessor:
    def __init__(self, news_df, entity_path=None, glove_path=None):
        self.news_df = news_df
        self.news2int = {nid: i+1 for i, nid in enumerate(self.news_df['news_id'])}
        self.cat2int = {'<PAD>': 0, '<UNK>': 1}
        self.word2int = {'<PAD>': 0, '<UNK>': 1}
        
        # 只在主進程打印 Log
        self.verbose = is_main_process()
        
        self.build_category_vocab()
        self.build_vocab()
        
        self.glove_matrix = None
        if glove_path:
            self.glove_matrix = self.load_glove(glove_path, embedding_dim=WORD_EMB_DIM)
            
        if self.verbose: print("Caching news features...")
        n_news = len(self.news2int) + 1
        self.news_title_matrix = np.zeros((n_news, MAX_TITLE_LEN), dtype=np.int64)
        self.news_cat_matrix = np.zeros((n_news,), dtype=np.int64)
        self.process_all_news()

    def clean_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        return re.sub(r'[^\w\s]', '', text)

    def build_vocab(self):
        if self.verbose: print("Building word vocabulary...")
        words = []
        # 使用簡單遍歷，避免多進程 tqdm 衝突
        for title in self.news_df['title']:
            words.extend(self.clean_text(title).split())
        counts = Counter(words)
        for word, _ in counts.most_common(40000):
            self.word2int[word] = len(self.word2int)
            
    def build_category_vocab(self):
        cats = self.news_df['category'].fillna('Unknown').unique()
        for c in cats:
            if c not in self.cat2int:
                self.cat2int[c] = len(self.cat2int)

    def load_glove(self, glove_path, embedding_dim):
        if self.verbose: print(f"Loading GloVe from {glove_path}...")
        embeddings_index = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                if len(values) != embedding_dim + 1: continue
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except: continue
        
        vocab_size = len(self.word2int)
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        embedding_matrix[1] = np.random.normal(scale=0.6, size=(embedding_dim, ))
        
        for word, i in self.word2int.items():
            vec = embeddings_index.get(word)
            if vec is not None: embedding_matrix[i] = vec
        return torch.FloatTensor(embedding_matrix)

    def process_all_news(self):
        # 簡單處理，不使用 tqdm 以保持 log 乾淨
        for idx, row in self.news_df.iterrows():
            nid_idx = self.news2int[row['news_id']]
            words = self.clean_text(row['title']).split()
            word_indices = [self.word2int.get(w, 1) for w in words][:MAX_TITLE_LEN]
            word_indices += [0] * (MAX_TITLE_LEN - len(word_indices))
            self.news_title_matrix[nid_idx] = word_indices
            cat = row['category'] if not pd.isna(row['category']) else 'Unknown'
            self.news_cat_matrix[nid_idx] = self.cat2int.get(cat, 1)

class BehaviorsDataset(Dataset):
    def __init__(self, behaviors_path, news_preprocessor, mode='train', num_negatives=4):
        self.preprocessor = news_preprocessor
        self.mode = mode
        self.num_negatives = num_negatives
        self.data = []
        
        if is_main_process():
            print(f"Loading behaviors from {behaviors_path}...")
        
        try:
            df = pd.read_csv(behaviors_path, sep='\t', header=0, names=['idx', 'user_id', 'time', 'history', 'impressions'])
        except:
            df = pd.read_csv(behaviors_path, sep='\t', header=None, names=['idx', 'user_id', 'time', 'history', 'impressions'])
            
        news2int = self.preprocessor.news2int
        
        # 為了加速讀取，這邊不使用太複雜的邏輯
        rows = df.values.tolist()
        for row in rows:
            # row: [idx, user_id, time, history, impressions]
            hist_raw = str(row[3])
            if hist_raw == 'nan': hist_list = []
            else: hist_list = hist_raw.split()
            
            hist_indices = [news2int.get(nid, 0) for nid in hist_list][-MAX_HISTORY_LEN:]
            hist_indices += [0] * (MAX_HISTORY_LEN - len(hist_indices))
            
            impressions_raw = str(row[4]).split()
            
            if mode == 'train':
                positives, negatives = [], []
                for imp in impressions_raw:
                    parts = imp.split('-')
                    if len(parts) != 2: continue
                    nid, label = parts
                    idx = news2int.get(nid, 0)
                    if int(label) == 1: positives.append(idx)
                    else: negatives.append(idx)
                
                for pos_idx in positives:
                    self.data.append({'history': hist_indices, 'candidate': pos_idx, 'label': 1})
                    if len(negatives) > 0:
                        neg_samples = random.choices(negatives, k=self.num_negatives)
                        for neg_idx in neg_samples:
                            self.data.append({'history': hist_indices, 'candidate': neg_idx, 'label': 0})
            else:
                cand_indices = [news2int.get(imp.split('-')[0], 0) for imp in impressions_raw]
                if len(cand_indices) != 15: continue
                self.data.append({'id': row[0], 'history': hist_indices, 'candidate': cand_indices})

    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        hist = torch.LongTensor(item['history'])
        if self.mode == 'train':
            cand = torch.LongTensor([item['candidate']])
            label = torch.FloatTensor([item['label']])
            return hist, cand, label
        else:
            cand = torch.LongTensor(item['candidate'])
            return item['id'], hist, cand

# --- 5. 訓練與預測函數 ---

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch):
    model.train()
    
    # DDP 需要在每個 epoch 設定 sampler
    if dataloader.sampler is not None:
        dataloader.sampler.set_epoch(epoch)
        
    total_loss = 0
    
    # 只在 rank 0 顯示進度條
    if is_main_process():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    else:
        pbar = dataloader

    for hist, cand, label in pbar:
        hist, cand, label = hist.to(device, non_blocking=True), cand.to(device, non_blocking=True), label.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # [優化] AMP 混合精度運算
        with autocast():
            scores = model(hist, cand)
            loss = criterion(scores, label)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if is_main_process():
            pbar.set_postfix({'loss': loss.item()})
            
    return total_loss / len(dataloader)

def generate_submission(model, test_loader, device, output_path='submission.csv'):
    model.eval()
    results = []
    print(f"Start inference on Rank 0...")
    with torch.no_grad():
        for imp_ids, hist, cand in tqdm(test_loader, desc="Predicting"):
            hist, cand = hist.to(device), cand.to(device)
            # 使用 model 進行推論 (注意 DDP 模式下可能是 model.module)
            scores = model(hist, cand)
            probs = torch.sigmoid(scores).cpu().numpy()
            imp_ids = imp_ids.numpy()
            
            for i in range(len(imp_ids)):
                results.append([imp_ids[i]] + probs[i].tolist())

    cols = ['id'] + [f'p{i}' for i in range(1, 16)]
    submission_df = pd.DataFrame(results, columns=cols)
    submission_df.sort_values(by='id').to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

# --- 6. 主程式 ---

def main():
    # 1. 初始化 DDP
    local_rank, rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # 2. 準備資料 (Preprocess)
    # 注意：這裡為了程式碼簡單，所有 GPU 都會讀取資料 (32GB VRAM 足夠)
    # 正式大規模系統可能會讓 Rank 0 處理完存成 cache，其他人讀 cache
    if rank == 0: print("Merging Train and Test News...")
    
    cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    df_train = pd.read_csv(PATH_TRAIN_NEWS, sep='\t', header=None, names=cols)
    df_test = pd.read_csv(PATH_TEST_NEWS, sep='\t', header=None, names=cols)
    df_all = pd.concat([df_train, df_test]).drop_duplicates(subset='news_id').reset_index(drop=True)
    
    preprocessor = NewsPreprocessor(df_all, glove_path=PATH_GLOVE)
    
    # 3. Dataset & DDP Sampler
    train_ds = BehaviorsDataset(PATH_TRAIN_BEHAVIORS, preprocessor, mode='train', num_negatives=NUM_NEGATIVES)
    
    # 使用 DistributedSampler 切分資料
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, # Shuffle 由 Sampler 負責
        num_workers=4, # 每個 GPU 開 4 個 subprocess，8 卡共 32
        pin_memory=True, 
        sampler=train_sampler,
        persistent_workers=True
    )
    
    # 4. Model Setup
    model = AdvancedNRMS(preprocessor).to(device)
    
    # 將 BatchNorm 轉為 SyncBatchNorm (如果有用的話，這裡主要是保險)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # DDP 包裝
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler() # 初始化 AMP Scaler
    
    # 5. Training Loop
    for epoch in range(NUM_EPOCHS):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch)
        
        if rank == 0:
            print(f"Epoch Loss (Rank 0 approx): {loss:.4f}")
            # 存檔時，記得存 model.module
            checkpoint = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(checkpoint, f"model_epoch_{epoch+1}.pth")
            
        # 等待所有 GPU 完成這一輪，再進入下一輪
        dist.barrier()

    # 6. Inference (Only Rank 0)
    if rank == 0:
        print("Training finished. Starting inference...")
        # 測試集不需要 DDP，單卡跑即可 (或你可以也寫成 DDP，但處理 ID 順序比較麻煩)
        test_ds = BehaviorsDataset(PATH_TEST_BEHAVIORS, preprocessor, mode='test')
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        eval_model = model.module if hasattr(model, 'module') else model
        generate_submission(eval_model, test_loader, device, output_path='submission_8gpu.csv')
    
    cleanup_ddp()

if __name__ == "__main__":
    main()