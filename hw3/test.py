import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import json
import re
import os
import random
from tqdm import tqdm

# --- 1. 設定參數 ---
MAX_TITLE_LEN = 30
MAX_HISTORY_LEN = 50
BATCH_SIZE = 64          # Self-Attention 比較吃顯存，若 OOM 可改為 32
NUM_EPOCHS = 4          # 因為有負採樣，訓練變快，可以多跑幾輪
LEARNING_RATE = 0.0005   # Transformer 類模型建議 LR 小一點
NUM_NEGATIVES = 4        # 負採樣比例 1:4

# Embedding 設定
WORD_EMB_DIM = 100       # 配合 GloVe 100d
CAT_EMB_DIM = 50         # Category 向量維度
NEWS_HEADS = 20          # News Attention Heads (100 / 20 = 5)
NEWS_DIM = 250           # News Title 經過 Attention 後的輸出維度
TOTAL_NEWS_DIM = NEWS_DIM + CAT_EMB_DIM # 250 + 50 = 300
USER_HEADS = 20          # User Attention Heads (300 / 20 = 15)

# 路徑設定 (請確認檔案存在)
PATH_TRAIN_NEWS = 'dataset/train/train_news.tsv'
PATH_TRAIN_BEHAVIORS = 'dataset/train/train_behaviors.tsv'
PATH_TRAIN_ENTITY = 'dataset/train/train_entity_embedding.vec'
PATH_GLOVE = 'dataset/glove_6B_100d.txt' # 請確保此檔案存在

PATH_TEST_NEWS = 'dataset/test/test_news.tsv'
PATH_TEST_BEHAVIORS = 'dataset/test/test_behaviors.tsv'

# --- 2. 工具模組: Attention Layers ---

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
        # x shape: (Batch, Seq, Dim)
        N, seq_length, _ = x.shape
        
        Q = self.query(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(N, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / (self.d_model ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(N, seq_length, self.d_model)
        return self.fc_out(out)

class AdditiveAttention(nn.Module):
    '''將 Sequence 壓縮成一個 Vector (Pooling)'''
    def __init__(self, input_dim, query_dim=200):
        super().__init__()
        self.linear = nn.Linear(input_dim, query_dim)
        self.query = nn.Parameter(torch.randn(query_dim, 1))
        
    def forward(self, x):
        # x: (Batch, Seq, Dim)
        attn_scores = torch.matmul(torch.tanh(self.linear(x)), self.query) # (Batch, Seq, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        # Weighted Sum
        output = torch.bmm(attn_weights.permute(0, 2, 1), x).squeeze(1) # (Batch, Dim)
        return output

# --- 3. 資料處理 (Preprocessor & Dataset) ---

class NewsPreprocessor:
    def __init__(self, news_df, entity_path=None, glove_path=None):
        self.news_df = news_df
        
        # ID Mapping
        self.news2int = {nid: i+1 for i, nid in enumerate(self.news_df['news_id'])}
        self.int2news = {i+1: nid for i, nid in enumerate(self.news_df['news_id'])}
        
        # Category Mapping
        self.cat2int = {'<PAD>': 0, '<UNK>': 1}
        self.build_category_vocab()
        
        # Word Vocab
        self.word2int = {'<PAD>': 0, '<UNK>': 1}
        self.build_vocab()
        
        # GloVe
        self.glove_matrix = None
        if glove_path:
            self.glove_matrix = self.load_glove(glove_path, embedding_dim=WORD_EMB_DIM)
            
        # Cache Matrices
        print("Caching news features...")
        n_news = len(self.news2int) + 1
        self.news_title_matrix = np.zeros((n_news, MAX_TITLE_LEN), dtype=np.int64)
        self.news_cat_matrix = np.zeros((n_news,), dtype=np.int64) # 儲存 Category ID
        
        self.process_all_news()

    def clean_text(self, text):
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def build_vocab(self):
        print("Building word vocabulary...")
        words = []
        for title in tqdm(self.news_df['title'], desc="Vocab"):
            words.extend(self.clean_text(title).split())
        counts = Counter(words)
        for word, _ in counts.most_common(40000):
            self.word2int[word] = len(self.word2int)
            
    def build_category_vocab(self):
        print("Building category vocabulary...")
        cats = self.news_df['category'].fillna('Unknown').unique()
        for c in cats:
            if c not in self.cat2int:
                self.cat2int[c] = len(self.cat2int)
        print(f"Categories: {len(self.cat2int)}")

    def load_glove(self, glove_path, embedding_dim):
        print(f"Loading GloVe from {glove_path}...")
        embeddings_index = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                
                # --- [修正重點] 改回嚴格檢查長度 ---
                # 確保這一行的長度剛好是 "單字" + "100維向量"
                # 如果長度不對 (例如遇到含有空白的髒資料)，就直接跳過
                if len(values) != embedding_dim + 1:
                    continue
                # --------------------------------
                
                word = values[0]
                try:
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
                except ValueError:
                    # 雙重保險：萬一還是轉不過，就跳過這行
                    continue
        
        vocab_size = len(self.word2int)
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        # 初始化 <UNK> (Index 1) 為隨機分佈, <PAD> (Index 0) 維持 0
        embedding_matrix[1] = np.random.normal(scale=0.6, size=(embedding_dim, ))
        
        hits = 0
        for word, i in self.word2int.items():
            vec = embeddings_index.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
                hits += 1
        print(f"GloVe loaded. Hits: {hits}/{vocab_size}")
        return torch.FloatTensor(embedding_matrix)

    def process_all_news(self):
        print("Processing news features...")
        for idx, row in tqdm(self.news_df.iterrows(), total=len(self.news_df), desc="Feature"):
            nid_idx = self.news2int[row['news_id']]
            
            # Title
            words = self.clean_text(row['title']).split()
            word_indices = [self.word2int.get(w, 1) for w in words][:MAX_TITLE_LEN]
            word_indices += [0] * (MAX_TITLE_LEN - len(word_indices))
            self.news_title_matrix[nid_idx] = word_indices
            
            # Category
            cat = row['category'] if not pd.isna(row['category']) else 'Unknown'
            self.news_cat_matrix[nid_idx] = self.cat2int.get(cat, 1)

class BehaviorsDataset(Dataset):
    def __init__(self, behaviors_path, news_preprocessor, mode='train', num_negatives=4):
        self.preprocessor = news_preprocessor
        self.mode = mode
        self.num_negatives = num_negatives
        self.data = []
        
        print(f"Loading behaviors from {behaviors_path}...")
        # 防呆讀取
        try:
            df = pd.read_csv(behaviors_path, sep='\t', header=0, names=['idx', 'user_id', 'time', 'history', 'impressions'])
        except:
            df = pd.read_csv(behaviors_path, sep='\t', header=None, names=['idx', 'user_id', 'time', 'history', 'impressions'])
            
        news2int = self.preprocessor.news2int
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing Behaviors"):
            # History
            hist_raw = str(row['history'])
            if hist_raw == 'nan': hist_list = []
            else: hist_list = hist_raw.split()
            
            hist_indices = [news2int.get(nid, 0) for nid in hist_list]
            hist_indices = hist_indices[-MAX_HISTORY_LEN:]
            hist_indices += [0] * (MAX_HISTORY_LEN - len(hist_indices))
            
            impressions_raw = str(row['impressions']).split()
            
            if mode == 'train':
                # Negative Sampling Logic
                positives = []
                negatives = []
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
                # Test Mode
                cand_indices = [news2int.get(imp.split('-')[0], 0) for imp in impressions_raw]
                if len(cand_indices) != 15: continue
                self.data.append({'id': row['idx'], 'history': hist_indices, 'candidate': cand_indices})

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

# --- 4. 模型定義 (NRMS with Category & GloVe) ---

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, cat_size, pretrained_emb=None):
        super().__init__()
        # 1. Title Word Embedding
        self.word_embedding = nn.Embedding(vocab_size, WORD_EMB_DIM, padding_idx=0)
        if pretrained_emb is not None:
            self.word_embedding.weight.data.copy_(pretrained_emb)
            self.word_embedding.weight.requires_grad = True # Fine-tune GloVe
            
        # 2. Category Embedding
        self.cat_embedding = nn.Embedding(cat_size, CAT_EMB_DIM, padding_idx=0)
        
        # 3. Self-Attention for Title
        self.self_attn = MultiHeadSelfAttention(WORD_EMB_DIM, NEWS_HEADS)
        self.dropout = nn.Dropout(0.2)
        
        # 4. Attention Pooling for Title
        self.attn_pool = AdditiveAttention(WORD_EMB_DIM, 200)
        
        # 5. Projection to combine Title + Category
        # Input: Title(100->Attn->100) + Cat(50) = 150
        # Output: NEWS_DIM (250)
        self.final_proj = nn.Linear(WORD_EMB_DIM + CAT_EMB_DIM, TOTAL_NEWS_DIM)
        
    def forward(self, title_inputs, cat_inputs):
        # title_inputs: (B, Seq)
        # cat_inputs: (B, ) -> 需要變成 (B, 1) 或者直接 cat
        
        # --- Title Processing ---
        w_emb = self.word_embedding(title_inputs) # (B, Seq, 100)
        w_emb = self.dropout(w_emb)
        
        # Self-Attention
        title_rep = self.self_attn(w_emb) # (B, Seq, 100)
        
        # Pooling -> 變成一個向量
        title_vec = self.attn_pool(title_rep) # (B, 100)
        
        # --- Category Processing ---
        c_emb = self.cat_embedding(cat_inputs) # (B, 50)
        
        # --- Concatenation ---
        # 結合 Title 和 Category
        combined = torch.cat([title_vec, c_emb], dim=1) # (B, 150)
        
        # Final Projection -> 擴展到 300 維以供 User Encoder 使用
        news_vec = F.relu(self.final_proj(combined)) # (B, 300)
        
        return news_vec

class UserEncoder(nn.Module):
    def __init__(self, news_dim):
        super().__init__()
        # 使用 Self-Attention 捕捉瀏覽歷史中新聞的關聯
        self.self_attn = MultiHeadSelfAttention(news_dim, USER_HEADS)
        self.dropout = nn.Dropout(0.2)
        self.attn_pool = AdditiveAttention(news_dim, 200)
        
    def forward(self, history_vecs):
        # history_vecs: (B, Hist_Len, News_Dim)
        h_rep = self.self_attn(history_vecs)
        h_rep = self.dropout(h_rep)
        user_vec = self.attn_pool(h_rep) # (B, News_Dim)
        return user_vec

class AdvancedNRMS(nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        # Load cached matrices
        self.register_buffer('news_title_matrix', torch.LongTensor(preprocessor.news_title_matrix))
        self.register_buffer('news_cat_matrix', torch.LongTensor(preprocessor.news_cat_matrix))
        
        vocab_size = len(preprocessor.word2int)
        cat_size = len(preprocessor.cat2int)
        
        self.news_encoder = NewsEncoder(vocab_size, cat_size, preprocessor.glove_matrix)
        self.user_encoder = UserEncoder(news_dim=TOTAL_NEWS_DIM) # 300
        
    def forward(self, history_ids, candidate_ids):
        # 1. 取得 History 的 Title 和 Category
        hist_title = self.news_title_matrix[history_ids] # (B, Hist, Seq)
        hist_cat = self.news_cat_matrix[history_ids]     # (B, Hist)
        
        # 2. 取得 Candidate 的 Title 和 Category
        cand_title = self.news_title_matrix[candidate_ids] # (B, 1 or 15, Seq)
        cand_cat = self.news_cat_matrix[candidate_ids]     # (B, 1 or 15)
        
        # 3. Encode History News
        batch_size, hist_len, seq_len = hist_title.shape
        hist_title_flat = hist_title.view(-1, seq_len)
        hist_cat_flat = hist_cat.view(-1)
        
        hist_vecs_flat = self.news_encoder(hist_title_flat, hist_cat_flat)
        hist_vecs = hist_vecs_flat.view(batch_size, hist_len, -1) # (B, Hist, 300)
        
        # 4. Encode User
        user_vec = self.user_encoder(hist_vecs) # (B, 300)
        
        # 5. Encode Candidate News
        cand_len = cand_title.shape[1]
        cand_title_flat = cand_title.view(-1, seq_len)
        cand_cat_flat = cand_cat.view(-1)
        
        cand_vecs_flat = self.news_encoder(cand_title_flat, cand_cat_flat)
        cand_vecs = cand_vecs_flat.view(batch_size, cand_len, -1) # (B, Cand, 300)
        
        # 6. Dot Product Similarity
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(2)).squeeze(2) # (B, Cand)
        return scores

# --- 5. 訓練與預測 ---

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for hist, cand, label in pbar:
        hist, cand, label = hist.to(device), cand.to(device), label.to(device)
        
        optimizer.zero_grad()
        scores = model(hist, cand) # scores: (B, 1)
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(dataloader)

def generate_submission(model, test_loader, device, output_path='submission.csv'):
    model.eval()
    results = []
    print(f"Start inference...")
    with torch.no_grad():
        for imp_ids, hist, cand in tqdm(test_loader, desc="Predicting"):
            hist, cand = hist.to(device), cand.to(device)
            scores = model(hist, cand) # (B, 15)
            probs = torch.sigmoid(scores).cpu().numpy()
            imp_ids = imp_ids.numpy()
            
            for i in range(len(imp_ids)):
                results.append([imp_ids[i]] + probs[i].tolist())

    cols = ['id'] + [f'p{i}' for i in range(1, 16)]
    submission_df = pd.DataFrame(results, columns=cols)
    submission_df.sort_values(by='id').to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

# --- 6. 主程式 ---

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 讀取與合併 News (為了 Preprocessing)
    print("Merging Train and Test News...")
    cols = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
    df_train = pd.read_csv(PATH_TRAIN_NEWS, sep='\t', header=None, names=cols)
    df_test = pd.read_csv(PATH_TEST_NEWS, sep='\t', header=None, names=cols)
    df_all = pd.concat([df_train, df_test]).drop_duplicates(subset='news_id').reset_index(drop=True)
    
    # 2. Preprocessing
    preprocessor = NewsPreprocessor(df_all, glove_path=PATH_GLOVE)
    
    # 3. Dataset & DataLoader
    train_ds = BehaviorsDataset(PATH_TRAIN_BEHAVIORS, preprocessor, mode='train', num_negatives=NUM_NEGATIVES)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    
    # 4. Model
    model = AdvancedNRMS(preprocessor).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # 5. Training
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch Loss: {loss:.4f}")
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        
    # 6. Inference
    test_ds = BehaviorsDataset(PATH_TEST_BEHAVIORS, preprocessor, mode='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    generate_submission(model, test_loader, device, output_path='submission_advanced.csv')
    print("All Done!")