import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import pickle
import sys
import datetime
import time

# ==========================================
# 工具函數 (Utils)
# ==========================================

def log(msg):
    """格式化輸出日誌，帶上時間戳"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def set_seed(seed=42):
    log(f"Setting random seed to {seed}...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_params(model):
    """計算模型參數量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==========================================
# 1. 資料處理與載入 (Data Processing)
# ==========================================

class NewsProcessor:
    def __init__(self, max_title_len=30):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.category2idx = {"<PAD>": 0}
        self.entity2idx = {"<PAD>": 0}
        self.max_title_len = max_title_len
        self.news_features = {} # NewsID -> Features
        self.entity_vectors = [] 
    
    def load_entity_embeddings(self, vec_path):
        log(f"Loading entity embeddings from {vec_path}...")
        start_time = time.time()
        entity_vectors = [np.zeros(100)] # padding vector (index 0)
        self.entity2idx = {"<PAD>": 0}
        
        if not os.path.exists(vec_path):
            log(f"Warning: Entity file {vec_path} not found. Using random initialization.")
            self.entity_vectors = np.random.rand(1, 100)
            return

        with open(vec_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                entity_id = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                self.entity2idx[entity_id] = len(self.entity2idx)
                entity_vectors.append(vector)
        
        self.entity_vectors = np.array(entity_vectors)
        elapsed = time.time() - start_time
        log(f"Loaded {len(self.entity_vectors)} entities (Shape: {self.entity_vectors.shape}) in {elapsed:.2f}s.")

    def build_vocab(self, news_path):
        log(f"Building vocabulary from {news_path}...")
        df = pd.read_csv(news_path, sep='\t', header=None, 
                         names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
        
        for title in df['title'].fillna(""):
            words = title.lower().split()
            for w in words:
                if w not in self.word2idx:
                    self.word2idx[w] = len(self.word2idx)
        
        for cat in df['category'].unique():
            if cat not in self.category2idx:
                self.category2idx[cat] = len(self.category2idx)
                
        log(f"Vocabulary built. Words: {len(self.word2idx)}, Categories: {len(self.category2idx)}")

    def process_news(self, news_path, is_train=True):
        mode_str = "Train" if is_train else "Test"
        log(f"Processing {mode_str} news from {news_path}...")
        
        df = pd.read_csv(news_path, sep='\t', header=None, 
                         names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
        
        success_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Parsing {mode_str} News"):
            news_id = row['news_id']
            
            # Title Processing
            title_words = row['title'].lower().split() if isinstance(row['title'], str) else []
            title_indices = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in title_words][:self.max_title_len]
            title_indices += [0] * (self.max_title_len - len(title_indices))
            
            # Category
            cat_idx = self.category2idx.get(row['category'], 0)
            
            # Entity Processing
            entity_idx = 0
            if isinstance(row['title_entities'], str):
                try:
                    ents = json.loads(row['title_entities'])
                    if ents:
                        wiki_id = ents[0].get('WikidataId')
                        entity_idx = self.entity2idx.get(wiki_id, 0)
                except:
                    pass
            
            self.news_features[news_id] = {
                'title': torch.tensor(title_indices, dtype=torch.long),
                'category': torch.tensor(cat_idx, dtype=torch.long),
                'entity': torch.tensor(entity_idx, dtype=torch.long)
            }
            success_count += 1
            
        log(f"Successfully processed {success_count} news items.")

    def save(self, path):
        log(f"Saving processor state to {path}...")
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'category2idx': self.category2idx,
                'entity2idx': self.entity2idx,
                'entity_vectors': self.entity_vectors
            }, f)

    def load(self, path):
        log(f"Loading processor state from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.category2idx = data['category2idx']
            self.entity2idx = data['entity2idx']
            self.entity_vectors = data['entity_vectors']
        log("Processor state loaded.")

class BehaviorDataset(Dataset):
    def __init__(self, behaviors_path, news_processor, max_history=50, is_test=False):
        self.news_processor = news_processor
        self.max_history = max_history
        self.is_test = is_test
        self.samples = []
        
        log(f"Loading behaviors from {behaviors_path}...")
        df = pd.read_csv(behaviors_path, sep='\t')
        
        # 統計資訊
        missing_news_count = 0
        empty_history_count = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing Behaviors"):
            imp_id = row['id']
            history = str(row['clicked_news']).split() if isinstance(row['clicked_news'], str) else []
            impressions = str(row['impressions']).split() if isinstance(row['impressions'], str) else []
            
            # Process History
            hist_indices = []
            for nid in history:
                if nid in self.news_processor.news_features:
                    hist_indices.append(self.news_processor.news_features[nid])
                else:
                    missing_news_count += 1
            
            hist_indices = hist_indices[:self.max_history]
            
            if not hist_indices:
                empty_history_count += 1
                dummy = {'title': torch.zeros(30, dtype=torch.long), 'category': torch.tensor(0), 'entity': torch.tensor(0)}
                hist_indices = [dummy]
            
            if self.is_test:
                candidate_news_ids = [imp.split('-')[0] for imp in impressions]
                self.samples.append({
                    'imp_id': imp_id,
                    'history': hist_indices,
                    'candidates': candidate_news_ids
                })
            else:
                candidate_features = []
                labels = []
                for imp in impressions:
                    parts = imp.split('-')
                    nid = parts[0]
                    label = int(parts[1])
                    if nid in self.news_processor.news_features:
                        candidate_features.append(self.news_processor.news_features[nid])
                        labels.append(label)
                
                if candidate_features:
                    self.samples.append({
                        'imp_id': imp_id,
                        'history': hist_indices,
                        'candidates': candidate_features,
                        'labels': labels
                    })
        
        log(f"Dataset created. Samples: {len(self.samples)}")
        log(f"Stats: Empty History Users: {empty_history_count}, Missing History News: {missing_news_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    return batch

# ==========================================
# 2. 模型架構 (Model Architecture)
# ==========================================

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, entity_vectors, category_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if len(entity_vectors) > 0:
            self.entity_embedding = nn.Embedding.from_pretrained(torch.tensor(entity_vectors).float(), freeze=False)
            self.entity_dim = entity_vectors.shape[1]
        else:
            self.entity_embedding = nn.Embedding(1, 100) # Dummy
            self.entity_dim = 100

        self.category_embedding = nn.Embedding(category_size, embed_dim)
        
        self.cnn = nn.Conv1d(embed_dim + self.entity_dim, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(256 + embed_dim, 256)

    def forward(self, title, category, entity):
        w_emb = self.word_embedding(title) 
        e_emb = self.entity_embedding(entity).unsqueeze(1).expand(-1, title.size(1), -1)
        x = torch.cat([w_emb, e_emb], dim=-1).permute(0, 2, 1)
        x = self.dropout(self.relu(self.cnn(x)))
        title_vec, _ = torch.max(x, dim=-1)
        cat_vec = self.category_embedding(category)
        vec = self.fc(torch.cat([title_vec, cat_vec], dim=-1))
        return vec

class UserEncoder(nn.Module):
    def __init__(self, news_dim):
        super().__init__()
        self.news_dim = news_dim
        self.attn_fc = nn.Linear(news_dim, 1)
        
    def forward(self, history_vectors):
        attn_scores = self.attn_fc(torch.tanh(history_vectors))
        attn_weights = torch.softmax(attn_scores, dim=1)
        user_vector = torch.sum(history_vectors * attn_weights, dim=1)
        return user_vector

class DualTowerModel(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.news_encoder = NewsEncoder(
            vocab_size=len(processor.word2idx),
            embed_dim=100,
            entity_vectors=processor.entity_vectors,
            category_size=len(processor.category2idx)
        )
        self.user_encoder = UserEncoder(news_dim=256)

    def forward_news(self, news_batch):
        titles = torch.stack([n['title'] for n in news_batch]).to(device)
        cats = torch.stack([n['category'] for n in news_batch]).to(device)
        ents = torch.stack([n['entity'] for n in news_batch]).to(device)
        return self.news_encoder(titles, cats, ents)

    def forward(self, history_list, candidate_list):
        # Batch encode history
        batch_history_vecs = [self.forward_news(hist) for hist in history_list]
        padded_hist = torch.nn.utils.rnn.pad_sequence(batch_history_vecs, batch_first=True)
        user_vecs = self.user_encoder(padded_hist)
        
        scores_list = []
        for i, candidates in enumerate(candidate_list):
            c_vecs = self.forward_news(candidates)
            u_vec = user_vecs[i].unsqueeze(0)
            scores = torch.sum(u_vec * c_vecs, dim=-1)
            scores_list.append(scores)
        return scores_list

# ==========================================
# 3. 訓練與推論 (Training & Inference)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    log("=== Initializing Training Pipeline ===")
    log(f"Device: {device}")
    
    train_news = os.path.join(args.train_data, 'train_news.tsv')
    train_behaviors = os.path.join(args.train_data, 'train_behaviors.tsv')
    train_vec = os.path.join(args.train_data, 'train_entity_embedding.vec')
    
    processor = NewsProcessor()
    processor.load_entity_embeddings(train_vec)
    processor.build_vocab(train_news)
    processor.process_news(train_news, is_train=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    processor.save(os.path.join(args.save_dir, 'processor.pkl'))
    
    dataset = BehaviorDataset(train_behaviors, processor, is_test=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = DualTowerModel(processor).to(device)
    log(f"Model Created. Total Parameters: {get_model_params(model):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    log("=== Start Training Loop ===")
    model.train()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            history_list = [item['history'] for item in batch]
            candidate_list = [item['candidates'] for item in batch]
            labels_list = [torch.tensor(item['labels']).float().to(device) for item in batch]
            
            preds_list = model(history_list, candidate_list)
            
            loss = 0
            count = 0
            for preds, labels in zip(preds_list, labels_list):
                loss += criterion(preds, labels)
                count += 1
            
            loss = loss / (count + 1e-9)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({'loss': total_loss/batch_count})
        
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        log(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}. Time: {epoch_time:.1f}s")
        
        save_path = os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        log(f"Checkpoint saved: {save_path}")

def inference(args):
    log("=== Initializing Inference Pipeline ===")
    log(f"Device: {device}")
    
    processor = NewsProcessor()
    processor.load(os.path.join(args.save_dir, 'processor.pkl'))
    
    test_news = os.path.join(args.test_data, 'test_news.tsv')
    test_behaviors = os.path.join(args.test_data, 'test_behaviors.tsv')
    
    # 注意：測試集可能有新新聞
    processor.process_news(test_news, is_train=False)
    
    model = DualTowerModel(processor).to(device)
    log(f"Loading weights from {args.load_model}...")
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.eval()
    
    dataset = BehaviorDataset(test_behaviors, processor, is_test=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    results = []
    unknown_news_encountered = 0
    
    log("=== Start Prediction ===")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            history_list = [item['history'] for item in batch]
            imp_ids = [item['imp_id'] for item in batch]
            
            candidate_feats_list = []
            
            for item in batch:
                c_feats = []
                for nid in item['candidates']:
                    if nid in processor.news_features:
                        c_feats.append(processor.news_features[nid])
                    else:
                        # Fallback
                        unknown_news_encountered += 1
                        c_feats.append(processor.news_features[list(processor.news_features.keys())[0]])
                candidate_feats_list.append(c_feats)
            
            scores_list = model(history_list, candidate_feats_list)
            
            for imp_id, scores in zip(imp_ids, scores_list):
                probs = torch.sigmoid(scores).cpu().numpy().tolist()
                # Ensure 15 probabilities
                if len(probs) < 15:
                    probs += [0.0] * (15 - len(probs))
                elif len(probs) > 15:
                    probs = probs[:15]
                
                results.append([imp_id] + probs)
    
    log(f"Inference Done. Unknown News Encounters: {unknown_news_encountered}")
    
    cols = ['id'] + [f'p{i+1}' for i in range(15)]
    sub_df = pd.DataFrame(results, columns=cols)
    out_file = unique_filename('submission.csv')
    sub_df.to_csv(out_file, index=False)
    log(f"Submission saved to {out_file} (Rows: {len(sub_df)})")

def unique_filename(base_name):
    """生成唯一的檔案名稱以避免覆蓋"""
    counter = 1
    new_name = base_name
    while os.path.exists(new_name):
        new_name = f"{os.path.splitext(base_name)[0]}_{counter}{os.path.splitext(base_name)[1]}"
        counter += 1
    return new_name

# ==========================================
# 4. 主程式入口 (Main Entry)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'])
    parser.add_argument('--train_data', type=str, default='./dataset/train')
    parser.add_argument('--test_data', type=str, default='./dataset/test')
    parser.add_argument('--save_dir', type=str, default='./save_models')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    set_seed()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if not args.load_model:
            print("Error: --load_model is required for inference.")
        else:
            inference(args)