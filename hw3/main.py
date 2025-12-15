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

# 設定隨機種子以確保可重現性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. 資料處理與載入 (Data Processing)
# ==========================================

class NewsProcessor:
    def __init__(self, max_title_len=30, max_abstract_len=50):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.category2idx = {"<PAD>": 0}
        self.entity2idx = {"<PAD>": 0}
        self.max_title_len = max_title_len
        self.news_features = {} # NewsID -> Features
        self.entity_vectors = [] 
    
    def load_entity_embeddings(self, vec_path):
        print(f"Loading entity embeddings from {vec_path}...")
        entity_vectors = [np.zeros(100)] # padding vector (index 0)
        self.entity2idx = {"<PAD>": 0}
        
        with open(vec_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                entity_id = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                
                self.entity2idx[entity_id] = len(self.entity2idx)
                entity_vectors.append(vector)
        
        self.entity_vectors = np.array(entity_vectors)
        print(f"Loaded {len(self.entity_vectors)} entities.")

    def build_vocab(self, news_path):
        print(f"Building vocabulary from {news_path}...")
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
                
        print(f"Vocab size: {len(self.word2idx)}, Category size: {len(self.category2idx)}")

    def process_news(self, news_path, is_train=True):
        print(f"Processing news from {news_path}...")
        df = pd.read_csv(news_path, sep='\t', header=None, 
                         names=['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'])
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing News"):
            news_id = row['news_id']
            
            # Title Processing
            title_words = row['title'].lower().split() if isinstance(row['title'], str) else []
            title_indices = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in title_words][:self.max_title_len]
            title_indices += [0] * (self.max_title_len - len(title_indices))
            
            # Category
            cat_idx = self.category2idx.get(row['category'], 0)
            
            # Entity Processing (Extract first entity from title for simplicity)
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

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'category2idx': self.category2idx,
                'entity2idx': self.entity2idx,
                'entity_vectors': self.entity_vectors
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.category2idx = data['category2idx']
            self.entity2idx = data['entity2idx']
            self.entity_vectors = data['entity_vectors']

class BehaviorDataset(Dataset):
    def __init__(self, behaviors_path, news_processor, max_history=50, is_test=False):
        self.news_processor = news_processor
        self.max_history = max_history
        self.is_test = is_test
        self.samples = []
        
        print(f"Loading behaviors from {behaviors_path}...")
        df = pd.read_csv(behaviors_path, sep='\t')
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing Behaviors"):
            imp_id = row['id']
            history = str(row['clicked_news']).split() if isinstance(row['clicked_news'], str) else []
            impressions = str(row['impressions']).split() if isinstance(row['impressions'], str) else []
            
            # Process History
            hist_indices = [self.news_processor.news_features[nid] for nid in history if nid in self.news_processor.news_features]
            hist_indices = hist_indices[:self.max_history]
            if not hist_indices: # Handle empty history
                 # Create a dummy empty news feature
                dummy = {'title': torch.zeros(30, dtype=torch.long), 'category': torch.tensor(0), 'entity': torch.tensor(0)}
                hist_indices = [dummy]
            
            # Pad History logic handled in collate_fn or simple list handling
            
            if self.is_test:
                # For test, we output all candidates
                candidate_news_ids = []
                for imp in impressions:
                    # In test, format might be 'Nxxxx' or 'Nxxxx-0', handle both
                    nid = imp.split('-')[0]
                    candidate_news_ids.append(nid)
                
                self.samples.append({
                    'imp_id': imp_id,
                    'history': hist_indices,
                    'candidates': candidate_news_ids
                })
            else:
                # For train, we split into positive and negative samples or keep as list
                # Here we implement pointwise training (one sample per candidate)
                # Or list-wise. Let's do list-wise simply or flattened.
                # To align with typical News Rec, let's store list and labels
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    # Custom collate to handle variable length history and candidates
    return batch

# ==========================================
# 2. 模型架構 (Model Architecture)
# ==========================================

class NewsEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, entity_vectors, category_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Load Pretrained Entity Embeddings
        self.entity_embedding = nn.Embedding.from_pretrained(torch.tensor(entity_vectors).float(), freeze=False)
        self.entity_dim = entity_vectors.shape[1]
        
        self.category_embedding = nn.Embedding(category_size, embed_dim)
        
        # CNN for Title
        self.cnn = nn.Conv1d(embed_dim + self.entity_dim, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Projection
        self.fc = nn.Linear(256 + embed_dim, 256)

    def forward(self, title, category, entity):
        # title: (B, L), category: (B), entity: (B)
        w_emb = self.word_embedding(title) # (B, L, D)
        e_emb = self.entity_embedding(entity).unsqueeze(1).expand(-1, title.size(1), -1) # (B, L, E_D)
        
        # Combine Word + Entity
        x = torch.cat([w_emb, e_emb], dim=-1) # (B, L, D+E_D)
        x = x.permute(0, 2, 1) # (B, C, L)
        
        # CNN
        x = self.cnn(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Pooling (Max)
        title_vec, _ = torch.max(x, dim=-1) # (B, 256)
        
        # Category
        cat_vec = self.category_embedding(category)
        
        # Final News Vector
        vec = torch.cat([title_vec, cat_vec], dim=-1)
        vec = self.fc(vec)
        return vec

class UserEncoder(nn.Module):
    def __init__(self, news_dim):
        super().__init__()
        self.news_dim = news_dim
        self.attn_fc = nn.Linear(news_dim, 1)
        
    def forward(self, history_vectors):
        # history_vectors: (B, Hist_Len, News_Dim)
        attn_scores = self.attn_fc(torch.tanh(history_vectors)) # (B, H, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        user_vector = torch.sum(history_vectors * attn_weights, dim=1) # (B, News_Dim)
        return user_vector

class DualTowerModel(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.news_encoder = NewsEncoder(
            vocab_size=len(processor.word2idx),
            embed_dim=100,
            entity_vectors=processor.entity_vectors,
            category_size=len(processor.category2idx)
        )
        self.user_encoder = UserEncoder(news_dim=256)
        self.sigmoid = nn.Sigmoid()

    def forward_news(self, news_batch):
        # Helper to batch process news features
        titles = torch.stack([n['title'] for n in news_batch]).to(device)
        cats = torch.stack([n['category'] for n in news_batch]).to(device)
        ents = torch.stack([n['entity'] for n in news_batch]).to(device)
        return self.news_encoder(titles, cats, ents)

    def forward(self, history_list, candidate_list):
        # This forward is per batch of users
        # 1. Encode History
        batch_history_vecs = []
        for hist in history_list:
            h_vecs = self.forward_news(hist) # (H, D)
            batch_history_vecs.append(h_vecs)
        
        # Pad history for batch processing
        padded_hist = torch.nn.utils.rnn.pad_sequence(batch_history_vecs, batch_first=True)
        user_vecs = self.user_encoder(padded_hist) # (B, D)
        
        # 2. Encode Candidates & Score
        scores_list = []
        for i, candidates in enumerate(candidate_list):
            c_vecs = self.forward_news(candidates) # (C, D)
            u_vec = user_vecs[i].unsqueeze(0) # (1, D)
            
            # Dot Product
            scores = torch.sum(u_vec * c_vecs, dim=-1) # (C)
            scores_list.append(scores)
            
        return scores_list

# ==========================================
# 3. 訓練與推論 (Training & Inference)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(args):
    # Setup Paths
    train_news = os.path.join(args.train_data, 'train_news.tsv')
    train_behaviors = os.path.join(args.train_data, 'train_behaviors.tsv')
    train_vec = os.path.join(args.train_data, 'train_entity_embedding.vec')
    
    # Initialize Processor
    processor = NewsProcessor()
    processor.load_entity_embeddings(train_vec)
    processor.build_vocab(train_news)
    processor.process_news(train_news, is_train=True)
    
    # Save Processor (Vocab)
    os.makedirs(args.save_dir, exist_ok=True)
    processor.save(os.path.join(args.save_dir, 'processor.pkl'))
    
    # Dataset
    dataset = BehaviorDataset(train_behaviors, processor, is_test=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # Model
    model = DualTowerModel(processor).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop
    print("Start Training...")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
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
            
            loss = loss / count
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataloader):.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))

def inference(args):
    print("Start Inference...")
    # Load Processor
    processor = NewsProcessor()
    processor.load(os.path.join(args.save_dir, 'processor.pkl'))
    
    # Process Test News (Note: Unknown words handled by UNK)
    test_news = os.path.join(args.test_data, 'test_news.tsv')
    test_behaviors = os.path.join(args.test_data, 'test_behaviors.tsv')
    processor.process_news(test_news, is_train=False)
    
    # Load Model
    model = DualTowerModel(processor).to(device)
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.eval()
    
    # Dataset
    dataset = BehaviorDataset(test_behaviors, processor, is_test=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            history_list = [item['history'] for item in batch]
            imp_ids = [item['imp_id'] for item in batch]
            
            # Need to get features for candidate IDs which might be in test_news
            candidate_feats_list = []
            valid_batch_indices = []
            
            for i, item in enumerate(batch):
                c_feats = []
                for nid in item['candidates']:
                    if nid in processor.news_features:
                        c_feats.append(processor.news_features[nid])
                    else:
                        # Fallback for unknown news in test
                        c_feats.append(processor.news_features[list(processor.news_features.keys())[0]])
                candidate_feats_list.append(c_feats)
            
            scores_list = model(history_list, candidate_feats_list)
            
            for imp_id, scores in zip(imp_ids, scores_list):
                probs = torch.sigmoid(scores).cpu().numpy().tolist()
                # Ensure we output 15 probabilities
                if len(probs) < 15:
                    probs += [0.0] * (15 - len(probs))
                elif len(probs) > 15:
                    probs = probs[:15]
                
                results.append([imp_id] + probs)
    
    # Write Submission
    cols = ['id'] + [f'p{i+1}' for i in range(15)]
    sub_df = pd.DataFrame(results, columns=cols)
    sub_df.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")

# ==========================================
# 4. 主程式入口 (Main Entry)
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'inference'], help='Run mode')
    parser.add_argument('--train_data', type=str, default='./dataset/train', help='Path to train dataset')
    parser.add_argument('--test_data', type=str, default='./dataset/test', help='Path to test dataset')
    parser.add_argument('--save_dir', type=str, default='./save_models', help='Directory to save models')
    parser.add_argument('--load_model', type=str, default=None, help='Path to .pth model file for inference')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    
    set_seed()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if not args.load_model:
            raise ValueError("Please provide --load_model path for inference")
        inference(args)