import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 設定與讀取
# ==========================================
TRAIN_PATH = './dataset/train.csv'
TEST_PATH = './dataset/test.csv'
SUBMISSION_PATH = 'submission_ensemble.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 標記來源
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['response'] = np.nan

full_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

# ==========================================
# 2. 進階特徵工程 (Magic Features)
# ==========================================

# --- A. 貝葉斯平滑的題目難度 (Smoothed Target Encoding) ---
# 避免某些題目只出現 1 次，答對率就變 100% 或 0% 的極端情況
def bayesian_smoothing(df, group_col, target_col, global_mean):
    stats = df.groupby(group_col)[target_col].agg(['count', 'sum'])
    # Smoothing factor (C): 假設每題都有 C 個虛擬樣本支撐 global mean
    C = 10 
    stats['smoothed_mean'] = (stats['sum'] + C * global_mean) / (stats['count'] + C)
    return stats['smoothed_mean']

global_mean = train_df['response'].mean()
# 只用 Train 計算統計值，避免 Leakage
train_stats = train_df.groupby('question_id')['response'].agg(['count', 'sum'])
train_stats['q_smooth_diff'] = (train_stats['sum'] + 10 * global_mean) / (train_stats['count'] + 10)

# Map 回 full_df
full_df = full_df.merge(train_stats[['q_smooth_diff']], on='question_id', how='left')
full_df['q_smooth_diff'] = full_df['q_smooth_diff'].fillna(global_mean)

# --- B. 學生能力特徵 (User Ability) ---
# 計算學生在 Train set 的表現，作為 Test set 的基準能力
user_stats = train_df.groupby('student_id')['response'].agg(['count', 'mean', 'sum']).reset_index()
user_stats.columns = ['student_id', 'u_count', 'u_mean_acc', 'u_correct_sum']
full_df = full_df.merge(user_stats, on='student_id', how='left')

# 填補沒見過的學生 (Cold Start)
full_df['u_mean_acc'] = full_df['u_mean_acc'].fillna(global_mean)
full_df['u_count'] = full_df['u_count'].fillna(0)

# ==========================================
# 3. NLP 內容相似度 (Content Similarity) - 關鍵加分項
# ==========================================
print("提取文本向量與計算相似度...")
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

# 1. 取得每題的 Embedding
unique_problems = full_df[['question_id', 'Problem']].drop_duplicates('question_id')
unique_problems['Problem'] = unique_problems['Problem'].fillna("")
problem_embs = model.encode(unique_problems['Problem'].tolist(), show_progress_bar=False)
qid_to_emb = dict(zip(unique_problems['question_id'], problem_embs))

# 2. 計算每個學生「擅長領域」的中心向量 (User Success Centroid)
# 找出 Train set 中學生答對的所有題目，取 Embedding 平均
user_success_centers = {}
train_correct = train_df[train_df['response'] == 1]

# 這裡稍微花點時間，但在小數據集上跑得動
for uid, group in train_correct.groupby('student_id'):
    qids = group['question_id'].unique()
    embs = np.array([qid_to_emb[q] for q in qids if q in qid_to_emb])
    if len(embs) > 0:
        user_success_centers[uid] = np.mean(embs, axis=0)

# 3. 計算特徵：當前題目 vs 學生擅長領域的相似度
def get_similarity(row):
    uid = row['student_id']
    qid = row['question_id']
    
    if uid not in user_success_centers or qid not in qid_to_emb:
        return 0.0 # 無歷史數據或無題目向量
    
    u_center = user_success_centers[uid].reshape(1, -1)
    q_vec = qid_to_emb[qid].reshape(1, -1)
    
    return cosine_similarity(u_center, q_vec)[0][0]

# 只對 Test set 或全體計算這個特徵
# 為了速度，我們用 apply (4000筆很快)
full_df['sim_to_success'] = full_df.apply(get_similarity, axis=1)

# 另外加上 PCA 降維後的文本特徵進模型
pca = PCA(n_components=15)
pca_embs = pca.fit_transform(problem_embs)
qid_to_pca = dict(zip(unique_problems['question_id'], pca_embs))
pca_cols = [f'pca_{i}' for i in range(15)]
pca_df = pd.DataFrame(full_df['question_id'].map(qid_to_pca).tolist(), columns=pca_cols)
full_df = pd.concat([full_df, pca_df], axis=1)

# ==========================================
# 4. 模型訓練與集成 (Ensemble)
# ==========================================
features = [
    'q_smooth_diff', 'u_mean_acc', 'u_count', 'sim_to_success'
] + pca_cols

# 類別特徵處理
cat_features = ['category']
le = LabelEncoder()
full_df['category'] = full_df['category'].fillna('unknown').astype(str)
full_df['category'] = le.fit_transform(full_df['category'])
features.append('category')

train_data = full_df[full_df['is_train'] == 1].reset_index(drop=True)
test_data = full_df[full_df['is_train'] == 0].reset_index(drop=True)

X = train_data[features]
y = train_data['response']
X_test = test_data[features]

# 定義三個強模型
models = {
    'cat': CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, verbose=0, random_state=42, eval_metric='AUC'),
    'lgb': LGBMClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42),
    'xgb': XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss')
}

oof_preds = {k: np.zeros(len(X)) for k in models}
test_preds = {k: np.zeros(len(X_test)) for k in models}

# 5-Fold Cross Validation
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n開始訓練 Ensemble 模型...")

for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # 1. CatBoost
    models['cat'].fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100, cat_features=['category'])
    oof_preds['cat'][val_idx] = models['cat'].predict_proba(X_val)[:, 1]
    test_preds['cat'] += models['cat'].predict_proba(X_test)[:, 1] / 5
    
    # 2. LightGBM
    models['lgb'].fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', 
                      callbacks=[]) # 新版 LGBM callback 寫法不同，這裡簡化
    oof_preds['lgb'][val_idx] = models['lgb'].predict_proba(X_val)[:, 1]
    test_preds['lgb'] += models['lgb'].predict_proba(X_test)[:, 1] / 5
    
    # 3. XGBoost
    models['xgb'].fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    oof_preds['xgb'][val_idx] = models['xgb'].predict_proba(X_val)[:, 1]
    test_preds['xgb'] += models['xgb'].predict_proba(X_test)[:, 1] / 5

# ==========================================
# 5. 加權融合 (Weighted Blending)
# ==========================================
# 計算個別分數
scores = {k: roc_auc_score(y, v) for k, v in oof_preds.items()}
print("\n單模型 CV 分數:")
for k, v in scores.items():
    print(f"{k}: {v:.4f}")

# 簡單平均融合 (也可以根據分數調整權重，例如 CatBoost 分數高就給 0.5)
# 這裡採用 40% Cat + 30% LGB + 30% XGB
final_train_pred = 0.4 * oof_preds['cat'] + 0.3 * oof_preds['lgb'] + 0.3 * oof_preds['xgb']
final_score = roc_auc_score(y, final_train_pred)

print(f"\n>>> Ensemble CV AUROC: {final_score:.4f} <<<")

# 產生最終預測
final_test_pred = 0.4 * test_preds['cat'] + 0.3 * test_preds['lgb'] + 0.3 * test_preds['xgb']

submission = pd.DataFrame({
    'interaction_id': test_data['interaction_id'],
    'response': final_test_pred
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")