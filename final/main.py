import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 設定與讀取 (Configuration)
# ==========================================
# 請確認你的資料集路徑是否正確
TRAIN_PATH = './dataset/train.csv'
TEST_PATH = './dataset/test.csv'
SUBMISSION_PATH = 'submission_ensemble.csv'

# 檢查 GPU 是否可用
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device: {DEVICE}")

def load_data():
    print("正在讀取資料...")
    # 讀取資料
    # 注意：請根據你實際的檔案內容確認哪一個是訓練集 (有 response 的那個)
    df1 = pd.read_csv(TRAIN_PATH)
    df2 = pd.read_csv(TEST_PATH)
    
    # 自動判斷哪個是 Train (有 'response' 欄位)
    if 'response' in df1.columns:
        train_df, test_df = df1, df2
    else:
        train_df, test_df = df2, df1
        
    print(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")
    
    # 標記來源
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    # Test set 若無 response 補 NaN
    if 'response' not in test_df.columns:
        test_df['response'] = np.nan
        
    # 合併以利統一特徵工程
    full_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    return full_df

# ==========================================
# 2. 特徵工程函數 (Feature Engineering)
# ==========================================

def get_bayesian_smoothed_mean(df, group_col, target_col, global_mean, C=10):
    """計算貝葉斯平滑後的平均值，防止樣本少導致極端值"""
    stats = df.groupby(group_col)[target_col].agg(['count', 'sum'])
    stats['smoothed_mean'] = (stats['sum'] + C * global_mean) / (stats['count'] + C)
    return stats['smoothed_mean']

def extract_nlp_features(full_df):
    """使用 BERT 提取文本特徵並計算相似度"""
    print("正在提取 NLP 特徵 (這需要一點時間)...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device=DEVICE)
    
    # 1. 針對唯一的題目文字做 Embedding
    unique_problems = full_df[['question_id', 'Problem']].drop_duplicates('question_id')
    unique_problems['Problem'] = unique_problems['Problem'].fillna("")
    
    # Encode
    problem_embs = model.encode(unique_problems['Problem'].tolist(), show_progress_bar=True)
    qid_to_emb = dict(zip(unique_problems['question_id'], problem_embs))
    
    # 2. PCA 降維 (從 384 -> 15 維，避免維度災難)
    pca = PCA(n_components=15, random_state=42)
    pca_embs = pca.fit_transform(problem_embs)
    qid_to_pca = dict(zip(unique_problems['question_id'], pca_embs))
    
    # 將 PCA 特徵合併回 full_df
    pca_cols = [f'pca_{i}' for i in range(15)]
    pca_df = pd.DataFrame(full_df['question_id'].map(qid_to_pca).tolist(), columns=pca_cols)
    full_df = pd.concat([full_df, pca_df], axis=1)
    
    # 3. 計算「內容相似度」(Content Similarity) - 關鍵加分特徵
    # 概念：計算「當前題目」與「該學生過去答對題目平均向量」的 Cosine Similarity
    print("正在計算內容相似度特徵...")
    
    # 找出 Train set 中學生答對的所有題目向量
    train_correct = full_df[(full_df['is_train'] == 1) & (full_df['response'] == 1)]
    user_success_centers = {}
    
    # 計算每個學生的成功中心向量 (User Success Centroid)
    for uid, group in train_correct.groupby('student_id'):
        qids = group['question_id'].unique()
        # 找出這些題目的向量
        embs = np.array([qid_to_emb[q] for q in qids if q in qid_to_emb])
        if len(embs) > 0:
            user_success_centers[uid] = np.mean(embs, axis=0)
            
    # 計算每一行的相似度
    def get_sim(row):
        uid = row['student_id']
        qid = row['question_id']
        if uid in user_success_centers and qid in qid_to_emb:
            u_vec = user_success_centers[uid].reshape(1, -1)
            q_vec = qid_to_emb[qid].reshape(1, -1)
            return cosine_similarity(u_vec, q_vec)[0][0]
        return 0.0 # 若無歷史成功紀錄或無題目向量，給 0
    
    # 使用 apply 計算 (資料量不大時可行)
    full_df['sim_to_success'] = full_df.apply(get_sim, axis=1)
    
    return full_df, pca_cols

# ==========================================
# 3. 主程式流程
# ==========================================
if __name__ == "__main__":
    # --- A. 載入資料 ---
    full_df = load_data()
    
    # --- B. 統計特徵工程 ---
    print("正在計算統計特徵...")
    # 1. 題目難度 (Bayesian Smoothed)
    # 只用 Train 計算 Global Mean 避免 Leakage
    train_only = full_df[full_df['is_train'] == 1]
    global_mean = train_only['response'].mean()
    
    # 計算 Smooth Mean
    smooth_map = get_bayesian_smoothed_mean(train_only, 'question_id', 'response', global_mean)
    full_df = full_df.merge(smooth_map.rename('q_smooth_diff'), on='question_id', how='left')
    full_df['q_smooth_diff'] = full_df['q_smooth_diff'].fillna(global_mean)
    
    # 2. 學生能力 (Student Ability)
    # 計算學生在 Train set 的歷史表現 (Mean Accuracy & Count)
    user_stats = train_only.groupby('student_id')['response'].agg(['count', 'mean']).reset_index()
    user_stats.columns = ['student_id', 'u_count', 'u_mean_acc']
    full_df = full_df.merge(user_stats, on='student_id', how='left')
    
    # Cold Start 填補
    full_df['u_mean_acc'] = full_df['u_mean_acc'].fillna(global_mean)
    full_df['u_count'] = full_df['u_count'].fillna(0)
    
    # 3. 類別特徵編碼
    le = LabelEncoder()
    full_df['category'] = full_df['category'].fillna('unknown').astype(str)
    full_df['category_code'] = le.fit_transform(full_df['category'])
    
    # --- C. NLP 特徵提取 ---
    full_df, pca_cols = extract_nlp_features(full_df)
    
    # --- D. 準備訓練 ---
    features = ['q_smooth_diff', 'u_mean_acc', 'u_count', 'sim_to_success', 'category_code'] + pca_cols
    print(f"使用特徵 ({len(features)}): {features}")
    
    train_data = full_df[full_df['is_train'] == 1].reset_index(drop=True)
    test_data = full_df[full_df['is_train'] == 0].reset_index(drop=True)
    
    X = train_data[features]
    y = train_data['response']
    X_test = test_data[features]
    
    # --- E. 定義模型 (已修正 XGBoost 參數) ---
    models = {
        'cat': CatBoostClassifier(
            iterations=2000, 
            learning_rate=0.02, 
            depth=6, 
            verbose=0, 
            random_state=42, 
            eval_metric='AUC',
            task_type='GPU' if torch.cuda.is_available() else 'CPU', # 自動切換 GPU
            early_stopping_rounds=200
        ),
        'lgb': LGBMClassifier(
            n_estimators=2000, 
            learning_rate=0.02, 
            max_depth=6, 
            random_state=42,
            verbosity=-1
        ),
        'xgb': XGBClassifier(
            n_estimators=2000, 
            learning_rate=0.02, 
            max_depth=6, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss',
            early_stopping_rounds=200 # 修正：移至初始化
        )
    }
    
    oof_preds = {k: np.zeros(len(X)) for k in models}
    test_preds = {k: np.zeros(len(X_test)) for k in models}
    
    # --- F. 訓練迴圈 (Cross Validation) ---
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n開始訓練 Ensemble 模型...")
    for fold, (train_idx, val_idx) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. CatBoost
        models['cat'].fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=['category_code'])
        oof_preds['cat'][val_idx] = models['cat'].predict_proba(X_val)[:, 1]
        test_preds['cat'] += models['cat'].predict_proba(X_test)[:, 1] / 5
        
        # 2. LightGBM (修正 Callbacks)
        models['lgb'].fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            eval_metric='auc',
            callbacks=[early_stopping(stopping_rounds=200, verbose=False), log_evaluation(0)]
        )
        oof_preds['lgb'][val_idx] = models['lgb'].predict_proba(X_val)[:, 1]
        test_preds['lgb'] += models['lgb'].predict_proba(X_test)[:, 1] / 5
        
        # 3. XGBoost (修正 fit 不帶 early_stopping_rounds)
        models['xgb'].fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)], 
            verbose=False
        )
        oof_preds['xgb'][val_idx] = models['xgb'].predict_proba(X_val)[:, 1]
        test_preds['xgb'] += models['xgb'].predict_proba(X_test)[:, 1] / 5
        
        print(f"Fold {fold+1} 完成.")

    # --- G. 評估與加權融合 (Ensemble) ---
    print("\n--- 模型效能 (CV AUROC) ---")
    for k, v in oof_preds.items():
        print(f"{k}: {roc_auc_score(y, v):.5f}")
        
    # 權重設定：CatBoost 通常最強，給較高權重 (可自行微調)
    w_cat, w_lgb, w_xgb = 0.5, 0.25, 0.25
    
    final_train_pred = (w_cat * oof_preds['cat'] + 
                        w_lgb * oof_preds['lgb'] + 
                        w_xgb * oof_preds['xgb'])
                        
    final_score = roc_auc_score(y, final_train_pred)
    print(f"\n>>> Ensemble CV AUROC: {final_score:.5f} <<<")
    
    # --- H. 產生 Submission ---
    final_test_pred = (w_cat * test_preds['cat'] + 
                       w_lgb * test_preds['lgb'] + 
                       w_xgb * test_preds['xgb'])
                       
    submission = pd.DataFrame({
        'interaction_id': test_data['interaction_id'],
        'response': final_test_pred
    })
    
    # 確保按照 interaction_id 排序 (Kaggle 有時要求)
    submission = submission.sort_values('interaction_id')
    
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\n預測結果已儲存至: {SUBMISSION_PATH}")
    print(submission.head())