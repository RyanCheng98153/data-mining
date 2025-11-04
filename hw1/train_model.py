#!/usr/bin/env python3
"""
train_model.py

用法示例:
  python train_model.py --train /path/to/train.csv --test /path/to/test.csv --out_model model.npz --out_pred predictions.csv

說明:
 - 只用 numpy 做 linear regression (batch gradient descent)
 - pandas 僅用於資料處理 / 檔案讀寫
"""
import argparse
import numpy as np
import pandas as pd
import json, math, os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train', default='/mnt/data/train.csv')
    p.add_argument('--test',  default='/mnt/data/test.csv')
    p.add_argument('--out_model', default='/mnt/data/model.npz')
    p.add_argument('--out_pred',  default='/mnt/data/predictions.csv')
    p.add_argument('--val_months', type=int, default=2, help='最後幾個月做為驗證集')
    p.add_argument('--window', type=int, default=9, help='用前 W 小時預測下一小時 (default 9)')
    p.add_argument('--lr', type=float, default=1.0, help='學習率 learning rate')
    p.add_argument('--lambda_l2', type=float, default=0.01, help='L2 正則化強度')
    p.add_argument('--epochs', type=int, default=2000, help='最大 epoch 數')
    p.add_argument('--patience', type=int, default=50, help='多少 epoch 沒進步就提早停止')
    p.add_argument('--min_delta', type=float, default=1e-4, help='進步幅度小於此值就不算進步')
    p.add_argument('--standardize', action='store_true', help='是否對 X 做標準化')
    p.add_argument('--features', default=None, help='逗號分隔 feature 名稱（不含空格）; 預設使用所有')
    return p.parse_args()

# ------------------ helper functions ------------------
def read_train(path):
    df = pd.read_csv(path)
    df['Location'] = df['Location'].astype(str).str.strip()
    df['ItemName'] = df['ItemName'].astype(str).str.strip()
    df['Date'] = df['Date'].astype(str).str.strip()
    def parse_month_day(s):
        try:
            left = s.split()[0]
            parts = left.split('/')
            month = int(parts[0]); day = int(parts[1])
            return month, day
        except:
            return (np.nan, np.nan)
    md = df['Date'].apply(parse_month_day)
    df['Month'] = md.apply(lambda x: x[0])
    df['Day']   = md.apply(lambda x: x[1])
    hour_cols = [str(i) for i in range(24)]
    for c in hour_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip().replace('', np.nan), errors='coerce')
    return df, hour_cols

def read_test(path):
    tdf = pd.read_csv(path, header=None)
    tdf = tdf.rename(columns={0:'id', 1:'ItemName'})
    n_extra = tdf.shape[1] - 2
    hour_cols = [str(i) for i in range(n_extra)]
    rename_map = {i+2: hour_cols[i] for i in range(n_extra)}
    tdf = tdf.rename(columns=rename_map)
    tdf['ItemName'] = tdf['ItemName'].astype(str).str.strip()
    for c in hour_cols:
        tdf[c] = pd.to_numeric(tdf[c].astype(str).str.strip().replace('', np.nan), errors='coerce')
    return tdf, hour_cols

def row_impute(arr):
    # arr: 1D numpy array (float with NaN allowed)
    vals = arr.astype(float)
    mask = np.isfinite(vals)
    cnt = mask.sum()
    if cnt == 0:
        return np.zeros_like(vals)
    elif cnt == 1:
        return np.full_like(vals, vals[mask][0], dtype=float)
    else:
        m = vals[mask].mean()
        out = vals.copy()
        out[~mask] = m
        return out

def build_month_matrices(train_df, hour_cols, feature_items):
    # build mapping (month, day, item)->24-array
    mapping = {}
    for _, row in train_df.iterrows():
        key = (int(row['Month']), int(row['Day']), row['ItemName'].strip())
        mapping[key] = row[hour_cols].values.astype(float)
    months = sorted(train_df['Month'].dropna().unique().astype(int).tolist())
    month_matrices = {}
    for m in months:
        days = sorted(train_df[train_df['Month']==m]['Day'].unique().astype(int).tolist())
        rows = len(days)*24
        mat = np.zeros((rows, len(feature_items)), dtype=float)
        r = 0
        for d in days:
            for h in range(24):
                for j, it in enumerate(feature_items):
                    key = (m, int(d), it)
                    if key in mapping:
                        mat[r,j] = mapping[key][h]
                    else:
                        mat[r,j] = 0.0
                r += 1
        month_matrices[m] = mat
    return month_matrices, months

def prepare_samples(month_matrices, months, W, feature_items, pm25_name):
    X_list, y_list, meta = [], [], []
    pm_index = feature_items.index(pm25_name)
    for m in months:
        mat = month_matrices[m]
        T = mat.shape[0]
        for t in range(W, T):
            past = mat[t-W:t, :]
            X_list.append(past.flatten(order='C'))
            y_list.append(mat[t, pm_index])
            meta.append((m, t))
    X = np.vstack(X_list)
    y = np.array(y_list).astype(float)
    return X, y, meta

def add_bias(X):
    return np.hstack([np.ones((X.shape[0],1), dtype=float), X])

def detect_pm25_name(feature_items):
    # try to find 'PM2.5' ignoring spaces/dots/case
    for it in feature_items:
        if it.strip().upper().replace('.', '').replace(' ', '') == 'PM25':
            return it
    for it in feature_items:
        if 'PM' in it.upper():
            return it
    raise ValueError("No PM2.5-like feature found")

# ------------------ training function (batch GD) ----------------------
def train_linear_gd(X_train, y_train, X_val, y_val, cfg, init_w=None):
    N, D = X_train.shape[0], X_train.shape[1]
    if init_w is None:
        w = np.zeros(D, dtype=float)
    else:
        w = init_w.copy()
    # heuristic learning rate scaling
    feat_scale = np.mean(np.sum(X_train**2, axis=1))
    lr = cfg['lr'] / max(1.0, feat_scale)
    best_val_rmse = float('inf')
    best_w = w.copy()
    wait = 0
    history = {'train_rmse': [], 'val_rmse': []}
    
    no_val = (len(y_val) == 0)  # ← 檢查是否有驗證集
    
    for epoch in range(1, cfg['epochs']+1):
        preds = X_train.dot(w)
        err = preds - y_train
        grad = (2.0/N) * (X_train.T.dot(err))
        reg_grad = np.hstack([0.0, 2.0 * cfg['lambda_l2'] * w[1:]])  # no reg on bias
        grad += reg_grad
        w -= lr * grad
        
        # 計算 train_rmse
        train_rmse = math.sqrt(np.mean((X_train.dot(w) - y_train)**2))
        history['train_rmse'].append(train_rmse)
        
        # 若沒有 validation，就跳過這步
        if no_val:
            val_rmse = float('nan')
            improved = False
            best_w = w.copy()  # 直接更新，不早停
            wait = 0  # 沒有 val 時不啟用 early stopping
        else:
            # 計算 val_rmse
            val_rmse = math.sqrt(np.mean((X_val.dot(w) - y_val)**2))
            history['val_rmse'].append(val_rmse)
            improved = (best_val_rmse - val_rmse) > cfg['min_delta']
            if improved:
                best_val_rmse = val_rmse
                best_w = w.copy()
                wait = 0
            else:
                wait += 1
        
        # 印出訓練進度
        if epoch % max(1, cfg['epochs']//20) == 0 or epoch==1:
            if no_val:
                print(f"Epoch {epoch:4d} train_rmse={train_rmse:.4f}")
            else:
                print(f"Epoch {epoch:4d} train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f} best_val={best_val_rmse:.4f} wait={wait}")
        
        # Early stopping 僅在有 val set 時生效
        if not no_val and wait >= cfg['patience']:
            print("Early stopping at epoch", epoch)
            break
    
    return best_w, history, epoch

# ------------------ main pipeline ------------------
def main(args):
    print("Loading train...")
    train_df, train_hours = read_train(args.train)
    print("Loading test...")
    test_df, test_hours = read_test(args.test)

    # per-row impute on training rows (24-hour rows)
    print("Imputing train rows...")
    for idx in train_df.index:
        arr = train_df.loc[idx, train_hours].values.astype(float)
        train_df.loc[idx, train_hours] = row_impute(arr)

    # determine features
    all_items = sorted(train_df['ItemName'].unique().astype(str).tolist())
    if args.features:
        req = [s.strip() for s in args.features.split(',')]
        feature_items = []
        for r in req:
            matched = [it for it in all_items if it.strip().lower() == r.lower()]
            if not matched:
                raise ValueError("Feature not found: " + r)
            feature_items.append(matched[0])
    else:
        feature_items = all_items

    print("Feature items count:", len(feature_items))
    pm_name = detect_pm25_name(feature_items)
    print("Detected PM2.5 column:", pm_name)

    month_mats, months = build_month_matrices(train_df, train_hours, feature_items)
    print("Built month matrices for months:", months)

    X, y, meta = prepare_samples(month_mats, months, args.window, feature_items, pm_name)
    print("Prepared samples, X shape:", X.shape, "y shape:", y.shape)

    # split by month
    if args.val_months >= len(months):
        raise ValueError("val_months too large")
    if args.val_months == 0:
        # 用全部月份訓練，不分 validation
        train_months = months
        val_months = []
        meta_months = np.array([m for (m,t) in meta])
        train_mask = np.ones(len(meta_months), dtype=bool)
        val_mask = np.zeros(len(meta_months), dtype=bool)
    else:
        train_months = months[:-args.val_months]
        val_months = months[-args.val_months:]
        meta_months = np.array([m for (m,t) in meta])
        train_mask = np.isin(meta_months, train_months)
        val_mask = np.isin(meta_months, val_months)
        
    X_train = X[train_mask]; y_train = y[train_mask]
    X_val   = X[val_mask];   y_val = y[val_mask]
    print("Train samples:", X_train.shape[0], "Val samples:", X_val.shape[0])

    # standardize
    if args.standardize:
        mean = X_train.mean(axis=0)
        std  = X_train.std(axis=0)
        std[std==0] = 1.0
        X_train_s = (X_train - mean) / std
        X_val_s   = (X_val   - mean) / std
    else:
        mean = np.zeros(X_train.shape[1])
        std  = np.ones(X_train.shape[1])
        X_train_s = X_train.copy(); X_val_s = X_val.copy()

    X_train_aug = add_bias(X_train_s)
    X_val_aug = add_bias(X_val_s)

    cfg = {
        'lr': args.lr,
        'lambda_l2': args.lambda_l2,
        'epochs': args.epochs,
        'patience': args.patience,
        'min_delta': args.min_delta
    }
    print("Start training (batch GD)...")
    w_best, history, epochs_ran = train_linear_gd(X_train_aug, y_train, X_val_aug, y_val, cfg)

    # save model
    np.savez(args.out_model, w=w_best, mean=mean, std=std, feature_items=np.array(feature_items), pm25_name=pm_name, cfg=json.dumps(vars(args)))
    print("Saved model to", args.out_model)

    # prepare test data to predict
    print("Preparing test inputs and predicting...")
    grouped = test_df.groupby('id')
    test_ids = []
    X_test_list = []
    W = args.window
    for gid, g in grouped:
        test_ids.append(gid)
        # row-impute each test row (9-length)
        item_map = {}
        for _, row in g.iterrows():
            item = row['ItemName'].strip()
            arr = row[[c for c in test_hours]].values.astype(float)
            item_map[item] = row_impute(arr)
        # build matrix W x n_features aligned with feature_items
        Tmat = np.zeros((W, len(feature_items)), dtype=float)
        for j, it in enumerate(feature_items):
            found = None
            for k in item_map.keys():
                if k.strip().lower() == it.strip().lower():
                    found = k; break
            if found is None:
                Tmat[:,j] = 0.0
            else:
                a = item_map[found]
                if len(a)!=W:
                    if len(a)<W:
                        b = np.zeros(W); b[-len(a):]=a; a=b
                    else:
                        a = a[-W:]
                Tmat[:,j] = a
        X_test_list.append(Tmat.flatten(order='C'))
    X_test = np.vstack(X_test_list)
    # standardize and add bias
    X_test_s = (X_test - mean) / std
    X_test_aug = add_bias(X_test_s)
    preds = X_test_aug.dot(w_best)
    outdf = pd.DataFrame({'index': test_ids, 'answer': preds})
    outdf.to_csv(args.out_pred, index=False)
    print("Saved predictions to", args.out_pred)
    print("Done. Train epochs run:", epochs_ran)

if __name__ == '__main__':
    args = parse_args()
    main(args)
