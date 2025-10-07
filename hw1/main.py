import re, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)

def data_cleaning(data_df, value_columns, invalid_pattern):
    # 對每個欄位進行檢查與轉換
    for col in value_columns:
        # 把包含 invalid 的地方設為 NaN
        data_df.iloc[:, col] = data_df.iloc[:, col].astype(str)  # 確保是字串
        data_df.iloc[:, col] = data_df.iloc[:, col].mask(
            data_df.iloc[:, col].str.contains(invalid_pattern, case=False, na=False),
            np.nan
        )
        
        # 再把無法轉成 float 的值設為 NaN
        data_df.iloc[:, col] = pd.to_numeric(data_df.iloc[:, col], errors='coerce')
    
    return data_df

# 資料內插
def data_interpolation(data_df, value_columns):
    for i, row in data_df.iterrows():
        # 找出該 row 中不是 NaN 的 value_columns 的 index
        not_nan = row.iloc[value_columns].notna().to_numpy().nonzero()[0]
        # 如果想不做 interpolation 的話，需要把這些 row 跳過
        if len(not_nan) == len(value_columns) or len(not_nan) == 0:
            continue
        
        # 如果該 row 中有多個 value_columns 不是 NaN，就用線性插值法 (linear interpolation) 補值
        np_row = row.iloc[value_columns].to_numpy(dtype=float)
        x = np.arange(len(np_row))
        data_df.iloc[i, value_columns] = np.interp(x, x[not_nan], np_row[not_nan])
    
    return data_df

def data_dropdown(data_df, value_columns, drop_type='item', skip_for_test=False):
    """
    只保留 PM2.5 相關的檢查，不刪除其他 feature 的數據
    skip_for_test=True 時，完全不刪除任何數據（用於測試集）
    """
    if skip_for_test:
        return data_df
    
    dropdown_items = []
    for i, row in data_df.iterrows():
        not_nan = row.iloc[value_columns].notna().to_numpy().nonzero()[0]

        # 只對 PM2.5 (target) 做嚴格檢查
        if row.ItemName.strip() == 'PM2.5' and len(not_nan) < 2:
            dropdown_items.append({'date': row.Date, 'feature': row.ItemName})

    for item in dropdown_items:
        if drop_type == 'item':
            data_df = data_df[~((data_df.Date == item['date']) & (data_df.ItemName == item['feature']))]
        elif drop_type == 'date':
            data_df = data_df[data_df.Date != item['date']]
        elif drop_type == 'feature':
            data_df = data_df[data_df.ItemName != item['feature']]
        
    return data_df


# ========== Linear Regression with NaN handling ==========

def train_linear_regression(X, y, w=None, b=None, lr=0.01, iterations=1000, verbose=True):
    """
    使用 Gradient Descent 訓練 Linear Regression 模型
    支援 NaN 值處理：在計算時動態忽略 NaN 的 feature
    """
    n_samples, n_features = X.shape

    # 初始化權重
    if w is None:
        w = np.zeros(n_features)
    if b is None:
        b = 0.0

    for iteration in range(iterations):
        # 對每個樣本分別計算預測值（因為每個樣本的有效 feature 可能不同）
        y_pred = np.zeros(n_samples)
        
        for i in range(n_samples):
            valid_mask = ~np.isnan(X[i])
            if valid_mask.any():
                y_pred[i] = np.dot(X[i][valid_mask], w[valid_mask]) + b
            else:
                y_pred[i] = b  # 如果所有 feature 都是 NaN，只用 bias
        
        error = y_pred - y
        
        # 計算梯度（對每個 feature 分別計算，只用有效的樣本）
        dw = np.zeros(n_features)
        for j in range(n_features):
            valid_samples = ~np.isnan(X[:, j])
            if valid_samples.any():
                dw[j] = np.mean(error[valid_samples] * X[valid_samples, j])
        
        db = np.mean(error)
        
        w -= lr * dw
        b -= lr * db

        if verbose and iteration % 10000 == 0:
            rmse = np.sqrt(np.mean(error ** 2))
            print(f"Iteration {iteration:4d} | RMSE = {rmse:.4f}")

    return w, b


def predict(X, w, b):
    """預測函數，支援 NaN 值"""
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples)
    
    for i in range(n_samples):
        valid_mask = ~np.isnan(X[i])
        if valid_mask.any():
            predictions[i] = np.dot(X[i][valid_mask], w[valid_mask]) + b
        else:
            predictions[i] = b  # 如果所有 feature 都是 NaN，只用 bias
    
    return predictions


def save_model(model_path, feature_names, w, b):
    model_data = {
        "feature_names": feature_names,
        "weights": w.tolist(),
        "bias": float(b)
    }
    with open(model_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, ensure_ascii=False, indent=4)
    print(f"✅ Model saved to {model_path}")


def load_model(model_path):
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    feature_names = model_data["feature_names"]
    w = np.array(model_data["weights"], dtype=float)
    b = float(model_data["bias"])
    print(f"✅ Model loaded from {model_path}")
    return feature_names, w, b


# ========== Data Preparation Helper ==========
def extract_features_and_target(train_df, feature_items, target_item, value_columns):
    """
    將 train_df 中特定 ItemName 的數據取出，組成 X (features) 與 y (target)
    允許 feature 為 NaN
    """
    # 建立一個 dict，key 是 item name，value 是每小時的值
    item_data = {}
    for item in feature_items + [target_item]:
        subset = train_df[train_df["ItemName"].str.strip() == item]
        if subset.empty:
            # 如果該 feature 不存在，用全 NaN 填充
            print(f"⚠️ Feature '{item}' not found, using NaN values")
            item_data[item] = np.full(len(value_columns), np.nan)
        else:
            item_data[item] = subset.iloc[0, value_columns].to_numpy(dtype=float)

    # 組成 X 和 y
    X = np.stack([item_data[f] for f in feature_items], axis=1)  # shape (24, n_features)
    y = item_data[target_item]                                   # shape (24,)

    return X, y

# ========== Testing Function ==========

def predict_next_hour_pm25(test_path, model_path="./model.json", cur_hour=9):
    next_hour = cur_hour + 1
    """
    使用訓練好的模型預測每筆資料的第next_hour小時 PM2.5。
    支援 feature 為 NaN 的情況，並預測所有日期的 PM2.5
    """
    # === 讀取 test.csv，並手動命名欄位 ===
    columns = ["Date", "ItemName"] + [str(i) for i in range(1, 11)]
    test_df = pd.read_csv(test_path, header=None, names=columns)
    
    test_df = data_cleaning(test_df, value_columns=[i+2 for i in range(cur_hour)], invalid_pattern='#|A|X|\*')
    test_df = data_interpolation(test_df, value_columns=[i+2 for i in range(cur_hour)])
    # 不刪除任何數據，保留所有日期
    test_df = data_dropdown(test_df, value_columns=[i+2 for i in range(cur_hour)], drop_type='date', skip_for_test=True)
    print("✅ Test data loaded and cleaned.")

    # === 載入模型 ===
    feature_names, w, b = load_model(model_path)
    if w is None or feature_names is None:
        raise ValueError("❌ Model not found or feature mismatch with trained model.")

    value_columns = [i+2 for i in range(cur_hour)]  # 第1~cur_hour小時欄位 (index=2~cur_hour+1)
    results = []
    y_true_list = []
    y_pred_list = []

    # 取得所有唯一的日期
    all_dates = test_df["Date"].unique()
    print(f"📊 Total dates to predict: {len(all_dates)}")

    for date in all_dates:
        subset = test_df[test_df["Date"] == date]

        # 收集該時段的 feature 資料（允許 NaN）
        feature_data = {}
        try:    
            for item in feature_names:
                row = subset[subset["ItemName"].str.strip() == item]
                if row.empty:
                    # 如果該 feature 不存在，使用 NaN
                    feature_data[item] = np.full(len(value_columns), np.nan)
                else:
                    feature_data[item] = row.iloc[0, value_columns].to_numpy(dtype=float)
        except Exception as e:
            print(f"⚠️ Error processing date {date}: {e}")
            continue

        # 取第cur_hour小時 feature 值作為輸入（允許 NaN）
        X_input = np.array([feature_data[f][-1] for f in feature_names]).reshape(1, -1)

        # 預測第next_hour小時 PM2.5
        y_pred_next_hour = predict(X_input, w, b)[0]
        results.append({"index": date, "answer": y_pred_next_hour})
        y_pred_list.append(y_pred_next_hour)

        # 若有 PM2.5 真實第next_hour小時數據，則計算 RMSE
        pm25_row = subset[subset["ItemName"].str.strip() == "PM2.5"]
        if not pm25_row.empty and str(next_hour) in pm25_row.columns:
            true_val = pm25_row.iloc[0][str(next_hour)]
            if not pd.isna(true_val):
                y_true_list.append(float(true_val))

    result_df = pd.DataFrame(results)
    print(f"✅ Prediction complete. Predicted {len(result_df)} dates.")

    # === RMSE 計算 ===
    if len(y_true_list) == len(y_pred_list) and len(y_true_list) > 0:
        rmse = np.sqrt(np.mean((np.array(y_true_list) - np.array(y_pred_list)) ** 2))
        print(f"✅ Prediction RMSE = {rmse:.4f}")
    else:
        rmse = None
        print("⚠️ RMSE cannot be calculated (missing true PM2.5 values).")

    return result_df, rmse

def unique_filename(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base}_v{counter}{ext}"
        counter += 1
    return new_file_path

def main_training(important_features=None, name=None, iterations=1000000, model_path=None):
    train_df = read_csv("./data/train.csv")
    
    # 對每個 row 中的 value_columns，
    # 如果是空值或包含 invalid 的資料 (ex: #, A, X, *) ，就改為 NaN
    value_columns = [i+3 for i in range(0, 24)]
    invalid_values = ['#', 'A', 'X', '*']
    invalid_pattern = '|'.join(map(re.escape, invalid_values))
    
    print("✅ Data loaded.")

    # 將 train_df 做資料清理
    train_df = data_cleaning(train_df, value_columns, invalid_pattern)
    train_df = data_interpolation(train_df, value_columns)
    train_df = data_dropdown(train_df, value_columns, drop_type='date')

    print(f"✅ Data cleaned. Remaining rows: {len(train_df)}")

    # --- 選擇 Feature 與 Target ---
    if important_features is None:
        important_features = [
            'AMB_TEMP',
            'CH4',
            'CO',
            'NMHC',
            'NO',
            'NO2',
            'NOx',
            'O3',
            'PM10',
            'PM2.5',
            'RAINFALL',
            'RH',
            'SO2',
            'THC',
            'WD_HR',
            'WIND_DIREC',
            'WIND_SPEED',
            'WS_HR'
        ]
    
    feature_items = important_features

    if name is not None:
        output_dir = f"./history_model/{name}"
    else:
        output_dir = "./history_model" + f"/{'_'.join(feature_items)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_item = "PM2.5"

    # --- 建立資料 ---
    X, y = extract_features_and_target(train_df, feature_items, target_item, value_columns)
    print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
    print(f"NaN count in X: {np.isnan(X).sum()} out of {X.size} values ({np.isnan(X).sum()/X.size*100:.2f}%)")
    print(f"NaN count in y: {np.isnan(y).sum()} out of {y.size} values")

    # --- 模型路徑 ---
    if model_path is None:
        model_path = "./model.json"

    # --- 載入現有模型（若有） ---
    feature_names, w, b = load_model(model_path)

    # --- 若 feature 不同或模型不存在，重新初始化 ---
    if w is None or feature_names != feature_items:
        print("🔁 No existing model found or feature mismatch. Starting new training.")
        w, b = None, None
    
    # --- 訓練模型（可續訓） ---
    w, b = train_linear_regression(X, y, w=w, b=b, lr=0.0001, iterations=iterations, verbose=True)

    # --- 儲存模型 ---
    save_model(model_path, feature_items, w, b)
    save_model(unique_filename(f"{output_dir}/model.json"), feature_items, w, b)

    # --- 驗證 ---
    y_pred = predict(X, w, b)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"✅ Final RMSE = {rmse:.4f}")

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Path to the model file (JSON)')

    args = parser.parse_args()
    model_path = args.model if args.model is not None else "./model.json"

    important_features = [
        'SO2',
        'CH4',
        'CO',
        'AMB_TEMP',
        'RAINFALL',
        'WIND_SPEED'
    ]
    
    # important_features = [
    #     "PM2.5",
    #     "PM10",
    #     "NO2",
    #     "NOx",
    #     "SO2",
    #     "CH4",
    #     "CO",
    #     "RH",
    #     "AMB_TEMP",
    #     "RAINFALL",
    #     "WIND_SPEED"
    # ]
    
    for i in range(1):    
        # Train the model
        # main_training(important_features=important_features, name=None, iterations=10000, model_path=model_path)

        # --- predict test data ---
        test_path = "./data/test.csv"
        
        # 使用前 9 小時的資料來預測第 10 小時的 PM2.5
        cur_hour = 9
        pred_df, rmse = predict_next_hour_pm25(
            test_path, 
            model_path=model_path, 
            cur_hour=cur_hour
        )
        
        clean_model_path = os.path.splitext(os.path.basename(model_path))[0]
        clean_model_path = f"new_10000_v{i}"

        if rmse is not None:
            rmse = f"{rmse:.4f}"
    
        print(pred_df.head())
        pred_df.to_csv(f"./prediction_{cur_hour + 1}_hour_pm25_{clean_model_path}_rmse_{rmse}.csv", index=False)
        print(f"✅ Saved prediction to prediction_{cur_hour + 1}_hour_pm25_{clean_model_path}_rmse_{rmse}.csv")