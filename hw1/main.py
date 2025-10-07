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

        # if len(not_nan) == 1:
        #     print(f"Row {i} \t, {row.Date}\t, {row.ItemName.strip()}\t has only 1 value, set interpolation:")
        #     print(data_df.iloc[i, value_columns].to_numpy(dtype=float))
    
    return data_df

def data_dropdown(data_df, value_columns, drop_type='item'):
    dropdown_items = []
    for i, row in data_df.iterrows():
        not_nan = row.iloc[value_columns].notna().to_numpy().nonzero()[0]

        # 如果 row 中的 value_columns 的非 NaN < threshold，就把該 row 刪掉
        if len(not_nan) < 2:
            dropdown_items.append({'date': row.Date, 'feature': row.ItemName})

    # for item in dropdown_items:
    #     print(f"Drop Date: {item['date']}, Feature: {item['feature']}")

    for item in dropdown_items:
        if drop_type == 'item':
            # 刪掉符合該 date 且 feature 的 row
            data_df = data_df[~((data_df.Date == item['date']) & (data_df.ItemName == item['feature']))]
        elif drop_type == 'date':
            data_df = data_df[data_df.Date != item['date']]
        elif drop_type == 'feature':
            data_df = data_df[data_df.ItemName != item['feature']]
        
    return data_df


# ========== Linear Regression Function ==========

def train_linear_regression(X, y, w=None, b=None, lr=0.01, iterations=1000, verbose=True):
    """
    使用 Gradient Descent 訓練 Linear Regression 模型
    支援繼續訓練（若 w, b 已存在）
    """
    n_samples, n_features = X.shape

    # 初始化權重
    if w is None:
        w = np.zeros(n_features)
    if b is None:
        b = 0.0

    for i in range(iterations):
        y_pred = X.dot(w) + b
        error = y_pred - y

        dw = (1 / n_samples) * X.T.dot(error)
        db = (1 / n_samples) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        if verbose and i % 100000 == 0:
            rmse = np.sqrt(np.mean(error ** 2))
            print(f"Iteration {i:4d} | RMSE = {rmse:.4f}")

    return w, b


def predict(X, w, b):
    return X.dot(w) + b


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
    """
    # 建立一個 dict，key 是 item name，value 是每小時的值
    item_data = {}
    for item in feature_items + [target_item]:
        subset = train_df[train_df["ItemName"].str.strip() == item]
        if subset.empty:
            continue
        item_data[item] = subset.iloc[0, value_columns].to_numpy(dtype=float)

    # 檢查是否所有 feature 都有資料
    if any(item not in item_data for item in feature_items + [target_item]):
        missing = [i for i in feature_items + [target_item] if i not in item_data]
        raise ValueError(f"❌ Missing items: {missing}")

    # 組成 X 和 y
    X = np.stack([item_data[f] for f in feature_items], axis=1)  # shape (24, n_features)
    y = item_data[target_item]                                   # shape (24,)

    return X, y

def unique_filename(file_path):
    base, ext = os.path.splitext(file_path)
    counter = 1
    new_file_path = file_path
    while os.path.exists(new_file_path):
        new_file_path = f"{base}_v{counter}{ext}"
        counter += 1
    return new_file_path

def main(important_features=None, name=None, iterations=1000000):
    train_df = read_csv("./data/train.csv")
    test_df = read_csv("./data/test.csv")

    # 對每個 row 中的 value_columns，
    # 如果是空值或包含 invalid 的資料 (ex: #, A, X, *) ，就改為 NaN
    value_columns = [i+3 for i in range(0, 24)]
    invalid_values = ['#', 'A', 'X', '*']
    invalid_pattern = '|'.join(map(re.escape, invalid_values))  # → "#|A|X|\*"
    
    print("✅ Data loaded.")

    # 將 train_df 做資料清理，有 invalid pattern 的地方設為 NaN
    # 全部是 NaN 的 row 就刪掉，少量的 NaN 用線性插值法補值
    train_df = data_cleaning(train_df, value_columns, invalid_pattern)
    train_df = data_interpolation(train_df, value_columns)
    train_df = data_dropdown(train_df, value_columns, drop_type='date')

    print(f"✅ Data cleaned. Remaining rows: {len(train_df)}")

    # --- 選擇 Feature 與 Target ---
    feature_items = [
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
    
    feature_items = important_features  # 使用較少的 feature 來訓練

    if name is not None:
        output_dir = f"./history_model/{name}_" + f"{'_'.join(feature_items)}"
    else:
        output_dir = "./history_model" + f"/{'_'.join(feature_items)}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    target_item = "PM2.5"

    # --- 建立資料 ---
    X, y = extract_features_and_target(train_df, feature_items, target_item, value_columns)
    print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
    print(f"check NaN in X and y: {np.isnan(X).any()}, {np.isnan(y).any()}")
    print(f"check Inf in X and y: {np.isinf(X).any()}, {np.isinf(y).any()}")

    # --- 模型路徑 ---
    model_path = "./model.json"

    # --- 載入現有模型（若有） ---
    feature_names, w, b = load_model(model_path)

    # --- 若 feature 不同或模型不存在，重新初始化 ---
    if w is None or feature_names != feature_items:
        print("🔁 No existing model found or feature mismatch. Starting new training.")
        w, b = None, None
    
    # # --- 訓練模型（可續訓） ---
    # w, b = train_linear_regression(X, y, w=w, b=b, lr=0.0001, iterations=iterations, verbose=True)

    # # --- 儲存模型 ---
    # save_model(model_path, feature_items, w, b)
    # save_model(unique_filename(f"{output_dir}/model.json"), feature_items, w, b)  # 備份
    print("✅ Model training skipped.")

    # --- 驗證 ---
    y_pred = predict(X, w, b)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"✅ Final RMSE = {rmse:.4f}")

if __name__ == "__main__":
    
    # important_features = [
    #     'PM2.5',
    #     'PM10',
    #     'NO2',
    #     'NOx',
    #     'SO2',
    #     'CH4',
    #     'CO',
    #     'RH',
    #     'AMB_TEMP',
    #     'RAINFALL',
    #     'WIND_SPEED'
    # ]
    
    # for _ in range(20):
    #     main(important_features=important_features, name="all", iterations=100000)
    
    # for _ in range(20):
    #     main(important_features=important_features, name="all_1000000", iterations=100000)
    
    # os.remove("./model.json")  # 刪掉模型，避免影響下一次的訓練
    
    important_features = [
        # 'PM2.5',
        # 'PM10',
        # 'NO2',
        # 'NOx',
        'SO2',
        'CH4',
        'CO',
        # 'RH',
        'AMB_TEMP',
        'RAINFALL',
        'WIND_SPEED'
    ]
    
    main(important_features=important_features, name="imp_10^7", iterations=10000000)