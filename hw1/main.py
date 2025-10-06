import re
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
        if len(not_nan) == len(value_columns) or len(not_nan) == 0:
            continue
        
        # 如果該 row 中只有一個 value_columns 不是 NaN，就把該 row 全部設為該 value
        # if len(not_nan) == 1:
        #     not_nan -= 1
        #     only_value = row.iloc[value_columns].iloc[not_nan[0]]
        #     data_df.iloc[i, value_columns] = only_value
        #     # print(f"Row {i} \t, {row.Date}\t, {row.ItemName.strip()}\t only has one value_column, set all value_columns to {only_value}")

        # 如果該 row 中有多個 value_columns 不是 NaN，就用線性插值法 (linear interpolation) 補值
        np_row = row.iloc[value_columns].to_numpy(dtype=float)
        x = np.arange(len(np_row))
        data_df.iloc[i, value_columns] = np.interp(x, x[not_nan], np_row[not_nan])
    
    return data_df

def data_dropdown(data_df, value_columns, drop_type='item'):
    dropdown_items = []
    for i, row in data_df.iterrows():
        not_nan = row.iloc[value_columns].notna().to_numpy().nonzero()[0]

        # 如果 row 中的 value_columns 的非 NaN < threshold，就把該 row 刪掉
        if len(not_nan) < 2:
            dropdown_items.append({'date': row.Date, 'feature': row.ItemName})

    for item in dropdown_items:
        if drop_type == 'item':
            data_df = data_df[data_df.Date != item['date'] and data_df.ItemName != item['feature']]
        elif drop_type == 'date':
            data_df = data_df[data_df.Date != item['date']]
        elif drop_type == 'feature':
            data_df = data_df[data_df.ItemName != item['feature']]
        
    return data_df

def main():
    train_df = read_csv("./data/train.csv")
    test_df = read_csv("./data/test.csv")

    # 對每個 row 中的 value_columns，
    # 如果是空值或包含 invalid 的資料 (ex: #, A, X, *) ，就改為 NaN
    value_columns = [i+3 for i in range(0, 24)]
    invalid_values = ['#', 'A', 'X', '*']
    invalid_pattern = '|'.join(map(re.escape, invalid_values))  # → "#|A|X|\*"

    # 將 train_df 做資料清理，有 invalid pattern 的地方設為 NaN
    # 全部是 NaN 的 row 就刪掉，少量的 NaN 用線性插值法補值
    train_df = data_cleaning(train_df, value_columns, invalid_pattern)
    train_df = data_interpolation(train_df, value_columns)
    train_df = data_dropdown(train_df, value_columns, drop_type='item')

if __name__ == "__main__":
    main()
