import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_csv(file_path):
    """Reads a CSV file and returns a pandas DataFrame."""
    return pd.read_csv(file_path)

def main():
    train_df = read_csv("./data/train.csv")
    test_df = read_csv("./data/test.csv")

    # 對每個 row 中的 value_columns，
    # 如果是空值或包含 invalid 的資料 (ex: #, A, X, *) ，就改為 NaN
    value_columns = [i+3 for i in range(0, 24)]
    invalid_values = ['#', 'A', 'X', '*']
    invalid_pattern = '|'.join(map(re.escape, invalid_values))  # → "#|A|X|\*"
    
    # 對每個欄位進行檢查與轉換
    for col in value_columns:
        # 把包含 invalid 的地方設為 NaN
        train_df.iloc[:, col] = train_df.iloc[:, col].astype(str)  # 確保是字串
        train_df.iloc[:, col] = train_df.iloc[:, col].mask(
            train_df.iloc[:, col].str.contains(invalid_pattern, case=False, na=False),
            np.nan
        )
    
    for i, row in train_df.iterrows():
        # 找出該 row 中不是 NaN 的 value_columns 的 index
        not_nan = row.iloc[value_columns].notna().to_numpy().nonzero()[0]
        
        if len(not_nan) == len(value_columns):
            # No need to fill, all values are present
            continue
        
        # 如果該 row 中只有一個 value_columns 不是 NaN，就把該 row 全部設為該 value
        if len(not_nan) == 1:
            train_df.iloc[i, value_columns] = np.nan  # 先設為 NaN
            # only_value = row.iloc[value_columns].iloc[not_nan[0]]
            # train_df.iloc[i, value_columns] = only_value
            # print(f"Row {i} \t, {row.Date}\t, {row.ItemName.strip()}\t only has one value_column, set all value_columns to {only_value}")
            continue
        
        # 如果 row 中的 value_columns 全部都是 NaN，就把該 row 刪掉
        if len(not_nan) == 0:
            train_df.drop(i, inplace=True)
            # print(f"Drop row {i}\t, {row.Date}\t, {row.ItemName.strip()}\t because all value_columns are NaN")
            continue
        
        # 如果該 row 中有多個 value_columns 不是 NaN，就用線性插值法 (linear interpolation) 補值
        np_row = row.iloc[value_columns].to_numpy(dtype=float)
        x = np.arange(len(np_row))
        train_df.iloc[i, value_columns] = np.interp(x, x[not_nan], np_row[not_nan])
    
        
if __name__ == "__main__":
    main()
