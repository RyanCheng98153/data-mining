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
    
    # 如果 row 中的 value_columns 全部都是 NaN，就把該 row 刪掉
    for i, row in train_df.iterrows():
        if row.iloc[value_columns].isna().all():
            train_df.drop(i, inplace=True)
            print(f"Drop row {i} because all value_columns are NaN")
    
if __name__ == "__main__":
    main()
