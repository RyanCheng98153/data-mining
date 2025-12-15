import pandas as pd
import os

# 假設 .csv 檔案跟 .py 檔案在同一個資料夾
FILE_NAME = './labels.csv'

try:
    # 讀取 CSV
    df = pd.read_csv(FILE_NAME)
    
    if 'label' in df.columns:
        # 獲取 'label' 欄位中的獨立值 (unique values)
        # .unique() 會回傳一個 NumPy array
        unique_labels = df['label'].unique()
        
        # 轉換為標準的 Python list (陣列)
        unique_labels_list = list(unique_labels)
        
        # 打印結果
        print(f"成功讀取檔案: {FILE_NAME}")
        print(f"共找到 {len(unique_labels_list)} 個獨立的 label：")
        print(sorted(unique_labels_list))
        
    else:
        print(f"錯誤: 在 '{FILE_NAME}' 中找不到 'label' 欄位")

except FileNotFoundError:
    print(f"錯誤: 找不到檔案 '{FILE_NAME}'")
except pd.errors.EmptyDataError:
     print(f"錯誤: 檔案 '{FILE_NAME}' 是空的")
except Exception as e:
    print(f"發生預期外的錯誤: {e}")