#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import hashlib
import csv
from PIL import Image
from tqdm import tqdm # 用於顯示進度條，如果沒有請 pip install tqdm

# --- Hash 輔助函式 ---

def get_file_hash(path):
    """
    計算檔案的 SHA256 雜湊值 (對應 compare_by_file_hash)
    """
    hash_sha256 = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            # 為了處理大檔案，分塊讀取
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        tqdm.write(f"[Warning] Error reading file hash for {path}: {e}")
        return None

def get_pixel_hash(path):
    """
    計算圖片像素內容的 SHA256 雜湊值 (對應 compare_by_pixel)
    """
    try:
        with Image.open(path) as img:
            # 轉換為 'RGB' 以標準化
            # 這等同於原版 compare_by_pixel 中的 .convert('RGB')
            pixel_data = img.convert('RGB').tobytes()
            
            # 額外加入圖片尺寸，確保 '1x1的黑' 和 '2x2的黑' Hash 不同
            # 這對應原版 compare_by_pixel 中的 img1.size != img2.size
            size_data = str(img.size).encode('utf-8')
            
            hash_sha256 = hashlib.sha256()
            hash_sha256.update(size_data)
            hash_sha256.update(pixel_data)
            
            return hash_sha256.hexdigest()
    except Exception as e:
        # 處理非圖片檔案或損壞的圖片
        tqdm.write(f"[Warning] Error reading pixel hash for {path}: {e}")
        return None

# --- 主要執行邏輯 ---

def main():
    """
    主函式：掃描、建立索引、比對並生成 labels.csv
    """
    # 定義路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')
    ARCHIVE_DIR = os.path.join(BASE_DIR, 'dataset', 'archive') # 根據您的腳本，這裡是 'archive'
    OUTPUT_CSV = os.path.join(BASE_DIR, 'labels.csv')
    
    IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    # --- 步驟 1: 掃描 Archieve 圖片清單 ---
    print(f"Scanning archive directory: {ARCHIVE_DIR}")
    archive_images = [] # 儲存 (full_path, label)
    
    for root, dirs, files in os.walk(ARCHIVE_DIR):
        rel_path = os.path.relpath(root, ARCHIVE_DIR)
        parts = rel_path.split(os.sep)
        
        # parts[0] 是 [item] (例如 'bottles')
        # parts[1] 是 'test'
        # parts[2] 是 [category] (例如 'goods')
        if len(parts) == 3 and parts[1] == 'test':
            
            # <--- 修改點
            # label = parts[2] # 原本的
            label = f"{parts[0]}/{parts[2]}" # 新的: "item/category"
            # <--- 修改點結束
            
            for f in files:
                if f.lower().endswith(IMAGE_EXTENSIONS):
                    full_path = os.path.join(root, f)
                    archive_images.append((full_path, label))
                    
    if not archive_images:
        print("Error: No images found in archive. Please check the directory structure.")
        return
        
    print(f"Found {len(archive_images)} images in archive.")

    # --- 步驟 2: 建立 Archive Hash 索引 ---
    # 這是加速的關鍵。
    print("Building hash indices for archive images... (This may take a while)")
    
    # 建立兩個字典 (Hash Map)
    archive_file_hash_index = {}
    archive_pixel_hash_index = {}

    for path, label in tqdm(archive_images, desc="Building archive index"):
        # 1. 檔案 Hash
        file_hash = get_file_hash(path)
        if file_hash and file_hash not in archive_file_hash_index:
            archive_file_hash_index[file_hash] = label
            
        # 2. 像素 Hash
        pixel_hash = get_pixel_hash(path)
        if pixel_hash and pixel_hash not in archive_pixel_hash_index:
            archive_pixel_hash_index[pixel_hash] = label

    print(f"Index built. Found {len(archive_file_hash_index)} unique file hashes and {len(archive_pixel_hash_index)} unique pixel hashes.")

    # --- 步驟 3: 取得 Test 圖片清單 ---
    print(f"Scanning test directory: {TEST_DIR}")
    test_images = [] # 儲存 (id, full_path)
    
    try:
        test_files = sorted(os.listdir(TEST_DIR))
    except FileNotFoundError:
        print(f"Error: Test directory not found: {TEST_DIR}")
        return

    for f in test_files:
        if f.lower().endswith(IMAGE_EXTENSIONS):
            img_id = os.path.splitext(f)[0]
            full_path = os.path.join(TEST_DIR, f)
            test_images.append((img_id, full_path))

    try:
        test_images.sort(key=lambda x: int(x[0]))
    except ValueError:
        print("Warning: Could not sort test images numerically, sorting alphabetically.")
        test_images.sort(key=lambda x: x[0])
        
    print(f"Found {len(test_images)} images in test set. Starting matching...")

    # --- 步驟 4: 執行比對 (使用索引) 並寫入 CSV ---
    
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label']) # 寫入標頭
        
        # O(N) 迴圈，不再是 O(N*M)
        for test_id, test_path in tqdm(test_images, desc="Matching images"):
            found_label = 'NOT_FOUND'
            
            # --- 策略 1: 查詢檔案 Hash (最快) ---
            test_file_hash = get_file_hash(test_path)
            if test_file_hash in archive_file_hash_index:
                found_label = archive_file_hash_index[test_file_hash]
                writer.writerow([test_id, found_label])
                continue # 找到，換下一張

            # --- 策略 2: 查詢像素 Hash (較慢，但仍比 O(N*M) 快) ---
            test_pixel_hash = get_pixel_hash(test_path)
            if test_pixel_hash in archive_pixel_hash_index:
                found_label = archive_pixel_hash_index[test_pixel_hash]
                writer.writerow([test_id, found_label])
                continue # 找到，換下一張

            # --- 策略 3: 都找不到 ---
            writer.writerow([test_id, found_label])
            tqdm.write(f"Warning: No match found for {test_path} (ID: {test_id})")

    print(f"\nDone. Results saved to {OUTPUT_CSV}")

# --- 程式進入點 ---
if __name__ == "__main__":
    try:
        from PIL import Image
        from tqdm import tqdm
    except ImportError:
        print("Error: Required libraries not found.")
        print("Please install them using:")
        print("pip install Pillow tqdm")
        exit(1)
        
    main()