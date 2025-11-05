import sys
import os
import hashlib
from PIL import Image, ImageChops

def compare_by_file_hash(path1, path2):
    """
    方法 1: 檔案雜湊 (SHA256)
    檢查兩個檔案是否在位元組層級完全相同。
    速度快，但最嚴格。只要中繼資料 (metadata) 或壓縮方式稍有不同，就會被判斷為不同。
    """
    hash1 = hashlib.sha256()
    hash2 = hashlib.sha256()

    try:
        with open(path1, 'rb') as f:
            hash1.update(f.read())
        
        with open(path2, 'rb') as f:
            hash2.update(f.read())
            
        return hash1.hexdigest() == hash2.hexdigest()
        
    except Exception as e:
        print(f"  [File Hash Error]: {e}")
        return False

def compare_by_pixel(path1, path2):
    """
    方法 2: 像素比對 (Pixel-by-Pixel)
    使用 Pillow (PIL) 檢查兩張圖片的尺寸、模式和所有像素值是否完全相同。
    這適用於檢查一張圖是否被儲存了兩次（例如，一個是 PNG，一個是 BMP，但內容完全一致）。
    """
    try:
        with Image.open(path1) as img1, Image.open(path2) as img2:
            # 檢查尺寸和模式是否相同
            if img1.size != img2.size or img1.mode != img2.mode:
                return False
            
            # 使用 ImageChops.difference 找出差異
            # 如果沒有差異，diff.getbbox() 會回傳 None
            diff = ImageChops.difference(img1.convert('RGB'), img2.convert('RGB'))
            
            if diff.getbbox() is None:
                return True
            else:
                return False
                
    except Exception as e:
        print(f"  [Pixel Compare Error]: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("用法: python compare_images.py <圖片路徑1> <圖片路徑2>")
        sys.exit(1)
        
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    
    # 檢查檔案是否存在
    if not os.path.exists(path1):
        print(f"錯誤: 找不到檔案 {path1}")
        sys.exit(1)
    if not os.path.exists(path2):
        print(f"錯誤: 找不到檔案 {path2}")
        sys.exit(1)

    print(f"正在比較 '{path1}' 和 '{path2}'...\n")
    
    # --- 方法 1 ---
    print("--- 1. 檔案雜湊比對 (Byte-for-byte) ---")
    if compare_by_file_hash(path1, path2):
        print("結果: 相同 (兩個檔案的位元組完全一致)")
    else:
        print("結果: 不同 (檔案內容或中繼資料不同)")
    
    # --- 方法 2 ---
    print("\n--- 2. 像素比對 (Pixel-perfect) ---")
    if compare_by_pixel(path1, path2):
        print("結果: 相同 (所有像素、尺寸、模式都一樣)")
    else:
        print("結果: 不同 (像素、尺寸或模式有差異)")
        
if __name__ == "__main__":
    main()