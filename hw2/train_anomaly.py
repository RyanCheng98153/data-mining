import os
import argparse
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# --- 1. Configuration ---
# 使用 config 字典來管理參數
CONFIG = {
    'data_dir': 'dataset/',
    'model_path': 'cae_model.pth',     # 儲存/讀取模型路徑
    'submission_dir': 'submissions',   # 儲存 submission 檔案的資料夾
    'batch_size': 32,
    'epochs': 25,                      # 訓練週期
    'lr': 1e-3,                        # 學習率
    'img_size': (128, 128),            # 影像統一 resize
    'latent_dim': 32                   # 潛在空間維度 (code)
}

# --- 2. Custom Dataset ---
# 建立一個客製化的 Dataset 來讀取影像
class MVTecDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # 讀取所有圖片檔案名稱 (例如 0.png, 1.png...)
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        
        # 讀取影像並轉為 'L' (灰階)
        # 異常檢測通常在灰階上效果不錯，因為顏色有時是干擾
        try:
            image = Image.open(img_path).convert('L')
        except IOError:
            print(f"Error opening image {img_path}")
            return None, None # 處理損壞圖片

        if self.transform:
            image = self.transform(image)
        
        # Test set 需要回傳檔名 (id)
        # Train set 只需要回傳影像
        if "test" in self.data_dir:
            return image, img_name
        else:
            return image, img_name # 保持回傳格式一致

# --- 3. Model Definition (Convolutional Autoencoder) ---
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder (編碼器)
        self.encoder = nn.Sequential(
            # Input: (B, 1, 128, 128)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # -> (B, 16, 64, 64)
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (B, 32, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (B, 64, 16, 16)
            nn.ReLU(True),
            nn.Flatten(), # -> (B, 64*16*16)
            nn.Linear(64 * 16 * 16, latent_dim) # -> (B, latent_dim)
        )
        
        # Decoder (解碼器)
        self.decoder_input = nn.Linear(latent_dim, 64 * 16 * 16)
        
        self.decoder = nn.Sequential(
            # nn.Unflatten(1, (64, 16, 16)) # 需要 PyTorch 1.8+
            # 手動 unflatten
            # Input: (B, 64*16*16)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 32, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 16, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 1, 128, 128)
            nn.Sigmoid() # 將輸出壓縮到 0-1 之間
        )

    def forward(self, x):
        code = self.encoder(x)
        x_unflat = self.decoder_input(code)
        # 手動 unflatten
        x_unflat = x_unflat.view(-1, 64, 16, 16) 
        reconstructed = self.decoder(x_unflat)
        return reconstructed

# --- 4. Save / Load Functions ---
def save_model(model, path):
    """儲存模型"""
    print(f"Saving model to {path}...")
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    """讀取模型"""
    if not os.path.exists(path):
        print(f"Error: Model file not found at {path}")
        sys.exit(1)
    print(f"Loading model from {path}...")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)

# --- 5. Train Function ---
def train(model, loader, optimizer, criterion, device, epochs):
    model.train() # 設置為訓練模式
    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for (images, _) in pbar:
            images = images.to(device)
            
            # 1. Forward pass
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # 2. Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Average Training Loss: {avg_loss:.6f}")

# --- 6. Evaluate Function ---
def evaluate(model, loader, criterion, device):
    model.eval() # 設置為評估模式
    
    all_losses = []
    all_filenames = []
    
    print("Evaluating test set...")
    with torch.no_grad(): # 評估時不需要計算梯度
        for (images, filenames) in tqdm(loader):
            images = images.to(device)
            
            # 1. 重建影像
            reconstructed = model(images)
            
            # 2. 計算每張影像的重建誤差 (MSE Loss)
            # criterion(reduction='none') 會回傳 (B, C, H, W) 的 loss
            # 我們需要對每張影像的 (C, H, W) 維度取平均，得到 (B,) 的 loss
            loss_per_sample = criterion(reconstructed, images)
            loss_per_sample = torch.mean(loss_per_sample, dim=[1, 2, 3])
            
            all_losses.extend(loss_per_sample.cpu().numpy())
            all_filenames.extend(filenames)
            
    print(f"Evaluation complete. Processed {len(all_filenames)} images.")
    return all_filenames, all_losses

# --- 7. Main Execution ---
def main():
    # --- Argparse Setup ---
    parser = argparse.ArgumentParser(description="HW2 Anomaly Detection")
    parser.add_argument('--load_model', type=str, default=None,
                        help=f"Path to load a pre-trained model (e.g., {CONFIG['model_path']})")
    parser.add_argument('--eval', action='store_true',
                        help="Only run evaluation on the test set. Requires --load_model.")
    args = parser.parse_args()

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 建立 submission 資料夾
    os.makedirs(CONFIG['submission_dir'], exist_ok=True)

    # --- Data Transforms & Loaders ---
    # 訓練和測試使用相同的 transform
    data_transform = transforms.Compose([
        transforms.Resize(CONFIG['img_size']),
        transforms.ToTensor(),
        # ToTensor() 已經將影像縮放到 [0, 1]
    ])

    # --- Model, Optimizer, Loss ---
    model = ConvAutoencoder(latent_dim=CONFIG['latent_dim']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # 訓練時使用 'mean' reduction
    train_criterion = nn.MSELoss(reduction='mean') 
    # 評估時使用 'none' reduction，以便計算每張影像的 loss
    eval_criterion = nn.MSELoss(reduction='none')

    # --- Control Flow (Train or Load) ---
    
    # 1. 檢查是否要讀取模型
    if args.load_model:
        load_model(model, args.load_model, device)
    
    # 2. 檢查是否要執行訓練 (如果不是 'eval_only')
    if not args.eval:
        if args.load_model:
            print("Continuing training from loaded model...")
        else:
            print("Starting new training...")
            
        # 準備訓練資料
        train_dataset = MVTecDataset(data_dir=os.path.join(CONFIG['data_dir'], 'train'), 
                                     transform=data_transform)
        train_loader = DataLoader(train_dataset, 
                                  batch_size=CONFIG['batch_size'], 
                                  shuffle=True, 
                                  num_workers=4,
                                  pin_memory=True)
        
        if len(train_dataset) == 0:
            print(f"Error: No images found in {os.path.join(CONFIG['data_dir'], 'train')}")
            sys.exit(1)
            
        # 執行訓練
        train(model, train_loader, optimizer, train_criterion, device, CONFIG['epochs'])
        
        # 儲存模型
        save_model(model, CONFIG['model_path'])
    
    elif not args.load_model:
        # 如果 --eval 被設置，但 --load_model 沒有被設置
        print("Error: --eval mode requires --load_model to be specified.")
        sys.exit(1)

    # --- Evaluation & Submission ---
    print("Preparing test data...")
    test_dataset = MVTecDataset(data_dir=os.path.join(CONFIG['data_dir'], 'test'), 
                                transform=data_transform)
    test_loader = DataLoader(test_dataset, 
                             batch_size=CONFIG['batch_size'], 
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True)
    
    if len(test_dataset) == 0:
            print(f"Error: No images found in {os.path.join(CONFIG['data_dir'], 'test')}")
            sys.exit(1)
            
    # 執行評估
    filenames, anomaly_scores = evaluate(model, test_loader, eval_criterion, device)
    
    # 從檔名 (e.g., "123.png") 提取 id (e.g., 123)
    # 並確保 id 為整數
    try:
        ids = [int(os.path.splitext(f)[0]) for f in filenames]
    except ValueError as e:
        print(f"Error: Could not parse ID from filenames. Example: {filenames[0]}. Error: {e}")
        sys.exit(1)

    # --- 建立 Submission 檔案 ---
    
    # 1. 浮點數版 (分數版)
    # 這個版本最適合用來計算 AUC 
    df_scores = pd.DataFrame({
        'id': ids,
        'prediction_score': anomaly_scores
    })
    df_scores = df_scores.sort_values(by='id') # 根據 id 排序
    score_path = os.path.join(CONFIG['submission_dir'], 'submission_scores.csv')
    df_scores.to_csv(score_path, index=False)
    print(f"Floating-point scores saved to {score_path}")

    # 2. Binary 版 (0/1 版) - 根據你的要求
    # 使用 test_set 分數的「中位數」當作 threshold
    threshold = np.median(anomaly_scores)
    print(f"Using threshold (median of scores): {threshold:.6f}")
    
    # 異常分數 > threshold 的為 anomaly (1) [cite: 103], 否則為 normal (0) [cite: 100]
    binary_predictions = [1 if score > threshold else 0 for score in anomaly_scores]
    
    df_binary = pd.DataFrame({
        'id': ids,
        'prediction': binary_predictions # Column name 必須為 id 和 prediction [cite: 104]
    })
    df_binary = df_binary.sort_values(by='id') # 根據 id 排序
    binary_path = os.path.join(CONFIG['submission_dir'], 'submission.csv')
    df_binary.to_csv(binary_path, index=False)
    print(f"Binary submission (for HW example) saved to {binary_path}")

if __name__ == "__main__":
    main()