import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm # 用於顯示進度條
import argparse  # <-- 1. 加入 import

# --- 3. Custom MVTec Dataset ---
class MVTecDataset(Dataset):
    """
    自訂資料集
    - 訓練模式 (is_test=False): 只返回圖片
    - 測試模式 (is_test=True): 返回圖片和其檔案名稱
    """
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        
        # 獲取所有圖片路徑 (支援 .png, .jpg, .jpeg)
        self.image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif'):
             self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
        
        # 確保測試集的 ID 順序正確
        if self.is_test:
            # 根據檔案名稱中的數字進行排序 (e.g., '0.png', '1.png', ...)
            self.image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 讀取圖片並確保為 RGB (MVTec AD 都是 RGB)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            # 測試集需要返回檔案名，以便生成 submission.csv
            filename = os.path.basename(img_path)
            return image, filename
        else:
            # 訓練集只需要返回圖片
            return image

# --- 4. Autoencoder Model (Convolutional) ---
class ConvAutoencoder(nn.Module):
    # 將 latent_dim 作為參數傳入
    def __init__(self, latent_dim):
        super(ConvAutoencoder, self).__init__()
        
        # --- Encoder ---
        # 根據作業要求，我們不使用預訓練模型
        # Input: (B, 3, 128, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # -> (B, 32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (B, 64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (B, 128, 16, 16)
            nn.ReLU(),
            # 使用傳入的 latent_dim
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=2, padding=1), # -> (B, 256, 8, 8)
            nn.ReLU()
        )
        
        # --- Decoder ---
        # Input: (B, 256, 8, 8)
        self.decoder = nn.Sequential(
            # 使用傳入的 latent_dim
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 128, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1), # -> (B, 3, 128, 128)
            nn.Tanh() # 輸出範圍 [-1, 1]，匹配我們的 Normalize
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 5. Train Function ---
# 傳入 config 字典來獲取參數
def train_model(model, train_loader, criterion, optimizer, config):
    print("Starting training...")
    model.train() # 設置為訓練模式
    
    device = config['DEVICE']
    num_epochs = config['EPOCHS']
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # 我們只使用 'normal' 圖片進行訓練
        for images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            
            # --- Autoencoder 訓練 ---
            # 1. 前向傳播
            outputs = model(images)
            
            # 2. 計算重建損失 (Reconstruction Loss)
            loss = criterion(outputs, images)
            
            # 3. 反向傳播與優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
    print("Training finished.")

# --- 6. Evaluation and Submission Function (Binary 0/1) ---
# 傳入 config 字典來獲取參數
# **注意**：此版本會產生 0/1 標籤，這對於 AUC 評分 *不是* 最優的
def evaluate_and_submit(model, train_loader, test_loader, criterion_eval, config):
    model.eval() # 設置為評估模式
    device = config['DEVICE']
    
    # --- 1. 從訓練集 (正常資料) 計算異常閾值 ---
    print("Calculating anomaly threshold from training data...")
    train_losses = []
    with torch.no_grad():
        for images in tqdm(train_loader, desc="Calculating threshold"):
            images = images.to(device)
            outputs = model(images)
            # 計算每個樣本的 MSE
            loss_per_sample = criterion_eval(outputs, images)
            loss_per_sample = loss_per_sample.view(loss_per_sample.size(0), -1).mean(dim=1)
            train_losses.extend(loss_per_sample.cpu().numpy())
    
    # 使用百分位數 (例如 99%) 作為閾值
    threshold = np.percentile(train_losses, config['THRESHOLD_PERCENTILE'])
    print(f"Anomaly Threshold ({config['THRESHOLD_PERCENTILE']}th percentile) set to: {threshold:.6f}")

    # --- 2. 評估測試集並產生 0/1 預測 ---
    print("Starting evaluation on test set...")
    results = []
    with torch.no_grad(): # 評估時不需要計算梯度
        for images, filenames in tqdm(test_loader, desc="Evaluating test set"):
            images = images.to(device)
            
            # 重建圖片
            outputs = model(images)
            
            # 計算每張圖片的 MSE 損失 (異常分數)
            loss_per_sample = criterion_eval(outputs, images)
            loss_per_sample = loss_per_sample.view(loss_per_sample.size(0), -1).mean(dim=1)
            
            # 將分數與閾值比較，產生 0 (normal) 或 1 (anomaly)
            # preds = (loss_per_sample > threshold).cpu().numpy().astype(int)
            preds = loss_per_sample.cpu().numpy()
            
            # 儲存結果
            for fname, pred_label in zip(filenames, preds):
                # 儲存 0 或 1
                results.append({'filename': fname, 'prediction': pred_label})

    # threshold 使用 results 中的中位數作為最終閾值
    all_scores = np.array([r['prediction'] for r in results])
    final_threshold = np.median(all_scores)
    print(f"Final threshold (median of scores): {final_threshold:.6f}")
    
    submission_results = []
    
    for k, r in enumerate(results):
        submission_results.append({'filename': r['filename'], 
                                   'prediction': 1 if r['prediction'] > final_threshold else 0})
    
    # --- 3. 產生 Kaggle 提交檔案 ---
    submission_filename = config['SUBMISSION_FILE']
    print(f"Generating submission file: {submission_filename}")

    def gen_csv(results, output_filename):
        df = pd.DataFrame(results)
        
        # 從 '0.png', '1.png' 中提取 'id'
        df['id'] = df['filename'].apply(lambda x: int(os.path.splitext(x)[0]))
        
        # 確保 ID 排序正確並只保留 'id' 和 'prediction' 欄位
        df = df.sort_values(by='id')
        df_submission = df[['id', 'prediction']]
        
        # 儲存為 CSV
        df_submission.to_csv(output_filename, index=False)
        print("Submission file created successfully!")
        print(df_submission.head())

    gen_csv(submission_results, submission_filename)
    gen_csv(results, submission_filename.replace('.csv', '_analysis.csv'))

def unique_filename(filename):
    # if file is exists, append a number to make it unique
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

# --- 2. Argument Parser ---
def argument_parser():
    parser = argparse.ArgumentParser(description="Anomaly Detection with Autoencoder")
    
    # --eval 參數: store_true 表示如果有此參數則為 True (預設 False)
    # 如果執行 'python script.py --eval'，則 args.eval 會是 True
    parser.add_argument('--eval', action='store_true', default=False, 
                        help='Set this flag to evaluation mode (skip training)')

    # --load_model 參數: 指定模型路徑
    parser.add_argument('--load_model', type=str, default=None, 
                        help='Path to load a pre-trained model for evaluation')
        
    return parser.parse_args()


# --- 7. Main Execution ---
if __name__ == "__main__":
    
    # --- 0. Parse Arguments ---
    args = argument_parser()
    
    # 根據 --eval 參數設定 TRAIN_MODE
    # 如果有 --eval, args.eval 為 True, TRAIN_MODE 為 False
    TRAIN_MODE = not args.eval
    
    # 取得要載入的模型路徑
    model_load_path = args.load_model
    
    # --- 1. Configuration & Hyperparameters ---
    config = {
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'DATA_PATH': './dataset/',
        'SUBMISSION_FILE': unique_filename('submission.csv'),
        # 'MODEL_SAVE_PATH' 用於 *儲存* 新訓練的模型
        'MODEL_SAVE_PATH': unique_filename('./anomaly_ae.pth'), 
        'TRAIN_MODE': TRAIN_MODE,
        
        # 模型超參數
        'IMG_SIZE': 128,
        'BATCH_SIZE': 64,
        'EPOCHS': 50,
        'LR': 1e-3,
        'LATENT_DIM': 256,
        'THRESHOLD_PERCENTILE': 99,           # 用於 0/1 分類的閾值 (百分位)
        
        # DataLoader 參數
        'NUM_WORKERS': 4,
        'PIN_MEMORY': True
    }
    
    # 從 config 衍生路徑
    config['TRAIN_DIR'] = os.path.join(config['DATA_PATH'], 'train')
    config['TEST_DIR'] = os.path.join(config['DATA_PATH'], 'test')

    print(f"Using device: {config['DEVICE']}")
    
    # 檢查資料夾是否存在
    if not os.path.exists(config['TRAIN_DIR']):
        print(f"Error: Training directory not found at {config['TRAIN_DIR']}")
        exit()
    if not os.path.exists(config['TEST_DIR']):
        print(f"Error: Test directory not found at {config['TEST_DIR']}")
        exit()

    # --- 2. Data Transforms ---
    data_transform = transforms.Compose([
        transforms.Resize((config['IMG_SIZE'], config['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- 準備 DataLoaders ---
    print("Loading datasets...")
    # 訓練集 (只有 normal 圖片) - 評估時也需要用它來算閾值
    train_dataset = MVTecDataset(root_dir=config['TRAIN_DIR'], transform=data_transform, is_test=False)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['BATCH_SIZE'], 
        # 訓練時 shuffle, 評估/計算閾值時不用
        shuffle=True if config['TRAIN_MODE'] else False, 
        num_workers=config['NUM_WORKERS'], 
        pin_memory=config['PIN_MEMORY']
    )
    
    # 測試集 (normal + anomaly 混合)
    test_dataset = MVTecDataset(root_dir=config['TEST_DIR'], transform=data_transform, is_test=True)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['BATCH_SIZE'], 
        shuffle=False, 
        num_workers=config['NUM_WORKERS'], 
        pin_memory=config['PIN_MEMORY']
    )

    # --- 初始化模型 ---
    model = ConvAutoencoder(latent_dim=config['LATENT_DIM']).to(config['DEVICE'])
    
    # 訓練用的損失函數 (計算 batch 平均)
    criterion_train = nn.MSELoss() 
    
    # 評估用的損失函數 (計算 每個樣本 的損失，以便排序或閾值)
    criterion_eval = nn.MSELoss(reduction='none') 
    
    
    # --- 執行訓練或載入模型 ---
    if config['TRAIN_MODE']:
        print("Mode: Training")
        optimizer = optim.Adam(model.parameters(), lr=config['LR'], weight_decay=1e-5)
        # T傳入 config 字典
        train_model(model, train_loader, criterion_train, optimizer, config)
        
        # 訓練後儲存模型
        print(f"Saving model to {config['MODEL_SAVE_PATH']}...")
        torch.save(model.state_dict(), config['MODEL_SAVE_PATH'])
        print(f"Model saved to {config['MODEL_SAVE_PATH']}")
    
    else: # 進入評估模式 (TRAIN_MODE = False)
        print(f"Mode: Evaluation")
        
        # 檢查是否提供了 --load_model 參數
        if model_load_path is None:
            print("Error: Evaluation mode selected (--eval) but no model path provided.")
            print("Please use --load_model <path_to_model.pth> to specify the model to load.")
            exit()
            
        # 檢查模型檔案是否存在
        if not os.path.exists(model_load_path):
            print(f"Error: Model file not found at {model_load_path}")
            exit()
            
        # 載入已儲存的權重
        print(f"Loading model from {model_load_path}...")
        model.load_state_dict(torch.load(model_load_path, map_location=config['DEVICE']))
        print("Model loaded successfully.")

    
    # --- 執行評估並產生提交檔案 ---
    # **注意**：evaluate_and_submit 現在需要 train_loader 來計算閾值
    evaluate_and_submit(model, train_loader, test_loader, criterion_eval, config)