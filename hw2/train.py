import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# from torchvision.datasets import ImageFolder #不再需要
import numpy as np
import cv2
import os
import argparse
import pandas as pd
from tqdm import tqdm
# from sklearn.metrics import roc_auc_score # --- 移除 sklearn ---

# --- 1. Config ---
config = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 50,
    "model_save_path": "./models",
    "submission_path": "./submissions",
    "dataset_root": "./dataset",
    "ground_truth_csv": "submission_h.csv" # 使用您提供的 Ground Truth
}

# --- 2. Preprocessing (FFT) ---
def preprocess_fft(image_path):
    """
    讀取圖片，轉灰階 (256x256)，並計算FFT幅度譜
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    img = cv2.resize(img, (256, 256))
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / \
                         (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
    
    return torch.tensor(magnitude_spectrum, dtype=torch.float32).unsqueeze(0)

# --- 3. Custom Dataset (已修正) ---
class FFTDataset(Dataset):
    def __init__(self, root_dir):
        # --- FIX: Don't use ImageFolder. Find images directly. ---
        self.paths = []
        if not os.path.isdir(root_dir):
            print(f"Error: Directory not found: {root_dir}")
            return
            
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.paths.append(os.path.join(root_dir, filename))
        
        if not self.paths:
            print(f"Warning: No images found in {root_dir}")
        # --- End of FIX ---

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        
        try:
            # 嘗試從 '.../1.png' 提取 '1'
            img_id = int(os.path.basename(img_path).split('.')[0])
        except ValueError:
            img_id = -1 # 訓練時用不到 ID
            
        fft_tensor = preprocess_fft(img_path)
        return fft_tensor, img_id

# --- 4. Model (Convolutional Autoencoder) ---
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # 1x256x256 -> 16x128x128
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x64x64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x32x32
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- 5. Save / Load Functions ---
def save_model(model, path, object_type):
    if not os.path.exists(path):
        os.makedirs(path)
    model_name = f"ae_model_{object_type}.pth"
    torch.save(model.state_dict(), os.path.join(path, model_name))
    print(f"Model saved to {os.path.join(path, model_name)}")

def load_model(model, path, object_type):
    model_name = f"ae_model_{object_type}.pth"
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded from {model_path}")
        return True
    else:
        print(f"No model found at {model_path}")
        return False

# --- 6. Train Function ---
def train(model, train_loader, object_type, device):
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    model.train()
    for epoch in range(config["num_epochs"]):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        total_loss = 0
        for data, _ in loop: # 訓練時我們不需要 ID
            normal_data = data.to(device)
            if normal_data.shape[0] == 0:
                continue

            output = model(normal_data)
            loss = criterion(output, normal_data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loop):.6f}")
    
    save_model(model, config["model_save_path"], object_type)

# --- 7. Evaluate Function ---
def evaluate(model, test_loader, device, true_labels_map):
    criterion = nn.MSELoss(reduction='none') # 我們要每個樣本的 loss
    model.eval()
    
    results = [] # 存 (id, score, true_label)
    
    with torch.no_grad():
        for data, img_ids in tqdm(test_loader, desc=f"Evaluating"):
            data = data.to(device)
            
            outputs = model(data)
            
            loss_per_sample = criterion(outputs, data).mean(dim=[1,2,3]).cpu().numpy()
            
            for i in range(len(img_ids)):
                img_id = img_ids[i].item()
                if img_id == -1: 
                    continue
                    
                score = loss_per_sample[i]
                true_label = true_labels_map.get(img_id, 0)
                
                results.append({
                    "id": img_id,
                    "score": score,
                    "true_label": int(true_label)
                })
                
    return pd.DataFrame(results)

# --- 8. Main Function (已修正) ---
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 讀取 Ground Truth (submission_h.csv)
    try:
        ground_truth_df = pd.read_csv(config["ground_truth_csv"])
        true_labels_map = pd.Series(
            ground_truth_df.prediction.values, 
            index=ground_truth_df.id
        ).to_dict()
        print(f"Loaded ground truth from {config['ground_truth_csv']}")
    except FileNotFoundError:
        print(f"Warning: Ground truth file '{config['ground_truth_csv']}' not found.")
        print("Local accuracy cannot be calculated.")
        true_labels_map = {}

    # --- FIX: We are training ONE model, not one per category ---
    model = ConvAutoencoder().to(device)
    
    # 'default_model' 是我們儲存單一模型的名稱
    model_category_name = "default_model" 
    
    # --- Train or Load Model ---
    if not args.eval: # 如果不是 '僅評估' 模式
        if args.load_model:
            load_model(model, args.load_model_path, model_category_name)
        
        print(f"Starting training for {model_category_name}...")
        
        # 直接指向 train 資料夾
        train_dir = os.path.join(config["dataset_root"], "train")
        if not os.path.exists(train_dir):
            print(f"Error: Train folder not found at {train_dir}.")
            return
            
        train_dataset = FFTDataset(root_dir=train_dir)
        if len(train_dataset) == 0:
            print(f"Error: No training images found in {train_dir}.")
            return

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        train(model, train_loader, model_category_name, device)
        
    else: # '僅評估' 模式
        if not args.load_model_path:
            print("Error: --eval mode requires --load_model_path.")
            return
        if not load_model(model, args.load_model_path, model_category_name):
            print(f"Exiting: Model not found for evaluation.")
            return

    # --- Evaluation ---
    print(f"Starting evaluation...")
    
    # 直接指向 test 資料夾
    test_dir = os.path.join(config["dataset_root"], "test")
    if not os.path.exists(test_dir):
        print(f"Error: Test folder not found at {test_dir}.")
        return
        
    test_dataset = FFTDataset(root_dir=test_dir)
    if len(test_dataset) == 0:
        print(f"Error: No test images found in {test_dir}.")
        return
        
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # category_results_df 現在是所有結果
    final_df = evaluate(model, test_loader, device, true_labels_map)
    
    # --- 聚合所有結果並產生 Submission ---
    if final_df.empty:
        print("No evaluation results to process. Exiting.")
        return
    # --- End of FIX ---

    if not os.path.exists(config["submission_path"]):
        os.makedirs(config["submission_path"])
        
    final_df = final_df.sort_values(by="id").reset_index(drop=True)
    
    # 1. 產生 submission_analysis.csv (浮點數版)
    submission_float = final_df[['id', 'score']]
    submission_float = submission_float.rename(columns={"score": "prediction"})
    float_path = os.path.join(config["submission_path"], "submission_analysis.csv")
    submission_float.to_csv(float_path, index=False)
    print(f"\nRaw score submission saved to: {float_path}")

    # 2. 產生 submission.csv (二進位版，使用中位數)
    all_scores = final_df['score'].values
    final_threshold = np.median(all_scores)
    print(f"Final threshold (median of scores): {final_threshold:.6f}")
    
    final_df['binary_pred'] = (final_df['score'] > final_threshold).astype(int)
    
    submission_binary = final_df[['id', 'binary_pred']]
    submission_binary = submission_binary.rename(columns={"binary_pred": "prediction"})
    binary_path = os.path.join(config["submission_path"], "submission.csv")
    submission_binary.to_csv(binary_path, index=False)
    print(f"Binary submission saved to: {binary_path}")
    
    # --- 修正：改成計算簡易準確率 ---
    if 'true_label' in final_df.columns and 'binary_pred' in final_df.columns:
        # 確保兩邊都是整數型態
        true_labels = final_df['true_label'].astype(int)
        binary_preds = final_df['binary_pred'].astype(int)
        
        correct_predictions = (binary_preds == true_labels).sum()
        total_predictions = len(final_df)
        
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            print(f"\n--- 簡易準確率 (與 {config['ground_truth_csv']} 比較) ---")
            print(f"總題數: {total_predictions}")
            print(f"答對題數: {correct_predictions}")
            print(f"準確率: {accuracy:.2f}%")
        else:
            print("\n無法計算準確率：沒有可比較的預測結果。")
            
    else:
        print(f"\n無法計算準確率：缺少 'true_label' 或 'binary_pred' 欄位。")
    # --- 修正結束 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HW2 Anomaly Detection")
    parser.add_argument(
        '--load_model_path', 
        type=str, 
        default=config["model_save_path"],
        help="Path to load model(s) from (used for --eval or continue training)"
    )
    parser.add_argument(
        '--load_model',
        action='store_true',
        default=False,
        help="Flag to load model for continuing training (use with --load_model_path)"
    )
    parser.add_argument(
        '--eval', 
        default=False,
        action='store_true', 
        help="Run evaluation only (requires --load_model_path)"
    )
    
    args = parser.parse_args()
    
    main(args)