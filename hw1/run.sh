# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--train', default='/mnt/data/train.csv')
#     p.add_argument('--test',  default='/mnt/data/test.csv')
#     p.add_argument('--out_model', default='/mnt/data/model.npz')
#     p.add_argument('--out_pred',  default='/mnt/data/predictions.csv')
#     p.add_argument('--val_months', type=int, default=2, help='最後幾個月做為驗證集')
#     p.add_argument('--window', type=int, default=9, help='用前 W 小時預測下一小時 (default 9)')
#     p.add_argument('--lr', type=float, default=1.0, help='學習率 learning rate')
#     p.add_argument('--lambda_l2', type=float, default=0.01, help='L2 正則化強度')
#     p.add_argument('--epochs', type=int, default=2000, help='最大 epoch 數')
#     p.add_argument('--patience', type=int, default=50, help='多少 epoch 沒進步就提早停止')
#     p.add_argument('--min_delta', type=float, default=1e-4, help='進步幅度小於此值就不算進步')
#     p.add_argument('--standardize', action='store_true', help='是否對 X 做標準化')
#     p.add_argument('--features', default=None, help='逗號分隔 feature 名稱（不含空格）; 預設使用所有')
#     return p.parse_args()

# ========= Scripts to run experiments ==========

# sh run.sh 2>&1 | tee -a test.log

echo -e "\n[From Echo] =============================== Separator ==============================="

# echo -e "\n[From Echo] Testing... : default parameters"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_default.npz --out_pred predictions_default.csv

# echo -e "\n[From Echo] Testing... : validation months = 3"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_default_val_3.npz --out_pred predictions_default_val_3.csv \
#     --val_months 3

# echo -e "\n[From Echo] Testing... : validation months = 4"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_default_val_4.npz --out_pred predictions_default_val_4.csv \
#     --val_months 4 

# echo -e "\n[From Echo] Testing... : epochs = 5000"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_5000.npz --out_pred predictions_epochs_5000.csv \
#     --epochs 5000

# echo -e "\n[From Echo] Testing... : epochs = 10000"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000.npz --out_pred predictions_epochs_10000.csv \
#     --epochs 10000

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 10000, val_months = 3"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000_val_3.npz --out_pred predictions_epochs_10000_val_3.csv \
#     --epochs 10000 --val_months 3 

# echo -e "\n[From Echo] Testing... : epochs = 10000, val_months = 4"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000_val_4.npz --out_pred predictions_epochs_10000_val_4.csv \
#     --epochs 10000 --val_months 4

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 2000, val_months = 1"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_2000_val_1.npz --out_pred predictions_epochs_2000_val_1.csv \
#     --epochs 2000 --val_months 1

# echo -e "\n[From Echo] Testing... : epochs = 5000, val_months = 1"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_5000_val_1.npz --out_pred predictions_epochs_5000_val_1.csv \
#     --epochs 5000 --val_months 1

# echo -e "\n[From Echo] Testing... : epochs = 10000, val_months = 1"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000_val_1.npz --out_pred predictions_epochs_10000_val_1.csv \
#     --epochs 10000 --val_months 1

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 15000, val_months = 2, patience = 100"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_15000_val_2_pat_100.npz --out_pred predictions_epochs_15000_val_2_pat_100.csv \
#     --epochs 15000 --val_months 2 --patience 100

# echo -e "\n[From Echo] Testing... : epochs = 20000, val_months = 2, patience = 100"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_20000_val_2_pat_100.npz --out_pred predictions_epochs_20000_val_2_pat_100.csv \
#     --epochs 20000 --val_months 2 --patience 100

# echo -e "\n[From Echo] Testing... : epochs = 30000, val_months = 2, patience = 500"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_30000_val_2_pat_500.npz --out_pred predictions_epochs_30000_val_2_pat_500.csv \
#     --epochs 30000 --val_months 2 --patience 500

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 40000, val_months = 2, patience = 1000"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_40000_val_2_pat_1000.npz --out_pred predictions_epochs_40000_val_2_pat_1000.csv \
#     --epochs 40000 --val_months 2 --patience 1000

# echo -e "\n[From Echo] Testing... : epochs = 10000, val_months = 2, patience = 1000, standardize"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000_val_2_pat_1000_standardized.npz --out_pred predictions_epochs_10000_val_2_pat_1000_standardized.csv \
#     --epochs 10000 --val_months 2 --patience 1000 --standardize

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 10000, val_months = 1, patience = 1000, standardize"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000_val_1_pat_1000_standardized.npz --out_pred predictions_epochs_10000_val_1_pat_1000_standardized.csv \
#     --epochs 10000 --val_months 1 --patience 1000 --standardize

# echo -e "\n[From Echo] Testing... : epochs = 10000, val_months = 3, patience = 1000, standardize"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000_val_3_pat_1000_standardized.npz --out_pred predictions_epochs_10000_val_3_pat_1000_standardized.csv \
#     --epochs 10000 --val_months 3 --patience 1000 --standardize

# echo -e "\n[From Echo] Testing... : epochs = 10000, val_months = 4, patience = 1000, standardize"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000_val_4_pat_1000_standardized.npz --out_pred predictions_epochs_10000_val_4_pat_1000_standardized.csv \
#     --epochs 10000 --val_months 4 --patience 1000 --standardize

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 10000, val_months = 0, patience = 1000, standardize, no validation set"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_10000_val_0_pat_1000_standardized_noval.npz --out_pred predictions_epochs_10000_val_0_pat_1000_standardized_noval.csv \
#     --epochs 10000 --val_months 0 --patience 1000 --standardize

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 20000, val_months = 0, patience = 1000, standardize, no validation set"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_20000_val_0_pat_1000_standardized_noval.npz --out_pred predictions_epochs_20000_val_0_pat_1000_standardized_noval.csv \
#     --epochs 20000 --val_months 0 --patience 1000 --standardize

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 20000, val_months = 0, patience = 1000, standardize, no validation set"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_20000_val_0_pat_1000_standardized_noval.npz --out_pred predictions_epochs_20000_val_0_pat_1000_standardized_noval.csv \
#     --epochs 20000 --val_months 0 --patience 1000 --standardize

# ========== New experiments ==========

# echo -e "\n[From Echo] Testing... : epochs = 20000, val_months = 1, patience = 100000, standardize, no validation set"
# python train_model.py --train ./data/train.csv --test ./data/test.csv \
#     --out_model model_epochs_20000_val_1_pat_100000_standardized_noval.npz --out_pred predictions_epochs_20000_val_1_pat_100000_standardized_noval.csv \
#     --epochs 20000 --val_months 1 --patience 100000 --standardize
