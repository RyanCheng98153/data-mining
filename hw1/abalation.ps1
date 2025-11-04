# Ablation experiment script for Data Mining HW1
# Baseline + remove-one-feature experiments (PowerShell version)

# 設定輸出資料夾
$OUT_DIR = "./results_ablation"
if (-not (Test-Path $OUT_DIR)) {
    New-Item -ItemType Directory -Path $OUT_DIR | Out-Null
}

# 參數設定
$EPOCHS = 20000
$VAL_MONTHS = 0      # 若已手動固定驗證集，則可設為 0
$PAT = 1000

# 所有特徵名稱
$ALL_FEATURES = @(
    "AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3",
    "PM10", "PM2.5", "RAINFALL", "RH", "SO2", "THC", "WD_HR",
    "WIND_DIREC", "WIND_SPEED", "WS_HR"
)

Write-Host "`n[From Echo] ========= Baseline: all features ==========" -ForegroundColor Cyan
Write-Output "`n[From Echo] ========= Baseline: all features =========="

python train_model.py `
    --train ./data/train.csv `
    --test ./data/test.csv `
    --out_model "$OUT_DIR/model_all.npz" `
    --out_pred "$OUT_DIR/pred_all.csv" `
    --epochs $EPOCHS `
    --val_months $VAL_MONTHS `
    --patience $PAT `
    --standardize

Write-Host "`n[From Echo] ========= Start ablation experiments ==========" -ForegroundColor Yellow
Write-Output "`n[From Echo] ========= Start ablation experiments =========="

foreach ($f in $ALL_FEATURES) {
    # 移除單一 feature
    $features = $ALL_FEATURES | Where-Object { $_ -ne $f }
    $features_str = ($features -join ",")

    $MODEL_NAME = "model_minus_${f}.npz"
    $PRED_NAME = "pred_minus_${f}.csv"

    Write-Host "`n[From Echo] Running without feature: $f" -ForegroundColor Green
    Write-Output "`n[From Echo] Running without feature: $f"

    python train_model.py `
        --train ./data/train.csv `
        --test ./data/test.csv `
        --out_model "$OUT_DIR/$MODEL_NAME" `
        --out_pred "$OUT_DIR/$PRED_NAME" `
        --features "$features_str" `
        --epochs $EPOCHS `
        --val_months $VAL_MONTHS `
        --patience $PAT `
        --standardize
}

Write-Host "`n[From Echo] ========= All ablation runs completed. Results saved to $OUT_DIR ==========" -ForegroundColor Cyan
Write-Output "`n[From Echo] ========= All ablation runs completed. Results saved to $OUT_DIR"
