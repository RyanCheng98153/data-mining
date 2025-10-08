#!/bin/bash
# Ablation experiment script for Data Mining HW1
# Baseline + remove-one-feature experiments

OUT_DIR="./results_ablation"
mkdir -p "$OUT_DIR"

EPOCHS=20000
VAL_MONTHS=2        # or 0 if you already fixed validation manually
PAT=1000

ALL_FEATURES=("AMB_TEMP" "CH4" "CO" "NMHC" "NO" "NO2" "NOx" "O3" "PM10" "PM2.5" "RAINFALL" "RH" "SO2" "THC" "WD_HR" "WIND_DIREC" "WIND_SPEED" "WS_HR")

echo -e "\n[From Echo] ========= Baseline: all features =========="
python train_model.py --train ./data/train.csv --test ./data/test.csv \
    --out_model "$OUT_DIR/model_all.npz" \
    --out_pred "$OUT_DIR/pred_all.csv" \
    --epochs $EPOCHS \
    --val_months $VAL_MONTHS \
    --patience $PAT \
    --standardize

echo -e "\n[From Echo] ========= Start ablation experiments =========="

for f in "${ALL_FEATURES[@]}"; do
    # remove one feature from the list
    FEATURES=$(printf ",%s" "${ALL_FEATURES[@]/$f/}")
    FEATURES=${FEATURES:1}  # remove leading comma

    MODEL_NAME="model_minus_${f}.npz"
    PRED_NAME="pred_minus_${f}.csv"

    echo "Running without feature: $f"
    python train_model.py --train ./data/train.csv --test ./data/test.csv \
        --out_model "$OUT_DIR/$MODEL_NAME" \
        --out_pred "$OUT_DIR/$PRED_NAME" \
        --features "$FEATURES" \
        --epochs $EPOCHS \
        --val_months $VAL_MONTHS \
        --patience $PAT \
        --standardize
done

echo -e "\n[From Echo] ========= All ablation runs completed. Results saved to $OUT_DIR =========="