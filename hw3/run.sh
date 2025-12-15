#!/bin/bash

# Run training for the model
python main.py --mode train \
    --train_data ./dataset/train \
    --save_dir ./save_models \
    --epochs 1000 \

for i in {1..1000}
do
    echo "----------------------------------------"
    echo "Running inference iteration $i"
    echo "----------------------------------------"
    # Run inference for the model
    python main.py --mode inference \
    --test_data ./dataset/test \
    --save_dir ./save_models \
    --load_model ./save_models/model_epoch.pth
done