#!/bin/bash

# Run training for the model
python main.py --mode train \
    --train_data ./dataset/train \
    --save_dir ./save_models

# Run inference for the model
# python main.py --mode inference \
#   --test_data ./dataset/test \
#   --save_dir ./save_models \
#   --load_model ./save_models/model_epoch_3.pth