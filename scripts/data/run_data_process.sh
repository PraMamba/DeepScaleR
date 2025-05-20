#!/bin/bash
set -eu

source ~/anaconda3/etc/profile.d/conda.sh
conda activate veRL

cd ~/DeepScaleR
python scripts/data/deepscaler_dataset.py \
    --local_dir ~/DeepScaleR/deepscaler/data/train_parquet