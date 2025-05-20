#!/bin/bash

set -eu  # 遇到错误时退出

# 加载 conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate OpenRLHF

# 进入脚本目录
cd ~/verl/scripts

# 定义多个 local_dir 目录
local_dirs=(
    "/mnt/pri_public/Mamba/Project/DeepScaleR/DeepSeek-R1-Distill-Qwen-7B/GRPO_DeepScaleR-Preview-Dataset_8192/global_step_480/actor"
    "/mnt/pri_public/Mamba/Project/DeepScaleR/DeepSeek-R1-Distill-Qwen-7B/GRPO_DeepScaleR-Preview-Dataset_8192/global_step_960/actor"
    "/mnt/pri_public/Mamba/Project/DeepScaleR/DeepSeek-R1-Distill-Qwen-7B/GRPO_DeepScaleR-Preview-Dataset_8192/global_step_1440/actor"
)

# 遍历 local_dirs 并依次运行
for dir in "${local_dirs[@]}"; do
    echo "Processing directory: $dir"
    python model_merger.py --local_dir "$dir"
done

echo "All tasks completed!"
