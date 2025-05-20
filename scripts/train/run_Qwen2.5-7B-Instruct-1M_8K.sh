#!/bin/bash
set -eu

source ~/anaconda3/etc/profile.d/conda.sh
conda activate veRL

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

ADV_ESTIMATOR=GRPO
RL_DATASET=DeepScaleR-Preview-Dataset
MAX_RESPONSE_LENGTH=4096

MODEL_NAME=Qwen2.5-7B-Instruct-1M
MODEL_PATH=/mnt/pri_public/Mamba/Project/OpenRLHF-R1/Qwen2.5-7B-Instruct-1M/SFT_OpenR1-Math-220k_8K/global_step2000_hf
OUTPUT_DIR=/mnt/pri_public/Mamba/Project/DeepScaleR/${MODEL_NAME}/${ADV_ESTIMATOR}_${RL_DATASET}_${MAX_RESPONSE_LENGTH}

PROJECT_NAME=DeepScaleR
EXPERIMENT_NAME=${MODEL_NAME}_${ADV_ESTIMATOR}_${RL_DATASET}_${MAX_RESPONSE_LENGTH}

DATA_DIR=$HOME/DeepScaleR/deepscaler/data/train_parquet
TRAIN_FILES=$DATA_DIR/deepscaler-preview-dataset.parquet
# VALID_FILES="$DATA_DIR/aime-2024.parquet,$DATA_DIR/amc_2022-2023.parquet,$DATA_DIR/math-500.parquet,$DATA_DIR/minerva.parquet,$DATA_DIR/olympiad_bench.parquet"
VALID_FILES=$DATA_DIR/aime-2024.parquet,$DATA_DIR/math-500.parquet

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}/model_train.log"
if [ -f "$log_file" ]; then
    echo "Overwrite Log: $log_file"
    > "$log_file"
else
    echo "Create Log: $log_file"
    touch "$log_file"
fi

echo "=============================================="
echo "Real-Time Training Log Monitoring"
echo "tail -f ${log_file}"
echo "=============================================="

cd ~/DeepScaleR
# Train over a single node, 8 A100-80GB GPUs.
PYTHONWARNINGS="ignore" python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV_ESTIMATOR,,} \
    data.train_files=$TRAIN_FILES \
    data.val_files=[$VALID_FILES] \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-06 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    +actor_rollout_ref.rollout.micro_batch_size=4 \
    +actor_rollout_ref.rollout.micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.total_epochs=1 "${@:1}" \
    >> "${log_file}" 2>&1