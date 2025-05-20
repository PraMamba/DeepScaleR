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
export RAY_TMPDIR=/data2/Mamba/Ray_Cache
export NCCL_TIMEOUT=600000

ADV_ESTIMATOR=GRPO
RL_DATASET=DeepScaleR-Preview-Dataset
MAX_RESPONSE_LENGTH=8192

MODEL_NAME=DeepSeek-R1-Distill-Qwen-7B
MODEL_PATH=/data1/Mamba/Model/DeepSeek/DeepSeek-R1-Distill-Qwen-7B
OUTPUT_DIR=/data2/Mamba/Project/DeepScaleR/${MODEL_NAME}/${ADV_ESTIMATOR}_${RL_DATASET}_${MAX_RESPONSE_LENGTH}_Test

PROJECT_NAME=DeepScaleR
DATE_SUFFIX=$(date +"%Y%m%d_%H%M")
EXPERIMENT_NAME=${MODEL_NAME}_${ADV_ESTIMATOR}_${RL_DATASET}_${MAX_RESPONSE_LENGTH}_${DATE_SUFFIX}

DATA_DIR=$HOME/DeepScaleR/deepscaler/data/train_parquet
TRAIN_FILES=$DATA_DIR/deepscaler-preview-dataset.parquet
# VALID_FILES="$DATA_DIR/aime-2024.parquet,$DATA_DIR/amc_2022-2023.parquet,$DATA_DIR/math-500.parquet,$DATA_DIR/minerva.parquet,$DATA_DIR/olympiad_bench.parquet"
# VALID_FILES=$DATA_DIR/aime-2024.parquet,$DATA_DIR/aime-2025.parquet,$DATA_DIR/hmmt-202502.parquet,$DATA_DIR/aimo-2_reference.parquet
VALID_FILES=$DATA_DIR/aime-2024.parquet

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
PYTHONWARNINGS="ignore" python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=${ADV_ESTIMATOR,,} \
    data.train_files=$TRAIN_FILES \
    data.val_files=[$VALID_FILES] \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 "${@:1}" \
    >> "${log_file}" 2>&1