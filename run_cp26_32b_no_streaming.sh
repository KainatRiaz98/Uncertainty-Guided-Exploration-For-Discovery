#!/bin/bash
# cp26, UGTTT arm, Qwen3-32B, layer-sharded across 2 GPUs, streaming_mi disabled.
# Same config as run_cp26.sh minus --streaming_mi* (compare run_ac1_no_streaming.sh
# vs run_ac1_baseline.sh for that pattern), base_model swapped to 32B, --gpus 2
# added since 32B does not fit on one GPU (see scripts/preflight_model_check.py).
set -euo pipefail

: "${WANDB_ENTITY:?WANDB_ENTITY must be set in the environment}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set in the environment}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export RAY_DEDUP_LOGS=0
export RAY_LOG_TO_STDERR=1
export RAY_ADDRESS=auto
export TOKENIZERS_PARALLELISM=false

# NOTE: expandable_segments MUST stay off on this (layer-sharded, --gpus 2) run.
# It allocates via the CUDA virtual-memory APIs, and those mappings do not hold
# up across the cross-device hops the shard path does: the SECOND forward of a
# batch faults with `illegal memory access` / CUBLAS_STATUS_INTERNAL_ERROR,
# reported at a random later op. Verified on this box (RTX PRO 6000 Blackwell,
# sm_120, torch 2.9.1): with it, ensemble scoring fails 3/3 at every sequence
# length tried; without it, 2/2 pass at the production shape (T=16000, 8 seqs).
# Single-GPU runs are unaffected either way.
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # <- do not re-enable while sharded

# Root volume on this box is small (~29GB) — model weights (32B fp16 is ~65GB)
# MUST land on the large ephemeral disk, not the default ~/.cache/huggingface.
# Set here (not just in the launching shell) since env vars from a separate
# shell invocation do not carry over to this script.
export HF_HOME=/opt/dlami/nvme/hf_cache
mkdir -p "$HF_HOME"

# Script is launched detached (setsid nohup ... &), so it does NOT inherit an
# activated venv — PATH may not have .venv/bin on it. Reference the venv's own
# ray binary explicitly rather than relying on bare `ray` resolving via PATH.
RAY="$SCRIPT_DIR/.venv/bin/ray"

if ! "$RAY" status >/dev/null 2>&1; then
  echo "No Ray head found — starting one (--num-cpus $(nproc))."
  "$RAY" start --head --num-cpus="$(nproc)" --disable-usage-stats >/dev/null
fi

PYTHONPATH="mLoRA:${PYTHONPATH:-}" "$SCRIPT_DIR/.venv/bin/python3" -m tinker_cookbook.rl.mlora_train \
    --env cp \
    --problem_idx "26" \
    --base_model Qwen/Qwen3-32B \
    --precision fp16 \
    --gpus 2 \
    --num_ensemble_members 5 \
    --adv_estimator entropic_adaptive_beta \
    --rmi_coef 0.1 \
    --nnm_coef 0.075 \
    --uncertainty_metric true_mi \
    --kl_penalty_coef 0.01 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 4e-5 \
    --group_size 8 \
    --groups_per_batch 8 \
    --num_epochs 6 \
    --max_tokens 260000 \
    --temperature 1.0 \
    --sampler_type puct_backprop \
    --initial_exp_type random \
    --two_phase_sampling \
    --phase1_max_tokens 26000 \
    --wandb_project "ttt-discover-uncertainty" \
    --wandb_name "cp26-32b-no-streaming-truemi-r16-nnm" \
    --log_path ./logs/cp26_32b_no_streaming \
    --save_every 1
