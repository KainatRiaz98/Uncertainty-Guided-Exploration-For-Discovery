#!/bin/bash
export RAY_DEDUP_LOGS=0
export RAY_LOG_TO_STDERR=1
export RAY_ADDRESS=auto
export TOKENIZERS_PARALLELISM=false

PYTHONPATH=mLoRA:$PYTHONPATH discovery/bin/python3 -m tinker_cookbook.rl.mlora_train \
    --env cp \
    --problem_idx "26" \
    --base_model Qwen/Qwen3-8B \
    --precision fp16 \
    --num_ensemble_members 5 \
    --adv_estimator entropic_adaptive_beta \
    --rmi_coef 0.1 \
    --uncertainty_metric rmi \
    --kl_penalty_coef 0.01 \
    --lora_rank 4 \
    --lora_alpha 8 \
    --learning_rate 4e-5 \
    --group_size 8 \
    --groups_per_batch 1 \
    --num_epochs 10 \
    --max_tokens 26000 \
    --temperature 1.0 \
    --sampler_type puct_backprop \
    --initial_exp_type random \
    --two_phase_sampling \
    --phase1_max_tokens 26000 \
    --streaming_mi \
    --streaming_mi_wrap_budget 6000 \
    --streaming_mi_warmup_epochs 3 \
    --kv_quantize \
    --wandb_project "ttt-discover-uncertainty" \
    --wandb_name "7-april-mi" \
    --log_path ./logs/cp26_rmi \
    --save_every 5
