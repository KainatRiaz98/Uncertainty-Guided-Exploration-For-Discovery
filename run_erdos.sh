#!/bin/bash
export RAY_DEDUP_LOGS=0
export RAY_LOG_TO_STDERR=1
export RAY_ADDRESS=auto
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHONPATH=mLoRA:$PYTHONPATH discovery/bin/python3 -m tinker_cookbook.rl.mlora_train \
    --env erdos \
    --problem_idx "200" \
    --budget_s 1000 \
    --base_model Qwen/Qwen3-8B \
    --precision fp16 \
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
    --num_epochs 10 \
    --max_tokens 260000 \
    --temperature 1.0 \
    --num_cpus_per_task 2 \
    --eval_timeout 1100 \
    --dataset_timeout 1200 \
    --sampler_type puct_backprop \
    --initial_exp_type random \
    --two_phase_sampling \
    --phase1_max_tokens 26000 \
    --streaming_mi \
    --streaming_mi_wrap_budget 6000 \
    --streaming_mi_warmup_epochs 3 \
    --streaming_mi_threshold_percentile 5.0 \
    --wandb_project "ttt-discover-uncertainty" \
    --wandb_name "erdos-real-run-truemi-r16-nnm" \
    --log_path ./logs/erdos_rmi_nnm \
    --save_every 5
