#!/bin/bash
# EXPLORATORY cross-family test: does UG-TTT transfer to a non-Qwen model?
#
# Runs Llama-3.1-8B (a genuinely different family) on CP26 as a UG-TTT vs
# baseline pair, SINGLE-PHASE (Llama has no <think> reasoning mode, so the
# two-phase Qwen3 path does not apply). Requires the family-aware templating
# added on branch exp/cross-family-llama (mlora_train.py: wrap_llama_chat_template,
# base_model_family, <|eot_id|> stop token).
#
# This is NOT validated on GPU yet — see scripts/aws/CROSS_FAMILY_NOTES.md for
# the exact things to check on the first run. Smoke-test ONE run before the pair.
#
#   bash scripts/aws/launch_llama_test.sh --list
#   bash scripts/aws/launch_llama_test.sh run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export RAY_ADDRESS=auto

NUM_GPUS="${NUM_GPUS:-2}"
TOTAL_CPUS="${TOTAL_CPUS:-$(nproc)}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"

ensure_ray_head() {
  if ! ray status >/dev/null 2>&1; then
    echo "No Ray head found — starting one (--num-cpus ${TOTAL_CPUS})."
    ray start --head --num-cpus="${TOTAL_CPUS}" --disable-usage-stats >/dev/null
  fi
}

# Single-phase config: NO --two_phase_sampling (that is the Qwen3 thinking path).
# max_tokens capped to 26k so a single generation fits Llama's context easily.
COMMON=(
  --base_model "${MODEL}"
  --precision fp16
  --adv_estimator entropic_adaptive_beta
  --kl_penalty_coef 0.01
  --lora_rank 16
  --lora_alpha 32
  --learning_rate 4e-5
  --group_size 8
  --groups_per_batch 8
  --num_epochs 6
  --max_tokens 26000
  --temperature 1.0
  --sampler_type puct_backprop
  --initial_exp_type random
  --num_cpus_per_task 2
  --save_every 2
)

UGTTT=(    --num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric true_mi )
BASELINE=( --num_ensemble_members 1 --rmi_coef 0.0 --nnm_coef 0.0 )

declare -a RUNS=(
  "cp26-llama8b-ugttt|cp|26|UGTTT"
  "cp26-llama8b-base|cp|26|BASELINE"
)

usage() { echo "usage: $0 [--list] run"; exit 1; }

if [[ "${1:-}" == "--list" ]]; then
  printf '%-22s %-5s %-9s %-9s %s\n' RUN ENV PROBLEM ARM MODEL
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r name env pidx arm <<< "$spec"
    printf '%-22s %-5s %-9s %-9s %s\n' "$name" "$env" "$pidx" "$arm" "$MODEL"
  done
  exit 0
fi

[[ "${1:-}" == "run" ]] || usage

ensure_ray_head
LOG_DIR="logs/aws/llama_test"
mkdir -p "$LOG_DIR"
echo "Launching ${#RUNS[@]} Llama runs (single-phase) on $NUM_GPUS GPUs — model: $MODEL"

gpu=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name env pidx arm <<< "$spec"
  declare -n arm_flags="$arm"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" \
  setsid \
    python3 -m tinker_cookbook.rl.mlora_train \
      "${COMMON[@]}" \
      "${arm_flags[@]}" \
      --env "$env" \
      --problem_idx "$pidx" \
      --log_path "./logs/aws/llama_test/${name}" \
      --wandb_project "ugttt-rebuttal" \
      --wandb_name "$name" \
    > "${LOG_DIR}/${name}.log" 2>&1 &
  echo "  GPU ${gpu}  ${name}  (pid $!)"
  gpu=$(( gpu + 1 ))
done
echo
echo "Smoke-test ONE first: tail -f ${LOG_DIR}/cp26-llama8b-base.log"
echo "Watch for the checks in scripts/aws/CROSS_FAMILY_NOTES.md before trusting results."
