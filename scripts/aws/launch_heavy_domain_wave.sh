#!/bin/bash
# Heavy-model domain wave for node 3 (8x A100-80GB): the wave4 domain pairs
# (denoising, ahc039) re-run on Qwen2.5-72B-Instruct instead of Qwen3-8B, at
# 2 seeds each to use all 8 GPUs.
#
# Why a SEPARATE script instead of adding to launch_wave.sh: true 2-GPU model
# sharding was ruled out (mLoRA's pipeline mode is multi-process training-only,
# no generation path; the custom forward has no cross-device .to() hops; see
# the session plan for the full investigation). Instead this uses QLoRA
# quantization (--precision nf4), which is ALREADY implemented in
# mLoRA/mlora/model/llm/model_llama.py (BitsAndBytesConfig) and needs zero
# forward-pass changes — a 70B model at nf4 is ~35-40GB, fitting on ONE 80GB
# GPU. So each run gets 1 GPU, not 2, and 8 runs fill the node instead of 4.
#
# Qwen2.5-72B-Instruct is NOT a <think>-reasoning model, so these runs are
# single-phase (no --two_phase_sampling). mlora_train.py's single-phase path
# now wraps prompts in plain ChatML (wrap_chatml_single_phase) and sets the
# <|im_end|> stop token — added alongside this script since the single-phase
# path previously sent raw, unwrapped prompts (dead code path until now: every
# run through launch_wave.sh sets --two_phase_sampling).
#
# REQUIRES: pip install bitsandbytes  (not in requirements/*.txt — quantized
# loading needs it; nothing else here does, so it isn't bundled into
# requirements-math.txt). Also requires the same per-domain infra as wave4:
# denoising's bio/openproblems stack in this venv, ahc039's Docker + the
# ALE-Bench container.
#
# SMOKE-TEST ONE RUN PER DOMAIN FIRST. This exact combination — nf4 + a new
# model family + single-phase + two brand-new domains — has no prior run
# anywhere. Confirm real reward + non-garbled completions before filling the
# node:
#   bash scripts/aws/launch_heavy_domain_wave.sh smoke
# Then, once both smoke runs look healthy:
#   bash scripts/aws/launch_heavy_domain_wave.sh run
#
#   bash scripts/aws/launch_heavy_domain_wave.sh --list
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
# Same shared-head model as launch_wave.sh — see that script's comment for why
# (task layer hardcodes ray.init("auto"); base_reward_task.py's cpu_scheduler
# actor partitions this node's CPUs across co-resident runs; no taskset).
export RAY_ADDRESS=auto

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_CPUS="${TOTAL_CPUS:-$(nproc)}"
MODEL="${MODEL:-Qwen/Qwen2.5-72B-Instruct}"

ensure_ray_head() {
  if ! ray status >/dev/null 2>&1; then
    echo "No Ray head found — starting one (--num-cpus ${TOTAL_CPUS})."
    ray start --head --num-cpus="${TOTAL_CPUS}" --disable-usage-stats >/dev/null
  fi
}

# Single-phase, quantized. NOTE: no --two_phase_sampling, no --streaming_mi —
# streaming stays off project-wide except the dedicated wave2 comparison pair.
COMMON=(
  --base_model "${MODEL}"
  --precision nf4
  --adv_estimator entropic_adaptive_beta
  --kl_penalty_coef 0.01
  --lora_rank 16
  --lora_alpha 32
  --learning_rate 4e-5
  --group_size 8
  --groups_per_batch 8
  --num_epochs 6
  # SINGLE-PHASE decode bound. Do NOT copy launch_wave.sh's 260000 here: that
  # is a TWO-phase total budget, and the two-phase path clamps itself to the
  # context window (ensemble.py phase2_budget). The single-phase decode loop
  # (`for step in range(1, max_tokens)`, generate_batch_multi_adapter) has NO
  # context bound, so max_tokens must stay under context_window (32768) minus
  # room for the prompt, or generation runs past the context and blows up.
  --max_tokens 24000
  --temperature 1.0
  --sampler_type puct_backprop
  --initial_exp_type random
  --num_cpus_per_task 2
  --save_every 2
)

UGTTT=(    --num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric true_mi )
BASELINE=( --num_ensemble_members 1 --rmi_coef 0.0 --nnm_coef 0.0 )

# Format: "<run-name>|<env>|<problem_idx>|<arm>|<extra flags>"
declare -a RUNS=(
  "denoise-ugttt-s2|denoising|improvement|UGTTT|--seed 2"
  "denoise-ugttt-s3|denoising|improvement|UGTTT|--seed 3"
  "denoise-base-s2|denoising|improvement|BASELINE|--seed 2"
  "denoise-base-s3|denoising|improvement|BASELINE|--seed 3"
  "ahc039-ugttt-s2|ahc039|ahc039|UGTTT|--seed 2"
  "ahc039-ugttt-s3|ahc039|ahc039|UGTTT|--seed 3"
  "ahc039-base-s2|ahc039|ahc039|BASELINE|--seed 2"
  "ahc039-base-s3|ahc039|ahc039|BASELINE|--seed 3"
)

# The two runs used to validate the setup before committing all 8 GPUs.
declare -a SMOKE=(
  "denoise-base-s2|denoising|improvement|BASELINE|--seed 2"
  "ahc039-base-s2|ahc039|ahc039|BASELINE|--seed 2"
)

usage() { echo "usage: $0 [--list] <smoke|run>"; exit 1; }

list_runs() {
  local -n arr=$1
  printf '%-20s %-11s %-12s %-9s %s\n' RUN ENV PROBLEM ARM EXTRA
  for spec in "${arr[@]}"; do
    IFS='|' read -r name env pidx arm extra <<< "$spec"
    printf '%-20s %-11s %-12s %-9s %s\n' "$name" "$env" "$pidx" "$arm" "$extra"
  done
}

[[ $# -ge 1 ]] || usage

if [[ "$1" == "--list" ]]; then
  echo "model: $MODEL   precision: nf4   two_phase_sampling: off"
  echo; echo "== smoke (run first) =="; list_runs SMOKE
  echo; echo "== run (all 8, after smoke passes) =="; list_runs RUNS
  exit 0
fi

MODE="$1"
case "$MODE" in
  smoke) SELECTED=("${SMOKE[@]}") ;;
  run)   SELECTED=("${RUNS[@]}") ;;
  *) usage ;;
esac

[[ ${#SELECTED[@]} -le $NUM_GPUS ]] || { echo "error: ${#SELECTED[@]} runs > $NUM_GPUS GPUs" >&2; exit 1; }

ensure_ray_head

LOG_DIR="logs/aws/heavy_domain_${MODE}"
mkdir -p "$LOG_DIR"

echo "Launching ${#SELECTED[@]} runs on $NUM_GPUS GPUs — model: $MODEL (nf4, 1 GPU/run)"
echo "Logs: $LOG_DIR"
echo

gpu=0
for spec in "${SELECTED[@]}"; do
  IFS='|' read -r name env pidx arm extra <<< "$spec"
  declare -n arm_flags="$arm"

  extra_flags=()
  for tok in $extra; do
    extra_flags+=( "$tok" )
  done

  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" \
  setsid \
    python3 -m tinker_cookbook.rl.mlora_train \
      "${COMMON[@]}" \
      "${arm_flags[@]}" \
      "${extra_flags[@]}" \
      --env "$env" \
      --problem_idx "$pidx" \
      --log_path "./logs/aws/heavy_domain_${MODE}/${name}" \
      --wandb_project "ugttt-rebuttal" \
      --wandb_name "$name" \
    > "${LOG_DIR}/${name}.log" 2>&1 &

  echo "  GPU ${gpu}  ${name}  (pid $!)"
  gpu=$(( gpu + 1 ))
done

echo
echo "All runs detached. Follow one with:"
echo "  tail -f ${LOG_DIR}/<run-name>.log"
echo "Check placement:  nvidia-smi   (one python process per GPU)"
if [[ "$MODE" == "smoke" ]]; then
  echo
  echo "Smoke test running. Before launching 'run', confirm in each log:"
  echo "  - nf4 load succeeded (no bitsandbytes import/CUDA error)"
  echo "  - completions look like real code, not garbled tokens (confirms ChatML wrapping)"
  echo "  - train/correctness/nonzero > 0 for at least one epoch"
fi
