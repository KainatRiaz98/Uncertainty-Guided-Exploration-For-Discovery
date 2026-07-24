#!/bin/bash
# Launch one wave of UG-TTT runs on a single 8-GPU g7e.48xlarge.
#
# Targets the UG-TTT implementation in the epistemic-uncertainty-for-test-time-discovery
# repo (module ug_ttt.rl.mlora_train). Run this from a checkout of THAT repo —
# flags mirror its scripts/run.sh. See docs/aws_g7e48xlarge.md.
#
#   bash scripts/aws/launch_wave.sh --list
#   bash scripts/aws/launch_wave.sh wave1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
# Each run gets its OWN local Ray instance. Do NOT set RAY_ADDRESS=auto here:
# utils/cpu_scheduler.py registers a detached actor named "cpu_scheduler", so 8
# runs sharing one Ray cluster would collide on that name and on its CPU pool.
unset RAY_ADDRESS || true

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_CPUS="${TOTAL_CPUS:-$(nproc)}"
CPUS_PER_RUN=$(( TOTAL_CPUS / NUM_GPUS ))

# Published UG-TTT configuration (paper Tables 3-4). The argparse defaults are
# upstream TTT-Discover values, so these are passed explicitly.
COMMON=(
  --base_model "Qwen/Qwen3-8B"
  --precision fp16
  --adv_estimator entropic_adaptive_beta
  --kl_penalty_coef 0.01
  --lora_rank 16
  --lora_alpha 32
  --learning_rate 4e-5
  --group_size 8
  --groups_per_batch 8
  --num_epochs 6
  --max_tokens 260000
  --two_phase_sampling
  --phase1_max_tokens 26000
  --temperature 1.0
  --sampler_type puct_backprop
  --initial_exp_type random
  --num_cpus_per_task 2
  --save_every 5
)

# Streaming MI early-stop (paper Table 4). Append to runs that need it.
STREAM=(
  --streaming_mi
  --streaming_mi_wrap_budget 6000
  --streaming_mi_warmup_epochs 3
  --streaming_mi_threshold_percentile 25.0
)

# ── Arms ─────────────────────────────────────────────────────────────────────
UGTTT=(    --num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric true_mi )
BASELINE=( --num_ensemble_members 1 --rmi_coef 0.0 --nnm_coef 0.0 )
ALPHA0=(   --num_ensemble_members 5 --rmi_coef 0.0 --nnm_coef 0.075 --uncertainty_metric true_mi )
ENTROPY=(  --num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric predictive_entropy )
NONNM=(    --num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.0   --uncertainty_metric true_mi )

# ── Wave definitions ─────────────────────────────────────────────────────────
# Format: "<run-name>|<env>|<problem_idx>|<arm>|<extra flags>"
#
# NOTE: --seed does not exist yet (ensemble.py:88 hardcodes the seed). The
# seeded runs below will fail on an unrecognised argument until it is added.
declare -a WAVE1=(
  "ac1-ugttt-seed2|ac1|improvement|UGTTT|--seed 2 STREAM"
  "ac1-ugttt-seed3|ac1|improvement|UGTTT|--seed 3 STREAM"
  "ac1-base-seed2|ac1|improvement|BASELINE|--seed 2"
  "ac1-base-seed3|ac1|improvement|BASELINE|--seed 3"
  "cp26-ugttt-seed2|cp|26|UGTTT|--seed 2 STREAM"
  "cp26-ugttt-seed3|cp|26|UGTTT|--seed 3 STREAM"
  "cp26-base-seed2|cp|26|BASELINE|--seed 2"
  "cp26-base-seed3|cp|26|BASELINE|--seed 3"
)

# Wave 2 needs no --seed and can run today.
declare -a WAVE2=(
  "cp26-alpha0|cp|26|ALPHA0|STREAM"
  "cp26-entropy|cp|26|ENTROPY|STREAM"
  "cp26-nonnm|cp|26|NONNM|STREAM"
  "ac1-nostream|ac1|improvement|UGTTT|"
  "cp26-nostream|cp|26|UGTTT|"
  "erdos-ugttt|erdos|improvement|UGTTT|"
  "denoise-base|denoising|improvement|BASELINE|"
  "denoise-ugttt|denoising|improvement|UGTTT|"
)

usage() { echo "usage: $0 [--list] <wave1|wave2>"; exit 1; }

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
  echo "== wave1 =="; list_runs WAVE1
  echo; echo "== wave2 =="; list_runs WAVE2
  exit 0
fi

WAVE_NAME="$1"
case "$WAVE_NAME" in
  wave1) RUNS=("${WAVE1[@]}") ;;
  wave2) RUNS=("${WAVE2[@]}") ;;
  *) usage ;;
esac

[[ ${#RUNS[@]} -le $NUM_GPUS ]] || { echo "error: ${#RUNS[@]} runs > $NUM_GPUS GPUs" >&2; exit 1; }

LOG_DIR="logs/aws/${WAVE_NAME}"
mkdir -p "$LOG_DIR"

echo "Launching ${#RUNS[@]} runs on $NUM_GPUS GPUs (${CPUS_PER_RUN} vCPUs each)"
echo "Logs: $LOG_DIR"
echo

gpu=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name env pidx arm extra <<< "$spec"
  declare -n arm_flags="$arm"

  # Expand the STREAM sentinel in the extra field.
  extra_flags=()
  for tok in $extra; do
    if [[ "$tok" == "STREAM" ]]; then
      extra_flags+=( "${STREAM[@]}" )
    else
      extra_flags+=( "$tok" )
    fi
  done

  cpu_lo=$(( gpu * CPUS_PER_RUN ))
  cpu_hi=$(( cpu_lo + CPUS_PER_RUN - 1 ))

  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" \
  setsid taskset -c "${cpu_lo}-${cpu_hi}" \
    python3 -m ug_ttt.rl.mlora_train \
      "${COMMON[@]}" \
      "${arm_flags[@]}" \
      "${extra_flags[@]}" \
      --env "$env" \
      --problem_idx "$pidx" \
      --log_path "./logs/aws/${WAVE_NAME}/${name}" \
      --wandb_project "ugttt-rebuttal" \
      --wandb_name "$name" \
    > "${LOG_DIR}/${name}.log" 2>&1 &

  echo "  GPU ${gpu}  cpus ${cpu_lo}-${cpu_hi}  ${name}  (pid $!)"
  gpu=$(( gpu + 1 ))
done

echo
echo "All runs detached. Follow one with:"
echo "  tail -f ${LOG_DIR}/<run-name>.log"
echo "Check placement:  nvidia-smi"
echo "Load average should settle near ${TOTAL_CPUS}; much higher means pinning failed."
