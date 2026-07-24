#!/bin/bash
# Launch one wave of UG-TTT runs on a single 8-GPU g7e.48xlarge.
#
# One run per GPU, each pinned to its own slice of vCPUs so the per-run CPU
# scheduler (utils/cpu_scheduler.py, which sizes itself from sched_getaffinity)
# does not oversubscribe the box. See docs/aws_g7e48xlarge.md.
#
#   bash scripts/aws/launch_wave.sh --list
#   bash scripts/aws/launch_wave.sh wave1
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."
export PYTHONPATH="${PWD}:${PWD}/tasks:${PYTHONPATH:-}"
unset RAY_ADDRESS

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_CPUS="${TOTAL_CPUS:-$(nproc)}"
CPUS_PER_RUN=$(( TOTAL_CPUS / NUM_GPUS ))

# Hyperparameters matching the published UG-TTT runs. The Config defaults in
# mlora_train.py are the upstream TTT-Discover values, so these are explicit.
COMMON=(
  --base_model "Qwen/Qwen3-8B"
  --precision fp16
  --lora_rank 16
  --lora_alpha 32
  --group_size 8
  --groups_per_batch 8
  --num_epochs 6
  --learning_rate 4e-5
  --kl_penalty_coef 0.01
  --max_tokens 26000
  --num_cpus_per_task 2
  --adv_estimator entropic_adaptive_beta
  --sampler_type puct_backprop
  --initial_exp_type random
)

# Arm definitions.
UGTTT=(   --num_ensemble_members 5 --rmi_coef 0.1 --uncertainty_metric rmi )
BASELINE=( --num_ensemble_members 1 --rmi_coef 0.0 )
ALPHA0=(  --num_ensemble_members 5 --rmi_coef 0.0 --uncertainty_metric rmi )
ENTROPY=( --num_ensemble_members 5 --rmi_coef 0.1 --uncertainty_metric predictive_entropy )

# ── Wave definitions ─────────────────────────────────────────────────────────
# Format: "<run-name>|<env>|<problem_idx>|<arm>|<extra flags>"
# NOTE: seeded runs require a --seed argument that does not yet exist; see
# docs/aws_g7e48xlarge.md. Add it before launching wave1.
declare -a WAVE1=(
  "ac1-ugttt-seed2|ac1|improvement|UGTTT|--seed 2"
  "ac1-ugttt-seed3|ac1|improvement|UGTTT|--seed 3"
  "ac1-base-seed2|ac1|improvement|BASELINE|--seed 2"
  "ac1-base-seed3|ac1|improvement|BASELINE|--seed 3"
  "cp26-ugttt-seed2|cp|26|UGTTT|--seed 2"
  "cp26-ugttt-seed3|cp|26|UGTTT|--seed 3"
  "cp26-base-seed2|cp|26|BASELINE|--seed 2"
  "cp26-base-seed3|cp|26|BASELINE|--seed 3"
)

declare -a WAVE2=(
  "cp26-alpha0|cp|26|ALPHA0|"
  "cp26-entropy|cp|26|ENTROPY|"
  "ac1-nostream|ac1|improvement|UGTTT|"
  "cp26-nostream|cp|26|UGTTT|"
  "denoise-base|denoising|improvement|BASELINE|"
  "denoise-ugttt|denoising|improvement|UGTTT|"
  "ac2-ugttt-seed2|ac2|improvement|UGTTT|--seed 2"
  "ac2-base-seed2|ac2|improvement|BASELINE|--seed 2"
)

usage() {
  echo "usage: $0 [--list] <wave1|wave2>"
  exit 1
}

list_runs() {
  local -n arr=$1
  printf '%-22s %-10s %-12s %s\n' RUN ENV PROBLEM ARM
  for spec in "${arr[@]}"; do
    IFS='|' read -r name env pidx arm extra <<< "$spec"
    printf '%-22s %-10s %-12s %s %s\n' "$name" "$env" "$pidx" "$arm" "$extra"
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

if [[ ${#RUNS[@]} -gt $NUM_GPUS ]]; then
  echo "error: ${#RUNS[@]} runs but only $NUM_GPUS GPUs" >&2
  exit 1
fi

LOG_DIR="logs/aws/${WAVE_NAME}"
mkdir -p "$LOG_DIR"

echo "Launching ${#RUNS[@]} runs on $NUM_GPUS GPUs (${CPUS_PER_RUN} vCPUs each)"
echo "Logs: $LOG_DIR"
echo

gpu=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name env pidx arm extra <<< "$spec"

  # Resolve the named arm array into flags.
  declare -n arm_flags="$arm"

  cpu_lo=$(( gpu * CPUS_PER_RUN ))
  cpu_hi=$(( cpu_lo + CPUS_PER_RUN - 1 ))

  CUDA_VISIBLE_DEVICES="$gpu" \
  setsid taskset -c "${cpu_lo}-${cpu_hi}" \
    python -m tinker_cookbook.rl.mlora_train \
      "${COMMON[@]}" \
      "${arm_flags[@]}" \
      ${extra} \
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
echo "Check GPU placement with: nvidia-smi"
echo "Load average should settle near ${TOTAL_CPUS}; much higher means pinning failed."
