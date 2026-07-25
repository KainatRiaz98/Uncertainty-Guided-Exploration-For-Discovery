#!/bin/bash
# Launch the model-size scaling matrix (14B / 32B / 72B) on one 8-GPU node,
# with each run layer-sharded across several GPUs.
#
# Differs from launch_wave.sh in exactly one way: a run may claim MORE THAN ONE
# GPU. The spec gains a gpus-per-run field and the allocator hands out a
# contiguous slice of device indices per run.
#
#   bash scripts/aws/launch_scaling_wave.sh --list
#   bash scripts/aws/launch_scaling_wave.sh scale72
#
# RUN THE PRE-FLIGHT FIRST — it reports the real per-model memory and the
# recommended --gpus, using the actual HF configs rather than these estimates:
#
#   python scripts/preflight_model_check.py \
#     --models Qwen/Qwen3-14B Qwen/Qwen3-32B Qwen/Qwen2.5-72B-Instruct \
#     --gpu-memory-gb 80 --num-gpus 8
#
# AND RUN THE ACCEPTANCE GATE BEFORE ANY OF THIS (see
# tests/test_sharded_equivalence.py). A sharding bug found after a 72B run has
# burned a day is not recoverable on this deadline.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
# ONE shared Ray head per node — same contract as launch_wave.sh: the task layer
# hardcodes ray.init("auto") and base_reward_task.py get-or-creates a detached
# host-keyed "cpu_scheduler" actor that partitions this node's CPUs across all
# co-resident runs. Do NOT taskset-pin.
export RAY_ADDRESS=auto

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_CPUS="${TOTAL_CPUS:-$(nproc)}"
GPU_START="${GPU_START:-0}"

ensure_ray_head() {
  if ! ray status >/dev/null 2>&1; then
    echo "No Ray head found — starting one (--num-cpus ${TOTAL_CPUS})."
    ray start --head --num-cpus="${TOTAL_CPUS}" --disable-usage-stats >/dev/null
  fi
}

# Published UG-TTT configuration, identical to launch_wave.sh except that
# --base_model is set per run. fp16 at EVERY model size on purpose: sharding is
# what makes that possible, and it keeps model size the only variable across the
# scaling curve (the earlier nf4-for-72B plan confounded size with quantization).
COMMON=(
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
  --save_every 1
)

UGTTT=(   --num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric true_mi )
BASELINE=( --num_ensemble_members 1 --rmi_coef 0.0 --nnm_coef 0.0 )

M14="Qwen/Qwen3-14B"
M32="Qwen/Qwen3-32B"
M72="Qwen/Qwen2.5-72B-Instruct"

# Spec: "<name>|<env>|<problem_idx>|<arm>|<gpus>|<extra flags>"
#
# GPU counts at ctx 32768 / group_size 8, from: fp16 weights + 1.0 GiB/layer of
# pre-allocated KV. KV is LINEAR in context_window and group_size, so if a run
# OOMs those are the levers — but changing them changes the experiment, so
# prefer giving the run another GPU.
#
#   14B  40 layers   ~29 GB weights + 40 GB KV  = ~73 GB  -> 2 GPUs (1 is marginal)
#   32B  64 layers   ~65 GB weights + 64 GB KV  = ~133 GB -> 2 GPUs (tight), 3 safe
#   72B  80 layers  ~145 GB weights + 80 GB KV  = ~231 GB -> 4 GPUs

# Both arms of the long pole together: 4 + 4 = 8.
WAVE_scale72=(
  "ac1-72b-ugttt|ac1|improvement|UGTTT|4|--base_model $M72"
  "ac1-72b-base|ac1|improvement|BASELINE|4|--base_model $M72"
)

# 14B and 32B pairs together: 2 + 2 + 2 + 2 = 8. 32B on 2 GPUs is ~84% full —
# if it OOMs during the prefill->N KV expansion, use scale32safe instead.
WAVE_scale_small=(
  "ac1-14b-ugttt|ac1|improvement|UGTTT|2|--base_model $M14"
  "ac1-14b-base|ac1|improvement|BASELINE|2|--base_model $M14"
  "ac1-32b-ugttt|ac1|improvement|UGTTT|2|--base_model $M32"
  "ac1-32b-base|ac1|improvement|BASELINE|2|--base_model $M32"
)

# More headroom for 32B: 3 + 3 + 2 = 8. Run the 14B baseline in a later wave.
WAVE_scale32safe=(
  "ac1-32b-ugttt|ac1|improvement|UGTTT|3|--base_model $M32"
  "ac1-32b-base|ac1|improvement|BASELINE|3|--base_model $M32"
  "ac1-14b-ugttt|ac1|improvement|UGTTT|2|--base_model $M14"
)

# One-run smoke on the largest model, to prove placement before committing the box.
WAVE_smoke72=(
  "smoke-72b|ac1|improvement|UGTTT|4|--base_model $M72 --num_epochs 1 --groups_per_batch 1 --group_size 2 --phase1_max_tokens 2000 --max_tokens 4000"
)

usage() {
  cat <<EOF
usage: bash scripts/aws/launch_scaling_wave.sh <wave>

waves:
  smoke72      1 run  x 4 GPUs   72B placement smoke (do this first)
  scale72      2 runs x 4 GPUs   72B UGTTT + BASELINE
  scale_small  4 runs x 2 GPUs   14B + 32B pairs (32B tight)
  scale32safe  2x3 + 1x2 GPUs    32B pair with headroom + 14B UGTTT

env: NUM_GPUS (default 8), GPU_START (default 0), TOTAL_CPUS (default nproc)
EOF
}

[[ $# -ge 1 ]] || { usage; exit 1; }
case "$1" in
  --list|-l|-h|--help) usage; exit 0 ;;
  smoke72)     WAVE_NAME="smoke72";     RUNS=( "${WAVE_smoke72[@]}" ) ;;
  scale72)     WAVE_NAME="scale72";     RUNS=( "${WAVE_scale72[@]}" ) ;;
  scale_small) WAVE_NAME="scale_small"; RUNS=( "${WAVE_scale_small[@]}" ) ;;
  scale32safe) WAVE_NAME="scale32safe"; RUNS=( "${WAVE_scale32safe[@]}" ) ;;
  *) echo "unknown wave: $1" >&2; usage; exit 1 ;;
esac

# Validate the GPU budget BEFORE launching anything — a partially-launched wave
# that dies on the last run leaves the box in a confusing half-busy state.
total_claimed=0
for spec in "${RUNS[@]}"; do
  IFS='|' read -r _name _env _pidx _arm gpus_per_run _extra <<< "$spec"
  total_claimed=$(( total_claimed + gpus_per_run ))
done
if (( GPU_START + total_claimed > NUM_GPUS )); then
  echo "error: wave claims ${total_claimed} GPUs starting at ${GPU_START}, but NUM_GPUS=${NUM_GPUS}" >&2
  exit 1
fi

ensure_ray_head

LOG_DIR="logs/aws/${WAVE_NAME}"
mkdir -p "$LOG_DIR"

echo "Launching ${#RUNS[@]} sharded runs claiming ${total_claimed}/${NUM_GPUS} GPUs"
echo "Logs: $LOG_DIR"
echo

gpu=$GPU_START
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name env pidx arm gpus_per_run extra <<< "$spec"
  declare -n arm_flags="$arm"

  extra_flags=()
  for tok in $extra; do extra_flags+=( "$tok" ); done

  # Contiguous slice of physical devices for this run. CUDA_VISIBLE_DEVICES
  # remaps them to 0..n-1 inside the process, which is what --gpus counts.
  devices=""
  for (( i = 0; i < gpus_per_run; i++ )); do
    devices+="${devices:+,}$(( gpu + i ))"
  done

  CUDA_VISIBLE_DEVICES="$devices" \
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" \
  setsid \
    python3 -m tinker_cookbook.rl.mlora_train \
      "${COMMON[@]}" \
      "${arm_flags[@]}" \
      "${extra_flags[@]}" \
      --gpus "$gpus_per_run" \
      --env "$env" \
      --problem_idx "$pidx" \
      --log_path "./logs/aws/${WAVE_NAME}/${name}" \
      --wandb_project "ugttt-rebuttal" \
      --wandb_name "$name" \
    > "${LOG_DIR}/${name}.log" 2>&1 &

  echo "  GPUs ${devices}  ${name}  (pid $!)"
  gpu=$(( gpu + gpus_per_run ))
done

cat <<EOF

All runs detached. Follow one with:
  tail -f ${LOG_DIR}/<run-name>.log

Verify sharding actually happened (each run should span its whole GPU slice):
  nvidia-smi
  grep -E "Shard layout|logits land on" ${LOG_DIR}/*.log

Watch for these run-invalidating signatures in the logs:
  "PROMPT TRUNCATED"        prompt exceeded the 4096-token cap; that rollout scores 0
  "partially offloaded"     a shard landed on CPU/disk — run would be ~100x slower
  uncertainty_true_mi = 0   ensemble collapsed; MI carries no signal
  reward_exec all zero      verifier or code-extraction failure, not a model result
EOF
