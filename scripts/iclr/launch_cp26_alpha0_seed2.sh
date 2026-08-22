#!/bin/bash
# ============================================================================
# ICLR resubmission rerun (OPTIONAL, third priority) — CP26 alpha=0 ablation
# arm at seed 2, in the wave-2 8192 config.
# Mirror of node2/wave2/cp26-alpha0 (seed 42): ONLY the seed changes (plus
# save_every 2 -> 1, run control only — see note below).
#
#   bash scripts/iclr/launch_cp26_alpha0_seed2.sh check   # preflight, ~10 s
#   bash scripts/iclr/launch_cp26_alpha0_seed2.sh run     # preflight + launch
#
# WHAT THIS TESTS
#   cp26-alpha0 is the "no MI bonus" arm (rmi_coef 0.0) with the ensemble and
#   NNM still on (K=5, nnm 0.075). A second seed tests whether the wave-2
#   ablation ordering holds beyond n=1. The rebuttal deferred the uniqueness /
#   arm-ordering question to exactly these ablation seed runs.
#
# EVERY FLAG BELOW IS COPIED FIELD-FOR-FIELD from
#   ugttt-trajectories-data/extracted_trajectories/node2/wave2/cp26-alpha0/config.json
# EXCEPT --seed 2. This is the wave-2 config: --context_window 8192,
# --max_tokens 16000, --phase1_max_tokens 6000. That is DELIBERATE — the
# comparison target is the wave-2 ablation family (all 8192), not paper
# Table 1 (32768). Do not "fix" these to the paper values; at 32768 this run
# would be comparable to nothing.
#
# DEVIATION (documented, run-control only): wave-2 used --save_every 2; this
# script uses --save_every 1 so the run is safe to kill at any epoch boundary.
# Checkpoint cadence does not touch the RNG stream or training trajectory.
#
# RUNTIME: ~2.98 h/epoch measured on the wave-2 cp26 config -> 6 epochs ~= 18 h.
#   Even 4 epochs is a usable comparison against the other arms truncated to
#   the same budget (the analysis pipeline's default convention). Do not
#   restart it to "get all 6" if you are short on time.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

RUN_NAME="${RUN_NAME:-cp26-alpha0-seed2}"
GPU="${GPU:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
SEED=2                              # <-- THE ONLY MECHANISM-VISIBLE CHANGE
LOG_DIR="logs/iclr"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export RAY_ADDRESS=auto
TOTAL_CPUS="${TOTAL_CPUS:-$(nproc)}"

# ---------------------------------------------------------------- preflight
fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok    $*"; }

preflight() {
  echo "Preflight (${RUN_NAME})"

  # ml_log.py silently no-ops wandb when this is unset.
  [[ -n "${WANDB_API_KEY:-}" ]] || fail "WANDB_API_KEY unset. ml_log.py skips wandb SILENTLY; metrics would only exist in metrics.jsonl."
  ok "WANDB_API_KEY set"
  [[ -n "${WANDB_ENTITY:-}" ]] || echo "  warn  WANDB_ENTITY unset (fine if your default entity is right)"

  ! pgrep -f "wandb_name ${RUN_NAME}" >/dev/null 2>&1 \
    || fail "a process with wandb_name ${RUN_NAME} is already running"
  ok "no duplicate ${RUN_NAME} process"

  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  local ngpu; ngpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  [[ "$GPU" -lt "$ngpu" ]] || fail "GPU=$GPU but only $ngpu GPUs present"
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [[ "$used" -lt 2000 ]] || fail "GPU $GPU already has ${used} MiB in use — pick a free GPU with GPU=<n>"
  ok "GPU $GPU free (${used} MiB used, ${ngpu} GPUs on box)"

  python3 -c "import torch" 2>/dev/null || fail "torch not importable in this python3"
  ok "torch importable"

  [[ -d mLoRA ]] || fail "mLoRA/ missing — PYTHONPATH would not resolve"
  ok "mLoRA present"

  local freegb; freegb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  [[ "$freegb" -ge 40 ]] || fail "only ${freegb} GB free; --save_every 1 writes K=5 adapter checkpoints per epoch"
  ok "${freegb} GB disk free"

  if ! ray status >/dev/null 2>&1; then
    echo "  ..    no Ray head; starting one (--num-cpus ${TOTAL_CPUS})"
    ray start --head --num-cpus="${TOTAL_CPUS}" --disable-usage-stats >/dev/null
  fi
  ok "Ray head up"

  echo "Preflight passed."
  echo
}

# --------------------------------------------------------------------- flags
# Copied field-for-field from node2/wave2/cp26-alpha0/config.json.
# Flags whose value already equals the argparse default are still listed
# explicitly, so this file is a complete record of the run.
ARGS=(
  --base_model            "Qwen/Qwen3-8B"
  --precision             fp16
  --env                   cp
  --problem_idx           26

  --seed                  "${SEED}"        # wave-2 run: 42. THE ONLY CHANGE.

  # --- arm: alpha=0 (ensemble + NNM on, MI bonus OFF) ---
  --num_ensemble_members  5
  --uncertainty_metric    true_mi          # still logged; bonus inert at rmi 0
  --nnm_coef              0.075
  --rmi_coef              0.0              # <-- the ablation
  --gamma_max_ratio       10.0             # default coupling, as recorded (inert at rmi 0)

  # --- identical to the wave-2 arm ---
  --adv_estimator         entropic_adaptive_beta
  --kl_penalty_coef       0.01
  --lora_rank             16
  --lora_alpha            32
  --learning_rate         4e-5
  --group_size            8
  --groups_per_batch      8
  --max_tokens            16000            # wave-2 value, NOT the wave script's 260000
  --two_phase_sampling
  --phase1_max_tokens     6000
  --context_window        8192             # wave-2 value — comparability over fidelity to paper
  --temperature           1.0
  --budget_s              1000
  --eval_timeout          1000
  --dataset_timeout       1000
  --sampler_type          puct_backprop
  --initial_exp_type      random
  --num_cpus_per_task     2
  --eval_every            3

  # --- run control ---
  --num_epochs            "${NUM_EPOCHS}"  # wave-2 ran 6
  --save_every            1                # wave-2 used 2; 1 = killable at any epoch (run control only)
)
# Streaming is OFF: cp26-alpha0 has streaming_mi_enabled=false, so
# --streaming_mi is deliberately NOT passed.

case "${1:-}" in
  check) preflight; echo "Nothing launched. Re-run with: bash $0 run"; exit 0 ;;
  run)   ;;
  *)     echo "usage: $0 <check|run>"; exit 1 ;;
esac

preflight
mkdir -p "$LOG_DIR"

echo "Launching ${RUN_NAME} on GPU ${GPU}"
echo "  seed=${SEED}  K=5  rmi_coef=0.0  nnm_coef=0.075  ctx=8192  epochs=${NUM_EPOCHS}"
echo "  ETA ~$(python3 -c "print(round(${NUM_EPOCHS}*2.98,1))") h at the measured ~2.98 h/epoch"
echo "  log: ${LOG_FILE}"
echo

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONPATH="mLoRA:${PYTHONPATH:-}" \
setsid nohup \
  python3 -m tinker_cookbook.rl.mlora_train \
    "${ARGS[@]}" \
    --log_path      "./${LOG_DIR}/${RUN_NAME}" \
    --wandb_project "ugttt-rebuttal" \
    --wandb_name    "${RUN_NAME}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
disown || true

echo "pid ${PID}"
echo
echo "VERIFY WITHIN THE FIRST 2 MINUTES — the trainer writes its config record"
echo "before training starts. If seed is not 2, kill it immediately; you are"
echo "burning 18 h on a duplicate of the wave-2 seed-42 arm."
echo
echo "  python3 -c \"import json; c=json.load(open('${LOG_DIR}/${RUN_NAME}/config.json')); print('seed', c['seed'], 'ctx', c['context_window'], 'K', c['num_ensemble_members'], 'rmi', c['rmi_coef'], 'nnm', c['nnm_coef'])\""
echo "     want: seed 2 ctx 8192 K 5 rmi 0.0 nnm 0.075"
echo
echo "Follow:  tail -f ${LOG_FILE}"
echo "Stop:    kill ${PID}    (safe at any epoch boundary — --save_every 1)"
