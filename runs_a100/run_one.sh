#!/bin/bash
# One UG-TTT run pinned to its own pair of GPUs, with its own private Ray head.
#
# Driven entirely by env vars so launch_all.sh can start four of these side by
# side on an 8x A100-40GB node without them colliding. Not meant to be run bare;
# see launch_all.sh (though `RUN_NAME=... ENV_NAME=... ./run_one.sh` works).
#
# Required: RUN_NAME ENV_NAME ARM GPU_IDS RAY_PORT
# Optional: PROBLEM_IDX BASE_MODEL PHASE1_MAX_TOKENS NUM_EPOCHS GROUP_SIZE
set -euo pipefail

: "${RUN_NAME:?}" ; : "${ENV_NAME:?}" ; : "${ARM:?}"
: "${GPU_IDS:?}"  ; : "${RAY_PORT:?}"
: "${WANDB_ENTITY:?WANDB_ENTITY must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
PROBLEM_IDX="${PROBLEM_IDX:-improvement}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
GROUP_SIZE="${GROUP_SIZE:-8}"
# Observed on the 32B cp26 run: rollouts used 10k-16k of a 25k budget and always
# stopped on EOS, so the cap was never the binding constraint. 26000 is kept to
# stay comparable with the existing ac1 baselines; lower it here to trade a
# little headroom for wall-clock.
PHASE1_MAX_TOKENS="${PHASE1_MAX_TOKENS:-26000}"

# ── Interpreter: this repo has been seen with either venv name ──────────────
if   [ -x "$REPO_DIR/.venv/bin/python3"     ]; then PY="$REPO_DIR/.venv/bin/python3"
elif [ -x "$REPO_DIR/discovery/bin/python3" ]; then PY="$REPO_DIR/discovery/bin/python3"
else echo "FATAL: no .venv or discovery venv in $REPO_DIR" >&2; exit 1; fi
RAY_BIN="$(dirname "$PY")/ray"

# ── GPU + cache placement ──────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPU_IDS"     # run sees exactly 2 -> cuda:0, cuda:1
export HF_HOME="${HF_HOME:-$REPO_DIR/.hf_cache}"
mkdir -p "$HF_HOME"

# NOTE: expandable_segments stays OFF for every layer-sharded (--gpus 2) run.
# It allocates through the CUDA virtual-memory APIs and those mappings do not
# survive the cross-device hops the shard path makes: the SECOND forward of a
# batch faults with `illegal memory access` / CUBLAS_STATUS_INTERNAL_ERROR,
# reported at a random later op. Confirmed by bisection on the 32B cp26 run
# (RTX PRO 6000, sm_120): with it, ensemble scoring failed 3/3 at every
# sequence length; without it, passed at the production shape. Root cause is
# the allocator + peer mapping, not the GPU model, so it is left off here too.
unset PYTORCH_CUDA_ALLOC_CONF
export TOKENIZERS_PARALLELISM=false

# ── Private Ray head per run ───────────────────────────────────────────────
# Four concurrent runs must NOT share one head: the reward path creates a
# detached actor named `cpu_scheduler`, and a shared head would let the runs
# bind to each other's actor. Separate port + temp dir keeps them isolated.
CPUS_PER_RUN="${CPUS_PER_RUN:-$(( $(nproc) / 4 ))}"
[ "$CPUS_PER_RUN" -lt 2 ] && CPUS_PER_RUN=2
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
RAY_TMP="${RAY_TMP_ROOT:-/tmp}/ray_${RUN_NAME}"
mkdir -p "$RAY_TMP"
export RAY_DEDUP_LOGS=0

if ! "$RAY_BIN" status >/dev/null 2>&1; then
  echo "[$RUN_NAME] starting private Ray head on :${RAY_PORT} (${CPUS_PER_RUN} cpus)"
  "$RAY_BIN" start --head \
      --port="$RAY_PORT" \
      --num-cpus="$CPUS_PER_RUN" \
      --temp-dir="$RAY_TMP" \
      --include-dashboard=false \
      --disable-usage-stats >/dev/null
fi

# ── Arm: baseline (no ensemble / no uncertainty) vs UG-TTT ─────────────────
case "$ARM" in
  baseline)
    ARM_ARGS=(--num_ensemble_members 1 --rmi_coef 0.0 --nnm_coef 0.0)
    ;;
  ugttt)
    ARM_ARGS=(--num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075
              --uncertainty_metric true_mi)
    ;;
  *) echo "FATAL: ARM must be 'baseline' or 'ugttt', got '$ARM'" >&2; exit 1 ;;
esac

LOG_DIR="./logs/${RUN_NAME}"
mkdir -p "$LOG_DIR"

echo "[$RUN_NAME] env=$ENV_NAME arm=$ARM gpus=$GPU_IDS model=$BASE_MODEL save_every=1"

PYTHONPATH="mLoRA:${PYTHONPATH:-}" "$PY" -m tinker_cookbook.rl.mlora_train \
    --env "$ENV_NAME" \
    --problem_idx "$PROBLEM_IDX" \
    --base_model "$BASE_MODEL" \
    --precision fp16 \
    --gpus 2 \
    "${ARM_ARGS[@]}" \
    --adv_estimator entropic_adaptive_beta \
    --kl_penalty_coef 0.01 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 4e-5 \
    --group_size "$GROUP_SIZE" \
    --groups_per_batch 8 \
    --num_epochs "$NUM_EPOCHS" \
    --max_tokens 260000 \
    --temperature 1.0 \
    --budget_s 1000 \
    --num_cpus_per_task 2 \
    --eval_timeout 1100 \
    --dataset_timeout 1200 \
    --sampler_type puct_backprop \
    --initial_exp_type random \
    --two_phase_sampling \
    --phase1_max_tokens "$PHASE1_MAX_TOKENS" \
    --wandb_project "ttt-discover-uncertainty" \
    --wandb_name "$RUN_NAME" \
    --log_path "$LOG_DIR" \
    --save_every 1
