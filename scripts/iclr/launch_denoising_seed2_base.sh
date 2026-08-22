#!/bin/bash
# ============================================================================
# ICLR resubmission rerun — denoising BASELINE arm, seed 2.
# Mirror of ugttt-vm-export/vm-export/runs/denoising-baseline (seed 42):
# ONLY the seed changes.
#
#   bash scripts/iclr/launch_denoising_seed2_base.sh check   # preflight, ~30 s
#   bash scripts/iclr/launch_denoising_seed2_base.sh run     # preflight + launch
#
# WHY THIS RUN
#   The seed-42 denoising pair gave UG-TTT +0.4469 R_max (5.1409 vs 4.6940),
#   ~100x the paper's per-domain deltas, on n=1 seed in a domain whose reward
#   is gated by a hard Poisson constraint — a single lucky rollout can produce
#   that margin. This pair is the second seed. Launch BOTH arms (this script on
#   GPU 0, launch_denoising_seed2_ugttt.sh on GPU 1) so the pair shares a box
#   and a code state, like the seed-42 pair did.
#
# EVERY FLAG BELOW IS COPIED FIELD-FOR-FIELD from
#   ugttt-vm-export/vm-export/runs/denoising-baseline/config.json
# (the trainer's own record of what executed), EXCEPT --seed 2. Flags whose
# value already equals the argparse default are still listed explicitly so this
# file is a complete record of the run. config.json fields with no CLI flag
# (adv_estimator_beta=2.0, lora_dropout=0.05, target_modules q/k/v/o) are
# Config-class defaults and cannot drift via the CLI.
#
# CODE STATE REQUIRED
#   The seed-42 pair ran on trainer base commit 1eb925f plus
#   vm-export/code/ugttt-fixes.patch (cutoff_len bound to context_window; judge
#   worker cap; scratch-dir reclaim). Launch from the same checkout that
#   produced the seed-42 pair, or one already carrying those fixes — otherwise
#   the seed-2 numbers are not a seed-only mirror.
#
# HOST REQUIREMENTS (vm-export/README.md "Host-level requirements")
#   Bio venv at /opt/dlami/nvme/venv-denoise — openproblems v0.8.0 (NOT
#   upstream main) + scanpy + MCV, imported IN-PROCESS by the verifier — plus
#   the np.int -> int patch in openproblems denoising datasets/utils.py.
#   Build it with scripts/aws/setup_denoising_venv.sh if missing.
#
# RUNTIME: seed-42 baseline arm took 15.0 h for 6 epochs on one RTX PRO 6000.
#   --save_every 1 and per-epoch logging: SAFE TO KILL at any epoch boundary.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

RUN_NAME="${RUN_NAME:-denoising-seed2-baseline}"
GPU="${GPU:-0}"
RAY_PORT="${RAY_PORT:-6402}"        # ugttt sibling defaults to 6403 — keep distinct
SEED=2                              # <-- THE ONLY CHANGE vs the seed-42 run
PY="${VENV_PY:-/opt/dlami/nvme/venv-denoise/bin/python}"
LOG_DIR="logs/iclr"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

# ── environment (mirrors vm-export runs_rtx6000/run_one.sh) ────────────────
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/hf}"
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/opt/dlami/nvme/scratch/tmp}"
# expandable_segments stays OFF — see runs_a100/run_one.sh bisection notes.
unset PYTORCH_CUDA_ALLOC_CONF
# Private Ray head per run: the reward path creates a detached actor named
# `cpu_scheduler`; a shared head would let the two arms bind each other's actor.
export RAY_ADDRESS="127.0.0.1:${RAY_PORT}"
export RAY_DEDUP_LOGS=0
RAY_TMP="${RAY_TMP_ROOT:-/opt/dlami/nvme/raytmp}/ray_${RUN_NAME}"
CPUS_PER_RUN="${CPUS_PER_RUN:-$(( $(nproc) / 2 ))}"
[ "$CPUS_PER_RUN" -lt 2 ] && CPUS_PER_RUN=2
RAY_BIN="$(dirname "$PY")/ray"

# ---------------------------------------------------------------- preflight
fail() { echo "PREFLIGHT FAIL: $*" >&2; exit 1; }
ok()   { echo "  ok    $*"; }

# A GCS whose raylet has died still answers `ray status`; require a registered
# raylet (CPU resources present), not merely a responding GCS. Otherwise the
# failure surfaces hours in, at the first reward call, as `RPC error:
# Deadline Exceeded`.
ray_head_healthy() {
  timeout 25 "$RAY_BIN" status 2>/dev/null | grep -qE "[0-9.]+/[0-9.]+ CPU"
}
ray_head_healthy_within() {
  local deadline=$(( SECONDS + ${1:-90} ))
  while [ "$SECONDS" -lt "$deadline" ]; do
    ray_head_healthy && return 0
    sleep 3
  done
  return 1
}

preflight() {
  echo "Preflight (${RUN_NAME})"

  # ml_log.py silently no-ops wandb when this is unset.
  [[ -n "${WANDB_API_KEY:-}" ]] || fail "WANDB_API_KEY unset. ml_log.py skips wandb SILENTLY; metrics would only exist in metrics.jsonl."
  ok "WANDB_API_KEY set"
  [[ -n "${WANDB_ENTITY:-}" ]] || echo "  warn  WANDB_ENTITY unset (fine if your default entity is right)"

  ! pgrep -f "wandb_name ${RUN_NAME}" >/dev/null 2>&1 \
    || fail "a process with wandb_name ${RUN_NAME} is already running"
  ok "no duplicate ${RUN_NAME} process"

  [[ -x "$PY" ]] || fail "no interpreter at $PY — build it with scripts/aws/setup_denoising_venv.sh (bio stack must be importable IN-PROCESS)"
  ok "interpreter $PY"
  "$PY" -c "import torch" 2>/dev/null || fail "torch not importable in $PY"
  ok "torch importable"
  "$PY" -c "import openproblems, scanpy" 2>/dev/null \
    || fail "openproblems/scanpy not importable in $PY — the denoising verifier is imported in-process; this venv must carry the bio stack (openproblems v0.8.0, np.int patch)"
  ok "openproblems + scanpy importable (denoising verifier deps)"

  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  local ngpu; ngpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  [[ "$GPU" -lt "$ngpu" ]] || fail "GPU=$GPU but only $ngpu GPUs present"
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [[ "$used" -lt 2000 ]] || fail "GPU $GPU already has ${used} MiB in use — pick a free GPU with GPU=<n>"
  ok "GPU $GPU free (${used} MiB used, ${ngpu} GPUs on box)"

  [[ -d mLoRA ]] || fail "mLoRA/ missing — PYTHONPATH would not resolve"
  ok "mLoRA present"

  # Guard against a pre-fix checkout: without the context_window flag (and the
  # cutoff_len binding shipped with it) prompts are truncated at 4096 tokens and
  # rollouts score 0.0 silently.
  grep -q 'parser.add_argument("--context_window"' tinker_cookbook/rl/mlora_train.py \
    || fail "--context_window not in argparse — this checkout predates the cutoff_len fix (1eb925f + ugttt-fixes.patch); rollouts would silently score 0.0"
  ok "--context_window wired (cutoff_len fix era checkout)"

  local freegb; freegb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  [[ "$freegb" -ge 40 ]] || fail "only ${freegb} GB free; --save_every 1 writes a checkpoint per epoch"
  ok "${freegb} GB disk free"

  mkdir -p "$TMPDIR" "$HF_HOME"
  if ! ray_head_healthy; then
    for pid in $(ss -ltnp 2>/dev/null | grep -E "[:.]${RAY_PORT}[[:space:]]" \
                 | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
      echo "  ..    killing stale Ray process $pid on :${RAY_PORT}"
      kill -9 "$pid" 2>/dev/null || true
    done
    rm -rf "$RAY_TMP"; mkdir -p "$RAY_TMP"
    echo "  ..    starting private Ray head on :${RAY_PORT} (${CPUS_PER_RUN} cpus)"
    "$RAY_BIN" start --head --port="$RAY_PORT" --num-cpus="$CPUS_PER_RUN" \
        --temp-dir="$RAY_TMP" --include-dashboard=false --disable-usage-stats >/dev/null
    ray_head_healthy_within 90 || fail "Ray head on :${RAY_PORT} never registered a raylet"
  fi
  ok "private Ray head healthy on :${RAY_PORT}"

  echo "Preflight passed."
  echo
}

# --------------------------------------------------------------------- flags
# Copied field-for-field from vm-export/runs/denoising-baseline/config.json.
ARGS=(
  --base_model            "Qwen/Qwen3-8B"
  --precision             fp16
  --gpus                  1
  --env                   denoising
  --problem_idx           improvement

  --seed                  "${SEED}"        # seed-42 run: 42. THE ONLY CHANGE.

  # --- arm: baseline (K=1, no exploration bonus, no NNM) ---
  --num_ensemble_members  1
  --rmi_coef              0.0
  --nnm_coef              0.0
  --uncertainty_metric    true_mi          # inert at rmi_coef 0; matches config record
  --gamma_max_ratio       10.0             # default coupling, as recorded

  # --- identical to the seed-42 pair ---
  --adv_estimator         entropic_adaptive_beta
  --kl_penalty_coef       0.01
  --lora_rank             16
  --lora_alpha            32
  --learning_rate         4e-5
  --group_size            8
  --groups_per_batch      8
  --num_epochs            6
  --max_tokens            260000
  --temperature           1.0
  --budget_s              1000
  --two_phase_sampling
  --phase1_max_tokens     26000
  --context_window        32768            # keep 32768 — do NOT copy wave-2's 8192
  --eval_timeout          1100
  --dataset_timeout       1200
  --num_cpus_per_task     2
  --sampler_type          puct_backprop
  --initial_exp_type      random
  --eval_every            3

  # --- run control ---
  --save_every            1                # killable at any epoch boundary
)
# Streaming MI is deliberately NOT passed: the seed-42 pair had
# streaming_mi_enabled=false (no paired calibration exists for denoising).

case "${1:-}" in
  check) preflight; echo "Nothing launched. Re-run with: bash $0 run"; exit 0 ;;
  run)   ;;
  *)     echo "usage: $0 <check|run>"; exit 1 ;;
esac

preflight
mkdir -p "$LOG_DIR"

echo "Launching ${RUN_NAME} on GPU ${GPU} (Ray :${RAY_PORT})"
echo "  seed=${SEED}  K=1  rmi_coef=0.0  nnm_coef=0.0  ctx=32768  epochs=6"
echo "  ETA ~15 h (seed-42 baseline arm measured 15.0 h on one RTX PRO 6000)"
echo "  log: ${LOG_FILE}"
echo

PYTHONPATH="mLoRA:${PYTHONPATH:-}" \
setsid nohup \
  "$PY" -m tinker_cookbook.rl.mlora_train \
    "${ARGS[@]}" \
    --log_path      "./${LOG_DIR}/${RUN_NAME}" \
    --wandb_project "ttt-discover-uncertainty" \
    --wandb_name    "${RUN_NAME}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!
disown || true

echo "pid ${PID}"
echo
echo "VERIFY WITHIN THE FIRST 2 MINUTES — the trainer writes its config record"
echo "before training starts. If seed is not 2 or context_window is not 32768,"
echo "kill it immediately; you are burning 15 h on a duplicate of seed 42."
echo
echo "  python3 -c \"import json; c=json.load(open('${LOG_DIR}/${RUN_NAME}/config.json')); print('seed', c['seed'], 'ctx', c['context_window'], 'K', c['num_ensemble_members'], 'rmi', c['rmi_coef'])\""
echo "     want: seed 2 ctx 32768 K 1 rmi 0.0"
echo
echo "Follow:  tail -f ${LOG_FILE}"
echo "Stop:    kill ${PID}    (safe at any epoch boundary — --save_every 1)"
