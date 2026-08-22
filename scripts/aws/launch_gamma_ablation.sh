#!/bin/bash
# ============================================================================
# beta-gamma coupling ablation (Remark 1 / Eq. 5) — Reviewer vGzb W2 + Q3.
#
#   bash scripts/aws/launch_gamma_ablation.sh check    # preflight only, ~10 s
#   bash scripts/aws/launch_gamma_ablation.sh run      # preflight + launch
#
# WHAT THIS TESTS
#   The paper claims exploration pressure should RISE with policy sharpness
#   (gamma_eff coupled to beta, Eq. 5). No run isolates that. This arm pins
#   gamma_eff to a CONSTANT while matching the coupled run's average strength,
#   so the two arms differ in SCHEDULE only.
#
#     coupled   (cp26-nostream): rmi_coef 0.10, gamma_max_ratio 10.0
#                                -> realised gamma_eff 0.459 0.268 0.333 0.413 (ep 0-3)
#     decoupled (this run):      rmi_coef 0.37, gamma_max_ratio 1.0
#                                -> gamma_eff pinned at 0.37 (beta >> beta_ref=2)
#
#   0.37 is the coupled run's realised mean gamma_eff over epochs 0-3 (0.3684).
#   Matching the mean is the whole point: a naive "coupling off" arm would just
#   explore 3.7x less and prove nothing.
#
# COMPARISON
#   Against the FIRST 4 EPOCHS (256 rollouts) of node2/wave2/cp26-nostream.
#   Budget-matching by truncation, the analysis pipeline's default convention.
#
# EVERY FLAG BELOW IS COPIED FROM cp26-nostream/config.json, NOT from
# launch_wave.sh. They disagree: the wave script passes --max_tokens 260000 /
# --phase1_max_tokens 26000, the run actually used 16000 / 6000 with
# --context_window 8192. Do not "fix" these to match the wave script.
#
# RUNTIME: ~2.98 h/epoch measured -> 4 epochs ~= 11.9 h.
#   --save_every 1 and per-epoch logging mean this run is SAFE TO KILL at any
#   epoch boundary. Even 3 epochs is a usable comparison. Do not restart it to
#   "get all 4" if you are short on time.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

RUN_NAME="${RUN_NAME:-cp26-gamma-fixed}"
GPU="${GPU:-0}"
NUM_EPOCHS="${NUM_EPOCHS:-4}"
RMI_COEF="${RMI_COEF:-0.37}"
LOG_DIR="logs/aws/gamma_ablation"
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
  echo "Preflight"

  # ml_log.py silently no-ops wandb when this is unset — you would lose the
  # gamma_eff series that is the entire point of the run.
  [[ -n "${WANDB_API_KEY:-}" ]] || fail "WANDB_API_KEY unset. ml_log.py skips wandb SILENTLY; gamma_eff would only exist in metrics.jsonl."
  ok "WANDB_API_KEY set"

  [[ -n "${WANDB_ENTITY:-}" ]] || echo "  warn  WANDB_ENTITY unset (fine if your default entity is right)"

  command -v nvidia-smi >/dev/null || fail "nvidia-smi not found"
  local ngpu; ngpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  [[ "$GPU" -lt "$ngpu" ]] || fail "GPU=$GPU but only $ngpu GPUs present"
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [[ "$used" -lt 2000 ]] || fail "GPU $GPU already has ${used} MiB in use — pick a free GPU with GPU=<n>"
  ok "GPU $GPU free (${used} MiB used, ${ngpu} GPUs on box)"

  # The flag this whole run depends on. If the patch is missing the run would
  # silently execute the COUPLED config and waste 12 h.
  grep -q 'parser.add_argument("--gamma_max_ratio"' tinker_cookbook/rl/mlora_train.py \
    || fail "--gamma_max_ratio not in argparse. This checkout predates the patch; the run would silently be COUPLED."
  grep -q 'gamma_max_ratio=args.gamma_max_ratio,' tinker_cookbook/rl/mlora_train.py \
    || fail "gamma_max_ratio parsed but never reaches Config. The run would silently be COUPLED."
  ok "--gamma_max_ratio wired to Config"

  if [[ -f tests/test_gamma_coupling.py ]]; then
    python3 tests/test_gamma_coupling.py >/dev/null 2>&1 \
      || fail "tests/test_gamma_coupling.py failed. Run it directly to see why."
    ok "tests/test_gamma_coupling.py passes"
  fi

  python3 -c "import torch" 2>/dev/null || fail "torch not importable in this python3"
  ok "torch importable"

  [[ -d mLoRA ]] || fail "mLoRA/ missing — PYTHONPATH would not resolve"
  ok "mLoRA present"

  local freegb; freegb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  [[ "$freegb" -ge 40 ]] || fail "only ${freegb} GB free; --save_every 1 writes a checkpoint per epoch"
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
# Copied field-for-field from node2/wave2/cp26-nostream/config.json.
# Flags whose value already equals the argparse default are still listed
# explicitly, so this file is a complete record of the run.
ARGS=(
  --base_model            "Qwen/Qwen3-8B"
  --precision             fp16
  --env                   cp
  --problem_idx           26
  --seed                  42

  # --- the ablation ---
  --num_ensemble_members  5
  --uncertainty_metric    true_mi
  --nnm_coef              0.075
  --rmi_coef              "${RMI_COEF}"     # coupled arm used 0.10
  --gamma_max_ratio       1.0               # coupled arm used 10.0  <-- THE ONLY MECHANISM CHANGE

  # --- identical to the coupled run ---
  --adv_estimator         entropic_adaptive_beta
  --kl_penalty_coef       0.01
  --lora_rank             16
  --lora_alpha            32
  --learning_rate         4e-5
  --group_size            8
  --groups_per_batch      8
  --max_tokens            16000
  --two_phase_sampling
  --phase1_max_tokens     6000
  --context_window        8192
  --temperature           1.0
  --sampler_type          puct_backprop
  --initial_exp_type      random
  --num_cpus_per_task     2
  --eval_every            3

  # --- run control ---
  --num_epochs            "${NUM_EPOCHS}"   # 4, not 6: fits a 12 h slot
  --save_every            1                 # killable at any epoch boundary
)
# Streaming is OFF: cp26-nostream has streaming_mi_enabled=False, so
# --streaming_mi is deliberately NOT passed.

case "${1:-}" in
  check) preflight; echo "Nothing launched. Re-run with: bash $0 run"; exit 0 ;;
  run)   ;;
  *)     echo "usage: $0 <check|run>"; exit 1 ;;
esac

preflight
mkdir -p "$LOG_DIR"

echo "Launching ${RUN_NAME} on GPU ${GPU}"
echo "  rmi_coef=${RMI_COEF}  gamma_max_ratio=1.0  epochs=${NUM_EPOCHS}"
echo "  ETA ~$(python3 -c "print(round(${NUM_EPOCHS}*2.98,1))") h at the measured 2.98 h/epoch"
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
echo "VERIFY WITHIN THE FIRST 2 MINUTES — the banner prints before training starts."
echo "If it does not say DECOUPLED, kill it immediately; you are burning 12 h on a duplicate."
echo
echo "  grep -m1 'Exploration coefficient' ${LOG_FILE}"
echo "     want: DECOUPLED (gamma_max_ratio=1.0) - gamma_eff pinned at ${RMI_COEF}"
echo
echo "AFTER EPOCH 0 (~3 h) confirm the schedule really is flat:"
echo "  python3 -c \"import json;print([round(json.loads(l)['train/gamma_eff/mean'],4) for l in open('${LOG_DIR}/${RUN_NAME}/metrics.jsonl') if 'train/gamma_eff/mean' in l])\""
echo "     want: ~${RMI_COEF} every epoch"
echo "     coupled cp26-nostream was: 0.4593, 0.2684, 0.3332, 0.4125"
echo
echo "Follow:  tail -f ${LOG_FILE}"
echo "Stop:    kill ${PID}    (safe at any epoch boundary — --save_every 1)"
