#!/bin/bash
# Resume the two seed-42 wave4 ahc039 runs killed by the 2026-07-30 root-volume
# overflow. See /opt/dlami/nvme/experiments/RECOVERY_NOTES.md.
#
# Why a separate script instead of launch_wave.sh: ONLY= is a SUBSTRING filter,
# and "ahc039-ugttt" is a prefix of "ahc039-ugttt-seed2", so no ONLY value
# selects just the seed-42 pair — ONLY=ahc039 would relaunch all four and
# clobber the seed-2 runs' state. The flags below reproduce
# COMMON + {UGTTT,BASELINE} + "--save_every 1" from launch_wave.sh exactly, which
# is what logs/aws/wave4/*/config.json records for these two runs.
#
# Resume is automatic and needs no flag: mlora_train.py reads last_epoch.txt,
# sets start_epoch = last_epoch + 1, and calls ensemble.load(log_path,
# start_epoch - 1). Nothing here should pass --seed: both runs took the argparse
# default of 42, matching the published runs.
#
#   bash scripts/aws/resume_wave4_ahc039_seed42.sh check   # verify, launch nothing
#   bash scripts/aws/resume_wave4_ahc039_seed42.sh run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=0
export RAY_ADDRESS=auto

# ── Everything off the 29GB root volume (see launch_wave.sh for the full story)
NVME_SCRATCH="${NVME_SCRATCH:-/opt/dlami/nvme/scratch}"
export ALE_BENCH_LOCAL_TMPDIR="${ALE_BENCH_LOCAL_TMPDIR:-${NVME_SCRATCH}/ale_bench}"
export TMPDIR="${TMPDIR:-${NVME_SCRATCH}/tmp}"
export HF_HOME="${HF_HOME:-/opt/dlami/nvme/hf-cache}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/opt/dlami/nvme/ray}"
# Backstop for any ray.init() that runs without RAY_ADDRESS=auto — those would
# otherwise build a local cluster at Ray's default /tmp/ray on the root volume.
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TEMP_DIR}}"
mkdir -p "$ALE_BENCH_LOCAL_TMPDIR" "$TMPDIR" "$RAY_TEMP_DIR"

PYTHON="${PYTHON:-/opt/dlami/nvme/ugttt-venv/bin/python3}"
RAY_BIN="$(dirname "$PYTHON")/ray"
[[ -x "$RAY_BIN" ]] || RAY_BIN="ray"
TOTAL_CPUS="${TOTAL_CPUS:-$(nproc)}"
GPU_START="${GPU_START:-0}"

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
  --save_every 1
)
UGTTT=(    --num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric true_mi )
BASELINE=( --num_ensemble_members 1 --rmi_coef 0.0 --nnm_coef 0.0 )

declare -a RUNS=(
  "ahc039-ugttt|UGTTT"
  "ahc039-base|BASELINE"
)

WAVE_DIR="logs/aws/wave4"

# Refuse to start if the recorded resume checkpoint is missing or the wrong size
# for ANY ensemble member. A truncated adapter.pt is exactly what the overflow
# produced (ahc039-ugttt/ensemble_0/step_2, 2.6MB vs 61444223B), and torch.load
# on one aborts the run minutes in.
ADAPTER_BYTES=61444223
preflight() {
  local ok=0
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r name arm <<< "$spec"
    local dir="${WAVE_DIR}/${name}"
    local le_file="${dir}/last_epoch.txt"
    if [[ ! -f "$le_file" ]]; then
      echo "  ${name}: NO last_epoch.txt — would start from epoch 0, refusing" >&2
      ok=1; continue
    fi
    local last_epoch start_epoch members
    last_epoch="$(tr -d '[:space:]' < "$le_file")"
    start_epoch=$(( last_epoch + 1 ))
    if [[ "$arm" == "UGTTT" ]]; then members=5; else members=1; fi

    local missing=0 bad=0 m
    for (( m=0; m<members; m++ )); do
      local ck="${dir}/ensemble_${m}/step_${last_epoch}/adapter.pt"
      if [[ ! -f "$ck" ]]; then
        echo "  ${name}: MISSING ${ck}" >&2; missing=$((missing+1)); continue
      fi
      local sz; sz=$(stat -c%s "$ck")
      if [[ "$sz" != "$ADAPTER_BYTES" ]]; then
        echo "  ${name}: TRUNCATED ${ck} (${sz}B, expected ${ADAPTER_BYTES}B)" >&2
        bad=$((bad+1))
      fi
    done
    if (( missing || bad )); then ok=1; continue; fi
    printf '  %-14s last_epoch=%s -> resumes at epoch %s, loads step_%s (%d/%d members OK)\n' \
      "$name" "$last_epoch" "$start_epoch" "$last_epoch" "$members" "$members"
  done
  return $ok
}

ensure_ray_head() {
  if ! "$RAY_BIN" status >/dev/null 2>&1; then
    echo "No Ray head found — starting one (--num-cpus ${TOTAL_CPUS}, temp-dir ${RAY_TEMP_DIR})."
    "$RAY_BIN" start --head --num-cpus="${TOTAL_CPUS}" --disable-usage-stats \
      --temp-dir="${RAY_TEMP_DIR}" >/dev/null
    return
  fi
  if [[ -e /tmp/ray/ray_current_cluster ]]; then
    echo "error: running Ray head is rooted at /tmp (root volume), not ${RAY_TEMP_DIR}." >&2
    echo "       That is what filled the disk. Recycle it first:" >&2
    echo "         ${RAY_BIN} stop --force && rm -rf /tmp/ray" >&2
    exit 1
  fi
  # ale_bench eval runs in Ray WORKERS forked by the raylet, so they inherit the
  # raylet's env — not this script's. A head started without
  # ALE_BENCH_LOCAL_TMPDIR silently writes case scratch to /tmp (observed
  # 2026-07-30, ~270MB/h). Refuse rather than leak.
  local rl
  rl="$(pgrep -f 'raylet/raylet' | head -1)"
  if [[ -n "$rl" ]] && ! grep -qz "ALE_BENCH_LOCAL_TMPDIR=" "/proc/${rl}/environ" 2>/dev/null; then
    echo "error: running Ray head's raylet (pid ${rl}) has no ALE_BENCH_LOCAL_TMPDIR." >&2
    echo "       Its workers will refill the root volume. Recycle it from here:" >&2
    echo "         ${RAY_BIN} stop --force" >&2
    exit 1
  fi
}

MODE="${1:-}"
[[ "$MODE" == "check" || "$MODE" == "run" ]] || {
  echo "usage: $0 <check|run>" >&2; exit 1; }

echo "repo:        $REPO_ROOT"
echo "python:      $PYTHON"
echo "ale_bench scratch: $ALE_BENCH_LOCAL_TMPDIR"
echo "TMPDIR:      $TMPDIR"
echo "ray temp:    $RAY_TEMP_DIR"
echo "root volume: $(df -h --output=avail / | tail -1 | tr -d ' ') available"
echo
echo "Resume preflight:"
preflight || { echo; echo "preflight FAILED — nothing launched." >&2; exit 1; }
echo

if [[ "$MODE" == "check" ]]; then
  echo "check mode — nothing launched."
  exit 0
fi

ensure_ray_head

gpu=$GPU_START
for spec in "${RUNS[@]}"; do
  IFS='|' read -r name arm <<< "$spec"
  declare -n arm_flags="$arm"
  ts="$(date +%Y%m%d_%H%M%S)"

  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" \
  setsid \
    "$PYTHON" -m tinker_cookbook.rl.mlora_train \
      "${COMMON[@]}" \
      "${arm_flags[@]}" \
      --env ahc039 \
      --problem_idx ahc039 \
      --log_path "./${WAVE_DIR}/${name}" \
      --wandb_project "ugttt-rebuttal" \
      --wandb_name "$name" \
    > "${WAVE_DIR}/${name}.resume_${ts}.log" 2>&1 &

  echo "  GPU ${gpu}  ${name}  (pid $!)  -> ${WAVE_DIR}/${name}.resume_${ts}.log"
  gpu=$(( gpu + 1 ))
done

echo
echo "Resumed. The prior stdout logs are preserved; resumes write *.resume_<ts>.log."
echo "Confirm each log shows 'Resumed from epoch N' before walking away."
echo "Watch the root volume — it must stay flat now:  watch -n60 'df -h /'"
