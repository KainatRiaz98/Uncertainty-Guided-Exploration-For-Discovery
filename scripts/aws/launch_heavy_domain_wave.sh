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
# requirements-math.txt).
#
# Per-domain infra — the two domains are NOT symmetric:
#   ahc039    needs NOTHING beyond requirements-ahc.txt on this branch. The
#             runbook's "install Docker + docker pull yimjk/ale-bench" is stale:
#             ale_bench/utils.py:487 docker_client() is a host-side Ray mock
#             ("deprecated - kept for compatibility but uses ray instead") that
#             strips the /bin/sh -c wrapper and runs the compile on the host.
#             No daemon is involved. Inputs and judges are vendored in-repo.
#   denoising needs the bio/openproblems stack IN THE TRAINING VENV — the
#             verifier is imported in-process by the trainer
#             (mlora_train.py:2426), so a side venv is never seen. See
#             requirements/denoising/README.md.
#
# Because denoising's install mutates the venv (NumPy 2.x vs torch), give it a
# CLONE of the training venv and point these runs at it with PYTHON=. Otherwise
# a numpy swap can break already-launched ahc039 runs when they auto-resume
# after the ~24h instance reboot.
#
# SMOKE-TEST ONE RUN PER DOMAIN FIRST. This exact combination — nf4 + a new
# model family + single-phase + two brand-new domains — has no prior run
# anywhere. Confirm real reward + non-garbled completions before filling the
# node. ahc039 has no setup, so launch it without waiting on denoising:
#   DOMAIN=ahc039 bash scripts/aws/launch_heavy_domain_wave.sh smoke
#   DOMAIN=ahc039 bash scripts/aws/launch_heavy_domain_wave.sh run
# Then, once denoising's deps are installed in a cloned venv:
#   DOMAIN=denoising GPU_START=4 PYTHON=~/venv-denoise/bin/python \
#     bash scripts/aws/launch_heavy_domain_wave.sh smoke
#   DOMAIN=denoising GPU_START=4 PYTHON=~/venv-denoise/bin/python \
#     bash scripts/aws/launch_heavy_domain_wave.sh run
#
# If denoising's setup is still fighting you, do NOT leave 4 GPUs idle — this
# takes ahc039 to n=3 per cell, which answers vGzb's one-random-seed weakness
# on the new domain:
#   GPU_START=4 bash scripts/aws/launch_heavy_domain_wave.sh ahc-extra
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
# Launch one domain at a time so the domain with no setup (ahc039) never waits
# on the one that does (denoising): DOMAIN=ahc039|denoising|all.
DOMAIN="${DOMAIN:-all}"
# First GPU index to use. Set this on the SECOND launch so it does not land on
# the GPUs the first launch already took (e.g. GPU_START=4).
GPU_START="${GPU_START:-0}"
# Interpreter for these runs. denoising needs its own cloned venv; ahc039 uses
# the normal training venv.
PYTHON="${PYTHON:-python3}"

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
  # Checkpoint EVERY epoch (not every 2). Instances reboot on a ~24h cycle, so
  # a resume should lose at most one epoch of compute. Checkpoints accumulate in
  # per-step dirs rather than overwriting, but LoRA adapters (rank 16, q/k/v/o)
  # are small, so the extra disk is cheap insurance.
  --save_every 1
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

# Fallback for the 4 GPUs denoising would have used, if its bio stack does not
# come up in time. Takes ahc039 to 3 seeds per arm (2,3 here + 4,5) so the
# new-domain result carries a spread instead of being n=1 like the paper's.
declare -a AHC_EXTRA=(
  "ahc039-ugttt-s4|ahc039|ahc039|UGTTT|--seed 4"
  "ahc039-ugttt-s5|ahc039|ahc039|UGTTT|--seed 5"
  "ahc039-base-s4|ahc039|ahc039|BASELINE|--seed 4"
  "ahc039-base-s5|ahc039|ahc039|BASELINE|--seed 5"
)

usage() {
  echo "usage: $0 [--list] <smoke|run|ahc-extra>"
  echo "  env: DOMAIN=ahc039|denoising|all  GPU_START=<n>  PYTHON=<interpreter>"
  exit 1
}

list_runs() {
  local -n arr=$1
  printf '%-20s %-11s %-12s %-9s %s\n' RUN ENV PROBLEM ARM EXTRA
  for spec in "${arr[@]}"; do
    IFS='|' read -r name env pidx arm extra <<< "$spec"
    # --list must show exactly what a launch with these env vars would start,
    # since the runbook says to always --list first.
    if [[ "$DOMAIN" == "all" || "$env" == "$DOMAIN" ]]; then
      printf '%-20s %-11s %-12s %-9s %s\n' "$name" "$env" "$pidx" "$arm" "$extra"
    fi
  done
}

[[ $# -ge 1 ]] || usage

if [[ "$1" == "--list" ]]; then
  echo "model: $MODEL   precision: nf4   two_phase_sampling: off"
  echo "domain filter: $DOMAIN   first gpu: $GPU_START   python: $PYTHON"
  echo; echo "== smoke (run first) =="; list_runs SMOKE
  echo; echo "== run (all 8, after smoke passes) =="; list_runs RUNS
  echo; echo "== ahc-extra (only if denoising is not ready) =="; list_runs AHC_EXTRA
  exit 0
fi

MODE="$1"
case "$MODE" in
  smoke)     SELECTED=("${SMOKE[@]}") ;;
  run)       SELECTED=("${RUNS[@]}") ;;
  ahc-extra) SELECTED=("${AHC_EXTRA[@]}") ;;
  *) usage ;;
esac

# Keep only the requested domain. Written as a full if/fi (not `[[ ]] && ...`)
# because under `set -e` a false test as the last statement in a loop body
# would exit the script.
if [[ "$DOMAIN" != "all" ]]; then
  declare -a FILTERED=()
  for spec in "${SELECTED[@]}"; do
    IFS='|' read -r _ spec_env _ _ _ <<< "$spec"
    if [[ "$spec_env" == "$DOMAIN" ]]; then
      FILTERED+=( "$spec" )
    fi
  done
  # Only expand FILTERED once it is known non-empty: "${FILTERED[@]-}" on an
  # empty array yields a single empty element, which would pass a count check
  # and then launch a garbage run.
  if [[ ${#FILTERED[@]} -eq 0 ]]; then
    echo "error: no runs match DOMAIN=$DOMAIN in mode $MODE" >&2; exit 1
  fi
  SELECTED=("${FILTERED[@]}")
fi

[[ $(( GPU_START + ${#SELECTED[@]} )) -le $NUM_GPUS ]] || {
  echo "error: ${#SELECTED[@]} runs starting at GPU $GPU_START exceeds $NUM_GPUS GPUs" >&2; exit 1; }

ensure_ray_head

LOG_DIR="logs/aws/heavy_domain_${MODE}"
mkdir -p "$LOG_DIR"

echo "Launching ${#SELECTED[@]} runs — model: $MODEL (nf4, 1 GPU/run)"
echo "Domain: $DOMAIN   GPUs: ${GPU_START}..$(( GPU_START + ${#SELECTED[@]} - 1 ))   Python: $PYTHON"
echo "Logs: $LOG_DIR"
echo

gpu=$GPU_START
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
    "$PYTHON" -m tinker_cookbook.rl.mlora_train \
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
