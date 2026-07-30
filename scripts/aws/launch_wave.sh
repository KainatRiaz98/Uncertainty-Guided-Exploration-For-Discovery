#!/bin/bash
# Launch one wave of UG-TTT runs on a single 8-GPU g7e.48xlarge.
#
# Targets the UG-TTT implementation on this branch (module
# tinker_cookbook.rl.mlora_train). Run this from a checkout of THIS repo.
# See docs/aws_g7e48xlarge.md.
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
# ONE shared Ray head per node. The task layer hardcodes ray.init("auto")
# (tasks/*/task.py, utils/cpu_scheduler.py) and base_reward_task.py get-or-creates
# a detached, host-keyed "cpu_scheduler" actor that partitions this node's CPUs
# across all co-resident runs. So every run must join the SAME head (matching
# scripts/run.sh, which also sets RAY_ADDRESS=auto) and must NOT be taskset-pinned
# — the scheduler does the CPU partitioning; taskset would fight it.
export RAY_ADDRESS=auto

# Keep ALL scratch off the 29GB root volume. On 2026-07-30 a wave died when
# root hit 0 bytes: ale_bench leaks one ~128KB dir per evaluated case (82,283 of
# them, ~10.6GB) and Ray spills objects + writes session logs under /tmp/ray.
# The first casualty was a torch.save of an adapter, which left a truncated
# checkpoint. NVME_SCRATCH has 6.5TB; point everything there.
NVME_SCRATCH="${NVME_SCRATCH:-/opt/dlami/nvme/scratch}"
# Consumed by ale_bench/tool_wrappers/case_runner.py:get_local_scratch_root.
export ALE_BENCH_LOCAL_TMPDIR="${ALE_BENCH_LOCAL_TMPDIR:-${NVME_SCRATCH}/ale_bench}"
# Catch-all for tempfile/mkdtemp elsewhere (compile dirs, HF, C++ builds).
export TMPDIR="${TMPDIR:-${NVME_SCRATCH}/tmp}"
# Ray session dir: holds worker logs AND the object-spilling directory.
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/opt/dlami/nvme/ray}"
# Backstop: --temp-dir only covers the head we start. Any code path that calls
# ray.init() WITHOUT RAY_ADDRESS=auto spins up its own local cluster at Ray's
# default /tmp/ray; RAY_TMPDIR redirects those too. Verified 2026-07-30.
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TEMP_DIR}}"
mkdir -p "$ALE_BENCH_LOCAL_TMPDIR" "$TMPDIR" "$RAY_TEMP_DIR"

NUM_GPUS="${NUM_GPUS:-8}"
TOTAL_CPUS="${TOTAL_CPUS:-$(nproc)}"

# First GPU index to use. Set this when part of the node is already busy so a
# launch does not land on GPUs another wave already took (e.g. GPU_START=2).
GPU_START="${GPU_START:-0}"
# Substring filter on the run name: launch only the matching subset of a wave.
# Lets a wave be split across nodes/times (e.g. ONLY=ahc039 for just that pair).
ONLY="${ONLY:-}"
# Interpreter for these runs. Must be the training venv — a bare `python3` is
# /usr/bin/python3 here, which has none of the deps.
PYTHON="${PYTHON:-python3}"
# `ray` lives next to the interpreter inside a venv, and is NOT on PATH.
RAY_BIN="$(dirname "$PYTHON")/ray"
[[ -x "$RAY_BIN" ]] || RAY_BIN="ray"

# Ensure a Ray head exists on this node, sized to the whole box so the
# cpu_scheduler pool covers all cores. Idempotent: skip if one is already up.
# Defined here, invoked only on a real launch (not on --list/usage).
ensure_ray_head() {
  if ! "$RAY_BIN" status >/dev/null 2>&1; then
    echo "No Ray head found — starting one (--num-cpus ${TOTAL_CPUS}, temp-dir ${RAY_TEMP_DIR})."
    "$RAY_BIN" start --head --num-cpus="${TOTAL_CPUS}" --disable-usage-stats \
      --temp-dir="${RAY_TEMP_DIR}" >/dev/null
    return
  fi
  # A head is already up. If it was started WITHOUT --temp-dir it is spilling
  # objects and writing worker logs to /tmp on the root volume — the exact
  # condition that filled the disk and killed wave4. Refuse to pile onto it:
  # --temp-dir cannot be changed on a live cluster, it needs a restart.
  if [[ -e /tmp/ray/ray_current_cluster ]]; then
    echo "error: the running Ray head is rooted at /tmp (root volume), not ${RAY_TEMP_DIR}." >&2
    echo "       This is what filled the disk on 2026-07-30. Recycle it first:" >&2
    echo "         ${RAY_BIN} stop --force" >&2
    echo "         rm -rf /tmp/ray" >&2
    echo "       then re-run this script (it will start the head on nvme)." >&2
    exit 1
  fi
  # A head on nvme is still not enough. ale_bench evaluation runs inside RAY
  # WORKERS, which the raylet forks — so they inherit the RAYLET's environment,
  # not this script's. A head started from a shell without ALE_BENCH_LOCAL_TMPDIR
  # (e.g. a bare `ray start`) silently sends every case dir back to /tmp. That
  # happened on 2026-07-30 and leaked ~270MB/h until a janitor was added.
  local rl
  rl="$(pgrep -f 'raylet/raylet' | head -1)"
  if [[ -n "$rl" ]] && ! grep -qz "ALE_BENCH_LOCAL_TMPDIR=" "/proc/${rl}/environ" 2>/dev/null; then
    echo "warning: the running Ray head's raylet (pid ${rl}) has no ALE_BENCH_LOCAL_TMPDIR." >&2
    echo "         Workers it ALREADY forked keep writing case scratch to /tmp." >&2
    echo "         Newly forked workers are safe: get_local_scratch_root() now" >&2
    echo "         defaults to /opt/dlami/nvme when no override is set." >&2
    echo "         To clear the tail, recycle the head once training is idle:" >&2
    echo "           ${RAY_BIN} stop --force   # then relaunch from this script" >&2
  fi
}

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
  --save_every 2
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
VARIANCE=( --num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric variance )

# ── Wave definitions ─────────────────────────────────────────────────────────
# Format: "<run-name>|<env>|<problem_idx>|<arm>|<extra flags>"
#
# Seed variance runs (DmAa Q1): 2 new seeds x {AC1, CP26} x {UG-TTT, baseline}.
# --seed is implemented (mlora_train.py; ensemble.py + sampler.py seed-derived,
# seed 42 == published). Streaming is OFF on every seed run by design — the
# streaming-vs-no-stream comparison lives separately in wave2.
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

# Wave 2: component ablations + the streaming comparison. Streaming is OFF on
# every run EXCEPT cp26-stream, which pairs with cp26-nostream to answer the
# post-hoc-config critique (DmAa W2) as a controlled ablation. denoising is not
# here because it needs its own node infra (see WAVE4), not because the dispatch
# is missing — mlora_train.py:2425/2487 handles denoising, ahc039 and ahc058.
declare -a WAVE2=(
  "cp26-alpha0|cp|26|ALPHA0|"
  "cp26-entropy|cp|26|ENTROPY|"
  "cp26-nonnm|cp|26|NONNM|"
  "cp26-variance|cp|26|VARIANCE|"
  "cp26-nostream|cp|26|UGTTT|"
  "cp26-stream|cp|26|UGTTT|STREAM"
  "ac1-nostream|ac1|improvement|UGTTT|"
  "erdos-ugttt|erdos|improvement|UGTTT|"
)

# Wave 3 (node 3): model-scaling generality — answers "single base model"
# (8mjY W3, vGzb Q1) and "would gains hold at scale" (vGzb Q2). Each is a
# UG-TTT vs baseline PAIR so the reported quantity is the delta, not an absolute.
# Combined with the 8B runs in waves 1-2 this gives an 8B -> 14B -> 32B trend
# with matched baselines. Streaming OFF. --base_model in the extra field
# overrides COMMON's Qwen3-8B (argparse takes the last value). All load via the
# existing model_type="llama" path (Qwen3 is Llama-compatible), one GPU each —
# the frozen base is shared across the ensemble, so 32B (~64GB fp16) fits in
# 80GB. NOTE: 32B is the tight one; smoke-test a single 32B run before filling
# the node. Cross-FAMILY (Llama-3.1-8B) is on branch exp/cross-family-llama.
#
# --save_every 1 is set PER RUN here rather than in COMMON on purpose: wave1 and
# wave2 are already running on nodes 1-2 at --save_every 2, and changing COMMON
# would alter their behaviour if they ever resume. Same argparse-last-wins
# mechanism as --base_model. Every epoch = a reboot costs at most 1 epoch.
declare -a WAVE3=(
  "cp26-qwen14b-ugttt|cp|26|UGTTT|--base_model Qwen/Qwen3-14B --save_every 1"
  "cp26-qwen14b-base|cp|26|BASELINE|--base_model Qwen/Qwen3-14B --save_every 1"
  "cp26-qwen32b-ugttt|cp|26|UGTTT|--base_model Qwen/Qwen3-32B --save_every 1"
  "cp26-qwen32b-base|cp|26|BASELINE|--base_model Qwen/Qwen3-32B --save_every 1"
  "ac1-qwen14b-ugttt|ac1|improvement|UGTTT|--base_model Qwen/Qwen3-14B --save_every 1"
  "ac1-qwen14b-base|ac1|improvement|BASELINE|--base_model Qwen/Qwen3-14B --save_every 1"
)

# Wave 4: NEW DOMAINS beyond the mathematical subset — the direct answer to
# 8mjY W3 / vGzb W1. UG-TTT vs baseline pair per domain, streaming off.
#   denoising = single-cell biology (pure-CPU verifier; needs the bio deps +
#     openproblems + pancreas dataset in a venv the TRAINER uses — the verifier
#     is imported in-process at mlora_train.py:2426, so a standalone venv is
#     never seen. Use a clone of the training venv; see
#     requirements/denoising/README.md).
#   ahc039    = AtCoder heuristic algorithm design (C++; NO Docker —
#     ale_bench/utils.py:487 docker_client() is a host-side Ray mock that
#     compiles on the host; any C++20 g++ works. Data/judges vendored).
# These two need DIFFERENT node infra; smoke-test one baseline per domain before
# launching its pair. Requires 4 GPUs on a node with the matching setup.
declare -a WAVE4=(
  "denoise-ugttt|denoising|improvement|UGTTT|"
  "denoise-base|denoising|improvement|BASELINE|"
  "ahc039-ugttt|ahc039|ahc039|UGTTT|--save_every 1"
  "ahc039-base|ahc039|ahc039|BASELINE|--save_every 1"
  # Second seed for the ahc039 pair, so the new-domain claim is not itself n=1
  # (vGzb W1). Identical to the two runs above in every respect except --seed;
  # the pair above takes the argparse default of 42, matching the published
  # runs, so together they give n=2 per arm. Select with ONLY=seed2.
  "ahc039-ugttt-seed2|ahc039|ahc039|UGTTT|--save_every 1 --seed 2"
  "ahc039-base-seed2|ahc039|ahc039|BASELINE|--save_every 1 --seed 2"
  # Third seed, added 2026-07-30 to take the ahc039 pair to n=3 while the
  # seed-42 pair finishes on GPUs 0-1. Select with ONLY=seed3 (a substring that
  # matches ONLY these two, unlike ONLY=ahc039 which matches all six).
  "ahc039-ugttt-seed3|ahc039|ahc039|UGTTT|--save_every 1 --seed 3"
  "ahc039-base-seed3|ahc039|ahc039|BASELINE|--save_every 1 --seed 3"
)

usage() {
  echo "usage: $0 [--list] <wave1|wave2|wave3|wave4>"
  echo "  env: GPU_START=<n>  ONLY=<run-name-substring>  PYTHON=<interpreter>"
  exit 1
}

list_runs() {
  local -n arr=$1
  printf '%-20s %-11s %-12s %-9s %s\n' RUN ENV PROBLEM ARM EXTRA
  for spec in "${arr[@]}"; do
    IFS='|' read -r name env pidx arm extra <<< "$spec"
    # --list must show exactly what a launch with these env vars would start,
    # so it honours ONLY rather than always printing the full wave.
    if [[ -z "$ONLY" || "$name" == *"$ONLY"* ]]; then
      printf '%-20s %-11s %-12s %-9s %s\n' "$name" "$env" "$pidx" "$arm" "$extra"
    fi
  done
}

[[ $# -ge 1 ]] || usage

if [[ "$1" == "--list" ]]; then
  echo "== wave1 =="; list_runs WAVE1
  echo; echo "== wave2 =="; list_runs WAVE2
  echo; echo "== wave3 =="; list_runs WAVE3
  echo; echo "== wave4 =="; list_runs WAVE4
  exit 0
fi

WAVE_NAME="$1"
case "$WAVE_NAME" in
  wave1) RUNS=("${WAVE1[@]}") ;;
  wave2) RUNS=("${WAVE2[@]}") ;;
  wave3) RUNS=("${WAVE3[@]}") ;;
  wave4) RUNS=("${WAVE4[@]}") ;;
  *) usage ;;
esac

# Keep only the runs whose name matches ONLY. Written as a full if/fi because
# under `set -e` a false test as the last statement in a loop body exits.
if [[ -n "$ONLY" ]]; then
  declare -a FILTERED=()
  for spec in "${RUNS[@]}"; do
    IFS='|' read -r spec_name _ _ _ _ <<< "$spec"
    if [[ "$spec_name" == *"$ONLY"* ]]; then
      FILTERED+=( "$spec" )
    fi
  done
  # Only expand FILTERED once known non-empty: "${FILTERED[@]-}" on an empty
  # array yields one empty element, which would then launch a garbage run.
  if [[ ${#FILTERED[@]} -eq 0 ]]; then
    echo "error: no runs in ${WAVE_NAME} match ONLY=${ONLY}" >&2; exit 1
  fi
  RUNS=("${FILTERED[@]}")
fi

[[ $(( GPU_START + ${#RUNS[@]} )) -le $NUM_GPUS ]] || {
  echo "error: ${#RUNS[@]} runs starting at GPU $GPU_START exceeds $NUM_GPUS GPUs" >&2; exit 1; }

ensure_ray_head

LOG_DIR="logs/aws/${WAVE_NAME}"
mkdir -p "$LOG_DIR"

echo "Launching ${#RUNS[@]} runs on GPUs ${GPU_START}..$(( GPU_START + ${#RUNS[@]} - 1 )) (shared Ray head, cpu_scheduler partitions ${TOTAL_CPUS} vCPUs)"
echo "Python: $PYTHON"
echo "Logs: $LOG_DIR"
echo

gpu=$GPU_START
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

  # Pin only the GPU. CPUs are partitioned by the shared cpu_scheduler actor
  # (base_reward_task.py) — do NOT taskset here; every run uses the same
  # --num_cpus_per_task from COMMON so the scheduler divides cores evenly.
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONPATH="mLoRA:${PYTHONPATH:-}" \
  setsid \
    "$PYTHON" -m tinker_cookbook.rl.mlora_train \
      "${COMMON[@]}" \
      "${arm_flags[@]}" \
      "${extra_flags[@]}" \
      --env "$env" \
      --problem_idx "$pidx" \
      --log_path "./logs/aws/${WAVE_NAME}/${name}" \
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
echo "Check the CPU pool: ray status   (cpu_scheduler should own ${TOTAL_CPUS} CPUs)"
echo "Load average should settle near ${TOTAL_CPUS}; much higher means the"
echo "cpu_scheduler didn't bound the pool (check --num_cpus_per_task is uniform)."
