#!/bin/bash
# Launch all four arms on an 8x A100-40GB node, two GPUs each, detached.
#
#   export WANDB_ENTITY=... WANDB_API_KEY=...
#   ./runs_a100/launch_all.sh
#
#   GPUs 0,1  ahc-baseline        (--env ac1)
#   GPUs 2,3  ahc-ugttt           (--env ac1)
#   GPUs 4,5  denoising-baseline  (--env denoising)
#   GPUs 6,7  denoising-ugttt     (--env denoising)
#
# Qwen3-8B fp16 is ~16GB of weights split across the pair, so 2x40GB leaves
# plenty for KV cache even with the 5-member ensemble on the ugttt arms.
# Each run gets its own Ray head (own port + temp dir) so the four cannot
# collide on Ray's detached `cpu_scheduler` actor.
#
# Every run checkpoints EVERY epoch (--save_every 1).
set -euo pipefail

: "${WANDB_ENTITY:?export WANDB_ENTITY first}"
: "${WANDB_API_KEY:?export WANDB_API_KEY first}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$GPU_COUNT" -lt 8 ]; then
  echo "WARNING: expected 8 GPUs, found $GPU_COUNT. Edit the table below." >&2
fi

mkdir -p logs

#            RUN_NAME             ENV_NAME    ARM       GPUS  RAY_PORT  PROBLEM_IDX
RUNS=(
  "ahc-baseline|ac1|baseline|0,1|6390|improvement"
  "ahc-ugttt|ac1|ugttt|2,3|6391|improvement"
  "denoising-baseline|denoising|baseline|4,5|6392|improvement"
  "denoising-ugttt|denoising|ugttt|6,7|6393|improvement"
)

for spec in "${RUNS[@]}"; do
  IFS='|' read -r name env arm gpus port pidx <<< "$spec"
  out="logs/${name}.out"
  if pgrep -f "wandb_name $name" >/dev/null 2>&1; then
    echo "SKIP $name (already running)"
    continue
  fi
  echo "launching $name  env=$env arm=$arm gpus=$gpus -> $out"
  RUN_NAME="$name" ENV_NAME="$env" ARM="$arm" GPU_IDS="$gpus" \
  RAY_PORT="$port" PROBLEM_IDX="$pidx" \
  setsid nohup "$SCRIPT_DIR/run_one.sh" > "$out" 2>&1 < /dev/null &
  disown
  sleep 20     # stagger: avoid four processes hitting the HF cache at once
done

echo
echo "all four launched. useful commands:"
echo "  tail -f logs/ahc-ugttt.out"
echo "  grep -hE 'Group [0-9]+:|\\[Epoch' logs/*.out | tail"
echo "  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
echo "  pkill -f mlora_train    # stop everything"
