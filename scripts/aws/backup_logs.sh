#!/bin/bash
# Back up run artefacts off the instance. RUN THIS PERIODICALLY.
#
# WHY THIS MATTERS: wandb stores ONLY scalars. The generated code/text — which
# the solution-family (Shannon) entropy metric is computed from — exists ONLY
# in trajectories.jsonl on this instance's disk. If the box is wiped, family
# entropy becomes permanently uncomputable for those runs. Nothing can
# reconstruct it from wandb.
#
# Usage:
#   bash scripts/aws/backup_logs.sh s3://my-bucket/ugttt          # to S3
#   bash scripts/aws/backup_logs.sh ~/persistent/ugttt-backup     # to a local/EBS path
#   WITH_CHECKPOINTS=1 bash scripts/aws/backup_logs.sh <dest>     # include adapter weights
#
# Cron it every 15 min (survives your SSH session dropping):
#   (crontab -l 2>/dev/null; echo "*/15 * * * * cd $PWD && bash scripts/aws/backup_logs.sh <dest> >> /tmp/ugttt_backup.log 2>&1") | crontab -
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

DEST="${1:-}"
[[ -n "$DEST" ]] || { echo "usage: $0 <s3://bucket/prefix | /local/path>" >&2; exit 1; }

SRC="logs"
[[ -d "$SRC" ]] || { echo "error: no ./logs directory here — run from the repo root of a node that has runs" >&2; exit 1; }

HOST="$(hostname -s)"

# Small, irreplaceable files. trajectories.jsonl holds prompt_text +
# generated_text per rollout (the code needed for family entropy).
# metrics.jsonl holds per-epoch scalars. last_epoch.txt + *sampler*.json are
# what a resume needs.
INCLUDES=(
  --include='*/'
  --include='trajectories.jsonl'
  --include='metrics.jsonl'
  --include='last_epoch.txt'
  --include='*sampler*.json'
  --include='*.log'
)
if [[ "${WITH_CHECKPOINTS:-0}" == "1" ]]; then
  # Adapter weights — large. Needed to resume mid-run rather than from epoch 0.
  INCLUDES+=( --include='*.bin' --include='*.pt' --include='*.safetensors' )
fi

echo "[$(date -Is)] backing up ${SRC} -> ${DEST}/${HOST}  (checkpoints=${WITH_CHECKPOINTS:-0})"

if [[ "$DEST" == s3://* ]]; then
  command -v aws >/dev/null || { echo "error: aws CLI not found" >&2; exit 1; }
  S3_ARGS=(--exclude '*' --include '*/trajectories.jsonl' --include '*/metrics.jsonl'
           --include '*/last_epoch.txt' --include '*/*sampler*.json' --include '*.log')
  if [[ "${WITH_CHECKPOINTS:-0}" == "1" ]]; then
    S3_ARGS+=(--include '*.bin' --include '*.pt' --include '*.safetensors')
  fi
  aws s3 sync "$SRC" "${DEST%/}/${HOST}" "${S3_ARGS[@]}"
else
  mkdir -p "${DEST%/}/${HOST}"
  if command -v rsync >/dev/null; then
    rsync -av --prune-empty-dirs "${INCLUDES[@]}" --exclude='*' "$SRC/" "${DEST%/}/${HOST}/"
  else
    cp -r "$SRC/." "${DEST%/}/${HOST}/"
  fi
fi

echo "[$(date -Is)] backup complete"
echo
echo "Sanity check — trajectories captured so far:"
find "$SRC" -name trajectories.jsonl -exec sh -c 'echo "  $(wc -l < "$1") rollouts  $1"' _ {} \; 2>/dev/null || true
