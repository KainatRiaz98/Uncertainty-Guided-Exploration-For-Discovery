#!/bin/bash
# Reap leaked ale_bench per-case scratch dirs from /tmp (the small root volume).
#
# WHY THIS EXISTS: ale_bench never cleans up its per-case scratch dirs. The code
# now defaults to /opt/dlami/nvme when no env override is set
# (case_runner.get_local_scratch_root), so NEW worker processes write to nvme.
# But Ray workers already running when that fix landed had imported the old
# module and keep writing to /tmp until they are recycled. This janitor covers
# that tail, and is cheap insurance against any future path that lands on /tmp.
#
# SAFETY: only removes /tmp/ale_bench_local_* dirs untouched for AGE_MIN minutes.
# One case evaluation is bounded by --eval_timeout (1000s ~= 17min), so the
# 30min floor cannot race a live case. EMERG_AGE_MIN is still above that bound.
set -uo pipefail

AGE_MIN="${AGE_MIN:-30}"
EMERG_AGE_MIN="${EMERG_AGE_MIN:-20}"
EMERG_FREE_GB="${EMERG_FREE_GB:-5}"
INTERVAL="${INTERVAL:-300}"
LOG="${LOG:-/opt/dlami/nvme/experiments/ale_bench_janitor.log}"

count() { find /tmp -maxdepth 1 -name 'ale_bench_local_*' -type d 2>/dev/null | wc -l; }
free_gb() { df --output=avail -BG / 2>/dev/null | tail -1 | tr -dc '0-9'; }

echo "$(date -Is) janitor started (age>${AGE_MIN}min, every ${INTERVAL}s, emergency<${EMERG_FREE_GB}G)" >> "$LOG"
while true; do
  before=$(count)
  find /tmp -maxdepth 1 -name 'ale_bench_local_*' -type d -mmin "+${AGE_MIN}" \
    -exec rm -rf {} + 2>/dev/null
  after=$(count); avail=$(free_gb)

  # Emergency: if the root volume is still tight, reap harder (but never below
  # the eval timeout, so a live case is never pulled out from under a worker).
  if [ -n "$avail" ] && [ "$avail" -lt "$EMERG_FREE_GB" ]; then
    echo "$(date -Is) EMERGENCY: only ${avail}G free — reaping >${EMERG_AGE_MIN}min" >> "$LOG"
    find /tmp -maxdepth 1 -name 'ale_bench_local_*' -type d -mmin "+${EMERG_AGE_MIN}" \
      -exec rm -rf {} + 2>/dev/null
    after=$(count); avail=$(free_gb)
    echo "$(date -Is) EMERGENCY done: ${after} dirs left, ${avail}G free" >> "$LOG"
  fi

  if [ "$before" != "$after" ]; then
    echo "$(date -Is) reaped $(( before - after )) dirs (${before} -> ${after}); root avail ${avail}G" >> "$LOG"
  fi
  sleep "$INTERVAL"
done
