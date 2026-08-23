#!/bin/bash
# ============================================================================
# ICLR rerun watchdog — keeps the four approved runs alive and the GPUs busy.
#
#   bash scripts/iclr/watchdog.sh check    # report current state, change nothing
#   bash scripts/iclr/watchdog.sh start    # start the daemon (setsid, detached)
#   bash scripts/iclr/watchdog.sh status   # daemon + per-run status
#   bash scripts/iclr/watchdog.sh stop     # stop the daemon (runs keep going)
#
# WHY A CLEAN RESTART AND NEVER A RESUME  (read before changing RESTART_POLICY)
#   mlora_train.py:2737-2749 reads <log_path>/last_epoch.txt on startup and
#   silently sets resume_step. So simply re-running a crashed launcher RESUMES
#   mid-run. Per scripts/iclr/README.md the seed-2 runs are seed-only mirrors of
#   the seed-42 pair; a resumed run is contaminated and "must be flagged, not
#   used". A resumed run therefore burns 100% of its remaining GPU hours
#   producing data nobody can put in the paper — strictly worse than restarting
#   from epoch 0, which costs the elapsed epochs but yields a usable run.
#
#   So on a crash this watchdog ARCHIVES the run directory (which removes
#   last_epoch.txt from the live path) and relaunches from epoch 0 with
#   byte-identical flags, by calling the run's own approved launcher. It never
#   edits a flag. If a failure would need a config change to fix, it does not
#   restart — it records the incident and leaves that GPU free for you.
#
# GUARANTEES
#   - Never modifies any training flag, config, or launcher.
#   - Never resumes (archives last_epoch.txt out of the way first).
#   - Never restarts a run that finished ("Training complete") or that failed
#     for a non-transient reason.
#   - Caps restarts (MAX_RESTARTS) and detects crash-loops, so a deterministic
#     failure cannot burn the box in a relaunch cycle.
#   - Every decision is appended to logs/iclr/_watchdog/events.log.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO"

# ── policy knobs ───────────────────────────────────────────────────────────
RESTART_POLICY="${RESTART_POLICY:-clean}"   # clean | never   (never = page only)
MAX_RESTARTS="${MAX_RESTARTS:-3}"           # per run, per watchdog lifetime
POLL_SECS="${POLL_SECS:-60}"
STALL_WARN_SECS="${STALL_WARN_SECS:-2700}"  # 45 min of no log growth -> WARN
STALL_KILL_SECS="${STALL_KILL_SECS:-7200}"  # 2 h  of no log growth -> hung
MIN_HEALTHY_SECS="${MIN_HEALTHY_SECS:-900}" # restart dying <15 min = crash-loop
MIN_FREE_GB="${MIN_FREE_GB:-40}"            # launcher preflight needs 40 GB

STATE_DIR="logs/iclr/_watchdog"
EVENTS="${STATE_DIR}/events.log"
INCIDENTS="${STATE_DIR}/incidents"
PIDFILE="${STATE_DIR}/watchdog.pid"
mkdir -p "$STATE_DIR" "$INCIDENTS"

# ── run table: name|log_file|run_dir|gpu|launcher|epochs|env-prefix ─────────
# env-prefix reproduces exactly how each run was launched, recovered from
# /proc/<pid>/environ of the live processes on 2026-08-22.
HF="/opt/dlami/nvme/hf-cache"
TMPD="/opt/dlami/nvme/scratch/tmp"
CPBIN="/opt/dlami/nvme/venvs/venv-cp/bin"

RUNS=(
"denoising-seed2-baseline|logs/iclr/denoising-seed2-baseline.log|logs/iclr/denoising-seed2-baseline|0|scripts/iclr/launch_denoising_seed2_base.sh|6|GPU=0 RAY_PORT=6402 HF_HOME=${HF} TMPDIR=${TMPD} RAY_TMP_ROOT=/nvme/raytmp"
"denoising-seed2-ugttt|logs/iclr/denoising-seed2-ugttt.log|logs/iclr/denoising-seed2-ugttt|1|scripts/iclr/launch_denoising_seed2_ugttt.sh|6|GPU=1 RAY_PORT=6403 HF_HOME=${HF} TMPDIR=${TMPD} RAY_TMP_ROOT=/nvme/raytmp"
"cp26-gamma-fixed|logs/aws/gamma_ablation/cp26-gamma-fixed.log|logs/aws/gamma_ablation/cp26-gamma-fixed|2|scripts/aws/launch_gamma_ablation.sh|4|GPU=2 HF_HOME=${HF} PATH=${CPBIN}:${PATH}"
"cp26-alpha0-seed2|logs/iclr/cp26-alpha0-seed2.log|logs/iclr/cp26-alpha0-seed2|3|scripts/iclr/launch_cp26_alpha0_seed2.sh|6|GPU=3 HF_HOME=${HF} PATH=${CPBIN}:${PATH}"
)

field() { echo "$1" | cut -d'|' -f"$2"; }

ts()  { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { echo "[$(ts)] $*" | tee -a "$EVENTS"; }

# ── liveness ───────────────────────────────────────────────────────────────
run_pid() {  # echo pid of the trainer for run $1, empty if not running
  pgrep -f "mlora_train.*--wandb_name ${1}\$" 2>/dev/null | head -1
}

log_age() {  # seconds since $1 last changed; huge number if missing
  [ -f "$1" ] || { echo 999999; return; }
  echo $(( $(date +%s) - $(stat -c %Y "$1") ))
}

last_epoch() { [ -f "$1/last_epoch.txt" ] && cat "$1/last_epoch.txt" 2>/dev/null || echo "-"; }

rollouts() { [ -f "$1/trajectories.jsonl" ] && wc -l < "$1/trajectories.jsonl" 2>/dev/null || echo 0; }

r_max() {    # R_max = max reward_exec over ALL rollouts (the reported headline)
  # epoch rows carry train/reward/exec/best; per-group rows carry
  # rollout_group/reward_exec_max. Take the max over both so R_max is correct
  # mid-epoch too, before the first epoch row is written.
  [ -f "$1/metrics.jsonl" ] || { echo "-"; return; }
  python3 - "$1/metrics.jsonl" <<'PY' 2>/dev/null || echo "-"
import json,sys
b=[]
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except Exception: continue
    for k in ("train/reward/exec/best","rollout_group/reward_exec_max"):
        v=d.get(k)
        if isinstance(v,(int,float)) and v==v: b.append(v)
print(round(max(b),4) if b else "-")
PY
}

finished() { grep -q "Training complete" "$1" 2>/dev/null; }

# ── crash classification ───────────────────────────────────────────────────
# TRANSIENT  -> infrastructure; relaunching unchanged is the right fix.
# CODE_ERROR -> the trainer raised a Python exception. Fixing that would mean
#               changing config or code, which is forbidden here -> page a human.
# DISK_FULL  -> relaunching cannot help until space is freed -> page a human.
# UNKNOWN    -> process vanished leaving NO traceback and NO known error string.
#               A config/code fault always leaves a traceback, so "no diagnostic
#               at all" means an external kill: host OOM-killer, SIGKILL, driver
#               abort, node event. Those are transient, so this DOES restart —
#               the crash-loop guard and MAX_RESTARTS stop it if it is not.
#               (Learned from testing: a SIGKILLed trainer logs nothing at all.)
classify() {
  local logf="$1" tail_txt
  tail_txt="$(tail -400 "$logf" 2>/dev/null)"
  case "$tail_txt" in
    *"No space left on device"*)                  echo "DISK_FULL";        return ;;
    *"CUDA out of memory"*|*"OutOfMemoryError"*)  echo "TRANSIENT:cuda_oom";  return ;;
    *"RayActorError"*|*"ActorDiedError"*)         echo "TRANSIENT:ray_actor"; return ;;
    *"Deadline Exceeded"*|*"RpcError"*)           echo "TRANSIENT:ray_rpc";   return ;;
    *"ConnectionError"*|*"Connection reset"*|*"Read timed out"*) echo "TRANSIENT:network"; return ;;
    *"NCCL"*error*|*"CUDA error"*|*"device-side assert"*)        echo "TRANSIENT:cuda_ctx"; return ;;
    *"Killed"*|*"SIGKILL"*|*"MemoryError"*)       echo "TRANSIENT:oom_host";  return ;;
    *"HTTPError"*|*"huggingface"*Error*)          echo "TRANSIENT:hf";        return ;;
    *"Traceback"*)                                echo "CODE_ERROR";      return ;;
  esac
  echo "UNKNOWN"
}

# ── drain a GPU before relaunching ─────────────────────────────────────────
# Every launcher's preflight refuses at >2000 MiB used. After a crash the dead
# trainer's Ray workers can keep a CUDA context alive for a while; without this
# the relaunch fails preflight and the GPU sits idle. Kill only processes that
# nvidia-smi reports as holding memory on THIS GPU index, so the sibling arm on
# another GPU is never touched.
drain_gpu() {
  local gpu="$1" waited=0 used pids
  while [ "$waited" -lt 300 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 0)
    [ "${used:-0}" -lt 2000 ] && { log "DRAIN    gpu$gpu clear (${used} MiB) after ${waited}s"; return 0; }
    if [ "$waited" -ge 60 ]; then
      pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null | tr -d ' ')
      if [ -n "$pids" ]; then
        log "DRAIN    gpu$gpu still ${used} MiB after ${waited}s; killing holders: $(echo $pids | tr '\n' ' ')"
        for p in $pids; do kill -9 "$p" 2>/dev/null; done
      fi
    fi
    sleep 15; waited=$((waited+15))
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo 0)
  log "DRAIN    gpu$gpu STILL ${used} MiB after ${waited}s — relaunch will likely fail preflight"
  return 1
}

# ── the restart path ───────────────────────────────────────────────────────
restart_run() {
  local name="$1" logf="$2" rundir="$3" gpu="$4" launcher="$5" envp="$6" reason="$7"
  local n stamp inc
  n=$(( $(cat "${STATE_DIR}/${name}.restarts" 2>/dev/null || echo 0) + 1 ))
  stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
  inc="${INCIDENTS}/${name}.attempt${n}.${stamp}"
  mkdir -p "$inc"

  # 1. preserve the wreckage, and in doing so remove last_epoch.txt from the
  #    live path -> the relaunch CANNOT resume. This is the whole point.
  [ -d "$rundir" ] && mv "$rundir" "${inc}/rundir"
  [ -f "$logf"  ] && mv "$logf"  "${inc}/console.log"
  tail -400 "${inc}/console.log" 2>/dev/null > "${inc}/tail-400.txt"
  {
    echo "run:      $name"
    echo "when:     $(ts)"
    echo "attempt:  $n"
    echo "reason:   $reason"
    echo "epochs done before crash: $(last_epoch "${inc}/rundir")"
    echo "rollouts before crash:    $(rollouts "${inc}/rundir")"
    echo "relaunch: env $envp bash $launcher run"
  } > "${inc}/INCIDENT.txt"

  echo "$n" > "${STATE_DIR}/${name}.restarts"
  log "RESTART  $name attempt=$n reason=$reason -> clean start from epoch 0 (archived: $inc)"

  # 2. make sure the GPU is actually free, or preflight will refuse.
  drain_gpu "$gpu"

  # 3. relaunch through the run's own approved launcher, flags untouched.
  #    The launcher runs its own preflight and refuses if the host is unfit.
  #    Retry a few times: a Ray head or GPU can need a moment to settle.
  local attempt ok=1
  for attempt in 1 2 3; do
    echo "--- relaunch attempt $attempt at $(ts) ---" >>"${STATE_DIR}/${name}.relaunch.log"
    if ( eval "env $envp bash $launcher run" ) >>"${STATE_DIR}/${name}.relaunch.log" 2>&1; then
      ok=0; break
    fi
    log "RETRY    $name relaunch attempt $attempt failed; waiting 60s"
    sleep 60
  done

  if [ "$ok" -eq 0 ]; then
    date +%s > "${STATE_DIR}/${name}.started"
    sleep 30
    if [ -n "$(run_pid "$name")" ]; then
      log "RESTART  $name relaunched OK pid=$(run_pid "$name") — running fresh from epoch 0"
    else
      log "FAILED   $name launcher returned 0 but no trainer process appeared — see ${STATE_DIR}/${name}.relaunch.log"
      echo "no_process_after_launch" > "${STATE_DIR}/${name}.halted"
    fi
  else
    log "FAILED   $name relaunch failed 3x — see ${STATE_DIR}/${name}.relaunch.log. GPU $gpu left free."
    echo "preflight_failed" > "${STATE_DIR}/${name}.halted"
  fi
}

# ── one inspection pass over one run ───────────────────────────────────────
check_run() {
  local spec="$1" mode="$2"
  local name logf rundir gpu launcher epochs envp pid age cls used
  name=$(field "$spec" 1); logf=$(field "$spec" 2); rundir=$(field "$spec" 3)
  gpu=$(field "$spec" 4);  launcher=$(field "$spec" 5); epochs=$(field "$spec" 6)
  envp=$(field "$spec" 7)

  pid="$(run_pid "$name")"
  age="$(log_age "$logf")"
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null || echo '?')"

  if [ "$mode" = "report" ]; then
    printf '%-26s pid=%-8s gpu%s=%sMiB  ep=%s/%s  rollouts=%-5s R_max=%-8s log_age=%ss %s\n' \
      "$name" "${pid:-DEAD}" "$gpu" "$used" "$(last_epoch "$rundir")" "$epochs" \
      "$(rollouts "$rundir")" "$(r_max "$rundir")" "$age" \
      "$( [ -f "${STATE_DIR}/${name}.halted" ] && echo '[HALTED]' )$( finished "$logf" && echo '[DONE]' )"
    return
  fi

  # already resolved states
  [ -f "${STATE_DIR}/${name}.halted" ] && return
  if finished "$logf"; then
    [ -f "${STATE_DIR}/${name}.done" ] || {
      log "DONE     $name last_epoch=$(last_epoch "$rundir") rollouts=$(rollouts "$rundir") R_max=$(r_max "$rundir")"
      touch "${STATE_DIR}/${name}.done"
    }
    return
  fi

  if [ -n "$pid" ]; then
    # alive — but is it doing anything?
    if [ "$age" -ge "$STALL_KILL_SECS" ]; then
      log "HUNG     $name pid=$pid no log growth for ${age}s (>${STALL_KILL_SECS}s). Killing for a clean restart."
      kill -9 "$pid" 2>/dev/null; sleep 20
      restart_run "$name" "$logf" "$rundir" "$gpu" "$launcher" "$envp" "hung_no_progress_${age}s"
    elif [ "$age" -ge "$STALL_WARN_SECS" ]; then
      local warned="${STATE_DIR}/${name}.stallwarn"
      if [ ! -f "$warned" ] || [ "$(( $(date +%s) - $(stat -c %Y "$warned") ))" -gt 3600 ]; then
        log "WARN     $name pid=$pid no log growth for ${age}s (epoch $(last_epoch "$rundir")/${epochs}) — watching"
        touch "$warned"
      fi
    fi
    return
  fi

  # ── dead ────────────────────────────────────────────────────────────────
  cls="$(classify "$logf")"
  local n; n=$(cat "${STATE_DIR}/${name}.restarts" 2>/dev/null || echo 0)
  log "CRASH    $name gone. class=$cls epoch=$(last_epoch "$rundir")/${epochs} rollouts=$(rollouts "$rundir") restarts_so_far=$n"
  log "CRASH    $name tail: $(tail -3 "$logf" 2>/dev/null | tr '\n' ' ' | cut -c1-400)"

  # crash-loop guard
  local started elapsed
  started="$(cat "${STATE_DIR}/${name}.started" 2>/dev/null || echo 0)"
  if [ "$started" -gt 0 ]; then
    elapsed=$(( $(date +%s) - started ))
    if [ "$elapsed" -lt "$MIN_HEALTHY_SECS" ]; then
      log "HALT     $name died ${elapsed}s after a restart — crash-loop. Not relaunching. GPU $gpu free."
      echo "crash_loop" > "${STATE_DIR}/${name}.halted"; return
    fi
  fi

  case "$cls" in
    DISK_FULL)
      log "HALT     $name out of disk. Free space then clear ${STATE_DIR}/${name}.halted. Not relaunching."
      echo "disk_full" > "${STATE_DIR}/${name}.halted"; return ;;
    CODE_ERROR)
      log "HALT     $name raised a Python exception — a fix would mean changing config/code, which this watchdog will not do. Reporting only. GPU $gpu free."
      echo "code_error" > "${STATE_DIR}/${name}.halted"; return ;;
    UNKNOWN)
      # no traceback, no known signature => external kill. Record the evidence.
      local oom; oom="$(dmesg 2>/dev/null | tail -60 | grep -i -m1 'out of memory\|oom-kill' || true)"
      [ -n "$oom" ] && log "CRASH    $name kernel says: ${oom:0:200}"
      log "CRASH    $name no traceback and no known error string — treating as an external kill (restartable)."
      ;;
  esac

  if [ "$RESTART_POLICY" = "never" ]; then
    log "HALT     $name RESTART_POLICY=never — reporting only."
    echo "policy_never" > "${STATE_DIR}/${name}.halted"; return
  fi
  if [ "$n" -ge "$MAX_RESTARTS" ]; then
    log "HALT     $name hit MAX_RESTARTS=$MAX_RESTARTS. Not relaunching. GPU $gpu free."
    echo "max_restarts" > "${STATE_DIR}/${name}.halted"; return
  fi

  local freegb; freegb=$(df -BG --output=avail "$REPO" 2>/dev/null | tail -1 | tr -dc '0-9')
  if [ -n "$freegb" ] && [ "$freegb" -lt "$MIN_FREE_GB" ]; then
    log "HALT     $name only ${freegb}GB free (<${MIN_FREE_GB}); launcher preflight would refuse. Not relaunching."
    echo "low_disk" > "${STATE_DIR}/${name}.halted"; return
  fi

  restart_run "$name" "$logf" "$rundir" "$gpu" "$launcher" "$envp" "$cls"
}

daemon() {
  log "WATCHDOG start pid=$$ policy=$RESTART_POLICY poll=${POLL_SECS}s max_restarts=$MAX_RESTARTS"
  for s in "${RUNS[@]}"; do
    n=$(field "$s" 1); [ -f "${STATE_DIR}/${n}.started" ] || date +%s > "${STATE_DIR}/${n}.started"
  done
  while true; do
    for s in "${RUNS[@]}"; do check_run "$s" "act"; done
    # all four resolved? then we are done.
    local live=0
    for s in "${RUNS[@]}"; do
      n=$(field "$s" 1)
      [ -f "${STATE_DIR}/${n}.done" ] || [ -f "${STATE_DIR}/${n}.halted" ] || live=$((live+1))
    done
    if [ "$live" -eq 0 ]; then log "WATCHDOG all runs resolved (done or halted). Exiting."; rm -f "$PIDFILE"; exit 0; fi
    sleep "$POLL_SECS"
  done
}

case "${1:-}" in
  check|status)
    echo "policy=$RESTART_POLICY  max_restarts=$MAX_RESTARTS  poll=${POLL_SECS}s"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "daemon: RUNNING pid=$(cat "$PIDFILE")"
    else echo "daemon: not running"; fi
    echo
    for s in "${RUNS[@]}"; do check_run "$s" "report"; done
    echo
    echo "disk: $(df -h "$REPO" | tail -1 | awk '{print $4" free on "$6}')"
    if [ -s "$EVENTS" ]; then echo; echo "last events:"; tail -8 "$EVENTS"; fi
    ;;
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running pid=$(cat "$PIDFILE")"; exit 0; fi
    setsid nohup bash "$0" _daemon >>"${STATE_DIR}/daemon.out" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"; disown || true
    sleep 2; echo "watchdog started pid=$(cat "$PIDFILE")  events: $EVENTS"
    ;;
  _daemon) echo $$ > "$PIDFILE"; daemon ;;
  stop)
    if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null && echo "stopped $(cat "$PIDFILE")"; rm -f "$PIDFILE"
    else echo "not running"; fi ;;
  *) echo "usage: $0 <check|start|status|stop>"; exit 1 ;;
esac
