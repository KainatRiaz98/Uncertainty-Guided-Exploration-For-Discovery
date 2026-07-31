# Disk-overflow recovery — 2026-07-30

The VM's root volume (`/dev/root`, 29 GB) hit **100% full / 0 bytes free**, which killed all
in-flight experiment runs. Last log write was `2026-07-30 01:45`. No training processes
survived; only idle Ray infrastructure (`gcs_server`, `raylet`, `ray::IDLE`) was still up.

## What was rescued

Everything now lives on the 6.9 TB nvme volume:

| Path | Contents |
| --- | --- |
| `/opt/dlami/nvme/experiments/Uncertainty-Guided-Exploration-For-Discovery/` | Full repo copy (2.3 GB), including all `logs/aws/**` run data |
| `/opt/dlami/nvme/experiments/ale_bench_local_scratch_outputs.tar.gz` | 164,540 `output.txt` + `profiles.json` files salvaged from the leaked `/tmp` eval scratch (8.8 MB) |

The copy was verified byte-identical to the source with `rsync -ni` (no differences).
The original at `/home/ubuntu/Uncertainty-Guided-Exploration-For-Discovery` is untouched.

## Run status — all 8 runs are restartable

Resume works via `last_epoch.txt`: `mlora_train.py` reads it, sets
`start_epoch = last_epoch + 1`, calls `ensemble.load(log_path, start_epoch - 1)`, and rebuilds
MI history from `metrics.jsonl`. Sampler state resumes from
`puct_backprop_sampler_step_*.json`.

All 32 intact `adapter.pt` checkpoints were load-tested with
`torch.load(..., weights_only=True)` — **every one loads, 288 tensors each**.

| Run | seed | last_epoch | resumes from | trajectories | status |
| --- | --- | --- | --- | --- | --- |
| `wave4/ahc039-ugttt` | **42** | 1 | step_1 (5 members) | 192 | incomplete, 4 epochs left |
| `wave4/ahc039-base` | **42** | 2 | step_2 (1 member) | 248 | incomplete, 3 epochs left |
| `wave4/ahc039-ugttt-seed2` | 2 | 0 | step_0 (5 members) | 80 | incomplete, 5 epochs left |
| `wave4/ahc039-base-seed2` | 2 | 0 | step_0 (1 member) | 80 | incomplete, 5 epochs left |
| `wave1-rerun/ac1-base-seed2` | 2 | 5 | step_5 | 384 | **complete** (num_epochs=6) |
| `wave1-rerun/ac1-base-seed3` | 3 | 5 | step_5 | 384 | **complete** |
| `wave1-rerun/cp26-base-seed2` | 2 | 5 | step_5 | 384 | **complete** |
| `wave1-rerun/cp26-base-seed3` | 3 | 5 | step_5 | 384 | **complete** |

Trajectory records carry the full 33-field schema (`reward_exec`, `reward_rmi`, `reward_total`,
`uncertainty_true_mi`, `uncertainty_rmi`, `ensemble_*` disagreement, `streaming_mi_*`, etc.).

## Crash damage found and repaired

1. **Truncated checkpoint.** `wave4/ahc039-ugttt/ensemble_0/step_2/adapter.pt` was cut off at
   2,621,440 bytes vs the full 61,444,223 — the write that ran out of disk. It is **not** on the
   resume path (`last_epoch=1` → loads `step_1`, which is complete for all 5 members).
   Renamed to `ensemble_0/step_2.CORRUPT-truncated` so it can't be mistaken for a usable
   checkpoint. Safe to delete.

2. **Truncated trajectory logs.** Three `trajectories.jsonl` files ended in a partial JSON line.
   The partial line was dropped and the original preserved as `*.jsonl.corrupt-backup`:
   - `wave4/ahc039-ugttt-seed2/` — 81 → 80 records
   - `wave4/ahc039-base/` — 249 → 248 records
   - `wave4/ahc039-base-seed2/` — 81 → 80 records

   `wave4/ahc039-ugttt` (seed 42) needed no repair — its 192 records were already clean.
   All `metrics.jsonl` and all 32 sampler-state JSON files validated clean.

## Root cause of the overflow

`ale_bench/tool_wrappers/case_runner.py` created a scratch dir per evaluation case with
`tempfile.mkdtemp(..., dir="/tmp")` hardcoded at two sites, and **never cleaned them up**. Each
dir holds a ~120 KB regenerable `input.txt` plus tiny `output.txt`/`profiles.json`.
**82,283 leaked dirs ≈ 10.6 GB** accumulated on the root volume over two days.

### Fix applied (nvme copy only)

Added `get_local_scratch_root()` in `case_runner.py` and pointed both `mkdtemp` calls at it.
It honours `ALE_BENCH_LOCAL_TMPDIR`, then `TMPDIR`, falling back to `/tmp` — so scratch can be
placed on nvme instead of the root filesystem. Syntax-checked and behaviour-verified.

Set this before any restart:

```bash
export ALE_BENCH_LOCAL_TMPDIR=/opt/dlami/nvme/scratch/ale_bench
```

Note this relocates the leak rather than closing it. The dirs still accumulate, just on a
volume with 6.5 TB free. A proper cleanup in the case-runner lifecycle is still worth doing.

The patch was applied to **both** copies (nvme and `/home/ubuntu`), verified identical, so a
launch from either path is safe.

## Everything moved off the root volume (2026-07-30)

There were **three** independent refill paths, not one. All are now on nvme:

| Path | Was | Now | Verified by |
| --- | --- | --- | --- |
| ale_bench per-case scratch | `/tmp/ale_bench_local_*` | `$ALE_BENCH_LOCAL_TMPDIR` = `/opt/dlami/nvme/scratch/ale_bench` | `mkdtemp` self-test landed on nvme |
| Ray session logs **+ object spilling** | `/tmp/ray/session_*` | `--temp-dir=/opt/dlami/nvme/ray` | decoded `object_spilling_config` now points to nvme |
| Stray `ray.init()` without `RAY_ADDRESS=auto` | `/tmp/ray` (Ray's default) | `RAY_TMPDIR=/opt/dlami/nvme/ray` | reproduced the leak, then confirmed the fix |
| General `tempfile`/compile dirs | `/tmp` | `TMPDIR=/opt/dlami/nvme/scratch/tmp` | — |
| HF weights cache | — | `HF_HOME=/opt/dlami/nvme/hf-cache` (pre-existing) | — |

The third row is a real trap: `--temp-dir` only governs the head *you* start. The self-test
above recreated `/tmp/ray` within seconds because it called `ray.init()` without
`RAY_ADDRESS=auto`. `RAY_TMPDIR` is the belt-and-braces fix and is now exported by both
launchers.

The old Ray cluster was recycled: it was poisoned anyway (workers dead `SYSTEM_ERROR`, metrics
agent unreachable) and `--temp-dir` cannot be changed on a live cluster. A fresh head now runs
on nvme — 96 CPU, 4 GPU, no failures. Root went 100% → **66% used, 9.7 GB free**.

`launch_wave.sh`'s `ensure_ray_head` now **refuses to launch** onto a head rooted at `/tmp`
rather than silently piling onto the failure condition.

## Resuming the seed-42 pair

`scripts/aws/resume_wave4_ahc039_seed42.sh` — `check` verifies and launches nothing, `run`
launches. It exists because `launch_wave.sh`'s `ONLY=` is a *substring* filter and
`ahc039-ugttt` is a prefix of `ahc039-ugttt-seed2`, so no `ONLY` value selects just the
seed-42 pair; `ONLY=ahc039` would relaunch all four and clobber the seed-2 runs.

Flags reproduce `COMMON + {UGTTT,BASELINE} + --save_every 1` exactly as recorded in each
`config.json`. It passes no `--seed`, since both runs took the argparse default of 42.

A preflight refuses to start unless every ensemble member's resume checkpoint exists at exactly
61,444,223 bytes — the specific failure the overflow caused. Current output:

```
ahc039-ugttt   last_epoch=1 -> resumes at epoch 2, loads step_1 (5/5 members OK)
ahc039-base    last_epoch=2 -> resumes at epoch 3, loads step_2 (1/1 members OK)
```

Resume logs go to `*.resume_<timestamp>.log` so the original crash logs stay intact.
