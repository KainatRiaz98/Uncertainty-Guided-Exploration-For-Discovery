# Rebuttal data — all experiments run on this VM

Snapshot: 2026-07-31 20:16 UTC

Base model `Qwen/Qwen3-8B`, fp16, LoRA r16/alpha32 on q/k/v/o, `puct_backprop` sampler,
two-phase sampling. UGTTT = 5 ensemble members with `rmi_coef 0.1`/`nnm_coef 0.075`;
BASELINE = 1 member with both coefficients at 0.

## Runs

| run | env | problem | arm | members | seed | epochs | rollouts (raw) | rollouts (deduped) | mean R_exec | best R_exec | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `wave1-rerun/ac1-base-seed2` | ac1 | improvement | BASELINE | 1 | 2 | 6/6 | 384 | 384 | 0.3376 | 0.6628 | COMPLETE |
| `wave1-rerun/ac1-base-seed3` | ac1 | improvement | BASELINE | 1 | 3 | 6/6 | 384 | 384 | 0.3451 | 0.6601 | COMPLETE |
| `wave1-rerun/cp26-base-seed2` | cp | 26 | BASELINE | 1 | 2 | 6/6 | 384 | 384 | 1.1405 | 2.6299 | COMPLETE |
| `wave1-rerun/cp26-base-seed3` | cp | 26 | BASELINE | 1 | 3 | 6/6 | 384 | 384 | 1.0469 | 2.6278 | COMPLETE |
| `wave4/ahc039-base-seed2` | ahc039 | ahc039 | BASELINE | 1 | 2 | 1/6 | 80 | 80 | 0.3483 | 1.2401 | incomplete |
| `wave4/ahc039-base-seed3` | ahc039 | ahc039 | BASELINE | 1 | 3 | 0/6 | 0 | 0 | - | - | incomplete |
| `wave4/ahc039-base` | ahc039 | ahc039 | BASELINE | 1 | 42 | 6/6 | 440 | 384 | 0.5768 | 1.2662 | COMPLETE |
| `wave4/ahc039-ugttt-seed2` | ahc039 | ahc039 | UGTTT | 5 | 2 | 1/6 | 80 | 80 | 0.3960 | 1.1893 | incomplete |
| `wave4/ahc039-ugttt-seed3` | ahc039 | ahc039 | UGTTT | 5 | 3 | 0/6 | 0 | 0 | - | - | incomplete |
| `wave4/ahc039-ugttt` | ahc039 | ahc039 | UGTTT | 5 | 42 | 6/6 | 448 | 384 | 0.5232 | 1.2671 | COMPLETE |

## Reading `trajectories.jsonl`

One JSON object per rollout, 33 fields, including `reward_exec` (task score),
`reward_rmi`, `reward_total`, `correctness`, `uncertainty_true_mi`, the `ensemble_*`
disagreement metrics and the `streaming_mi_*` counters.

**Deduplicate before aggregating.** The two seed-42 ahc039 runs were killed by a disk
overflow on 2026-07-30 and resumed; the resumed process APPENDS to the same file, so
recomputed epochs appear twice. Group by `(epoch, group_idx, rollout_idx, adapter_idx)`
and keep the last occurrence — the raw vs deduped columns above show where this bites.
Without it you double-count, and for `ahc039-base` you also blend in epoch-3 rollouts
that scored `0.0` because the disk-full condition broke their evaluation, not the model.

**Do not trust the `best=` field in the stdout logs.** It is an in-memory tracker that
resets on resume: `ahc039-base` reports `best=1.2558` for epochs 3-5 having forgotten the
1.2662 it found at epoch 1. Compute the global best from the trajectories instead.

`*.corrupt-backup` files are the pre-repair originals: each had one truncated final JSON
line from the crash, dropped during recovery.

## Not included

- **LoRA checkpoints** (`ensemble_*/step_*/adapter.pt`) — 55 files, 3.09 GB, 58.6 MB each.
  Too large for git; they live on the VM's nvme volume only.
- **wandb run dirs** — already synced to the `ugttt-rebuttal` project on wandb.ai.

`ahc039-ugttt/ensemble_0/step_2.CORRUPT-truncated/` on the VM is a 2.6 MB partial write
(vs the full 61,444,223 B) left when the disk filled mid-`torch.save`. Not on the resume path.
