# AWS node handoff — UG-TTT NeurIPS rebuttal runs

Paste-able brief for a fresh Claude session on the AWS instance.
Repo: `KainatRiaz98/Uncertainty-Guided-Exploration-For-Discovery`, branch
**`aws-multi-gpu-launch`**. Everything below is already committed and pushed.

## Context

NeurIPS 2026 submission 17343 (UG-TTT). Author response due **Jul 27**, all
posting ends **Aug 3**. These runs produce the numbers for that response.
3 nodes × 8 A100-80GB = 24 GPUs.

The runnable module is **`tinker_cookbook.rl.mlora_train`** (NOT `ug_ttt.*` —
that package name only exists in the public mirror repo).

## Launch commands

All runs: 6 epochs, checkpoint every 2 epochs, streaming OFF (except one
deliberate wave2 comparison pair). Always `--list` first.

```bash
bash scripts/aws/launch_wave.sh --list
```

| Node | Command | What it is |
|---|---|---|
| 1 | `bash scripts/aws/launch_wave.sh wave1` | 8 seed runs: seeds 2,3 × {AC1, CP26} × {UG-TTT, baseline} |
| 2 | `bash scripts/aws/launch_wave.sh wave2` | 4 CP26 ablations + streaming/no-stream pair + AC1 + Erdős |
| 3 | `bash scripts/aws/launch_wave.sh wave3` | Qwen3-14B/32B model-scaling pairs (6 runs) |
| 3 (alt) | `bash scripts/aws/launch_heavy_domain_wave.sh smoke` then `run` | NEW domains on Qwen2.5-72B (see below) |

`wave3` and the heavy-domain wave both target node 3 — pick one, or run the
heavy wave after wave3 finishes.

## Before launching, on each node

```bash
export WANDB_API_KEY=<key>
export WANDB_ENTITY=kriaz-msee20seecs
```

`ml_log.py` **silently skips wandb** if the key is unset — you would lose all
remote logging without an error. Verify with `wandb login --verify $WANDB_API_KEY`.

Ray starts automatically (the launcher runs `ray start --head` if none is up).
To do it manually:

```bash
ray start --head --num-cpus=$(nproc) --disable-usage-stats
```

**One shared Ray head per node, and no `taskset`.** The task layer hardcodes
`ray.init("auto")` and `base_reward_task.py` get-or-creates a host-keyed
`cpu_scheduler` actor that partitions CPUs across all co-resident runs. Pinning
with `taskset` fights it. Every run must pass the same `--num_cpus_per_task`.

## Extra setup — only for the new-domain wave

Neither domain runs until its verifier deps exist **on that node**:

- **denoising** (single-cell biology): install
  `requirements/denoising/new_reqs.txt`, then git-install `simscity` and
  `molecular-cross-validation`, clone `openproblems` and
  `git apply requirements/denoising/openproblems_api_fix.patch`. See
  `requirements/denoising/README.md`. Watch the documented NumPy-2.x-vs-torch
  conflict. Pancreas data downloads to `~/.cache/denoising_datasets` on first run.
  **This is the fragile one — budget setup time.**
- **ahc039** (AtCoder heuristics, C++): `pip install -r requirements/requirements-ahc.txt`,
  ensure Docker works, `docker pull yimjk/ale-bench:cpp20-202301`. Inputs and
  judges are vendored in-repo, and compilation is container-isolated — more
  reliable than denoising.
- **heavy-model wave only**: `pip install bitsandbytes` (required for `nf4`
  quantized loading; not in any requirements file).

## The heavy-model wave (`launch_heavy_domain_wave.sh`)

8 runs on Qwen2.5-72B-Instruct at `--precision nf4` (QLoRA), **1 GPU per run**.
True 2-GPU model sharding was investigated and rejected: mLoRA's `pipeline`
mode is multi-process RPC, training-only, with no generation path, and the
custom forward has no cross-device hops. Quantization was already fully
implemented, so a 70B model fits on one 80GB GPU — 8 runs instead of 4.

Qwen2.5 is not a `<think>` model, so these runs are **single-phase** (no
`--two_phase_sampling`); prompts are wrapped in plain ChatML.

**Run the smoke test first — this combination has no prior successful run.**

```bash
bash scripts/aws/launch_heavy_domain_wave.sh smoke
```

Before launching the other 6, confirm in each smoke log:
- nf4 load succeeded (no bitsandbytes / CUDA error)
- completions are real code, not garbled tokens (confirms ChatML wrapping)
- `train/correctness/nonzero > 0` for at least one epoch
- **no** `eos_id_ != <|im_end|>` warning (that means generation won't stop and
  every rollout runs to `--max_tokens`)

Do **not** raise `--max_tokens` above ~24000 in this script. The single-phase
decode loop has no context-window bound, unlike the two-phase path, so a larger
value generates past the 32768 context.

If completions look degenerate, retry the smoke test at `--precision int8`
before abandoning the wave.

## Results — do not lose them

`logs/` is **gitignored**. Everything lives only on the instance:
`--log_path` holds `metrics.jsonl`, `trajectories.jsonl`, `last_epoch.txt` and
ensemble checkpoints; stdout goes to `logs/aws/<wave>/<run>.log`.

**`rsync` the logs off before terminating any instance.** Parse `metrics.jsonl`
for the rebuttal tables — the response rules forbid links, so wandb is
convenience only.

Runs resume from `last_epoch.txt` (losing at most 2 epochs) if restarted with
the same `--log_path`.

## Gotchas that have already bitten

- `--save_every 2` counts **epochs**, not steps.
- Published seed-42 runs used streaming **ON**; the new streaming-OFF seeds
  cannot be pooled with them into one n=3 cell — different config.
- `--seed 42` reproduces published behaviour exactly; other seeds give
  independent adapter inits *and* independent initial constructions.
- The mlora path dispatches only `ac1/ac2/cp/erdos/denoising/ahc039/ahc058`.
  `trimul`/`mla` need paid Modal H100/H200 and are deliberately out of scope.
- `cp32` needs no new code: `--env cp --problem_idx 32`.
