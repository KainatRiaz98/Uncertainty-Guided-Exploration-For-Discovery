# Running UG-TTT on a single AWS `g7e.48xlarge`

Guide for running the full experiment matrix on one 8-GPU AWS box instead of many
single-GPU instances. Written for the rebuttal-period runs, but applies to any
multi-run sweep.

## Why this instance

| | `g7e.48xlarge` |
|---|---|
| GPUs | 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 96 GB each (768 GB total) |
| vCPUs | 192 |
| System RAM | 2 TiB |
| Local NVMe | up to 15.2 TB |
| On-demand price | ≈ \$33/hr (us-east-1) |

The 96 GB per GPU is the **same card the published runs used**, so no memory
retuning is required — the chunked LM head, the `(K, chunk, V)` fp32 MI tensor,
and the sequential per-adapter backward all fit exactly as before.

Eight GPUs is the point: one training run per GPU, eight runs in parallel. For a
sweep of N runs, wall-clock is `ceil(N/8) × run_time`, not `N × run_time`.

An H200 box (`p5e.48xlarge` / `p5en.48xlarge`, ≈\$63/hr) is roughly twice the
price for memory this workload does not need. Prefer it only if you intend to
raise `num_ensemble_members`, `lora_rank`, or `max_tokens` beyond the published
configuration.

## Before you launch: service quota

`g7e.48xlarge` consumes 192 vCPUs of the **"Running On-Demand G and VT
instances"** quota. New accounts are typically far below this. Request the
increase **before** you need the box — approval can take hours to a day, and it
is the most common cause of a blocked launch.

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --desired-value 192 \
  --region us-east-1
```

## Instance setup

Launch with a Deep Learning AMI and a large root volume (checkpoints and
trajectory logs for 8 concurrent runs add up quickly):

```bash
aws ec2 run-instances \
  --instance-type g7e.48xlarge \
  --image-id <deep-learning-base-ami-id> \
  --key-name <your-keypair> \
  --security-group-ids <your-sg> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":1000,"VolumeType":"gp3"}}]' \
  --region us-east-1
```

Then, on the box:

```bash
git clone <this-repo> && cd Uncertainty-Guided-Exploration-For-Discovery
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/requirements-math.txt
export WANDB_API_KEY="..." WANDB_ENTITY="..."
nvidia-smi --list-gpus          # expect 8
nproc                           # expect 192
```

For the denoising domain, additionally install
`requirements/denoising/requirements-denoising.txt` and dry-run its verifier once
before committing a GPU to a full run.

## The CPU-oversubscription trap (read this)

Verification runs on CPU. [`utils/cpu_scheduler.py`](../utils/cpu_scheduler.py)
sizes its worker pool from `os.sched_getaffinity(0)` — the CPUs the process is
actually allowed to use — and partitions them into groups of
`num_cpus_per_task`.

If you start 8 runs plainly, **each one sees all 192 vCPUs** and builds a pool as
if it owned the whole machine. With `num_cpus_per_task=2` that is 96 verification
slots per run, 768 across the box, on 192 real cores — roughly 4× oversubscribed.
Every run then thrashes and the box gets *slower* than a single-GPU instance.

The fix is to pin each run to its own slice of cores. `sched_getaffinity`
respects `taskset`, so the scheduler self-limits with no code change:

```bash
# 192 vCPUs / 8 runs = 24 vCPUs per run
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23    python -m tinker_cookbook.rl.mlora_train ...
CUDA_VISIBLE_DEVICES=1 taskset -c 24-47   python -m tinker_cookbook.rl.mlora_train ...
```

[`scripts/aws/launch_wave.sh`](../scripts/aws/launch_wave.sh) does this
assignment automatically. Use it rather than launching by hand.

## Launching a wave

```bash
bash scripts/aws/launch_wave.sh wave1        # start 8 runs, one per GPU
bash scripts/aws/launch_wave.sh --list       # show the configured runs
tail -f logs/aws/wave1/<run-name>.log        # follow one run
nvidia-smi                                   # confirm 8 processes, one per GPU
```

Each run is detached (`setsid`), so closing the SSH session does not kill it.
Logs land in `logs/aws/<wave>/<run-name>.log`.

## Verifying the run mix is healthy

Within a few minutes of launch:

- `nvidia-smi` — exactly one python process per GPU, memory well under 96 GB.
- `uptime` — load average should sit near 192, not far above it. Much higher
  means the CPU pinning did not take effect.
- Each log should be producing rollouts; a run stuck before the first rollout is
  usually a missing task dependency, not a GPU problem.

## Hyperparameters that differ from the code defaults

The `Config` defaults in
[`tinker_cookbook/rl/mlora_train.py`](../tinker_cookbook/rl/mlora_train.py)
are the upstream TTT-Discover values, **not** the published UG-TTT ones. The
launch script sets these explicitly; if you launch by hand, pass them yourself:

| Flag | Code default | Published UG-TTT runs |
|---|---|---|
| `--lora_rank` | 32 | **16** |
| `--lora_alpha` | 64 | **32** |
| `--groups_per_batch` | 64 | **8** |
| `--num_epochs` | 50 | **6** |
| `--num_cpus_per_task` | 1 | **2** |

`--group_size 8`, `--learning_rate 4e-5`, `--kl_penalty_coef 0.01`,
`--max_tokens 26000`, and `--num_ensemble_members 5` match the defaults.

Note also that `Config.target_modules` defaults to `q_proj` and `o_proj` only,
while the paper's hyperparameter table lists all four attention projections
(q, k, v, o). Reconcile this before publishing the configuration.

## Arm → flag mapping

| Arm | Flags |
|---|---|
| UG-TTT (full) | `--num_ensemble_members 5 --rmi_coef 0.1 --uncertainty_metric rmi` |
| Baseline (TTT-Discover) | `--num_ensemble_members 1 --rmi_coef 0.0` |
| MI bonus off (α=0) | `--num_ensemble_members 5 --rmi_coef 0.0` |
| Entropy bonus | `--num_ensemble_members 5 --rmi_coef 0.1 --uncertainty_metric predictive_entropy` |

`--uncertainty_metric` already supports `rmi`, `variance`, and
`predictive_entropy` ([`tinker_cookbook/rl/uncertainty.py`](../tinker_cookbook/rl/uncertainty.py)),
so the entropy-bonus arm needs no new code.

Environment strings are `ac1`, `ac2`, and `cp` (circle packing takes its size
from `--problem_idx`, e.g. `--problem_idx 26`). Confirm the string for `erdos`
and `denoising` against `cli_main()` before launching those.

## Two blockers that are not solved by hardware

1. **No seed flag.** [`ensemble.py:86`](../tinker_cookbook/rl/ensemble.py)
   hardcodes `torch.manual_seed(42 + k * 1000)`, and no `--seed` argument is
   parsed. Every run initialises identically, so multi-seed experiments cannot be
   produced by launching the same command twice. A `--seed` argument must be
   added and threaded into ensemble init and rollout sampling first.
2. **No nuclear-norm regulariser in the training loop.** The loss assembled at
   `mlora_train.py:752-790` is the per-adapter policy-gradient term only; a
   repository-wide search for `nuc`, `svdvals`, `linalg.svd`, `matrix_norm`, and
   similar finds no implementation, and no `λ_NNM` argument is parsed. Any run
   launched from this tree is an unregularised ensemble.

Resolve both before spending GPU hours, or the runs will not answer the
questions they were designed to answer.

## Cost planning

At ≈\$33/hr:

| | Wall-clock | Cost |
|---|---|---|
| One wave of 8 runs | ~24–32 h | ~\$800–1,050 |
| Two waves (16 runs) | ~48–64 h | ~\$1,600–2,100 |

Stop the instance the moment the last wave finishes — an idle box bills at the
same rate. Use on-demand rather than spot for long runs; a spot reclaim mid-run
loses the whole run.
