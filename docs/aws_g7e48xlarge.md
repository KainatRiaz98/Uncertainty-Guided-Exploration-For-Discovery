# Running UG-TTT on a single AWS `g7e.48xlarge`

Guide for running the full experiment matrix as 8 parallel runs on one 8-GPU AWS
box instead of serialising across single-GPU instances.

> **Which repo runs the experiments.** The UG-TTT implementation lives in
> `epistemic-uncertainty-for-test-time-discovery` (module `ug_ttt.rl.mlora_train`,
> canonical launcher `scripts/run.sh`). Clone **that** repo on the AWS box.
> This repo is the TTT-Discover fork and does not contain the nuclear-norm
> regulariser or the streaming-MI implementation.

## Why this instance

| | `g7e.48xlarge` |
|---|---|
| GPUs | 8 × NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU memory | 96 GB each (768 GB total) |
| vCPUs | 192 |
| System RAM | 2 TiB |
| Local NVMe | up to 15.2 TB |
| On-demand price | ≈ \$33/hr (us-east-1) |

96 GB per GPU is the **same card the published runs used**, so the chunked LM
head, the `(K, chunk, V)` fp32 MI tensor, and the sequential per-adapter backward
all fit without retuning.

Eight GPUs is the point: one run per GPU. For N runs, wall-clock is
`ceil(N/8) × run_time`, not `N × run_time`.

An H200 box (`p5e.48xlarge` / `p5en.48xlarge`, ≈\$63/hr) is roughly twice the
price for memory this workload does not need. Prefer it only when raising
`num_ensemble_members`, `lora_rank`, or the token budget beyond published values.

## Before you launch: service quota

`g7e.48xlarge` consumes 192 vCPUs of the **"Running On-Demand G and VT
instances"** quota. Request the increase early — approval can take hours to a
day and is the most common cause of a blocked launch.

```bash
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-DB2E81BA \
  --desired-value 192 \
  --region us-east-1
```

## Instance setup

```bash
aws ec2 run-instances \
  --instance-type g7e.48xlarge \
  --image-id <deep-learning-base-ami-id> \
  --key-name <your-keypair> \
  --security-group-ids <your-sg> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":1000,"VolumeType":"gp3"}}]' \
  --region us-east-1
```

On the box:

```bash
git clone https://github.com/KainatRiaz98/epistemic-uncertainty-for-test-time-discovery.git
cd epistemic-uncertainty-for-test-time-discovery
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/requirements-math.txt
export WANDB_API_KEY="..." WANDB_ENTITY="..."
nvidia-smi --list-gpus     # expect 8
nproc                      # expect 192
```

Copy `scripts/aws/launch_wave.sh` from this repo into that checkout, or run a
single job first with its own `scripts/run.sh` to confirm the environment.

For the denoising domain, additionally install
`requirements/denoising/requirements-denoising.txt` and dry-run its verifier once
before committing a GPU.

## Two things that break 8 concurrent runs

**1. CPU oversubscription.** Verification runs on CPU.
[`utils/cpu_scheduler.py`](../utils/cpu_scheduler.py) sizes its worker pool from
`os.sched_getaffinity(0)` and partitions it into groups of `num_cpus_per_task`.
Start 8 runs plainly and **each sees all 192 vCPUs**, building a pool as if it
owned the machine — with `num_cpus_per_task=2` that is 96 slots per run, 768
across the box, on 192 real cores. Every run thrashes and the box ends up slower
than a single-GPU instance.

`sched_getaffinity` respects `taskset`, so pinning each run to its own slice
makes the scheduler self-limit with no code change:

```bash
# 192 vCPUs / 8 runs = 24 vCPUs per run
CUDA_VISIBLE_DEVICES=0 taskset -c 0-23  python3 -m ug_ttt.rl.mlora_train ...
CUDA_VISIBLE_DEVICES=1 taskset -c 24-47 python3 -m ug_ttt.rl.mlora_train ...
```

**2. Shared Ray cluster.** `scripts/run.sh` sets `RAY_ADDRESS=auto`, which joins
an existing cluster. `cpu_scheduler.py` registers a **detached actor named
`cpu_scheduler`**, so 8 runs on one Ray cluster collide on that name and share a
single CPU pool. Give each run its own local Ray instance by leaving
`RAY_ADDRESS` unset.

[`scripts/aws/launch_wave.sh`](../scripts/aws/launch_wave.sh) handles both. Use
it rather than launching by hand, and smoke-test one run before filling the box.

## Launching a wave

```bash
bash scripts/aws/launch_wave.sh --list        # preview
bash scripts/aws/launch_wave.sh wave2         # start 8 runs, one per GPU
tail -f logs/aws/wave2/<run-name>.log
nvidia-smi                                    # one process per GPU
uptime                                        # load ≈ 192, not far above
```

Runs are detached with `setsid`, so closing SSH does not kill them.

## Configuration reference

The argparse defaults in `ug_ttt/rl/mlora_train.py` are upstream TTT-Discover
values, **not** the published UG-TTT ones. The launcher passes these explicitly:

| Flag | Argparse default | Published UG-TTT |
|---|---|---|
| `--lora_rank` | 32 | **16** |
| `--lora_alpha` | 64 | **32** |
| `--groups_per_batch` | 64 | **8** |
| `--num_epochs` | 50 | **6** |
| `--nnm_coef` | 0.0 | **0.075** |
| `--num_cpus_per_task` | 1 | **2** |
| `--uncertainty_metric` | `true_mi` | `true_mi` |

`--group_size 8`, `--learning_rate 4e-5`, `--kl_penalty_coef 0.01`,
`--num_ensemble_members 5`, and `--two_phase_sampling --phase1_max_tokens 26000`
match both the defaults and the paper.

### Arm → flag mapping

| Arm | Flags |
|---|---|
| UG-TTT (full) | `--num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric true_mi` |
| Baseline (TTT-Discover) | `--num_ensemble_members 1 --rmi_coef 0.0 --nnm_coef 0.0` |
| MI bonus off (α=0) | `--num_ensemble_members 5 --rmi_coef 0.0 --nnm_coef 0.075` |
| Entropy bonus | `--num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.075 --uncertainty_metric predictive_entropy` |
| No NNM (paper Tab. 2) | `--num_ensemble_members 5 --rmi_coef 0.1 --nnm_coef 0.0` |

`--uncertainty_metric` supports `true_mi`, `rmi`, `variance`, and
`predictive_entropy`, so the entropy-bonus ablation needs **no new code**.

Environment strings: `ac1`, `ac2`, `cp` (size from `--problem_idx`, e.g. `26`),
`erdos`. Streaming MI is opt-in via `--streaming_mi` plus its sub-flags.

## Open issues to settle before the runs

1. **No `--seed` argument.** `ug_ttt/rl/ensemble.py:88` hardcodes
   `torch.manual_seed(42 + k * 1000)` and no `--seed` is parsed. Launching the
   same command twice reproduces the same run, so multi-seed experiments are not
   currently possible. Must be added and threaded into ensemble init and rollout
   sampling before any seed run.
2. **`--num_epochs`**: `scripts/run.sh` uses 10; the paper reports 6. Reruns
   should match whichever produced the published numbers.
3. **`--streaming_mi_threshold_percentile`**: `scripts/run.sh` uses 5.0; paper
   Table 4 states the 25th percentile. Reconcile before rerunning streamed arms.

## Cost planning

At ≈\$33/hr:

| | Wall-clock | Cost |
|---|---|---|
| One wave of 8 runs | ~24–32 h | ~\$800–1,050 |
| Two waves (16 runs) | ~48–64 h | ~\$1,600–2,100 |

Stop the instance as soon as the last wave finishes — an idle box bills at the
same rate. Use on-demand, not spot: a reclaim mid-run loses the whole run.
