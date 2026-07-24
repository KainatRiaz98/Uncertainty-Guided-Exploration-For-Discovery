# Running UG-TTT on a single AWS `g7e.48xlarge`

Guide for running the full experiment matrix as 8 parallel runs on one 8-GPU AWS
box instead of serialising across single-GPU instances.

> **Which repo runs the experiments.** Clone **this** repo on the AWS box and
> run the `aws-multi-gpu-launch` branch (module `tinker_cookbook.rl.mlora_train`).
> As of the feat/parallelism rebase this branch contains the nuclear-norm
> regulariser (`tinker_cookbook/rl/nuclear_norm.py`) and the streaming-MI
> implementation. The public `epistemic-uncertainty-for-test-time-discovery`
> repo is the mirror where the same code lives under the `ug_ttt.*` package name.

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
git clone https://github.com/KainatRiaz98/Uncertainty-Guided-Exploration-For-Discovery.git
cd Uncertainty-Guided-Exploration-For-Discovery
git checkout aws-multi-gpu-launch
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

## Concurrency model: one shared Ray head per node

The task layer decides this, and the launcher follows it — do **not** use the
per-run-`taskset` / separate-Ray model an earlier draft suggested; it is wrong
for this code.

**Why one head.** The verifiers hardcode `ray.init("auto")`
([`tasks/alphaevolve_ac2/task.py:72`](../tasks/alphaevolve_ac2/task.py),
[`tasks/erdos_min_overlap/task.py:69`](../tasks/erdos_min_overlap/task.py),
[`utils/cpu_scheduler.py:97`](../utils/cpu_scheduler.py)), so every run needs a
running Ray head — with none, `ray.init("auto")` **crashes the run** at first
verification. `scripts/run.sh` already sets `RAY_ADDRESS=auto` for this reason.

**Why no `taskset`.** [`tasks/base_reward_task.py:460-476`](../tasks/base_reward_task.py)
get-or-creates a **detached, host-keyed `cpu_scheduler` actor** that partitions
this node's CPUs into groups of `num_cpus_per_task` across *all* co-resident
runs. That is the intended CPU-sharing mechanism. Pinning each run with
`taskset` fights it and can leave cores idle or oversubscribed. Instead:

- Start one head per node, sized to the whole box: `ray start --head --num-cpus=<total>`.
- Give every run the **same** `--num_cpus_per_task` (the launcher uses 2).
- Let the scheduler divide the cores.

[`scripts/aws/launch_wave.sh`](../scripts/aws/launch_wave.sh) does all of this:
it exports `RAY_ADDRESS=auto`, starts a head if none is running, and launches
one run per GPU with no `taskset`. Use it rather than launching by hand, and
smoke-test one run before filling the box.

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

The argparse defaults in `tinker_cookbook/rl/mlora_train.py` are upstream TTT-Discover
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

1. **`--seed` — implemented.** `--seed N` (default 42) seeds Python/NumPy/Torch
   via `seed_everything()`, threads a base into per-adapter init
   (`ensemble.py`, `seed + k*1000`), and shifts the initial-construction RNGs
   (`sampler.py::set_initial_state_seed`). Seed 42 reproduces the published
   constants exactly; other seeds give independent adapter inits and starting
   constructions. Residual: `sampler.py:412`'s standalone `default_rng()` is
   still unseeded — acceptable (it adds independence, not a reproducibility
   hazard) but note it if you need bitwise repeatability.
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
