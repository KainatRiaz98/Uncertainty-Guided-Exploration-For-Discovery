# UG-TTT Rebuttal — Experiment Plan (3 nodes × 8 A100 = 24 GPUs)

Deadlines: initial response **Jul 27** (target Jul 26) · discussion **Jul 27–Aug 3** ·
authors locked out **Aug 3**. Results can be posted throughout Phase 2, so the Jul 27
response carries clarifications + analyses + commitments, and run results land as they finish.

Runnable assets live in the **public** repo `epistemic-uncertainty-for-test-time-discovery`
under `scripts/cluster/` (preflight, smoke test, launcher, monitor, wave manifests).
This file is the private plan and the reviewer mapping.

## Capacity

24 GPUs × ~24–36 h per run. Three waves fit before Aug 3:

| Wave | Launch | Expect done | Purpose |
|---|---|---|---|
| 1 | Jul 24–25 | Jul 26–27 | Variance evidence + component isolation |
| 2 | Jul 27 | Jul 28–29 | New domains, new models, larger-scale config |
| 3 | Jul 29–30 | Jul 31–Aug 1 | Tighter error bars + reviewer follow-ups |

Wave 1 results land in time for the initial response. Wave 3 is deliberately half
reserve — by Jul 29 you know what reviewers actually asked in discussion.

## Wave 1 — 24 runs (the score-movers)

**Seeds, all four tasks, both arms, seeds 2 and 3 (16 runs).** With the published
run as seed 42 this gives n=3 per cell. Answers **DmAa Q1** (who asked only for AC1
and CP26 — we deliver all four), **vGzb W1** ("apparently only one random seed"),
and **vm7q Q2** ("how statistically significant").
Report mean ± std and min/max for R_max and final-epoch H. With n=3 do **not** claim
p-values; say only whether the arm gap exceeds the observed spread.

**Component ablations on CP26 at identical budget (6 runs).**

| Run | Isolates | Answers |
|---|---|---|
| `cp26-alpha0` | NNM on, MI bonus off (α=0) | **DmAa Q2** — is the gain the ensemble, not the epistemic signal? |
| `cp26-entropy` | token-entropy bonus instead of MI | **DmAa Q2** ("ideally"), **8mjY W1**, **vGzb W2** |
| `cp26-nonnm` | reproduce Table 2 under seed control | **vGzb W2** |
| `cp26-variance` | variance instead of BALD MI | **8mjY W1** — epistemic *specifically* |
| `cp26-k3`, `cp26-k10` | ensemble size | **vm7q Q2** ("why 5?"), **vGzb W2** ("multi-adapter ensemble") |

**Uniform-configuration table (2 runs).** `ac1-nostream`, `cp26-nostream` complete a
Table 1 with streaming OFF everywhere (AC2/Erdős no-stream numbers already exist in
App. F). Answers **DmAa W2** — the headline stops depending on a post-hoc per-problem
choice. If UG-TTT still wins here, present this as the new primary table and demote
streaming to an efficiency result.

## Wave 2 — 24 runs (generality)

**New domains (8).** `denoising` (biology), `trimul` (GPU kernels — and these are
A100s, the exact hardware TTT-Discover reported TriMul on), `ahc039` (algorithm
design). Directly answers **8mjY W3** ("restricted to the mathematical subset") and
**vGzb W1**. Two seeds on denoising so the new-domain claim is not itself n=1.

**Models (8).** Qwen3-14B on CP26 and AC1, Qwen3-32B on CP26 (80GB only), and
Qwen2.5-7B as a different model *family*. Answers **8mjY W3** and **vGzb Q1**.
GPT-OSS-120B remains out of reach — say so plainly and show the 8B→14B→32B trend
instead; a monotone trend is the honest, checkable version of that claim.

**Scale of the RL setup (6).** `bigcfg` = rank 32 + group 32 (4× the rollouts) and
`2x` = 12 epochs, each with a matched baseline. Answers **vGzb Q2** ("why were
parameters reduced, would gains hold at original scale"). Full original scale
(25,600 rollouts) is not reachable; state that and report the trend you can show.

**Seed 4 on CP26 (2).** Pushes the headline task to n=4.

## Wave 3 — 24 runs (half reserve)

Seeds 4–5 on the headline tasks, second seeds for every ablation (so the ablation
table also carries a spread), and 8 slots explicitly reserved for whatever reviewers
raise after Jul 27.

## No-GPU work — do this first, it carries the Jul 27 response

| Analysis | Answers | Status |
|---|---|---|
| Taxonomy audit: % "other" per task/condition, when the regexes were fixed, label-free clustering check | **DmAa Q3** | TODO |
| MI→novelty: U_i of first-occurrence-of-family rollouts vs repeats; MI at family-defining tokens vs boilerplate | **8mjY W1** | TODO |
| C1/C2/C5/CP26 constants table vs published records | **vm7q Q3** | TODO |
| Clarity pack: frozen base, K=5, β is bisection-solved not scheduled, "within-group", Props 1–2 in plain English, page-number fix | **vm7q** all, **AC** | TODO |
| Claim-tempering list: exact wording changes for "disagreement = epistemic = novelty" | **8mjY W1** | TODO |

## Code status

- **`--seed` — DONE.** Added to `ug_ttt/rl/mlora_train.py` (Config field, argparse,
  `seed_everything()`, threaded to `LoRAEnsemble` and `create_sampler`),
  `ug_ttt/rl/ensemble.py` (`torch.manual_seed(seed + k*1000)`), and
  `ug_ttt/recipes/ttt/sampler.py` (module seed replacing hardcoded `default_rng(12345)`
  / `default_rng(42)`). seed=42 reproduces the published behaviour.
  **This mattered more than expected:** the sampler's hardcoded `12345` meant every
  AC1/AC2 run started from a byte-identical initial construction, so "extra seeds"
  without this patch would not have been independent runs.
- **Ablation arms — no code needed.** `--rmi_coef 0.0`, `--uncertainty_metric
  predictive_entropy|variance`, `--nnm_coef 0.0`, `--num_ensemble_members N` all exist.
- **Open:** `--num_epochs` (run.sh 10 vs paper 6) and
  `--streaming_mi_threshold_percentile` (run.sh 5.0 vs paper Table 4's 25.0) must be
  reconciled before publishing numbers. Launcher currently uses 6 and 25.0.
- A100 40GB vs 80GB is unconfirmed; it decides whether the 32B rows in wave 2 are viable.

## Reporting conventions

- Always label new tables "new experiments run during the author response period".
- Include the paper's original run as seed 42 in every seed table.
- Ablations all at 384 rollouts, matching Table 2's budget.
- No wandb links in responses — the rules forbid links.
