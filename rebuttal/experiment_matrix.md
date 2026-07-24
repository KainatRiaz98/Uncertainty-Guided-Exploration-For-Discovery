# UG-TTT NeurIPS Rebuttal — Experiment Matrix

Deadlines: initial response **Jul 27** (target Jul 26) · last results postable **Aug 3** · last safe 32h-run start **Aug 1 AM**.
One run ≈ 32h on RTX Pro 6000 96GB. Update Status in place; keep one row per run.

## P0 — direct reviewer questions (start first)

| # | Run | Task | Config | Answers | Status | Result (Rmax / H_final) |
|---|-----|------|--------|---------|--------|--------------------------|
| 1 | ugttt-seed2 | AC1 | full UG-TTT, new seed | DmAa Q1, vGzb W1, vm7q Q2 | TODO | |
| 2 | ugttt-seed3 | AC1 | full UG-TTT, new seed | DmAa Q1 | TODO | |
| 3 | baseline-seed2 | AC1 | K=1 TTT-Discover, new seed | DmAa Q1 | TODO | |
| 4 | baseline-seed3 | AC1 | K=1 TTT-Discover, new seed | DmAa Q1 | TODO | |
| 5 | ugttt-seed2 | CP26 | full UG-TTT, new seed | DmAa Q1 | TODO | |
| 6 | ugttt-seed3 | CP26 | full UG-TTT, new seed | DmAa Q1 | TODO | |
| 7 | baseline-seed2 | CP26 | K=1 TTT-Discover, new seed | DmAa Q1 | TODO | |
| 8 | baseline-seed3 | CP26 | K=1 TTT-Discover, new seed | DmAa Q1 | TODO | |
| 9 | alpha0 | CP26 | K=5, NNM on, MI bonus OFF (α=0) | DmAa Q2, vGzb W2, 8mjY W1 | TODO | |
| 10 | entropy-bonus | CP26 | K=5, NNM on, token-entropy bonus instead of MI | DmAa Q2, vGzb W2, 8mjY W1 | TODO | |

## P1 — kills post-hoc-selection criticism + adds a domain

> **Note:** if ≥6 VMs are available, launch the denoising pair (13–14) on day 1 in parallel with P0 — it is the highest-risk run (new environment, unproven verifier setup) and needs the longest runway. Set up `requirements/denoising/` and dry-run the verifier before committing a VM. P0 stays first-priority because DmAa's questions name AC1/CP26 explicitly and ablations must run on the tasks where the Table 1 claims live.

| # | Run | Task | Config | Answers | Status | Result |
|---|-----|------|--------|---------|--------|--------|
| 11 | nostream | AC1 | full UG-TTT, streaming OFF | DmAa W2 | TODO | |
| 12 | nostream | CP26 | full UG-TTT, streaming OFF | DmAa W2 | TODO | |
| 13 | denoising-baseline | denoising | K=1 TTT-Discover | 8mjY W3, vGzb W1 | TODO | |
| 14 | denoising-ugttt | denoising | full UG-TTT | 8mjY W3, vGzb W1 | TODO | |

## P2 — scale evidence (only if VMs remain)

| # | Run | Task | Config | Answers | Status | Result |
|---|-----|------|--------|---------|--------|--------|
| 15 | model14b-baseline | CP26 | Qwen3-14B, K=1 | 8mjY W3, vGzb Q1 | TODO | |
| 16 | model14b-ugttt | CP26 | Qwen3-14B, full UG-TTT | 8mjY W3, vGzb Q1 | TODO | |
| 17 | budget2x-baseline | CP26 | K=1, 12 epochs | vGzb Q2 | TODO | |
| 18 | budget2x-ugttt | CP26 | full UG-TTT, 12 epochs | vGzb Q2 | TODO | |

## No-GPU analyses (from existing logs — do before Jul 26)

| Analysis | Answers | Status |
|----------|---------|--------|
| Taxonomy audit: % "other" per task/condition; when regexes were fixed; label-free clustering robustness check | DmAa Q3 | TODO |
| MI→novelty: U_i of first-occurrence-of-family rollouts vs repeat rollouts; MI at family-defining tokens vs boilerplate | 8mjY W1 | TODO |
| C1/C2/C5/CP26 constants table vs published records | vm7q Q3 | TODO |
| Clarity pack: frozen-base why, K=5 why, β schedule, "within-group" definition, Props 1–2 plain-English | vm7q all, AC | TODO |

## Code changes needed before launching — VERIFIED against the tree 2026-07-24

**BLOCKER 1 — no seed argument.** `tinker_cookbook/rl/ensemble.py:86` hardcodes
`torch.manual_seed(42 + k * 1000)`; `cli_main()` in `mlora_train.py` parses no
`--seed`. Launching the same command twice reproduces the same run. **Every P0
seed run (#1–8) is blocked until a `--seed` flag is added** and threaded into
ensemble init and rollout sampling. Highest-priority code fix.

**RESOLVED — NNM exists.** The regulariser is implemented in the *other* repo,
`epistemic-uncertainty-for-test-time-discovery`: `ug_ttt/rl/nuclear_norm.py`,
exposed as `--nnm_coef` (default 0.0; paper uses 0.075) plus `--nnm_use_fro_norm`.
Streaming MI is there too (`--streaming_mi` + 6 sub-flags). **All runs must be
launched from that repo**, module `ug_ttt.rl.mlora_train`. This repo is the
TTT-Discover fork and lacks both. `target_modules` there is q/k/v/o, matching the
paper.

- [x] `alpha0` arm — no code needed: `--rmi_coef 0.0 --nnm_coef 0.075`.
- [x] `entropy-bonus` arm — no code needed: `--uncertainty_metric predictive_entropy`.
- [x] `no-NNM` arm — no code needed: `--nnm_coef 0.0`.
- [ ] **Add `--seed`** (Blocker 1) — gates runs #1–8.
- [ ] Decide `--num_epochs`: `scripts/run.sh` uses 10, paper reports 6.
- [ ] Decide `--streaming_mi_threshold_percentile`: `run.sh` uses 5.0, paper
      Table 4 states 25th percentile.
- [ ] Confirm the `denoising` env string in `cli_main()` (`ac1`, `ac2`, `cp`,
      `erdos` confirmed; circle-packing size comes from `--problem_idx`).
- [ ] Denoising env: install `requirements/denoising/requirements-denoising.txt`;
      dry-run the verifier once before committing a GPU.

## Hardware

One AWS `g7e.48xlarge` (8 × RTX PRO 6000 96 GB, 192 vCPUs). Setup, the CPU-pinning
requirement, and cost planning: [docs/aws_g7e48xlarge.md](../docs/aws_g7e48xlarge.md).
Launcher: `bash scripts/aws/launch_wave.sh wave1`.
