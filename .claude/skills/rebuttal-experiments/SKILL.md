---
name: rebuttal-experiments
description: Manage the UG-TTT rebuttal experiment matrix — run configs, VM assignments, statuses, and converting finished runs into rebuttal-ready markdown tables. Use when launching runs, recording results, or checking what is left before the Aug 3 deadline.
---

# Rebuttal Experiment Tracker — UG-TTT (Submission 17343)

Single source of truth: [rebuttal/experiment_matrix.md](../../rebuttal/experiment_matrix.md) in the repo root. Always read it first, update it after any run launches or finishes, and regenerate the summary tables from it.

## Statuses

`TODO` → `RUNNING (vm-name, started YYYY-MM-DD HH:MM)` → `DONE (Rmax=…, H=…)` or `FAILED (reason)`.

## Deadlines that gate everything

- Initial rebuttal posted: **Jul 27, 2026** (target Jul 26). Runs started Jul 24 finish ~Jul 25–26 and CAN make the initial response.
- Last possible posting of results: **Aug 3**. A 32h run must therefore START by ~Aug 1 morning at the absolute latest.
- One run ≈ 32h on RTX Pro 6000 96GB (baseline K=1 runs are somewhat faster).

## Arm definitions (exact configs — keep these consistent across all runs)

All arms use the paper's Table 3/4 hyperparameters unless stated. Base: Qwen3-8B, K=5 rank-16, λ_NNM=0.075, α=0.1, G=8, 8 groups/epoch, 6 epochs.

1. **ugttt-seedN** — full UG-TTT, new random seed N (change only the seed; streaming per Table 1 setting for that task).
2. **baseline-seedN** — K=1, α=0, λ_NNM=0, no streaming (TTT-Discover config), new seed.
3. **alpha0** — K=5, NNM ON (λ_NNM=0.075), MI bonus OFF (α=0). Isolates "does the epistemic bonus matter beyond ensemble diversity". (DmAa Q2, vGzb W2)
4. **entropy-bonus** — K=5, NNM ON, but replace U_i with single-model mean token entropy H(p_k) of the generating adapter over the same top-7% positions; identical standardization, clipping, and β–γ coupling, same α=0.1. Tests "cheaper diversity signal would do the same". (DmAa Q2 "ideally", 8mjY W1, vGzb W2)
5. **nostream** — full UG-TTT with streaming early-stop disabled, on AC1 and CP26 (the two tasks where Table 1 used streaming). Enables a uniform-config main table. (DmAa W2)
6. **denoising-{baseline,ugttt}** — new domain (single-cell denoising, biology). Task harness exists: `tasks/denoising/`, launcher `scripts/single_host/denoising.sh`. Score metric per TTT-Discover (MAGIC baseline 0.64, TTT-Discover 0.71). (8mjY W3, vGzb W1)
7. **model14b-{baseline,ugttt}** — second base model (Qwen3-14B fits 96GB with K=5 rank-16 + chunked scoring; Qwen3-32B only if an H200/141GB VM is available), one task (CP26 recommended — clearest headline). (8mjY W3, vGzb Q1)
8. **budget2x** (stretch) — UG-TTT + baseline on CP26 at doubled budget (12 epochs), to answer "would gains hold at larger scale". (vGzb Q2)

## Priority order (if VMs are scarce, cut from the bottom)

P0: 2× ugttt-seed + 2× baseline-seed on AC1; same on CP26 (8 runs) + alpha0 (CP26) + entropy-bonus (CP26).
P1: nostream-AC1, nostream-CP26, denoising-baseline, denoising-ugttt.
P2: model14b pair; budget2x pair.

## Reporting conventions (what goes into the rebuttal)

- Seeds: report per-seed values AND mean ± std over all seeds (including the paper's original run as seed 1), for both Rmax and final-epoch H. Also report min/max. With n=3, do not claim p-values; say "the UG-TTT–baseline gap exceeds the run-to-run spread" only if it actually does.
- Ablation table (CP26, all at 384 rollouts): Full UG-TTT / alpha0 / entropy-bonus / no-NNM (paper Table 2) / baseline — columns: Rmax, final H, mean MI at epoch 5.
- Uniform-config table: Table 1 rebuilt with streaming OFF for all four tasks (AC2/Erdős numbers already exist in the paper's App. F; add the new AC1/CP26 no-stream runs).
- Always mark tables "new experiments run during the author response period" — reviewers and ACs value this explicitly.
- Log every finished run's key numbers into the matrix file immediately; wandb links must NOT go into the rebuttal (no links allowed).

## Sanity checks before reporting any number

- Confirm the verifier version and task config are identical to the paper's runs (prompts/verifiers unchanged from TTT-Discover).
- Confirm Rmax is read as best-during-training (cumulative), matching Table 1's definition.
- For entropy H: final-epoch family entropy over correct rollouts, same regex taxonomy, same "other" handling.
