# scripts/iclr — approved ICLR-resubmission rerun launchers

Built 2026-08-13. **Nothing in this directory has been executed.** Every
launcher has a `check` mode (preflight only, launches nothing) and a `run`
mode. Run `check` first, always.

Priority order:

| # | Script | Run | Where | ETA | Status |
|---|--------|-----|-------|-----|--------|
| 1 | `launch_denoising_seed2_base.sh` | denoising baseline, seed 2, ctx 32768 | RTX PRO 6000 box (GPU 0) | ~15 h | ready, not launched |
| 1 | `launch_denoising_seed2_ugttt.sh` | denoising UG-TTT, seed 2, ctx 32768 | RTX PRO 6000 box (GPU 1) | ~19.5 h | ready, not launched |
| 2 | `../aws/launch_gamma_ablation.sh` | CP26 γ-decoupled, seed 42, ctx 8192 | wave box | ~11.9 h (4 ep) | pre-existing; see config note below |
| 3 (optional) | `launch_cp26_alpha0_seed2.sh` | CP26 α=0 arm, seed 2, ctx 8192 | wave box | ~18 h (6 ep) | ready, not launched |

## 1. Denoising pair, seed 2 (highest priority)

**Why first:** the headline denoising margin (UG-TTT R_max 5.1409 vs baseline
4.6940, +0.4469) is ~100x the paper's per-domain deltas and rests on a single
seed (42 — and that seed was itself accidental: `--seed` was never passed).
One more seed either turns it into a claimable extension result or kills it.
Nothing else in the queue changes the resubmission story as much.

**What they are:** field-for-field replicas of the seed-42 pair's
`config.json` records (`ugttt-vm-export/vm-export/runs/denoising-{baseline,ugttt}/config.json`)
with **only `--seed` changed to 2**. Qwen3-8B fp16, 1 GPU per run, 6 epochs,
group 8 x 8 groups/batch, ctx 32768, max_tokens 260000, phase1 26000,
`entropic_adaptive_beta`, `puct_backprop`. Baseline: K=1, rmi 0, nnm 0.
UG-TTT: K=5, rmi 0.1, nnm 0.075, `true_mi`. Streaming MI off in both (no
calibration exists for denoising). The K=5/G=8 divisibility wart of the
seed-42 UG arm is preserved on purpose — fixing it would break the mirror.

**Code state:** must run on the checkout that produced the seed-42 pair —
trainer base `1eb925f` + `vm-export/code/ugttt-fixes.patch` (cutoff_len bound
to context_window; judge worker cap; scratch reclaim) — or a descendant
carrying those fixes. Preflight refuses to launch if `--context_window` is not
in the trainer's argparse.

**Host:** needs the bio venv (`/opt/dlami/nvme/venv-denoise`, openproblems
v0.8.0 + scanpy + MCV, `np.int` patch) because the verifier is imported
in-process; preflight import-checks it. Each run gets a private Ray head
(ports 6402/6403) — the reward path's detached `cpu_scheduler` actor must not
be shared between the arms. Launch base first, wait ~20 s, then ugttt (HF
cache contention at startup).

**Logs land in:** `logs/iclr/denoising-seed2-{baseline,ugttt}/` (run dir:
config.json, metrics.jsonl, trajectories.jsonl, sampler state, checkpoints)
plus console log `logs/iclr/denoising-seed2-{baseline,ugttt}.log`.
wandb: project `ttt-discover-uncertainty` (same as the seed-42 pair), names
`denoising-seed2-baseline` / `denoising-seed2-ugttt`.

## 2. CP26 gamma-decoupled ablation (pre-existing script) — CONFIG NOTE, READ THIS

`scripts/aws/launch_gamma_ablation.sh` (tested by
`tests/test_gamma_coupling.py`) runs the Remark-1 decoupling arm:
`rmi_coef 0.37`, `gamma_max_ratio 1.0`, seed 42, 4 epochs, in the **wave-2
CP26 config: ctx 8192 / max_tokens 16000 / phase1 6000**.

**Discrepancy flagged (V3):** the plan's drop-in slot describes this run as
ctx 32768, seed 42. The script as written uses ctx 8192. This is **not** an
oversight in the script — it is a copy of `node2/wave2/cp26-nostream/config.json`
— but the two descriptions disagree, and per prep rules we did not silently
change the script.

**Recommendation: run it at ctx 8192, exactly as the script stands.** Reasons:

- The decoupled arm's whole construction is *matched mean exploration
  strength*: rmi 0.37 is the realised mean γ_eff (0.3684) of the **8192**
  coupled run (`cp26-nostream`, per-epoch γ_eff 0.4593 / 0.2684 / 0.3332 /
  0.4125 over epochs 0–3). At ctx 32768 the β trajectory — and therefore the
  coupled γ_eff schedule being matched — would be different, invalidating the
  0.37 calibration.
- The comparison target (`cp26-nostream`, first 4 epochs / 256 rollouts)
  exists **only at 8192**. Running decoupled at 32768 would require also
  rerunning the coupled arm at 32768 (~12 h more GPU) before any comparison
  is possible.
- Precedent: all wave-2 ablation arms share ctx 8192, so within-family
  comparisons (the Table-2-style ablation table) remain valid; that is the
  established convention for the rebuttal ablations.

**Tradeoff to document wherever the numbers are used:** at 8192 the result is
an *ablation-internal* comparison (schedule vs constant γ_eff, everything else
matched). It is **not** comparable to the paper's Table 1 numbers (ctx 32768),
and must never be placed next to them. If a reviewer-facing claim ever needs
the decoupling story *at the paper's config*, both arms must be rerun at
32768 — that is a new, more expensive experiment (2 x ~4 epochs at 32768),
not a re-parameterisation of this one.

Logs land in `logs/aws/gamma_ablation/cp26-gamma-fixed/` per the script.

## 3. CP26 alpha=0, seed 2 (optional, run last)

`launch_cp26_alpha0_seed2.sh` — the wave-2 "MI bonus off" arm (K=5,
nnm 0.075, rmi 0.0) at seed 2, field-for-field from
`node2/wave2/cp26-alpha0/config.json` (ctx 8192), only the seed changed.
One documented run-control deviation: `--save_every 1` instead of wave-2's 2,
so the run is killable at any epoch boundary (checkpoint cadence does not
affect the training trajectory). 6 epochs at ~2.98 h/epoch ≈ 18 h; even 4
epochs is usable under the budget-truncation convention. Logs:
`logs/iclr/cp26-alpha0-seed2/` + `logs/iclr/cp26-alpha0-seed2.log`; wandb
project `ugttt-rebuttal`, name `cp26-alpha0-seed2`.

## After the runs: what to copy back where

Copy each finished run directory **whole** (config.json, metrics.jsonl,
trajectories.jsonl, logs, `last_epoch.txt`, `puct_backprop_sampler_step_*.json`;
adapter checkpoints `ensemble_*/` optional — large) to the analysis machine:

| Run | Remote dir | Copy to (local) |
|-----|-----------|-----------------|
| denoising-seed2-baseline | `logs/iclr/denoising-seed2-baseline/` | `C:\Users\kaina\ugttt-iclr-data\runs\denoising-seed2-baseline\` |
| denoising-seed2-ugttt | `logs/iclr/denoising-seed2-ugttt/` | `C:\Users\kaina\ugttt-iclr-data\runs\denoising-seed2-ugttt\` |
| cp26-gamma-fixed | `logs/aws/gamma_ablation/cp26-gamma-fixed/` | `C:\Users\kaina\ugttt-iclr-data\runs\cp26-gamma-fixed\` |
| cp26-alpha0-seed2 | `logs/iclr/cp26-alpha0-seed2/` | `C:\Users\kaina\ugttt-iclr-data\runs\cp26-alpha0-seed2\` |

Also grab the console `.log` files next to the run dirs. Then register the
new runs in the `RUNS` registry of `analysis/scripts/rebuttal_final.py` so
`load_rows` / `ep_curves` can see them. Keep them OUT of
`ugttt-vm-export/` and `ugttt-trajectories-data/` — those mirror what actually
ran in the rebuttal window and must stay frozen.

## Ground rules

- Do not launch anything from this directory without running `check` first.
- Never launch the denoising pair at ctx 8192 "for consistency with wave 2";
  the seed-42 denoising pair is 32768 and the mirror must match it.
- Never compare the gamma-ablation (8192) numbers to paper Table 1 (32768).
- Both denoising arms on one box: base on GPU 0 / port 6402, ugttt on GPU 1 /
  port 6403, staggered by ~20 s.
