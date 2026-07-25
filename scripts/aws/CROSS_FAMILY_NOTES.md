# Cross-family (Llama-3.1-8B) experiment — status & validation checklist

**Branch:** `exp/cross-family-llama` (off `aws-multi-gpu-launch`). Exploratory —
**not validated on GPU.** Goal: show UG-TTT's gain is not specific to Qwen3,
answering 8mjY W3 and vGzb Q1 ("single base model", "different base models
would be better").

## What was changed vs the main branch

`tinker_cookbook/rl/mlora_train.py`:
- `wrap_llama_chat_template()` — Llama-3.1 chat format, no `<think>` (BOS added by
  the tokenizer, so omitted here to avoid a doubled BOS).
- `base_model_family()` — `"llama"` if the model name contains "llama", else the
  Qwen3 thinking path (unchanged default).
- Two-phase setup: when two-phase is off **and** family is llama, the stop token
  becomes `<|eot_id|>` and the Llama template is applied at the prompt site.

`scripts/aws/launch_llama_test.sh` — 2-run pair (UG-TTT vs baseline) on CP26,
**single-phase** (`--two_phase_sampling` omitted), `--max_tokens 26000`,
model `meta-llama/Llama-3.1-8B-Instruct`. CP26 is chosen because its prompt is
built from free functions (`build_cp_prompt`), not the renderer-coupled Env class.

## Why single-phase

The published two-phase path is Qwen3-specific: it wraps prompts in ChatML with
`<think>`, prefetches a "out of thinking tokens" prefill, and stops on `<|im_end|>`.
Llama-3.1 has no thinking mode, so two-phase is semantically wrong for it. The
Llama runs therefore use one generation pass with the Llama chat template. This
is a **different configuration** from the Qwen3 headline — report it as
"UG-TTT signal on a different family, single-phase", not as a like-for-like
Table-1 row.

## Validate on the FIRST GPU run (before trusting any numbers)

1. **Model loads.** mLoRA's `MODEL_TYPE_DICT` only has `LlamaModel`. Llama-3.1-8B
   is native Llama arch, so `from_pretrained` should work — but confirm no config
   mismatch (rope scaling, vocab). If it errors on load, this needs a loader fix.
2. **No renderer assert fires.** `env_ac.py:154` asserts a `GptOssRenderer` and
   `env_ac.py:211` checks `Qwen3Renderer`. The mlora path wires free functions
   (`build_cp_prompt`/`verify_cp`), so for **CP26** these should not trigger. If
   you later try AC1/AC2 on Llama, expect to hit these.
3. **Prompt looks right.** Grep the first prompt in `logs/.../trajectories.jsonl`
   — it should start with `<|start_header_id|>user<|end_header_id|>` and contain
   no `<|im_start|>` / `<think>`.
4. **Generation stops.** Confirm rollouts terminate at `<|eot_id|>` and are not
   running to `--max_tokens` every time (a sign the stop token id was empty).
5. **HF access.** `meta-llama/Llama-3.1-8B-Instruct` is gated — the box needs a
   HF token with access (`huggingface-cli login` or `HF_TOKEN`).

## If it works

Port the three `mlora_train.py` changes back to `aws-multi-gpu-launch` and add a
`llama8b` pair there. If it doesn't, the fallback with zero code risk is the
Qwen3 scale ladder already on the main branch (8B→14B→32B), which answers the
same reviewers without a cross-family run.
