#!/usr/bin/env python3
"""
Pre-flight check for multi-GPU UG-TTT runs.

Downloads only the config + tokenizer (a few KB — no weights), then reports
whether a model can run under mLoRA at all, how much memory it needs, and what
--gpus / --gpu_layer_balance to pass.

Run this BEFORE pulling 145 GB of 72B weights.

    python scripts/preflight_model_check.py \
        --models Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B Qwen/Qwen2.5-72B-Instruct \
        --gpu-memory-gb 80 --num-gpus 8 --context-window 32768 --group-size 8

Exit code is non-zero if any hard precondition fails.
"""

import argparse
import sys
from typing import List, Optional

# mLoRA only accepts these; anything else asserts inside
# LlamaModel.from_pretrained (model_llama.py:252).
LLAMA_COMPATIBLE = ["mistral", "qwen2", "qwen3", "llama"]

BYTES_PER_DTYPE = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "nf4": 0.5, "fp4": 0.5}


def gib(n_bytes: float) -> float:
    return n_bytes / (1024 ** 3)


def analyse(
    name: str,
    precision: str,
    context_window: int,
    group_size: int,
    num_ensemble: int,
    gpu_memory_gb: float,
    num_gpus: int,
) -> bool:
    from transformers import AutoConfig

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    ok = True

    try:
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
    except Exception as exc:  # network / gated repo / typo
        print(f"  FAIL  could not load config: {exc}")
        return False

    n_layers = cfg.num_hidden_layers
    hidden = cfg.hidden_size
    vocab = cfg.vocab_size
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads
    max_pos = getattr(cfg, "max_position_embeddings", None)
    sliding = getattr(cfg, "sliding_window", None)
    rope_scaling = getattr(cfg, "rope_scaling", None)
    tied = getattr(cfg, "tie_word_embeddings", False)

    print(f"  model_type            {cfg.model_type}")
    print(f"  layers / hidden       {n_layers} / {hidden}")
    print(f"  heads (q/kv), head_d  {n_heads}/{n_kv}, {head_dim}")
    print(f"  vocab                 {vocab}")
    print(f"  max_position_embed    {max_pos}")
    print(f"  sliding_window        {sliding}")
    print(f"  rope_theta            {getattr(cfg, 'rope_theta', None)}")
    print(f"  rope_scaling          {rope_scaling}")
    print(f"  tie_word_embeddings   {tied}")

    # ── Hard preconditions ────────────────────────────────────────────────
    if cfg.model_type not in LLAMA_COMPATIBLE:
        print(f"  FAIL  model_type {cfg.model_type!r} not in {LLAMA_COMPATIBLE}; "
              "LlamaModel.from_pretrained will assert.")
        ok = False

    if tied:
        print("  FAIL  tie_word_embeddings=True: lm_head shares storage with "
              "embed_tokens, which layer sharding cannot split. Single GPU only.")
        ok = False

    if rope_scaling:
        print("  FAIL  rope_scaling is set but LLMModelArgs DROPS it "
              "(mLoRA/mlora/model/args.py:33-66) — RoPE would be silently wrong.")
        ok = False

    # This mirrors LLMModelArgs.__from_pretrained_config exactly.
    eff_max_seq = 4096
    if hasattr(cfg, "max_sequence_length"):
        eff_max_seq = cfg.max_sequence_length
    elif max_pos is not None:
        eff_max_seq = max_pos
    if sliding is not None and eff_max_seq > sliding:
        eff_max_seq = sliding
        print(f"  WARN  sliding_window shrinks max_seq_len_ to {eff_max_seq}")

    print(f"  -> mLoRA max_seq_len_ {eff_max_seq}  (sizes the RoPE tables)")
    if eff_max_seq < context_window:
        print(f"  FAIL  max_seq_len_ ({eff_max_seq}) < context_window "
              f"({context_window}). Attention.forward slices cos_[pos:pos+len] "
              "with NO bounds check, so this fails deep inside apply_rotary_emb.")
        ok = False

    # ── Memory ────────────────────────────────────────────────────────────
    wbytes = BYTES_PER_DTYPE.get(precision, 2)
    # Transformer body only; embedding + lm_head counted separately below.
    per_layer_params = (
        hidden * n_heads * head_dim          # q_proj
        + 2 * hidden * n_kv * head_dim       # k_proj, v_proj
        + n_heads * head_dim * hidden        # o_proj
        + 3 * hidden * getattr(cfg, "intermediate_size", 4 * hidden)  # gate/up/down
    )
    body = per_layer_params * n_layers * wbytes
    embed = vocab * hidden * wbytes
    lm_head = vocab * hidden * wbytes
    weights = body + embed + lm_head

    # Pre-allocated KV: 2 (K and V) * N * n_kv * ctx * head_dim, fp16.
    kv_per_layer = 2 * group_size * n_kv * context_window * head_dim * 2
    kv_total = kv_per_layer * n_layers

    # RoPE cos/sin tables, fp32, one pair per layer.
    rope = 2 * eff_max_seq * head_dim * 4 * n_layers

    # Transient fp32 buffers that all land on the LAST device.
    mi_buffers = 4 * num_ensemble * 256 * vocab * 4

    total = weights + kv_total + rope + mi_buffers

    print(f"\n  weights ({precision})    {gib(weights):8.1f} GiB")
    print(f"  KV cache @ ctx {context_window}, N={group_size}"
          f"   {gib(kv_total):8.1f} GiB  ({gib(kv_per_layer):.2f} GiB/layer)")
    print(f"  RoPE tables           {gib(rope):8.1f} GiB")
    print(f"  fp32 MI/logit buffers {gib(mi_buffers):8.1f} GiB  (last GPU only)")
    print(f"  {'-' * 44}\n  TOTAL                 {gib(total):8.1f} GiB")

    # ── Recommendation ────────────────────────────────────────────────────
    # Leave ~12% headroom for activations, fragmentation and the reward path.
    usable = gpu_memory_gb * 0.88
    need = int(gib(total) / usable) + 1
    need = max(1, need)

    if need > num_gpus:
        print(f"\n  FAIL  needs ~{need} GPUs of {gpu_memory_gb:.0f} GB but only "
              f"{num_gpus} available. Reduce --context_window (KV is linear in "
              "it) or --group_size, or use a smaller model.")
        ok = False
    elif need == 1:
        print(f"\n  OK    fits on 1 GPU. Run with --gpus 1 (default).")
    else:
        print(f"\n  OK    needs ~{need} GPUs. Run with --gpus {need}")
        print(f"        (~{gib(total) / need:.0f} GiB/GPU; if the LAST GPU OOMs, "
              f"shift 1-2 layers off it with --gpu_layer_balance)")

    return ok


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--precision", default="fp16", choices=list(BYTES_PER_DTYPE))
    p.add_argument("--context-window", type=int, default=32768)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--num-ensemble", type=int, default=5)
    p.add_argument("--gpu-memory-gb", type=float, default=80.0)
    p.add_argument("--num-gpus", type=int, default=8)
    args = p.parse_args(argv)

    all_ok = True
    for name in args.models:
        all_ok &= analyse(
            name, args.precision, args.context_window, args.group_size,
            args.num_ensemble, args.gpu_memory_gb, args.num_gpus,
        )

    print(f"\n{'=' * 78}")
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED — see FAIL lines above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
